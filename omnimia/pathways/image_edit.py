import argparse
import glob
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from omnimia.pathways.image import _debug_decode_built_one, _save_reconstructed_image_from_sequence
from omnimia.trajectory import score_probability_trajectories
from omnimia.evaluation import binary_roc_auc

autocast = torch.autocast


def _load_processed_record_keys(*, results_path: str, input_key: str, output_key: str, instruction_key: str, label_key: str) -> set[Tuple[str, str, str, int]]:
    """Load processed record keys from an existing results shard."""
    processed: set[Tuple[str, str, str, int]] = set()
    if not results_path or not os.path.exists(results_path):
        return processed
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    inp = obj.get(str(input_key), None)
                    out = obj.get(str(output_key), None)
                    instr = obj.get(str(instruction_key), None)
                    lbl = obj.get(str(label_key), None)
                    if inp is None or out is None or instr is None or lbl is None:
                        continue
                    processed.add((str(inp), str(out), str(instr), int(lbl)))
                except Exception:
                    continue
    except Exception:
        return processed
    return processed


def _load_processed_record_keys_from_shard_dir(
    *, shard_dir: str, input_key: str, output_key: str, instruction_key: str, label_key: str
) -> set[Tuple[str, str, str, int]]:
    processed: set[Tuple[str, str, str, int]] = set()
    if not shard_dir or not os.path.isdir(shard_dir):
        return processed
    for path in sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl"))):
        processed |= _load_processed_record_keys(
            results_path=path,
            input_key=str(input_key),
            output_key=str(output_key),
            instruction_key=str(instruction_key),
            label_key=str(label_key),
        )
    return processed


def _stable_record_hash(key: Tuple[str, str, str, int]) -> int:
    import hashlib

    inp, out, instr, lbl = key
    payload = (str(inp) + "\n" + str(out) + "\n" + str(instr) + "\n" + str(int(lbl))).encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass
class Config:
    backend: str
    input_path: str
    output_dir: str
    pretrained_model_path: str
    vq_model_name: Optional[str]

    input_key: str
    output_key: str
    instruction_key: str
    label_key: str
    limit: Optional[int]

    noise_target: str  # input, output, both

    device: str
    torch_dtype: str

    batch_size: int
    max_seq_len: int
    devices: Optional[str]
    num_workers: Optional[int]
    seed: int
    mask_ratio: float
    num_trajectories: int
    steps_mode: str
    steps_value: float
    top_d: int
    save_probs: bool
    save_traj_outputs: bool
    save_traj_tokens: bool
    save_dtype: str
    aggregation_topk: int
    eps: float
    progress: bool
    resume: bool
    image_size: Optional[Tuple[int, int]]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {name}")


def _parse_devices(devices: Optional[str]) -> Optional[List[str]]:
    if devices is None:
        return None
    items = [x.strip() for x in str(devices).split(",") if x.strip()]
    return items or None


def _load_backend(
    *,
    backend: str,
    pretrained_model_path: str,
    vq_model_name: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Any, Any, int]:
    if backend == "mmada":
        from models.MMaDA.pipeline import MMaDAPipeline

        pipe = MMaDAPipeline.from_pretrained(
            task="img2img",
            pretrained_model_path=pretrained_model_path,
            vq_model_name=vq_model_name,
            device=device,
            torch_dtype=dtype,
            padding_side="right",
        )
        model = pipe.model
        tokenizer = pipe.tokenizer
        text_vocab_end = int(len(pipe.uni_prompting.text_tokenizer))
        setattr(model, "_vdlm_text_vocab_end", text_vocab_end)
        mask_id = int(
            getattr(model.config, "mask_token_id", None)
            or getattr(pipe.model.config, "mask_token_id")
        )
        return {"pipe": pipe}, model, tokenizer, mask_id

    if backend == "lumina":
        from models.LuminaDiMOO.pipeline import LuminaDiMOOPipeline
        from models.LuminaDiMOO.trajectory_prob import mask_token_id as lumina_mask_token_id

        # Initialize as i2i task
        pipe = LuminaDiMOOPipeline.from_pretrained(
            task="i2i",
            pretrained_model_path=pretrained_model_path,
            vq_model_name=vq_model_name,
            device=device,
            torch_dtype=dtype,
            device_map=None,
        )
        model = pipe.model
        tokenizer = pipe.tokenizer
        mask_id = int(lumina_mask_token_id())
        return {"pipe": pipe}, model, tokenizer, mask_id

    raise ValueError(f"Unsupported backend for this script: {backend}")


def _build_inputs_and_image_pos_mask(
    *,
    backend: str,
    state: Dict[str, Any],
    tokenizer: Any,
    allow_token_mask: torch.BoolTensor,
    input_path: str,
    output_path: str,
    instruction: str,
    max_seq_len: int,
    noise_target: str,
    image_size: Optional[Tuple[int, int]],
) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, Tuple[int, int]]:
    
    pipe = state["pipe"]

    if backend == "mmada":
        from models.MMaDA.prompting_utils import reserved_token_mapping
        from models.MMaDA.utils import image_transform, image_transform_squash

        if pipe.vq_model is None:
            raise RuntimeError("MMaDA i2i requires a VQ model")
        if not noise_target == "output":
            raise ValueError("MMaDA i2i PPPL currently supports noise_target='output' only")

        resolution = int(image_size[0]) if image_size is not None else 256
        if image_size is not None and int(image_size[0]) != int(image_size[1]):
            raise ValueError("MMaDA image preprocessing requires a square image_size")

        def _encode(path: str) -> torch.LongTensor:
            image = Image.open(str(path)).convert("RGB")
            filename = os.path.basename(str(path))
            if any(tag in filename for tag in ["ai2d", "clevr", "docvqa", "geo", "llava"]):
                image_tensor = image_transform_squash(image, resolution=resolution)
            else:
                image_tensor = image_transform(image, resolution=resolution)
            image_tensor = image_tensor.to(pipe.device).unsqueeze(0)
            vq_param = next(pipe.vq_model.parameters(), None)
            vq_dtype = vq_param.dtype if vq_param is not None else image_tensor.dtype
            with torch.autocast(pipe.device.type, enabled=False):
                image_codes = pipe.vq_model.get_code(image_tensor.to(dtype=vq_dtype))
            return image_codes[0].to(torch.long) + len(pipe.uni_prompting.text_tokenizer)

        source_tokens = _encode(input_path)
        target_tokens = _encode(output_path)

        text_ids = pipe.uni_prompting.text_tokenizer.apply_chat_template(
            [{"role": "user", "content": str(instruction)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(text_ids, list) and text_ids and isinstance(text_ids[0], list):
            text_ids = text_ids[0]
        text_ids = torch.tensor(list(map(int, text_ids)), device=pipe.device, dtype=torch.long)

        v2v_id = int(reserved_token_mapping["<|v2v|>"])
        soi_id = int(reserved_token_mapping["<|soi|>"])
        eoi_id = int(reserved_token_mapping["<|eoi|>"])
        bos_id = int(pipe.uni_prompting.text_tokenizer.bos_token_id)
        mask_id = int(getattr(pipe.model.config, "mask_token_id"))

        condition = torch.cat(
            [
                torch.tensor([v2v_id, soi_id], device=pipe.device, dtype=torch.long),
                source_tokens,
                torch.tensor([eoi_id], device=pipe.device, dtype=torch.long),
                text_ids,
            ]
        )
        text_generation_length = int(getattr(pipe.runtime, "max_seq_length", 128))
        output_suffix = torch.full(
            (text_generation_length,),
            mask_id,
            device=pipe.device,
            dtype=torch.long,
        )
        output_suffix[0] = bos_id
        full_sequence = torch.cat(
            [
                condition,
                torch.tensor([soi_id], device=pipe.device, dtype=torch.long),
                target_tokens,
                torch.tensor([eoi_id], device=pipe.device, dtype=torch.long),
                output_suffix,
            ]
        ).unsqueeze(0)
        attention_mask = torch.ones_like(full_sequence, dtype=torch.bool)

        if int(full_sequence.shape[1]) > int(max_seq_len):
            full_sequence = full_sequence[:, : int(max_seq_len)]
            attention_mask = attention_mask[:, : int(max_seq_len)]

        target_start = int(condition.numel()) + 1
        target_end = min(target_start + int(target_tokens.numel()), int(full_sequence.shape[1]))
        pos_mask = torch.zeros_like(full_sequence, dtype=torch.bool)
        if target_start < target_end:
            pos_mask[0, target_start:target_end] = True
        pos_mask &= attention_mask
        pos_mask &= allow_token_mask.to(pipe.device)[full_sequence]
        if not bool(pos_mask.any()):
            raise RuntimeError("No valid MMaDA output-image positions found")
        pos_any = pos_mask[0].detach().cpu().numpy().astype(bool)
        start = int(np.argmax(pos_any))
        end = int(len(pos_any) - np.argmax(pos_any[::-1]))
        return full_sequence, attention_mask, pos_mask, (start, end)

    if backend == "lumina":
        from models.LuminaDiMOO.utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop
        from models.LuminaDiMOO.config import SPECIAL_TOKENS

        # Construct the i2i conditioning prompt used by the Lumina pipeline.
        system_text = "Generate an image applying the following editing instruction based on the original image."
        input_prompt = f"<system>{system_text}</system><user>{instruction}</user>"
        
        prompt_ids = tokenizer(input_prompt)["input_ids"]
        # Insert source image tokens before the final prompt token, matching
        # the model pipeline's conditioning sequence construction.
        src_img = Image.open(str(input_path)).convert("RGB")
        tgt_img = Image.open(str(output_path)).convert("RGB")
        
        if image_size is not None:
             w, h = image_size
             src_img = src_img.resize((w, h), Image.BICUBIC)
             tgt_img = tgt_img.resize((w, h), Image.BICUBIC)
        else:
             # Use original resolution, snapped to the VQ grid.
             w, h = src_img.size
             w = max(32, (w // 32) * 32)
             h = max(32, (h // 32) * 32)
             src_img = src_img.resize((w, h), Image.BICUBIC)
             
             w, h = tgt_img.size
             w = max(32, (w // 32) * 32)
             h = max(32, (h // 32) * 32)
             tgt_img = tgt_img.resize((w, h), Image.BICUBIC)

        src_tokens = encode_img_with_breaks(src_img, pipe.vqvae) # List[int]
        tgt_tokens = encode_img_with_breaks(tgt_img, pipe.vqvae) # List[int]
        
        # Target sequence: answer-start, image codes, answer-end.
        BOA = SPECIAL_TOKENS["answer_start"]
        EOA = SPECIAL_TOKENS["answer_end"]
        
        con_input = prompt_ids[:-1] + src_tokens + prompt_ids[-1:]
        tgt_seq = [BOA] + tgt_tokens + [EOA]
        
        full_seq = con_input + tgt_seq
        
        input_ids = torch.tensor(full_seq, device=pipe.device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        if int(input_ids.shape[1]) > int(max_seq_len):
            input_ids = input_ids[:, : int(max_seq_len)]
            attention_mask = attention_mask[:, : int(max_seq_len)]
            
        # Region 1 is the source image; region 2 is the target image.
        src_start = len(prompt_ids[:-1])
        src_end = src_start + len(src_tokens)
        
        tgt_start = len(con_input) + 1
        tgt_end = tgt_start + len(tgt_tokens)
        
        pos_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        vocab_allow = allow_token_mask.to(input_ids.device)[input_ids]
        
        if noise_target in ["input", "both"]:
            s = min(src_start, input_ids.shape[1])
            e = min(src_end, input_ids.shape[1])
            if s < e:
                pos_mask[0, s:e] = True
                
        if noise_target in ["output", "both"]:
            s = min(tgt_start, input_ids.shape[1])
            e = min(tgt_end, input_ids.shape[1])
            if s < e:
                pos_mask[0, s:e] = True
        
        # Final combined mask
        pos_mask = pos_mask & attention_mask & vocab_allow
        
        pos_any = pos_mask[0].detach().cpu().numpy().astype(bool)
        if not pos_any.any():
            raise RuntimeError(f"No valid positions found for noise_target={noise_target}")
            
        start = int(np.argmax(pos_any))
        end = int(len(pos_any) - np.argmax(pos_any[::-1]))
        
        return input_ids.to(torch.long), attention_mask.to(torch.bool), pos_mask.to(torch.bool), (start, end)

    raise ValueError(f"Unknown backend: {backend}")


# Reuse helper functions from original file
def _infer_text_vocab_end(*, backend: str, model: Any, tokenizer: Any) -> Optional[int]:
    if backend == "lumina":
        try:
            from models.LuminaDiMOO.config import SPECIAL_TOKENS
            return int(dict(SPECIAL_TOKENS)["image_token_offset"])
        except Exception:
            return None
    if backend == "mmada":
        cached = getattr(model, "_vdlm_text_vocab_end", None)
        if cached is not None:
            return int(cached)
        try:
            return int(len(tokenizer))
        except Exception:
            try:
                value = getattr(getattr(model, "config", None), "llm_vocab_size", None)
                return int(value) if value is not None else None
            except Exception:
                return None
    return None

def _collect_forbid_token_ids(*, backend: str, tokenizer: Any, mask_id: int) -> set[int]:
    forbid: set[int] = set()
    for tid in [
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        int(mask_id),
    ]:
        if tid is not None:
            forbid.add(int(tid))
    try:
        for tid in getattr(tokenizer, "all_special_ids", []) or []:
            forbid.add(int(tid))
    except Exception:
        pass
    if backend == "lumina":
        try:
            from models.LuminaDiMOO.config import SPECIAL_TOKENS
            for _k, v in dict(SPECIAL_TOKENS).items():
                forbid.add(int(v))
        except Exception:
            pass
    if backend == "mmada":
        try:
            from models.MMaDA.pipeline import _SPECIAL_TOKENS

            for token in _SPECIAL_TOKENS:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    forbid.add(int(token_id))
        except Exception:
            pass
    return forbid

def _build_allow_token_mask(*, backend: str, tokenizer: Any, mask_id: int, vocab_size: int, text_vocab_end: Optional[int], image_vocab_size: Optional[int] = None) -> torch.BoolTensor:
    if vocab_size <= 0: raise ValueError("vocab_size > 0")
    if text_vocab_end is None: raise ValueError("text_vocab_end is None")
    
    start = max(0, min(int(text_vocab_end), int(vocab_size)))
    if image_vocab_size is None:
        end = int(vocab_size)
    else:
        end = min(int(vocab_size), start + max(0, int(image_vocab_size)))
        
    allow = torch.zeros((int(vocab_size),), dtype=torch.bool)
    allow[start:end] = True
    forbid = _collect_forbid_token_ids(backend=backend, tokenizer=tokenizer, mask_id=int(mask_id))
    for tid in forbid:
        if 0 <= int(tid) < int(vocab_size):
            allow[int(tid)] = False
    return allow

def _build_maskable_token_mask(*, backend: str, tokenizer: Any, mask_id: int, vocab_size: int) -> torch.BoolTensor:
    maskable = torch.ones((int(vocab_size),), dtype=torch.bool)
    forbid = _collect_forbid_token_ids(backend=backend, tokenizer=tokenizer, mask_id=int(mask_id))
    for tid in forbid:
        if 0 <= int(tid) < int(vocab_size):
            maskable[int(tid)] = False
    return maskable

def _infer_model_vocab_size(*, model: Any, device: torch.device) -> int:
    try:
        if getattr(model, "vocab_size", None):
            return int(model.vocab_size)
        emb = model.get_input_embeddings()
        if emb is not None:
            return int(emb.num_embeddings)
    except (AttributeError, TypeError):
        pass
    x = torch.zeros((1, 1), device=device, dtype=torch.long)
    attention_bias = torch.ones((1, 1, 1, 1), device=device, dtype=torch.bool)
    logits = model(x, attention_bias=attention_bias).logits
    return int(logits.size(-1))

def _choose_mask_positions_in_region(
    input_ids_1d: torch.LongTensor,
    *,
    region_pos_mask: torch.BoolTensor,
    mask_ratio: float,
    rng: np.random.Generator,
    maskable_token_mask: torch.BoolTensor,
) -> torch.BoolTensor:
    # Logic remains identical: it samples from region_pos_mask
    if input_ids_1d.ndim != 1: raise ValueError("input_ids_1d must be 1D")
    allow_ids = maskable_token_mask.detach().cpu().to(torch.bool)
    vocab_size = int(allow_ids.numel())
    cand: List[int] = []
    ids_cpu = input_ids_1d.detach().cpu().to(torch.long)
    pos_cpu = region_pos_mask.detach().cpu().to(torch.bool)
    for pos, (tid, ok_pos) in enumerate(zip(ids_cpu.tolist(), pos_cpu.tolist())):
        if not bool(ok_pos): continue
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size and bool(allow_ids[tid_i].item()):
            cand.append(int(pos))
    if not cand: return torch.zeros_like(input_ids_1d, dtype=torch.bool)
    k = int(round(len(cand) * float(mask_ratio)))
    if float(mask_ratio) > 0: k = max(1, k)
    k = min(k, len(cand))
    idx = np.array(cand, dtype=np.int64)
    rng.shuffle(idx)
    chosen = idx[:k]
    mask = torch.zeros_like(input_ids_1d, dtype=torch.bool)
    mask[torch.from_numpy(chosen).to(mask.device)] = True
    return mask

def _select_topk_candidates_from_logits(logits: torch.Tensor, *, topk: int, allow_token_mask: torch.BoolTensor) -> torch.LongTensor:
    # Reuse original logic
    allow = allow_token_mask.to(device=logits.device, dtype=torch.bool)
    min_val = torch.finfo(logits.dtype).min
    logits_m = logits.masked_fill((~allow).view(1, 1, -1), min_val)
    _, idx = torch.topk(logits_m, k=int(topk), dim=-1)
    return idx.to(torch.long)

@torch.no_grad()
def _denoise_candidate_prob_trajectories(
    *,
    backend: str,
    model: Any,
    x_init: torch.LongTensor,
    candidate_ids: torch.LongTensor,
    mask_id: int,
    steps: int,
    allowed_range: Tuple[int, int],
    temperature: float,
    attention_bias: Optional[torch.Tensor],
    return_x_history: bool = False,
    return_x_final: bool = False,
) -> np.ndarray:
    if backend == "mmada":
        from models.MMaDA.modeling_mmada import add_gumbel_noise, get_num_transfer_tokens
    elif backend == "lumina":
        from models.LuminaDiMOO.utils.generation_utils import add_gumbel_noise, get_num_transfer_tokens
    else:
        raise ValueError(f"Unknown backend: {backend}")

    x = x_init.clone()
    start, end = allowed_range # Note: this range is used for checking allowed tokens in loop logic.
    # Our _build function returns (min_start, max_end). 
    # But mask_pos in logic respects the 'region_pos_mask' which can have holes (text in between).
    # This function uses allowed_range mainly for slicing logits?
    # Original: logits_allowed_full = logits[:, start:end, :]
    # This assumes contiguous range of interest? 
    # If we have [Img1] [Text] [Img2]. min_start=0, max_end=N.
    # logits[:, 0:N, :] includes Text.
    # But mask_index = (x == int(mask_id)) & allowed
    # allowed[:, start:end] = True.
    # Text tokens are NOT mask_id (they are never masked).
    # So logic should hold even if range encompasses unmasked text.

    allowed = torch.zeros_like(x, dtype=torch.bool)
    allowed[:, start:end] = True

    mask_index0 = (x == int(mask_id)) & allowed
    num_transfer_tokens = get_num_transfer_tokens(mask_index0, int(steps))

    bsz = int(x.shape[0])
    l = int(end - start)
    k = int(candidate_ids.shape[-1])
    out = np.zeros((bsz, int(steps), l, k), dtype=np.float16)

    x_hist: Optional[List[np.ndarray]] = [] if bool(return_x_history) else None

    for i in range(int(steps)):
        mask_index = (x == int(mask_id)) & allowed

        logits = model(x, attention_bias=attention_bias).logits
        logits_allowed_full = logits[:, start:end, :]

        logits_f = logits_allowed_full.to(torch.float32)
        denom = torch.logsumexp(logits_f, dim=-1, keepdim=True)
        cand = candidate_ids[:, start:end, :].to(device=logits_f.device, dtype=torch.long)
        sel = torch.gather(logits_f, dim=-1, index=cand)
        probs = torch.exp(sel - denom).to(torch.float16)
        out[:, i, :, :] = probs.detach().cpu().numpy()

        if int(mask_index[:, start:end].sum().item()) == 0:
            break

        logits_with_noise = add_gumbel_noise(logits, temperature=float(temperature))
        x0 = torch.argmax(logits_with_noise, dim=-1)

        x0_allowed = x0[:, start:end]
        mask_allowed = mask_index[:, start:end]

        mask_flat = mask_allowed.reshape(-1)
        masked_flat_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if int(masked_flat_indices.numel()) == 0:
            break

        batch_ids = masked_flat_indices // l
        pos_in_allowed = masked_flat_indices % l
        pos_abs = pos_in_allowed + int(start)

        logits_flat = logits_allowed_full.reshape(-1, logits_allowed_full.size(-1)).to(torch.float32)
        x0_flat = x0_allowed.reshape(-1)
        logits_masked = logits_flat[mask_flat]
        x0_masked = x0_flat[mask_flat]

        log_denom = torch.logsumexp(logits_masked, dim=-1)
        sel = torch.gather(logits_masked, dim=-1, index=x0_masked.unsqueeze(-1)).squeeze(-1)
        logp_masked = sel - log_denom

        x0 = torch.where(mask_index, x0, x)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for b in range(int(x.shape[0])):
            kb = int(num_transfer_tokens[b, i].item())
            if kb <= 0: continue
            b_sel = batch_ids == int(b)
            n_masked_b = int(b_sel.sum().item())
            if n_masked_b <= 0: continue
            kb = min(kb, n_masked_b)
            conf_b = logp_masked[b_sel]
            pos_b = pos_abs[b_sel]
            _, top_local = torch.topk(conf_b, k=int(kb))
            transfer_index[b, pos_b[top_local]] = True

        x[transfer_index] = x0[transfer_index]
        if x_hist is not None:
            x_hist.append(x.detach().cpu().to(torch.int32).numpy())

    if x_hist is not None:
        if len(x_hist) == 0:
            hist = np.zeros((bsz, 0, int(x.shape[1])), dtype=np.int32)
        else:
            hist = np.stack(x_hist, axis=0).transpose(1, 0, 2).astype(np.int32, copy=False)
        return out, hist

    if not bool(return_x_final):
        return out
    return out, x.detach().cpu().to(torch.int32)


def _process_batch(
    batch: Sequence[Tuple[int, str, str, str, int]],
    *,
    out_f,
    probs_dir: Optional[str],
    traj_outputs_dir: Optional[str],
    traj_tokens_dir: Optional[str],
    backend: str,
    state: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    mask_id: int,
    allow_token_mask: torch.BoolTensor,
    maskable_token_mask: torch.BoolTensor,
    rng_global: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
    args: Config,
) -> None:
    sample_ids = [x[0] for x in batch]
    input_paths = [x[1] for x in batch]
    output_paths = [x[2] for x in batch]
    instructions = [x[3] for x in batch]
    labels = [x[4] for x in batch]

    built = []
    for inp, out, instr in zip(input_paths, output_paths, instructions):
        built.append(
            _build_inputs_and_image_pos_mask(
                backend=backend,
                state=state,
                tokenizer=tokenizer,
                allow_token_mask=allow_token_mask,
                input_path=inp,
                output_path=out,
                instruction=instr,
                max_seq_len=int(args.max_seq_len),
                noise_target=args.noise_target,
                image_size=args.image_size,
            )
        )

	# Quick sanity check (optional): export one decoded text+image from the first built sample.
	# Usage: VDLM_MIA_DEBUG_DECODE=1 python -m omnimia.pathways.image_edit ...
    if os.environ.get("VDLM_MIA_DEBUG_DECODE", "").strip() == "1" and built:
        try:
            ids_1xL, am_1xL, _pm_1xL, _span = built[0]
            _debug_decode_built_one(
				backend=str(backend),
				state=state,
				tokenizer=tokenizer,
				input_ids_1xL=ids_1xL,
				attention_mask_1xL=am_1xL,
				allow_token_mask=allow_token_mask,
				out_dir=os.path.join(str(args.output_dir), "_debug_decode"),
				name=f"rank_debug_sample{int(sample_ids[0])}",
			)
        except (OSError, RuntimeError, ValueError):
            pass


    max_len = max(int(x[0].shape[1]) for x in built)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None: raise RuntimeError("Tokenizer has no pad_token_id")

    input_ids = torch.full((len(built), max_len), int(pad_id), device=device, dtype=torch.long)
    attention_mask = torch.zeros((len(built), max_len), device=device, dtype=torch.bool)
    pos_mask = torch.zeros((len(built), max_len), device=device, dtype=torch.bool)
    spans: List[Tuple[int, int]] = []

    for i, (ids, am, pm, span) in enumerate(built):
        L = int(ids.shape[1])
        input_ids[i, :L] = ids[0]
        attention_mask[i, :L] = am[0]
        pos_mask[i, :L] = pm[0]
        spans.append(span)

    attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).unsqueeze(1)

    with torch.no_grad():
        with (autocast("cuda", dtype=dtype) if device.type == "cuda" else autocast("cpu", enabled=False)):
            logits_clean = model(input_ids, attention_bias=attention_bias).logits
            candidate_ids = _select_topk_candidates_from_logits(
                logits_clean,
                topk=int(args.top_d),
                allow_token_mask=allow_token_mask,
            )
            logits_f = logits_clean.to(torch.float32)
            logp = logits_f - torch.logsumexp(logits_f, dim=-1, keepdim=True)
            pos_logp = torch.gather(logp, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B,L]
            pos_logp = pos_logp.to(torch.float16)

    traj_probs_by_sample: List[List[np.ndarray]] = [[] for _ in range(len(batch))]
    span0 = min(s for s, _ in spans)
    span1 = max(e for _, e in spans)
    allowed_range = (int(span0), int(span1))

    for t_global in range(int(args.num_trajectories)):
        mask_pos = torch.zeros_like(input_ids, dtype=torch.bool)
        masked_counts: List[int] = []
        for b in range(int(input_ids.shape[0])):
            rng = np.random.default_rng(int(rng_global.integers(0, 2**31 - 1)))
            m = _choose_mask_positions_in_region(
                input_ids[b],
                region_pos_mask=pos_mask[b],
                mask_ratio=float(args.mask_ratio),
                rng=rng,
                maskable_token_mask=maskable_token_mask,
            )
            mask_pos[b] = m.to(device=device)
            masked_counts.append(int(m.sum().item()))

        if int(mask_pos.sum().item()) == 0:
            continue

        x_init = input_ids.clone()
        x_init[mask_pos] = int(mask_id)

        steps_per_sample = []
        for mc in masked_counts:
            if mc <= 0: steps_per_sample.append(0)
            elif args.steps_mode == "abs": steps_per_sample.append(int(args.steps_value))
            else: steps_per_sample.append(max(1, int(math.ceil(int(mc) * float(args.steps_value)))))

        buckets = {}
        for b, st in enumerate(steps_per_sample):
            if st <= 0: continue
            buckets.setdefault(int(st), []).append(int(b))

        with torch.no_grad():
            with (autocast("cuda", dtype=dtype) if device.type == "cuda" else autocast("cpu", enabled=False)):
                for st, idxs in buckets.items():
                    x_sub = x_init[idxs]
                    ab_sub = attention_bias[idxs]
                    cand_sub = candidate_ids[idxs]
                    need_hist = traj_tokens_dir is not None
                    denoise_out = _denoise_candidate_prob_trajectories(
                        backend=backend,
                        model=model,
                        x_init=x_sub,
                        candidate_ids=cand_sub,
                        mask_id=int(mask_id),
                        steps=int(st),
                        allowed_range=allowed_range,
                        temperature=1.0,
                        attention_bias=ab_sub,
                        return_x_history=bool(need_hist),
                        return_x_final=(traj_outputs_dir is not None and not bool(need_hist)),
                    )
                    if need_hist:
                        probs, x_hist = denoise_out
                    elif traj_outputs_dir is not None:
                        probs, x_final = denoise_out
                    else:
                        probs = denoise_out

                    for j, b in enumerate(idxs):
                        traj_probs_by_sample[b].append(probs[j])
                        if traj_tokens_dir is not None:
                            # Save tokens logic (simplified)
                            save_path = os.path.join(str(traj_tokens_dir), f"sample_{int(sample_ids[b])}_traj{int(t_global)}.jsonl")
                            with open(save_path, "w", encoding="utf-8") as f:
                                # Meta info
                                meta = {"sample_id": int(sample_ids[b]), "input_path": str(input_paths[b]), "output_path": str(output_paths[b]), "noise_target": str(args.noise_target)}
                                f.write(json.dumps(meta) + "\n")
                                steps_done = int(x_hist[j].shape[0])
                                for s in range(steps_done):
                                     f.write(json.dumps({"step": s, "token_ids": x_hist[j][s].tolist()}) + "\n")
                        
                        if traj_outputs_dir is not None:
                            try:
                                if traj_tokens_dir is not None:
                                    if int(x_hist[j].shape[0]) <= 0: x_last = x_sub[j].detach().cpu().to(torch.long)
                                    else: x_last = torch.from_numpy(x_hist[j][-1]).to(torch.long)
                                    x_1xL = x_last.unsqueeze(0)
                                else:
                                    x_1xL = x_final[j].unsqueeze(0).to(torch.long)
                                am_1xL = attention_mask[b].unsqueeze(0).to(torch.bool)
                                sp = os.path.join(str(traj_outputs_dir), f"sample_{int(sample_ids[b])}_traj{int(t_global)}.png")
                                _save_reconstructed_image_from_sequence(
									backend=str(backend),
									state=state,
									tokenizer=tokenizer,
									input_ids_1xL=x_1xL,
									attention_mask_1xL=am_1xL,
									allow_token_mask=allow_token_mask,
									save_path=sp,
								)
                                pass 
                            except (OSError, RuntimeError, ValueError):
                                pass

    for b in range(len(batch)):
        trajectories = traj_probs_by_sample[b]
        allowed_positions = (
            pos_mask[b, allowed_range[0] : allowed_range[1]]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        record = {
            "sample_id": int(sample_ids[b]),
            args.input_key: str(input_paths[b]),
            args.output_key: str(output_paths[b]),
            args.instruction_key: str(instructions[b]),
            args.label_key: int(labels[b]),
            "backend": str(backend),
            "mask_ratio": float(args.mask_ratio),
            "steps_mode": str(args.steps_mode),
            "top_d": int(args.top_d),
        }

        if len(trajectories) < 2:
            record["status"] = "skipped_insufficient_trajectories"
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        evidence = score_probability_trajectories(
            np.stack(trajectories, axis=0),
            position_mask=allowed_positions,
            aggregation_topk=int(args.aggregation_topk),
            eps=float(args.eps),
        )
        record.update(
            {
                "status": "ok",
                "trajectory_evidence": evidence.to_dict(),
                "membership_score": evidence.membership_score,
            }
        )
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if probs_dir is not None:
            candidate_ids_for_sample = (
                candidate_ids[b, allowed_range[0] : allowed_range[1], :]
                .detach()
                .cpu()
                .to(torch.int32)
                .numpy()
            )
            probabilities = np.stack(trajectories, axis=0)
            probabilities = probabilities.astype(
                np.float32 if args.save_dtype == "float32" else np.float16
            )
            save_path = os.path.join(probs_dir, f"sample_{int(sample_ids[b])}.npz")
            np.savez_compressed(
                save_path,
                candidate_ids=candidate_ids_for_sample,
                trajectory_probabilities=probabilities,
                semantic_position_mask=allowed_positions,
                observed_token_log_probabilities=(
                    pos_logp[b, allowed_range[0] : allowed_range[1]].detach().cpu().numpy()
                ),
                allowed_range=np.asarray(allowed_range, dtype=np.int32),
            )


def _iter_valid_samples_shard(
    *,
    input_path: str,
    input_key: str,
    output_key: str,
    instruction_key: str,
    label_key: str,
    limit: Optional[int],
    rank: int,
    world_size: int,
) -> Iterator[Tuple[int, str, str, str, int]]:
    def _emit(sample_id: int, row: Dict[str, Any]) -> Optional[Tuple[int, str, str, str, int]]:
        inp = row.get(input_key, None)
        out = row.get(output_key, None)
        instr = row.get(instruction_key, None)
        lbl = row.get(label_key, None)
        if inp is None or out is None or instr is None or lbl is None: return None
        return sample_id, str(inp), str(out), str(instr), int(lbl)

    if input_path.endswith(".jsonl"):
        row_idx = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                if limit is not None and row_idx >= limit: break
                if (row_idx % world_size) == rank:
                    try:
                        out = _emit(row_idx, json.loads(line))
                        if out: yield out
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                row_idx += 1
        return

    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(".json input must contain a list of records")
        if limit is not None:
            rows = rows[:limit]
        for sample_id, row in enumerate(rows):
            if sample_id % world_size != rank or not isinstance(row, dict):
                continue
            sample = _emit(sample_id, row)
            if sample is not None:
                yield sample
        return

    raise ValueError("input_path must end with .json or .jsonl")


def _count_raw_records(*, input_path: str, limit: Optional[int], rank: int, world_size: int) -> int:
    if input_path.endswith(".json"):
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                rows = json.load(handle)
            if not isinstance(rows, list):
                return 0
            count = min(len(rows), limit) if limit is not None else len(rows)
            quotient, remainder = divmod(count, world_size)
            return quotient + int(rank < remainder)
        except (OSError, ValueError, json.JSONDecodeError):
            return 0
    n = 0
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                if (i % world_size) == rank:
                    n += 1
    except Exception:
        pass
    return n


def _run_worker(rank: int, world_size: int, device_str: str, args: Config, resume_total_override: Optional[int] = None) -> None:
    backend = str(args.backend)
    device = torch.device(device_str)
    if device.type == "cuda": torch.cuda.set_device(device)
    
    state, model, tokenizer, mask_id = _load_backend(
        backend=backend,
        pretrained_model_path=args.pretrained_model_path,
        vq_model_name=args.vq_model_name,
        device=device,
        dtype=_torch_dtype(args.torch_dtype),
    )
    
    text_vocab_end = _infer_text_vocab_end(backend=backend, model=model, tokenizer=tokenizer)
    image_vocab_size = 8192
    try:
        image_vocab_size = int(model.config.codebook_size)
    except (AttributeError, TypeError):
        pass
    vocab_size = _infer_model_vocab_size(model=model, device=device)
    
    allow_token_mask = _build_allow_token_mask(
        backend=backend,
        tokenizer=tokenizer,
        mask_id=mask_id,
        vocab_size=vocab_size,
        text_vocab_end=text_vocab_end,
        image_vocab_size=image_vocab_size,
    ).to(device)
    
    maskable_token_mask = _build_maskable_token_mask(
        backend=backend, tokenizer=tokenizer, mask_id=mask_id, vocab_size=vocab_size
    ).to(device)
    
    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    results_path = os.path.join(shard_dir, f"results_rank{rank}.jsonl")
    
    probs_dir = os.path.join(shard_dir, f"probs_rank{rank}")
    if args.save_probs: os.makedirs(probs_dir, exist_ok=True)
    
    traj_outputs_dir = os.path.join(shard_dir, f"traj_outputs_rank{rank}")
    if args.save_traj_outputs: os.makedirs(traj_outputs_dir, exist_ok=True)
    
    traj_tokens_dir = os.path.join(shard_dir, f"traj_tokens_rank{rank}")
    if args.save_traj_tokens: os.makedirs(traj_tokens_dir, exist_ok=True)

    processed_keys = set()
    if args.resume:
        processed_keys = _load_processed_record_keys_from_shard_dir(
            shard_dir=shard_dir, input_key=args.input_key, output_key=args.output_key,
            instruction_key=args.instruction_key, label_key=args.label_key
        )

    # Calculate total for tqdm
    total = _count_raw_records(input_path=args.input_path, limit=args.limit, rank=rank, world_size=world_size)
    
    sample_iter = _iter_valid_samples_shard(
        input_path=args.input_path,
        input_key=args.input_key,
        output_key=args.output_key,
        instruction_key=args.instruction_key,
        label_key=args.label_key,
        limit=args.limit,
        rank=rank,
        world_size=world_size
    )
    
    buffer = []
    rng_global = np.random.default_rng(args.seed + rank)
    
    pbar = tqdm(total=total, desc=f"rank{rank}", position=rank, leave=(rank==0), disable=not args.progress)
    
    with open(results_path, "a", encoding="utf-8") as out_f:
        for sample in sample_iter:
            key = (str(sample[1]), str(sample[2]), str(sample[3]), int(sample[4]))
            
            # Check if processed
            if args.resume and key in processed_keys:
                pbar.update(1)
                continue
            
            buffer.append(sample)
            if len(buffer) < args.batch_size: continue
            
            _process_batch(
                buffer, out_f=out_f, 
                probs_dir=probs_dir if args.save_probs else None, 
                traj_outputs_dir=traj_outputs_dir if args.save_traj_outputs else None, 
                traj_tokens_dir=traj_tokens_dir if args.save_traj_tokens else None,
                backend=backend, state=state, model=model, tokenizer=tokenizer, mask_id=mask_id,
                allow_token_mask=allow_token_mask, maskable_token_mask=maskable_token_mask,
                rng_global=rng_global, device=device, dtype=_torch_dtype(args.torch_dtype), args=args
            )
            pbar.update(len(buffer))
            buffer.clear()
        
        if buffer:
            _process_batch(
                buffer, out_f=out_f, 
                probs_dir=probs_dir if args.save_probs else None, 
                traj_outputs_dir=traj_outputs_dir if args.save_traj_outputs else None, 
                traj_tokens_dir=traj_tokens_dir if args.save_traj_tokens else None,
                backend=backend, state=state, model=model, tokenizer=tokenizer, mask_id=mask_id,
                allow_token_mask=allow_token_mask, maskable_token_mask=maskable_token_mask,
                rng_global=rng_global, device=device, dtype=_torch_dtype(args.torch_dtype), args=args
            )
            pbar.update(len(buffer))
    pbar.close()

def _parse_image_size(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s: return None
    s = s.lower().replace("*", "x").replace(",", "x")
    parts = s.split("x")
    if len(parts) == 2:
        return (int(parts[0].strip()), int(parts[1].strip()))
    return None


def _parse_steps(value: str) -> Tuple[str, float]:
    value = value.strip()
    if not value:
        raise ValueError("--steps cannot be empty")
    if any(character in value for character in (".", "e", "E")):
        relative_steps = float(value)
        if not 0.0 < relative_steps <= 1.0:
            raise ValueError("relative --steps must be in (0, 1]")
        return "rel", relative_steps
    absolute_steps = int(value)
    if absolute_steps < 1:
        raise ValueError("absolute --steps must be >= 1")
    return "abs", float(absolute_steps)

def _parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Run the gray-box OmniMIA image-to-image probing pathway."
    )
    p.add_argument("--backend", default="lumina", choices=["mmada", "lumina"])
    p.add_argument(
        "--input_path",
        "--meta_file",
        dest="input_path",
        required=True,
        help="Input .jsonl file with source image, target image, instruction, and label fields.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pretrained_model_path", required=True)
    p.add_argument("--vq_model_name", default=None)
    p.add_argument("--input_key", default="input_path")
    p.add_argument("--output_key", default="output_path")
    p.add_argument("--instruction_key", default="instruction")
    p.add_argument("--label_key", default="label")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=20480)
    p.add_argument("--devices", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--num_trajectories", type=int, default=4)
    p.add_argument("--steps", default="18")
    p.add_argument(
        "--top_d",
        "--topk",
        dest="top_d",
        type=int,
        default=32,
        help="Truncated probability-vector dimension d (default: 32).",
    )
    
    p.add_argument("--save_probs", dest="save_probs", action="store_true", default=True)
    p.add_argument("--no_save_probs", dest="save_probs", action="store_false")
    p.add_argument("--save_traj_outputs", dest="save_traj_outputs", action="store_true", default=True)
    p.add_argument("--no_save_traj_outputs", dest="save_traj_outputs", action="store_false")
    p.add_argument("--save_traj_tokens", dest="save_traj_tokens", action="store_true", default=True)
    p.add_argument("--no_save_traj_tokens", dest="save_traj_tokens", action="store_false")
    p.add_argument("--save_dtype", default="float16")
    p.add_argument(
        "--aggregation_topk",
        type=int,
        default=32,
        help="k in the paper's pessimistic/optimistic hierarchical aggregation (default: 32).",
    )
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--progress", dest="progress", action="store_true", default=True)
    p.add_argument("--no_progress", dest="progress", action="store_false")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    
    p.add_argument("--noise_target", default="output", choices=["input", "output", "both"], help="Region to apply noise/masking")
    p.add_argument("--image_size", default=None, help="Force image resolution WxH (e.g. 512x512). If not set, use original resolution (snapped to 32).")

    a = p.parse_args()
    
    mode, val = _parse_steps(a.steps)
    if not (0.0 <= float(a.mask_ratio) <= 1.0):
        raise ValueError("--mask_ratio must be in [0, 1]")
    if int(a.num_trajectories) < 2:
        raise ValueError("--num_trajectories must be >= 2")
    if int(a.top_d) < 1:
        raise ValueError("--top_d must be >= 1")
    if int(a.aggregation_topk) < 1:
        raise ValueError("--aggregation_topk must be >= 1")
    
    return Config(
        backend=str(a.backend),
        input_path=str(a.input_path),
        output_dir=str(a.output_dir),
        pretrained_model_path=str(a.pretrained_model_path),
        vq_model_name=a.vq_model_name,
        input_key=str(a.input_key),
        output_key=str(a.output_key),
        instruction_key=str(a.instruction_key),
        label_key=str(a.label_key),
        limit=a.limit,
        noise_target=str(a.noise_target),
        device=str(a.device),
        torch_dtype=str(a.torch_dtype),
        batch_size=int(a.batch_size),
        max_seq_len=int(a.max_seq_len),
        devices=a.devices,
        num_workers=a.num_workers,
        seed=int(a.seed),
        mask_ratio=float(a.mask_ratio),
        num_trajectories=int(a.num_trajectories),
        steps_mode=mode,
        steps_value=val,
        top_d=int(a.top_d),
        save_probs=bool(a.save_probs),
        save_traj_outputs=bool(a.save_traj_outputs),
        save_traj_tokens=bool(a.save_traj_tokens),
        save_dtype=str(a.save_dtype),
        aggregation_topk=int(a.aggregation_topk),
        eps=float(a.eps),
        progress=bool(a.progress),
        resume=bool(a.resume),
        image_size=_parse_image_size(a.image_size),
    )

def _merge_results(args: Config) -> float:
    """Write one standardized result file and return the pathway ROC-AUC."""
    shard_dir = os.path.join(args.output_dir, "_shards")
    result_paths = sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl")))
    records_by_id: Dict[int, Dict[str, Any]] = {}
    for result_path in result_paths:
        with open(result_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records_by_id[int(record["sample_id"])] = record

    labels: List[int] = []
    scores: List[float] = []
    merged_path = os.path.join(args.output_dir, "results.jsonl")
    with open(merged_path, "w", encoding="utf-8") as handle:
        for sample_id in sorted(records_by_id):
            record = records_by_id[sample_id]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if record.get("status") != "ok":
                continue
            labels.append(1 if int(record[args.label_key]) != 0 else 0)
            scores.append(float(record["membership_score"]))

    auc = binary_roc_auc(labels, scores)
    print(f"AUC={auc:.6f} (n={len(scores)})")
    return auc


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    devices = _parse_devices(args.devices)
    if devices:
        world_size = int(args.num_workers) if args.num_workers else len(devices)
        ctx = mp.get_context("spawn")
        procs = []
        for r in range(world_size):
            p = ctx.Process(target=_run_worker, args=(r, world_size, devices[r], args, None))
            p.start()
            procs.append(p)
        for process in procs:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"worker failed with exit code {process.exitcode}")
    else:
        _run_worker(0, 1, str(args.device), args)
        
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    _merge_results(args)

if __name__ == "__main__":
    main()
