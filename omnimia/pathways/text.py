import argparse
import glob
import json
import math
import os
import multiprocessing as mp
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from omnimia.evaluation import binary_roc_auc
from omnimia.trajectory import score_probability_trajectories


autocast = torch.autocast


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        raise ValueError(".json must be a list[dict] or {data: list[dict]}")
    raise ValueError("input_path must end with .json or .jsonl")


def _iter_valid_samples_shard(
    *,
    input_path: str,
    text_key: str,
    label_key: str,
    limit: Optional[int],
    rank: int,
    world_size: int,
) -> Iterator[Tuple[int, str, int]]:
    """Yield (global_sample_id, text, label_int) for this rank.

    - For .jsonl: streams line-by-line (memory friendly).
    - For .json: loads list and iterates.
    - limit: applied on raw record count (same as slicing `rows[:limit]`).
    - Sharding: by record index modulo world_size.
    """
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if not (0 <= rank < world_size):
        raise ValueError("rank must be in [0, world_size)")

    def _emit(sample_id: int, row: Dict[str, Any]) -> Optional[Tuple[int, str, int]]:
        text = row.get(text_key, None)
        label = row.get(label_key, None)
        if text is None or label is None:
            return None
        try:
            label_i = int(label)
        except Exception:
            return None
        return sample_id, str(text), int(label_i)

    if input_path.endswith(".jsonl"):
        row_idx = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and row_idx >= int(limit):
                    break
                if (row_idx % world_size) == rank:
                    row = json.loads(line)
                    out = _emit(row_idx, row)
                    if out is not None:
                        yield out
                row_idx += 1
        return

    rows = _read_json_or_jsonl(input_path)
    if limit is not None:
        rows = rows[: int(limit)]
    for row_idx, row in enumerate(rows):
        if (row_idx % world_size) != rank:
            continue
        out = _emit(row_idx, row)
        if out is not None:
            yield out


def _count_raw_records_for_rank(*, input_path: str, limit: Optional[int], rank: int, world_size: int) -> Optional[int]:
    """Best-effort total for tqdm progress bar.

    Notes:
    - Counts raw records/lines before filtering invalid samples.
    - For .jsonl this scans the file once (can be slow on huge files).
    - Returning None falls back to indeterminate tqdm (no bar).
    """
    try:
        if input_path.endswith(".json"):
            rows = _read_json_or_jsonl(input_path)
            n = len(rows[: int(limit)]) if limit is not None else len(rows)
            q, r = divmod(int(n), int(world_size))
            return int(q + (1 if int(rank) < int(r) else 0))

        if input_path.endswith(".jsonl"):
            n = 0
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if limit is not None and n >= int(limit):
                        break
                    # count even empty lines consistently with _iter_valid_samples_shard's row_idx logic
                    n += 1
            q, r = divmod(int(n), int(world_size))
            return int(q + (1 if int(rank) < int(r) else 0))
    except Exception:
        return None
    return None


def _load_processed_sample_ids(*, results_path: str) -> set[int]:
    """Load processed sample_id values from an existing results shard JSONL."""
    processed: set[int] = set()
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
                    sid = obj.get("sample_id", None)
                    if sid is None:
                        continue
                    processed.add(int(sid))
                except Exception:
                    continue
    except Exception:
        return processed
    return processed


def _load_processed_sample_ids_from_shard_dir(*, shard_dir: str) -> set[int]:
    """Load processed sample_id values from ALL shard result files."""
    processed: set[int] = set()
    if not shard_dir or not os.path.isdir(shard_dir):
        return processed
    for path in sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl"))):
        processed |= _load_processed_sample_ids(results_path=path)
    return processed


def _stable_record_hash(key: Tuple[str, int]) -> int:
    """Deterministic hash for repartitioning and resume.

    We intentionally do NOT use Python's built-in hash() because it's salted per-process.
    """
    import hashlib

    txt, lbl = key
    payload = (str(txt) + "\n" + str(int(lbl))).encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _load_processed_record_hashes(
    *,
    results_path: str,
    text_key: str,
    label_key: str,
) -> set[int]:
    """Load processed record hashes from an existing results shard JSONL.

    Prefers an explicit 'record_hash' field if present; otherwise hashes (text,label) from the row.
    """
    processed: set[int] = set()
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
                    rh = obj.get("record_hash", None)
                    if rh is not None:
                        processed.add(int(rh))
                        continue
                    txt = obj.get(str(text_key), None)
                    lbl = obj.get(str(label_key), None)
                    if txt is None or lbl is None:
                        continue
                    processed.add(int(_stable_record_hash((str(txt), int(lbl)))))
                except Exception:
                    continue
    except Exception:
        return processed
    return processed


def _load_processed_record_hashes_from_shard_dir(
    *,
    shard_dir: str,
    text_key: str,
    label_key: str,
) -> set[int]:
    """Load processed record hashes from ALL shard result files."""
    processed: set[int] = set()
    if not shard_dir or not os.path.isdir(shard_dir):
        return processed
    for path in sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl"))):
        processed |= _load_processed_record_hashes(results_path=path, text_key=text_key, label_key=label_key)
    return processed


def _count_assigned_samples_for_rank_resume(
    *,
    input_path: str,
    text_key: str,
    label_key: str,
    limit: Optional[int],
    rank: int,
    world_size: int,
) -> int:
    """Count how many valid samples are assigned to this rank under resume repartitioning."""
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if not (0 <= rank < world_size):
        raise ValueError("rank must be in [0, world_size)")

    n = 0
    for sid, txt, lbl in _iter_valid_samples_shard(
        input_path=str(input_path),
        text_key=str(text_key),
        label_key=str(label_key),
        limit=limit,
        rank=0,
        world_size=1,
    ):
        key = (str(txt), int(lbl))
        assigned_rank = int(_stable_record_hash(key) % int(world_size))
        if assigned_rank == int(rank):
            n += 1
    return int(n)


def _count_remaining_samples_for_rank_resume(
    *,
    input_path: str,
    text_key: str,
    label_key: str,
    limit: Optional[int],
    rank: int,
    world_size: int,
    processed_hashes: set[int],
) -> int:
    """Count how many valid samples remain for this rank under resume repartitioning.

    This counts only samples that:
    - are assigned to this rank by stable content hash modulo current world_size
    - and are NOT already present in any existing shard result file.
    """
    n = 0
    for _sid, _txt, _lbl, rh in _iter_valid_samples_resume(
        input_path=str(input_path),
        text_key=str(text_key),
        label_key=str(label_key),
        limit=limit,
        rank=int(rank),
        world_size=int(world_size),
    ):
        if processed_hashes and int(rh) in processed_hashes:
            continue
        n += 1
    return int(n)


def _count_remaining_samples_for_all_ranks_resume(
    *,
    input_path: str,
    text_key: str,
    label_key: str,
    limit: Optional[int],
    world_size: int,
    processed_hashes: set[int],
) -> List[int]:
    """Count remaining samples for every rank in one pass (resume repartitioning).

    This avoids N workers each scanning the full input, which can stall tqdm and GPU utilization.
    """
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    counts = [0 for _ in range(int(world_size))]
    for _sid, txt, lbl in _iter_valid_samples_shard(
        input_path=str(input_path),
        text_key=str(text_key),
        label_key=str(label_key),
        limit=limit,
        rank=0,
        world_size=1,
    ):
        rh = int(_stable_record_hash((str(txt), int(lbl))))
        if processed_hashes and rh in processed_hashes:
            continue
        r = int(rh % int(world_size))
        counts[r] += 1
    return [int(x) for x in counts]


def _iter_valid_samples_resume(
    *,
    input_path: str,
    text_key: str,
    label_key: str,
    limit: Optional[int],
    rank: int,
    world_size: int,
) -> Iterator[Tuple[int, str, int, int]]:
    """Yield (sample_id,row_text,label,record_hash) for this rank under resume repartitioning."""
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if not (0 <= rank < world_size):
        raise ValueError("rank must be in [0, world_size)")

    for sid, txt, lbl in _iter_valid_samples_shard(
        input_path=str(input_path),
        text_key=str(text_key),
        label_key=str(label_key),
        limit=limit,
        rank=0,
        world_size=1,
    ):
        key = (str(txt), int(lbl))
        rh = int(_stable_record_hash(key))
        if int(rh % int(world_size)) != int(rank):
            continue
        yield int(sid), str(txt), int(lbl), int(rh)


def _assert_resume_config_compatible(*, output_dir: str, current: Dict[str, Any]) -> None:
    """On resume, ensure config.json matches current args (except device/worker related fields)."""
    cfg_path = os.path.join(str(output_dir), "config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        return

    ignore = {"devices", "num_workers", "device", "progress", "resume"}
    diffs: List[str] = []
    for k, prev_v in prev.items() if isinstance(prev, dict) else []:
        if k in ignore:
            continue
        if k not in current:
            continue
        cur_v = current.get(k)
        if prev_v != cur_v:
            diffs.append(k)
    if diffs:
        diffs_sorted = ", ".join(sorted(diffs))
        raise RuntimeError(
            "Resume config mismatch (must match previous run except devices/num_workers/device/progress). "
            f"Mismatched keys: {diffs_sorted}. Previous={cfg_path}"
        )


def _parse_steps(value: str) -> Tuple[str, float]:
    """Returns (mode, val) where mode is 'abs' or 'rel'.

    - abs: integer steps >= 1 (e.g., '18')
    - rel: float in (0,1] (e.g., '0.5') meaning ceil(masked_tokens * rel)
    """
    v = value.strip()
    if not v:
        raise ValueError("--steps empty")
    if any(ch in v for ch in [".", "e", "E"]):
        rel = float(v)
        if not (0.0 < rel <= 1.0):
            raise ValueError("Relative --steps must be in (0,1]")
        return "rel", rel
    steps = int(v)
    if steps <= 0:
        raise ValueError("Absolute --steps must be >= 1")
    return "abs", float(steps)


def _choose_mask_positions(
    input_ids_1d: torch.LongTensor,
    *,
    mask_ratio: float,
    rng: np.random.Generator,
    maskable_token_mask: torch.BoolTensor,
) -> torch.BoolTensor:
    if input_ids_1d.ndim != 1:
        raise ValueError("input_ids_1d must be 1D")

    if maskable_token_mask.ndim != 1:
        raise ValueError("maskable_token_mask must be 1D")

    cand: List[int] = []
    allow_cpu = maskable_token_mask.detach().cpu().to(torch.bool)
    vocab_size = int(allow_cpu.numel())
    for i, tid in enumerate(input_ids_1d.tolist()):
        tid_i = int(tid)
        if 0 <= tid_i < vocab_size and bool(allow_cpu[tid_i].item()):
            cand.append(int(i))
    if not cand:
        return torch.zeros_like(input_ids_1d, dtype=torch.bool)

    k = int(round(len(cand) * float(mask_ratio)))
    if float(mask_ratio) > 0:
        k = max(1, k)
    k = min(k, len(cand))
    if k <= 0:
        return torch.zeros_like(input_ids_1d, dtype=torch.bool)

    idx = np.array(cand, dtype=np.int64)
    rng.shuffle(idx)
    chosen = idx[:k]

    mask = torch.zeros_like(input_ids_1d, dtype=torch.bool)
    mask[torch.from_numpy(chosen).to(mask.device)] = True
    return mask


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
) -> np.ndarray:
    """Run mask-based denoising and record per-step candidate-token probabilities.

    candidate_ids: [B, L, K] (fixed from clean forward)
    Returns: float32/float16 numpy array [B, steps, L, K]
    """

    if backend == "mmada":
        from models.MMaDA.modeling_mmada import add_gumbel_noise, get_num_transfer_tokens
    elif backend == "lumina":
        from models.LuminaDiMOO.utils.generation_utils import add_gumbel_noise, get_num_transfer_tokens
    else:
        raise ValueError(f"Unknown backend: {backend}")

    x = x_init.clone()
    start, end = allowed_range
    if start < 0 or end > x.shape[1] or start >= end:
        raise ValueError(f"Invalid allowed_range {allowed_range} for seq_len={x.shape[1]}")

    if candidate_ids.ndim != 3:
        raise ValueError(f"candidate_ids must be [B,L,K], got {tuple(candidate_ids.shape)}")
    if int(candidate_ids.shape[0]) != int(x.shape[0]) or int(candidate_ids.shape[1]) != int(x.shape[1]):
        raise ValueError(
            f"candidate_ids shape {tuple(candidate_ids.shape)} must match x_init {tuple(x.shape)} on [B,L]"
        )

    allowed = torch.zeros_like(x, dtype=torch.bool)
    allowed[:, start:end] = True
    mask_index0 = (x == int(mask_id)) & allowed
    num_transfer_tokens = get_num_transfer_tokens(mask_index0, int(steps))

    bsz = int(x.shape[0])
    l = int(end - start)
    k = int(candidate_ids.shape[-1])
    # store as float16 to reduce memory; caller can cast if needed
    out = np.zeros((bsz, int(steps), l, k), dtype=np.float16)

    x_hist: Optional[List[np.ndarray]] = [] if bool(return_x_history) else None

    for i in range(int(steps)):
        mask_index = (x == int(mask_id)) & allowed
        mask_allowed = mask_index[:, start:end]
        # if nothing is masked, we still record probabilities for consistency

        logits = model(x, attention_bias=attention_bias).logits
        logits_allowed_full = logits[:, start:end, :]

        # record candidate probabilities at this step
        logits_f = logits_allowed_full.to(torch.float32)
        denom = torch.logsumexp(logits_f, dim=-1, keepdim=True)  # [B,L,1]
        cand = candidate_ids[:, start:end, :].to(device=logits_f.device, dtype=torch.long)
        sel = torch.gather(logits_f, dim=-1, index=cand)  # [B,L,K]
        probs = torch.exp(sel - denom).to(torch.float16)
        out[:, i, :, :] = probs.detach().cpu().numpy()

        if int(mask_allowed.sum().item()) == 0:
            break

        logits_with_noise = add_gumbel_noise(logits, temperature=float(temperature))
        x0 = torch.argmax(logits_with_noise, dim=-1)

        x0_allowed = x0[:, start:end]

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
            if kb <= 0:
                continue
            b_sel = batch_ids == int(b)
            n_masked_b = int(b_sel.sum().item())
            if n_masked_b <= 0:
                continue
            kb = min(kb, n_masked_b)
            conf_b = logp_masked[b_sel]
            pos_b = pos_abs[b_sel]
            _, top_local = torch.topk(conf_b, k=int(kb))
            transfer_index[b, pos_b[top_local]] = True

        x[transfer_index] = x0[transfer_index]

        if x_hist is not None:
            x_hist.append(x.detach().cpu().to(torch.int32).numpy())

    if x_hist is None:
        return out

    # [steps_done, B, L] -> [B, steps_done, L]
    if len(x_hist) == 0:
        hist = np.zeros((bsz, 0, int(x.shape[1])), dtype=np.int32)
    else:
        hist = np.stack(x_hist, axis=0).transpose(1, 0, 2).astype(np.int32, copy=False)
    return out, hist


@torch.no_grad()
def _select_topk_candidates(
    *,
    model: Any,
    input_ids: torch.LongTensor,
    attention_bias: Optional[torch.Tensor],
    topk: int,
    allow_token_mask: torch.BoolTensor,
) -> torch.LongTensor:
    if topk <= 0:
        raise ValueError("topk must be > 0")
    logits = model(input_ids, attention_bias=attention_bias).logits  # [B,L,V]

    if allow_token_mask.ndim != 1:
        raise ValueError("allow_token_mask must be 1D")

    vocab_size = int(logits.size(-1))
    if int(allow_token_mask.numel()) != vocab_size:
        raise ValueError(f"allow_token_mask length {int(allow_token_mask.numel())} != vocab_size {vocab_size}")

    allow = allow_token_mask.to(device=logits.device, dtype=torch.bool)
    valid_count = int(allow.sum().item())
    if valid_count <= 0:
        raise ValueError(f"No valid tokens left for top-k selection (vocab_size={vocab_size})")
    if int(topk) > valid_count:
        raise ValueError(f"topk={topk} exceeds valid token count={valid_count} (vocab_size={vocab_size})")

    # mask out disallowed token ids before top-k
    min_val = torch.finfo(logits.dtype).min
    logits = logits.masked_fill((~allow).view(1, 1, -1), min_val)

    # Use logits directly (argmax of softmax is argmax of logits)
    _v, idx = torch.topk(logits, k=int(topk), dim=-1)
    return idx.to(torch.long)


def _select_topk_candidates_from_logits(
    logits: torch.Tensor,
    *,
    topk: int,
    allow_token_mask: torch.BoolTensor,
) -> torch.LongTensor:
    """Select top-k ids from precomputed logits [B,L,V] with allow_token_mask applied."""
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if allow_token_mask.ndim != 1:
        raise ValueError("allow_token_mask must be 1D")
    vocab_size = int(logits.size(-1))
    if int(allow_token_mask.numel()) != vocab_size:
        raise ValueError(f"allow_token_mask length {int(allow_token_mask.numel())} != vocab_size {vocab_size}")

    allow = allow_token_mask.to(device=logits.device, dtype=torch.bool)
    valid_count = int(allow.sum().item())
    if valid_count <= 0:
        raise ValueError(f"No valid tokens left for top-k selection (vocab_size={vocab_size})")
    if int(topk) > valid_count:
        raise ValueError(f"topk={topk} exceeds valid token count={valid_count} (vocab_size={vocab_size})")

    min_val = torch.finfo(logits.dtype).min
    logits_m = logits.masked_fill((~allow).view(1, 1, -1), min_val)
    _v, idx = torch.topk(logits_m, k=int(topk), dim=-1)
    return idx.to(torch.long)


def _infer_text_vocab_end(*, backend: str, model: Any, tokenizer: Any) -> Optional[int]:
    """Infer the exclusive upper bound of text-token ids.

    For multimodal tokenization, image/codebook tokens are typically appended after text vocab.
    We only want to operate on text-token ids (< text_vocab_end).
    """
    if backend == "lumina":
        try:
            from models.LuminaDiMOO.config import SPECIAL_TOKENS

            return int(dict(SPECIAL_TOKENS)["image_token_offset"])
        except Exception:
            # Fallback: infer from model vocab and known codebook size.
            try:
                from models.LuminaDiMOO.trajectory_prob import text_vocab_size

                return int(text_vocab_size(model=model, codebook_size=8192))
            except Exception:
                return None

    if backend == "mmada":
        # In MMaDA, image/codebook tokens are offset by len(text_tokenizer).
        try:
            return int(len(tokenizer))
        except Exception:
            return None

    return None


def _infer_model_vocab_size(*, model: Any, device: torch.device) -> int:
    # Best-effort without a forward pass.
    for attr in ["vocab_size", "llm_vocab_size"]:
        try:
            v = getattr(getattr(model, "config", None), attr, None)
            if v is not None:
                vi = int(v)
                if vi > 0:
                    return vi
        except Exception:
            pass

    # Try embeddings
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            n = int(getattr(emb, "num_embeddings", 0))
            if n > 0:
                return n
    except Exception:
        pass

    # Fallback: minimal forward pass
    x = torch.zeros((1, 1), device=device, dtype=torch.long)
    attention_bias = torch.ones((1, 1, 1, 1), device=device, dtype=torch.bool)
    logits = model(x, attention_bias=attention_bias).logits
    return int(logits.size(-1))


def _collect_forbid_token_ids(*, backend: str, tokenizer: Any, mask_id: int) -> set[int]:
    """Token ids that should be treated as special/control and excluded.

    This is used to build both:
    - allow_token_mask (domain-limited + excludes forbid ids)
    - maskable_token_mask (all ids + excludes forbid ids)
    """
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
            from models.MMaDA.pipeline import _SPECIAL_TOKENS  # type: ignore

            for tok in list(_SPECIAL_TOKENS):
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                except Exception:
                    tid = None
                if tid is None:
                    continue
                forbid.add(int(tid))
        except Exception:
            pass

    return forbid


def _build_allow_token_mask(
    *,
    backend: str,
    tokenizer: Any,
    mask_id: int,
    vocab_size: int,
    topk_domain: str,
    text_vocab_end: Optional[int],
) -> torch.BoolTensor:
    """Build a 1D boolean mask over token ids indicating which ids are allowed.

    This mask is the only thing downstream code needs for:
    - choosing which positions can be masked/compared (based on token id)
    - selecting top-k candidate token ids
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    domain = str(topk_domain)
    if domain not in {"text", "image", "all"}:
        raise ValueError(f"Unknown topk_domain: {topk_domain}")

    allow = torch.zeros((int(vocab_size),), dtype=torch.bool)
    if domain == "all":
        allow[:] = True
    elif domain == "text":
        end = int(text_vocab_end) if text_vocab_end is not None else int(vocab_size)
        end = max(0, min(end, int(vocab_size)))
        allow[:end] = True
    else:  # image
        if text_vocab_end is None:
            raise ValueError("topk_domain=image requires text_vocab_end, but got None")
        start = max(0, min(int(text_vocab_end), int(vocab_size)))
        allow[start:] = True

    forbid = _collect_forbid_token_ids(backend=backend, tokenizer=tokenizer, mask_id=int(mask_id))

    for tid in forbid:
        if 0 <= int(tid) < int(vocab_size):
            allow[int(tid)] = False

    if int(allow.sum().item()) <= 0:
        raise ValueError(
            f"allow_token_mask is empty after filtering (domain={domain}, vocab_size={vocab_size}, text_vocab_end={text_vocab_end})"
        )

    return allow


def _build_maskable_token_mask(
    *,
    backend: str,
    tokenizer: Any,
    mask_id: int,
    vocab_size: int,
) -> torch.BoolTensor:
    """Token-id mask for selecting positions to mask.

    Requirement: can mask any token except padding/special/control tokens.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    maskable = torch.ones((int(vocab_size),), dtype=torch.bool)
    forbid = _collect_forbid_token_ids(backend=backend, tokenizer=tokenizer, mask_id=int(mask_id))
    for tid in forbid:
        if 0 <= int(tid) < int(vocab_size):
            maskable[int(tid)] = False
    if int(maskable.sum().item()) <= 0:
        raise ValueError(f"maskable_token_mask is empty after filtering (vocab_size={vocab_size})")
    return maskable


@dataclass
class Config:
    backend: str
    input_path: str
    output_dir: str
    pretrained_model_path: str
    config: Optional[str]

    text_key: str
    label_key: str
    limit: Optional[int]

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
    topk_domain: str
    save_probs: bool
    save_denoise: bool
    save_dtype: str

    aggregation_topk: int
    eps: float

    progress: bool

    resume: bool


def _parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=(
            "Text-only multi-trajectory diffusion denoise logging: \
" "mask->unmask prob density per step -> trajectory similarity -> AUC"
        )
    )

    p.add_argument("--backend", default="mmada", choices=["mmada", "lumina"], help="Model backend")
    p.add_argument("--input_path", required=True, help=".json or .jsonl file containing text+label")
    p.add_argument("--output_dir", required=True, help="Where to write results")

    p.add_argument("--pretrained_model_path", required=True, help="Pretrained model path")
    p.add_argument("--config", default=None, help="Optional YAML config (MMaDA only; ignored by Lumina)")

    p.add_argument("--text_key", default="text")
    p.add_argument("--label_key", default="label")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=1024)

    p.add_argument(
        "--top_d",
        "--topk",
        dest="top_d",
        type=int,
        default=32,
        help="Truncated probability-vector dimension d (default: 32).",
    )
    p.add_argument(
        "--topk_domain",
        type=str,
        default="text",
        choices=["text", "image", "all"],
        help=(
            "Which vocabulary segment to select top-k candidates from. "
            "text: [0, text_vocab_end) (default). "
            "image: [text_vocab_end, vocab_size) (requires text_vocab_end). "
            "all: full vocab (special/control tokens are still excluded)."
        ),
    )
    p.add_argument(
        "--save_probs",
        dest="save_probs",
        action="store_true",
        default=True,
        help="Save per-step candidate probabilities for all trajectories (default: enabled)",
    )
    p.add_argument(
        "--no_save_probs",
        dest="save_probs",
        action="store_false",
        help="Do not save per-step candidate probabilities",
    )

    p.add_argument(
        "--save_denoise",
        dest="save_denoise",
        action="store_true",
        default=True,
        help=(
            "Save denoise results for each trajectory: decoded text at every denoising step (default: enabled). "
            "Files are written under output_dir/_shards/denoise_rank*/."
        ),
    )
    p.add_argument(
        "--no_save_denoise",
        dest="save_denoise",
        action="store_false",
        help="Disable saving per-step denoise decoded texts",
    )
    p.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])

    p.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices for multi-process run, e.g. 'cuda:0,cuda:1'. If set, one process per device.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override number of worker processes (default: len(devices)). Only used when --devices is set.",
    )

    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--num_trajectories", type=int, default=4)
    p.add_argument(
        "--steps",
        type=str,
        default="18",
        help=("Denoising steps: integer (abs) or float in (0,1] (rel to #masked tokens). E.g. 18 or 0.5"),
    )

    p.add_argument(
        "--aggregation_topk",
        type=int,
        default=32,
        help="k in the paper's pessimistic/optimistic hierarchical aggregation (default: 32).",
    )
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show tqdm progress bars (default: enabled)",
    )
    p.add_argument(
        "--no_progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm progress bars",
    )

    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help=(
            "Resume from existing shard results in output_dir: skip already-processed sample_id and append. "
            "This is safe across worker-count changes because sample_id is the input row index."
        ),
    )
    p.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Disable resume behavior (overwrite shard outputs)",
    )

    a = p.parse_args()
    mode, val = _parse_steps(a.steps)

    if not (0.0 <= float(a.mask_ratio) <= 1.0):
        raise ValueError("--mask_ratio must be in [0,1]")
    if int(a.num_trajectories) < 2:
        raise ValueError("--num_trajectories must be >= 2 (need trajectory similarity)")
    if int(a.top_d) < 1:
        raise ValueError("--top_d must be >= 1")
    if int(a.aggregation_topk) < 1:
        raise ValueError("--aggregation_topk must be >= 1")

    return Config(
        backend=str(a.backend),
        input_path=a.input_path,
        output_dir=a.output_dir,
        pretrained_model_path=a.pretrained_model_path,
        config=a.config,
        text_key=a.text_key,
        label_key=a.label_key,
        limit=a.limit,
        device=a.device,
        torch_dtype=a.torch_dtype,
        batch_size=int(a.batch_size),
        max_seq_len=int(a.max_seq_len),
        devices=a.devices,
        num_workers=a.num_workers,
        seed=int(a.seed),
        mask_ratio=float(a.mask_ratio),
        num_trajectories=int(a.num_trajectories),
        steps_mode=mode,
        steps_value=float(val),
        top_d=int(a.top_d),
        topk_domain=str(a.topk_domain),
        save_probs=bool(a.save_probs),
        save_denoise=bool(a.save_denoise),
        save_dtype=str(a.save_dtype),
        aggregation_topk=int(a.aggregation_topk),
        eps=float(a.eps),
        progress=bool(a.progress),

        resume=bool(a.resume),
    )


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
    if not items:
        return None
    return items


def _load_backend_model_and_tokenizer(
    *,
    backend: str,
    pretrained_model_path: str,
    config: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, Any, int]:
    """Returns (model, tokenizer, mask_id)."""
    if backend == "mmada":
        from models.MMaDA.pipeline import MMaDAPipeline

        pipe = MMaDAPipeline.from_pretrained(
            task="lm",
            pretrained_model_path=pretrained_model_path,
            config=config,
            device=device,
            torch_dtype=dtype,
            padding_side="right",
        )
        model = pipe.model
        tokenizer = pipe.tokenizer
        mask_id = int(getattr(model.config, "mask_token_id", None) or getattr(pipe.model.config, "mask_token_id"))
        return model, tokenizer, mask_id

    if backend == "lumina":
        from transformers import AutoTokenizer
        from models.LuminaDiMOO.modeling_xllmx_dimoo import LLaDAForMultiModalGeneration
        from models.LuminaDiMOO.trajectory_prob import mask_token_id as lumina_mask_token_id

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration.from_pretrained(
            pretrained_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            device_map=None,
        ).to(device)
        model.eval()
        mask_id = int(lumina_mask_token_id())
        return model, tokenizer, mask_id

    raise ValueError(f"Unknown backend: {backend}")


def _run_worker(
    rank: int,
    world_size: int,
    device_str: str,
    args: Config,
    resume_total_override: Optional[int] = None,
) -> None:
    """One worker process on one GPU/device."""
    backend = str(args.backend)
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype = _torch_dtype(args.torch_dtype)
    seed = int(args.seed) + int(rank) * 1000003
    rng_global = np.random.default_rng(seed)

    model, tokenizer, mask_id = _load_backend_model_and_tokenizer(
        backend=backend,
        pretrained_model_path=args.pretrained_model_path,
        config=args.config,
        device=device,
        dtype=dtype,
    )

    text_vocab_end = _infer_text_vocab_end(backend=backend, model=model, tokenizer=tokenizer)
    vocab_size = _infer_model_vocab_size(model=model, device=device)
    allow_token_mask = _build_allow_token_mask(
        backend=backend,
        tokenizer=tokenizer,
        mask_id=int(mask_id),
        vocab_size=int(vocab_size),
        topk_domain=str(args.topk_domain),
        text_vocab_end=text_vocab_end,
    ).to(device=device)

    maskable_token_mask = _build_maskable_token_mask(
        backend=backend,
        tokenizer=tokenizer,
        mask_id=int(mask_id),
        vocab_size=int(vocab_size),
    ).to(device=device)

    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    results_path = os.path.join(shard_dir, f"results_rank{rank}.jsonl")

    processed_hashes: set[int] = set()
    if bool(args.resume):
        processed_hashes = _load_processed_record_hashes_from_shard_dir(
            shard_dir=shard_dir,
            text_key=str(args.text_key),
            label_key=str(args.label_key),
        )
        if processed_hashes:
            print(f"[resume] rank{rank}: loaded {len(processed_hashes)} processed record hashes from {shard_dir}")

    probs_dir = os.path.join(shard_dir, f"probs_rank{rank}")
    if bool(args.save_probs):
        os.makedirs(probs_dir, exist_ok=True)

    denoise_dir = os.path.join(shard_dir, f"denoise_rank{rank}")
    if bool(args.save_denoise):
        os.makedirs(denoise_dir, exist_ok=True)

    batch_size = max(1, int(args.batch_size))

    # Stream samples for this rank
    if bool(args.resume):
        sample_iter_resume = _iter_valid_samples_resume(
            input_path=str(args.input_path),
            text_key=str(args.text_key),
            label_key=str(args.label_key),
            limit=args.limit,
            rank=int(rank),
            world_size=int(world_size),
        )
    else:
        sample_iter = _iter_valid_samples_shard(
            input_path=args.input_path,
            text_key=args.text_key,
            label_key=args.label_key,
            limit=args.limit,
            rank=rank,
            world_size=world_size,
        )

    # Multi-process friendly progress bars.
    # - position=rank keeps each process on its own line.
    # - Use --no_progress to disable explicitly.
    if bool(args.resume):
        if resume_total_override is not None:
            total = int(resume_total_override)
        else:
            total = _count_remaining_samples_for_rank_resume(
                input_path=str(args.input_path),
                text_key=str(args.text_key),
                label_key=str(args.label_key),
                limit=args.limit,
                rank=int(rank),
                world_size=int(world_size),
                processed_hashes=processed_hashes,
            )
    else:
        total = _count_raw_records_for_rank(
            input_path=args.input_path,
            limit=args.limit,
            rank=rank,
            world_size=world_size,
        )
    pbar = tqdm(
        total=total,
        desc=f"rank{rank}",
        position=int(rank),
        leave=(int(rank) == 0),
        dynamic_ncols=True,
        disable=not bool(args.progress),
    )

    buffer: List[Tuple[int, str, int]] = []
    try:
        open_mode = "a" if bool(args.resume) and os.path.exists(results_path) else "w"
        with open(results_path, open_mode, encoding="utf-8") as out_f:
            if bool(args.resume):
                for sid, txt, lbl, rh in sample_iter_resume:
                    if processed_hashes and int(rh) in processed_hashes:
                        continue
                    buffer.append((int(sid), str(txt), int(lbl)))
                    if len(buffer) < batch_size:
                        continue

                    _process_batch(
                        buffer,
                        out_f=out_f,
                        probs_dir=probs_dir if bool(args.save_probs) else None,
                        denoise_dir=denoise_dir if bool(args.save_denoise) else None,
                        backend=backend,
                        model=model,
                        tokenizer=tokenizer,
                        mask_id=int(mask_id),
                        allow_token_mask=allow_token_mask,
                        maskable_token_mask=maskable_token_mask,
                        rng_global=rng_global,
                        device=device,
                        dtype=dtype,
                        args=args,
                    )
                    pbar.update(len(buffer))
                    buffer.clear()
            else:
                for sample in sample_iter:
                    buffer.append(sample)
                    if len(buffer) < batch_size:
                        continue

                    _process_batch(
                        buffer,
                        out_f=out_f,
                        probs_dir=probs_dir if bool(args.save_probs) else None,
                        denoise_dir=denoise_dir if bool(args.save_denoise) else None,
                        backend=backend,
                        model=model,
                        tokenizer=tokenizer,
                        mask_id=int(mask_id),
                        allow_token_mask=allow_token_mask,
                        maskable_token_mask=maskable_token_mask,
                        rng_global=rng_global,
                        device=device,
                        dtype=dtype,
                        args=args,
                    )
                    pbar.update(len(buffer))
                    buffer.clear()

            if buffer:
                _process_batch(
                    buffer,
                    out_f=out_f,
                    probs_dir=probs_dir if bool(args.save_probs) else None,
                    denoise_dir=denoise_dir if bool(args.save_denoise) else None,
                    backend=backend,
                    model=model,
                    tokenizer=tokenizer,
                    mask_id=int(mask_id),
                    allow_token_mask=allow_token_mask,
                    maskable_token_mask=maskable_token_mask,
                    rng_global=rng_global,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
                pbar.update(len(buffer))
    finally:
        pbar.close()


def _process_batch(
    batch: Sequence[Tuple[int, str, int]],
    *,
    out_f,
    probs_dir: Optional[str],
    denoise_dir: Optional[str],
    backend: str,
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
    texts = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(args.max_seq_len),
        add_special_tokens=True,
    )
    input_ids = tok["input_ids"].to(device=device, dtype=torch.long)  # [B,L]
    attn_mask = tok.get("attention_mask", None)
    if attn_mask is None:
        attn_mask = torch.ones_like(input_ids, dtype=torch.long)
    attention_mask = attn_mask.to(device=device, dtype=torch.bool)
    seq_len = int(input_ids.shape[1])
    attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).unsqueeze(1)

    # clean forward: select top-k candidates per position
    with torch.no_grad():
        with (autocast("cuda", dtype=dtype) if device.type == "cuda" else autocast("cpu", enabled=False)):
            logits_clean = model(input_ids, attention_bias=attention_bias).logits  # [B,L,V]
            candidate_ids = _select_topk_candidates_from_logits(
                logits_clean,
                topk=int(args.top_d),
                allow_token_mask=allow_token_mask,
            )  # [B,L,K]

            # Per-position clean-forward log-likelihood of the observed token id.
            # pos_logp[b, pos] = log p(x[pos] | x) under the clean forward pass.
            logits_f = logits_clean.to(torch.float32)
            logp = logits_f - torch.logsumexp(logits_f, dim=-1, keepdim=True)  # [B,L,V]
            pos_logp = torch.gather(logp, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B,L]
            pos_logp = pos_logp.to(torch.float16)

    # restrict similarity to non-pad + non-special tokens (same filter as masking)
    allowed_pos_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    allow_cpu = allow_token_mask.detach().cpu().to(torch.bool)
    vocab_size = int(allow_cpu.numel())
    for b in range(int(input_ids.shape[0])):
        ids = input_ids[b].detach().cpu().to(torch.long)
        for pos, tid in enumerate(ids.tolist()):
            if not bool(attention_mask[b, pos].item()):
                continue
            tid_i = int(tid)
            if 0 <= tid_i < vocab_size and bool(allow_cpu[tid_i].item()):
                allowed_pos_mask[b, pos] = True

    traj_probs_by_sample: List[List[np.ndarray]] = [[] for _ in range(len(batch))]
    traj_xhist_by_sample: Optional[List[List[np.ndarray]]] = None
    if denoise_dir is not None:
        traj_xhist_by_sample = [[] for _ in range(len(batch))]

    for _t in range(int(args.num_trajectories)):
        mask_pos = torch.zeros_like(input_ids, dtype=torch.bool)
        masked_counts: List[int] = []
        for b in range(int(input_ids.shape[0])):
            rng = np.random.default_rng(int(rng_global.integers(0, 2**31 - 1)))
            m = _choose_mask_positions(
                input_ids[b].detach().cpu(),
                mask_ratio=args.mask_ratio,
                rng=rng,
                maskable_token_mask=maskable_token_mask,
            )
            mask_pos[b] = m.to(device=device)
            masked_counts.append(int(m.sum().item()))

        if int(mask_pos.sum().item()) == 0:
            continue

        x_init = input_ids.clone()
        x_init[mask_pos] = int(mask_id)

        steps_per_sample: List[int] = []
        for mc in masked_counts:
            if mc <= 0:
                steps_per_sample.append(0)
            elif args.steps_mode == "abs":
                steps_per_sample.append(int(args.steps_value))
            else:
                steps_per_sample.append(max(1, int(math.ceil(int(mc) * float(args.steps_value)))))

        buckets: Dict[int, List[int]] = {}
        for b, st in enumerate(steps_per_sample):
            if st <= 0:
                continue
            buckets.setdefault(int(st), []).append(int(b))

        with torch.no_grad():
            with (autocast("cuda", dtype=dtype) if device.type == "cuda" else autocast("cpu", enabled=False)):
                for st, idxs in buckets.items():
                    x_sub = x_init[idxs]
                    ab_sub = attention_bias[idxs]
                    cand_sub = candidate_ids[idxs]
                    denoise_out = _denoise_candidate_prob_trajectories(
                        backend=backend,
                        model=model,
                        x_init=x_sub,
                        candidate_ids=cand_sub,
                        mask_id=int(mask_id),
                        steps=int(st),
                        allowed_range=(0, seq_len),
                        temperature=1.0,
                        attention_bias=ab_sub,
                        return_x_history=(denoise_dir is not None),
                    )
                    if denoise_dir is not None:
                        probs, x_hist = denoise_out  # type: ignore[misc]
                    else:
                        probs = denoise_out  # type: ignore[assignment]
                    for j, b in enumerate(idxs):
                        traj_probs_by_sample[b].append(probs[j])  # [S,L,K]
                        if traj_xhist_by_sample is not None:
                            traj_xhist_by_sample[b].append(x_hist[j])  # [S_done, L]

    for b in range(len(batch)):
        traj_probs = traj_probs_by_sample[b]
        if len(traj_probs) < 2:
            continue

        allowed = allowed_pos_mask[b].detach().cpu().numpy().astype(bool)
        evidence = score_probability_trajectories(
            np.stack(traj_probs, axis=0),
            position_mask=allowed,
            aggregation_topk=int(args.aggregation_topk),
            eps=float(args.eps),
        )

        rec = {
            "sample_id": int(sample_ids[b]),
            args.text_key: texts[b],
            args.label_key: int(labels[b]),
            "record_hash": int(_stable_record_hash((str(texts[b]), int(labels[b])))),
            "trajectory_evidence": evidence.to_dict(),
            "membership_score": evidence.membership_score,
            "steps_mode": args.steps_mode,
            "mask_ratio": float(args.mask_ratio),
            "backend": backend,
            "top_d": int(args.top_d),
            "topk_domain": str(args.topk_domain),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Optional: save all trajectories' per-step candidate probs
        if probs_dir is not None:
            true_len = int(attention_mask[b].sum().detach().cpu().item())
            cand_ids_b = candidate_ids[b, :true_len, :].detach().cpu().to(torch.int32).numpy()
            probs_stack = np.stack([p[:, :true_len, :] for p in traj_probs], axis=0)  # [T,S,L,K]
            if str(args.save_dtype) == "float32":
                probs_stack = probs_stack.astype(np.float32)
            else:
                probs_stack = probs_stack.astype(np.float16)
            save_path = os.path.join(probs_dir, f"sample_{int(sample_ids[b])}.npz")
            allowed_pos = allowed_pos_mask[b, :true_len].detach().cpu().numpy().astype(np.bool_)
            pos_logp_b = pos_logp[b, :true_len].detach().cpu().numpy()
            np.savez_compressed(
                save_path,
                candidate_ids=cand_ids_b,
                trajectory_probabilities=probs_stack,
                semantic_position_mask=allowed_pos,
                observed_token_log_probabilities=pos_logp_b,
            )

        # Optional: save per-step denoised tokens+decoded text for each trajectory (JSONL)
        if denoise_dir is not None and traj_xhist_by_sample is not None:
            xh_traj = traj_xhist_by_sample[b]
            if xh_traj:
                true_len = int(attention_mask[b].sum().detach().cpu().item())
                save_path = os.path.join(denoise_dir, f"sample_{int(sample_ids[b])}.jsonl")
                with open(save_path, "w", encoding="utf-8") as f:
                    # meta line
                    meta = {
                        "type": "meta",
                        "sample_id": int(sample_ids[b]),
                        args.text_key: str(texts[b]),
                        args.label_key: int(labels[b]),
                        "backend": str(backend),
                        "mask_id": int(mask_id),
                        "mask_ratio": float(args.mask_ratio),
                        "top_d": int(args.top_d),
                        "steps_mode": str(args.steps_mode),
                        "steps_value": float(args.steps_value),
                        "true_len": int(true_len),
                        "num_trajectories": int(len(xh_traj)),
                    }
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                    for t_idx, xhist in enumerate(xh_traj):
                        steps_done = int(xhist.shape[0])
                        for s in range(steps_done):
                            token_ids = [int(x) for x in xhist[s, :true_len].tolist()]
                            try:
                                txt_step = tokenizer.decode(token_ids, skip_special_tokens=True)
                            except Exception:
                                txt_step = str(token_ids[:200])
                            n_mask = 0
                            try:
                                n_mask = int(np.sum(np.asarray(token_ids, dtype=np.int64) == int(mask_id)))
                            except Exception:
                                n_mask = 0
                            rec_step = {
                                "type": "step",
                                "sample_id": int(sample_ids[b]),
                                "trajectory_id": int(t_idx),
                                "step": int(s),
                                "token_ids": token_ids,
                                "text": str(txt_step),
                                "num_mask_tokens": int(n_mask),
                            }
                            f.write(json.dumps(rec_step, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()

    backend = str(args.backend)

    os.makedirs(args.output_dir, exist_ok=True)

    if bool(args.resume):
        _assert_resume_config_compatible(output_dir=str(args.output_dir), current=vars(args))

    devices = _parse_devices(args.devices)
    if devices:
        world_size = int(args.num_workers) if args.num_workers is not None else len(devices)
        world_size = min(world_size, len(devices))
        if world_size <= 0:
            raise ValueError("--num_workers must be > 0")
        if args.num_workers is not None and int(args.num_workers) < int(len(devices)):
            try:
                print(
                    f"[warn] --num_workers={int(args.num_workers)} < len(--devices)={len(devices)}; "
                    f"only first {world_size} devices will be used: {devices[:world_size]}"
                )
            except Exception:
                pass

        resume_totals: Optional[List[int]] = None
        if bool(args.resume):
            shard_dir = os.path.join(args.output_dir, "_shards")
            processed_hashes = _load_processed_record_hashes_from_shard_dir(
                shard_dir=shard_dir,
                text_key=str(args.text_key),
                label_key=str(args.label_key),
            )
            resume_totals = _count_remaining_samples_for_all_ranks_resume(
                input_path=str(args.input_path),
                text_key=str(args.text_key),
                label_key=str(args.label_key),
                limit=args.limit,
                world_size=int(world_size),
                processed_hashes=processed_hashes,
            )
            try:
                print(f"[resume] remaining per rank (world_size={world_size}): {resume_totals}")
            except Exception:
                pass

        ctx = mp.get_context("spawn")
        procs: List[mp.Process] = []
        for r in range(world_size):
            tot_r: Optional[int] = None
            if resume_totals is not None and 0 <= int(r) < len(resume_totals):
                tot_r = int(resume_totals[int(r)])
            p = ctx.Process(target=_run_worker, args=(r, world_size, devices[r], args, tot_r))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

        # merge shards + compute AUC
        shard_dir = os.path.join(args.output_dir, "_shards")
        merged_path = os.path.join(args.output_dir, "results.jsonl")
        y_true: List[int] = []
        y_score: List[float] = []
        with open(merged_path, "w", encoding="utf-8") as out_f:
            paths: List[str]
            if bool(args.resume):
                paths = sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl")))
            else:
                paths = [os.path.join(shard_dir, f"results_rank{r}.jsonl") for r in range(world_size)]
            for rp in paths:
                if not os.path.exists(rp):
                    continue
                with open(rp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        out_f.write(line + "\n")
                        try:
                            obj = json.loads(line)
                            y_true.append(1 if int(obj.get(args.label_key, 0)) != 0 else 0)
                            y_score.append(float(obj.get("membership_score", 0.0)))
                        except Exception:
                            continue

        auc = binary_roc_auc(y_true, y_score)
        print(f"AUC={auc:.6f} (n={len(y_true)})")
    else:
        # single-process path: run as rank0 on args.device
        _run_worker(0, 1, str(args.device), args)
        # compute AUC from shard file
        shard_dir = os.path.join(args.output_dir, "_shards")
        merged_path = os.path.join(args.output_dir, "results.jsonl")
        y_true: List[int] = []
        y_score: List[float] = []
        with open(merged_path, "w", encoding="utf-8") as out_f:
            if bool(args.resume):
                paths = sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl")))
                for rp in paths:
                    if not os.path.exists(rp):
                        continue
                    with open(rp, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            out_f.write(line + "\n")
                            try:
                                obj = json.loads(line)
                                y_true.append(1 if int(obj.get(args.label_key, 0)) != 0 else 0)
                                y_score.append(float(obj.get("membership_score", 0.0)))
                            except Exception:
                                continue
            else:
                rp0 = os.path.join(shard_dir, "results_rank0.jsonl")
                with open(rp0, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        out_f.write(line + "\n")
                        try:
                            obj = json.loads(line)
                            y_true.append(1 if int(obj.get(args.label_key, 0)) != 0 else 0)
                            y_score.append(float(obj.get("membership_score", 0.0)))
                        except Exception:
                            continue
        auc = binary_roc_auc(y_true, y_score)
        print(f"AUC={auc:.6f} (n={len(y_true)})")

    # save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
