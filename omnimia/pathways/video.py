#!/usr/bin/env python3

import argparse
import copy
from contextlib import contextmanager
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import traceback
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import decord
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import CLIPModel, CLIPProcessor

# Import OmniVideo modules from the sibling Omni-Video-main checkout.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
_OMNI_ROOT = os.path.join(_REPO_ROOT, "Omni-Video-main")
if os.path.isdir(_OMNI_ROOT) and _OMNI_ROOT not in sys.path:
    sys.path.insert(0, _OMNI_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omnivideo.configs import SIZE_CONFIGS, WAN_CONFIGS
from omnivideo.utils.utils import cache_video, str2bool
from omnivideo.x2x_gen_unified_1_3B import OmniVideoX2XUnified1_3B
import omnivideo.x2x_gen_unified as omni_x2x_module


def _init_logging(output_root: str) -> None:
    os.makedirs(output_root, exist_ok=True)
    log_path = os.path.join(output_root, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    logging.info("Logging initialized: %s", log_path)


def _align_to_4n_plus_1(x: int) -> int:
    if x <= 1:
        return 1
    return int(math.ceil((x - 1) / 4.0) * 4 + 1)


def _resolve_clip_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _clip_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    return torch.float32


def _validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"input_jsonl not found: {args.input_jsonl}")
    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(f"ckpt_dir not found: {args.ckpt_dir}")
    if args.num_repeats < 2:
        raise ValueError("num_repeats must be >= 2")
    if args.prefix_frames < 1:
        raise ValueError("prefix_frames must be >= 1")
    if args.continuation_frames < 1:
        raise ValueError("continuation_frames must be >= 1")
    if args.input_sampling_rate < 1:
        raise ValueError("input_sampling_rate must be >= 1")
    if args.sample_steps < 1:
        raise ValueError("sample_steps must be >= 1")
    if args.clip_batch_size < 1:
        raise ValueError("clip_batch_size must be >= 1")
    if args.seed_stride < 1:
        raise ValueError("seed_stride must be >= 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be >= 1 when set")
    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if args.continuation_noise_level < 0.0 or args.continuation_noise_level > 1.0:
        raise ValueError("continuation_noise_level must be in [0, 1]")
    if args.denoise_strength < 0.0 or args.denoise_strength > 1.0:
        raise ValueError("denoise_strength must be in [0, 1]")
    if args.start_step_index is not None and args.start_step_index < 0:
        raise ValueError("start_step_index must be >= 0 when set")

    if args.new_checkpoint is None or str(args.new_checkpoint).strip() == "":
        default_ckpt = os.path.join(args.ckpt_dir, "transformer", "pytorch_model.pt")
        if os.path.exists(default_ckpt):
            args.new_checkpoint = default_ckpt
    if args.new_checkpoint is not None and not os.path.exists(args.new_checkpoint):
        raise FileNotFoundError(f"new_checkpoint not found: {args.new_checkpoint}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OmniVideo inference")

    parsed_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if len(parsed_gpu_ids) == 0:
        raise RuntimeError("No usable GPU ids were found")
    if args.num_workers > len(parsed_gpu_ids):
        raise ValueError(
            f"num_workers={args.num_workers} exceeds available gpu slots={len(parsed_gpu_ids)}. "
            "Please reduce num_workers or provide more gpu_ids."
        )
    args.gpu_ids_list = parsed_gpu_ids

    clip_device = _resolve_clip_device(args.clip_device)
    if clip_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("clip_device=cuda requested but CUDA is unavailable")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OmniVideo 1.3B continuation generation from prefix clips, repeat each sample N times, "
            "and export per-frame CLIP similarity matrix with shape [N*(N-1)/2, S]."
        )
    )

    parser.add_argument("--input_jsonl", type=str, required=True, help="Input JSONL with keys: label, video_path")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory for videos and matrices")

    parser.add_argument("--ckpt_dir", type=str, required=True, help="OmniVideo checkpoint root directory")
    parser.add_argument(
        "--new_checkpoint",
        type=str,
        default=None,
        help="Optional trained checkpoint file. Defaults to ckpt_dir/transformer/pytorch_model.pt when present",
    )

    parser.add_argument("--size", type=str, default="832*480", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--sample_fps", type=int, default=8)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=40)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=3.0)
    parser.add_argument("--classifier_free_ratio", type=float, default=0.0)

    parser.add_argument("--prefix_frames", type=int, required=True, help="Number of prefix frames used as Omni input")
    parser.add_argument("--continuation_frames", type=int, required=True, help="Number of continuation frames S")
    parser.add_argument("--num_repeats", type=int, default=4, help="Number of generations per sample N")
    parser.add_argument("--input_sampling_rate", type=int, default=1, help="Frame stride when reading source video")

    parser.add_argument(
        "--continuation_noise_level",
        type=float,
        default=1.0,
        help="Mask strength for continuation latent noising (0.0=keep clean, 1.0=fully noised)",
    )
    parser.add_argument(
        "--denoise_strength",
        type=float,
        default=0.35,
        help="Img2img denoising strength for init_latents branch (0.0 preserves latent, 1.0 full sampling)",
    )
    parser.add_argument(
        "--start_step_index",
        type=int,
        default=None,
        help="Optional scheduler start index. Overrides denoise_strength when provided",
    )
    parser.add_argument(
        "--continuation_only_noising",
        type=str2bool,
        default=True,
        help="If true, apply latent noising mask only on continuation latent timesteps",
    )

    parser.add_argument("--base_seed", type=int, default=1818)
    parser.add_argument("--seed_stride", type=int, default=100000)
    parser.add_argument("--fixed_edit_prompt", type=str, default="Infer the subsequent scenes.")

    parser.add_argument("--use_usp", type=str2bool, default=False)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", type=str2bool, default=False)
    parser.add_argument("--dit_fsdp", type=str2bool, default=False)
    parser.add_argument("--max_context_len", type=int, default=6272)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (one worker binds one GPU)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="auto",
        help="Comma-separated GPU ids, e.g. '0,1,2'. Use 'auto' for all visible GPUs",
    )

    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_device", type=str, default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--clip_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument(
        "--offload_omni_during_clip",
        type=str2bool,
        default=True,
        help="When clip_device is cuda, offload Omni model to CPU while CLIP computes features",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_existing", type=str2bool, default=False)
    parser.add_argument("--save_full_video", type=str2bool, default=False)
    parser.add_argument("--save_gt_continuation", type=str2bool, default=False)

    args = parser.parse_args()
    _validate_args(args)
    return args


def _load_input_records(input_jsonl: str, max_samples: Optional[int]) -> List[Dict]:
    records: List[Dict] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                logging.warning("Skip malformed JSON at line %d", line_idx + 1)
                continue

            if "video_path" not in obj or "label" not in obj:
                logging.warning("Skip line %d because label/video_path is missing", line_idx + 1)
                continue

            sample_id = obj.get("sample_id", obj.get("id", line_idx))
            video_path = str(obj["video_path"])
            label = obj["label"]
            records.append(
                {
                    "sample_id": int(sample_id) if str(sample_id).isdigit() else line_idx,
                    "line_idx": int(line_idx),
                    "label": label,
                    "video_path": video_path,
                }
            )

            if max_samples is not None and len(records) >= max_samples:
                break

    return records


def _parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    total = int(torch.cuda.device_count())
    if total <= 0:
        return []

    if gpu_ids_arg is None:
        return list(range(total))

    text = str(gpu_ids_arg).strip()
    if text == "" or text.lower() == "auto":
        return list(range(total))

    ids: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if token == "":
            continue
        idx = int(token)
        if idx < 0 or idx >= total:
            raise ValueError(f"Invalid gpu id {idx}. Visible GPU count: {total}")
        ids.append(idx)

    if len(ids) == 0:
        raise ValueError("gpu_ids resolved to empty list")

    # Preserve order while removing duplicates.
    uniq_ids: List[int] = []
    seen = set()
    for idx in ids:
        if idx not in seen:
            uniq_ids.append(idx)
            seen.add(idx)
    return uniq_ids


def _compute_center_crop_size(src_h: int, src_w: int, target_h: int, target_w: int) -> Tuple[int, int]:
    ratio = float(target_w) / float(target_h)
    if src_w < src_h * ratio:
        crop_h = max(int(float(src_w) / ratio), 1)
        crop_w = src_w
    else:
        crop_h = src_h
        crop_w = max(int(float(src_h) * ratio), 1)
    return crop_h, crop_w


def _transform_frames(
    frames: np.ndarray,
    target_size_hw: Tuple[int, int],
) -> Tuple[np.ndarray, torch.Tensor]:
    target_h, target_w = target_size_hw
    src_h, src_w = frames[0].shape[:2]
    crop_h, crop_w = _compute_center_crop_size(src_h, src_w, target_h, target_w)

    out_uint8: List[np.ndarray] = []
    out_norm: List[torch.Tensor] = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = TF.center_crop(img, [crop_h, crop_w])
        img = TF.resize(img, [target_h, target_w], interpolation=InterpolationMode.BICUBIC)

        arr = np.asarray(img, dtype=np.uint8)
        out_uint8.append(arr)

        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)
        tensor = tensor / 127.5 - 1.0
        out_norm.append(tensor)

    return np.stack(out_uint8, axis=0), torch.stack(out_norm, dim=0)


def _load_prefix_and_gt(
    video_path: str,
    prefix_frames: int,
    continuation_frames: int,
    sampling_rate: int,
    target_size_hw: Tuple[int, int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[int]]:
    if not os.path.exists(video_path):
        return None, None, None, None, "video_not_found", None

    try:
        vr = decord.VideoReader(video_path)
    except Exception as e:
        return None, None, None, None, f"video_open_failed: {e}", None

    total_frames = len(vr)
    need = prefix_frames + continuation_frames
    indices = [i * sampling_rate for i in range(need)]
    if len(indices) == 0 or indices[-1] >= total_frames:
        return None, None, None, None, "video_too_short", total_frames

    try:
        raw = vr.get_batch(indices).asnumpy()
    except Exception as e:
        return None, None, None, None, f"video_read_failed: {e}", total_frames

    frames_uint8, frames_norm = _transform_frames(raw, target_size_hw=target_size_hw)
    prefix_uint8 = frames_uint8[:prefix_frames].copy()
    prefix_norm = frames_norm[:prefix_frames].contiguous()
    full_norm = frames_norm[:need].contiguous()
    gt_cont_uint8 = frames_uint8[prefix_frames : need].copy()
    return prefix_norm, full_norm, prefix_uint8, gt_cont_uint8, None, total_frames


def _move_vae_to_device(omni_video: OmniVideoX2XUnified1_3B, device: torch.device) -> None:
    omni_video.vae.model.to(device)
    omni_video.vae.mean = omni_video.vae.mean.to(device)
    omni_video.vae.std = omni_video.vae.std.to(device)
    omni_video.vae.scale = [omni_video.vae.mean, 1.0 / omni_video.vae.std]


def _encode_prefix_latent(
    omni_video: OmniVideoX2XUnified1_3B,
    cfg,
    prefix_norm: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        _move_vae_to_device(omni_video, device)
        latent = omni_video.vae.encode(prefix_norm.to(device).transpose(0, 1).unsqueeze(0))[0]
        _move_vae_to_device(omni_video, torch.device("cpu"))

    if latent.dim() == 4:
        latent = latent.unsqueeze(0)

    vc_patch_size = cfg.visual_context_adapter_patch_size
    if isinstance(vc_patch_size, (list, tuple)) and len(vc_patch_size) > 0 and int(vc_patch_size[0]) > 1:
        t_patch = int(vc_patch_size[0])
        t = int(latent.shape[2])
        if t % t_patch != 0:
            num_to_pad = t_patch - (t % t_patch)
            pad_tensor = latent[:, :, 0:1].repeat(1, 1, num_to_pad, 1, 1)
            latent = torch.cat([pad_tensor, latent], dim=2)

    return latent


def _video_tensor_to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    if not isinstance(video, torch.Tensor):
        video = torch.as_tensor(video)

    if video.dim() != 4:
        raise ValueError(f"Expected [C,T,H,W], got shape {tuple(video.shape)}")

    if int(video.shape[0]) not in (1, 3):
        raise ValueError(f"Expected channel-first video with C in (1,3), got shape {tuple(video.shape)}")

    x = video.detach().cpu()
    if x.dtype == torch.uint8:
        out = x
    else:
        out = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)

    out = out.permute(1, 2, 3, 0).contiguous().numpy()
    if out.shape[-1] == 1:
        out = np.repeat(out, 3, axis=-1)
    elif out.shape[-1] > 3:
        out = out[..., :3]
    return np.ascontiguousarray(out)


def _save_uint8_video(frames_thwc: np.ndarray, save_path: str, fps: int) -> None:
    tensor = torch.from_numpy(frames_thwc)
    tensor = tensor.permute(0, 3, 1, 2).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    saved_path = cache_video(
        tensor=tensor,
        save_file=save_path,
        fps=fps,
        nrow=1,
        normalize=False,
        value_range=(0, 255),
    )
    if saved_path is None:
        raise RuntimeError(f"Failed to save video: {save_path}")


def _init_omni_model(args: argparse.Namespace, gpu_id: int):
    cfg = copy.deepcopy(WAN_CONFIGS["t2v-1.3B"])
    precision_dtype = cfg.param_dtype
    local_device = torch.device(f"cuda:{int(gpu_id)}")
    torch.cuda.set_device(local_device)

    omni_video = OmniVideoX2XUnified1_3B(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        vlm_in_dim=cfg.vlm_in_dim,
        device_id=int(gpu_id),
        rank=0,
        use_usp=args.use_usp,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        sp_size=args.sp_size,
        use_visual_context_adapter=cfg.use_visual_context_adapter,
        visual_context_adapter_patch_size=cfg.visual_context_adapter_patch_size,
        max_context_len=args.max_context_len,
        init_on_cpu=True,
        wan_config=cfg if args.new_checkpoint else None,
    )

    if args.new_checkpoint:
        logging.info("Loading trained checkpoint: %s", args.new_checkpoint)
        state_dict = torch.load(args.new_checkpoint, map_location="cpu")
        if "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        for k in list(state_dict.keys()):
            if isinstance(state_dict[k], torch.Tensor):
                state_dict[k] = state_dict[k].to(precision_dtype)

        missing, unexpected = omni_video.model.load_state_dict(state_dict, strict=False)
        logging.info(
            "Checkpoint loaded. missing_keys=%d unexpected_keys=%d",
            len(missing),
            len(unexpected),
        )
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    return omni_video, cfg, local_device


def _resolve_worker_clip_device(args: argparse.Namespace, gpu_id: int) -> torch.device:
    clip_name = str(args.clip_device).strip().lower()
    if clip_name in ("auto", "cuda"):
        return torch.device(f"cuda:{int(gpu_id)}")
    return _resolve_clip_device(args.clip_device)


def _init_clip(args: argparse.Namespace, gpu_id: int):
    clip_device = _resolve_worker_clip_device(args, gpu_id)
    dtype = _clip_dtype(args.clip_dtype, clip_device)

    logging.info(
        "Loading CLIP model %s on %s with dtype=%s",
        args.clip_model_name,
        str(clip_device),
        str(dtype),
    )
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    if clip_device.type == "cuda":
        model = CLIPModel.from_pretrained(args.clip_model_name, torch_dtype=dtype)
    else:
        model = CLIPModel.from_pretrained(args.clip_model_name)

    model = model.to(clip_device)
    model.eval()
    return processor, model, clip_device


def _encode_clip_features(
    frames_thwc: np.ndarray,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    feats: List[torch.Tensor] = []
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for start in range(0, int(frames_thwc.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(frames_thwc.shape[0]))
            images = [Image.fromarray(frames_thwc[i]) for i in range(start, end)]
            batch = processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device)
            if device.type == "cuda":
                pixel_values = pixel_values.to(model_dtype)

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)
            feats.append(image_features.cpu())

    return torch.cat(feats, dim=0).numpy().astype(np.float32)


def _build_similarity_matrix(gen_feats: List[np.ndarray]) -> Tuple[np.ndarray, List[Dict]]:
    """Return Equation-2 similarities for unordered generated-trajectory pairs."""
    rows: List[np.ndarray] = []
    row_meta: List[Dict] = []

    for i, j in combinations(range(len(gen_feats)), 2):
        sim = np.sum(gen_feats[i] * gen_feats[j], axis=-1, dtype=np.float32)
        rows.append(sim.astype(np.float32))
        row_meta.append({"pair_type": "gen_gen", "i": int(i), "j": int(j)})

    matrix = np.stack(rows, axis=0)
    return matrix, row_meta


def _seed_for_repeat(base_seed: int, seed_stride: int, sample_index: int, repeat_index: int) -> int:
    return int(base_seed + sample_index * seed_stride + repeat_index)


def _to_jsonable_label(label):
    try:
        return int(label)
    except Exception:
        return str(label)


@contextmanager
def _disable_omni_sampling_tqdm():
    """Temporarily disable Omni internal diffusion tqdm without affecting outer tqdm bars."""
    orig_tqdm = getattr(omni_x2x_module, "tqdm", None)
    if orig_tqdm is None:
        yield
        return

    def _passthrough_tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    omni_x2x_module.tqdm = _passthrough_tqdm
    try:
        yield
    finally:
        omni_x2x_module.tqdm = orig_tqdm


def _process_one_sample(
    args: argparse.Namespace,
    cfg,
    omni_video: OmniVideoX2XUnified1_3B,
    omni_device: torch.device,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
    clip_device: torch.device,
    rec: Dict,
    record_index: int,
) -> Dict:
    sample_id = int(rec["sample_id"])
    label = _to_jsonable_label(rec["label"])
    video_path = str(rec["video_path"])

    sample_dir = os.path.join(args.output_root, f"sample_{record_index:06d}_id{sample_id}")
    os.makedirs(sample_dir, exist_ok=True)

    matrix_path = os.path.join(sample_dir, "similarity_matrix.npy")
    row_map_path = os.path.join(sample_dir, "similarity_row_map.json")
    meta_path = os.path.join(sample_dir, "sample_meta.json")

    if args.skip_existing and os.path.exists(matrix_path) and os.path.exists(row_map_path):
        return {
            "status": "skipped_existing",
            "record_index": int(record_index),
            "sample_id": int(sample_id),
            "label": label,
            "video_path": video_path,
            "matrix_path": os.path.relpath(matrix_path, args.output_root),
        }

    target_size = SIZE_CONFIGS[args.size]
    target_size_hw = (int(target_size[1]), int(target_size[0]))
    prefix_norm, full_norm, prefix_uint8, gt_cont_frames, err, total_frames = _load_prefix_and_gt(
        video_path=video_path,
        prefix_frames=args.prefix_frames,
        continuation_frames=args.continuation_frames,
        sampling_rate=args.input_sampling_rate,
        target_size_hw=target_size_hw,
    )
    if err is not None:
        return {
            "status": "skipped_invalid_video",
            "reason": err,
            "record_index": int(record_index),
            "sample_id": int(sample_id),
            "label": label,
            "video_path": video_path,
            "total_frames": total_frames,
        }

    if args.save_gt_continuation:
        gt_path = os.path.join(sample_dir, "gt_continuation.mp4")
        _save_uint8_video(gt_cont_frames, gt_path, fps=args.sample_fps)

    full_visual_emb = _encode_prefix_latent(
        omni_video=omni_video,
        cfg=cfg,
        prefix_norm=full_norm,
        device=omni_device,
    )

    t_prefix_latent = (args.prefix_frames - 1) // 4 + 1
    noise_lvl = float(getattr(args, "continuation_noise_level", 1.0))
    if args.continuation_only_noising:
        latent_noise_mask = torch.zeros_like(full_visual_emb)
        if t_prefix_latent < int(full_visual_emb.shape[2]):
            latent_noise_mask[:, :, t_prefix_latent:, :, :] = noise_lvl
    else:
        latent_noise_mask = torch.full_like(full_visual_emb, fill_value=noise_lvl)

    denoise_strength = float(args.denoise_strength)
    if args.start_step_index is None and noise_lvl <= 0.0:
        # If no continuation noising is requested, keep init latents unchanged.
        denoise_strength = 0.0

    visual_emb = full_visual_emb

    generated_cont_frames: List[np.ndarray] = []
    generated_paths: List[str] = []
    generated_seeds: List[int] = []
    full_paths: List[str] = []

    total_model_frames = _align_to_4n_plus_1(args.prefix_frames + args.continuation_frames)

    for repeat_idx in range(args.num_repeats):
        seed = _seed_for_repeat(args.base_seed, args.seed_stride, record_index, repeat_idx)
        with _disable_omni_sampling_tqdm():
            video = omni_video.generate(
                input_prompt=args.fixed_edit_prompt,
                precomputed_context=None,
                ar_vision_input=None,
                visual_emb=visual_emb,
                init_latents=full_visual_emb,
                denoise_strength=denoise_strength,
                start_step_index=args.start_step_index,
                latent_noise_mask=latent_noise_mask,
                size=SIZE_CONFIGS[args.size],
                frame_num=total_model_frames,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                classifier_free_ratio=args.classifier_free_ratio,
                unconditioned_context=None,
                condition_mode=cfg.condition_mode,
                precision_dtype=cfg.param_dtype,
            )

        if video is None:
            return {
                "status": "failed_generation",
                "reason": "omni_generate_returned_none",
                "record_index": int(record_index),
                "sample_id": int(sample_id),
                "label": label,
                "video_path": video_path,
                "repeat_idx": int(repeat_idx),
                "seed": int(seed),
            }

        full_frames = _video_tensor_to_uint8_frames(video)
        need_end = args.prefix_frames + args.continuation_frames
        if int(full_frames.shape[0]) < int(need_end):
            return {
                "status": "failed_generation",
                "reason": "generated_video_too_short",
                "record_index": int(record_index),
                "sample_id": int(sample_id),
                "label": label,
                "video_path": video_path,
                "repeat_idx": int(repeat_idx),
                "seed": int(seed),
                "generated_frames": int(full_frames.shape[0]),
                "required_frames": int(need_end),
            }

        cont_frames = full_frames[args.prefix_frames:need_end]
        out_path = os.path.join(sample_dir, f"gen_repeat{repeat_idx:02d}_seed{seed}.mp4")
        _save_uint8_video(cont_frames, out_path, fps=args.sample_fps)

        generated_cont_frames.append(cont_frames)
        generated_paths.append(os.path.relpath(out_path, args.output_root))
        generated_seeds.append(int(seed))

        if args.save_full_video:
            full_path = os.path.join(sample_dir, f"gen_full_repeat{repeat_idx:02d}_seed{seed}.mp4")
            stitched_full = np.concatenate([prefix_uint8, cont_frames], axis=0)
            _save_uint8_video(np.ascontiguousarray(stitched_full), full_path, fps=args.sample_fps)
            full_paths.append(os.path.relpath(full_path, args.output_root))

        del video

    if clip_device.type == "cuda" and args.offload_omni_during_clip:
        omni_video.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    gen_feats = [
        _encode_clip_features(
            frames_thwc=frames,
            processor=clip_processor,
            model=clip_model,
            device=clip_device,
            batch_size=args.clip_batch_size,
        )
        for frames in generated_cont_frames
    ]

    if clip_device.type == "cuda" and args.offload_omni_during_clip:
        omni_video.model.to(omni_device)
        gc.collect()
        torch.cuda.empty_cache()

    sim_matrix, row_map = _build_similarity_matrix(gen_feats=gen_feats)

    np.save(matrix_path, sim_matrix)
    with open(row_map_path, "w", encoding="utf-8") as f:
        json.dump(row_map, f, ensure_ascii=False, indent=2)

    sample_meta = {
        "status": "ok",
        "record_index": int(record_index),
        "sample_id": int(sample_id),
        "label": label,
        "video_path": video_path,
        "source_total_frames": int(total_frames),
        "input_sampling_rate": int(args.input_sampling_rate),
        "prefix_frames": int(args.prefix_frames),
        "continuation_frames": int(args.continuation_frames),
        "num_repeats": int(args.num_repeats),
        "continuation_noise_level": float(noise_lvl),
        "denoise_strength": float(denoise_strength),
        "start_step_index": None if args.start_step_index is None else int(args.start_step_index),
        "continuation_only_noising": bool(args.continuation_only_noising),
        "model_frame_num": int(total_model_frames),
        "generated_paths": generated_paths,
        "generated_full_paths": full_paths,
        "generated_seeds": generated_seeds,
        "matrix_path": os.path.relpath(matrix_path, args.output_root),
        "row_map_path": os.path.relpath(row_map_path, args.output_root),
        "matrix_shape": [int(sim_matrix.shape[0]), int(sim_matrix.shape[1])],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sample_meta, f, ensure_ascii=False, indent=2)

    return sample_meta


def _run_worker_records(
    args: argparse.Namespace,
    worker_id: int,
    gpu_id: int,
    records: List[Dict],
) -> Tuple[Dict[str, int], str]:
    worker_results_path = os.path.join(args.output_root, f"results.worker{worker_id:02d}.jsonl")

    # Make RNG streams worker-specific while preserving per-sample seed determinism in generation.
    random.seed(args.base_seed + worker_id * 1000003)
    np.random.seed(args.base_seed + worker_id * 1000003)
    torch.manual_seed(args.base_seed + worker_id * 1000003)
    torch.cuda.manual_seed_all(args.base_seed + worker_id * 1000003)

    omni_video, cfg, omni_device = _init_omni_model(args, gpu_id=gpu_id)
    clip_processor, clip_model, clip_device = _init_clip(args, gpu_id=gpu_id)

    statuses: Dict[str, int] = {}
    with open(worker_results_path, "w", encoding="utf-8") as out_f:
        progress_bar = tqdm(
            records,
            desc=f"worker{worker_id:02d}-gpu{gpu_id}",
            position=worker_id,
            leave=True,
            dynamic_ncols=True,
        )
        for local_idx, rec in enumerate(progress_bar):
            global_idx = int(rec["record_index"])
            logging.info(
                "[worker=%d gpu=%d] Processing shard sample %d/%d | global_idx=%d | line=%d | sample_id=%s",
                worker_id,
                gpu_id,
                local_idx + 1,
                len(records),
                global_idx,
                int(rec["line_idx"]),
                str(rec["sample_id"]),
            )
            try:
                result = _process_one_sample(
                    args=args,
                    cfg=cfg,
                    omni_video=omni_video,
                    omni_device=omni_device,
                    clip_processor=clip_processor,
                    clip_model=clip_model,
                    clip_device=clip_device,
                    rec=rec,
                    record_index=global_idx,
                )
            except Exception as e:
                logging.exception(
                    "[worker=%d gpu=%d] Sample failed with exception | global_idx=%d line=%d sample_id=%s",
                    worker_id,
                    gpu_id,
                    global_idx,
                    int(rec["line_idx"]),
                    str(rec["sample_id"]),
                )
                result = {
                    "status": "failed_exception",
                    "reason": str(e),
                    "record_index": int(global_idx),
                    "sample_id": int(rec["sample_id"]),
                    "label": _to_jsonable_label(rec["label"]),
                    "video_path": str(rec["video_path"]),
                }

            status = str(result.get("status", "unknown"))
            statuses[status] = int(statuses.get(status, 0) + 1)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    return statuses, worker_results_path


def _worker_main(
    worker_id: int,
    gpu_id: int,
    args_dict: Dict,
    worker_records: List[Dict],
    report_queue,
) -> None:
    args = argparse.Namespace(**args_dict)
    try:
        statuses, worker_results_path = _run_worker_records(
            args=args,
            worker_id=worker_id,
            gpu_id=gpu_id,
            records=worker_records,
        )
        report_queue.put(
            {
                "worker_id": int(worker_id),
                "gpu_id": int(gpu_id),
                "ok": True,
                "statuses": statuses,
                "worker_results_path": worker_results_path,
                "num_records": int(len(worker_records)),
            }
        )
    except Exception as e:
        report_queue.put(
            {
                "worker_id": int(worker_id),
                "gpu_id": int(gpu_id),
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "num_records": int(len(worker_records)),
            }
        )


def _build_worker_shards(records: List[Dict], num_workers: int) -> List[List[Dict]]:
    shards: List[List[Dict]] = [[] for _ in range(num_workers)]
    for idx, rec in enumerate(records):
        new_rec = dict(rec)
        new_rec["record_index"] = int(idx)
        shards[idx % num_workers].append(new_rec)
    return shards


def _aggregate_worker_outputs(worker_reports: List[Dict], final_results_path: str) -> Dict[str, int]:
    merged: List[Dict] = []
    status_counts: Dict[str, int] = {}

    for rep in worker_reports:
        if not rep.get("ok", False):
            continue
        worker_results_path = str(rep["worker_results_path"])
        with open(worker_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                merged.append(obj)

    merged.sort(key=lambda x: int(x.get("record_index", -1)))
    with open(final_results_path, "w", encoding="utf-8") as out_f:
        for obj in merged:
            status = str(obj.get("status", "unknown"))
            status_counts[status] = int(status_counts.get(status, 0) + 1)
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return status_counts


def main() -> None:
    args = _parse_args()
    _init_logging(args.output_root)

    logging.info("Loading input records from %s", args.input_jsonl)
    records = _load_input_records(args.input_jsonl, args.max_samples)
    if len(records) == 0:
        raise RuntimeError("No valid records were found in input_jsonl")
    logging.info("Loaded %d records", len(records))

    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)

    results_path = os.path.join(args.output_root, "results.jsonl")
    summary_path = os.path.join(args.output_root, "summary.json")

    num_workers = int(args.num_workers)
    gpu_ids = list(args.gpu_ids_list)
    use_workers = min(num_workers, len(gpu_ids))

    logging.info(
        "Parallel plan: num_workers=%d gpu_ids=%s",
        use_workers,
        ",".join(str(x) for x in gpu_ids[:use_workers]),
    )

    worker_shards = _build_worker_shards(records, use_workers)

    mp_ctx = mp.get_context("spawn")
    report_queue = mp_ctx.Queue()
    processes: List[mp.Process] = []

    args_dict = dict(vars(args))
    worker_reports: List[Dict] = []

    for worker_id in range(use_workers):
        shard = worker_shards[worker_id]
        if len(shard) == 0:
            continue

        p = mp_ctx.Process(
            target=_worker_main,
            args=(
                int(worker_id),
                int(gpu_ids[worker_id]),
                args_dict,
                shard,
                report_queue,
            ),
            daemon=False,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while True:
        try:
            rep = report_queue.get_nowait()
        except queue.Empty:
            break
        worker_reports.append(rep)

    report_map = {int(rep["worker_id"]): rep for rep in worker_reports}
    for worker_id in range(use_workers):
        if worker_id not in report_map:
            report_map[worker_id] = {
                "worker_id": int(worker_id),
                "gpu_id": int(gpu_ids[worker_id]),
                "ok": False,
                "error": "worker_exited_without_report",
                "num_records": int(len(worker_shards[worker_id])),
            }

    worker_reports = [report_map[i] for i in sorted(report_map.keys())]

    statuses = _aggregate_worker_outputs(worker_reports, results_path)

    for rep in worker_reports:
        if not rep.get("ok", False):
            logging.error(
                "Worker failed | worker_id=%d gpu_id=%d num_records=%d error=%s",
                int(rep["worker_id"]),
                int(rep["gpu_id"]),
                int(rep.get("num_records", 0)),
                str(rep.get("error", "unknown")),
            )
            if rep.get("traceback"):
                logging.error("Worker traceback:\n%s", str(rep["traceback"]))

            statuses["failed_worker"] = int(statuses.get("failed_worker", 0) + int(rep.get("num_records", 0)))

    summary = {
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "output_root": os.path.abspath(args.output_root),
        "num_input_records": int(len(records)),
        "num_success": int(statuses.get("ok", 0)),
        "status_counts": statuses,
        "matrix_shape_formula": "[N*(N-1)/2, S]",
        "settings": {
            "prefix_frames": int(args.prefix_frames),
            "continuation_frames": int(args.continuation_frames),
            "num_repeats": int(args.num_repeats),
            "continuation_noise_level": float(getattr(args, 'continuation_noise_level', 1.0)),
            "denoise_strength": float(args.denoise_strength),
            "start_step_index": None if args.start_step_index is None else int(args.start_step_index),
            "continuation_only_noising": bool(args.continuation_only_noising),
            "fixed_edit_prompt": str(args.fixed_edit_prompt),
            "size": str(args.size),
            "sample_fps": int(args.sample_fps),
            "sample_solver": str(args.sample_solver),
            "sample_steps": int(args.sample_steps),
            "sample_shift": float(args.sample_shift),
            "sample_guide_scale": float(args.sample_guide_scale),
            "classifier_free_ratio": float(args.classifier_free_ratio),
            "input_sampling_rate": int(args.input_sampling_rate),
            "base_seed": int(args.base_seed),
            "seed_stride": int(args.seed_stride),
            "num_workers": int(use_workers),
            "gpu_ids": [int(x) for x in gpu_ids[:use_workers]],
            "clip_model_name": str(args.clip_model_name),
            "clip_device": str(args.clip_device),
            "clip_batch_size": int(args.clip_batch_size),
        },
        "worker_reports": worker_reports,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("All done. results=%s summary=%s", results_path, summary_path)


if __name__ == "__main__":
    main()
