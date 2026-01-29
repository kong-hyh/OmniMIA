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


autocast = torch.autocast


def _load_processed_record_keys(*, results_path: str, image_key: str, text_key: str, label_key: str) -> set[Tuple[str, str, int]]:
	"""Load processed record keys from an existing results shard.

	Resume should NOT rely on sample_id because it is derived from input row index.
	We instead treat (image_path, caption/text, label) as the record identity.

	Best-effort: ignores malformed lines/JSON.
	"""
	processed: set[Tuple[str, str, int]] = set()
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
					img = obj.get(str(image_key), None)
					txt = obj.get(str(text_key), None)
					lbl = obj.get(str(label_key), None)
					if img is None or txt is None or lbl is None:
						continue
					processed.add((str(img), str(txt), int(lbl)))
				except Exception:
					continue
	except Exception:
		return processed
	return processed


def _load_processed_record_keys_from_shard_dir(
	*, shard_dir: str, image_key: str, text_key: str, label_key: str
) -> set[Tuple[str, str, int]]:
	"""Load processed record keys from ALL per-rank shard files.

	This is important for resume when the number of devices/workers changes.
	"""
	processed: set[Tuple[str, str, int]] = set()
	if not shard_dir or not os.path.isdir(shard_dir):
		return processed
	for path in sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl"))):
		processed |= _load_processed_record_keys(
			results_path=path,
			image_key=str(image_key),
			text_key=str(text_key),
			label_key=str(label_key),
		)
	return processed


def _stable_record_hash(key: Tuple[str, str, int]) -> int:
	"""Deterministic hash for repartitioning across ranks.

	Do NOT use Python's built-in hash() because it is salted per-process.
	"""
	import hashlib

	img, txt, lbl = key
	payload = (str(img) + "\n" + str(txt) + "\n" + str(int(lbl))).encode("utf-8", errors="replace")
	digest = hashlib.blake2b(payload, digest_size=8).digest()
	return int.from_bytes(digest, byteorder="little", signed=False)


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
	if isinstance(prev, dict):
		for k, prev_v in prev.items():
			if k in ignore:
				continue
			if k not in current:
				continue
			cur_v = current.get(k)
			if prev_v != cur_v:
				diffs.append(str(k))
	if diffs:
		diffs_sorted = ", ".join(sorted(diffs))
		raise RuntimeError(
			"Resume config mismatch (must match previous run). "
			f"Mismatched keys: {diffs_sorted}. Previous={cfg_path}"
		)


def _count_assigned_samples_for_rank_resume(
	*,
	input_path: str,
	image_key: str,
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
	for sample in _iter_valid_samples_shard(
		input_path=str(input_path),
		image_key=str(image_key),
		text_key=str(text_key),
		label_key=str(label_key),
		limit=limit,
		rank=0,
		world_size=1,
	):
		key = (str(sample[1]), str(sample[2]), int(sample[3]))
		assigned_rank = int(_stable_record_hash(key) % int(world_size))
		if assigned_rank == int(rank):
			n += 1
	return int(n)


def _count_remaining_samples_for_rank_resume(
	*,
	input_path: str,
	image_key: str,
	text_key: str,
	label_key: str,
	limit: Optional[int],
	rank: int,
	world_size: int,
	processed_keys: set[Tuple[str, str, int]],
) -> int:
	"""Count how many valid samples remain for this rank under resume repartitioning."""
	if world_size <= 0:
		raise ValueError("world_size must be > 0")
	if not (0 <= rank < world_size):
		raise ValueError("rank must be in [0, world_size)")

	n = 0
	for sample in _iter_valid_samples_shard(
		input_path=str(input_path),
		image_key=str(image_key),
		text_key=str(text_key),
		label_key=str(label_key),
		limit=limit,
		rank=0,
		world_size=1,
	):
		img = str(sample[1])
		txt = str(sample[2])
		lbl = int(sample[3])
		key = (img, txt, lbl)
		assigned_rank = int(_stable_record_hash(key) % int(world_size))
		if assigned_rank != int(rank):
			continue
		if processed_keys and key in processed_keys:
			continue
		n += 1
	return int(n)


def _count_remaining_samples_for_all_ranks_resume(
	*,
	input_path: str,
	image_key: str,
	text_key: str,
	label_key: str,
	limit: Optional[int],
	world_size: int,
	processed_keys: set[Tuple[str, str, int]],
) -> List[int]:
	"""Count remaining samples for every rank in one pass (resume repartitioning)."""
	if world_size <= 0:
		raise ValueError("world_size must be > 0")
	counts = [0 for _ in range(int(world_size))]
	for sample in _iter_valid_samples_shard(
		input_path=str(input_path),
		image_key=str(image_key),
		text_key=str(text_key),
		label_key=str(label_key),
		limit=limit,
		rank=0,
		world_size=1,
	):
		img = str(sample[1])
		txt = str(sample[2])
		lbl = int(sample[3])
		key = (img, txt, lbl)
		if processed_keys and key in processed_keys:
			continue
		r = int(_stable_record_hash(key) % int(world_size))
		counts[r] += 1
	return [int(x) for x in counts]


def _roc_auc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
	y_true_np = np.asarray(y_true, dtype=np.int64)
	y_score_np = np.asarray(y_score, dtype=np.float64)
	if y_true_np.ndim != 1 or y_score_np.ndim != 1 or y_true_np.shape[0] != y_score_np.shape[0]:
		raise ValueError("y_true/y_score must be 1D arrays of same length")

	n_pos = int((y_true_np == 1).sum())
	n_neg = int((y_true_np == 0).sum())
	if n_pos == 0 or n_neg == 0:
		return float("nan")

	order = np.argsort(y_score_np, kind="mergesort")
	ranks = np.empty_like(order, dtype=np.float64)
	ranks[order] = np.arange(1, len(y_score_np) + 1, dtype=np.float64)

	sorted_scores = y_score_np[order]
	start = 0
	while start < len(sorted_scores):
		end = start + 1
		while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
			end += 1
		if end - start > 1:
			avg = (start + 1 + end) / 2.0
			ranks[order[start:end]] = avg
		start = end

	pos_ranks_sum = ranks[y_true_np == 1].sum()
	u = pos_ranks_sum - (n_pos * (n_pos + 1) / 2.0)
	return float(u / (n_pos * n_neg))


def _cosine_similarity_1d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
	a = a.astype(np.float64)
	b = b.astype(np.float64)
	num = float(np.dot(a, b))
	denom = float(np.linalg.norm(a) * np.linalg.norm(b))
	return num / max(denom, eps)


def _parse_steps(value: str) -> Tuple[str, float]:
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


def _infer_model_vocab_size(*, model: Any, device: torch.device) -> int:
	for attr in ["vocab_size", "llm_vocab_size"]:
		try:
			v = getattr(getattr(model, "config", None), attr, None)
			if v is not None:
				vi = int(v)
				if vi > 0:
					return vi
		except Exception:
			pass

	try:
		emb = model.get_input_embeddings()
		if emb is not None:
			n = int(getattr(emb, "num_embeddings", 0))
			if n > 0:
				return n
	except Exception:
		pass

	x = torch.zeros((1, 1), device=device, dtype=torch.long)
	attention_bias = torch.ones((1, 1, 1, 1), device=device, dtype=torch.bool)
	logits = model(x, attention_bias=attention_bias).logits
	return int(logits.size(-1))


def _infer_text_vocab_end(*, backend: str, model: Any, tokenizer: Any) -> Optional[int]:
	if backend == "lumina":
		try:
			from models.LuminaDiMOO.config import SPECIAL_TOKENS

			return int(dict(SPECIAL_TOKENS)["image_token_offset"])
		except Exception:
			try:
				from models.LuminaDiMOO.trajectory_prob import text_vocab_size

				return int(text_vocab_size(model=model, codebook_size=8192))
			except Exception:
				return None

	if backend == "mmada":
		try:
			return int(len(tokenizer))
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
	text_vocab_end: Optional[int],
	image_vocab_size: Optional[int] = None,
) -> torch.BoolTensor:
	if vocab_size <= 0:
		raise ValueError("vocab_size must be > 0")
	if text_vocab_end is None:
		raise ValueError("image-vocab similarity requires text_vocab_end, but got None")

	start = max(0, min(int(text_vocab_end), int(vocab_size)))
	if image_vocab_size is None:
		end = int(vocab_size)
	else:
		end = min(int(vocab_size), start + max(0, int(image_vocab_size)))
	if end <= start:
		raise ValueError(
			f"Invalid image vocab slice: start={start}, end={end} (vocab_size={vocab_size}, image_vocab_size={image_vocab_size})"
		)

	allow = torch.zeros((int(vocab_size),), dtype=torch.bool)
	allow[start:end] = True

	forbid = _collect_forbid_token_ids(backend=backend, tokenizer=tokenizer, mask_id=int(mask_id))
	for tid in forbid:
		if 0 <= int(tid) < int(vocab_size):
			allow[int(tid)] = False

	if int(allow.sum().item()) <= 0:
		raise ValueError(
			f"allow_token_mask is empty after filtering (vocab_size={vocab_size}, text_vocab_end={text_vocab_end}, image_vocab_size={image_vocab_size})"
		)

	return allow


def _build_maskable_token_mask(*, backend: str, tokenizer: Any, mask_id: int, vocab_size: int) -> torch.BoolTensor:
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

	logits = model(input_ids, attention_bias=attention_bias).logits
	if allow_token_mask.ndim != 1:
		raise ValueError("allow_token_mask must be 1D")

	vocab_size = int(logits.size(-1))
	if int(allow_token_mask.numel()) != vocab_size:
		raise ValueError(f"allow_token_mask length {int(allow_token_mask.numel())} != vocab_size {vocab_size}")

	allow = allow_token_mask.to(device=logits.device, dtype=torch.bool)
	valid_count = int(allow.sum().item())
	if valid_count <= 0:
		raise ValueError("No valid tokens left for top-k selection")
	if int(topk) > valid_count:
		raise ValueError(f"topk={topk} exceeds valid token count={valid_count}")

	min_val = torch.finfo(logits.dtype).min
	logits = logits.masked_fill((~allow).view(1, 1, -1), min_val)

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
		raise ValueError("No valid tokens left for top-k selection")
	if int(topk) > valid_count:
		raise ValueError(f"topk={topk} exceeds valid token count={valid_count}")

	min_val = torch.finfo(logits.dtype).min
	logits_m = logits.masked_fill((~allow).view(1, 1, -1), min_val)
	_v, idx = torch.topk(logits_m, k=int(topk), dim=-1)
	return idx.to(torch.long)


def _choose_mask_positions_in_region(
	input_ids_1d: torch.LongTensor,
	*,
	region_pos_mask: torch.BoolTensor,
	mask_ratio: float,
	rng: np.random.Generator,
	maskable_token_mask: torch.BoolTensor,
) -> torch.BoolTensor:
	if input_ids_1d.ndim != 1:
		raise ValueError("input_ids_1d must be 1D")
	if region_pos_mask.ndim != 1:
		raise ValueError("region_pos_mask must be 1D")
	if int(region_pos_mask.numel()) != int(input_ids_1d.numel()):
		raise ValueError("region_pos_mask must have same length as input_ids_1d")

	allow_ids = maskable_token_mask.detach().cpu().to(torch.bool)
	vocab_size = int(allow_ids.numel())

	cand: List[int] = []
	ids_cpu = input_ids_1d.detach().cpu().to(torch.long)
	pos_cpu = region_pos_mask.detach().cpu().to(torch.bool)
	for pos, (tid, ok_pos) in enumerate(zip(ids_cpu.tolist(), pos_cpu.tolist())):
		if not bool(ok_pos):
			continue
		tid_i = int(tid)
		if 0 <= tid_i < vocab_size and bool(allow_ids[tid_i].item()):
			cand.append(int(pos))

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
	return_x_final: bool = False,
) -> np.ndarray:
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
		raise ValueError(f"candidate_ids shape {tuple(candidate_ids.shape)} must match x_init {tuple(x.shape)}")

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

	if x_hist is not None:
		if len(x_hist) == 0:
			hist = np.zeros((bsz, 0, int(x.shape[1])), dtype=np.int32)
		else:
			# [steps_done, B, L] -> [B, steps_done, L]
			hist = np.stack(x_hist, axis=0).transpose(1, 0, 2).astype(np.int32, copy=False)
		return out, hist

	if not bool(return_x_final):
		return out
	return out, x.detach().cpu().to(torch.int32)


def _save_reconstructed_image_from_sequence(
	*,
	backend: str,
	state: Dict[str, Any],
	tokenizer: Any,
	input_ids_1xL: torch.LongTensor,
	attention_mask_1xL: torch.BoolTensor,
	allow_token_mask: torch.BoolTensor,
	save_path: str,
) -> bool:
	"""Reconstruct and save an image from a (possibly denoised) multimodal token sequence.

	Best-effort: returns False if decoding fails.
	"""
	try:
		pipe = state.get("pipe", None)
		if pipe is None:
			return False
		ids = input_ids_1xL[0].detach().cpu().to(torch.long)
		am = attention_mask_1xL[0].detach().cpu().to(torch.bool)

		text_vocab_end = _infer_text_vocab_end(backend=str(backend), model=getattr(pipe, "model", None), tokenizer=tokenizer)
		if text_vocab_end is None:
			if str(backend) == "mmada":
				try:
					text_vocab_end = int(len(pipe.uni_prompting.text_tokenizer))  # type: ignore[union-attr]
				except Exception:
					text_vocab_end = None
			elif str(backend) == "lumina":
				text_vocab_end = 126356

		allow_ids_cpu = allow_token_mask.detach().cpu().to(torch.bool)
		vocab_size = int(allow_ids_cpu.numel())
		img_tok: List[int] = []
		for tid, ok in zip(ids.tolist(), am.tolist()):
			if not ok:
				continue
			tid_i = int(tid)
			if 0 <= tid_i < vocab_size and bool(allow_ids_cpu[tid_i].item()):
				img_tok.append(tid_i)

		if not img_tok:
			return False

		os.makedirs(os.path.dirname(save_path), exist_ok=True)

		if str(backend) == "mmada":
			if text_vocab_end is None:
				return False
			codes = (torch.tensor(img_tok, dtype=torch.long) - int(text_vocab_end)).unsqueeze(0)
			vq_model = getattr(pipe, "vq_model", None)
			if vq_model is None:
				return False
			try:
				vq_device = next(vq_model.parameters()).device
			except Exception:
				vq_device = None
			if vq_device is not None:
				codes = codes.to(vq_device)
			img = vq_model.decode_code(codes)
			img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
			img_u8 = (img * 255.0).permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")[0]
			from PIL import Image

			Image.fromarray(img_u8).save(save_path)
			return True

		if str(backend) == "lumina":
			from models.LuminaDiMOO.utils.image_utils import decode_vq_to_image

			START_ID = 126349
			END_ID = 126350
			NEWLINE_ID = 126084
			try:
				seq = ids.tolist()
				si = seq.index(START_ID)
				ei = seq.index(END_ID, si + 1)
				mid = seq[si + 1 : ei]
				newline_pos = [i for i, t in enumerate(mid) if int(t) == NEWLINE_ID]
				if not newline_pos:
					raise ValueError("no newline tokens")
				lat_w = int(newline_pos[0])
				lat_h = int(len(newline_pos))
			except Exception:
				n = int(len(img_tok))
				s = int(math.isqrt(max(1, n)))
				lat_h = s
				lat_w = s
				if lat_h * lat_w != n:
					lat_h = 1
					lat_w = n

			codes = torch.tensor(img_tok, dtype=torch.long).unsqueeze(0)
			vqvae = getattr(pipe, "vqvae", None)
			if vqvae is None:
				return False
			scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
			img = decode_vq_to_image(codes, image_height=int(lat_h * scale), image_width=int(lat_w * scale), vqvae=vqvae)
			img.save(save_path)
			return True

		return False
	except Exception:
		return False


def _aggregate(values: Sequence[float], how: str) -> float:
	if not values:
		return float("nan")
	if how == "mean":
		return float(np.mean(values))
	if how == "max":
		return float(np.max(values))
	if how == "min":
		return float(np.min(values))
	raise ValueError(f"Unknown aggregate: {how}")


def _debug_decode_built_one(
	*,
	backend: str,
	state: Dict[str, Any],
	tokenizer: Any,
	input_ids_1xL: torch.LongTensor,
	attention_mask_1xL: torch.BoolTensor,
	allow_token_mask: torch.BoolTensor,
	out_dir: str,
	name: str,
) -> None:
	"""Minimal debug helper: decode text + reconstruct image from a built sequence.

	Enabled only when called (recommended behind an env-flag). Intended for quick sanity checks and can be deleted.
	"""
	try:
		os.makedirs(out_dir, exist_ok=True)
		pipe = state.get("pipe", None)
		ids = input_ids_1xL[0].detach().cpu().to(torch.long)
		am = attention_mask_1xL[0].detach().cpu().to(torch.bool)

		seq_len = int(ids.numel())
		attn_len = int(am.sum().item())

		# Infer text/image vocab split.
		text_vocab_end = _infer_text_vocab_end(backend=str(backend), model=getattr(pipe, "model", None), tokenizer=tokenizer)
		if text_vocab_end is None:
			# Fallbacks
			if str(backend) == "mmada":
				try:
					text_vocab_end = int(len(pipe.uni_prompting.text_tokenizer))  # type: ignore[union-attr]
				except Exception:
					text_vocab_end = None
			elif str(backend) == "lumina":
				text_vocab_end = 126356

		# Decode text (strip out image-vocab ids; keep only attended positions).
		text_ids: List[int] = []
		for tid, ok in zip(ids.tolist(), am.tolist()):
			if not ok:
				continue
			tid_i = int(tid)
			if text_vocab_end is not None and 0 <= tid_i < int(text_vocab_end):
				text_ids.append(tid_i)
		try:
			text = tokenizer.decode(text_ids, skip_special_tokens=True)
		except Exception:
			text = str(text_ids[:200])
		print(f"[debug_decode] {name} backend={backend} seq_len={seq_len} attn_len={attn_len} text=", text)

		# Extract image code tokens by allow_token_mask on token ids (robust across both backends).
		allow_ids_cpu = allow_token_mask.detach().cpu().to(torch.bool)
		vocab_size = int(allow_ids_cpu.numel())
		img_tok: List[int] = []
		img_pos: List[int] = []
		for tid, ok in zip(ids.tolist(), am.tolist()):
			if not ok:
				img_tok.append(-1)
				continue
			tid_i = int(tid)
			if 0 <= tid_i < vocab_size and bool(allow_ids_cpu[tid_i].item()):
				img_tok.append(tid_i)
			else:
				img_tok.append(-1)

		for i, (tid_i, ok) in enumerate(zip(ids.tolist(), am.tolist())):
			if not ok:
				continue
			t = int(tid_i)
			if 0 <= t < vocab_size and bool(allow_ids_cpu[t].item()):
				img_pos.append(int(i))

		# Print image-token positions in the prompt.
		if not img_pos:
			print(f"[debug_decode] {name} no image tokens found")
			return

		# Merge contiguous positions into ranges for readability.
		ranges: List[Tuple[int, int]] = []
		start = img_pos[0]
		prev = img_pos[0]
		for p in img_pos[1:]:
			if p == prev + 1:
				prev = p
				continue
			ranges.append((start, prev + 1))  # [start, end)
			start = p
			prev = p
		ranges.append((start, prev + 1))

		ranges_preview = ", ".join([f"{s}-{e}" for (s, e) in ranges[:12]])
		if len(ranges) > 12:
			ranges_preview += f", ... (+{len(ranges) - 12} ranges)"

		pos_preview = img_pos[:24]
		if len(img_pos) > 24:
			pos_preview_str = f"{pos_preview} ... (+{len(img_pos) - 24} positions)"
		else:
			pos_preview_str = str(pos_preview)

		print(
			f"[debug_decode] {name} img_pos_count={len(img_pos)} first={img_pos[0]} last={img_pos[-1]} ranges=[{ranges_preview}]"
		)
		print(f"[debug_decode] {name} img_pos_preview={pos_preview_str}")

		# Also print token-id preview at those positions.
		id_preview = [int(ids[p].item()) for p in img_pos[:24]]
		if len(img_pos) > 24:
			print(f"[debug_decode] {name} img_token_id_preview={id_preview} ...")
		else:
			print(f"[debug_decode] {name} img_token_id_preview={id_preview}")

		# Build a mixed prompt string with an inline [img] placeholder.
		mixed_parts: List[str] = []
		buf_text: List[int] = []

		def _flush_text_buf() -> None:
			nonlocal buf_text
			if not buf_text:
				return
			try:
				seg = tokenizer.decode(buf_text, skip_special_tokens=True)
			except Exception:
				seg = ""
			if seg:
				mixed_parts.append(seg)
			buf_text = []

		ids_list = ids.tolist()
		am_list = am.tolist()
		i = 0
		while i < len(ids_list):
			if not bool(am_list[i]):
				i += 1
				continue
			tid_i = int(ids_list[i])
			is_img = 0 <= tid_i < vocab_size and bool(allow_ids_cpu[tid_i].item())
			is_text = (text_vocab_end is not None) and (0 <= tid_i < int(text_vocab_end))
			if is_img:
				_flush_text_buf()
				# Collapse a contiguous run of image tokens into a single placeholder.
				if not mixed_parts or mixed_parts[-1] != "[img]":
					mixed_parts.append("[img]")
				j = i + 1
				while j < len(ids_list) and bool(am_list[j]):
					tj = int(ids_list[j])
					if not (0 <= tj < vocab_size and bool(allow_ids_cpu[tj].item())):
						break
					j += 1
				i = j
				continue
			if is_text:
				buf_text.append(tid_i)
			else:
				_flush_text_buf()
				mixed_parts.append(f"[{tid_i}]")
			i += 1
		_flush_text_buf()

		mixed_prompt = " ".join([p for p in mixed_parts if p])
		if len(mixed_prompt) > 1200:
			mixed_prompt = mixed_prompt[:1200] + " ... (truncated)"
		print(f"[debug_decode] {name} mixed_prompt=", mixed_prompt)

		# Filter out sentinels from img_tok for decoding.
		img_tok = [t for t in img_tok if t >= 0]

		if str(backend) == "mmada":
			# MMaDA image tokens are VQ indices offset by len(text_tokenizer).
			if text_vocab_end is None:
				raise RuntimeError("debug_decode: text_vocab_end is None for mmada")
			codes = (torch.tensor(img_tok, dtype=torch.long) - int(text_vocab_end)).unsqueeze(0)
			vq_model = getattr(pipe, "vq_model", None)
			if vq_model is None:
				raise RuntimeError("debug_decode: pipe.vq_model is None")
			try:
				vq_device = next(vq_model.parameters()).device
			except Exception:
				vq_device = None
			if vq_device is not None:
				codes = codes.to(vq_device)
			img = vq_model.decode_code(codes)
			img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
			img_u8 = (img * 255.0).permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")[0]
			from PIL import Image

			Image.fromarray(img_u8).save(os.path.join(out_dir, f"{name}_recon.png"))
			print(f"[debug_decode] {name} wrote {os.path.join(out_dir, f'{name}_recon.png')}")
			return

		if str(backend) == "lumina":
			# Lumina image tokens already include the offset (126356). Need to infer (lat_h, lat_w) from newline layout.
			from models.LuminaDiMOO.utils.image_utils import decode_vq_to_image

			START_ID = 126349
			END_ID = 126350
			NEWLINE_ID = 126084
			try:
				seq = ids.tolist()
				si = seq.index(START_ID)
				ei = seq.index(END_ID, si + 1)
				mid = seq[si + 1 : ei]
				newline_pos = [i for i, t in enumerate(mid) if int(t) == NEWLINE_ID]
				if not newline_pos:
					raise ValueError("no newline tokens")
				lat_w = int(newline_pos[0])
				lat_h = int(len(newline_pos))
			except Exception:
				n = int(len(img_tok))
				s = int(math.isqrt(max(1, n)))
				lat_h = s
				lat_w = s
				if lat_h * lat_w != n:
					lat_h = 1
					lat_w = n

			codes = torch.tensor(img_tok, dtype=torch.long).unsqueeze(0)
			vqvae = getattr(pipe, "vqvae", None)
			if vqvae is None:
				raise RuntimeError("debug_decode: pipe.vqvae is None")
			scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
			img = decode_vq_to_image(codes, image_height=int(lat_h * scale), image_width=int(lat_w * scale), vqvae=vqvae)
			save_path = os.path.join(out_dir, f"{name}_recon.png")
			img.save(save_path)
			print(f"[debug_decode] {name} wrote {save_path} (lat={lat_h}x{lat_w}, scale={scale})")
			return
	except Exception as e:
		print(f"[debug_decode] failed: {type(e).__name__}: {e}")


def _iter_valid_samples_shard(
	*,
	input_path: str,
	image_key: str,
	text_key: str,
	label_key: str,
	limit: Optional[int],
	rank: int,
	world_size: int,
) -> Iterator[Tuple[int, str, str, int]]:
	if world_size <= 0:
		raise ValueError("world_size must be > 0")
	if not (0 <= rank < world_size):
		raise ValueError("rank must be in [0, world_size)")

	def _emit(sample_id: int, row: Dict[str, Any]) -> Optional[Tuple[int, str, str, int]]:
		img = row.get(image_key, None)
		txt = row.get(text_key, None)
		lbl = row.get(label_key, None)
		if img is None or txt is None or lbl is None:
			return None
		try:
			lbl_i = int(lbl)
		except Exception:
			return None
		return sample_id, str(img), str(txt), int(lbl_i)

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

	if input_path.endswith(".json"):
		with open(input_path, "r", encoding="utf-8") as f:
			rows = json.load(f)
		if not isinstance(rows, list):
			raise ValueError(".json input must be a list of dicts")
		if limit is not None:
			rows = rows[: int(limit)]
		for row_idx, row in enumerate(rows):
			if (row_idx % world_size) != rank:
				continue
			if not isinstance(row, dict):
				continue
			out = _emit(row_idx, row)
			if out is not None:
				yield out
		return

	raise ValueError("input_path must end with .json or .jsonl")


def _count_raw_records_for_rank(*, input_path: str, limit: Optional[int], rank: int, world_size: int) -> Optional[int]:
	try:
		if input_path.endswith(".json"):
			with open(input_path, "r", encoding="utf-8") as f:
				rows = json.load(f)
			n = len(rows[: int(limit)]) if limit is not None else len(rows)
			q, r = divmod(int(n), int(world_size))
			return int(q + (1 if int(rank) < int(r) else 0))

		if input_path.endswith(".jsonl"):
			n = 0
			with open(input_path, "r", encoding="utf-8") as f:
				for _line in f:
					if limit is not None and n >= int(limit):
						break
					n += 1
			q, r = divmod(int(n), int(world_size))
			return int(q + (1 if int(rank) < int(r) else 0))
	except Exception:
		return None
	return None


@dataclass
class Config:
	task: str
	backend: str

	input_path: str
	output_dir: str

	pretrained_model_path: str
	config: Optional[str]
	vq_model_name: Optional[str]

	image_key: str
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

	topk: int
	save_probs: bool
	save_traj_outputs: bool
	save_traj_tokens: bool
	save_dtype: str

	traj_aggregate: str
	eps: float

	progress: bool
	resume: bool


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
	task: str,
	backend: str,
	pretrained_model_path: str,
	config: Optional[str],
	vq_model_name: Optional[str],
	device: torch.device,
	dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Any, Any, int]:
	if backend == "mmada":
		from models.MMaDA.pipeline import MMaDAPipeline

		pipe = MMaDAPipeline.from_pretrained(
			task=("t2i" if task == "t2i" else "mmu"),
			pretrained_model_path=pretrained_model_path,
			config=config,
			vq_model_name=vq_model_name,
			device=device,
			torch_dtype=dtype,
			padding_side="right",
		)
		model = pipe.model
		tokenizer = pipe.tokenizer
		mask_id = int(getattr(model.config, "mask_token_id", None) or getattr(pipe.model.config, "mask_token_id"))
		return {"pipe": pipe}, model, tokenizer, mask_id

	if backend == "lumina":
		from models.LuminaDiMOO.pipeline import LuminaDiMOOPipeline
		from models.LuminaDiMOO.trajectory_prob import mask_token_id as lumina_mask_token_id

		pipe = LuminaDiMOOPipeline.from_pretrained(
			task=("t2i" if task == "t2i" else "mmu"),
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

	raise ValueError(f"Unknown backend: {backend}")


def _mmada_encode_image_tokens(pipe, image_path: str, *, resolution: int) -> torch.LongTensor:
	from PIL import Image
	from models.MMaDA.utils import image_transform, image_transform_squash

	file_name = os.path.basename(str(image_path))
	image_obj = Image.open(str(image_path)).convert("RGB")
	if any(tag in file_name for tag in ["ai2d", "clevr", "docvqa", "geo", "llava"]):
		image_tensor = image_transform_squash(image_obj, resolution=resolution).to(pipe.device)
	else:
		image_tensor = image_transform(image_obj, resolution=resolution).to(pipe.device)
	image_tensor = image_tensor.unsqueeze(0)

	vq_param = next(pipe.vq_model.parameters(), None)
	vq_dtype = vq_param.dtype if vq_param is not None else image_tensor.dtype
	with torch.autocast(pipe.device.type, enabled=False):
		image_tokens = pipe.vq_model.get_code(image_tensor.to(dtype=vq_dtype))
	return image_tokens + len(pipe.uni_prompting.text_tokenizer)


def _build_inputs_and_image_pos_mask(
	*,
	task: str,
	backend: str,
	state: Dict[str, Any],
	tokenizer: Any,
	allow_token_mask: torch.BoolTensor,
	image_path: str,
	text: str,
	max_seq_len: int,
) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, Tuple[int, int]]:
	def _find_changed_span_in_b(a: List[int], b: List[int]) -> Tuple[int, int]:
		"""Return [start,end) span in b that differs from a.

		Assumes b is created from a by inserting/replacing a small contiguous region.
		Uses longest common prefix/suffix; robust to BPE whitespace tokenization quirks.
		"""
		i = 0
		na = len(a)
		nb = len(b)
		while i < na and i < nb and int(a[i]) == int(b[i]):
			i += 1
		# If identical, no changed span.
		if i == na and i == nb:
			return -1, -1
		# Longest common suffix, without overlapping prefix.
		j = 0
		while (na - 1 - j) >= i and (nb - 1 - j) >= i and int(a[na - 1 - j]) == int(b[nb - 1 - j]):
			j += 1
		start = int(i)
		end = int(nb - j)
		if end < start:
			end = start
		return start, end

	pipe = state["pipe"]

	if backend == "mmada":
		if pipe.vq_model is None:
			raise RuntimeError("MMaDA requires vq_model for image-token evaluation")

		resolution = 512
		try:
			if pipe.config is not None:
				resolution = int(pipe.config.dataset.preprocessing.resolution)
		except Exception:
			pass

		img_tokens = _mmada_encode_image_tokens(pipe, image_path, resolution=resolution)

		if task == "t2i":
			input_ids, attention_mask = pipe.uni_prompting(([str(text)], img_tokens), "t2i_gen")
		else:
			# Unified i2t/MMU prompt template: "describe the image [img] : {caption}"
			# NOTE: [img] is a placeholder; the model should receive REAL image tokens at that position.
			# We use literal "[img]" only as a marker during tokenization, then replace it with:
			#   <|soi|> + img_tokens + <|eoi|>
			caption = str(text)
			marker = "[img]"
			prompt_with_markers = f"describe the image {marker} : {caption}"
			prompt_no_marker = f"describe the image : {caption}"
			messages = [{"role": "user", "content": prompt_with_markers}]
			ids = pipe.uni_prompting.text_tokenizer.apply_chat_template(
				messages,
				tokenize=True,
				add_generation_prompt=True,
				return_tensors=None,
			)
			if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
				ids = ids[0]
			messages_nm = [{"role": "user", "content": prompt_no_marker}]
			ids_nm = pipe.uni_prompting.text_tokenizer.apply_chat_template(
				messages_nm,
				tokenize=True,
				add_generation_prompt=True,
				return_tensors=None,
			)
			if isinstance(ids_nm, list) and len(ids_nm) > 0 and isinstance(ids_nm[0], list):
				ids_nm = ids_nm[0]

			mmu_token_id = int(pipe.uni_prompting.sptids_dict["<|mmu|>"].item())
			soi_token_id = int(pipe.uni_prompting.sptids_dict["<|soi|>"].item())
			eoi_token_id = int(pipe.uni_prompting.sptids_dict["<|eoi|>"].item())

			ids_list = list(map(int, list(ids)))
			ids_list_nm = list(map(int, list(ids_nm)))
			s, e = _find_changed_span_in_b(ids_list_nm, ids_list)
			if s < 0 or e < 0:
				raise RuntimeError("Failed to locate [img] marker span in i2t prompt (no diff)")
			if s == e:
				raise RuntimeError("Failed to locate [img] marker span in i2t prompt (empty diff)")

			# Replace the marker span with: [soi] + img_tokens + [eoi]
			# (img_tokens already include the correct vocab offset).
			rebuilt = ids_list[:s] + [int(soi_token_id)] + img_tokens[0].tolist() + [int(eoi_token_id)] + ids_list[e:]
			text_token_ids = torch.tensor(rebuilt, device=pipe.device, dtype=torch.long).unsqueeze(0)

			input_ids = torch.cat(
				[
					torch.full((1, 1), mmu_token_id, device=pipe.device, dtype=torch.long),
					text_token_ids,
				],
				dim=1,
			)
			attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

		if int(input_ids.shape[1]) > int(max_seq_len):
			input_ids = input_ids[:, : int(max_seq_len)]
			attention_mask = attention_mask[:, : int(max_seq_len)]

		pos_mask = allow_token_mask.to(input_ids.device)[input_ids] & attention_mask
		pos_any = pos_mask[0].detach().cpu().numpy().astype(bool)
		if not pos_any.any():
			raise RuntimeError("No image-vocab positions found in built sequence (mmada)")
		start = int(np.argmax(pos_any))
		end = int(len(pos_any) - np.argmax(pos_any[::-1]))
		return input_ids.to(torch.long), attention_mask.to(torch.bool), pos_mask.to(torch.bool), (start, end)

	if backend == "lumina":
		if task == "t2i":
			from models.LuminaDiMOO.trajectory_prob import t2i_build_tokens

			tokens, _code_start, _seq_len, _h, _w = t2i_build_tokens(pipe=pipe, prompt=str(text), image_path=str(image_path))
			input_ids = torch.tensor(tokens, device=pipe.device, dtype=torch.long).unsqueeze(0)
			attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
		else:
			from PIL import Image
			from models.LuminaDiMOO.utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop
			from models.LuminaDiMOO.utils.prompt_utils import generate_multimodal_understanding_prompt

			img = Image.open(str(image_path)).convert("RGB")
			crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
			img = var_center_crop(img, crop_size_list=crop_size_list)

			# Unified i2t/MMU prompt template: "describe the image [img] : {caption}"
			# NOTE: [img] is a placeholder; the model should receive REAL image tokens at that position.
			caption = str(text)
			marker = "[img]"
			question = f"describe the image {marker} : {caption}"
			question_nm = f"describe the image : {caption}"
			input_prompt = generate_multimodal_understanding_prompt(question)
			input_prompt_nm = generate_multimodal_understanding_prompt(question_nm)
			ids = list(map(int, list(pipe.tokenizer(input_prompt)["input_ids"])))
			ids_nm = list(map(int, list(pipe.tokenizer(input_prompt_nm)["input_ids"])))
			s, e = _find_changed_span_in_b(ids_nm, ids)
			if s < 0 or e < 0:
				raise RuntimeError("Failed to locate [img] marker span in Lumina prompt (no diff)")
			if s == e:
				raise RuntimeError("Failed to locate [img] marker span in Lumina prompt (empty diff)")

			img_tok = encode_img_with_breaks(img, pipe.vqvae)
			input_token = ids[:s] + list(img_tok) + ids[e:]
			input_ids = torch.tensor(input_token, device=pipe.device, dtype=torch.long).unsqueeze(0)
			attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

		if int(input_ids.shape[1]) > int(max_seq_len):
			input_ids = input_ids[:, : int(max_seq_len)]
			attention_mask = attention_mask[:, : int(max_seq_len)]

		pos_mask = allow_token_mask.to(input_ids.device)[input_ids] & attention_mask
		pos_any = pos_mask[0].detach().cpu().numpy().astype(bool)
		if not pos_any.any():
			raise RuntimeError("No image-vocab positions found in built sequence (lumina)")
		start = int(np.argmax(pos_any))
		end = int(len(pos_any) - np.argmax(pos_any[::-1]))
		return input_ids.to(torch.long), attention_mask.to(torch.bool), pos_mask.to(torch.bool), (start, end)

	raise ValueError(f"Unknown backend: {backend}")


def _process_batch(
	batch: Sequence[Tuple[int, str, str, int]],
	*,
	out_f,
	probs_dir: Optional[str],
	traj_outputs_dir: Optional[str],
	traj_tokens_dir: Optional[str],
	task: str,
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
	image_paths = [x[1] for x in batch]
	texts = [x[2] for x in batch]
	labels = [x[3] for x in batch]

	built: List[Tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, Tuple[int, int]]] = []
	for img, txt in zip(image_paths, texts):
		built.append(
			_build_inputs_and_image_pos_mask(
				task=task,
				backend=backend,
				state=state,
				tokenizer=tokenizer,
				allow_token_mask=allow_token_mask,
				image_path=img,
				text=txt,
				max_seq_len=int(args.max_seq_len),
			)
		)

	# Quick sanity check (optional): export one decoded text+image from the first built sample.
	# Usage: VDLM_MIA_DEBUG_DECODE=1 python image_trajectory_similarity_auc.py ...
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
		except Exception as _e:
			pass

	max_len = max(int(x[0].shape[1]) for x in built)
	pad_id = getattr(tokenizer, "pad_token_id", None)
	if pad_id is None:
		raise RuntimeError("Tokenizer has no pad_token_id")

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
			logits_clean = model(input_ids, attention_bias=attention_bias).logits  # [B,L,V]
			candidate_ids = _select_topk_candidates_from_logits(
				logits_clean,
				topk=int(args.topk),
				allow_token_mask=allow_token_mask,
			)
			logits_f = logits_clean.to(torch.float32)
			logp = logits_f - torch.logsumexp(logits_f, dim=-1, keepdim=True)
			pos_logp = torch.gather(logp, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B,L]
			pos_logp = pos_logp.to(torch.float16)

	traj_probs_by_sample: List[List[np.ndarray]] = [[] for _ in range(len(batch))]
	span0 = min(s for s, _e in spans)
	span1 = max(e for _s, e in spans)
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
						probs, x_hist = denoise_out  # type: ignore[misc]
					elif traj_outputs_dir is not None:
						probs, x_final = denoise_out  # type: ignore[misc]
					else:
						probs = denoise_out  # type: ignore[assignment]
					for j, b in enumerate(idxs):
						traj_probs_by_sample[b].append(probs[j])
						true_len_b = int(attention_mask[b].sum().detach().cpu().item())
						# Save per-step token ids (JSONL) for this trajectory
						if traj_tokens_dir is not None:
							try:
								save_path = os.path.join(
									str(traj_tokens_dir),
									f"sample_{int(sample_ids[b])}_traj{int(t_global)}.jsonl",
								)
								with open(save_path, "w", encoding="utf-8") as f:
									meta = {
										"type": "meta",
										"sample_id": int(sample_ids[b]),
										"trajectory_id": int(t_global),
										"task": str(task),
										"backend": str(backend),
										"mask_id": int(mask_id),
										"mask_ratio": float(args.mask_ratio),
										"topk": int(args.topk),
										"steps_mode": str(args.steps_mode),
										"steps_value": float(args.steps_value),
										"true_len": int(true_len_b),
										"allowed_range": [int(allowed_range[0]), int(allowed_range[1])],
										"image_path": str(image_paths[b]),
										"text": str(texts[b]),
										"label": int(labels[b]),
									}
									f.write(json.dumps(meta, ensure_ascii=False) + "\n")
									steps_done = int(x_hist[j].shape[0])
									for s in range(steps_done):
										token_ids = [int(x) for x in x_hist[j][s, :true_len_b].tolist()]
										rec = {
											"type": "step",
											"sample_id": int(sample_ids[b]),
											"trajectory_id": int(t_global),
											"step": int(s),
											"token_ids": token_ids,
										}
										f.write(json.dumps(rec, ensure_ascii=False) + "\n")
							except Exception:
								pass

						if traj_outputs_dir is not None:
							try:
								if traj_tokens_dir is not None:
									# reconstruct from last denoise step
									if int(x_hist[j].shape[0]) <= 0:
										x_last = x_sub[j].detach().cpu().to(torch.long)
									else:
										x_last = torch.from_numpy(x_hist[j][-1]).to(torch.long)
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
							except Exception:
								pass

	allow_ids_cpu = allow_token_mask.detach().cpu().to(torch.bool)
	vocab_size = int(allow_ids_cpu.numel())

	for b in range(len(batch)):
		traj_probs = traj_probs_by_sample[b]
		if len(traj_probs) < 2:
			continue

		ids_b = input_ids[b].detach().cpu().to(torch.long)
		am_b = attention_mask[b].detach().cpu().to(torch.bool)

		allowed_pos = np.zeros((allowed_range[1] - allowed_range[0],), dtype=np.bool_)
		for pos in range(int(allowed_range[0]), int(allowed_range[1])):
			if not bool(am_b[pos].item()):
				continue
			tid = int(ids_b[pos].item())
			if 0 <= tid < vocab_size and bool(allow_ids_cpu[tid].item()):
				allowed_pos[pos - int(allowed_range[0])] = True

		feats: List[np.ndarray] = []
		for p in traj_probs:
			feats.append(p[:, allowed_pos, :].reshape(-1).astype(np.float64))

		sims: List[float] = []
		for i in range(len(feats)):
			for j in range(i + 1, len(feats)):
				sims.append(_cosine_similarity_1d(feats[i], feats[j], eps=float(args.eps)))

		score = _aggregate(sims, how=str(args.traj_aggregate))

		rec = {
			"sample_id": int(sample_ids[b]),
			"task": str(task),
			args.image_key: str(image_paths[b]),
			args.text_key: str(texts[b]),
			args.label_key: int(labels[b]),
			"score": float(score),
			"pairwise_sims": sims,
			"num_trajectories": len(traj_probs),
			"steps_mode": str(args.steps_mode),
			"mask_ratio": float(args.mask_ratio),
			"backend": str(backend),
			"topk": int(args.topk),
			"allowed_range": [int(allowed_range[0]), int(allowed_range[1])],
		}
		out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

		if probs_dir is not None:
			cand_ids_b = candidate_ids[b, allowed_range[0] : allowed_range[1], :].detach().cpu().to(torch.int32).numpy()
			probs_stack = np.stack([p[:, :, :] for p in traj_probs], axis=0)
			if str(args.save_dtype) == "float32":
				probs_stack = probs_stack.astype(np.float32)
			else:
				probs_stack = probs_stack.astype(np.float16)
			save_path = os.path.join(probs_dir, f"sample_{int(sample_ids[b])}.npz")
			pos_logp_b = (
				pos_logp[b, int(allowed_range[0]) : int(allowed_range[1])].detach().cpu().numpy()
			)
			np.savez_compressed(
				save_path,
				candidate_ids=cand_ids_b,
				traj_probs=probs_stack,
				allowed_pos=allowed_pos,
				pos_logp=pos_logp_b,
				allowed_range=np.asarray([allowed_range[0], allowed_range[1]], dtype=np.int32),
			)


def _run_worker(rank: int, world_size: int, device_str: str, args: Config, resume_total_override: Optional[int] = None) -> None:
	task = str(args.task)
	backend = str(args.backend)
	device = torch.device(device_str)
	if device.type == "cuda":
		torch.cuda.set_device(device)

	dtype = _torch_dtype(args.torch_dtype)
	seed = int(args.seed) + int(rank) * 1000003
	rng_global = np.random.default_rng(seed)

	state, model, tokenizer, mask_id = _load_backend(
		task=task,
		backend=backend,
		pretrained_model_path=args.pretrained_model_path,
		config=args.config,
		vq_model_name=args.vq_model_name,
		device=device,
		dtype=dtype,
	)

	# Align image-vocab slice with each backend's generation code:
	# - Lumina: logits[..., vocab_offset:vocab_offset+codebook_size]
	# - MMaDA: logits[..., len(uni_prompting.text_tokenizer)+num_new_special_tokens : +codebook_size]
	text_vocab_end: Optional[int]
	image_vocab_size: Optional[int]
	if backend == "mmada":
		pipe = state.get("pipe", None)
		try:
			text_vocab_end = int(len(pipe.uni_prompting.text_tokenizer))  # type: ignore[union-attr]
		except Exception:
			text_vocab_end = _infer_text_vocab_end(backend=backend, model=model, tokenizer=tokenizer)
		try:
			image_vocab_size = int(getattr(getattr(pipe, "runtime", None), "codebook_size", None))  # type: ignore[arg-type]
		except Exception:
			image_vocab_size = None
		if image_vocab_size is None or image_vocab_size <= 0:
			try:
				image_vocab_size = int(getattr(getattr(model, "config", None), "codebook_size", 0))
			except Exception:
				image_vocab_size = 0
		if image_vocab_size is None or int(image_vocab_size) <= 0:
			image_vocab_size = 8192
	elif backend == 'lumina':
		text_vocab_end = _infer_text_vocab_end(backend=backend, model=model, tokenizer=tokenizer)
		try:
			image_vocab_size = int(getattr(getattr(model, "config", None), "codebook_size", 0))
		except Exception:
			image_vocab_size = 0
		if image_vocab_size is None or int(image_vocab_size) <= 0:
			image_vocab_size = 8192
	else:
		raise ValueError(f"Unknown backend: {backend}")
	vocab_size = _infer_model_vocab_size(model=model, device=device)

	allow_token_mask = _build_allow_token_mask(
		backend=backend,
		tokenizer=tokenizer,
		mask_id=int(mask_id),
		vocab_size=int(vocab_size),
		text_vocab_end=text_vocab_end,
		image_vocab_size=image_vocab_size,
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

	processed_keys: set[Tuple[str, str, int]] = set()
	if bool(args.resume):
		processed_keys = _load_processed_record_keys_from_shard_dir(
			shard_dir=shard_dir,
			image_key=str(args.image_key),
			text_key=str(args.text_key),
			label_key=str(args.label_key),
		)
		if processed_keys:
			print(f"[resume] rank{rank}: loaded {len(processed_keys)} processed record keys from {shard_dir}")

	probs_dir = os.path.join(shard_dir, f"probs_rank{rank}")
	if bool(args.save_probs):
		os.makedirs(probs_dir, exist_ok=True)

	traj_outputs_dir = os.path.join(shard_dir, f"traj_outputs_rank{rank}")
	if bool(args.save_traj_outputs):
		os.makedirs(traj_outputs_dir, exist_ok=True)

	traj_tokens_dir = os.path.join(shard_dir, f"traj_tokens_rank{rank}")
	if bool(args.save_traj_tokens):
		os.makedirs(traj_tokens_dir, exist_ok=True)

	batch_size = max(1, int(args.batch_size))

	# NOTE: For resume we re-partition work across the CURRENT world_size
	# (devices/workers count may differ from previous runs). We therefore iterate
	# all valid samples and assign them deterministically by record-key hash.
	if bool(args.resume):
		if resume_total_override is not None:
			total = int(resume_total_override)
		else:
			total = _count_remaining_samples_for_rank_resume(
				input_path=str(args.input_path),
				image_key=str(args.image_key),
				text_key=str(args.text_key),
				label_key=str(args.label_key),
				limit=args.limit,
				rank=int(rank),
				world_size=int(world_size),
				processed_keys=processed_keys,
			)
		sample_iter = _iter_valid_samples_shard(
			input_path=args.input_path,
			image_key=args.image_key,
			text_key=args.text_key,
			label_key=args.label_key,
			limit=args.limit,
			rank=0,
			world_size=1,
		)
	else:
		sample_iter = _iter_valid_samples_shard(
			input_path=args.input_path,
			image_key=args.image_key,
			text_key=args.text_key,
			label_key=args.label_key,
			limit=args.limit,
			rank=rank,
			world_size=world_size,
		)
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

	buffer: List[Tuple[int, str, str, int]] = []
	try:
		open_mode = "a" if bool(args.resume) and os.path.exists(results_path) else "w"
		with open(results_path, open_mode, encoding="utf-8") as out_f:
			for sample in sample_iter:
				img = str(sample[1])
				txt = str(sample[2])
				lbl = int(sample[3])
				key = (img, txt, lbl)

				if bool(args.resume):
					assigned_rank = int(_stable_record_hash(key) % int(world_size))
					if assigned_rank != int(rank):
						continue

				if processed_keys and key in processed_keys:
					continue

				buffer.append(sample)
				if len(buffer) < batch_size:
					continue

				_process_batch(
					buffer,
					out_f=out_f,
					probs_dir=probs_dir if bool(args.save_probs) else None,
					traj_outputs_dir=traj_outputs_dir if bool(args.save_traj_outputs) else None,
					traj_tokens_dir=traj_tokens_dir if bool(args.save_traj_tokens) else None,
					task=task,
					backend=backend,
					state=state,
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
				for s in buffer:
					processed_keys.add((str(s[1]), str(s[2]), int(s[3])))
				buffer.clear()

			if buffer:
				_process_batch(
					buffer,
					out_f=out_f,
					probs_dir=probs_dir if bool(args.save_probs) else None,
					traj_outputs_dir=traj_outputs_dir if bool(args.save_traj_outputs) else None,
					traj_tokens_dir=traj_tokens_dir if bool(args.save_traj_tokens) else None,
					task=task,
					backend=backend,
					state=state,
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
				for s in buffer:
					processed_keys.add((str(s[1]), str(s[2]), int(s[3])))
				buffer.clear()
	finally:
		pbar.close()

def _parse_args(*, task_override: Optional[str] = None) -> Config:
	p = argparse.ArgumentParser(
		description=(
			"Image-vocab multi-trajectory denoise logging: mask->unmask prob density per step -> trajectory similarity -> AUC. "
			"This always scores on IMAGE vocab, and the detection object is the image."
		)
	)

	p.add_argument("--task", default="t2i", choices=["t2i", "i2t"], help="t2i=text2img, i2t=img2text")
	p.add_argument("--backend", default="mmada", choices=["mmada", "lumina"], help="Model backend")

	p.add_argument("--input_path", required=True, help=".json or .jsonl file containing path+caption+label")
	p.add_argument("--output_dir", required=True, help="Where to write results")

	p.add_argument("--pretrained_model_path", required=True, help="Pretrained model path")
	p.add_argument("--config", default=None, help="Optional YAML config (MMaDA only; ignored by Lumina)")
	p.add_argument(
		"--vq_model_name",
		default=None,
		help="VQ model name/path. Required for MMaDA mmu/t2i; required for Lumina (VQModel.from_pretrained).",
	)

	p.add_argument("--image_key", default="path")
	p.add_argument("--text_key", default="caption")
	p.add_argument("--label_key", default="label")
	p.add_argument("--limit", type=int, default=None)

	p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--torch_dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])

	p.add_argument("--batch_size", type=int, default=1)
	p.add_argument("--max_seq_len", type=int, default=4096)

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

	p.add_argument("--mask_ratio", type=float, default=0.3)
	p.add_argument("--num_trajectories", type=int, default=4)
	p.add_argument(
		"--steps",
		type=str,
		default="18",
		help=("Denoising steps: integer (abs) or float in (0,1] (rel to #masked tokens). E.g. 18 or 0.5"),
	)

	p.add_argument("--topk", type=int, default=5, help="Top-k candidate tokens per position from clean forward")

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
		"--save_traj_outputs",
		dest="save_traj_outputs",
		action="store_true",
		default=True,
		help=(
			"Save each trajectory's final denoised output. For image-vocab scoring this reconstructs and saves the final image "
			"as PNG under output_dir/_shards/traj_outputs_rank*/ (default: enabled)."
		),
	)
	p.add_argument(
		"--no_save_traj_outputs",
		dest="save_traj_outputs",
		action="store_false",
		help="Disable saving per-trajectory final reconstructed outputs",
	)

	p.add_argument(
		"--save_traj_tokens",
		dest="save_traj_tokens",
		action="store_true",
		default=True,
		help=(
			"Save each trajectory's per-step token ids (JSONL) under output_dir/_shards/traj_tokens_rank*/ (default: enabled)."
		),
	)
	p.add_argument(
		"--no_save_traj_tokens",
		dest="save_traj_tokens",
		action="store_false",
		help="Disable saving per-step trajectory token ids",
	)
	p.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "float32"])

	p.add_argument("--traj_aggregate", type=str, default="mean", choices=["mean", "max", "min"])
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
			"Resume from existing shard results in output_dir: skip already-processed (path,text,label) and append. "
			"When resuming, remaining samples are repartitioned across the current world_size (devices/workers may change)."
		),
	)
	p.add_argument(
		"--no_resume",
		dest="resume",
		action="store_false",
		help="Disable resume behavior (overwrite shard outputs)",
	)

	a = p.parse_args()
	if task_override is not None:
		a.task = task_override

	mode, val = _parse_steps(a.steps)

	if not (0.0 <= float(a.mask_ratio) <= 1.0):
		raise ValueError("--mask_ratio must be in [0,1]")
	if int(a.num_trajectories) < 2:
		raise ValueError("--num_trajectories must be >= 2")

	return Config(
		task=str(a.task),
		backend=str(a.backend),
		input_path=a.input_path,
		output_dir=a.output_dir,
		pretrained_model_path=a.pretrained_model_path,
		config=a.config,
		vq_model_name=a.vq_model_name,
		image_key=str(a.image_key),
		text_key=str(a.text_key),
		label_key=str(a.label_key),
		limit=a.limit,
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
		steps_value=float(val),
		topk=int(a.topk),
		save_probs=bool(a.save_probs),
		save_traj_outputs=bool(a.save_traj_outputs),
		save_traj_tokens=bool(a.save_traj_tokens),
		save_dtype=str(a.save_dtype),
		traj_aggregate=str(a.traj_aggregate),
		eps=float(a.eps),
		progress=bool(a.progress),
		resume=bool(a.resume),
	)


def main(*, task_override: Optional[str] = None) -> None:
	args = _parse_args(task_override=task_override)
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
			processed_keys = _load_processed_record_keys_from_shard_dir(
				shard_dir=shard_dir,
				image_key=str(args.image_key),
				text_key=str(args.text_key),
				label_key=str(args.label_key),
			)
			resume_totals = _count_remaining_samples_for_all_ranks_resume(
				input_path=str(args.input_path),
				image_key=str(args.image_key),
				text_key=str(args.text_key),
				label_key=str(args.label_key),
				limit=args.limit,
				world_size=int(world_size),
				processed_keys=processed_keys,
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

		shard_dir = os.path.join(args.output_dir, "_shards")
		merged_path = os.path.join(args.output_dir, "results.jsonl")
		y_true: List[int] = []
		y_score: List[float] = []
		with open(merged_path, "w", encoding="utf-8") as out_f:
			# If resuming, include ALL shard result files (prior world_size may differ).
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
							y_score.append(float(obj.get("score", 0.0)))
						except Exception:
							continue

		auc = _roc_auc_binary(y_true, y_score)
		print(f"AUC={auc:.6f} (n={len(y_true)})")
	else:
		_run_worker(0, 1, str(args.device), args)

		shard_dir = os.path.join(args.output_dir, "_shards")
		rp = os.path.join(shard_dir, "results_rank0.jsonl")
		merged_path = os.path.join(args.output_dir, "results.jsonl")
		y_true: List[int] = []
		y_score: List[float] = []
		with open(merged_path, "w", encoding="utf-8") as out_f:
			if bool(args.resume):
				paths = sorted(glob.glob(os.path.join(shard_dir, "results_rank*.jsonl")))
				for rp_i in paths:
					if not os.path.exists(rp_i):
						continue
					with open(rp_i, "r", encoding="utf-8") as f:
						for line in f:
							line = line.strip()
							if not line:
								continue
							out_f.write(line + "\n")
							try:
								obj = json.loads(line)
								y_true.append(1 if int(obj.get(args.label_key, 0)) != 0 else 0)
								y_score.append(float(obj.get("score", 0.0)))
							except Exception:
								continue
			else:
				with open(rp, "r", encoding="utf-8") as f:
					for line in f:
						line = line.strip()
						if not line:
							continue
						out_f.write(line + "\n")
						try:
							obj = json.loads(line)
							y_true.append(1 if int(obj.get(args.label_key, 0)) != 0 else 0)
							y_score.append(float(obj.get("score", 0.0)))
						except Exception:
							continue
		auc = _roc_auc_binary(y_true, y_score)
		print(f"AUC={auc:.6f} (n={len(y_true)})")

	config_path = os.path.join(args.output_dir, "config.json")
	with open(config_path, "w", encoding="utf-8") as f:
		json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
	main()
