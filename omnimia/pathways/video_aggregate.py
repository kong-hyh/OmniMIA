"""Aggregate video trajectory similarities with the gray-box OmniMIA views.

``video`` writes one CLIP-similarity row per repeated-generation
pair and one column per continuation frame.  Those rows are the trajectory
pairs in Equation 2; this script applies the same three-view aggregation as
the text and image pathways rather than fitting a task-specific regressor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from omnimia.evaluation import binary_roc_auc
from omnimia.trajectory import aggregate_consistency_tensor


# Equation 2 compares two independently initialized trajectories.  Every row
# emitted by the current video runner has precisely that ``gen_gen`` meaning.
PAIR_TYPES = ("gen_gen",)


def _iter_completed_samples(output_dir: Path) -> Iterable[tuple[int, int, Path, Path]]:
    for sample_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        metadata_path = sample_dir / "sample_meta.json"
        if not metadata_path.is_file():
            metadata_path = sample_dir / "meta.json"  # Compatibility with early result folders.
        matrix_path = sample_dir / "similarity_matrix.npy"
        row_map_path = sample_dir / "similarity_row_map.json"
        if not (metadata_path.is_file() and matrix_path.is_file() and row_map_path.is_file()):
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("status") != "ok":
            continue
        yield int(metadata["sample_id"]), int(metadata["label"]), matrix_path, row_map_path


def _evidence_for_pair_type(
    matrix_path: Path,
    row_map_path: Path,
    *,
    pair_type: str,
    aggregation_topk: int,
) -> dict[str, float | int | list[float]]:
    similarities = np.asarray(np.load(matrix_path), dtype=np.float64)
    if similarities.ndim != 2 or any(size == 0 for size in similarities.shape):
        raise ValueError(f"{matrix_path} must be a non-empty [trajectory-pairs, steps] matrix")
    with row_map_path.open("r", encoding="utf-8") as handle:
        row_map = json.load(handle)
    if not isinstance(row_map, list) or len(row_map) != similarities.shape[0]:
        raise ValueError(f"{row_map_path} must map every similarity-matrix row")

    row_indices = [
        index for index, row in enumerate(row_map) if row.get("pair_type") == pair_type
    ]
    if not row_indices:
        raise ValueError(f"{matrix_path} has no {pair_type!r} rows")

    # [trajectory-pairs, steps] -> F[steps, one-position, trajectory-pairs]
    consistency = similarities[row_indices, :].T[:, None, :]
    return aggregate_consistency_tensor(
        consistency, aggregation_topk=aggregation_topk
    ).to_dict()


def _load_pathway_evidence(
    output_dir: Path,
    *,
    pair_type: str,
    aggregation_topk: int,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for sample_id, label, matrix_path, row_map_path in _iter_completed_samples(output_dir):
        pair_types = (pair_type,)
        evidence = {
            name: _evidence_for_pair_type(
                matrix_path,
                row_map_path,
                pair_type=name,
                aggregation_topk=aggregation_topk,
            )
            for name in pair_types
        }
        records[sample_id] = {"label": label, "evidence": evidence}
    if not records:
        raise ValueError(f"no completed video samples found in {output_dir}")
    return records


def fuse_video_pathways(
    output_dirs: list[str],
    *,
    pair_type: str,
    aggregation_topk: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], float]:
    """Fuse evidence across one or more independently probed video pathways."""
    pathway_maps = [
        _load_pathway_evidence(
            Path(directory), pair_type=pair_type, aggregation_topk=aggregation_topk
        )
        for directory in output_dirs
    ]
    sample_ids = set(pathway_maps[0])
    for directory, records in zip(output_dirs[1:], pathway_maps[1:]):
        if set(records) != sample_ids:
            raise ValueError(f"sample IDs in {directory} do not match the first pathway")

    feature_names: list[str] = []
    for directory in output_dirs:
        for current_pair_type in (pair_type,):
            feature_names.extend(
                f"{Path(directory).name}.{current_pair_type}.{view}"
                for view in ("expected", "pessimistic", "optimistic")
            )

    fused: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    for sample_id in sorted(sample_ids):
        label = pathway_maps[0][sample_id]["label"]
        values: list[float] = []
        for records in pathway_maps:
            record = records[sample_id]
            if record["label"] != label:
                raise ValueError(f"sample {sample_id} has inconsistent labels across pathways")
            for current_pair_type in (pair_type,):
                current = record["evidence"][current_pair_type]
                values.extend(float(current[view]) for view in ("expected", "pessimistic", "optimistic"))
        score = float(sum(values))
        fused.append(
            {
                "sample_id": sample_id,
                "label": label,
                "membership_score": score,
                "features": dict(zip(feature_names, values)),
            }
        )
        labels.append(1 if label != 0 else 0)
        scores.append(score)
    return fused, tuple(feature_names), binary_roc_auc(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply paper-defined OmniMIA aggregation to video continuation trajectories."
    )
    parser.add_argument("pathway_dirs", nargs="+", help="Output directories from the video pathway.")
    parser.add_argument("--output", "--output_file", dest="output", required=True)
    parser.add_argument("--pair-type", choices=PAIR_TYPES, default="gen_gen")
    parser.add_argument("--aggregation-topk", type=int, default=32)
    args = parser.parse_args()
    if args.aggregation_topk < 1:
        raise ValueError("--aggregation-topk must be >= 1")

    samples, feature_names, auc = fuse_video_pathways(
        args.pathway_dirs,
        pair_type=args.pair_type,
        aggregation_topk=args.aggregation_topk,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "OmniMIA video gray-box fusion",
                "feature_names": feature_names,
                "auc": auc,
                "samples": samples,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"AUC={auc:.6f} (n={len(samples)})")
    print(f"Wrote fused results to {output}")


if __name__ == "__main__":
    main()
