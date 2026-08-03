"""Evaluation and cross-pathway fusion for gray-box OmniMIA records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


VIEW_NAMES = ("expected", "pessimistic", "optimistic")


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return ROC-AUC without requiring scikit-learn; ties receive mean rank."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.size != y_score.size:
        raise ValueError("labels and scores must be one-dimensional and equally sized")
    positive = y_true == 1
    negative = y_true == 0
    if not positive.any() or not negative.any():
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    ordered_scores = y_score[order]
    start = 0
    while start < ordered_scores.size:
        end = start + 1
        while end < ordered_scores.size and ordered_scores[end] == ordered_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    mann_whitney_u = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(mann_whitney_u / (n_positive * n_negative))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record in {path} at line {line_number} is not an object")
            yield record


def _record_evidence(record: dict[str, Any]) -> dict[str, float]:
    evidence = record.get("trajectory_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("each record must contain a trajectory_evidence object")
    values: dict[str, float] = {}
    for name in VIEW_NAMES:
        if name not in evidence:
            raise ValueError(f"trajectory_evidence is missing {name!r}")
        values[name] = float(evidence[name])
    return values


def fuse_pathway_results(
    result_paths: Sequence[str | Path],
    *,
    label_key: str = "label",
) -> tuple[list[dict[str, Any]], tuple[str, ...], float]:
    """Fuse aligned pathway records according to Section 3.5.

    Each pathway contributes its expected, pessimistic, and optimistic
    evidence.  The final score is their sum across views and pathways, exactly
    matching ``e_mia = e_exp + e_pess + e_opt`` followed by cross-modal
    evidence integration.  Results are aligned by ``sample_id`` and labels
    must agree across all pathways.
    """
    if not result_paths:
        raise ValueError("at least one pathway results file is required")

    path_maps: list[dict[int, dict[str, Any]]] = []
    pathway_names: list[str] = []
    name_counts: dict[str, int] = {}
    for value in result_paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        records: dict[int, dict[str, Any]] = {}
        for record in _read_jsonl(path):
            if "sample_id" not in record:
                raise ValueError(f"{path}: a record has no sample_id")
            sample_id = int(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"{path}: duplicate sample_id {sample_id}")
            if label_key not in record:
                raise ValueError(f"{path}: sample {sample_id} has no {label_key!r}")
            _record_evidence(record)
            records[sample_id] = record
        if not records:
            raise ValueError(f"{path}: no records")
        base_name = path.parent.name if path.stem == "results" else path.stem
        occurrence = name_counts.get(base_name, 0)
        name_counts[base_name] = occurrence + 1
        pathway_name = base_name if occurrence == 0 else f"{base_name}_{occurrence + 1}"
        path_maps.append(records)
        pathway_names.append(pathway_name)

    common_ids = set(path_maps[0])
    for path, records in zip(result_paths[1:], path_maps[1:]):
        if set(records) != common_ids:
            raise ValueError(
                f"sample IDs in {path} do not exactly match the first pathway; "
                "run each pathway on the same input order"
            )

    feature_names = tuple(
        f"{pathway}.{view}" for pathway in pathway_names for view in VIEW_NAMES
    )
    fused: list[dict[str, Any]] = []
    labels: list[int] = []
    scores: list[float] = []
    for sample_id in sorted(common_ids):
        label = int(path_maps[0][sample_id][label_key])
        features: list[float] = []
        for path_name, records in zip(pathway_names, path_maps):
            record = records[sample_id]
            if int(record[label_key]) != label:
                raise ValueError(f"sample {sample_id} has inconsistent labels across pathways")
            evidence = _record_evidence(record)
            features.extend(evidence[view] for view in VIEW_NAMES)
        membership_score = float(sum(features))
        fused.append(
            {
                "sample_id": sample_id,
                label_key: label,
                "membership_score": membership_score,
                "features": dict(zip(feature_names, features)),
            }
        )
        labels.append(1 if label != 0 else 0)
        scores.append(membership_score)
    return fused, feature_names, binary_roc_auc(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse paper-defined gray-box OmniMIA evidence from cross-modal pathways."
    )
    parser.add_argument(
        "--pathway-results",
        nargs="+",
        required=True,
        help="One results.jsonl file per pathway, all with matching sample_id values.",
    )
    parser.add_argument("--output", required=True, help="Output JSON file for fused evidence and ROC-AUC.")
    parser.add_argument("--label-key", default="label")
    args = parser.parse_args()

    results, feature_names, auc = fuse_pathway_results(
        args.pathway_results, label_key=args.label_key
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "OmniMIA gray-box cross-modal fusion",
                "feature_names": feature_names,
                "auc": auc,
                "samples": results,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"AUC={auc:.6f} (n={len(results)})")
    print(f"Wrote fused results to {output}")


if __name__ == "__main__":
    main()
