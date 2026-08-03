"""Model-agnostic implementation of Black-Box OmniMIA.

Only decoded API outputs enter this module. Target-model logits, probabilities,
hidden states, and tokens are deliberately outside the interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np


class AccessMode(str, Enum):
    """Content exposure supplied by the target API."""

    MULTI_STEP = "multi_step"
    # Kept as an enum alias so existing integrations continue to work.
    MULTIPLE_STEP = "multi_step"
    FINAL_STEP = "final_step"


VIEW_NAMES = ("exp", "pess", "opt")

Encoder = Callable[[Sequence[Any]], np.ndarray]
QueryBuilder = Callable[[Any, str], Any]
TargetAPI = Callable[[Any, str, AccessMode], Any]


def _mode(value: AccessMode | str) -> AccessMode:
    if isinstance(value, AccessMode):
        return value
    # ``multiple_step`` was the spelling used by the previous public API.
    return AccessMode.MULTI_STEP if value == "multiple_step" else AccessMode(value)


def _aggregate(values: np.ndarray, view: str, axis: int, k: int) -> np.ndarray:
    """Apply one OmniMIA consistency view along ``axis``.

    The pessimistic and optimistic views retain the average of the lowest or
    highest ``k`` observations respectively; they are not min/max views.
    """
    if view == "exp":
        return values.mean(axis=axis)
    ordered = np.sort(values, axis=axis)
    if view == "pess":
        return np.take(ordered, np.arange(k), axis=axis).mean(axis=axis)
    if view == "opt":
        return np.take(ordered, np.arange(-k, 0), axis=axis).mean(axis=axis)
    raise ValueError(f"unknown consistency view: {view!r}")


def _validate_k(k: int, count: int, *, context: str) -> None:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if not 1 <= k <= count:
        raise ValueError(f"k must satisfy 1 <= k <= {count} for {context}")


def features_from_embeddings(
    embeddings: np.ndarray,
    mode: AccessMode | str,
    k: int = 1,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compute one task's expected, pessimistic, and optimistic evidence.

    ``embeddings`` must be ``[R, L, D]`` for multi-step access and ``[R, D]``
    for final-step access.  Returned values are ordered ``(exp, pess, opt)``.
    In multi-step mode each view is aggregated first across independent-query
    pairs at every exposed step, then across the exposed steps using that same
    view.  In final-step mode only the first aggregation is performed.
    """
    access_mode = _mode(mode)
    z = np.asarray(embeddings, dtype=np.float64)
    expected_ndim = 3 if access_mode is AccessMode.MULTI_STEP else 2
    if z.ndim != expected_ndim:
        raise ValueError(f"{access_mode.value} embeddings must have {expected_ndim} dimensions")
    if z.shape[0] < 2:
        raise ValueError("at least two independent queries are required")
    if 0 in z.shape:
        raise ValueError("embedding dimensions and exposed steps must be non-empty")

    if access_mode is AccessMode.FINAL_STEP:
        z = z[:, None, :]
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("semantic encoder returned a zero vector")
    z = z / norms

    first, second = np.triu_indices(z.shape[0], k=1)
    similarities = np.sum(z[first] * z[second], axis=-1)  # [pairs, steps]
    _validate_k(k, similarities.shape[0], context="pairwise consistencies")

    step_views = np.stack(
        [_aggregate(similarities, view, axis=0, k=k) for view in VIEW_NAMES], axis=0
    )  # [views, steps]

    if access_mode is AccessMode.FINAL_STEP:
        return step_views[:, 0], VIEW_NAMES

    _validate_k(k, step_views.shape[1], context="exposed generation steps")
    evidence = np.array(
        [_aggregate(step_views[index], view, axis=0, k=k) for index, view in enumerate(VIEW_NAMES)]
    )
    return evidence, VIEW_NAMES


def extract_pathway_features(
    repeated_outputs: Sequence[Any],
    encoder: Encoder,
    mode: AccessMode | str,
    k: int = 1,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode one task's decoded outputs and compute its three evidence views."""
    access_mode = _mode(mode)
    if len(repeated_outputs) < 2:
        raise ValueError("at least two independent API executions are required")

    if access_mode is AccessMode.FINAL_STEP:
        flat_outputs = list(repeated_outputs)
        shape = (len(flat_outputs),)
    else:
        executions = []
        for outputs in repeated_outputs:
            if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
                raise ValueError("each multiple-step execution must return an ordered output sequence")
            executions.append(list(outputs))
        step_count = len(executions[0])
        if step_count == 0 or any(len(outputs) != step_count for outputs in executions):
            raise ValueError("all executions must expose the same non-empty ordered step set")
        flat_outputs = [output for execution in executions for output in execution]
        shape = (len(executions), step_count)

    embeddings = np.asarray(encoder(flat_outputs), dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(flat_outputs):
        raise ValueError("encoder must return [number_of_outputs, embedding_dimension]")
    embeddings = embeddings.reshape(*shape, embeddings.shape[-1])
    return features_from_embeddings(embeddings, access_mode, k=k)


def extract_features(
    pathway_outputs: Mapping[str, Sequence[Any]],
    encoders: Mapping[str, Encoder],
    mode: AccessMode | str,
    pathway_order: Sequence[str] | None = None,
    k: int = 1,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the algorithm's unified membership score across all tasks.

    For each pathway, its expected, pessimistic, and optimistic evidence is
    added directly.  The resulting task scores are then added across pathways.
    No learned fusion or calibration is part of Black-Box OmniMIA.
    """
    pathways = tuple(pathway_order or pathway_outputs.keys())
    if not pathways:
        raise ValueError("at least one cross-modal pathway is required")

    membership_score = 0.0
    for pathway in pathways:
        if pathway not in pathway_outputs or pathway not in encoders:
            raise KeyError(f"missing outputs or encoder for pathway {pathway!r}")
        pathway_values, _ = extract_pathway_features(
            pathway_outputs[pathway], encoders[pathway], mode, k=k
        )
        membership_score += float(pathway_values.sum())
    return np.array([membership_score]), ("membership_score",)


def query_and_extract(
    sample: Any,
    pathways: Sequence[str],
    query_builder: QueryBuilder,
    target_api: TargetAPI,
    encoders: Mapping[str, Encoder],
    repeats: int,
    mode: AccessMode | str,
    k: int = 1,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Run the repeated-query procedure from the Black-Box OmniMIA algorithm."""
    access_mode = _mode(mode)
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    outputs: dict[str, list[Any]] = {}
    for pathway in pathways:
        query = query_builder(sample, pathway)
        outputs[pathway] = [target_api(query, pathway, access_mode) for _ in range(repeats)]
    return extract_features(outputs, encoders, access_mode, pathways, k=k)
