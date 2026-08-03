"""Trajectory-consistency features from OmniMIA Sections 3.3 and 3.4.

The target model supplies a truncated top-d probability vector for each
denoising step, semantic-token position, and stochastic trajectory.  This
module turns those vectors into the paper's consistency tensor
``F[s, p, j]`` and its expected, pessimistic, and optimistic views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectoryEvidence:
    """The three paper-defined views and their fused membership evidence."""

    expected: float
    pessimistic: float
    optimistic: float
    membership_score: float
    num_trajectories: int
    num_pairs: int
    num_steps: int
    num_positions: int
    aggregation_topk: int
    pair_mean_similarities: tuple[float, ...]

    def to_dict(self) -> dict[str, float | int | list[float]]:
        """Return a JSON-serializable representation for a result record."""
        return {
            "expected": self.expected,
            "pessimistic": self.pessimistic,
            "optimistic": self.optimistic,
            "membership_score": self.membership_score,
            "num_trajectories": self.num_trajectories,
            "num_pairs": self.num_pairs,
            "num_steps": self.num_steps,
            "num_positions": self.num_positions,
            "aggregation_topk": self.aggregation_topk,
            "pair_mean_similarities": list(self.pair_mean_similarities),
        }


def _as_trajectory_tensor(trajectories: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    values = np.asarray(trajectories, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            "trajectory probabilities must have shape "
            "[num_trajectories, num_steps, num_positions, top_d]"
        )
    if values.shape[0] < 2:
        raise ValueError("at least two stochastic trajectories are required")
    if any(size == 0 for size in values.shape[1:]):
        raise ValueError("steps, positions, and top-d probability vectors must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError("trajectory probabilities must be finite")
    return values


def pairwise_consistency_tensor(
    trajectories: Sequence[np.ndarray] | np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build ``F[s, p, j]`` using cosine similarity (Equation 2).

    The candidate-token coordinate must be aligned across trajectories.  The
    pathway runners guarantee this by selecting a fixed clean-forward top-d
    candidate list before stochastic reconstruction begins.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")

    probabilities = _as_trajectory_tensor(trajectories)
    first, second = np.triu_indices(probabilities.shape[0], k=1)
    left = probabilities[first]
    right = probabilities[second]
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    similarities = np.divide(
        np.sum(left * right, axis=-1),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > eps,
    )
    return np.moveaxis(similarities, 0, -1)  # [steps, positions, trajectory-pairs]


def _select_positions(
    consistency: np.ndarray,
    position_mask: np.ndarray | None,
) -> np.ndarray:
    if position_mask is None:
        return consistency

    mask = np.asarray(position_mask, dtype=bool)
    if mask.shape == consistency.shape[:2]:
        selected = consistency[mask]
    elif mask.shape == (consistency.shape[1],):
        return consistency[:, mask, :]
    else:
        raise ValueError(
            "position_mask must have shape [num_positions] or [num_steps, num_positions]"
        )
    if selected.size == 0:
        raise ValueError("position_mask selected no semantic token positions")
    return selected.reshape(1, -1, consistency.shape[-1])


def _mean_extreme(values: np.ndarray, count: int, *, largest: bool) -> np.ndarray:
    if count < 1:
        raise ValueError("aggregation_topk must be positive")
    use_count = min(int(count), values.shape[-1])
    partition_index = values.shape[-1] - use_count if largest else use_count - 1
    partitioned = np.partition(values, partition_index, axis=-1)
    selected = partitioned[..., -use_count:] if largest else partitioned[..., :use_count]
    return selected.mean(axis=-1)


def aggregate_consistency_tensor(
    consistency: np.ndarray,
    *,
    aggregation_topk: int = 32,
) -> TrajectoryEvidence:
    """Aggregate ``F`` into the three views defined by Equations 3 and 4.

    For the pessimistic (optimistic) view, we first average the k smallest
    (largest) trajectory-pair similarities at every step-position cell, then
    average the k smallest (largest) resulting cells.  ``k`` is clipped to the
    available count at each level, which makes the paper's default k=32 valid
    for small trajectory counts such as T=4.
    """
    tensor = np.asarray(consistency, dtype=np.float64)
    if tensor.ndim != 3 or any(size == 0 for size in tensor.shape):
        raise ValueError("consistency tensor must be non-empty [steps, positions, pairs]")
    if not np.isfinite(tensor).all():
        raise ValueError("consistency tensor must be finite")
    if aggregation_topk < 1:
        raise ValueError("aggregation_topk must be positive")

    flat_cells = tensor.reshape(-1, tensor.shape[-1])
    expected = float(flat_cells.mean())

    pessimistic_cells = _mean_extreme(flat_cells, aggregation_topk, largest=False)
    optimistic_cells = _mean_extreme(flat_cells, aggregation_topk, largest=True)
    pessimistic = float(_mean_extreme(pessimistic_cells[None, :], aggregation_topk, largest=False)[0])
    optimistic = float(_mean_extreme(optimistic_cells[None, :], aggregation_topk, largest=True)[0])

    num_pairs = int(tensor.shape[-1])
    trajectories = int((1 + np.sqrt(1 + 8 * num_pairs)) / 2)
    if trajectories * (trajectories - 1) // 2 != num_pairs:
        raise ValueError("the final tensor axis must contain all unordered trajectory pairs")

    pair_means = tuple(float(value) for value in flat_cells.mean(axis=0))
    return TrajectoryEvidence(
        expected=expected,
        pessimistic=pessimistic,
        optimistic=optimistic,
        membership_score=expected + pessimistic + optimistic,
        num_trajectories=trajectories,
        num_pairs=num_pairs,
        num_steps=int(tensor.shape[0]),
        num_positions=int(tensor.shape[1]),
        aggregation_topk=int(aggregation_topk),
        pair_mean_similarities=pair_means,
    )


def score_probability_trajectories(
    trajectories: Sequence[np.ndarray] | np.ndarray,
    *,
    position_mask: np.ndarray | None = None,
    aggregation_topk: int = 32,
    eps: float = 1e-12,
) -> TrajectoryEvidence:
    """Compute the complete gray-box OmniMIA evidence for one sample."""
    consistency = pairwise_consistency_tensor(trajectories, eps=eps)
    selected = _select_positions(consistency, position_mask)
    return aggregate_consistency_tensor(selected, aggregation_topk=aggregation_topk)
