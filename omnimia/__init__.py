"""Reference implementation utilities for gray-box OmniMIA experiments.

The package implements the method in Sections 3.3-3.5 of the accompanying
paper: stochastic trajectory consistency, three-view aggregation, and
cross-modal evidence fusion.
"""

from .trajectory import TrajectoryEvidence, score_probability_trajectories

__all__ = ("TrajectoryEvidence", "score_probability_trajectories")
