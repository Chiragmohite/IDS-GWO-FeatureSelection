"""
src/gwo/wolf.py

Defines the Wolf class: a single search agent in the Grey Wolf
Optimization population. Each wolf carries:
  - a continuous position vector in [lb, ub]^dim (the "raw" GWO position)
  - a binary feature mask derived from that position (which features
    are "selected")
  - a fitness score (from fitness.py) and the raw classification
    accuracy that produced it

This is a direct extraction of the `wolf` class that lived inline in
the original repository's notebook, with type hints and docstrings
added and no behavioural changes.
"""

import random
from typing import Callable, List


class Wolf:
    """
    A single grey wolf search agent.

    Attributes
    ----------
    position : List[float]
        Continuous position vector, one value per feature dimension,
        randomly initialized within [lb, ub].
    features : List[int]
        Binary feature-selection mask (1 = feature selected, 0 = not).
        Initialized to "all features selected" and updated during
        optimization as the wolf's position is decoded.
    score : float
        The raw classification performance (e.g. accuracy %) achieved
        with this wolf's currently selected feature subset.
    fitness : float
        The combined fitness value (accuracy + feature-count penalty),
        computed by a fitness function such as gwo.fitness.fitness_score.
    """

    def __init__(
        self,
        fitness_fn: Callable[[float, int], float],
        dim: int,
        seed: int,
        lb: float,
        ub: float,
        initial_score: float = 0.0,
    ):
        """
        Parameters
        ----------
        fitness_fn : Callable[[float, int], float]
            Function mapping (score, n_features) -> fitness value.
        dim : int
            Number of feature dimensions (search space size).
        seed : int
            Seed for this wolf's private random generator, so
            population initialization is reproducible per-wolf.
        lb : float
            Lower bound of the continuous position space.
        ub : float
            Upper bound of the continuous position space.
        initial_score : float
            Starting classification score before any optimization step
            (e.g. the baseline all-features accuracy).
        """
        self.rand = random.Random(seed)
        self.features: List[int] = [1 for _ in range(dim)]
        self.position: List[float] = [
            (ub - lb) * self.rand.random() + lb for _ in range(dim)
        ]
        self.score: float = initial_score
        self.fitness: float = fitness_fn(self.score, dim)

    def __repr__(self) -> str:
        return (
            f"Wolf(fitness={self.fitness:.4f}, score={self.score:.4f}, "
            f"n_features={sum(self.features)})"
        )
