"""
src/gwo/fitness.py

Fitness evaluation for Grey Wolf Optimization feature selection, plus
the small numeric helpers (sigmoid transfer, binary step functions,
three-way crossover) used to decode a wolf's continuous position into
a binary feature-selection mask.

This is a direct, cleaned-up extraction of the standalone functions
that lived inline in the original repository's notebook: stepc, stepb,
X_d, cross_over, sigmoid, and Fitness.
"""

import random
from typing import List

import numpy as np


def fitness_score(accuracy: float, n_features: int, alpha: float = 0.6, beta: float = 0.4) -> float:
    """
    Combined GWO fitness function balancing classification accuracy
    against the number of selected features.

        fitness = alpha * accuracy + beta * (1 / n_features)

    where alpha is the weight given to classification performance and
    beta = (1 - alpha) is the weight given to feature-selection
    parsimony (fewer features -> higher fitness, all else equal).

    Parameters
    ----------
    accuracy : float
        Classification accuracy (or detection rate), expected as a
        percentage (0-100) to match the original repository's scale.
    n_features : int
        Number of features selected (dimensionality of the current
        feature subset).
    alpha : float
        Weight on classification accuracy, in [0, 1].
    beta : float
        Weight on feature-selection quality, typically 1 - alpha.

    Returns
    -------
    float
        The combined fitness value. If n_features is 0 (no features
        selected), the feature-count term is dropped rather than
        raising a ZeroDivisionError.
    """
    if n_features == 0:
        return alpha * accuracy
    return alpha * accuracy + beta * (1.0 / n_features)


def sigmoid_transfer(x: float) -> int:
    """
    Standard sigmoid transfer function used to squash a continuous GWO
    position value into a binary decision (used by the non-binary
    "GWO" variant to decode positions into a feature mask).

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        1 if sigmoid(x) >= 0.5, else 0.
    """
    return int(x >= 0)  # Equivalent threshold without exponential overflow.


def _step_probability(A: float, D: float) -> float:
    """
    Probability-of-flip function 'c' used by the binary GWO (bGWO)
    variant's step function. Mirrors the original 'stepc'.
    """
    z = float(np.clip(10 * (A * D - 0.5), -700, 700))
    return 1.0 / (1.0 + np.exp(z))


def _step_binary(A: float, D: float, r: float) -> int:
    """
    Binary step decision: 1 if the step probability exceeds a random
    draw r, else 0. Mirrors the original 'stepb'.
    """
    return 1 if _step_probability(A, D) >= r else 0


def binary_position_component(x_current: int, A: float, D: float, rand: random.Random) -> int:
    """
    Compute one binary "vote" component (X_d1 / X_d2 / X_d3 in the
    original code) for the bGWO crossover step: combines the wolf's
    current binary feature value with a stochastic step decision.
    Mirrors the original 'X_d'.

    Parameters
    ----------
    x_current : int
        The current binary feature value (0 or 1) relative to one of
        alpha/beta/delta.
    A : float
        The GWO 'A' coefficient for this leader wolf.
    D : float
        The distance term for this leader wolf and dimension.
    rand : random.Random
        Random generator to draw the stochastic threshold from.

    Returns
    -------
    int
        1 or 0.
    """
    r = rand.random()
    return 1 if (x_current + _step_binary(A, D, r)) >= 1 else 0


def three_way_crossover(x_d1: int, x_d2: int, x_d3: int, rand: random.Random) -> int:
    """
    Randomly select one of three binary candidate values with equal
    (1/3) probability each. Used by bGWO to combine the alpha/beta/
    delta-derived binary votes into a single new feature-mask bit.
    Mirrors the original 'cross_over'.

    Parameters
    ----------
    x_d1 : int
    x_d2 : int
    x_d3 : int
    rand : random.Random

    Returns
    -------
    int
    """
    r = rand.random()
    if r < 1 / 3:
        return x_d1
    elif r < 2 / 3:
        return x_d2
    return x_d3
