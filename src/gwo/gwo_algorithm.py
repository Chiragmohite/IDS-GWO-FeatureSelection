"""
src/gwo/gwo_algorithm.py

Core Grey Wolf Optimization loops: the standard sigmoid-based GWO
variant and the binary GWO (bGWO) variant with three-way crossover,
as used in the original repository. Refactored out of one large
notebook cell into a clearly-staged implementation with explicit
alpha/beta/delta (gamma) wolf roles, population initialization,
position updates, and a stopping criterion (max_iter).

Design change from the original: instead of hardcoding a global
`classifier` and reading/reloading CSVs from disk inside the
objective function, this module takes an `evaluate_fn` callback
(dependency injection). That callback is responsible for turning a
binary feature mask into (fitness, score) -- see
gwo/feature_selector.py for the concrete implementation that wires
this up to a real classifier and dataset. This keeps gwo_algorithm.py
fully reusable and testable independent of any specific model or data.
"""

import copy
import logging
import random
from typing import Callable, List, Tuple

from src.gwo.fitness import (
    binary_position_component,
    sigmoid_transfer,
    three_way_crossover,
)
from src.gwo.wolf import Wolf

logger = logging.getLogger(__name__)

# (fitness, score) for a given binary feature mask
EvaluateFn = Callable[[List[int]], Tuple[float, float]]


def _initialize_leaders(population: List[Wolf]) -> Tuple[Wolf, Wolf, Wolf]:
    """
    Sort the population by fitness (descending) and return copies of
    the top three wolves as (alpha, beta, delta/gamma).
    """
    population.sort(key=lambda w: w.fitness, reverse=True)
    alpha, beta, delta = copy.copy(population[0]), copy.copy(population[1]), copy.copy(population[2])
    return alpha, beta, delta


def _coefficients(b: float, rand: random.Random) -> Tuple[float, float, float, float, float, float]:
    """
    Draw fresh A1/A2/A3 and C1/C2/C3 GWO coefficients for one update
    step, given the current linearly-decreasing 'b' parameter.
    """
    A1 = b * (2 * rand.random() - 1)
    A2 = b * (2 * rand.random() - 1)
    A3 = b * (2 * rand.random() - 1)
    C1 = 2 * rand.random()
    C2 = 2 * rand.random()
    C3 = 2 * rand.random()
    return A1, A2, A3, C1, C2, C3


def grey_wolf_optimizer(
    evaluate_fn: EvaluateFn,
    dim: int,
    n_wolves: int,
    max_iter: int,
    lb: float,
    ub: float,
    seed: int = 42,
    *, feature_budget=None, on_iteration=None,
) -> Wolf:
    """
    Standard (sigmoid-decoded) Grey Wolf Optimization for feature
    selection. Each wolf's continuous position is decoded into a
    binary feature mask via a sigmoid transfer function.

    Parameters
    ----------
    evaluate_fn : EvaluateFn
        Callback: binary feature mask (List[int]) -> (fitness, score).
    dim : int
        Number of feature dimensions (search space size).
    n_wolves : int
        Population size (N).
    max_iter : int
        Maximum number of iterations (stopping criterion).
    lb : float
        Lower bound of the continuous position space.
    ub : float
        Upper bound of the continuous position space.

    Returns
    -------
    Wolf
        The best wolf (alpha) found across all iterations, with its
        final `.features` binary mask, `.fitness`, and `.score`.
    """
    from src.gwo.fitness import fitness_score

    if dim < 1 or n_wolves < 3 or max_iter < 0 or lb >= ub:
        raise ValueError("Require dim >= 1, wolves >= 3, iterations >= 0 and lower < upper.")
    if feature_budget is not None and not 1 <= feature_budget <= dim:
        raise ValueError("Feature budget must be in 1..dim.")
    def budget_mask(position):
        chosen = set(sorted(range(dim), key=lambda j: (-position[j], j))[:feature_budget])
        return [int(j in chosen) for j in range(dim)]
    rand = random.Random(seed)
    population = [Wolf(fitness_score, dim, seed=seed + i, lb=lb, ub=ub) for i in range(n_wolves)]
    # Every score must correspond to the actual mask. Include an all-feature
    # reference and diverse starting subsets; never reuse another mask's score.
    for index, wolf in enumerate(population):
        if feature_budget is not None:
            wolf.features = budget_mask(wolf.position)
        elif index:
            wolf.features = [int(x >= (lb + ub) / 2) for x in wolf.position]
            if not any(wolf.features):
                wolf.features[index % dim] = 1
        wolf.fitness, wolf.score = evaluate_fn(wolf.features)


    b = 2.0
    A1, A2, A3, C1, C2, C3 = _coefficients(b, rand)
    alpha_wolf, beta_wolf, delta_wolf = _initialize_leaders(population)
    logger.info(
        "Initial leaders: alpha=%.4f beta=%.4f delta=%.4f",
        alpha_wolf.fitness, beta_wolf.fitness, delta_wolf.fitness,
    )

    for iteration in range(1, max_iter + 1):
        for i in range(n_wolves):
            x_new = [0.0] * dim
            x_decoded = [0] * dim

            for j in range(dim):
                d_alpha = abs(C1 * alpha_wolf.position[j] - population[i].position[j])
                d_beta = abs(C2 * beta_wolf.position[j] - population[i].position[j])
                d_delta = abs(C3 * delta_wolf.position[j] - population[i].position[j])

                x1 = alpha_wolf.position[j] - A1 * d_alpha
                x2 = beta_wolf.position[j] - A2 * d_beta
                x3 = delta_wolf.position[j] - A3 * d_delta
                x_new[j] = (x1 + x2 + x3) / 3.0

                x_decoded[j] = sigmoid_transfer(x_new[j])

            if feature_budget is not None:
                x_decoded = budget_mask(x_new)
            fitness_val, score_val = evaluate_fn(x_decoded)

            if fitness_val > population[i].fitness:
                population[i].position = copy.copy(x_new)
                population[i].fitness = fitness_val
                population[i].features = copy.copy(x_decoded)
                population[i].score = score_val

            b = 2 - iteration * (2 / max_iter)
            A1, A2, A3, C1, C2, C3 = _coefficients(b, rand)

        alpha_wolf, beta_wolf, delta_wolf = _initialize_leaders(population)
        if on_iteration is not None:
            on_iteration(iteration, alpha_wolf.fitness, alpha_wolf.score, sum(alpha_wolf.features))
        logger.info(
            "Iteration %d/%d | best fitness=%.4f score=%.4f n_features=%d",
            iteration, max_iter, alpha_wolf.fitness, alpha_wolf.score, sum(alpha_wolf.features),
        )

    return alpha_wolf


def binary_grey_wolf_optimizer(
    evaluate_fn: EvaluateFn,
    dim: int,
    n_wolves: int,
    max_iter: int,
    lb: float,
    ub: float,
    seed: int = 42,
    *, on_iteration=None,
) -> Wolf:
    """
    Binary Grey Wolf Optimization (bGWO) for feature selection. Each
    wolf's position update produces three candidate binary votes (one
    derived from each of alpha, beta, delta) which are then combined
    via three-way crossover into the new feature mask. This variant
    generally explores the discrete feature-selection space more
    directly than the plain sigmoid-decoded GWO above.

    Parameters and return value: identical to grey_wolf_optimizer().
    """
    from src.gwo.fitness import fitness_score

    if dim < 1 or n_wolves < 3 or max_iter < 0 or lb >= ub:
        raise ValueError("Require dim >= 1, wolves >= 3, iterations >= 0 and lower < upper.")
    rand = random.Random(seed)
    population = [Wolf(fitness_score, dim, seed=seed + i, lb=lb, ub=ub) for i in range(n_wolves)]
    # Every score must correspond to the actual mask. Include an all-feature
    # reference and diverse starting subsets; never reuse another mask's score.
    for index, wolf in enumerate(population):
        if index:
            wolf.features = [int(x >= (lb + ub) / 2) for x in wolf.position]
            if not any(wolf.features):
                wolf.features[index % dim] = 1
        wolf.fitness, wolf.score = evaluate_fn(wolf.features)


    b = 2.0
    A1, A2, A3, C1, C2, C3 = _coefficients(b, rand)
    alpha_wolf, beta_wolf, delta_wolf = _initialize_leaders(population)
    logger.info(
        "Initial leaders: alpha=%.4f beta=%.4f delta=%.4f",
        alpha_wolf.fitness, beta_wolf.fitness, delta_wolf.fitness,
    )

    for iteration in range(1, max_iter + 1):
        for i in range(n_wolves):
            x_new = [0.0] * dim
            x_decoded = [0] * dim

            for j in range(dim):
                d_alpha = abs(C1 * alpha_wolf.position[j] - population[i].position[j])
                d_beta = abs(C2 * beta_wolf.position[j] - population[i].position[j])
                d_delta = abs(C3 * delta_wolf.position[j] - population[i].position[j])

                x1 = alpha_wolf.position[j] - A1 * d_alpha
                x2 = beta_wolf.position[j] - A2 * d_beta
                x3 = delta_wolf.position[j] - A3 * d_delta
                combined = (x1 + x2 + x3) / 3.0
                x_new[j] = min(max(combined, lb), ub)

                vote_alpha = binary_position_component(alpha_wolf.position[j] >= 0.5, A1, d_alpha, rand)
                vote_beta = binary_position_component(beta_wolf.position[j] >= 0.5, A2, d_beta, rand)
                vote_delta = binary_position_component(delta_wolf.position[j] >= 0.5, A3, d_delta, rand)
                x_decoded[j] = three_way_crossover(vote_alpha, vote_beta, vote_delta, rand)

            fitness_val, score_val = evaluate_fn(x_decoded)

            if fitness_val > population[i].fitness:
                population[i].position = copy.copy(x_new)
                population[i].fitness = fitness_val
                population[i].features = copy.copy(x_decoded)
                population[i].score = score_val

            b = 2 - iteration * (2 / max_iter)
            A1, A2, A3, C1, C2, C3 = _coefficients(b, rand)

        alpha_wolf, beta_wolf, delta_wolf = _initialize_leaders(population)
        if on_iteration is not None:
            on_iteration(iteration, alpha_wolf.fitness, alpha_wolf.score, sum(alpha_wolf.features))
        logger.info(
            "Iteration %d/%d | best fitness=%.4f score=%.4f n_features=%d",
            iteration, max_iter, alpha_wolf.fitness, alpha_wolf.score, sum(alpha_wolf.features),
        )

    return alpha_wolf


def run_gwo(
    variant: str,
    evaluate_fn: EvaluateFn,
    dim: int,
    n_wolves: int,
    max_iter: int,
    lb: float,
    ub: float,
    seed: int = 42,
) -> Wolf:
    """
    Dispatch to the requested GWO variant.

    Parameters
    ----------
    variant : str
        "GWO" or "bGWO".
    (remaining parameters as in grey_wolf_optimizer / binary_grey_wolf_optimizer)

    Returns
    -------
    Wolf
    """
    if variant == "GWO":
        return grey_wolf_optimizer(evaluate_fn, dim, n_wolves, max_iter, lb, ub, seed)
    elif variant == "bGWO":
        return binary_grey_wolf_optimizer(evaluate_fn, dim, n_wolves, max_iter, lb, ub, seed)
    else:
        raise ValueError(f"Unknown GWO variant: {variant!r}. Use 'GWO' or 'bGWO'.")
