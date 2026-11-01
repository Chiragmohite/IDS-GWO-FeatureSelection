"""
src/gwo/feature_selector.py

Orchestration layer that wires the abstract GWO algorithm (gwo_algorithm.py)
to a concrete classifier and dataset. This is the "objective_function" /
"generate" / "AAP" / "ANF" logic from the original notebook, refactored
into a reusable, class-based selector instead of relying on module-level
globals (the original notebook reassigned a global `classifier` and
`sc_X` inside nested functions, which is fragile and hard to test).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.gwo.fitness import fitness_score
from src.gwo.gwo_algorithm import run_gwo
from src.gwo.wolf import Wolf

logger = logging.getLogger(__name__)


@dataclass
class GWORunResult:
    """Container for the outcome of one or more GWO runs."""
    best_wolf: Wolf
    best_accuracy: float
    best_accuracy_features: List[int]
    all_scores: List[float] = field(default_factory=list)
    all_feature_counts: List[int] = field(default_factory=list)

    @property
    def average_accuracy_performance(self) -> float:
        """AAP: mean accuracy across all GWO runs."""
        return float(np.mean(self.all_scores)) if self.all_scores else 0.0

    @property
    def average_selected_features(self) -> float:
        """ANF: mean number of selected features across all GWO runs."""
        return float(np.mean(self.all_feature_counts)) if self.all_feature_counts else 0.0


class GWOFeatureSelector:
    """
    Runs Grey Wolf Optimization feature selection against a given
    classifier and a fixed training/validation split, evaluating each candidate
    feature subset by retraining the classifier on just those features
    and scoring it against the validation set (never the final test set).

    Parameters
    ----------
    classifier_factory : callable
        A zero-argument callable that returns a *fresh, unfitted*
        classifier instance (e.g. `lambda: RandomForestClassifier(**params)`).
        A factory (not a shared instance) is required because each
        fitness evaluation must retrain from scratch on a different
        feature subset.
    X_train, y_train, X_test, y_test : np.ndarray
        Pre-encoded (categoricals already label-encoded) but
        *unscaled* feature matrices/targets. Scaling is applied
        per-candidate inside this class because the set of numeric
        columns to scale changes with the selected feature subset.
    categorical_indices : List[int]
        Column indices (in the ORIGINAL full feature space) that are
        categorical and must not be passed through StandardScaler.
    alpha, beta : float
        Fitness function weights, see gwo.fitness.fitness_score.
    """

    def __init__(
        self,
        classifier_factory,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        categorical_indices: List[int],
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.classifier_factory = classifier_factory
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.categorical_indices = set(categorical_indices)
        self.alpha = alpha
        self.beta = beta
        self.dim = X_train.shape[1]

        # Baseline: all features, no selection -- used as the fitness
        # reference score for the initial wolf population.
        self._cache = {}
        self.baseline_accuracy = self._evaluate_feature_mask([1] * self.dim)[1]

    def _evaluate_full_feature_set(self) -> float:
        clf = self.classifier_factory()
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        return accuracy_score(self.y_test, preds) * 100

    def _evaluate_feature_mask(self, feature_mask: List[int]) -> Tuple[float, float]:
        """
        Train and evaluate the classifier restricted to the columns
        selected by `feature_mask`. Returns (fitness, accuracy).
        """
        key = tuple(feature_mask)
        if len(key) != self.dim or any(v not in (0, 1) for v in key):
            raise ValueError("Feature mask must contain one binary value per feature.")
        if key in self._cache:
            return self._cache[key]
        n_selected = sum(feature_mask)
        if n_selected == 0:
            # No features selected is not a usable model; heavily
            # penalize rather than crash.
            return float("-inf"), 0.0

        selected_idx = [i for i in range(self.dim) if feature_mask[i] == 1]
        scale_idx = [i for i in selected_idx if i not in self.categorical_indices]

        X_train_subset = self.X_train[:, selected_idx].astype(float)
        X_test_subset = self.X_test[:, selected_idx].astype(float)

        # Recompute which columns *within the subset* need scaling.
        subset_scale_positions = [selected_idx.index(i) for i in scale_idx]
        if subset_scale_positions:
            scaler = StandardScaler()
            X_train_subset[:, subset_scale_positions] = scaler.fit_transform(
                X_train_subset[:, subset_scale_positions]
            )
            X_test_subset[:, subset_scale_positions] = scaler.transform(
                X_test_subset[:, subset_scale_positions]
            )

        clf = self.classifier_factory()
        clf.fit(X_train_subset, self.y_train)
        preds = clf.predict(X_test_subset)
        accuracy = accuracy_score(self.y_test, preds) * 100

        fitness_val = fitness_score(accuracy, n_selected, alpha=self.alpha, beta=self.beta)
        self._cache[key] = (fitness_val, accuracy)
        return fitness_val, accuracy

    def run(self, variant: str, n_wolves: int, max_iter: int, lb: float = 0.0, ub: float = 1.0, seed: int = 42) -> Wolf:
        """
        Execute a single GWO (or bGWO) run and return the best wolf found.
        """
        return run_gwo(
            variant=variant,
            evaluate_fn=self._evaluate_feature_mask,
            dim=self.dim,
            n_wolves=n_wolves,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            seed=seed,
        )

    def run_multiple(
        self,
        variant: str,
        n_runs: int,
        n_wolves: int,
        max_iter: int,
        lb: float = 0.0,
        ub: float = 1.0,
        seed: int = 42,
    ) -> GWORunResult:
        """
        Execute several independent GWO runs (as in the original
        notebook's `run` parameter) and aggregate results, tracking
        both the single best wolf (by fitness) and the best raw
        accuracy seen across all runs -- these can differ, since
        fitness also rewards fewer features.

        Returns
        -------
        GWORunResult
        """
        if n_runs < 1:
            raise ValueError("At least one GWO run is required.")
        best_wolf: Wolf = None
        best_accuracy = self.baseline_accuracy
        best_accuracy_features = [1] * self.dim
        all_scores, all_feature_counts = [], []

        for run_idx in range(n_runs):
            logger.info("=== GWO run %d/%d (variant=%s) ===", run_idx + 1, n_runs, variant)
            candidate = self.run(variant, n_wolves, max_iter, lb, ub, seed + run_idx)

            all_scores.append(candidate.score)
            all_feature_counts.append(sum(candidate.features))

            if candidate.score > best_accuracy:
                best_accuracy = candidate.score
                best_accuracy_features = candidate.features

            if best_wolf is None or candidate.fitness >= best_wolf.fitness:
                best_wolf = candidate

        return GWORunResult(
            best_wolf=best_wolf,
            best_accuracy=best_accuracy,
            best_accuracy_features=best_accuracy_features,
            all_scores=all_scores,
            all_feature_counts=all_feature_counts,
        )
