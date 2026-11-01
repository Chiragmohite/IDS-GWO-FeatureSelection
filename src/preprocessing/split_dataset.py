"""
src/preprocessing/split_dataset.py

Single responsibility: dataset splitting. The KDD-derived dataset used
by this project already ships with a dedicated train/test file pair, so
the primary split is just loading those two files separately (handled
in load_dataset.py). This module additionally provides a stratified
train/validation splitter for anyone who wants to carve a validation
set out of the training data (e.g. for hyperparameter tuning without
touching the held-out test set).
"""

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a feature matrix and target vector into training and
    validation subsets.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    test_size : float
        Fraction of data to hold out for validation.
    stratify : bool
        Whether to preserve class proportions in the split (recommended
        given the class imbalance typical of intrusion-detection
        datasets).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_val, y_train, y_val)
    """
    stratify_arg = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=stratify_arg, random_state=random_state
    )
    return X_train, X_val, y_train, y_val
