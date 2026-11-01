"""
src/preprocessing/normalization.py

Single responsibility: scale numeric (non-categorical) feature columns
using StandardScaler, fit on training data only and applied to test
data via transform() -- never fit twice, which is a common source of
train/test leakage bugs.
"""

from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_numeric_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    numeric_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the numeric columns of X_train and apply
    the same fitted transform to X_test.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (categorical columns already encoded
        to integers so the whole array is numeric-castable).
    X_test : np.ndarray
        Testing feature matrix.
    numeric_indices : List[int]
        Column indices to scale (i.e. every column except the
        categorical ones -- categorical columns are left as their
        encoded integer codes and are not standard-scaled).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[:, numeric_indices] = scaler.fit_transform(X_train[:, numeric_indices])
    X_test[:, numeric_indices] = scaler.transform(X_test[:, numeric_indices])

    return X_train, X_test, scaler


def numeric_indices_excluding(total_features: int, categorical_indices: List[int]) -> List[int]:
    """
    Compute the list of numeric column indices as the complement of the
    categorical indices within range(total_features).

    Parameters
    ----------
    total_features : int
        Total number of feature columns.
    categorical_indices : List[int]
        Indices that are categorical and should be excluded.

    Returns
    -------
    List[int]
    """
    categorical_set = set(categorical_indices)
    return [i for i in range(total_features) if i not in categorical_set]
