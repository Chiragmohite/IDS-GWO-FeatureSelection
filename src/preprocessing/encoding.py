"""
src/preprocessing/encoding.py

Single responsibility: label-encode the categorical feature columns
(protocol_type, service, flag) using a single shared LabelEncoder,
mirroring the original repository's approach but making the fitted
encoders explicit, returnable, and reusable at inference time instead
of silently refit on both train and test (a subtle bug in the
original notebook -- see README "Design Flaws Fixed" section).
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    categorical_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, LabelEncoder]]:
    """
    Fit a LabelEncoder per categorical column on the TRAINING data only,
    then use that same fitted encoder to transform both train and test.

    NOTE ON A FIXED DESIGN FLAW: the original notebook called
    `fit_transform` separately on X_train and X_test for each categorical
    column, which means the train and test encoders could assign
    different integer codes to the same category (e.g. 'tcp' might not
    map to the same integer in both sets). This refactor fits once per
    column on the training data and reuses that mapping for the test
    data, which is the statistically correct approach and prevents
    silent label leakage/misalignment.

    Any category seen in test but not in train falls back to a new
    unseen-label bucket rather than raising, so inference on live data
    with an unfamiliar category value does not crash the pipeline.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (object dtype, pre-scaling).
    X_test : np.ndarray
        Testing feature matrix (object dtype, pre-scaling).
    categorical_indices : List[int]
        Column indices that are categorical and require encoding.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[int, LabelEncoder]]
        (X_train_encoded, X_test_encoded, fitted_encoders_by_column_index)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    encoders: Dict[int, LabelEncoder] = {}

    for col_idx in categorical_indices:
        encoder = LabelEncoder()
        encoder.fit(list(X_train[:, col_idx].astype(str)) + ["__unknown__"])
        X_train[:, col_idx] = encoder.transform(X_train[:, col_idx].astype(str))
        known = set(encoder.classes_)
        X_test[:, col_idx] = encoder.transform([
            str(v) if str(v) in known else "__unknown__" for v in X_test[:, col_idx]
        ])

        encoders[col_idx] = encoder

    return X_train, X_test, encoders


def split_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a cleaned dataframe into an independent-variable matrix (X)
    and a dependent-variable vector (y), assuming the target is the
    final column (matches the original repository's iloc[:, :-1] /
    iloc[:, -1] convention).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X, y)
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y
