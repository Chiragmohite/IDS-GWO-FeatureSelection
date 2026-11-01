"""
src/preprocessing/unsw_preprocessing.py

UNSW-NB15-specific preprocessing. Reuses the existing generic cleaning
utilities in clean_data.py, but the encoding/scaling/target-selection
logic here is dataset-specific because:
  - UNSW-NB15's categorical columns and cardinality differ from NSL-KDD's
  - `proto` has 130+ raw values and needs rare-category bucketing
  - both a binary (`label`) and multiclass (`attack_cat`) target exist
  - every fitted preprocessing object MUST be persisted to disk, because
    the real-time runtime pipeline (src/runtime/feature_extraction.py)
    has to apply the EXACT SAME encoder/scaler to live-extracted
    features -- refitting at inference time would silently corrupt
    predictions.

This module never fits on test data -- every encoder/scaler is fit on
the training split only and reused (via .transform()) on test data and,
later, on live runtime data.
"""

import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

logger = logging.getLogger(__name__)


def load_unsw_raw(path: str) -> pd.DataFrame:
    """Load a raw UNSW-NB15 CSV and normalize column names to lowercase."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean_unsw(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNSW-specific cleaning: drop the `id` column (not predictive, just a
    row index), drop exact duplicate rows, and verify there are no
    missing values (the raw files have none, but this fails loudly
    rather than silently if a different UNSW variant does).
    """
    if config.UNSW_ID_COLUMN in df.columns:
        df = df.drop(columns=[config.UNSW_ID_COLUMN])

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows (%d -> %d).", n_before - len(df), n_before, len(df))

    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        logger.warning("Found %d missing values -- dropping affected rows.", n_missing)
        df = df.dropna().reset_index(drop=True)

    return df


def bucket_rare_protocols(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: int = 20,
    column: str = "proto",
) -> Tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    Replace any `proto` value occurring fewer than `threshold` times in
    the TRAINING data with a shared "other" category, applied to both
    train and test using the same training-derived rare-value set. This
    keeps the categorical dimension from exploding across 130+ mostly-rare
    protocol values and, critically, makes the fitted encoder robust to
    protocol values seen live at runtime that never appeared (or barely
    appeared) in training -- those also collapse to "other" instead of
    crashing the encoder on an unseen category.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, set]
        (df_train, df_test, rare_value_set) -- rare_value_set is saved
        into the feature schema so the runtime pipeline applies the
        identical bucketing rule.
    """
    counts = df_train[column].value_counts()
    rare_values = set(counts[counts < threshold].index)

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train[column] = df_train[column].apply(lambda v: "other" if v in rare_values else v)
    df_test[column] = df_test[column].apply(lambda v: "other" if v in rare_values else v)

    logger.info(
        "Bucketed %d rare '%s' values (< %d occurrences) into 'other'.",
        len(rare_values), column, threshold,
    )
    return df_train, df_test, rare_values


def encode_categoricals(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Fit one LabelEncoder per categorical column on TRAINING data only,
    apply to both train and test. Any category seen in test (or later,
    at runtime) that was not seen during training is mapped to a
    reserved "__unknown__" bucket included when fitting the encoder, so the
    pipeline never crashes on a novel category value.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]
        (df_train, df_test, fitted_encoders_by_column_name)
    """
    df_train = df_train.copy()
    df_test = df_test.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit(list(df_train[col].astype(str)) + ["__unknown__"])
        known = set(encoder.classes_)
        df_train[col] = encoder.transform(df_train[col].astype(str))
        values = df_test[col].astype(str)
        df_test[col] = encoder.transform(values.where(values.isin(known), "__unknown__"))

        encoders[col] = encoder

    return df_train, df_test, encoders


def scale_numeric(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    numeric_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on the numeric columns of the training data only,
    apply to both train and test.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    scaler = StandardScaler()
    df_train[numeric_columns] = scaler.fit_transform(df_train[numeric_columns])
    df_test[numeric_columns] = scaler.transform(df_test[numeric_columns])

    return df_train, df_test, scaler


def encode_target(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode the target column. For `label` (binary, already 0/1 integers)
    this is a no-op wrapped in a trivial encoder for interface
    consistency. For `attack_cat` (multiclass strings) this fits a real
    LabelEncoder on the training labels.
    """
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df_train[target_column])
    y_test = encoder.transform(df_test[target_column])
    return y_train, y_test, encoder


def save_preprocessing_artifacts(
    categorical_encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    rare_proto_values: set,
) -> None:
    """
    Persist every fitted preprocessing object so the real-time runtime
    pipeline can apply IDENTICAL transforms to live-extracted features,
    with no refitting and no guessing.
    """
    joblib.dump(
        {"categorical_encoders": categorical_encoders, "rare_proto_values": rare_proto_values},
        config.ENCODER_PATH,
    )
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    logger.info("Saved encoder -> %s", config.ENCODER_PATH)
    logger.info("Saved scaler -> %s", config.SCALER_PATH)
    logger.info("Saved label encoder -> %s", config.LABEL_ENCODER_PATH)


def load_preprocessing_artifacts():
    """
    Load previously saved preprocessing artifacts for inference (used by
    both offline re-evaluation and the real-time runtime pipeline).

    Returns
    -------
    Tuple[dict, StandardScaler, LabelEncoder]
        (encoder_bundle, scaler, label_encoder), where encoder_bundle is
        {"categorical_encoders": {...}, "rare_proto_values": {...}}.
    """
    encoder_bundle = joblib.load(config.ENCODER_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    return encoder_bundle, scaler, label_encoder
