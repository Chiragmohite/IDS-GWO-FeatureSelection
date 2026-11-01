"""
src/utils/helper.py

General-purpose I/O and bookkeeping helpers: output-directory creation,
model save/load, and selected-feature-mask persistence. Centralizing
these avoids every module reinventing its own `os.makedirs` / pickle
boilerplate.
"""

import json
import logging
import os
from typing import List

import joblib

logger = logging.getLogger(__name__)


def ensure_directories(paths: List[str]) -> None:
    """Create every directory in `paths` if it does not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
    logger.info("Verified %d output directories.", len(paths))


def save_model(model, path: str) -> None:
    """Persist a fitted sklearn-compatible model to disk via joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str):
    """Load a previously saved model from disk."""
    return joblib.load(path)


def save_selected_features(feature_mask: List[int], feature_names: List[str], path: str) -> None:
    """
    Persist the GWO-selected binary feature mask alongside the
    corresponding feature names as a small CSV, so the selection is
    human-readable and directly reusable at inference time.
    """
    import pandas as pd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "feature_name": feature_names,
        "selected": feature_mask,
    })
    df.to_csv(path, index=False)
    logger.info("Saved selected features (%d/%d selected) to %s",
                sum(feature_mask), len(feature_mask), path)


def save_json(data: dict, path: str) -> None:
    """Persist an arbitrary dict as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON report to %s", path)
