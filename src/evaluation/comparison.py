"""
src/evaluation/comparison.py

Builds a side-by-side comparison table between the baseline model
(trained on ALL features) and the GWO-selected-feature model, which
is the concrete evidence that GWO feature selection actually helped
-- required for the project report/viva, not just "a model that works".
"""

from typing import Dict, List

import pandas as pd

from src.evaluation.metrics import compute_summary_metrics


def build_comparison_table(
    y_test,
    baseline_preds,
    gwo_preds,
    n_total_features: int,
    n_selected_features: int,
) -> pd.DataFrame:
    """
    Build a DataFrame comparing baseline (all-features) vs GWO-selected
    feature model performance.

    Parameters
    ----------
    y_test : array-like
        Ground-truth test labels.
    baseline_preds : array-like
        Predictions from the model trained on all features.
    gwo_preds : array-like
        Predictions from the model trained on GWO-selected features.
    n_total_features : int
        Total number of features available before selection.
    n_selected_features : int
        Number of features GWO selected.

    Returns
    -------
    pd.DataFrame
        One row per model ("Baseline (All Features)", "GWO-Selected
        Features"), columns: accuracy, precision, recall, f1_score,
        n_features, feature_reduction_pct.
    """
    baseline_metrics = compute_summary_metrics(y_test, baseline_preds)
    gwo_metrics = compute_summary_metrics(y_test, gwo_preds)

    reduction_pct = 100 * (1 - n_selected_features / n_total_features) if n_total_features else 0.0

    rows = [
        {
            "model": "Baseline (All Features)",
            "n_features": n_total_features,
            "feature_reduction_pct": 0.0,
            **baseline_metrics,
        },
        {
            "model": "GWO-Selected Features",
            "n_features": n_selected_features,
            "feature_reduction_pct": round(reduction_pct, 2),
            **gwo_metrics,
        },
    ]
    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, path: str) -> None:
    """Persist the comparison table to CSV for direct inclusion in a report."""
    df.to_csv(path, index=False)
