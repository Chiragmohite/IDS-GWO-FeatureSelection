"""
src/evaluation/metrics.py

Core classification metrics: accuracy, precision, recall, F1 (both
per-class and averaged), plus the TP/FP/TN/FN breakdown used by the
original repository's AC (accuracy-from-confusion-counts) and DR
(detection rate) helper functions.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy as a percentage (0-100), matching the original notebook's scale."""
    return accuracy_score(y_true, y_pred) * 100


def compute_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Full sklearn classification report (per-class precision/recall/F1/support)."""
    return classification_report(y_true, y_pred, zero_division=0)


def compute_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Same as compute_classification_report but as a nested dict (for tables/JSON)."""
    return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


def compute_summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> Dict[str, float]:
    """
    Single-row summary metrics dict: accuracy, precision, recall, F1
    (macro-averaged by default, appropriate for the class-imbalanced
    multi-class intrusion categories in this dataset).
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0) * 100,
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0) * 100,
    }


def perf_measure(y_actual: np.ndarray, y_pred: np.ndarray, value) -> Tuple[int, int, int, int]:
    """
    One-vs-rest TP/FP/TN/FN counts for a single class label `value`.
    Direct, vectorized refactor of the original notebook's Python-loop
    `perf_measure` function.

    Parameters
    ----------
    y_actual : np.ndarray
    y_pred : np.ndarray
    value : the class label to treat as the "positive" class

    Returns
    -------
    Tuple[int, int, int, int]
        (TP, FP, TN, FN)
    """
    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)

    is_actual_positive = y_actual == value
    is_pred_positive = y_pred == value

    tp = int(np.sum(is_actual_positive & is_pred_positive))
    fp = int(np.sum(~is_actual_positive & is_pred_positive))
    tn = int(np.sum(~is_actual_positive & ~is_pred_positive))
    fn = int(np.sum(is_actual_positive & ~is_pred_positive))
    return tp, fp, tn, fn


def accuracy_from_confusion(tp: int, fp: int, tn: int, fn: int) -> float:
    """AC = (TP + TN) / (TP + FP + TN + FN). Mirrors the original 'AC'."""
    total = tp + fp + tn + fn
    return (tp + tn) / total if total else 0.0


def detection_rate(tp: int, fn: int) -> float:
    """DR = TP / (TP + FN), i.e. recall for the positive class. Mirrors the original 'DR'."""
    denom = tp + fn
    return tp / denom if denom else 0.0


def average_accuracy_performance(accuracies: list) -> float:
    """AAP: mean accuracy across multiple runs. Mirrors the original 'AAP'."""
    return float(np.mean(accuracies)) if accuracies else 0.0


def average_number_of_features(feature_counts: list) -> float:
    """ANF: mean number of selected features across multiple runs. Mirrors the original 'ANF'."""
    return float(np.mean(feature_counts)) if feature_counts else 0.0


def false_positive_rate(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    FPR = FP / (FP + TN) -- fraction of actual "normal" traffic that got
    wrongly flagged as an attack. High FPR in an IDS means a lot of
    false alarms, which is operationally costly (analysts start ignoring
    alerts).
    """
    denom = fp + tn
    return fp / denom if denom else 0.0


def false_negative_rate(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    FNR = FN / (FN + TP) -- fraction of actual attacks the model missed
    entirely. This is the more dangerous error for a security system:
    a missed attack is a successful intrusion.
    """
    denom = fn + tp
    return fn / denom if denom else 0.0


def compute_binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, positive_label=1) -> Dict[str, int]:
    """
    Convenience wrapper: compute TP/FP/TN/FN for a binary classification
    task in one call, plus FPR/FNR, returned as a flat dict ready for a
    report/JSON.
    """
    tp, fp, tn, fn = perf_measure(y_true, y_pred, positive_label)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "false_positive_rate": false_positive_rate(tp, fp, tn, fn),
        "false_negative_rate": false_negative_rate(tp, fp, tn, fn),
    }
