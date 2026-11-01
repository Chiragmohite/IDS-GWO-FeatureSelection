"""
src/evaluation/confusion_matrix.py

Confusion matrix computation and plotting.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> np.ndarray:
    """Compute the confusion matrix; labels controls row/column class order."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    Render a confusion matrix heatmap and optionally save it to disk.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    labels : Optional[List]
        Class label order for rows/columns. Defaults to sorted unique
        labels found in y_true/y_pred.
    title : str
    save_path : Optional[str]
        If given, the figure is saved to this path (PNG).
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = compute_confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
