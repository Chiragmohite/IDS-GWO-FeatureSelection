"""
src/evaluation/roc_curve.py

ROC curve computation and plotting for the multi-class intrusion
classification target, using one-vs-rest binarization (the original
repository did not include ROC analysis; this is new functionality
added per the "Implementation Requirements" spec).
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: List,
    title: str = "ROC Curve (One-vs-Rest)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot one ROC curve per class using one-vs-rest binarization, plus
    each class's AUC in the legend.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_score : np.ndarray
        Predicted probability/decision scores, shape (n_samples, n_classes),
        e.g. from classifier.predict_proba(X_test).
    class_labels : List
        Ordered list of class labels matching the columns of y_score.
    title : str
    save_path : Optional[str]
        If given, the figure is saved to this path (PNG) instead of shown.
    """
    y_true_bin = label_binarize(y_true, classes=class_labels)
    n_classes = len(class_labels)

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        if y_true_bin[:, i].sum() == 0:
            # Class not present in this evaluation split -- skip rather
            # than error out on an undefined ROC curve.
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
