"""
src/utils/visualization.py

Plots that are not tied to a single evaluation metric: GWO convergence
curves (fitness/accuracy over iterations) and feature-count comparison
bars. Model-specific evaluation plots (confusion matrix, ROC) live in
src/evaluation/ instead, since they are metric outputs rather than
general visual diagnostics.
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt


def plot_convergence(fitness_per_iteration: List[float], title: str = "GWO Convergence", save_path: Optional[str] = None) -> None:
    """
    Plot best-fitness-so-far across GWO iterations, to visually confirm
    the optimizer is actually converging rather than wandering.

    Parameters
    ----------
    fitness_per_iteration : List[float]
        Best fitness value recorded at each iteration.
    title : str
    save_path : Optional[str]
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(fitness_per_iteration) + 1), fitness_per_iteration, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_feature_count_comparison(
    labels: List[str],
    feature_counts: List[int],
    title: str = "Feature Count: Baseline vs GWO-Selected",
    save_path: Optional[str] = None,
) -> None:
    """
    Simple bar chart comparing the number of features used by each
    model variant (e.g. "All Features" vs "GWO-Selected").
    """
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, feature_counts, color=["#4C72B0", "#DD8452"])
    for bar, count in zip(bars, feature_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha="center")
    plt.ylabel("Number of Features")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
