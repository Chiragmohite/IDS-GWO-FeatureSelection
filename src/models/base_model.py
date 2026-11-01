"""
src/models/base_model.py

Shared base interface all classifier wrappers implement, so any model
in this package can be swapped in wherever a classifier is expected
(GWO fitness evaluation, main.py pipeline, evaluation/comparison.py)
without special-casing per model type.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Minimal common interface: build, fit, predict."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self):
        """Instantiate the underlying sklearn estimator."""
        raise NotImplementedError

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseModel":
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been built/fit yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{type(self.model).__name__} does not support predict_proba.")
        return self.model.predict_proba(X)
