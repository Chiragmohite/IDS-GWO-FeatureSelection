"""
src/models/model_factory.py

Central factory so main.py and the GWO feature selector can obtain a
*fresh, unfitted* classifier instance by name, driven entirely by
config.py -- no other module should import a specific sklearn
estimator class directly.
"""

from typing import Callable

import config
from src.models.base_model import BaseModel
from src.models.decision_tree import DecisionTreeModel
from src.models.knn import KNNModel
from src.models.random_forest import RandomForestModel
from src.models.svm import SVMModel

_MODEL_REGISTRY = {
    "random_forest": (RandomForestModel, "RANDOM_FOREST_PARAMS"),
    "decision_tree": (DecisionTreeModel, "DECISION_TREE_PARAMS"),
    "svm": (SVMModel, "SVM_PARAMS"),
    "knn": (KNNModel, "KNN_PARAMS"),
}


def get_model_factory(name: str) -> Callable[[], BaseModel]:
    """
    Return a zero-argument factory function that constructs a fresh
    instance of the named model, with hyperparameters pulled from
    config.py.

    Parameters
    ----------
    name : str
        One of "random_forest", "decision_tree", "svm", "knn".

    Returns
    -------
    Callable[[], BaseModel]
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")

    model_cls, params_attr = _MODEL_REGISTRY[name]
    params = getattr(config, params_attr)

    def factory():
        instance = model_cls(**params)
        instance.build()
        return instance.model  # return the raw sklearn estimator for GWO's fit/predict use

    return factory


def build_model(name: str) -> BaseModel:
    """
    Construct and return a ready-to-use (built, unfitted) BaseModel
    wrapper instance for the named model.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")

    model_cls, params_attr = _MODEL_REGISTRY[name]
    params = getattr(config, params_attr)
    instance = model_cls(**params)
    instance.build()
    return instance
