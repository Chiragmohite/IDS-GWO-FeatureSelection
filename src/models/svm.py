"""
src/models/svm.py

Support Vector Machine classifier wrapper (RBF kernel by default),
matching the classifier used in the original repository's notebook.
"""

from sklearn.svm import SVC

from src.models.base_model import BaseModel


class SVMModel(BaseModel):
    def __init__(self, **params):
        super().__init__()
        self.params = dict(params)

    def build(self):
        # SVC only accepts random_state when probability=True; drop it
        # otherwise so config.SVM_PARAMS can safely always include it.
        params = dict(self.params)
        if not params.get("probability", False):
            params.pop("random_state", None)
        self.model = SVC(**params)
        return self.model
