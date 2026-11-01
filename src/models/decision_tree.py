"""
src/models/decision_tree.py

Single Decision Tree classifier wrapper. Useful as a fast, highly
interpretable baseline against which the ensemble/kernel models are
compared.
"""

from sklearn.tree import DecisionTreeClassifier

from src.models.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def build(self):
        self.model = DecisionTreeClassifier(**self.params)
        return self.model
