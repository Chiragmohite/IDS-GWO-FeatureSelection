"""
src/models/random_forest.py

Random Forest classifier wrapper. Ensemble of decision trees; robust
default choice for tabular intrusion-detection data, handles the
mixed categorical/numeric-encoded feature space well and gives usable
feature importances as a bonus diagnostic.
"""

from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def build(self):
        self.model = RandomForestClassifier(**self.params)
        return self.model
