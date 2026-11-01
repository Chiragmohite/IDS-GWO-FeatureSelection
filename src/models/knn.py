"""
src/models/knn.py

K-Nearest Neighbors classifier wrapper. Useful as a simple
distance-based comparison point, particularly sensitive to feature
scaling and to irrelevant/noisy features -- which makes it a good
model for visibly demonstrating the value of GWO feature selection.
"""

from sklearn.neighbors import KNeighborsClassifier

from src.models.base_model import BaseModel


class KNNModel(BaseModel):
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def build(self):
        self.model = KNeighborsClassifier(**self.params)
        return self.model
