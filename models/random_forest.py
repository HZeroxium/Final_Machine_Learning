# models/random_forest.py

from models.base import ModelTrainer
from models.tree import DecisionTreeClassifier
import numpy as np


class RandomForestModel(ModelTrainer):
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.trees = []

    def train(self, X_train, y_train):
        for i in range(self.n_estimators):
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_sample = X_train[indices]
            y_sample = y_train[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
            )
            tree.train(X_sample, y_sample)
            self.trees.append(tree)
            self.log(f"Trained tree {i + 1}/{self.n_estimators}")

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions
        )
        return majority_vote
