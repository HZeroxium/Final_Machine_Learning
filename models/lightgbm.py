# models/lightgbm.py

from models.base import ModelTrainer
from models.tree import DecisionTreeRegressor
import numpy as np


class LightGBMModel(ModelTrainer):

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=None,
        lambda_l1=0.0,
        lambda_l2=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.trees = []
        self.tree_weights = []

    def train(self, X_train, y_train):
        residuals = y_train.astype(np.float64).copy()
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=2, min_samples_leaf=1
            )
            tree.train(X_train, residuals)
            predictions = tree.predict(X_train)

            # Add regularization (scaled by number of samples)
            l1_penalty = self.lambda_l1 * np.abs(predictions).sum() / len(predictions)
            l2_penalty = self.lambda_l2 * (predictions**2).sum() / len(predictions)

            # Update residuals
            residuals_update = self.learning_rate * (
                predictions + l1_penalty + l2_penalty
            )
            residuals_update = np.nan_to_num(
                residuals_update, nan=0.0, posinf=1e6, neginf=-1e6
            )
            residuals -= residuals_update

            self.trees.append(tree)
            self.tree_weights.append(self.learning_rate)
            self.log(
                f"Trained tree {i + 1}/{self.n_estimators} with residual MSE: {np.mean(residuals**2):.4f}"
            )

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree, weight in zip(self.trees, self.tree_weights):
            predictions += weight * tree.predict(X)
        return np.round(predictions)
