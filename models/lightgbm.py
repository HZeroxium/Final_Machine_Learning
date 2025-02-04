# models/lightgbm.py

from models.base import ModelTrainer
from models.tree import DecisionTreeRegressor
import numpy as np
import json


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

    def save_model_weights(self, filepath):
        """Save model weights, trees, and parameters to a file."""
        model_data = {
            "trees": [tree.tree for tree in self.trees],
            "tree_weights": self.tree_weights,
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "lambda_l1": self.lambda_l1,
                "lambda_l2": self.lambda_l2,
            },
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f)
        self.log(f"Model weights saved to {filepath}")

    def load_model_weights(self, filepath):
        """Load model weights, trees, and parameters from a file."""
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # Restore parameters
        params = model_data["params"]
        self.n_estimators = params["n_estimators"]
        self.learning_rate = params["learning_rate"]
        self.max_depth = params["max_depth"]
        self.lambda_l1 = params["lambda_l1"]
        self.lambda_l2 = params["lambda_l2"]

        # Restore trees and weights
        self.trees = [
            DecisionTreeRegressor(max_depth=self.max_depth) for _ in model_data["trees"]
        ]
        for tree, tree_data in zip(self.trees, model_data["trees"]):
            tree.tree = tree_data

        self.tree_weights = model_data["tree_weights"]
        self.log(f"Model weights loaded from {filepath}")
