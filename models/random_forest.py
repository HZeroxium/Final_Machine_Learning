# models/random_forest.py

from models.base import ModelTrainer
from models.tree import DecisionTreeClassifier
import numpy as np
import json


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

    def save_model_weights(self, filepath):
        model_weights = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "trees": [self._serialize_tree(tree.tree) for tree in self.trees],
        }

        with open(filepath, "w") as f:
            json.dump(model_weights, f)
        self.log(f"Saved model weights to {filepath}.")

    def load_model_weights(self, filepath):
        with open(filepath, "r") as f:
            model_weights = json.load(f)

        self.n_estimators = model_weights["n_estimators"]
        self.max_depth = model_weights["max_depth"]
        self.min_samples_split = model_weights["min_samples_split"]
        self.min_samples_leaf = model_weights["min_samples_leaf"]
        self.criterion = model_weights["criterion"]

        self.trees = [self._deserialize_tree(tree) for tree in model_weights["trees"]]
        self.log(f"Loaded model weights from {filepath}.")

    def _serialize_tree(self, tree):
        """Chuyển đổi cây thành định dạng có thể lưu trữ JSON."""
        if tree is None:
            return None
        if tree["leaf"]:
            return {"leaf": True, "class": int(tree["class"])}  # Đảm bảo kiểu int
        return {
            "leaf": False,
            "feature": int(tree["feature"]),
            "threshold": float(tree["threshold"]),
            "left": self._serialize_tree(tree["left"]),
            "right": self._serialize_tree(tree["right"]),
        }

    def _deserialize_tree(self, tree_data):
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
        )
        tree.tree = tree_data
        return tree
