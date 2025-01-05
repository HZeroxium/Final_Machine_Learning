# models/tree.py

import numpy as np


class DecisionTreeClassifier:
    def __init__(
        self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="gini"
    ):
        """
        Initialize Decision Tree Classifier.
        Parameters:
        - max_depth: Maximum depth of the tree.
        - min_samples_split: Minimum number of samples required to split.
        - min_samples_leaf: Minimum number of samples required in a leaf.
        - criterion: "gini" or "entropy" for split quality measure.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None

    def _gini(self, y):
        """Compute Gini Impurity."""
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        gini = 1 - np.sum(prob**2)
        return gini

    def _entropy(self, y):
        """Compute Entropy."""
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-9))  # Add epsilon to avoid log(0)
        return entropy

    def _split(self, X, y):
        """Find the best split for the data."""
        best_score = float("inf")
        best_split = None
        m, n = X.shape

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Skip if split creates empty groups
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate the chosen criterion
                if self.criterion == "gini":
                    left_score = self._gini(y[left_mask])
                    right_score = self._gini(y[right_mask])
                elif self.criterion == "entropy":
                    left_score = self._entropy(y[left_mask])
                    right_score = self._entropy(y[right_mask])
                else:
                    raise ValueError(f"Unknown criterion: {self.criterion}")

                # Weighted score
                weighted_score = (
                    len(y[left_mask]) / m * left_score
                    + len(y[right_mask]) / m * right_score
                )

                # Update the best split if the score improves
                if weighted_score < best_score:
                    best_score = weighted_score
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left": left_mask,
                        "right": right_mask,
                    }
        return best_split

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        if (
            len(np.unique(y)) == 1
            or depth == self.max_depth
            or len(y) < self.min_samples_split
        ):
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        split = self._split(X, y)
        if (
            split is None
            or np.sum(split["left"]) < self.min_samples_leaf
            or np.sum(split["right"]) < self.min_samples_leaf
        ):
            return {"leaf": True, "class": np.argmax(np.bincount(y))}

        left_subtree = self._build_tree(X[split["left"]], y[split["left"]], depth + 1)
        right_subtree = self._build_tree(
            X[split["right"]], y[split["right"]], depth + 1
        )

        return {
            "leaf": False,
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": left_subtree,
            "right": right_subtree,
        }

    def train(self, X, y):
        """Train the decision tree classifier."""
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_single(self, node, x):
        """Predict for a single sample."""
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(node["left"], x)
        else:
            return self._predict_single(node["right"], x)

    def predict(self, X):
        """Predict for multiple samples."""
        return np.array([self._predict_single(self.tree, x) for x in X])


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the DecisionTreeRegressor.
        Parameters:
        - max_depth: Maximum depth of the tree.
        - min_samples_split: Minimum number of samples required to split.
        - min_samples_leaf: Minimum number of samples required in a leaf.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _mse(self, y):
        """Compute Mean Squared Error (MSE)."""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _split(self, X, y):
        """Find the best split for the data."""
        best_mse = float("inf")
        best_split = None
        m, n = X.shape

        for feature in range(n):
            # Optimize by considering only midpoints between sorted unique values
            thresholds = np.unique((X[:-1, feature] + X[1:, feature]) / 2)
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Skip if split results in empty groups
                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                # Compute MSE for left and right splits
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])

                # Compute weighted MSE
                weighted_mse = (
                    len(y[left_mask]) / m * left_mse
                    + len(y[right_mask]) / m * right_mse
                )

                # Update best split if current split improves MSE
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left": left_mask,
                        "right": right_mask,
                    }
        return best_split

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        # Stop conditions
        if (
            len(y) <= self.min_samples_split
            or depth == self.max_depth
            or self._mse(y) < 1e-6  # Stop if variance is very small
        ):
            return {"leaf": True, "value": np.mean(y)}

        # Find the best split
        split = self._split(X, y)
        if split is None:
            return {"leaf": True, "value": np.mean(y)}

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[split["left"]], y[split["left"]], depth + 1)
        right_subtree = self._build_tree(
            X[split["right"]], y[split["right"]], depth + 1
        )

        return {
            "leaf": False,
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": left_subtree,
            "right": right_subtree,
        }

    def train(self, X, y):
        """Train the Decision Tree Regressor."""
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_single(self, node, x):
        """Predict for a single sample."""
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(node["left"], x)
        else:
            return self._predict_single(node["right"], x)

    def predict(self, X):
        """Predict for multiple samples."""
        return np.array([self._predict_single(self.tree, x) for x in X])
