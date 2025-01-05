# models/base.py

import numpy as np


class ModelTrainer:
    def __init__(self, verbose=False, **kwargs):
        self.params = kwargs
        self.verbose = verbose
        self.model = None

    def log(self, message):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def load_model_weights(self, filepath):
        import json

        with open(filepath, "r") as f:
            self.model_weights = json.load(f)
        self.log(f"Loaded model weights from {filepath}.")

    def save_model_weights(self, filepath):
        import json

        with open(filepath, "w") as f:
            json.dump(self.model_weights, f)
        self.log(f"Saved model weights to {filepath}.")

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
