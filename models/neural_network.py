# models/neural_network.py

from models.base import ModelTrainer
import numpy as np
from utils.activation import Activation


class NeuralNetworkModel(ModelTrainer):
    def __init__(
        self,
        hidden_layer_sizes=[64, 32],
        activation="relu",
        learning_rate=0.01,
        epochs=50,
        batch_size=32,
        dropout_rate=0.5,
        l2_lambda=0.01,
        seed=42,
        early_stopping=False,
        patience=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        np.random.seed(seed)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.early_stopping = early_stopping
        self.patience = patience
        self.weights = []
        self.biases = []
        self.best_weights = None
        self.best_biases = None
        self.training = True

    def initialize_weights(self, n_features, n_classes):
        """Khởi tạo trọng số và bias ngẫu nhiên."""
        layer_sizes = [n_features] + self.hidden_layer_sizes + [n_classes]
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [
            np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, X):
        """Lan truyền xuôi qua mạng."""
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._activation(z)
            if self.training:
                a = self.apply_dropout(a, self.dropout_rate)
            activations.append(a)
        # Lớp đầu ra
        activations.append(self.output_layer(activations[-1]))
        return activations

    def output_layer(self, X):
        """Tính toán lớp đầu ra."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        z = np.dot(X, self.weights[-1]) + self.biases[-1]
        return Activation.softmax(z)

    def backward(self, activations, y_one_hot):
        """Lan truyền ngược để tính gradient."""
        m = y_one_hot.shape[0]
        gradients = {"dW": [], "db": []}
        delta = activations[-1] - y_one_hot
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients["dW"].insert(0, dW)
            gradients["db"].insert(0, db)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(
                    activations[i]
                )
        return gradients

    def update_parameters(self, gradients):
        """Cập nhật trọng số và bias với L2 Regularization."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * (
                gradients["dW"][i] + self.l2_lambda * self.weights[i]
            )
            self.biases[i] -= self.learning_rate * gradients["db"][i]

    def apply_dropout(self, X, dropout_rate):
        """Áp dụng Dropout."""
        mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape)
        return X * mask / (1 - dropout_rate)

    def _activation(self, x):
        """Áp dụng hàm kích hoạt."""
        return getattr(Activation, self.activation)(x)

    def _activation_derivative(self, x):
        """Đạo hàm của hàm kích hoạt."""
        return getattr(Activation, f"{self.activation}_derivative")(x)

    def mini_batch_generator(self, X, y, batch_size):
        """Tạo mini-batch từ dữ liệu."""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            yield X[start_idx:end_idx], y[start_idx:end_idx]

    def train(self, X_train, y_train):
        """Huấn luyện mạng nơ-ron."""
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        print(f"Training on {X_train.shape[0]} samples...")
        print(f"Number of features: {n_features}")
        print(f"Number of classes: {n_classes}")

        self.initialize_weights(n_features, n_classes)
        y_train_one_hot = np.eye(n_classes)[y_train]
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in self.mini_batch_generator(
                X_train, y_train_one_hot, self.batch_size
            ):
                activations = self.forward(X_batch)
                gradients = self.backward(activations, y_batch)
                self.update_parameters(gradients)
                batch_loss = -np.mean(
                    np.sum(y_batch * np.log(activations[-1] + 1e-8), axis=1)
                )
                epoch_loss += batch_loss
            epoch_loss /= len(X_train) // self.batch_size
            self.log(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

            # Early stopping
            if self.early_stopping:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.log(f"Early stopping at epoch {epoch + 1}")
                        break

        # Khôi phục trọng số tốt nhất (nếu Early Stopping được kích hoạt)
        if self.early_stopping and self.best_weights:
            self.weights = self.best_weights
            self.biases = self.best_biases

    def predict(self, X):
        """Dự đoán nhãn cho dữ liệu đầu vào."""
        probabilities = self.forward(X)[-1]
        return np.argmax(probabilities, axis=1)
