# utils/activation.py

import numpy as np


class Activation:
    """Lớp xử lý các hàm kích hoạt và đạo hàm."""

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = Activation.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    def softmax(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)  # Ensure input is numpy array and float
        x = np.clip(x, -1e9, 1e9)  # Avoid overflow/underflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilize softmax
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
