import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred += 1e-7
        n = len(y_pred)
        loss = 0
        
        for i in range(n):
            loss -= (y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i]))

        loss = loss / n

        return round(loss, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        y_pred += 1e-7
        categories = len(y_pred[0])
        n = len(y_pred)
        loss = 0


        for C in range(categories):
            for i in range(n):
                loss -= (y_true[i][C] * np.log(y_pred[i][C]))

        loss = loss / n

        return round(loss, 4) 