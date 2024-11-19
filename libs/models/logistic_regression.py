import numpy as np
from libs.math import sigmoid


class LogisticRegression:
    def __init__(self, num_features: int):
        """
        Initialize the Logistic Regression model with random parameters.

        Args:
            num_features: The number of features in the input data.
        """
        # Initialize parameters (weights) with small random values
        self.parameters = np.random.normal(0, 0.01, num_features)

    def predict(self, x: np.array) -> np.array:
        """
        Method to compute the predictions for the input features.

        Args:
            x: The input data matrix. Shape: (m, num_features),
               where m is the number of samples.

        Returns:
            preds: The predictions for the input features. Shape: (m,).
        """
        # Compute predictions using the sigmoid function
        preds = sigmoid(np.dot(x, self.parameters))
        return preds

    @staticmethod
    def likelihood(preds: np.array, y: np.array) -> float:
        """
        Function to compute the log likelihood of the model parameters
        according to the predictions and true labels.

        Args:
            preds: The predicted probabilities. Shape: (m,).
            y: The true labels. Shape: (m,).

        Returns:
            log_l: The log likelihood value.
        """
        # Compute the log-likelihood using cross-entropy
        log_l = np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
        return log_l

    def update_theta(self, gradient: np.array, lr: float = 0.5):
        """
        Function to update the weights in-place using gradient ascent.

        Args:
            gradient: The gradient of the log likelihood. Shape: (num_features,).
            lr: The learning rate.

        Returns:
            None
        """
        # Update parameters using the gradient ascent rule
        self.parameters += lr * gradient

    @staticmethod
    def compute_gradient(x: np.array, y: np.array, preds: np.array) -> np.array:
        """
        Function to compute the gradient of the log likelihood.

        Args:
            x: The input data matrix. Shape: (m, num_features).
            y: The true labels. Shape: (m,).
            preds: The predicted probabilities. Shape: (m,).

        Returns:
            gradient: The gradient of the log likelihood. Shape: (num_features,).
        """
        # Compute the gradient of the log-likelihood
        gradient = np.dot(x.T, (y - preds)) / x.shape[0]
        return gradient
