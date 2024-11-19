import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: The predicted scores/logits. Shape is (N, K), where N is the number of samples
           and K is the number of classes.

    Returns:
        softmax_scores: The matrix containing probability for each sample and each class.
                        Shape is (N, K).
    """
    # Subtract the maximum value in each row for numerical stability
    y_stable = y - np.max(y, axis=1, keepdims=True)

    # Compute the exponential of each value
    exp_y = np.exp(y_stable)

    # Normalize by dividing each row by the sum of exponentials in that row
    softmax_scores = exp_y / np.sum(exp_y, axis=1, keepdims=True)

    return softmax_scores
