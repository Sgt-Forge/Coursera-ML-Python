""" Compute the cost and gradeint with regularization
"""
import numpy as np
from typing import List

from sigmoid import sigmoid


def regularized_cost_function(theta, X, y, lambda_):
    """Calculates cost of current theta paremeters
    Args:
        X: input features
        y: labels
        theta: weights for input features

    Returns:
        Cost and Gradient
    """
    grad = np.zeros(theta.shape)
    num_examples = y.size

    # X => (100, 3)
    # theta => (3, 1)
    # (X * theta) => (100 x 3) * (3 x 1) => (100 x 1)
    hypothesis = sigmoid(np.dot(X, theta))
    # y => (100 x 1)
    # y.T => (1 x 100)
    # y.T * log(X * theta) => (1 x 100) x (100 x 1) => (1 x 1) scalar
    '''
    Note: It appears the np.dot will handle converting a matrix to its
    transpose if it detects that the transpose will make the operation
    succeed. Hence, we don't need the tranpose of y, but we take it
    because it's right
    '''
    regularization_value = (lambda_ / (2 * num_examples)) * np.sum(theta ** 2)
    J = (1/num_examples) * np.sum(
                        np.dot(-y, np.log(hypothesis)) -
                        np.dot(1-y, np.log(1 - hypothesis))
                    ) + regularization_value
    regularization_value = (lambda_ / num_examples) * theta
    grad = (1/num_examples) * np.dot((hypothesis - y), X)
    grad += regularization_value
    print(regularization_value)

    return J, grad
