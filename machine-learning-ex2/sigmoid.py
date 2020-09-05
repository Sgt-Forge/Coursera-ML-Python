""" Implement sigmoid function.
"""
import numpy as np
from typing import List


def sigmoid(z: List[object]) -> List[object]:
    """Returns sigmoid of matrix or vector

    Args:
        Matrix or vector of values

    Returns:
        The values of the input matrix or vector with the sigmoid function
        applied to each value
    """

    z = np.array(z)
    sigmoid_vals = np.zeros(z.shape)

    sigmoid_vals = 1 / (1 + np.exp(-z))

    return sigmoid_vals
