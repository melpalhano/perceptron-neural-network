"""
@file compute_cost.py
@brief Computes the cost for linear regression.
"""

import numpy as np

def compute_cost(X, y, theta):
    """
    Compute the cost for linear regression.

    This function calculates the mean squared error cost function J(θ) for linear regression:
    J(θ) = (1 / (2 * m)) * Σ (h(θ) - y)^2

    where:
    - J(θ) is the cost
    - m is the number of training examples
    - h(θ) is the hypothesis function (X @ theta)
    - y is the vector of observed values

    @param X: np.ndarray
        Feature matrix including the intercept term (shape: m x n).
    @param y: np.ndarray
        Target variable vector (shape: m,).
    @param theta: np.ndarray
        Parameter vector for linear regression (shape: n,).

    @return: float
        The computed cost value as a single float.
    """
    # get the number of training examples
    m = len(y)

    # Compute the predictions using the linear model by formula h(θ) = X @ θ
    h_o = np.dot(X, theta)

    # Compute the error vector between predictions and actual values
    errors = h_o - y

    # Compute the cost as the mean squared error cost function
    J_o = (1 / (2 * m)) * np.sum(errors ** 2)

    return J_o
