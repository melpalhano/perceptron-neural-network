# Functions/compute_cost_multi.py
"""
@file compute_cost_multi.py
@brief Computes the cost for multivariate linear regression.
@details Este módulo contém uma função para calcular o custo de um modelo de regressão linear
          multivariada utilizando a função de custo de erro quadrático médio.
@author Melissa Rodrigues Palhano
"""

import numpy as np


def compute_cost_multi(X, y, theta):
    """
    Calcula o custo para regressão linear multivariada.

    A função de custo é definida como:
        J(θ) = (1 / (2m)) * (Xθ - y)ᵀ (Xθ - y)

    :param (ndarray) X: Matriz de features incluindo o termo de intercepto (shape: m × n+1).
    :param (ndarray) y: Vetor de valores alvo (shape: m,).
    :param (ndarray) theta: Vetor de parâmetros (shape: n+1,).
    :return (float): Valor do custo calculado.
    """
    # Verifica as dimensões das entradas
    if X.shape[0] != y.shape[0]:
        raise ValueError("Número de exemplos em X e y deve ser igual")
    if X.shape[1] != theta.shape[0]:
        raise ValueError("Número de features em X deve ser igual ao número de parâmetros em theta")
    
    # get the number of training examples
    m = X.shape[0]
    
    # compute the predictions using the linear model by formula h(θ) = X @ θ
    # where @ is the matrix multiplication operator
    predictions = X @ theta
    # compute the error vector between predictions and actual values
    # The error is the difference between the predicted values and the actual values
    # errors = predictions - y

    errors = predictions - y
    
    # compute the cost as the mean squared error cost function using the formula in the docstring
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

if __name__ == "__main__":
    print("Testando função de custo...")
    # Cria dados de exemplo
    m = 5  # número de exemplos
    n = 2  # número de features
    
    # Gera matriz X com termo de intercepto (coluna de 1s)
    X = np.hstack([np.ones((m, 1)), np.random.randn(m, n)])
    y = np.random.randn(m)  # Gera vetor y de valores alvo
    theta = np.random.randn(n + 1) # Gera vetor theta de parâmetros
    
    try:
        cost = compute_cost_multi(X, y, theta)
        print("\nDados de entrada:")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("theta shape:", theta.shape)
        print("\nCusto calculado:", cost)
    except ValueError as e:
        print("Erro:", str(e))
