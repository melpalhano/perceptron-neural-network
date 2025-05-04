# Functions/normal_eqn.py
"""
@file normal_eqn.py
@brief Calcula os parâmetros θ usando a Equação Normal.
@details Este módulo contém uma função para calcular os parâmetros de um modelo
          de regressão linear utilizando a equação normal.
@author Melissa Palhano
"""

import numpy as np


def normal_eqn(X, y):
    """
    Resolve os parâmetros θ utilizando a equação normal.

    A equação normal é definida como:
        θ = (XᵀX)⁻¹ Xᵀ y

    :param (ndarray) X: Matriz de features com bias, onde cada linha é uma amostra
                        e cada coluna é uma feature (shape: m × n+1).
    :param (ndarray) y: Vetor de valores alvo (shape: m,).
    :return (ndarray): Vetor de parâmetros θ (shape: n+1,).
    """
    # Calcula os parâmetros θ utilizando a equação normal
    # A equação normal é uma solução fechada para o problema de regressão linear
    # que minimiza a soma dos erros quadráticos entre as previsões e os valores reais
    # Implemente aqui a equação normal descrita na docstring. Use a função np.linalg.pinv
    # para calcular a pseudo-inversa de uma matriz, que é útil quando a matriz não é quadrada
    # ou não é invertível.
    # A pseudo-inversa é uma generalização da inversa de uma matriz e pode ser usada para resolver
    # sistemas de equações lineares que não têm uma solução única ou que são mal condicionados.

    # Verifica se X é uma matriz numpy
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X e y devem ser arrays numpy")

    # Verifica dimensões
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Número de amostras em X ({X.shape[0]}) não corresponde ao número de alvos em y ({y.shape[0]})")

    if len(y.shape) != 1:
        raise ValueError("y deve ser um vetor unidimensional")

    try:
        # Calcula X^T
        X_transpose = X.T
        
        # Calcula X^T X
        XtX = X_transpose.dot(X)
        
        # Calcula (X^T X)^(-1) usando pseudo-inversa
        # A pseudo-inversa lida com casos onde a matriz não é invertível
        XtX_inv = np.linalg.pinv(XtX)
        
        # Calcula X^T y
        Xty = X_transpose.dot(y)
        
        # Calcula θ = (X^T X)^(-1) X^T y
        theta = XtX_inv.dot(Xty)
        
        return theta
    except np.linalg.LinAlgError as e:
        raise ValueError("Erro ao calcular a equação normal. A matriz X^T X pode ser singular ou mal condicionada.") from e
    except Exception as e:
        raise RuntimeError(f"Erro inesperado ao calcular a equação normal: {str(e)}") from e 
