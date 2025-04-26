"""
@file gradient_descent.py
@brief Implementa o algoritmo de descida do gradiente para regressão linear.
"""

import numpy as np
from Functions.compute_cost import compute_cost

def normalize_features(X):
    """
    Normaliza as features usando z-score normalization.
    
    @param X: np.ndarray
        Matriz de features (m x n)
    @return: tuple(np.ndarray, np.ndarray, np.ndarray)
        X_norm: Matriz normalizada
        mu: Médias das features
        sigma: Desvios padrão das features
    """
    # Não normaliza a primeira coluna (termo de bias)
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    for i in range(1, X.shape[1]):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        if sigma[i] > 0:
            X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]
    
    return X_norm, mu, sigma

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Executa a descida do gradiente para minimizar a função de custo J(θ)
    no contexto de regressão linear.

    @param X: np.ndarray
        Matriz de entrada (m amostras × n atributos), incluindo termo de bias.
    @param y: np.ndarray
        Vetor de saída esperado com dimensão (m,).
    @param theta: np.ndarray
        Vetor de parâmetros inicial (n,).
    @param alpha: float
        Taxa de aprendizado (learning rate).
    @param num_iters: int
        Número de iterações da descida do gradiente.

    @return: tuple[np.ndarray, np.ndarray, np.ndarray]
        theta: vetor otimizado de parâmetros (n,).
        J_history: histórico do valor da função de custo.
        theta_history: histórico dos parâmetros.
    """
    m = len(y)
    
    # Normaliza as features
    X_norm, mu, sigma = normalize_features(X)
    
    # Inicializa históricos
    J_history = np.zeros(num_iters)
    theta_history = np.zeros((num_iters + 1, len(theta)))
    theta_history[0] = theta.copy()
    
    # Copia theta para não modificar o original
    theta = theta.copy()
    
    for i in range(num_iters):
        # Compute predictions
        h = np.dot(X_norm, theta)
        
        # Compute errors
        errors = h - y
        
        # Compute gradient with normalization adjustment
        gradient = np.dot(X_norm.T, errors) / m
        
        # Update parameters with gradient clipping
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1:
            gradient = gradient / grad_norm
            
        # Update theta
        theta = theta - alpha * gradient
        
        # Store cost and parameters
        J_history[i] = compute_cost(X_norm, y, theta)
        theta_history[i + 1] = theta.copy()
        
        # Check for convergence
        if i > 0 and abs(J_history[i] - J_history[i-1]) < 1e-10:
            # Truncate histories to actual iterations
            J_history = J_history[:i+1]
            theta_history = theta_history[:i+2]
            break
    
    return theta, J_history, theta_history