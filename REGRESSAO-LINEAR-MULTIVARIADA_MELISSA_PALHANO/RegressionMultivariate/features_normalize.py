# Functions/feature_normalize.py
"""
@file features_normalizes.py
@brief Funções para normalização de features em datasets.
@details Este módulo contém funções para normalizar as features de um dataset
          utilizando diferentes abordagens, como média e desvio padrão, ou
          mínimo e máximo.
@author Melissa Rodrigues Palhano
"""
import numpy as np


def features_normalize_by_std(X):
    """
    Normaliza as features de um dataset para média zero e desvio padrão unitário.
    Matematicamente, a formula utilizada é:
        X_norm = (X - mu) / sigma
    onde:
        - X é a matriz de entrada (m x n) onde m é o número de amostras e n é o número de features.
        - mu é o vetor de médias (1 x n) de cada feature.
        - sigma é o vetor de desvios padrão (1 x n) de cada feature.

    :param (ndarray) X: Matriz de entrada onde cada linha é uma amostra e cada coluna é uma feature.
    :return (tuple): Uma tripla contendo:
        - X_norm (ndarray): Matriz normalizada.
        - mu (ndarray): Vetor com as médias de cada feature.
        - sigma (ndarray): Vetor com os desvios padrão de cada feature.
    """
    # Calcula a média de cada feature (coluna)
    mu = np.mean(X, axis=0)

    # Calcula o desvio padrão de cada feature (coluna)
    sigma = np.std(X, axis=0)
    
    # Normaliza as features subtraindo a média e dividindo pelo desvio padrão
    # Verifica se sigma é zero (o que indicaria que todas as amostras têm o mesmo valor na feature)
    # Se sigma for zero, substitui por 1 para evitar divisão por zero
    # Isso garante que a normalização não cause problemas numéricos
    # e que a feature não seja eliminada do conjunto de dados
    if np.any(sigma == 0):
        sigma[sigma == 0] = 1
    # Normaliza as features
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def features_normalizes_by_min_max(X):
    """
    Normaliza as features de um dataset para o intervalo [0, 1] utilizando o mínimo e o máximo.
    Matematicamente, a formula utilizada é:
        X_norm = (X - min) / (max - min)
    onde:
        - X é a matriz de entrada (m x n) onde m é o número de amostras e n é o número de features.
        - min é o vetor de mínimos (1 x n) de cada feature.
        - max é o vetor de máximos (1 x n) de cada feature.

    :param (ndarray) X: Matriz de entrada onde cada linha é uma amostra e cada coluna é uma feature.
    :return (tuple): Uma tupla contendo:
        - X_norm (ndarray): Matriz normalizada.
        - min (ndarray): Vetor com os valores mínimos de cada feature.
        - max (ndarray): Vetor com os valores máximos de cada feature.
    """
    # Calcula o mínimo de cada feature (coluna)
    min_val = np.min(X, axis=0)
    # Calcula o máximo de cada feature (coluna)
    max_val = np.max(X, axis=0)
    
    # Normaliza as features subtraindo o mínimo e dividindo pela diferença entre máximo e mínimo
    # Verifica se max - min é zero (o que indicaria que todas as amostras têm o mesmo valor na feature)
    # Se max - min for zero, substitui por 1 para evitar divisão por zero
    # Isso garante que a normalização não cause problemas numéricos
    # e que a feature não seja eliminada do conjunto de dados
    range_val = max_val - min_val
    if np.any(range_val == 0):
        range_val[range_val == 0] = 1
    
    # Normaliza as features
    X_norm = (X - min_val) / range_val
    return X_norm, min_val, max_val


def revert_z_score_normalization(X_norm, mu, sigma):
    """
    Reverte a normalização z-score para recuperar os valores originais.
    
    :param (ndarray) X_norm: Matriz normalizada.
    :param (ndarray) mu: Vetor de médias.
    :param (ndarray) sigma: Vetor de desvios padrão.
    :return (ndarray): Matriz com valores originais.
    """
    return X_norm * sigma + mu


def revert_min_max_normalization(X_norm, min_val, max_val):
    """
    Reverte a normalização min-max para recuperar os valores originais.
    
    :param (ndarray) X_norm: Matriz normalizada.
    :param (ndarray) min_val: Vetor de valores mínimos.
    :param (ndarray) max_val: Vetor de valores máximos.
    :return (ndarray): Matriz com valores originais.
    """
    return X_norm * (max_val - min_val) + min_val


if __name__ == "__main__":
    # Exemplo de uso
    print("Testando funções de normalização...")
    
    # Cria dados de exemplo
    X = np.array([
        [100, 2],
        [120, 3],
        [150, 4],
        [80, 2],
        [200, 4]
    ])
    
    print("\nDados originais:")
    print(X)
    
    # Testa normalização z-score
    X_norm_z, mu, sigma = features_normalize_by_std(X)
    print("\nNormalização Z-score:")
    print("Dados normalizados:")
    print(X_norm_z)
    print("Média:", mu)
    print("Desvio padrão:", sigma)
    
    # Reverte normalização z-score
    X_original = revert_z_score_normalization(X_norm_z, mu, sigma)
    print("\nDados originais recuperados (z-score):")
    print(X_original)
    
    # Testa normalização min-max
    X_norm_mm, min_val, max_val = features_normalizes_by_min_max(X)
    print("\nNormalização Min-Max:")
    print("Dados normalizados:")
    print(X_norm_mm)
    print("Mínimos:", min_val)
    print("Máximos:", max_val)
    
    # Reverte normalização min-max
    X_original = revert_min_max_normalization(X_norm_mm, min_val, max_val)
    print("\nDados originais recuperados (min-max):")
    print(X_original)
