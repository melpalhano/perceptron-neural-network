"""
Script para testar as funções de aquecimento
"""

import numpy as np
from Functions.warm_up_exercises import (
    warm_up_exercise1,
    warm_up_exercise2,
    warm_up_exercise3,
    warm_up_exercise4,
    warm_up_exercise5,
    warm_up_exercise6,
    warm_up_exercise7
)

def test_warm_up_exercises():
    print("Testando exercícios de aquecimento...")
    
    # Teste 1: Matriz identidade 5x5
    print("\n1. Testando matriz identidade 5x5:")
    identity = warm_up_exercise1()
    print(identity)
    
    # Teste 2: Vetor coluna de 1s
    print("\n2. Testando vetor coluna de 1s (m=3):")
    ones = warm_up_exercise2(3)
    print(ones)
    
    # Teste 3: Adição de bias
    print("\n3. Testando adição de bias:")
    x = np.array([1, 2, 3])
    X_with_bias = warm_up_exercise3(x)
    print(X_with_bias)
    
    # Teste 4: Multiplicação matricial
    print("\n4. Testando multiplicação matricial:")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    theta = np.array([0.5, 0.5])
    predictions = warm_up_exercise4(X, theta)
    print(predictions)
    
    # Teste 5: Erros quadráticos
    print("\n5. Testando cálculo de erros quadráticos:")
    y = np.array([2, 4, 6])
    squared_errors = warm_up_exercise5(predictions, y)
    print(squared_errors)
    
    # Teste 6: Custo médio
    print("\n6. Testando cálculo do custo médio:")
    mean_cost = warm_up_exercise6(squared_errors)
    print(mean_cost)
    
    # Teste 7: Função de custo completa
    print("\n7. Testando função de custo completa:")
    final_cost = warm_up_exercise7(X, y, theta)
    print(final_cost)

if __name__ == "__main__":
    test_warm_up_exercises() 