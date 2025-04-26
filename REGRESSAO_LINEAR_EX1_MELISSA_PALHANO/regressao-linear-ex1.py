"""
@file regressao-linear-ex1.py
@brief Exercise 2 - Linear Regression implementation with visualization.

This script performs the following tasks:
1. Runs a warm-up exercise.
2. Loads and plots training data.
3. Implements cost function and gradient descent.
4. Predicts values for new inputs.
5. Visualizes the cost function surface and contour.

@author Teacher Thales Levi Azevedo Valente
@subject Foundations of Neural Networks
@course Computer Engineering
@university Federal University of Maranhão
@date 2025
"""


import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import os

from Functions.warm_up_exercises import warm_up_exercise1, warm_up_exercise2, warm_up_exercise3, warm_up_exercise4
from Functions.warm_up_exercises import warm_up_exercise5, warm_up_exercise6, warm_up_exercise7
from Functions.plot_data import plot_data
from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent


def main():
    """
    @brief Executa todos os passos do exercício de regressão linear.

    Esta função serve como ponto de partida para o exercício completo de regressão linear.
    Ela executa uma série de etapas fundamentais, utilizadas como base para o aprendizado
    de modelos supervisionados em redes neurais.

    As principais etapas executadas são:
      1. Executa o exercício de aquecimento (warm-up), imprimindo uma matriz identidade 5x5.
      2. Carrega e plota os dados de treinamento de uma regressão linear simples.
      3. Calcula o custo com diferentes valores de theta usando a função de custo J(θ).
      4. Executa o algoritmo de descida do gradiente para minimizar a função de custo.
      5. Plota a linha de regressão ajustada sobre os dados originais.
      6. Realiza previsões para valores populacionais de 35.000 e 70.000.
      7. Visualiza a função de custo J(θ₀, θ₁) em gráfico de superfície 3D e gráfico de contorno.

    @instructions
    - Os alunos devem garantir que todas as funções auxiliares estejam implementadas corretamente:
        * warm_up_exercise()
        * plot_data()
        * compute_cost()
        * gradient_descent()
    - Todas as funções devem seguir padrão PEP8 e possuir docstrings no formato Doxygen.
    - O script deve ser executado a partir do `main()`.

    @note
    O dataset de entrada `ex1data1.txt` deve estar no mesmo diretório Data.
    A estrutura esperada dos dados é: [population, profit].

    @return None
    """

    # Garante que a pasta de figuras existe
    os.makedirs("Figures", exist_ok=True)

    print('Executando o exercício de aquecimento (warm_up_exercise)...')
    print('Matriz identidade 5x5:')
    # Executa a função de aquecimento
    # Essa função deve retornar uma matriz identidade 5x5
    # representada como um array do NumPy.
    # A função está definida em Functions/warm_up_exercise.py
    # e foi importada no início deste arquivo.
    print('Executando os exercícios de aquecimento...')

    # Exercício 1: Matriz identidade 5x5
    print('\nExercício 1: Matriz identidade 5x5') 
    print(warm_up_exercise1()) # Esperado: [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

    # Exercício 2: Vetor coluna de 1s
    print('\nExercício 2: Vetor de 1s (m=3)')
    print(warm_up_exercise2(3)) # Esperado: [[1], [1], [1]]

    # Exercício 3: Adiciona bias ao vetor x
    print('\nExercício 3: Adiciona coluna de 1s ao vetor [2, 4, 6]')
    x_ex3 = np.array([2, 4, 6])
    print(warm_up_exercise3(x_ex3)) # Esperado: [[1, 2], [1, 4], [1, 6]]

    # Exercício 4: Produto matricial X @ theta
    print('\nExercício 4: Produto X @ theta')
    X_ex4 = warm_up_exercise3(x_ex3)
    theta_ex4 = np.array([1, 2])
    print(warm_up_exercise4(X_ex4, theta_ex4))  # Esperado: [5, 9, 13]

    # Exercício 5: Erros quadráticos
    print('\nExercício 5: Erros quadráticos entre predições e y')
    preds = warm_up_exercise4(X_ex4, theta_ex4)
    y_ex5 = np.array([5, 9, 13])
    print(warm_up_exercise5(preds, y_ex5))  # Esperado: [0, 0, 0]

    # Exercício 6: Custo médio a partir dos erros
    print('\nExercício 6: Custo médio')
    errors_ex6 = warm_up_exercise5(preds, y_ex5)
    print(warm_up_exercise6(errors_ex6))  # Esperado: 0.0

    # Exercício 7: Custo médio com base em X, y e theta
    print('\nExercício 7: Cálculo do custo médio completo')
    print(warm_up_exercise7(X_ex4, y_ex5, theta_ex4))  # Esperado: 0.0


    input("Programa pausado. Pressione Enter para continuar...")

    print('Plotando os dados...')
    # Carrega os dados de treinamento
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
    x = data[:, 0]  # População
    y = data[:, 1]  # Lucro
    m = len(y)  # número de exemplos

    # Plotagem dos dados iniciais
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'rx', markersize=10, label='Dados de Treino')
    plt.xlabel('População da Cidade em 10,000s')
    plt.ylabel('Lucro em $10,000s')
    plt.title('Dados de Treinamento')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/dados_treino.png')
    plt.savefig('Figures/dados_treino.svg')
    plt.show()

    # Preparação dos dados
    x_aug = np.column_stack((np.ones(m), x))
    
    # Parâmetros iniciais
    theta = np.zeros(2)
    iterations = 1500
    alpha = 0.01

    print('\nTestando a função de custo...')
    cost = compute_cost(x_aug, y, theta)
    print(f'Com theta = [0 ; 0]\nCusto calculado = {cost:.2f}')

    # Executa a descida do gradiente
    print('\nExecutando a descida do gradiente...')
    theta, J_history, theta_history = gradient_descent(x_aug, y, theta, alpha, iterations)
    print('Parâmetros theta encontrados pela descida do gradiente:')
    print(f'theta = [{theta[0]:.4f}, {theta[1]:.4f}]')

    # Plot da convergência do custo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(J_history) + 1), J_history, 'b-')
    plt.xlabel('Número de Iterações')
    plt.ylabel('Custo J(θ)')
    plt.title('Convergência da Descida do Gradiente')
    plt.grid(True)
    plt.savefig('Figures/convergencia_basica.png')
    plt.savefig('Figures/convergencia_basica.svg')
    plt.show()

    # Plot da linha de regressão
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'rx', markersize=10, label='Dados de Treino')
    plt.plot(x, x_aug @ theta, 'b-', label='Regressão Linear')
    plt.xlabel('População da Cidade em 10,000s')
    plt.ylabel('Lucro em $10,000s')
    plt.title('Regressão Linear Ajustada')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/regressao_ajustada.png')
    plt.savefig('Figures/regressao_ajustada.svg')
    plt.show()

    # Experimentos com diferentes taxas de aprendizado (α)
    print('\nExecutando experimentos com diferentes taxas de aprendizado...')
    alphas = [0.003, 0.01, 0.03]  # Valores mais conservadores
    thetas_init = np.zeros(2)
    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        theta, J_history, _ = gradient_descent(x_aug, y, thetas_init.copy(), alpha, iterations)
        plt.plot(range(1, len(J_history) + 1), J_history, label=f'α = {alpha}')
    
    plt.xlabel('Número de Iterações')
    plt.ylabel('Custo J(θ)')
    plt.title('Convergência para Diferentes Taxas de Aprendizado')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/convergencia_alphas.png')
    plt.savefig('Figures/convergencia_alphas.svg')
    plt.show()

    # Experimentos com diferentes inicializações de theta
    print('\nExecutando experimentos com diferentes inicializações de theta...')
    alpha_fixed = 0.01
    thetas_init = [
        np.array([0., 0.]),
        np.array([1., 1.]),
        np.array([-1., 1.]),
        np.random.randn(2) * 0.5,  # Reduzindo a escala das inicializações aleatórias
        np.random.randn(2) * 0.5,
        np.random.randn(2) * 0.5
    ]

    # Configuração dos subplots
    plt.figure(figsize=(15, 6))
    
    # Plot das diferentes retas de regressão
    plt.subplot(121)
    plt.plot(x, y, 'rx', markersize=10, label='Dados de Treino')
    
    for i, theta_init in enumerate(thetas_init[:3]):
        theta, _, _ = gradient_descent(x_aug, y, theta_init.copy(), alpha_fixed, iterations)
        plt.plot(x, x_aug @ theta, label=f'θ inicial = [{theta_init[0]:.1f}, {theta_init[1]:.1f}]')
    
    plt.xlabel('População da Cidade em 10,000s')
    plt.ylabel('Lucro em $10,000s')
    plt.title('Regressão Linear com Diferentes Inicializações')
    plt.legend()
    plt.grid(True)

    # Gráfico de contorno com trajetórias
    plt.subplot(122)
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            J_vals[i, j] = compute_cost(x_aug, y, np.array([t0, t1]))
    
    J_vals = J_vals.T
    plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, theta_init in enumerate(thetas_init):
        _, _, theta_history = gradient_descent(x_aug, y, theta_init.copy(), alpha_fixed, iterations)
        plt.plot(theta_history[:, 0], theta_history[:, 1], 
                f'{colors[i]}.-', label=f'Trajetória {i+1}')
    
    plt.xlabel('θ₀')
    plt.ylabel('θ₁')
    plt.title('Contorno da Função Custo com Trajetórias')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Figures/experimentos_theta.png')
    plt.savefig('Figures/experimentos_theta.svg')
    plt.show()

    # Gráfico 3D da superfície de custo
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    surf = ax.plot_surface(theta0_mesh, theta1_mesh, J_vals, 
                          cmap='viridis', alpha=0.6)
    
    for i, theta_init in enumerate(thetas_init[:3]):
        _, J_hist, theta_history = gradient_descent(x_aug, y, theta_init.copy(), alpha_fixed, iterations)
        costs = np.array([compute_cost(x_aug, y, th) for th in theta_history])
        ax.plot3D(theta_history[:, 0], theta_history[:, 1], costs, 
                 f'{colors[i]}.-', label=f'Trajetória {i+1}')
    
    ax.set_xlabel('θ₀')
    ax.set_ylabel('θ₁')
    ax.set_zlabel('J(θ)')
    ax.set_title('Superfície da Função Custo com Trajetórias')
    plt.colorbar(surf)
    ax.legend()
    
    plt.savefig('Figures/superficie_3d.png')
    plt.savefig('Figures/superficie_3d.svg')
    plt.show()


if __name__ == '__main__':
    main()
