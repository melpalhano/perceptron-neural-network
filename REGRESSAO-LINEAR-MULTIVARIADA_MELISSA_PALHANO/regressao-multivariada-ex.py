"""
@file regressao-multivariada-ex.py
@brief Multivariate linear regression exercise with gradient descent and normal equation.
@details Este script executa um fluxo de trabalho completo para regressão linear multivariada,
          incluindo normalização de features, cálculo de parâmetros via gradiente descendente
          e equação normal, além de comparação de custos.
@author Melissa Palhano (melissa.palhano@discente.ufma.br)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as pe

from RegressionMultivariate.features_normalize import features_normalize_by_std
from RegressionMultivariate.features_normalize import features_normalizes_by_min_max
from RegressionMultivariate.compute_cost_multi import compute_cost_multi
from RegressionMultivariate.gradient_descent_multi import gradient_descent_multi
from RegressionMultivariate.gradient_descent_multi import gradient_descent_multi_with_history
from RegressionMultivariate.normal_eqn import normal_eqn

# Configuração global do estilo
sns.set_theme(style="darkgrid")

def plot_convergence_comparison(J_history_std, J_history_minmax, num_iters):
    """Plota comparação da convergência com diferentes normalizações"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    iter_range = np.arange(1, num_iters + 1)
    ax.plot(iter_range, J_history_std, 
            color='#2E86C1', linewidth=2.5, 
            label='Normalização Z-score',
            path_effects=[pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()])
    ax.plot(iter_range, J_history_minmax, 
            color='#E74C3C', linewidth=2.5, 
            label='Normalização Min-Max',
            path_effects=[pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()])
    
    ax.set_xlabel('Número de Iterações', fontsize=12, fontweight='bold')
    ax.set_ylabel('Custo J(θ)', fontsize=12, fontweight='bold')
    ax.set_title('Gráfico 01 - Curva de Convergência do Custo do GD', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    min_std = np.min(J_history_std)
    min_minmax = np.min(J_history_minmax)
    
    ax.annotate(f'Custo mínimo (Z-score): {min_std:.2e}',
                xy=(num_iters, min_std),
                xytext=(num_iters-100, min_std*1.5),
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    ax.annotate(f'Custo mínimo (Min-Max): {min_minmax:.2e}',
                xy=(num_iters, min_minmax),
                xytext=(num_iters-100, min_minmax*1.5),
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-.2'))
    
    ax.legend(fontsize=10, loc='upper right',
             bbox_to_anchor=(1.15, 1),
             fancybox=True, shadow=True)
    
    ax.set_yscale('log')
    ax.set_xlim(0, num_iters + 50)

    
    plt.tight_layout()
    plt.savefig('Figures/Grafico_01_Convergencia_Custo_GD.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_gd_vs_ne_comparison(J_history, cost_ne, num_iters, theta_gd, theta_ne):
    """Plota comparação entre GD e Equação Normal"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plotando GD
    iter_range = np.arange(1, num_iters + 1)
    ax.plot(iter_range, J_history,
            color='blue', linewidth=2.5,
            label='Gradiente Descendente')
    
    # Linha do NE correto (linha tracejada vermelha)
    ax.axhline(y=cost_ne, color='red', linestyle='--', linewidth=2,
               label='Equação Normal (correto)')
    
    # Linha do NE com erro simulado (linha pontilhada preta)
    ax.axhline(y=cost_ne * 10, color='black', linestyle=':', linewidth=2,
               label='NE (errado)')
    
    ax.set_xlabel('Iteração', fontsize=12, fontweight='bold')
    ax.set_ylabel('Custo J(θ)', fontsize=12, fontweight='bold')
    ax.set_title('Gráfico 02 - Comparação do Menor Custo: GD vs NE', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Configurações do gráfico
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, loc='upper right')
    
    ax.set_yscale('log')
    ax.set_xlim(0, num_iters)
    
    # Ajustando os limites do eixo y para melhor visualização
    y_min = min(cost_ne * 0.5, min(J_history))
    y_max = max(cost_ne * 15, max(J_history))
    ax.set_ylim(y_min, y_max)
 
    plt.tight_layout()
    plt.savefig('Figures/Grafico_02_Comparacao_GD_vs_NE.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_regression_plane_3d(X, y, theta_gd_orig):
    """Plota o plano de regressão 3D com os pontos de treino"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev=20, azim=45)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], y,
                        c=y, cmap='viridis',
                        marker='o', s=100, alpha=0.6,
                        edgecolors='white',
                        linewidth=0.5)
    
    x_surf = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_surf = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = theta_gd_orig[0] + theta_gd_orig[1] * x_surf + theta_gd_orig[2] * y_surf
    
    surf = ax.plot_surface(x_surf, y_surf, z_surf,
                          cmap='coolwarm',
                          alpha=0.3,
                          linewidth=0,
                          antialiased=True)
    
    ax.set_xlabel('Tamanho (pés²)', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Número de Quartos', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('Preço (US$)', fontsize=12, fontweight='bold', labelpad=15)
    
    plt.title('Gráfico 03 - Plano de Regressão 3D',
              fontsize=14, fontweight='bold', pad=30)
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Preço Real (US$)', fontsize=10, fontweight='bold')
    
    eq_text = f'z = {theta_gd_orig[0]:.2f} + {theta_gd_orig[1]:.2f}x + {theta_gd_orig[2]:.2f}y'
    ax.text2D(0.02, 0.98, eq_text,
              transform=ax.transAxes,
              fontsize=10,
              bbox=dict(facecolor='white', 
                       edgecolor='gray',
                       alpha=0.8))
    
    
    plt.tight_layout()
    plt.savefig('Figures/Grafico_03_Plano_Regressao_3D.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_cost_surface_and_contour(T1, T2, J_mesh, theta_history, theta_ne_norm, X_b, y):
    """Plota superfície de custo e contorno com trajetória"""
    fig = plt.figure(figsize=(20, 8))
    
    # Subplot 1: Superfície 3D
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(T1, T2, J_mesh,
                           cmap='viridis',
                           alpha=0.8,
                           linewidth=0)
    
    # Trajetória do GD em 3D
    t1_hist, t2_hist = theta_history[:, 1], theta_history[:, 2]
    costs = np.array([compute_cost_multi(X_b, y, th) for th in theta_history])
    ax1.plot(t1_hist, t2_hist, costs,
             color='red', linewidth=2.5,
             marker='.', markersize=4,
             label='Trajetória GD',
             path_effects=[pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()])
    
    # Ponto da Equação Normal em 3D
    cost_ne = compute_cost_multi(X_b, y, theta_ne_norm)
    ax1.scatter([theta_ne_norm[1]], [theta_ne_norm[2]], [cost_ne],
                color='yellow', s=200, marker='*',
                edgecolors='black', linewidth=1,
                label='Ponto Ótimo (NE)')
    
    ax1.set_xlabel('θ₁', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('θ₂', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_zlabel('J(θ)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.view_init(elev=30, azim=45)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Subplot 2: Contorno
    ax2 = fig.add_subplot(122)
    levels = np.logspace(np.log10(J_mesh.min()),
                        np.log10(J_mesh.max()),
                        20)
    contour = ax2.contour(T1, T2, J_mesh,
                         levels=levels,
                         cmap='viridis',
                         norm=LogNorm())
    plt.colorbar(contour, ax=ax2, label='J(θ)')
    
    # Trajetória do GD no contorno
    ax2.plot(t1_hist, t2_hist,
             color='red', linewidth=2.5,
             marker='.', markersize=6,
             label='Trajetória GD',
             path_effects=[pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()])
    
    # Ponto da Equação Normal no contorno
    ax2.scatter([theta_ne_norm[1]], [theta_ne_norm[2]],
                color='yellow', s=200, marker='*',
                edgecolors='black', linewidth=1,
                label='Ponto Ótimo (NE)')
    
    ax2.set_xlabel('θ₁', fontsize=12, fontweight='bold')
    ax2.set_ylabel('θ₂', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.suptitle('Gráfico 04 - Superfície e Contorno com Trajetória do GD e NE',
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig('Figures/Grafico_04_Superficie_Contorno_GD_NE.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def costs_from_history(X_b: np.ndarray, y: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """Calcula o custo J(θ) para cada θ em *thetas*."""
    return np.array([compute_cost_multi(X_b, y, th) for th in thetas])

def main():
    """
    Executa o fluxo de trabalho de regressão multivariada.
    1. Carrega os dados de ex1data2.txt.
    2. Normaliza features e adiciona bias.
    3. Roda gradient descent e plota convergência.
    4. Prediz preço com θ encontrado.
    5. Calcula θ via equação normal e prediz.
    6. Compara custo das duas abordagens.
    7. Plota superfície de custo e trajetória do GD.
    8. Plota contorno de custo e trajetória do GD.
    9. Plota plano de regressão ajustado com dados originais.

    Ao fim, escreva um relatório com os resultados obtidos.
    Descreva:
    - As diferentes abordagens utilizadas (GD e NE). Qual foi a vantagem que você encontrou 
    na equação normal? Porque ela é mais rápida? Compare o custo minimo encontrado com as duas abordagens.
    - O que você aprendeu sobre a relação entre o custo e os parâmetros θ. Como o custo muda à medida que os parâmetros são ajustados?
    - O que você aprendeu sobre a normalização de features e como ela afeta o desempenho do GD. Qual a importância da normalização de features?
    ( Faça testes com e sem normalização de features e testando ambos os tipos de normalização. Faça gráficos comparativos explicando eles.)
    - Explique a diferença entre o custo calculado com θ_ne em X_ne (original) e o custo calculado com θ_gd em X_b (normalizado). (Discutir a escala dos dados.)
    - Explique porque a ultima plotagem originou um plano sobre os dados. O que isso significa?

    Obs. O relatório não precisa ser grande, mas precisa ter os gráficos e as explicações. Separe as discuções com subtítulos ou tópicos.

    """
    os.makedirs("Figures", exist_ok=True)

    data = np.loadtxt('Data/ex1data2.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = len(y)

    print('Primeiros 10 exemplos de treinamento:')
    print(np.column_stack((X[:10], y[:10])))
    """
    Resposta esperada:
    Primeiros 10 exemplos de treinamento:
    [[2.10400e+03 3.00000e+00 3.99900e+05]
    [1.60000e+03 3.00000e+00 3.29900e+05]
    [2.40000e+03 3.00000e+00 3.69000e+05]
    [1.41600e+03 2.00000e+00 2.32000e+05]
    [3.00000e+03 4.00000e+00 5.39900e+05]
    [1.98500e+03 4.00000e+00 2.99900e+05]
    [1.53400e+03 3.00000e+00 3.14900e+05]
    [1.42700e+03 3.00000e+00 1.98999e+05]
    [1.38000e+03 3.00000e+00 2.12000e+05]
    [1.49400e+03 3.00000e+00 2.42500e+05]]
    """

    X_norm_std, mu_std, sigma_std = features_normalize_by_std(X)
    X_norm_minmax, min_vals, max_vals = features_normalizes_by_min_max(X)

    X_b_std = np.column_stack((np.ones(m), X_norm_std))
    X_b_minmax = np.column_stack((np.ones(m), X_norm_minmax))
    X_b = X_b_std 

    alpha = 0.01
    num_iters = 400
    theta_init = np.zeros(3)

    theta_gd_std, J_history_std = gradient_descent_multi(X_b_std, y, theta_init, alpha, num_iters)
    theta_gd_minmax, J_history_minmax = gradient_descent_multi(X_b_minmax, y, theta_init, alpha, num_iters)

    X_ne = np.column_stack((np.ones(m), X))
    theta_ne = normal_eqn(X_ne, y)

    cost_ne = compute_cost_multi(X_ne, y, theta_ne)

    plot_convergence_comparison(J_history_std, J_history_minmax, num_iters)
    plot_gd_vs_ne_comparison(J_history_std, cost_ne, num_iters, theta_gd_std, theta_ne)

    theta_gd_orig = np.zeros_like(theta_ne)
    theta_gd_orig[1:] = theta_gd_std[1:] / sigma_std
    theta_gd_orig[0] = theta_gd_std[0] - np.sum((mu_std / sigma_std) * theta_gd_std[1:])
    plot_regression_plane_3d(X, y, theta_gd_orig)

    theta_gd_std_with_history, _, theta_history = gradient_descent_multi_with_history(
        X_b_std, y, theta_init, alpha, num_iters
    )

    t1_hist, t2_hist = theta_history[:, 1], theta_history[:, 2]
    max_dev1 = np.max(np.abs(t1_hist - theta_ne[1]))
    max_dev2 = np.max(np.abs(t2_hist - theta_ne[2]))
    span = 1.5 * max(max_dev1, max_dev2)

    t1_vals = np.linspace(theta_ne[1] - span, theta_ne[1] + span, 100)
    t2_vals = np.linspace(theta_ne[2] - span, theta_ne[2] + span, 100)
    T1, T2 = np.meshgrid(t1_vals, t2_vals)

    J_mesh = np.zeros_like(T1)
    for i in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            theta_test = np.array([theta_ne[0], T1[i,j], T2[i,j]])
            J_mesh[i,j] = compute_cost_multi(X_ne, y, theta_test)

    plot_cost_surface_and_contour(T1, T2, J_mesh, theta_history, theta_ne, X_b, y)

if __name__ == '__main__':
    main()
