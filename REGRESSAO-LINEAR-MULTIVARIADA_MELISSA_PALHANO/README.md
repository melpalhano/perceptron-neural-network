# Regressão Linear Multivariada


<img src="./ufma_logo.png" alt="UFMA" width="200"/><br>
<img src="./eng_comp_logo.png" alt="Engenharia da Computação" width="200"/>


## Universidade Federal do Maranhão  
### Engenharia da Computação  
### Disciplina: EECP0053 – **Tópicos em Engenharia da Computação II – Fundamentos de Redes Neurais**  
**Professor:** Dr. Thales Levi Azevedo Valente  
**E-mail:** <thales.levi@ufma.br> / <thales.l.a.valente@gmail.com>  
**Semestre:** 2025.1  

---

## 🎯 Objetivos

Este trabalho individual aprofunda a regressão linear **multivariada** com ênfase em **(i)** o impacto da _normalização das features_ e **(ii)** a comparação entre **Gradiente Descendente (GD)** e **Equação Normal (NE)** para estimação dos parâmetros \( \theta \).  

Objetivos específicos:

1. **Comparar métodos de normalização**  
   - Sem normalização  
   - Normalização **z‑score** (`features_normalize_by_std`)  
   - Normalização **min‑max** (`features_normalizes_by_min_max`)
2. **Comparar métodos de otimização**  
   - Gradiente Descendente clássico  
   - Solução fechada pela Equação Normal
3. **Implementar e documentar** (ou revisar) os componentes essenciais:
   - `RegressionMultivariate/features_normalize.py`
   - `RegressionMultivariate/compute_cost_multi.py`
   - `RegressionMultivariate/gradient_descent_multi.py`
   - `RegressionMultivariate/gradient_descent_multi_with_history.py`
   - `RegressionMultivariate/normal_eqn.py`
   - `regressao-multivariada-ex.py`
4. **Redigir um relatório ABNT** contendo:
   - Descrição dos experimentos e gráficos gerados
   - Discussão crítica dos resultados
   - Explicação do efeito da escala das features sobre GD e NE
   - Conclusões sobre desempenho, velocidade e precisão de cada abordagem

---

## 📚 Tópicos de Implementação & Gráficos

| Item | Conteúdo a gerar/entregar                                                                                                     |
|------|-------------------------------------------------------------------------------------------------------------------------------|
| 1    | **Curva de convergência** de custo do GD (uma linha por variante de normalização)                                             |
| 2    | **Comparação direta** entre menor custo obtido por GD × NE                                                                    |
| 3    | **Plano de regressão 3‑D** (tamanho × quartos × preço) ajustado com θ<sub>GD</sub>, sobre pontos de treino                    |
| 4    | **Superfície** e **contorno** de \( J(\theta_1,\theta_2) \) com trajetória do GD e ponto da NE (θ normalizado)            |

---

## 🗂️ Estrutura do Repositório

```
regressao-linear-multivariada_<SeuNome>/
│
├─ Data/
│   └─ ex1data2.txt
│
├─ RegressionMultivariate/
│   ├─ __init__.py
│   ├─ features_normalize.py
│   ├─ compute_cost_multi.py
│   ├─ gradient_descent_multi.py
│   ├─ gradient_descent_multi_with_history.py
│   └─ normal_eqn.py
│
├─ Figures/                 # gráficos (.png / .svg) produzidos pelo script
│
├─ regressao-multivariada-ex.py            # **script principal**
├─ README.md                # **este arquivo**
├─ ufma_logo.png
├─ eng_comp_logo.png
├─ requirements.txt         # dependências mínimas (numpy, matplotlib)
├─ regressao-multi.yml      # ambiente Conda (opcional)
└─ setup_env.py             # cria venv + instala libs a partir de requirements.txt
```

---


### Reconhecimentos e Direitos Autorais

```
@autor:                Melissa Rodrigues Palhano
@contato:              melissa.palhano@discente.ufma.br
@data última versão:   04/05/2025
@versão:               2.0
@outros repositórios:  -
@Agradecimentos:       Universidade Federal do Maranhão (UFMA),
                       Prof. Dr. Thales Levi Azevedo Valente,
                       colegas de curso.
```

---

### Licença (MIT)

> Este material é resultado de um trabalho acadêmico para a disciplina *EECP0053 - TÓPICOS EM ENGENHARIA DA COMPUTAÇÃO II - FUNDAMENTOS DE REDES NEURAIS*, semestre letivo 2025.1, curso Engenharia da Computação, UFMA.

```
MIT License

Copyright (c) 04/05/2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
