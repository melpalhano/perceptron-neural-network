"""
RegressionMultivariate - Pacote para implementação de regressão linear multivariada
"""

__version__ = '1.0.0'
__author__ = 'Melissa Rodrigues Palhano'

from .features_normalize import features_normalize_by_std, features_normalizes_by_min_max
from .compute_cost_multi import compute_cost_multi
from .gradient_descent_multi import gradient_descent_multi, gradient_descent_multi_with_history
from .normal_eqn import normal_eqn

__all__ = [
    'features_normalize_by_std',
    'features_normalizes_by_min_max',
    'compute_cost_multi',
    'gradient_descent_multi',
    'gradient_descent_multi_with_history',
    'normal_eqn'
] 