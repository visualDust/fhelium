"""Polynomial approximations and encrypted evaluation schedules."""

from fhelium.experimental.bootstrap.polynomial.even import (
    EvenChebyshevEvaluator,
)
from fhelium.experimental.bootstrap.polynomial.blocked import (
    BlockedChebyshevEvaluator,
    ChebyshevBlockPlan,
)
from fhelium.experimental.bootstrap.polynomial.approximation import (
    ChebyshevInterpolator,
    PolynomialApproximation,
)
from fhelium.experimental.bootstrap.polynomial.chebyshev import (
    BinaryDecompositionChebyshevEvaluator,
)
from fhelium.experimental.bootstrap.polynomial.power import (
    BalancedPowerEvaluator,
    HornerPowerEvaluator,
    PatersonStockmeyerPowerEvaluator,
)

__all__ = [
    'EvenChebyshevEvaluator',
    'BlockedChebyshevEvaluator',
    'ChebyshevBlockPlan',
    'BalancedPowerEvaluator',
    'BinaryDecompositionChebyshevEvaluator',
    'ChebyshevInterpolator',
    'HornerPowerEvaluator',
    'PatersonStockmeyerPowerEvaluator',
    'PolynomialApproximation',
]
