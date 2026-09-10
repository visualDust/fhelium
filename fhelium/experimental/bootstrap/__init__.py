"""CKKS bootstrap components and full-slot bootstrap execution.

The package exposes independently replaceable polynomial, linear-transform,
and modular-reduction mechanisms plus one full-slot callable configured for
one Engine. Preconfigured constructors live in
:mod:`fhelium.experimental.bootstrap.presets`.
"""

from fhelium.experimental.bootstrap.full_slot import FullSlotBootstrap
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.linear import (
    DiagonalBSGSEvaluator,
    DiagonalLinearTransform,
    DirectDiagonalEvaluator,
    Radix2FourierTransformCompiler,
)
from fhelium.experimental.bootstrap.reduction import (
    CosineDoubleAngleReduction,
    ExponentialSquaringReduction,
)
from fhelium.experimental.bootstrap.polynomial import (
    BalancedPowerEvaluator,
    BinaryDecompositionChebyshevEvaluator,
    ChebyshevInterpolator,
    HornerPowerEvaluator,
    PatersonStockmeyerPowerEvaluator,
    PolynomialApproximation,
)

__all__ = [
    'BalancedPowerEvaluator',
    'BootstrapArithmetic',
    'BinaryDecompositionChebyshevEvaluator',
    'ChebyshevInterpolator',
    'CosineDoubleAngleReduction',
    'DiagonalBSGSEvaluator',
    'DiagonalLinearTransform',
    'DirectDiagonalEvaluator',
    'ExponentialSquaringReduction',
    'FullSlotBootstrap',
    'HornerPowerEvaluator',
    'PatersonStockmeyerPowerEvaluator',
    'PolynomialApproximation',
    'Radix2FourierTransformCompiler',
]
