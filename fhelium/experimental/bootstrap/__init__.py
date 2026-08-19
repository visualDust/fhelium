"""CKKS bootstrap components and full-slot bootstrap execution.

The package exposes independently replaceable polynomial, linear-transform,
and modular-reduction mechanisms plus one engine-bound callable full-slot
composition. Preconfigured constructors live in
:mod:`fhelium.experimental.bootstrap.presets`.
"""

from fhelium.experimental.bootstrap._full_slot import FullSlotBootstrap
from fhelium.experimental.bootstrap._linear import (
    DiagonalBSGSEvaluator,
    DiagonalLinearTransform,
    DirectDiagonalEvaluator,
    Radix2FourierTransformCompiler,
)
from fhelium.experimental.bootstrap._modular import (
    CosineDoubleAngleReduction,
    ExponentialSquaringReduction,
)
from fhelium.experimental.bootstrap._polynomial import (
    BalancedPowerEvaluator,
    BinaryDecompositionChebyshevEvaluator,
    ChebyshevInterpolator,
    HornerPowerEvaluator,
    PatersonStockmeyerPowerEvaluator,
    PolynomialApproximation,
)

__all__ = [
    'BalancedPowerEvaluator',
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
