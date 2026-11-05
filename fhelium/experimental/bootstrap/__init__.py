"""CKKS bootstrap components and full-slot bootstrap execution.

The package exposes independently replaceable polynomial, linear-transform,
and modular-reduction mechanisms plus one full-slot callable configured for
one Engine. Preconfigured constructors live in
:mod:`fhelium.experimental.bootstrap.presets`.
"""

from fhelium.experimental.bootstrap.polynomial.blocked import (
    BlockedChebyshevEvaluator,
    ChebyshevBlockPlan,
)
from fhelium.experimental.bootstrap.full_slot import FullSlotBootstrap
from fhelium.experimental.bootstrap.structural import (
    CenteredModRaise,
    EncapsulatedModRaise,
    ScaledModRaise,
    ModulusRaising,
)
from fhelium.experimental.bootstrap._core import (
    PreparedBootstrapCore,
    PreparedBootstrap,
)
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.linear import (
    DiagonalBSGSEvaluator,
    DiagonalLinearTransform,
    DirectDiagonalEvaluator,
    Radix2FourierTransformCompiler,
)
from fhelium.experimental.bootstrap.reduction import (
    CosineDoubleAngleReduction,
)
from fhelium.experimental.bootstrap.polynomial import (
    EvenChebyshevEvaluator,
    BalancedPowerEvaluator,
    BinaryDecompositionChebyshevEvaluator,
    ChebyshevInterpolator,
    HornerPowerEvaluator,
    PatersonStockmeyerPowerEvaluator,
    PolynomialApproximation,
)

from fhelium.experimental.bootstrap.presets.cosine import cosine_bootstrap


__all__ = [
    "cosine_bootstrap",
    "CenteredModRaise",
    "EncapsulatedModRaise",
    "ScaledModRaise",
    "ModulusRaising",
    "EvenChebyshevEvaluator",
    'BlockedChebyshevEvaluator',
    'ChebyshevBlockPlan',
    'BalancedPowerEvaluator',
    'BootstrapArithmetic',
    'BinaryDecompositionChebyshevEvaluator',
    'ChebyshevInterpolator',
    'CosineDoubleAngleReduction',
    'DiagonalBSGSEvaluator',
    'DiagonalLinearTransform',
    'DirectDiagonalEvaluator',
    'FullSlotBootstrap',
    'PreparedBootstrapCore',
    'PreparedBootstrap',
    'HornerPowerEvaluator',
    'PatersonStockmeyerPowerEvaluator',
    'PolynomialApproximation',
    'Radix2FourierTransformCompiler',
]
