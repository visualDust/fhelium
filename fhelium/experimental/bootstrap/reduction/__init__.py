"""Periodic functions that remove approximate CKKS carries."""

from fhelium.experimental.bootstrap.reduction.cosine import (
    CosineDoubleAngleReduction,
)
from fhelium.experimental.bootstrap.reduction.exponential import (
    ExponentialSquaringReduction,
)

__all__ = ['CosineDoubleAngleReduction', 'ExponentialSquaringReduction']
