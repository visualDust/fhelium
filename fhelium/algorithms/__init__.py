r"""Homomorphic polynomial evaluation and slot-wise linear transforms.

The package provides mathematical objects (:class:`Polynomial`,
:class:`LinearTransform`), frozen evaluation plans that report consumed depth,
products and rotation steps, and functions that evaluate these objects on a
ciphertext with an explicit :class:`~fhelium.eager.Engine`. Every function is
composed of ordinary Eager operations, so ``fhelium.compile`` captures it.
Plan objects implement ``evaluate(arithmetic, ...)``, which receives an
:class:`Arithmetic` carrying the per-depth target scales; a custom plan
following the same protocol can replace a built-in one.
"""

from fhelium.algorithms import linear, polynomial
from fhelium.algorithms._arithmetic import Arithmetic
from fhelium.algorithms.linear import LinearTransform
from fhelium.algorithms.polynomial import Polynomial, chebyshev_fit

__all__ = [
    "Arithmetic",
    "LinearTransform",
    "Polynomial",
    "chebyshev_fit",
    "linear",
    "polynomial",
]
