"""Triton code generation for component-aware modular arithmetic and transforms.

Implementations are selected through Backend registries. Importing this package
neither imports Triton nor changes native defaults. Generated arithmetic uses
integer Montgomery reduction and preserves the native lazy residue ranges.
"""

from ._pointwise import TritonFusionImplementation, TritonRnsImplementation

__all__ = ["TritonFusionImplementation", "TritonRnsImplementation"]

from ._tensor import TritonTensorFusionImplementation

__all__ += ["TritonTensorFusionImplementation"]
