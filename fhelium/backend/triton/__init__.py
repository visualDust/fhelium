"""Triton code generation for component-aware modular arithmetic and transforms.

Backend registries select the generated implementations. Modular arithmetic uses integer Montgomery reduction and preserves native lazy residue ranges.
"""

from ._pointwise import TritonFusionImplementation, TritonRnsImplementation

__all__ = ["TritonFusionImplementation", "TritonRnsImplementation"]

from ._tensor import TritonTensorFusionImplementation

__all__ += ["TritonTensorFusionImplementation"]
