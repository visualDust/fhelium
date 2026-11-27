"""Triton code generation for component-aware modular arithmetic and transforms.

Backend registries select the generated implementations. Modular arithmetic uses integer Montgomery reduction and preserves native lazy residue ranges.

Generated kernels take data strides, prime-row counts and lengths as non-specialized int64 arguments, allowing compatible CKKS depths to share compiled kernels. ``warm_executable`` compiles the kernels of a linked executable ahead of its first call.
"""

from ._pointwise import TritonFusionImplementation
from ._tensor import TritonTensorFusionImplementation
from ._warm import warm_executable

__all__ = [
    "TritonFusionImplementation",
    "TritonTensorFusionImplementation",
    "warm_executable",
]
