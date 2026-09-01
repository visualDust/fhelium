"""Define shared CKKS lowering and its compile-pass adapter.

The package decomposes CKKS operations into logical RNS/NTT operations without
selecting backend implementations. Eager and JIT callers use the lowering
library directly, while compile pipelines use `LowerCkksToRnsNttPass`.
"""

from ._core import (
    CkksLoweringDefinition,
    CkksLoweringRegistry,
    LoweredCkksOperation,
)
from ._driver import DEFAULT_CKKS_LOWERINGS, lower_ckks_program
from ._ckks_to_rns_ntt import LowerCkksToRnsNttPass

__all__ = [
    "CkksLoweringDefinition",
    "CkksLoweringRegistry",
    "DEFAULT_CKKS_LOWERINGS",
    "LowerCkksToRnsNttPass",
    "LoweredCkksOperation",
    "lower_ckks_program",
]
