"""Define shared CKKS lowering and its compile-pass adapter.

The package decomposes CKKS operations into logical RNS/NTT operations.
`LowerCkksToRnsNttPass` applies caller-selected decompositions;
`SelectExecutionLoweringsPass` chooses locally between whole-operation coverage
and registered lowerings. Callers can also invoke the lowering library directly.
"""

from ._core import (
    CkksLoweringDefinition,
    CkksLoweringRegistry,
    LoweredCkksOperation,
)
from ._driver import DEFAULT_CKKS_LOWERINGS, lower_ckks_program
from ._ckks_to_rns_ntt import LowerCkksToRnsNttPass
from ._select_execution import SelectExecutionLoweringsPass

__all__ = [
    "CkksLoweringDefinition",
    "CkksLoweringRegistry",
    "DEFAULT_CKKS_LOWERINGS",
    "LowerCkksToRnsNttPass",
    "SelectExecutionLoweringsPass",
    "LoweredCkksOperation",
    "lower_ckks_program",
]
