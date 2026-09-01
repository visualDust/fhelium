"""Assign one implementation identity to the registered NTT operations."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

from dataclasses import dataclass

from fhelium.ir import Program
from fhelium.ir.dialects import ntt

from ._assign_implementations import AssignImplementationsPass

_NTT_OPERATION_NAMES = tuple(
    operation_type.name
    for operation_type in (
        ntt.CoefficientStandardToNttMontgomeryOp,
        ntt.CoefficientMontgomeryToNttMontgomeryOp,
        ntt.NttMontgomeryToCoefficientStandardOp,
        ntt.InverseMontgomeryOp,
    )
)


@dataclass(frozen=True)
class AssignNttImplementationPass:
    """Request one named implementation for every matched NTT operation."""

    implementation: str
    overwrite: bool = False
    name: str = "assign-ntt-implementation"

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("NTT implementation name must be nonempty")
        if type(self.overwrite) is not bool:
            raise TypeError("NTT implementation overwrite must be bool")

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Delegate exact NTT matches to the generic assignment pass."""

        return AssignImplementationsPass(
            {name: self.implementation for name in _NTT_OPERATION_NAMES},
            overwrite=self.overwrite,
        ).run(program, workspace)


__all__ = ["AssignNttImplementationPass"]
