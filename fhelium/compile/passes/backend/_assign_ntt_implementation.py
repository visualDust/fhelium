"""Assign caller constraints to NTT algorithms and execution implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass
from typing import cast

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Operation

from fhelium.config.ntt import resolve_ntt_backend_policy
from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks, ntt

from ..._pipeline import DecisionRecord, PassResult, PassStats
from .._operation_transforms import program_operations

_NTT_TYPES = tuple(
    cast(type[Operation], spec.operation_type) for spec in ntt.OPERATION_SPECS
)
_TRANSFORM_TYPES = (*_NTT_TYPES, ckks.ToNttOp, ckks.FromNttOp)
_COMPOUND_TYPES = (
    ckks.RescaleOp,
    ckks.RelinearizeOp,
    ckks.SwitchKeyOp,
    ckks.ConjugateOp,
    ckks.RotateOp,
    ckks.RotateManyOp,
    ckks.GroupedRotationWeightedSumOp,
    ckks.PrepareMultiplyMessageOp,
)


@dataclass(frozen=True)
class AssignNttImplementationPass:
    """Assign a complete schedule or partial algorithm constraints.

    Algorithm constraints apply to logical NTT operations and CKKS domain
    transitions. A concrete ``implementation`` applies to logical NTT operations;
    place this pass after CKKS lowering when assigning that implementation.
    Unspecified fields remain available to subsequent selection passes.
    """

    implementation: str | None = None
    overwrite: bool = False
    ntt_backend: str | None = None
    algorithm: str | None = None
    group_width: int | None = None
    radix: int | None = None
    name: str = "assign-ntt-implementation"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        workspace = compilation.workspace
        attributes = {}
        if self.ntt_backend is not None:
            attributes["ntt_backend"] = StringAttr(
                resolve_ntt_backend_policy(self.ntt_backend).name
            )
        if self.algorithm is not None:
            attributes["ntt_algorithm"] = StringAttr(self.algorithm)
        if self.group_width is not None:
            attributes["ntt_group_width"] = IntegerAttr(self.group_width, 64)
        if self.radix is not None:
            attributes["ntt_radix"] = IntegerAttr(self.radix, 64)
        matched = transformed = skipped = 0
        decisions = []
        for operation in program_operations(program):
            if isinstance(operation, _TRANSFORM_TYPES):
                selected = dict(attributes)
            elif (
                isinstance(operation, _COMPOUND_TYPES)
                and "ntt_backend" in attributes
            ):
                selected = {"ntt_backend": attributes["ntt_backend"]}
            else:
                continue
            if (
                isinstance(operation, _NTT_TYPES)
                and self.implementation is not None
            ):
                selected[EXECUTION_IMPLEMENTATION_ATTRIBUTE] = StringAttr(
                    self.implementation
                )
            if not selected:
                continue
            matched += 1
            changed = False
            for key, value in selected.items():
                previous = operation.attributes.get(key)
                if previous == value:
                    continue
                if previous is not None and not self.overwrite:
                    raise ValueError(
                        f"{operation.name} already has {key}; use overwrite=True to replace it"
                    )
                operation.attributes[key] = value
                changed = True
            transformed += int(changed)
            skipped += int(not changed)
            decisions.append(
                DecisionRecord(
                    operation.name,
                    self.implementation or self.ntt_backend or self.algorithm,
                    details=tuple(selected),
                )
            )
        return PassResult(
            program,
            PassStats(
                matched=matched, transformed=transformed, skipped=skipped
            ),
            decisions=tuple(decisions),
        )


__all__ = ["AssignNttImplementationPass"]
