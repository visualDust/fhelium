"""Native implementations of NTT transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ntt

from fhelium.backend.rns._operand_state import _operand_basis
from fhelium.backend.ntt.context import NttContext


class _OperandResourceImplementation:
    """Declare that resources arrive as operation operands in IR order."""

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()


_NTT_OPERATION_TYPES = (
    ntt.CoefficientStandardToNttMontgomeryOp,
    ntt.CoefficientMontgomeryToNttMontgomeryOp,
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.InverseMontgomeryOp,
)


@dataclass(frozen=True)
class NativeNttImplementation(_OperandResourceImplementation):
    """Execute logical NTT operations through one named schedule executor."""

    name: str
    required_backend_name: str | None
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = _NTT_OPERATION_TYPES

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NTT implementation name must be non-empty")
        if (self.required_backend_name is None) != (
            self.name == "supplied-ntt-plan"
        ):
            raise ValueError(
                "Only supplied-ntt-plan may accept any supplied NTT executor"
            )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        source = values[0]
        resource = cast(NttContext, resources[0].value)
        if (
            self.required_backend_name is not None
            and resource.ntt_backend_name != self.required_backend_name
        ):
            raise ValueError(
                f"NTT resource backend {resource.ntt_backend_name!r} differs "
                f"from implementation requirement "
                f"{self.required_backend_name!r}"
            )
        include_p = _operand_basis(invocation) == "QP"
        row_start = resource.rns_context.parameter_row_start_for(
            source,
            include_p=include_p,
        )
        if (
            invocation.operation_type
            is ntt.CoefficientStandardToNttMontgomeryOp
            and not in_place
        ):
            return (
                resource.forward_to_montgomery(
                    source,
                    parameter_row_start=row_start,
                ),
            )
        output_data = source if in_place else source.clone()
        if (
            invocation.operation_type
            is ntt.CoefficientStandardToNttMontgomeryOp
        ):
            resource.forward_to_montgomery_(
                output_data,
                parameter_row_start=row_start,
            )
        elif (
            invocation.operation_type
            is ntt.CoefficientMontgomeryToNttMontgomeryOp
        ):
            resource.forward_montgomery_(
                output_data,
                parameter_row_start=row_start,
            )
        elif (
            invocation.operation_type
            is ntt.NttMontgomeryToCoefficientStandardOp
        ):
            resource.inverse_to_standard_(
                output_data,
                parameter_row_start=row_start,
            )
        else:
            resource.inverse_montgomery_(
                output_data,
                parameter_row_start=row_start,
            )
        return (output_data,)


__all__ = [
    "NativeNttImplementation",
]
