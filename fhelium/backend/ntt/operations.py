"""Native implementations of NTT transitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.lowering import ExecBuilder
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.config.ntt import (
    CompactRadix2Policy,
    IndexedRadix2Policy,
    resolve_ntt_backend_policy,
)
from fhelium.ir.dialects import ntt

from ._selection import select_policy, table_count, table_layout
from .executors import compact_radix2, indexed_radix2, power_of_two_radix


class _TensorOperandImplementation:
    """Receive numerical data as ordinary Tensor operands."""

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()


_NTT_OPERATION_TYPES = (
    ntt.CoefficientStandardToNttMontgomeryOp,
    ntt.CoefficientMontgomeryToNttMontgomeryOp,
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.NttMontgomeryToCoefficientMontgomeryOp,
)


@dataclass(frozen=True)
class NativeNttImplementation(_TensorOperandImplementation):
    """Lower logical NTT operations to one named native schedule."""

    name: str
    required_backend_name: str | None
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = _NTT_OPERATION_TYPES

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NTT implementation name must be non-empty")
        if (self.required_backend_name is None) != (self.name == "native-ntt"):
            raise ValueError(
                "Only native-ntt may accept any supplied NTT executor"
            )

    def select_ntt_schedule(self, **facts):
        selected = facts.get("selected")
        if self.required_backend_name is not None:
            if selected is not None and selected != self.required_backend_name:
                raise ValueError(
                    "NTT schedule conflicts with the selected implementation"
                )
            facts["selected"] = self.required_backend_name
        policy = select_policy(**facts)
        if policy is None:
            return None
        return {
            "ntt_backend": policy.name,
            "ntt_table_layout": table_layout(policy),
            "table_count": table_count(policy),
        }

    def lower(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources, builder
        policy_name = invocation.attributes.get(
            "ntt_backend", self.required_backend_name
        )
        if policy_name is None:
            raise ValueError(
                "NTT schedule remains unresolved; run SelectNttImplementationsPass"
            )
        if (
            self.required_backend_name is not None
            and policy_name != self.required_backend_name
        ):
            raise ValueError(
                "Supplied NTT schedule differs from the selected implementation"
            )
        transition = _TRANSITIONS[invocation.operation_type]
        return (
            prepare_transition(
                transition, str(policy_name), in_place=invocation.in_place
            )(values),
        )


_TRANSITIONS = {
    ntt.CoefficientStandardToNttMontgomeryOp: "forward_to_montgomery_",
    ntt.CoefficientMontgomeryToNttMontgomeryOp: "forward_montgomery_",
    ntt.NttMontgomeryToCoefficientStandardOp: "inverse_to_standard_",
    ntt.NttMontgomeryToCoefficientMontgomeryOp: "inverse_montgomery_",
}


@lru_cache(maxsize=128)
def prepare_transition(transition: str, policy_name: str, *, in_place: bool):
    """Bind a native algorithm and transition without retaining operand data."""
    policy = resolve_ntt_backend_policy(policy_name)
    copy_input = not in_place
    if copy_input and transition == "forward_to_montgomery_":
        transition = "forward_to_montgomery"
        copy_input = False
    if isinstance(policy, IndexedRadix2Policy):
        native = indexed_radix2.prepare_transition(transition)

        def call(values):
            source, parameters, twiddles, even, odd = values
            output = source.clone() if copy_input else source
            return native(output, even, odd, twiddles, parameters)
    elif isinstance(policy, CompactRadix2Policy):
        native = compact_radix2.prepare_transition(transition)

        def call(values):
            source, parameters, twiddles = values
            output = source.clone() if copy_input else source
            return native(output, twiddles, parameters)
    else:
        native = power_of_two_radix.prepare_transition(transition)

        def call(values):
            source, parameters, twiddles, roots = values
            output = source.clone() if copy_input else source
            return native(output, twiddles, roots, parameters)

    return call


__all__ = ["NativeNttImplementation"]
