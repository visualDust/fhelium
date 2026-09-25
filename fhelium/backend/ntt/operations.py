"""Native implementations of NTT transitions."""

from __future__ import annotations

from collections.abc import Mapping

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
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
    """Execute logical NTT operations through one named schedule executor."""

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

    def prepare_operation(self, operation: Operation):
        from xdsl.dialects.builtin import StringAttr

        attribute = operation.attributes.get("ntt_backend")
        name = (
            attribute.data
            if isinstance(attribute, StringAttr)
            else self.required_backend_name
        )
        if name is None:
            raise ValueError(
                "NTT schedule remains unresolved; run SelectNttImplementationsPass"
            )
        self.select_ntt_schedule(selected=name)
        transition = _TRANSITIONS[type(operation)]
        return _PreparedNativeNtt(
            self.name,
            prepare_transition(transition, name, in_place=False),
            prepare_transition(transition, name, in_place=True),
        )

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        policy_name = str(
            invocation.attributes.get(
                "ntt_backend", self.required_backend_name or ""
            )
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
            execute_transition(
                transition, values, policy_name, in_place=in_place
            ),
        )


_TRANSITIONS = {
    ntt.CoefficientStandardToNttMontgomeryOp: "forward_to_montgomery_",
    ntt.CoefficientMontgomeryToNttMontgomeryOp: "forward_montgomery_",
    ntt.NttMontgomeryToCoefficientStandardOp: "inverse_to_standard_",
    ntt.NttMontgomeryToCoefficientMontgomeryOp: "inverse_montgomery_",
}


@dataclass(frozen=True)
class _PreparedNativeNtt(_TensorOperandImplementation):
    name: str
    functional: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]
    inplace: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]
    supports_in_place: bool = True
    operation_types: tuple[type[Operation], ...] = _NTT_OPERATION_TYPES

    def execute(self, invocation, values, resources, *, in_place):
        return ((self.inplace if in_place else self.functional)(values),)


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
        group = policy.grouped_radix2_stage_count
        native = compact_radix2.prepare_transition(transition)

        def call(values):
            source, parameters, twiddles = values
            output = source.clone() if copy_input else source
            return native(output, twiddles, parameters, group)
    else:
        native = power_of_two_radix.prepare_transition(transition)

        def call(values):
            source, parameters, twiddles, roots = values
            output = source.clone() if copy_input else source
            return native(output, twiddles, roots, parameters)

    return call


def execute_transition(transition, values, policy_name, *, in_place):
    """Execute current Tensor operands through the shared prepared native call."""
    return prepare_transition(transition, policy_name, in_place=in_place)(
        values
    )


__all__ = ["NativeNttImplementation"]


def execute_named_transform(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
    basis: str,
    transition: str,
    *,
    in_place: bool = True,
) -> torch.Tensor:
    """Pass the selected Tensor tables to the shared native NTT dispatch."""
    direction = "inverse" if transition.startswith("inverse") else "forward"
    prefix = f"{basis}_{direction}_"
    policy_name = str(attributes["ntt_backend"])
    count = (
        4
        if policy_name == "radix2_indexed"
        else 2
        if policy_name.startswith("radix2_compact")
        else 3
    )
    return execute_transition(
        transition,
        (source, *(tensors[f"{prefix}{i}"] for i in range(count))),
        policy_name,
        in_place=in_place,
    )
