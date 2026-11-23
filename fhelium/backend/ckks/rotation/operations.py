"""Independent, hoisted, and weighted CKKS rotation implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend._native_operands import key_switch_options, named_tables
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops

from .._key_switch_preparation import (
    native_key_switch_requirements,
    rotation_sum_requirements,
    prepare_native_key_switch,
)


@dataclass(frozen=True)
class NativeRotateImplementation:
    """Rotate CT2 through one automorphism and streaming hybrid key switch."""

    supports_in_place: bool = False
    name: str = "native-rotate-streaming"
    operation_types: tuple[type[Operation], ...] = (ckks.RotateOp,)

    tensor_requirements = staticmethod(native_key_switch_requirements)
    prepare_operation = prepare_native_key_switch

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        attributes = invocation.attributes
        return (
            ckks_ops.rotate(
                inputs[0],
                inputs[1],
                int(cast(int, attributes["rotation_step"])),
                *named_tables(attributes["parameter_names"], inputs[2:]),
                **key_switch_options(attributes),
            ),
        )


@dataclass(frozen=True)
class NativeRotationSumImplementation:
    """Stream all rotation key products and execute one quotient tail."""

    name: str = "native-rotation-sum-aggregate"
    supports_in_place: bool = False
    operation_types: tuple = (ckks.RotationSumOp,)
    tensor_requirements = staticmethod(rotation_sum_requirements)
    prepare_operation = prepare_native_key_switch

    def resource_requirements(self, invocation):
        return ()

    def execute(self, invocation, inputs, resources, *, in_place):
        del resources, in_place
        attributes = invocation.attributes
        count = int(attributes["source_count"])
        steps = [int(step) for step in attributes["rotation_steps"]]
        key_count = sum(step != 0 for step in steps)
        return (
            ckks_ops.rotation_sum(
                list(inputs[:count]),
                list(inputs[count : count + key_count]),
                steps,
                *named_tables(
                    attributes["parameter_names"], inputs[count + key_count :]
                ),
                **key_switch_options(attributes),
                remaining_q_count=int(attributes["sum_remaining_q_count"]),
            ),
        )


@dataclass(frozen=True)
class NativeHoistedRotateManyImplementation:
    """Execute one scheduled rotation group without changing its membership."""

    supports_in_place: bool = False
    name: str = "native-rotate-many-hoisted"
    operation_types: tuple[type[Operation], ...] = (ckks.RotateManyOp,)

    tensor_requirements = staticmethod(native_key_switch_requirements)
    prepare_operation = prepare_native_key_switch

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        attributes = invocation.attributes
        count = invocation.result_count
        return tuple(
            ckks_ops.rotate_many(
                inputs[0],
                list(inputs[1 : 1 + count]),
                [int(step) for step in attributes["rotation_steps"]],
                *named_tables(
                    attributes["parameter_names"], inputs[1 + count :]
                ),
                **key_switch_options(attributes),
            )
        )


@dataclass(frozen=True)
class NativeGroupedRotationWeightedSumImplementation:
    """Execute one supplied direct-rotation/plaintext group matrix."""

    supports_in_place: bool = False
    name: str = "native-grouped-rotation-weighted-sum"
    operation_types: tuple[type[Operation], ...] = (
        ckks.GroupedRotationWeightedSumOp,
    )
    tensor_requirements = staticmethod(native_key_switch_requirements)
    prepare_operation = prepare_native_key_switch

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        attributes = invocation.attributes
        steps = [int(step) for step in attributes["baby_steps"]]
        term_count = int(cast(int, attributes["term_count"]))
        group_count = int(cast(int, attributes["group_count"]))
        key_count = sum(step != 0 for step in steps)
        end = 1 + key_count + term_count * group_count
        return (
            ckks_ops.grouped_rotation_weighted_sum(
                inputs[0],
                list(inputs[1 : 1 + key_count]),
                steps,
                list(inputs[1 + key_count : end]),
                group_count,
                *named_tables(attributes["parameter_names"], inputs[end:]),
                **key_switch_options(attributes),
            ),
        )
