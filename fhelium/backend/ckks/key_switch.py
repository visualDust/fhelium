"""Hybrid RNS key switching for CKKS ciphertext components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.backend._native_operands import key_switch_options, named_tables
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops

from ._key_switch_preparation import (
    native_key_switch_requirements,
    prepare_native_key_switch,
)


@dataclass(frozen=True)
class NativeKeySwitchImplementation:
    r"""Switch CT2 secret relations with one streaming QP accumulator.

    For input phase $c_0+c_1s_{src}$, the key produces corrections
    $(d_0,d_1)$ under $s_{dst}$ and the result is $(c_0+d_0,d_1)$.
    Conjugation first applies $X\mapsto X^{-1}$ to both components.
    Inputs and outputs use coefficient-domain standard Q residues.
    """

    supports_in_place: bool = False
    name: str = "native-key-switch-streaming"
    operation_types: tuple[type[Operation], ...] = (
        ckks.SwitchKeyOp,
        ckks.ConjugateOp,
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
        return (
            ckks_ops.key_switch(
                inputs[0],
                inputs[1],
                *named_tables(attributes["parameter_names"], inputs[2:]),
                **key_switch_options(attributes),
                conjugate=invocation.operation_type is ckks.ConjugateOp,
            ),
        )


@dataclass(frozen=True)
class NativeRelinearizeImplementation:
    """Relinearize CT3 through one streaming hybrid key-switch schedule."""

    supports_in_place: bool = False
    name: str = "native-relinearize-streaming"
    operation_types: tuple[type[Operation], ...] = (ckks.RelinearizeOp,)

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
        rescale = bool(attributes.get("rescale", False))
        names, tables = named_tables(attributes["parameter_names"], inputs[2:])
        drop_count = attributes.get("joint_drop_count")
        if rescale and drop_count is None:
            drop_count = tables[names.index("joint_drop_positions")].numel()
        return (
            ckks_ops.relinearize(
                inputs[0],
                inputs[1],
                names,
                tables,
                **key_switch_options(attributes),
                rescale=rescale,
                q_count=int(attributes.get("q_count", 0)),
                joint_drop_count=int(drop_count or 0),
                joint_drop_is_prefix=bool(
                    attributes.get("joint_drop_is_prefix", False)
                ),
            ),
        )
