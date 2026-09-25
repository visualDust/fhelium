"""Independent, hoisted, and weighted CKKS rotation implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops, rns_ops

from fhelium.backend.rns.automorphism import coefficient_galois_gather_indices
from fhelium.backend.ntt.automorphism import ntt_galois_indices
from fhelium.backend.ntt.operations import execute_named_transform
from ..key_switch import key_switch_corrections
from .._key_switch_preparation import (
    native_key_switch_requirements,
    prepare_native_key_switch,
)
from ._hoisted import prepare_digits, apply_rotation
from ._galois import rotation_galois_element


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
        source, key = inputs[:2]
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[2:],
                strict=True,
            )
        )
        if source.size(0) != 2:
            raise ValueError("Rotation requires ciphertext components=2")
        shift = cast(int, invocation.attributes["rotation_step"])
        generator = cast(int, invocation.attributes["galois_generator"])
        input_ntt = (
            invocation.attributes.get("input_domain", "coefficient") == "ntt"
        )
        if input_ntt:
            indices = ntt_galois_indices(
                source.size(-1),
                rotation_galois_element(source.size(-1), shift, generator),
                source.device,
            )
            rotated = ckks_ops.apply_ntt_galois_automorphism(source, indices)
            component = rotated[1].clone()
            execute_named_transform(
                component,
                tensors,
                invocation.attributes,
                "q",
                "inverse_to_standard_",
            )
        else:
            element = rotation_galois_element(source.size(-1), shift, generator)
            indices, signs = coefficient_galois_gather_indices(
                source.size(-1), element, source.device
            )
            rotated = ckks_ops.apply_coefficient_galois_automorphism(
                source, indices, signs, tensors["q_parameters"][0]
            )
            component = rotated[1]
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        corrections = key_switch_corrections(
            component,
            tensors=tensors,
            attributes=invocation.attributes,
            key=key,
            output_ntt=output_ntt,
            coefficient_c0=rotated[0] if output_ntt and not input_ntt else None,
        )
        if output_ntt:
            return (
                torch.stack(
                    (
                        rns_ops.add_lazy(
                            rotated[0], corrections[0], tensors["q_parameters"]
                        ),
                        corrections[1],
                    )
                )
                if input_ntt
                else corrections,
            )
        component0 = rotated[0]
        if input_ntt:
            component0 = component0.clone()
            execute_named_transform(
                component0,
                tensors,
                invocation.attributes,
                "q",
                "inverse_to_standard_",
            )
        return (
            torch.stack(
                (
                    rns_ops.add_standard(
                        component0, corrections[0], tensors["q_parameters"]
                    ),
                    corrections[1],
                )
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
        source = inputs[0]
        count = invocation.result_count
        keys = inputs[1 : 1 + count]
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[1 + count :],
                strict=True,
            )
        )
        steps = cast(tuple[int, ...], invocation.attributes["rotation_steps"])
        prepared = prepare_digits(source[1], tensors, invocation.attributes)
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        c0 = (
            execute_named_transform(
                source[0],
                tensors,
                invocation.attributes,
                "q",
                "forward_to_montgomery_",
                in_place=False,
            )
            if output_ntt
            else source[0]
        )
        return tuple(
            torch.stack(
                apply_rotation(
                    c0,
                    prepared,
                    key,
                    step,
                    tensors,
                    invocation.attributes,
                    output_ntt=output_ntt,
                )
            )
            for key, step in zip(keys, steps, strict=True)
        )


@dataclass(frozen=True)
class NativeGroupedRotationWeightedSumImplementation:
    """Execute one supplied direct-rotation/plaintext group matrix."""

    supports_in_place: bool = False
    name: str = "native-grouped-rotation-weighted-sum"
    operation_types: tuple[type[Operation], ...] = (
        ckks.GroupedRotationWeightedSumOp,
    )

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
        steps = cast(tuple[int, ...], invocation.attributes["baby_steps"])
        term_count = cast(int, invocation.attributes["term_count"])
        group_count = cast(int, invocation.attributes["group_count"])
        key_count = sum(step != 0 for step in steps)
        source = inputs[0]
        keys = iter(inputs[1 : 1 + key_count])
        end = 1 + key_count + term_count * group_count
        plaintexts = inputs[1 + key_count : end]
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[end:],
                strict=True,
            )
        )
        prepared = prepare_digits(source[1], tensors, invocation.attributes)
        source_ntt = execute_named_transform(
            source,
            tensors,
            invocation.attributes,
            "q",
            "forward_to_montgomery_",
            in_place=False,
        )
        babies = [
            source_ntt
            if step == 0
            else torch.stack(
                apply_rotation(
                    source_ntt[0],
                    prepared,
                    next(keys),
                    step,
                    tensors,
                    invocation.attributes,
                    output_ntt=True,
                )
            )
            for step in steps
        ]
        return (
            rns_ops.montgomery_weighted_sums(
                babies, list(plaintexts), group_count, tensors["q_parameters"]
            ),
        )
