"""Compose hybrid RNS key switching for CKKS ciphertext components."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import ckks
from fhelium.native.wrapper import ckks_ops, rns_ops, ntt_ops

from fhelium.config.ntt import CompactRadix2Policy, resolve_ntt_backend_policy
from fhelium.backend.rns.modup import modup_digit
from fhelium.backend.rns.crt import CRT_OPERANDS, crt_basis_extend
from fhelium.backend.rns.moddown import (
    moddown_ntt_qp_to_q,
    moddown_qp_to_q,
    moddown_add_rescale,
)
from fhelium.backend.rns.automorphism import coefficient_galois_gather_indices
from fhelium.backend.ntt.operations import execute_named_transform
from ._key_switch_preparation import (
    native_key_switch_requirements,
    prepare_native_key_switch,
)


def lift_key_switch_digit(
    source, tensors, attributes, digit_index, span, *, out=None, scratch=None
):
    r"""Lift one supplied digit a modulo D=product(q_i) into QP Montgomery form.

    Exact lifting uses the representative in [0,D). For D_i=D/q_i and
    u_i=D_i^{-1} mod q_i, raw CRT uses sum_i(a_i D_i u_i); normalized CRT
    uses sum_i((a_i u_i mod q_i) D_i). CRT tables are live Tensor operands.
    Only exact lifting consumes the optional Garner scratch buffer.
    """
    method = attributes.get("modup_method", "exact")
    if method == "exact":
        return modup_digit(
            source, tensors, digit_index, span, out=out, scratch=scratch
        )
    if method not in {"crt", "normalized_crt"}:
        raise ValueError("modup_method must be exact, crt, or normalized_crt")
    start, stop, _ = span
    return crt_basis_extend(
        source[..., start:stop, :],
        {
            name: tensors[f"digit{digit_index}_crt_{name}"]
            for name in CRT_OPERANDS
        },
        out=out,
    )


def accumulate_key_switch_products(
    source: torch.Tensor,
    *,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
    key: torch.Tensor,
    accumulator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Accumulate hybrid digit key products into CT2 NTT/Montgomery QP.

    ``source`` is one coefficient/standard Q polynomial. The optional
    accumulator is owned by the caller and is updated in place. Keys and
    digit tables have the same active QP order for every contribution.
    """
    policy = resolve_ntt_backend_policy(str(attributes["ntt_backend"]))
    for index, span in enumerate(
        cast(tuple[tuple[int, int, int], ...], attributes["digits"])
    ):
        lifted = lift_key_switch_digit(source, tensors, attributes, index, span)
        if accumulator is None:
            accumulator = torch.zeros(
                (2, *lifted.shape), dtype=lifted.dtype, device=lifted.device
            )
        if isinstance(policy, CompactRadix2Policy):
            ntt_ops.forward_ntt_montgomery_compact_keyswitch_accumulate_(
                lifted,
                tensors["qp_forward_1"],
                tensors["qp_parameters"],
                key[span[2]],
                accumulator[0],
                accumulator[1],
                cast(int, attributes["key_row_start"]),
                policy.grouped_radix2_stage_count,
            )
        else:
            execute_named_transform(
                lifted, tensors, attributes, "qp", "forward_montgomery_"
            )
            ckks_ops.keyswitch_accumulate_digit_products_(
                accumulator[0],
                accumulator[1],
                lifted,
                key[span[2]],
                tensors["qp_parameters"],
                cast(int, attributes["key_row_start"]),
            )
        del lifted
    if accumulator is None:
        raise ValueError("Key switching requires at least one RNS digit")
    return accumulator


def key_switch_corrections(
    source: torch.Tensor,
    *,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
    key: torch.Tensor,
    output_ntt: bool = False,
    coefficient_c0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stream hybrid digit products into QP accumulators and remove P."""
    accumulator = accumulate_key_switch_products(
        source, tensors=tensors, attributes=attributes, key=key
    )
    if output_ntt:
        return moddown_ntt_qp_to_q(
            accumulator, tensors, attributes, coefficient_c0=coefficient_c0
        )
    execute_named_transform(
        accumulator, tensors, attributes, "qp", "inverse_to_standard_"
    )
    return moddown_qp_to_q(accumulator, tensors, attributes)


@dataclass(frozen=True)
class NativeKeySwitchImplementation:
    r"""Switch CT2 secret relations with a shared streaming QP accumulator.

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
        source, key = inputs[:2]
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[2:],
                strict=True,
            )
        )
        if source.size(0) != 2:
            raise ValueError("Key switching requires ciphertext components=2")
        if invocation.operation_type is ckks.ConjugateOp:
            indices, signs = coefficient_galois_gather_indices(
                source.size(-1), 2 * source.size(-1) - 1, source.device
            )
            source = ckks_ops.apply_coefficient_galois_automorphism(
                source, indices, signs, tensors["q_parameters"][0]
            )
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        corrections = key_switch_corrections(
            source[1],
            tensors=tensors,
            attributes=invocation.attributes,
            key=key,
            output_ntt=output_ntt,
            coefficient_c0=source[0],
        )
        if output_ntt:
            return (corrections,)
        return (
            torch.stack(
                (
                    rns_ops.add_standard(
                        source[0], corrections[0], tensors["q_parameters"]
                    ),
                    corrections[1],
                )
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
        source, key = inputs[:2]
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                inputs[2:],
                strict=True,
            )
        )
        if source.size(0) != 3:
            raise ValueError("Relinearization requires ciphertext components=3")
        output_ntt = (
            invocation.attributes.get("output_domain", "coefficient") == "ntt"
        )
        component = source[2].clone()
        execute_named_transform(
            component,
            tensors,
            invocation.attributes,
            "q",
            "inverse_to_standard_",
        )
        if bool(invocation.attributes.get("rescale", False)):
            products = accumulate_key_switch_products(
                component,
                tensors=tensors,
                attributes=invocation.attributes,
                key=key,
            )
            return (
                moddown_add_rescale(
                    products, source, tensors, invocation.attributes
                ),
            )
        corrections = key_switch_corrections(
            component,
            tensors=tensors,
            attributes=invocation.attributes,
            key=key,
            output_ntt=output_ntt,
        )
        if output_ntt:
            return (
                rns_ops.add_lazy(
                    source[:2], corrections, tensors["q_parameters"]
                ),
            )
        coefficient = source[:2].clone()
        execute_named_transform(
            coefficient,
            tensors,
            invocation.attributes,
            "q",
            "inverse_to_standard_",
        )
        return (
            rns_ops.add_standard(
                coefficient, corrections, tensors["q_parameters"]
            ),
        )
