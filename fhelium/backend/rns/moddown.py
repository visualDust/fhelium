r"""Auxiliary-basis removal in coefficient and NTT representations."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import ckks_ops, rns_ops, ntt_ops

from fhelium.config.ntt import CompactRadix2Policy, resolve_ntt_backend_policy
from fhelium.backend.ntt.operations import execute_named_transform
from ._preparation import native_moddown_requirements, prepare_native_moddown


def divide_trailing_rows(
    source: torch.Tensor,
    inverses: torch.Tensor,
    parameters: torch.Tensor,
    drop_count: int,
    *,
    half: torch.Tensor | None = None,
    row_order: torch.Tensor | None = None,
    rounding: str = "nearest",
) -> torch.Tensor:
    r"""Return floor((x+h)/D) after arranging the divisor rows last.

    D is the product of the trailing ``drop_count`` moduli. ``half`` supplies
    (D-1)/2 in the ordered basis for nearest rounding. Input storage is read-only.
    """
    if row_order is not None:
        source = source.index_select(-2, row_order)
    return ckks_ops.keyswitch_moddown_qp_to_q(
        source[..., :-drop_count, :],
        source[..., -drop_count:, :],
        inverses,
        parameters,
        half,
        rounding=rounding,
    )


def moddown_qp_to_q(source, tensors, attributes):
    r"""Return floor((x + h)/P) mod Q, with h=0 or (P-1)/2.

    ``rounding`` selects floor or nearest. Depth and CKKS scale are preserved.
    """
    p_count = cast(int, attributes["p_count"])
    return divide_trailing_rows(
        source,
        tensors["moddown_inverses"],
        tensors["qp_parameters"],
        p_count,
        half=tensors.get("moddown_half"),
        rounding=str(
            attributes.get(
                "rounding", attributes.get("moddown_rounding", "nearest")
            )
        ),
    )


def moddown_ntt_qp_to_q(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
    *,
    coefficient_c0: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Return $\widehat{x}_Q/P+\operatorname{NTT}_Q(-r/P)$.

    ``source`` holds NTT/Montgomery residues over active QP. Inverting only
    the P rows determines the coefficient remainder $r$ used by
    coefficient-domain ModDown. Floor uses $r\in[0,P)$; nearest uses
    $r\in[-(P-1)/2,(P-1)/2]$. Applying that same ModDown to zero Q
    rows produces $-r/P\bmod q_i$. Its forward NTT is added to the retained
    Q evaluations multiplied by $P^{-1}$. The result is a non-aliasing
    active-Q NTT/Montgomery tensor; source storage is unchanged.
    When ``coefficient_c0`` is supplied, it is added to component zero's
    coefficient correction before the forward NTT, adding NTT(c0) without
    another transform. The input c0 remains unchanged.
    """

    p_count = cast(int, attributes["p_count"])
    p_rows = source[..., -p_count:, :].clone()
    execute_named_transform(
        p_rows, tensors, attributes, "p", "inverse_to_standard_"
    )
    q_rows = source[..., :-p_count, :]
    correction = ckks_ops.keyswitch_moddown_qp_to_q(
        torch.zeros_like(q_rows),
        p_rows,
        tensors["moddown_inverses"],
        tensors["qp_parameters"],
        tensors.get("moddown_half"),
        rounding=str(
            attributes.get(
                "rounding", attributes.get("moddown_rounding", "nearest")
            )
        ),
    )
    if coefficient_c0 is not None:
        rns_ops.add_standard_(
            correction[0], coefficient_c0, tensors["q_parameters"]
        )
    policy = resolve_ntt_backend_policy(str(attributes["ntt_backend"]))
    if isinstance(policy, CompactRadix2Policy):
        ntt_ops.forward_ntt_to_montgomery_compact_add_scaled_(
            correction,
            q_rows,
            tensors["p_inverse"],
            tensors["q_forward_1"],
            tensors["q_parameters"],
            policy.grouped_radix2_stage_count,
        )
        return correction
    execute_named_transform(
        correction, tensors, attributes, "q", "forward_to_montgomery_"
    )
    q_rows = q_rows.clone()
    rns_ops.montgomery_mul_row_scalars_(
        q_rows, tensors["p_inverse"], tensors["q_parameters"]
    )
    return rns_ops.add_lazy(q_rows, correction, tensors["q_parameters"])


@dataclass(frozen=True)
class NativeModDownImplementation:
    """Divide RNS accumulators by auxiliary and depth-group products."""

    name: str = "native-key-switch-moddown"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.ModDownQpToQOp,
        rns.ModDownNttQpToQOp,
        rns.ModDownAddRescaleOp,
        rns.RoundedTrailingQuotientOp,
    )

    @staticmethod
    def tensor_requirements(operation, config):
        if isinstance(operation, rns.RoundedTrailingQuotientOp):
            return None
        return native_moddown_requirements(operation, config)

    def prepare_operation(self, operation):
        if isinstance(operation, rns.RoundedTrailingQuotientOp):
            return self
        return prepare_native_moddown(self, operation)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del resources, in_place
        source = values[0]
        names = cast(tuple[str, ...], invocation.attributes["parameter_names"])
        parameter_start = len(values) - len(names)
        tensors = dict(zip(names, values[parameter_start:], strict=True))
        if invocation.operation_type is rns.RoundedTrailingQuotientOp:
            return (
                divide_trailing_rows(
                    source,
                    tensors["moddown_inverses"],
                    tensors["parameters"],
                    cast(int, invocation.attributes["drop_count"]),
                    half=tensors.get("half"),
                    row_order=tensors.get("row_order"),
                    rounding=str(invocation.attributes["rounding"]),
                ),
            )
        if invocation.operation_type is rns.ModDownAddRescaleOp:
            return (
                moddown_add_rescale(
                    values[0], values[1], tensors, invocation.attributes
                ),
            )
        if invocation.operation_type is rns.ModDownNttQpToQOp:
            return (
                moddown_ntt_qp_to_q(
                    source,
                    tensors,
                    invocation.attributes,
                    coefficient_c0=values[1] if parameter_start == 2 else None,
                ),
            )
        return (moddown_qp_to_q(source, tensors, invocation.attributes),)


def moddown_add_rescale(
    products: torch.Tensor,
    base: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    attributes: Mapping[str, object],
) -> torch.Tensor:
    r"""Compute $\lfloor(v+Pa+H_P+PH_D)/(PD)\rfloor$ on surviving Q rows.

    Inputs are NTT/Montgomery bundles. The products' leading component extent selects the matching prefix of base components. Source storage and numerical operands are read-only. Output is coefficient/standard or NTT/Montgomery according to output_domain.
    """
    from fhelium.backend.ntt.operations import execute_named_transform
    from fhelium.config.ntt import (
        CompactRadix2Policy,
        resolve_ntt_backend_policy,
    )

    q_count = cast(int, attributes["q_count"])
    drop_count = cast(
        int,
        attributes.get(
            "joint_drop_count", tensors["joint_drop_positions"].numel()
        ),
    )

    def rows(value, *, dropped):
        if bool(attributes.get("joint_drop_is_prefix", False)):
            return (
                value[..., :drop_count, :]
                if dropped
                else value[..., drop_count:, :]
            )
        return value.index_select(
            -2,
            tensors[
                "joint_drop_positions"
                if dropped
                else "joint_remaining_positions"
            ],
        )

    numerator_q = rns_ops.montgomery_mul_row_scalars_standard(
        base[: products.size(0)],
        tensors["joint_p_times_base"],
        tensors["q_parameters"],
    )
    rns_ops.add_standard_(
        numerator_q, products[..., :q_count, :], tensors["q_parameters"]
    )
    dropped = torch.cat(
        (rows(numerator_q, dropped=True), products[..., q_count:, :]), dim=-2
    )
    execute_named_transform(
        dropped, tensors, attributes, "joint_dropped", "inverse_to_standard_"
    )
    rns_ops.add_standard_(
        dropped,
        tensors["joint_dropped_offset"],
        tensors["joint_dropped_parameters"],
    )
    remaining = rows(numerator_q, dropped=False)
    if attributes.get("output_domain", "coefficient") == "coefficient":
        remaining = remaining.clone()
        execute_named_transform(
            remaining,
            tensors,
            attributes,
            "joint_remaining",
            "inverse_to_standard_",
        )
        rns_ops.add_standard_(
            remaining,
            tensors["joint_remaining_offset"],
            tensors["joint_remaining_parameters"],
        )
        return ckks_ops.keyswitch_moddown_qp_to_q(
            remaining,
            dropped,
            tensors["joint_inverses"],
            tensors["joint_ordered_parameters"],
            None,
            rounding="floor",
        )
    offset = tensors["joint_remaining_offset"].expand(
        *dropped.shape[:-2], -1, -1
    )
    correction = ckks_ops.keyswitch_moddown_qp_to_q(
        offset,
        dropped,
        tensors["joint_inverses"],
        tensors["joint_ordered_parameters"],
        None,
        rounding="floor",
    )
    policy = resolve_ntt_backend_policy(str(attributes["ntt_backend"]))
    if isinstance(policy, CompactRadix2Policy):
        ntt_ops.forward_ntt_to_montgomery_compact_add_scaled_(
            correction,
            remaining,
            tensors["joint_product_inverse"],
            tensors["joint_remaining_forward_1"],
            tensors["joint_remaining_parameters"],
            policy.grouped_radix2_stage_count,
        )
        return correction
    execute_named_transform(
        correction,
        tensors,
        attributes,
        "joint_remaining",
        "forward_to_montgomery_",
    )
    scaled = rns_ops.montgomery_mul_row_scalars_standard(
        remaining,
        tensors["joint_product_inverse"],
        tensors["joint_remaining_parameters"],
    )
    return rns_ops.add_lazy(
        scaled, correction, tensors["joint_remaining_parameters"]
    )
