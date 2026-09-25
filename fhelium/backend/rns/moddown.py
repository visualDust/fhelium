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


def moddown_qp_to_q(source, tensors, attributes):
    """Remove P from a coefficient-domain QP polynomial or component bundle."""
    p_count = cast(int, attributes["p_count"])
    return ckks_ops.keyswitch_moddown_qp_to_q(
        source[..., :-p_count, :],
        source[..., -p_count:, :],
        tensors["moddown_inverses"],
        tensors["qp_parameters"],
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
    the P rows determines the coefficient representative $r\in[0,P)$ used
    by coefficient-domain ModDown. Applying that same ModDown to zero Q
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
    """Divide QP accumulators by P in coefficient or NTT representation."""

    name: str = "native-key-switch-moddown"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        rns.ModDownQpToQOp,
        rns.ModDownNttQpToQOp,
    )

    tensor_requirements = staticmethod(native_moddown_requirements)
    prepare_operation = prepare_native_moddown

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
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                values[1:],
                strict=True,
            )
        )
        if invocation.operation_type is rns.ModDownNttQpToQOp:
            return (
                moddown_ntt_qp_to_q(source, tensors, invocation.attributes),
            )
        return (moddown_qp_to_q(source, tensors, invocation.attributes),)
