r"""Auxiliary-basis removal in coefficient and NTT representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.native.tables import table_slots
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import rns_ops

from ._preparation import native_moddown_requirements, prepare_native_moddown
from fhelium.backend.lowering import ExecBuilder


@dataclass(frozen=True)
class NativeModDownImplementation:
    r"""Divide RNS accumulators by auxiliary and depth-group products.

    ``ModDownQpToQ`` returns $\lfloor (x + h)/P \rfloor \bmod Q$ with
    $h = 0$ or $(P - 1)/2$. ``ModDownNttQpToQ`` returns
    $\widehat{x}_Q P^{-1} + \operatorname{NTT}_Q(-r/P)$ for the coefficient
    remainder $r$ of the P rows, adding the optional coefficient component
    zero before the forward NTT. ``ModDownAddRescale`` returns
    $\lfloor (v + Pa + H_P + PH_D)/(PD) \rfloor$ on the surviving Q rows.
    ``RoundedTrailingQuotient`` divides by the product of the trailing rows
    after the optional row reordering. Depth and CKKS scale are preserved
    except for the joint rescale.
    """

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

    def lower(
        self,
        invocation: OperationInvocation,
        values: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del resources
        attributes = invocation.attributes
        names = cast(tuple[str, ...], attributes["parameter_names"])
        tensors = dict(zip(names, values[len(values) - len(names) :]))
        operation_type = invocation.operation_type
        rounding = str(
            attributes.get(
                "rounding", attributes.get("moddown_rounding", "nearest")
            )
        )
        if operation_type is rns.RoundedTrailingQuotientOp:
            return (
                rns_ops.rounded_trailing_quotient(
                    values[0],
                    table_slots("trailing_quotient", tensors),
                    int(cast(int, attributes["drop_count"])),
                    str(attributes["rounding"]),
                ),
            )
        if operation_type is rns.ModDownAddRescaleOp:
            drop_count = attributes.get("joint_drop_count")
            if drop_count is None:
                drop_count = tensors["joint_drop_positions"].numel()
            return (
                rns_ops.moddown_add_rescale(
                    values[0],
                    values[1],
                    table_slots("key_switch", tensors),
                    str(attributes["ntt_backend"]),
                    int(cast(int, attributes["q_count"])),
                    int(cast(int, drop_count)),
                    bool(attributes.get("joint_drop_is_prefix", False)),
                    str(attributes.get("output_domain", "coefficient")),
                ),
            )
        p_count = int(cast(int, attributes["p_count"]))
        if operation_type is rns.ModDownNttQpToQOp:
            return (
                rns_ops.moddown_ntt_qp_to_q(
                    values[0],
                    values[1] if len(values) - len(names) == 2 else None,
                    table_slots("key_switch", tensors),
                    str(attributes["ntt_backend"]),
                    p_count,
                    rounding,
                ),
            )
        return (
            rns_ops.moddown_qp_to_q(
                values[0], table_slots("key_switch", tensors), p_count, rounding
            ),
        )
