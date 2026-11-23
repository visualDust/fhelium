r"""CRT lifts defined by supplied coefficient and modular-reduction Tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement

import torch

from fhelium.ir.dialects import rns
from fhelium.native.wrapper import rns_ops

if TYPE_CHECKING:
    from fhelium.backend.rns.context import RnsContext


CRT_OPERANDS = (
    "source_parameters",
    "target_parameters",
    "matrix",
    "target_reciprocals",
    "source_normalizers",
    "source_reciprocals",
)


def crt_basis_extension_tables(
    context: RnsContext,
    source_prime_ids: tuple[int, ...],
    target_prime_ids: tuple[int, ...],
    *,
    method: Literal["raw", "normalized"],
) -> dict[str, torch.Tensor]:
    r"""Construct the selected lift's matrix in target Montgomery form.

    Let D=product(q_i), D_i=D/q_i and u_i=D_i^{-1} mod q_i.
    Raw lift evaluates sum(a_i D_i u_i); normalized lift evaluates
    sum((a_i u_i mod q_i) D_i). Both reduce to a_i on source prime rows.
    Target matrix entries include one factor R, so the output is in
    Montgomery form. The two lifts need not choose the canonical integer
    representative, and their evaluation-key error differs.
    """
    if method not in {"raw", "normalized"}:
        raise ValueError("CRT lift method must be 'raw' or 'normalized'")
    source_moduli = [context.config.moduli[i] for i in source_prime_ids]
    target_moduli = [context.config.moduli[i] for i in target_prime_ids]
    matrix, target_reciprocals, source_normalizers, source_reciprocals = (
        rns_ops.crt_basis_extension_tables(
            source_moduli, target_moduli, method == "normalized", context.dtype
        )
    )
    return {
        "source_parameters": context.rns_parameter_tensor.index_select(
            1,
            torch.tensor(
                source_prime_ids, dtype=torch.int64, device=context.device
            ),
        ),
        "target_parameters": context.rns_parameter_tensor.index_select(
            1,
            torch.tensor(
                target_prime_ids, dtype=torch.int64, device=context.device
            ),
        ),
        "matrix": matrix.to(context.device),
        "target_reciprocals": target_reciprocals.to(context.device),
        "source_normalizers": source_normalizers.to(context.device),
        "source_reciprocals": source_reciprocals.to(context.device),
    }


@dataclass(frozen=True)
class NativeCrtBasisExtensionImplementation:
    """Evaluate a raw or source-normalized CRT matrix over live Tensor operands.

    Source rows hold coefficient residues in $[0, 2q_i)$. A nonempty
    ``source_normalizers`` operand applies $a_i u_i \\bmod q_i$ before the
    matrix. The output holds Montgomery residues of the target rows.
    """

    name: str = "native-crt-basis-extension"
    operation_types: tuple[type[Operation], ...] = (rns.CrtBasisExtendOp,)
    supports_in_place: bool = False

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
        del invocation, resources, in_place
        tensors = dict(zip(CRT_OPERANDS, inputs[1:], strict=True))
        normalized = tensors["source_normalizers"].numel() != 0
        return (
            rns_ops.crt_basis_extend(
                inputs[0],
                tensors["source_parameters"],
                tensors["target_parameters"],
                tensors["matrix"],
                tensors["target_reciprocals"],
                tensors["source_normalizers"] if normalized else None,
                tensors["source_reciprocals"] if normalized else None,
            ),
        )
