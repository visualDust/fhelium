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
    "target_offsets",
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
    sum(c_i D_i) with the centered residue c_i=[a_i u_i]_{q_i} in
    [-(q_i-1)/2, (q_i-1)/2]. Both reduce to a_i on source prime rows. The
    normalized lift is odd in a, so its excess multiple of D has zero mean for
    sign-symmetric inputs.
    Target matrix entries include one factor R, so the output is in
    Montgomery form. The two lifts need not choose the canonical integer
    representative, and their evaluation-key error differs.
    """
    if method not in {"raw", "normalized"}:
        raise ValueError("CRT lift method must be 'raw' or 'normalized'")
    source_moduli = [context.config.moduli[i] for i in source_prime_ids]
    target_moduli = [context.config.moduli[i] for i in target_prime_ids]
    (
        matrix,
        target_reciprocals,
        source_normalizers,
        source_reciprocals,
        target_offsets,
    ) = rns_ops.crt_basis_extension_tables(
        source_moduli, target_moduli, method == "normalized", context.dtype
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
        "target_offsets": target_offsets.to(context.device),
    }


def centered_quotient_tables(
    context: RnsContext,
    source_prime_ids: tuple[int, ...],
    target_prime_ids: tuple[int, ...],
    prefix: str,
    divisor_prime_ids: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    r"""Construct a centered CRT lift that returns the negated quotient correction.

    With source product $D$, centered $c_i=[a_iu_i]_{q_i}$, an additional
    divisor $E$ formed by ``divisor_prime_ids`` and matrix entries
    $-q_i^{-1}E^{-1}\bmod p_t$ in standard form, the CRT lift evaluates
    $-\sum_i c_i/(q_iE) = -(r+kD)/(DE) \bmod p_t$, where $r$ is the centered
    remainder of the source integer modulo $D$ and $|k|$ is smaller than half
    the number of source primes. Adding $x_t(DE)^{-1}$ completes
    $(x-r-kD)/(DE)$, the nearest quotient by $D$ up to the integer error $-k$,
    divided by $E$. Names carry ``prefix`` followed by the
    :data:`CRT_OPERANDS` names.
    """
    moduli = context.config.moduli
    tables: dict[str, torch.Tensor] = dict(
        zip(
            CRT_OPERANDS[2:],
            rns_ops.crt_quotient_tables(
                [moduli[i] for i in source_prime_ids],
                [moduli[i] for i in target_prime_ids],
                [moduli[i] for i in divisor_prime_ids],
                context.dtype,
            ),
            strict=True,
        )
    )
    for name, ids in (
        ("source_parameters", source_prime_ids),
        ("target_parameters", target_prime_ids),
    ):
        tables[name] = context.rns_parameter_tensor.index_select(
            1, torch.tensor(ids, dtype=torch.int64, device=context.device)
        )
    tables = {name: tables[name].to(context.device) for name in CRT_OPERANDS}
    return {prefix + name: value for name, value in tables.items()}


@dataclass(frozen=True)
class NativeCrtBasisExtensionImplementation:
    """Evaluate a raw or centered normalized CRT matrix over live Tensor operands.

    Source rows hold coefficient residues in $[0, 2q_i)$. A nonempty
    ``source_normalizers`` operand selects the centered residue
    $[a_i u_i]_{q_i}$ before the matrix. The output holds Montgomery residues of the target rows.
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
                tensors["target_offsets"] if normalized else None,
            ),
        )
