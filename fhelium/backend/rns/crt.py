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
from fhelium.backend.lowering import ExecBuilder

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
    return dict(
        zip(
            *rns_ops.crt_table_set(
                context.rns_parameter_tensor,
                list(context.config.moduli),
                context.config.logN,
                list(source_prime_ids),
                list(target_prime_ids),
                method == "normalized",
            ),
            strict=True,
        )
    )


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
    names, tensors = rns_ops.centered_quotient_table_set(
        context.rns_parameter_tensor,
        list(context.config.moduli),
        context.config.logN,
        list(source_prime_ids),
        list(target_prime_ids),
        list(divisor_prime_ids),
    )
    return {
        prefix + name: value for name, value in zip(names, tensors, strict=True)
    }


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

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources
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
