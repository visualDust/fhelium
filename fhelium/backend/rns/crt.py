r"""CRT lifts defined by supplied coefficient and modular-reduction Tensors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
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
    source_moduli = tuple(context.config.moduli[i] for i in source_prime_ids)
    target_moduli = tuple(context.config.moduli[i] for i in target_prime_ids)
    product = prod(source_moduli)
    cofactors = tuple(product // q for q in source_moduli)
    inverses = tuple(
        pow(partial, -1, q) for partial, q in zip(cofactors, source_moduli)
    )
    radix = context.montgomery_parameters.R
    matrix = [
        [
            partial * (inverse if method == "raw" else 1) * radix % q
            for q in target_moduli
        ]
        for partial, inverse in zip(cofactors, inverses)
    ]
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
        "matrix": torch.tensor(
            matrix, dtype=context.dtype, device=context.device
        ),
        "target_reciprocals": torch.tensor(
            [(1 << 64) // q for q in target_moduli],
            dtype=torch.int64,
            device=context.device,
        ),
        "source_normalizers": torch.tensor(
            inverses if method == "normalized" else (),
            dtype=context.dtype,
            device=context.device,
        ),
        "source_reciprocals": torch.tensor(
            [(1 << 64) // q for q in source_moduli]
            if method == "normalized"
            else (),
            dtype=torch.int64,
            device=context.device,
        ),
    }


def crt_basis_extend(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Evaluate the supplied CRT matrix, optionally normalizing source rows.

    Source is coefficient/standard [*batch, source row, coefficient], with
    residues in [0, 2*q_i); each row is canonicalized before evaluation. Matrix
    entry (i,j) is a canonical Montgomery coefficient in [0, p_j).
    Nonempty source_normalizers in [0, q_i) apply a_i*u_i mod q_i before the matrix.
    Reciprocals are floor(2**64 / modulus) for their corresponding parameter
    rows. CUDA int32 execution requires primes below 2**30.
    Output is canonical Montgomery [*batch, target row, coefficient]. Inputs
    and numerical tables remain read-only. An out buffer has the same shape,
    dtype and device and must be disjoint from every read operand. The portable
    route requires separate storage; the CUDA route also permits proven
    disjoint views within shared storage.
    """
    source_parameters = tensors["source_parameters"]
    target_parameters = tensors["target_parameters"]
    matrix = tensors["matrix"]
    normalizers = tensors["source_normalizers"]
    normalized = normalizers.numel() != 0
    width, destinations = matrix.shape
    if (
        source.is_cuda
        and torch.version.hip is None
        and source.dtype == torch.int32
        and width * (128 + destinations) * source.element_size() <= 48 * 1024
    ):
        operands = (
            source,
            source_parameters,
            target_parameters,
            matrix,
            tensors["target_reciprocals"],
            normalizers if normalized else None,
            tensors["source_reciprocals"] if normalized else None,
        )
        if out is None:
            return rns_ops.crt_basis_extend(*operands)
        rns_ops.crt_basis_extend_out_(*operands, out)
        return out

    if out is not None:
        shape = (*source.shape[:-2], destinations, source.size(-1))
        if (
            out.shape != shape
            or out.dtype != source.dtype
            or out.device != source.device
        ):
            raise ValueError("CRT output buffer shape, dtype or device differs")
        if any(
            torch._C._overlaps(out, value)
            for value in (source, *tensors.values())
        ):
            raise ValueError(
                "CRT output buffer must use storage disjoint from its operands"
            )
    values = source.clone()
    rns_ops.reduce_to_standard_(values, source_parameters)
    if normalized:
        normalizers_montgomery = normalizers.unsqueeze(-1).clone()
        rns_ops.to_montgomery_(normalizers_montgomery, source_parameters)
        values = rns_ops.montgomery_mul_row_scalars_standard(
            values, normalizers_montgomery.squeeze(-1), source_parameters
        )
        rns_ops.reduce_to_standard_(values, source_parameters)
    # A second factor R lets ordinary Montgomery multiplication map an
    # unencoded source residue directly to the desired encoded matrix result.
    coefficients = matrix.transpose(0, 1).clone(
        memory_format=torch.contiguous_format
    )
    rns_ops.to_montgomery_(coefficients, target_parameters)
    modulus_shape = (*([1] * (source.ndim - 2)), -1, 1)
    moduli = (target_parameters[0] // 2).view(modulus_shape)
    result = torch.zeros(
        (*source.shape[:-2], destinations, source.size(-1)),
        dtype=source.dtype,
        device=source.device,
    )
    for row in range(width):
        residues = torch.remainder(values[..., row : row + 1, :], moduli)
        term = rns_ops.montgomery_mul_row_scalars_standard(
            residues, coefficients[:, row], target_parameters
        )
        result = rns_ops.add_lazy(result, term, target_parameters)
    rns_ops.reduce_to_standard_(result, target_parameters)
    if out is not None:
        out.copy_(result)
        return out
    return result


@dataclass(frozen=True)
class NativeCrtBasisExtensionImplementation:
    """Execute a raw or source-normalized CRT matrix over live Tensor operands."""

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
        tensors: dict[str, torch.Tensor] = dict(
            zip(CRT_OPERANDS, inputs[1:], strict=True)
        )
        return (crt_basis_extend(inputs[0], tensors),)
