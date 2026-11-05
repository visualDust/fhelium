"""Hybrid-digit decomposition and RNS basis extension."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import cast

import torch
from xdsl.ir import Operation
from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import rns
from fhelium.native.wrapper import rns_ops


def modup_digit(
    source: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    digit_index: int,
    span: tuple[int, int, int],
    *,
    out: torch.Tensor | None = None,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Extend one coefficient digit to QP Montgomery residues.

    Garner decomposition represents the selected source residues as mixed-radix
    digits; basis extension evaluates those digits modulo each target prime and
    multiplies by its Montgomery radix. ``out`` receives the QP result when
    supplied. ``scratch`` receives the decomposition and has the source batch
    and coefficient dimensions with at least ``stop - start`` rows. Neither
    buffer may overlap the read operands or the other buffer. Without these
    buffers, the corresponding native operations allocate their outputs.

    Multi-prime CUDA digits use a shared-memory coefficient tile when its
    decomposition, extension, and parameter data fit 48 KiB and the caller supplies no decomposition scratch. Single-prime digits, CPU/HIP, larger tiles, and caller-supplied scratch use the separate native operations. Both routes
    read the supplied numerical tables and preserve the same lazy residues.
    """
    start, stop, _ = span
    mixed = source[..., start:stop, :]
    parameters = tensors["qp_parameters"]
    width, destinations = stop - start, parameters.size(1)
    columns = 128 if source.dtype == torch.int32 else 64
    tile_bytes = (
        width * columns
        + width
        + width * width
        + 4 * width
        + (width + 8) * destinations
    ) * source.element_size()
    if (
        source.is_cuda
        and not torch.version.hip
        and width > 1
        and scratch is None
        and tile_bytes <= 48 * 1024
    ):
        tables = [
            tensors[f"digit{digit_index}_normalizers"],
            tensors[f"digit{digit_index}_propagation"],
            *(
                tensors[f"digit{digit_index}_reduction{part}"]
                for part in range(4)
            ),
        ]
        extension = tensors[f"digit{digit_index}_extension"]
        if out is None:
            return rns_ops.mixed_radix_modup_tiled(
                mixed, tables, extension, parameters
            )
        rns_ops.mixed_radix_modup_tiled_out_(
            mixed, tables, extension, parameters, out
        )
        return out
    if stop - start > 1:
        arguments = (
            mixed,
            tensors[f"digit{digit_index}_normalizers"],
            tensors[f"digit{digit_index}_propagation"],
            *(
                tensors[f"digit{digit_index}_reduction{part}"]
                for part in range(4)
            ),
        )
        if scratch is None:
            mixed = rns_ops.mixed_radix_decompose(*arguments)
        else:
            mixed = scratch[..., : stop - start, :]
            rns_ops.mixed_radix_decompose_out_(*arguments, out=mixed)
    extension_arguments = (
        mixed,
        tensors[f"digit{digit_index}_extension"],
        parameters,
        parameters.size(1),
    )
    if out is None:
        return rns_ops.mixed_radix_basis_extend_to_montgomery(
            *extension_arguments
        )
    rns_ops.mixed_radix_basis_extend_to_montgomery_out_(
        *extension_arguments, out
    )
    return out


@dataclass(frozen=True)
class NativeHybridModUpImplementation:
    """Execute one represented hybrid-digit ModUp using concrete RNS tables."""

    name: str = "native-hybrid-modup"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (rns.HybridModUpDigitOp,)

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
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
        tensors = dict(
            zip(
                cast(tuple[str, ...], invocation.attributes["parameter_names"]),
                values[1:],
                strict=True,
            )
        )
        index = cast(int, invocation.attributes["digit_index"])
        spans = cast(
            tuple[tuple[int, int, int], ...], invocation.attributes["digits"]
        )
        return (modup_digit(values[0], tensors, index, spans[index]),)
