"""Rank-local implementations of distributed IR operations as process-group collectives of the schedule.

Rank and group size are read from the bound process group when the schedule runs. A ciphertext sum reduce-scatters the residue payloads with integer addition, reduces each rank's row block modulo its primes and all-gathers the blocks; every rank holds the same bits because the integer sum does not depend on the summation order. A reduction with a combine region, which the default pipeline produces for two-rank groups, gathers every rank's payload with ``all_gather`` and folds the gathered payloads in group rank order with the region's arithmetic, which the schedule contains as ordinary operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from xdsl.ir import Operation

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.config._derived import config_from_text
from fhelium.ir.dialects import distributed

from .resources import ProcessGroupExecutionResource
from fhelium.backend.lowering import ExecBuilder

TensorCombine = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Combine two rank-local Tensor payloads into a functional result."""


def _bound_group(
    resources: tuple[BoundResource, ...],
) -> tuple[BoundResource, ProcessGroupExecutionResource]:
    if not resources:
        raise ValueError(
            "Distributed operation requires a process-group resource"
        )
    return resources[0], cast(ProcessGroupExecutionResource, resources[0].value)


def _fold(
    builder: ExecBuilder,
    value: torch.Tensor,
    resources: tuple[BoundResource, ...],
    combine: TensorCombine,
) -> torch.Tensor:
    """All-gather equal-layout payloads and fold them in group rank order."""
    bound, group = _bound_group(resources)
    local = value.contiguous()
    gathered = builder.alloc(
        (group.size, *local.shape), dtype=local.dtype, device=local.device
    )
    builder.collective("all_gather", bound, (local,), (gathered,))
    result = gathered[0]
    for rank in range(1, group.size):
        result = combine(result, gathered[rank])
    return result


class _OperandResourceImplementation:
    """Declare that the process group arrives as an IR resource operand."""

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()


@dataclass(frozen=True)
class TorchProcessGroupQueryImplementation(_OperandResourceImplementation):
    """Read group rank and size from the bound process group when the schedule runs."""

    name: str = "torch-process-group-query"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        distributed.RankOp,
        distributed.GroupSizeOp,
    )

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del inputs
        bound, group = _bound_group(resources)
        kind = (
            "rank"
            if invocation.operation_type is distributed.RankOp
            else "size"
        )
        # The result is a host integer read from the group when the schedule
        # runs; the lowering returns it as a 0-dim int64 CPU Tensor, or uses
        # the current value when it drives structured control flow.
        value = builder.group_query(
            kind, bound, group.rank if kind == "rank" else group.size
        )
        return (cast(torch.Tensor, value),)


@dataclass(frozen=True)
class TorchBroadcastImplementation(_OperandResourceImplementation):
    """Broadcast one equal-layout Tensor from a group rank into a new buffer."""

    name: str = "torch-broadcast"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (distributed.BroadcastOp,)

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        bound, group = _bound_group(resources)
        root = cast(int, invocation.attributes["root"])
        if not 0 <= root < group.size:
            raise IndexError(f"group rank {root} is outside [0, {group.size})")
        local = inputs[0].contiguous()
        output = builder.alloc(
            local.shape, dtype=local.dtype, device=local.device
        )
        builder.collective("broadcast", bound, (local,), (output,), (root,))
        return (output,)


@dataclass(frozen=True)
class TorchGenericAllReduceImplementation(_OperandResourceImplementation):
    """Fold all rank-local payloads with the operation's combine region."""

    name: str = "torch-generic-all-reduce"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (distributed.AllReduceOp,)

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        del invocation
        if len(builder.regions) != 1:
            raise ValueError("Generic all-reduce requires one combine region")
        region = builder.regions[0]

        def combine(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            outputs = builder.inline(region, (lhs, rhs))
            if len(outputs) != 1:
                raise ValueError(
                    "Generic all-reduce combine region returned "
                    f"{len(outputs)} values"
                )
            return cast(torch.Tensor, outputs[0])

        return (_fold(builder, inputs[0], resources, combine),)


def _modular_reduction(
    builder: ExecBuilder,
    value: torch.Tensor,
    resources: tuple[BoundResource, ...],
    moduli: tuple[int, ...],
    root: int | None,
) -> torch.Tensor:
    """Sum residue payloads across the group modulo the moduli of their rows.

    ``value`` has shape ``[..., L, N]`` with row ``l`` modulo ``moduli[l]``. Every rank receives the sum, or with ``root`` only that group rank does and the other ranks receive zeros.
    """
    bound, group = _bound_group(resources)
    size = group.size
    local = value.contiguous()
    shape = local.shape
    coefficients, limbs = shape[-1], len(moduli)
    rows = local.numel() // coefficients
    owned = -(-rows // size)
    padded = owned * size
    largest = max(moduli)
    narrow = local.dtype == torch.int32
    # Signed SUM wraps in two's complement, so the summed bits read as
    # unsigned are exact while P * r_max <= B, with r_max = 2q - 1 for lazy
    # residues or q - 1 after normalization, and B = 2^32 - 1 for int32
    # payloads (read through a 32-bit mask) or 2^63 - 1 for int64 payloads.
    # P, q and the dtype are static: int32 beyond the reduced bound is widened
    # to int64, int64 beyond it is all-gathered and folded modulo q.
    bound_sum = (1 << 32) - 1 if narrow else (1 << 63) - 1
    widen = size * (largest - 1) > bound_sum
    normalize = not widen and size * (2 * largest - 1) > bound_sum
    if widen and not narrow:
        return _fold(
            builder,
            local,
            resources,
            lambda lhs, rhs: _modular_add(lhs, rhs, moduli),
        )
    row_moduli = torch.tensor(
        [moduli[row % limbs] for row in range(padded)],
        dtype=torch.int64,
        device=local.device,
    )
    flat = local.view(rows, coefficients)
    dtype = torch.int64 if widen else local.dtype
    if normalize or widen or padded != rows:
        source = builder.alloc(
            (padded, coefficients),
            dtype=dtype,
            device=local.device,
            zero=padded != rows,
        )
        if normalize:
            modulus = row_moduli[:rows].view(rows, 1).to(local.dtype)
            flat = torch.where(flat >= modulus, flat - modulus, flat)
        builder.copy(source[:rows], flat.to(dtype) if widen else flat)
    else:
        source = flat
    chunk = builder.alloc(
        (owned, coefficients), dtype=dtype, device=local.device
    )
    builder.collective("reduce_scatter", bound, (source,), (chunk,))
    rank = torch.ops.aten.full.default(
        [1],
        builder.group_query("rank", bound, group.rank),
        dtype=torch.int64,
        device=local.device,
    )
    modulus = torch.index_select(row_moduli.view(size, owned), 0, rank).view(
        owned, 1
    )
    summed = chunk.to(torch.int64)
    if narrow and not widen:
        summed = torch.bitwise_and(summed, (1 << 32) - 1)
    reduced = torch.remainder(summed, modulus).to(local.dtype)
    gathered = builder.alloc(
        (padded, coefficients),
        dtype=local.dtype,
        device=local.device,
        zero=root is not None,
    )
    if root is None:
        builder.collective("all_gather", bound, (reduced,), (gathered,))
    else:
        builder.collective("gather", bound, (reduced,), (gathered,), (root,))
    return gathered[:rows].view(shape)


def _modular_add(
    lhs: torch.Tensor, rhs: torch.Tensor, moduli: tuple[int, ...]
) -> torch.Tensor:
    """Add int64 residue payloads of lazy range and reduce each row modulo its prime."""
    modulus = torch.tensor(moduli, dtype=lhs.dtype, device=lhs.device).view(
        len(moduli), 1
    )
    return torch.remainder(lhs + rhs, modulus)


@dataclass(frozen=True)
class TorchCiphertextAllReduceImplementation(_OperandResourceImplementation):
    """Sum ciphertext payloads modulo their primes by reduce-scatter and all-gather.

    Each rank receives the integer sum of its row block of every rank's payload, reduces it modulo the rows' primes and all-gathers the reduced blocks; with ``root``, the blocks are gathered on that group rank only. The operation's ``ckks_config`` and its operand's prime ids give the row moduli.
    """

    name: str = "torch-ciphertext-all-reduce"
    supports_in_place: bool = False
    operation_types: tuple[type[Operation], ...] = (
        distributed.AllReduceAddCiphertextOp,
    )

    def lower(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        builder: ExecBuilder,
        /,
    ) -> tuple[torch.Tensor, ...]:
        config = config_from_text(
            cast(str, invocation.attributes["ckks_config"])
        )
        prime_ids = cast(tuple[int, ...], invocation.operand_prime_ids[0])
        root = invocation.attributes.get("root")
        return (
            _modular_reduction(
                builder,
                inputs[0],
                resources,
                tuple(config.moduli[prime] for prime in prime_ids),
                None if root is None else int(cast(int, root)),
            ),
        )


def distributed_operation_contributions() -> tuple[
    TorchProcessGroupQueryImplementation
    | TorchBroadcastImplementation
    | TorchGenericAllReduceImplementation
    | TorchCiphertextAllReduceImplementation,
    ...,
]:
    """Return the distributed implementations of the default registry."""
    return (
        TorchProcessGroupQueryImplementation(),
        TorchBroadcastImplementation(),
        TorchGenericAllReduceImplementation(),
        TorchCiphertextAllReduceImplementation(),
    )


__all__ = [
    "TorchBroadcastImplementation",
    "TorchCiphertextAllReduceImplementation",
    "TorchGenericAllReduceImplementation",
    "TorchProcessGroupQueryImplementation",
    "TensorCombine",
    "distributed_operation_contributions",
]
