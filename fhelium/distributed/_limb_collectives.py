"""Structural scatter and gather collectives for ciphertext RNS limbs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from fhelium.values import Ciphertext
from fhelium.distributed._collective_common import (
    _collect_argument_errors,
    _group_info,
)
from fhelium.distributed._value_collectives import (
    _gather_values,
    _scatter_values,
)


def scatter_ciphertext_limbs(
    value_or_none: Ciphertext | None,
    *,
    limb_ranges: Sequence[tuple[int, int]] | None = None,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> Ciphertext:
    """Slice and scatter caller-selected RNS intervals of one ciphertext.

    The source provides one nonempty half-open interval per group rank. The
    intervals refer to positions on the source's limb axis, form a consecutive
    range in group-rank order, and may have unequal lengths. ``slice_limbs``
    selects both Tensor rows and their declared ``prime_ids``. The caller
    chooses the partition; the collective performs no arithmetic or balancing.
    It is synchronous and accepts no ``async_op`` argument.

    Args:
        value_or_none: Source ciphertext on ``src``; ``None`` on every other
            rank. The source may itself represent an interval of the full basis.
        limb_ranges: On ``src``, one ``(start, stop)`` pair per process-group
            rank, indexing the source's stored rows rather than global prime
            IDs. Other ranks supply ``None``. The intervals may select a
            consecutive portion of the source without covering every row.
        src: Global rank of the source process, which must belong to ``group``.
        group: Participating process group. ``None`` selects the default group.

    Returns:
        The local ciphertext shard with its selected prime IDs and unchanged
        arithmetic state. On ``src``, it shares the source Tensor storage;
        other ranks receive newly allocated storage. World size one returns
        the requested source view without communication or copying.

    Raises:
        ValueError: If the source rank or source/non-source arguments are
            invalid; the range count differs from group size; or intervals
            are empty, out of bounds, or not consecutive in group-rank order.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    info = _group_info(group)
    local_error = None
    shards: list[Ciphertext] | None = None
    if info.global_rank == src:
        if not isinstance(value_or_none, Ciphertext) or limb_ranges is None:
            local_error = "source must supply a Ciphertext and limb_ranges"
        else:
            try:
                shards = [
                    value_or_none.slice_limbs(start, stop)
                    for start, stop in limb_ranges
                ]
                if any(
                    left[1] != right[0]
                    for left, right in zip(limb_ranges, limb_ranges[1:])
                ):
                    raise ValueError(
                        "limb_ranges must be consecutive in group-rank order"
                    )
            except (TypeError, ValueError) as exc:
                local_error = str(exc)
    elif value_or_none is not None or limb_ranges is not None:
        local_error = (
            "non-source ranks must supply None for value and limb_ranges"
        )
    _collect_argument_errors("scatter_ciphertext_limbs", local_error, info)
    return _scatter_values(
        shards,
        src=src,
        expected_type=Ciphertext,
        operation="scatter_ciphertext_limbs",
        group=group,
    )


def gather_ciphertext_limbs(
    local_shard: Ciphertext,
    *,
    dst: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> Ciphertext | None:
    """Gather and reconstruct one ciphertext from disjoint RNS limb shards.

    Group-rank order defines prime-interval order.  On ``dst``, compatible,
    nonempty, contiguous parameter intervals are concatenated along the RNS
    limb dimension.  This is structural reconstruction, not ciphertext
    addition; use ``reduce_ciphertext`` for additive partials.  The operation
    is synchronous, accepts no ``async_op`` argument, and returns no
    :class:`torch.distributed.Work`.

    Args:
        local_shard: Caller-owned rank-local ciphertext shard.  It is read but
            not mutated.
        dst: Global rank that reconstructs the ciphertext, which must belong to
            ``group``.  This is not a process-group-relative rank.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        The reconstructed ciphertext on ``dst`` and ``None`` on every other
        rank.  The destination allocates receive buffers and the concatenated
        result.  For world size one, the validated ``local_shard`` object is
        returned unchanged without concatenation.

    Raises:
        ValueError: If ``dst`` is outside ``group`` or shards differ in logical
            ciphertext metadata, dtype, device, or contiguous prime layout.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    parts = _gather_values(
        local_shard,
        dst=dst,
        expected_type=Ciphertext,
        operation="gather_ciphertext_limbs",
        group=group,
    )
    if parts is None:
        return None
    shards = cast(list[Ciphertext], parts)
    _validate_limb_shards(shards, check_device=True)
    return _concatenate_limb_shards(shards)


def _validate_limb_shards(
    shards: Sequence[Ciphertext],
    *,
    check_device: bool,
) -> None:
    if not shards:
        raise ValueError("Ciphertext limb shard sequence cannot be empty")
    if not all(isinstance(shard, Ciphertext) for shard in shards):
        raise TypeError("Ciphertext limb shards must all be Ciphertext values")

    first = shards[0]
    common_fields = (
        "depth",
        "scale",
        "polynomial_domain",
        "modulus_basis",
        "residue_representation",
        "component_count",
        "batch_shape",
        "ring_dimension",
    )
    previous_last_prime_id: int | None = None
    for rank, shard in enumerate(shards):
        mismatches = [
            name
            for name in common_fields
            if getattr(shard, name) != getattr(first, name)
        ]
        if shard.data.dtype != first.data.dtype:
            mismatches.append("dtype")
        if check_device and shard.data.device != first.data.device:
            mismatches.append("device")
        if mismatches:
            raise ValueError(
                "Ciphertext limb shards describe different logical values at "
                f"rank {rank}: mismatches={mismatches}"
            )
        if not shard.prime_ids:
            raise ValueError(f"Ciphertext limb shard {rank} is empty")
        start = shard.prime_ids[0]
        stop = start + len(shard.prime_ids)
        if shard.prime_ids != tuple(range(start, stop)):
            raise ValueError(
                "Each ciphertext limb shard must contain a contiguous parameter "
                f"prime interval; rank={rank}, prime_ids={shard.prime_ids}"
            )
        if (
            previous_last_prime_id is not None
            and start != previous_last_prime_id + 1
        ):
            raise ValueError(
                "Ciphertext limb shards must be contiguous in rank order: "
                f"rank={rank}, previous_prime={previous_last_prime_id}, "
                f"next_prime={start}"
            )
        previous_last_prime_id = shard.prime_ids[-1]


def _concatenate_limb_shards(
    shards: Sequence[Ciphertext],
) -> Ciphertext:
    if len(shards) == 1:
        return shards[0]
    first = shards[0]
    return Ciphertext(
        data=torch.cat([shard.data for shard in shards], dim=-2),
        depth=first.depth,
        scale=first.scale,
        prime_ids=tuple(
            prime_id for shard in shards for prime_id in shard.prime_ids
        ),
        polynomial_domain=first.polynomial_domain,
        modulus_basis=first.modulus_basis,
        residue_representation=first.residue_representation,
    )
