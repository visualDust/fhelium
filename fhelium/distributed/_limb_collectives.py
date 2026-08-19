"""Structural scatter and gather collectives for ciphertext RNS limbs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from fhelium.core import Ciphertext
from fhelium.distributed._collective_common import (
    _collect_argument_errors,
    _group_info,
)
from fhelium.distributed._value_collectives import (
    _gather_values,
    _scatter_values,
)


def scatter_ciphertext_limbs(
    shards_or_none: Sequence[Ciphertext] | None,
    *,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> Ciphertext:
    """Scatter RNS limb shards of one logical ciphertext from one source.

    The source sequence must partition one ciphertext into nonempty,
    contiguous prime intervals in process-group-rank order.  This function
    distributes those existing shards; it neither performs ciphertext
    arithmetic nor reconstructs the full ciphertext.  It is synchronous,
    accepts no ``async_op`` argument, and returns no
    :class:`torch.distributed.Work`.

    Args:
        shards_or_none: On ``src``, exactly one caller-prepared ciphertext limb
            shard per group rank, ordered by process-group rank and increasing
            canonical prime interval.  Every non-source rank must pass
            ``None``.
        src: Global rank of the source process, which must belong to ``group``.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        The shard assigned to the caller's process-group rank.  The source
        receives its existing sequence element; other ranks receive newly
        allocated shards.  For world size one, the sole validated shard is
        returned unchanged without communication or allocation.

    Raises:
        ValueError: If ``src`` is outside ``group``; the source/non-source
            calling rules or sequence-length requirements are violated; or
            shards do not describe compatible, contiguous prime intervals of
            one ciphertext.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    info = _group_info(group)
    local_error = None
    if info.global_rank == src and shards_or_none is not None:
        try:
            _validate_limb_shards(shards_or_none, check_device=False)
        except (TypeError, ValueError) as exc:
            local_error = str(exc)
    _collect_argument_errors("scatter_ciphertext_limbs", local_error, info)
    return _scatter_values(
        shards_or_none,
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
    nonempty, contiguous canonical intervals are concatenated along the RNS
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
        "context_id",
        "level",
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
                "Each ciphertext limb shard must contain a contiguous canonical "
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
        level=first.level,
        scale=first.scale,
        context_id=first.context_id,
        prime_ids=tuple(
            prime_id for shard in shards for prime_id in shard.prime_ids
        ),
        polynomial_domain=first.polynomial_domain,
        modulus_basis=first.modulus_basis,
        residue_representation=first.residue_representation,
    )
