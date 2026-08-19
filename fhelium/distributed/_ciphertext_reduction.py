"""Arithmetic ciphertext reduction collectives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fhelium.core import Ciphertext
from fhelium.distributed._collective_common import (
    _check_global_rank,
    _check_identical_layout,
    _collect_argument_errors,
    _group_info,
    _GroupInfo,
)

if TYPE_CHECKING:
    from fhelium.engine import CkksEngine


def _tree_reduce_phases(
    world_size: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return logical sender/receiver edges for a general binomial tree.

    The construction deliberately supports non-power-of-two world sizes.  At
    phase ``mask``, every available ``receiver + mask`` subtree sends its
    accumulated value to ``receiver``.  Exactly ``world_size - 1`` messages are
    emitted, and no rank needs more than one incoming ciphertext buffer.
    """

    if world_size < 1:
        raise ValueError("world_size must be positive")
    phases = []
    mask = 1
    while mask < world_size:
        phases.append(
            tuple(
                (receiver + mask, receiver)
                for receiver in range(0, world_size, 2 * mask)
                if receiver + mask < world_size
            )
        )
        mask <<= 1
    return tuple(phases)


def _validate_ciphertext_reduce_layout(
    value: Ciphertext,
    *,
    operation: str,
    info: _GroupInfo,
) -> None:
    local_error = (
        None
        if isinstance(value, Ciphertext)
        else f"expected Ciphertext, got {type(value).__name__}"
    )
    _collect_argument_errors(operation, local_error, info)
    _check_identical_layout(value, operation, info)


def _reduce_ciphertext_tree(
    value: Ciphertext,
    *,
    dst: int,
    engine: CkksEngine,
    info: _GroupInfo,
) -> None:
    _check_global_rank(dst, info, "reduce_ciphertext dst")
    root_group_rank = info.global_ranks.index(dst)
    logical_rank = (info.group_rank - root_group_rank) % info.world_size

    for phase, edges in enumerate(_tree_reduce_phases(info.world_size)):
        sender_to_receiver = dict(edges)
        if logical_rank in sender_to_receiver:
            receiver_logical = sender_to_receiver[logical_rank]
            receiver_group_rank = (
                receiver_logical + root_group_rank
            ) % info.world_size
            requests = torch.distributed.batch_isend_irecv(
                [
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        value.data,
                        info.global_ranks[receiver_group_rank],
                        group=info.group,
                        tag=phase,
                    )
                ]
            )
            for request in requests:
                request.wait()
            return

        sender_logical = next(
            (sender for sender, receiver in edges if receiver == logical_rank),
            None,
        )
        if sender_logical is None:
            continue
        sender_group_rank = (sender_logical + root_group_rank) % info.world_size
        incoming = value.with_data(torch.empty_like(value.data))
        requests = torch.distributed.batch_isend_irecv(
            [
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    incoming.data,
                    info.global_ranks[sender_group_rank],
                    group=info.group,
                    tag=phase,
                )
            ]
        )
        for request in requests:
            request.wait()
        engine.add_(value, incoming)


def reduce_ciphertext(
    value: Ciphertext,
    *,
    dst: int = 0,
    engine: CkksEngine,
    group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """Synchronously sum ciphertext partials onto ``dst`` using ``add``.

    A binomial tree combines complete ciphertexts and works for every positive
    process-group size, not only powers of two.  The tree uses ``P - 1`` full
    ciphertext messages, ``O(log P)`` critical-path rounds, and one temporary
    receive buffer per active rank.  Non-destination values are partial or
    unchanged after return and must not be interpreted as the global sum.  This
    is component-wise CKKS ciphertext addition modulo the active Q primes, not
    gathering independent values or concatenating RNS limbs.  The operation
    accepts no ``async_op`` argument and returns no
    :class:`torch.distributed.Work`.

    Args:
        value: Rank-local additive ciphertext partial.  The object on ``dst``
            is updated in place; intermediate receivers may also be mutated as
            the tree accumulates subtrees.
        dst: Global rank that receives the complete sum, which must belong to
            ``group``.  This is not a process-group-relative rank.
        engine: Rank-local CKKS engine used for in-place modular ciphertext
            addition.  Every participating rank must provide an engine
            compatible with its ``value``.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        None.  Only ``value`` on ``dst`` is guaranteed to contain the global
        sum.  For world size one, the sole value is already the sum and remains
        unchanged.

    Raises:
        ValueError: If ``dst`` is outside ``group``; rank-local values are not
            ciphertexts with identical shape and arithmetic metadata; or the
            engine rejects an incompatible value.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    info = _group_info(group)
    _validate_ciphertext_reduce_layout(
        value,
        operation="reduce_ciphertext",
        info=info,
    )
    _reduce_ciphertext_tree(
        value,
        dst=dst,
        engine=engine,
        info=info,
    )


def all_reduce_ciphertext(
    value: Ciphertext,
    *,
    engine: CkksEngine,
    group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """Synchronously sum ciphertext partials and update every rank in place.

    The reduction phase uses the arbitrary-world-size binomial tree documented
    by ``reduce_ciphertext``.  The root then uses a representation-preserving
    tensor broadcast because broadcasting ciphertext payload bits requires no
    arithmetic specialization.  Asynchronous composite Work semantics are
    intentionally deferred until a measured workload demonstrates useful
    communication/``add`` overlap.  The implicit root is the first global rank
    in process-group-rank order.

    Args:
        value: Rank-local additive ciphertext partial.  Every rank's object is
            overwritten in place with the component-wise CKKS sum modulo the
            active Q primes.
        engine: Rank-local CKKS engine used during tree reduction.  Every
            participating rank must provide an engine compatible with its
            ``value``.
        group: Participating process group.  ``None`` selects the default
            process group.  No caller-selected root is used.

    Returns:
        None.  Every rank's existing ``value`` object contains the global sum
        after return.  For world size one, the value remains unchanged.  The
        function accepts no ``async_op`` argument and returns no
        :class:`torch.distributed.Work`.

    Raises:
        ValueError: If rank-local values are not ciphertexts with identical
            shape and arithmetic metadata, or an engine rejects an
            incompatible value.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    info = _group_info(group)
    _validate_ciphertext_reduce_layout(
        value,
        operation="all_reduce_ciphertext",
        info=info,
    )
    root = info.global_ranks[0]
    _reduce_ciphertext_tree(
        value,
        dst=root,
        engine=engine,
        info=info,
    )
    torch.distributed.broadcast(value.data, src=root, group=group)
