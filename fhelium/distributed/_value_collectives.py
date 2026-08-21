"""Typed whole-value broadcast, scatter, gather, and all-gather."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from fhelium.core import Ciphertext, CompressedPlaintext, Plaintext
from fhelium.distributed._collective_common import (
    _all_gather_descriptors,
    _all_gather_tensor,
    _broadcast_descriptor,
    _broadcast_descriptor_list,
    _broadcast_transfer_tensor,
    _check_global_rank,
    _check_identical_layout,
    _collect_argument_errors,
    _group_info,
    _GroupInfo,
    _KeyT,
    _p2p_transfer_tensor,
    _wait_p2p_ops,
    _workload_tensors,
    _WorkloadT,
)
from fhelium.distributed._state import local_device
from fhelium.distributed._transfer import (
    allocate_key,
    allocate_value,
    describe_key,
    describe_value,
)


def _broadcast_typed_value(
    value_or_none: _WorkloadT | None,
    *,
    expected_type: type[_WorkloadT],
    src: int,
    group: torch.distributed.ProcessGroup | None,
    operation: str,
) -> _WorkloadT:
    info = _group_info(group)
    _check_global_rank(src, info, f"{operation} src")
    local_error = None
    if info.global_rank == src:
        if not isinstance(value_or_none, expected_type):
            local_error = (
                f"source rank {src} must supply {expected_type.__name__}, "
                f"got {type(value_or_none).__name__}"
            )
    elif value_or_none is not None:
        local_error = (
            f"non-source rank {info.global_rank} must supply None, got "
            f"{type(value_or_none).__name__}"
        )
    _collect_argument_errors(operation, local_error, info)
    source_value = cast(_WorkloadT, value_or_none)
    if info.world_size == 1:
        return source_value

    descriptor = (
        describe_value(source_value) if info.global_rank == src else None
    )
    descriptor = _broadcast_descriptor(descriptor, src=src, info=info)
    result = (
        source_value
        if info.global_rank == src
        else cast(
            _WorkloadT,
            allocate_value(descriptor, local_device=local_device()),
        )
    )
    for tensor in _workload_tensors(result):
        _broadcast_transfer_tensor(tensor, src=src, info=info)
    return result


def broadcast_ciphertext(
    value_or_none: Ciphertext | None,
    *,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> Ciphertext:
    """Broadcast one complete ciphertext and allocate non-source receivers.

    A small internal object descriptor carries shape and CKKS metadata; the
    dense payload then uses the ordinary tensor broadcast.  No placement or
    owner policy is inferred.  The operation is synchronous: it accepts no
    ``async_op`` argument and returns no :class:`torch.distributed.Work`.

    Args:
        value_or_none: Ciphertext supplied only by ``src``.  Every non-source
            rank must pass ``None``.
        src: Global rank of the source process, which must belong to ``group``.
            This is not a process-group-relative rank.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        The source's original ciphertext on ``src`` and a newly allocated,
        metadata-equivalent ciphertext on every other group rank.  For a
        world-size-one group, the source object is returned unchanged and no
        payload communication or receiver allocation occurs.

    Raises:
        ValueError: If ``src`` is outside ``group`` or ranks violate the
            source/non-source argument rules.
        RuntimeError: If distributed communication is uninitialized, the
            caller is not a group member, or a valid transfer descriptor
            cannot be exchanged.
    """

    return _broadcast_typed_value(
        value_or_none,
        expected_type=Ciphertext,
        src=src,
        group=group,
        operation="broadcast_ciphertext",
    )


def broadcast_plaintext(
    value_or_none: Plaintext | None,
    *,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> Plaintext:
    """Broadcast one complete plaintext and allocate non-source receivers.

    The message or encoded data representation and its arithmetic metadata are
    preserved.  The operation does not mutate the source plaintext, accepts no
    ``async_op`` argument, and returns no :class:`torch.distributed.Work`.

    Args:
        value_or_none: Plaintext supplied only by ``src``.  Every non-source
            rank must pass ``None``.
        src: Global rank of the source process, which must belong to ``group``.
            This is not a process-group-relative rank.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        The source's original plaintext on ``src`` and a newly allocated,
        metadata-equivalent plaintext on every other group rank.  For a
        world-size-one group, the source object is returned unchanged without
        payload communication or receiver allocation.

    Raises:
        ValueError: If ``src`` is outside ``group`` or ranks violate the
            source/non-source argument rules.
        RuntimeError: If distributed communication is uninitialized, the
            caller is not a group member, or a valid transfer descriptor
            cannot be exchanged.
    """

    return _broadcast_typed_value(
        value_or_none,
        expected_type=Plaintext,
        src=src,
        group=group,
        operation="broadcast_plaintext",
    )


def broadcast_compressed_plaintext(
    value_or_none: CompressedPlaintext | None,
    *,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> CompressedPlaintext:
    """Broadcast one compressed plaintext with typed allocation.

    The compact tensor, ring dimension, encoded repetition layout, and all
    arithmetic metadata are preserved. The operation is synchronous and does
    not infer ownership, placement, or residency policy.
    """

    return _broadcast_typed_value(
        value_or_none,
        expected_type=CompressedPlaintext,
        src=src,
        group=group,
        operation="broadcast_compressed_plaintext",
    )


def broadcast_key(
    key_or_none: _KeyT | None,
    *,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> _KeyT:
    """Broadcast selected key material with typed allocation.

    The function performs no key generation, owner inference, or automatic
    placement.  Calling it is the program's visible decision to communicate
    one selected key.  It is synchronous, accepts no ``async_op`` argument,
    and returns no :class:`torch.distributed.Work`.

    Args:
        key_or_none: Supported dense FHElium key supplied only by ``src``.
            Every non-source rank must pass ``None``.
        src: Global rank of the source process, which must belong to ``group``.
            This is not a process-group-relative rank.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        The source's original key object on ``src`` and a newly allocated key
        of the same concrete type and metadata on every other group rank. For a
        world-size-one group, the source object is returned unchanged and no
        payload communication or receiver allocation occurs.

    Raises:
        ValueError: If ``src`` is outside ``group``, the source value is not a
            supported key, or ranks violate the source/non-source argument rules.
        RuntimeError: If distributed communication is uninitialized, the
            caller is not a group member, or a valid transfer descriptor
            cannot be exchanged.
    """

    info = _group_info(group)
    _check_global_rank(src, info, "broadcast_key src")
    local_error = None
    if info.global_rank == src:
        try:
            describe_key(key_or_none)
        except (TypeError, ValueError) as exc:
            local_error = str(exc)
    elif key_or_none is not None:
        local_error = (
            f"non-source rank {info.global_rank} must supply None, got "
            f"{type(key_or_none).__name__}"
        )
    _collect_argument_errors("broadcast_key", local_error, info)
    source_key = cast(_KeyT, key_or_none)
    if info.world_size == 1:
        return source_key

    descriptor = describe_key(source_key) if info.global_rank == src else None
    descriptor = _broadcast_descriptor(descriptor, src=src, info=info)
    result = (
        source_key
        if info.global_rank == src
        else cast(
            _KeyT,
            allocate_key(descriptor, local_device=local_device()),
        )
    )
    for tensor in _workload_tensors(result):
        _broadcast_transfer_tensor(tensor, src=src, info=info)
    return result


def _check_scatter_source(
    values_or_none: Sequence[_WorkloadT] | None,
    *,
    src: int,
    expected_type: type[_WorkloadT],
    operation: str,
    info: _GroupInfo,
) -> Sequence[_WorkloadT]:
    _check_global_rank(src, info, f"{operation} src")
    local_error = None
    if info.global_rank == src:
        if values_or_none is None:
            local_error = f"source rank {src} must supply values"
        elif len(values_or_none) != info.world_size:
            local_error = (
                "source must supply exactly group world_size values; "
                f"got {len(values_or_none)} for world_size={info.world_size}"
            )
        elif not all(
            isinstance(value, expected_type) for value in values_or_none
        ):
            local_error = f"all source values must be {expected_type.__name__}"
    elif values_or_none is not None:
        local_error = f"non-source rank {info.global_rank} must supply None"
    _collect_argument_errors(operation, local_error, info)
    return cast(Sequence[_WorkloadT], values_or_none or ())


def _scatter_values(
    values_or_none: Sequence[_WorkloadT] | None,
    *,
    src: int,
    expected_type: type[_WorkloadT],
    operation: str,
    group: torch.distributed.ProcessGroup | None,
) -> _WorkloadT:
    info = _group_info(group)
    source_values = _check_scatter_source(
        values_or_none,
        src=src,
        expected_type=expected_type,
        operation=operation,
        info=info,
    )
    source_group_rank = info.global_ranks.index(src)
    if info.world_size == 1:
        return source_values[0]

    descriptors = (
        [describe_value(value) for value in source_values]
        if info.global_rank == src
        else None
    )
    descriptors = _broadcast_descriptor_list(
        descriptors,
        src=src,
        info=info,
    )
    result = (
        source_values[info.group_rank]
        if info.global_rank == src
        else cast(
            _WorkloadT,
            allocate_value(
                descriptors[info.group_rank],
                local_device=local_device(),
            ),
        )
    )

    operations: list[torch.distributed.P2POp] = []
    receive_copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    if info.global_rank == src:
        for destination_group_rank, destination_value in enumerate(
            source_values
        ):
            if destination_group_rank == source_group_rank:
                continue
            destination = info.global_ranks[destination_group_rank]
            for tag, tensor in enumerate(_workload_tensors(destination_value)):
                transfer = _p2p_transfer_tensor(
                    tensor,
                    receive_copies,
                    receiving=False,
                    info=info,
                )
                operations.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        transfer,
                        destination,
                        group=group,
                        tag=tag,
                    )
                )
    else:
        for tag, tensor in enumerate(_workload_tensors(result)):
            transfer = _p2p_transfer_tensor(
                tensor,
                receive_copies,
                receiving=True,
                info=info,
            )
            operations.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    transfer,
                    src,
                    group=group,
                    tag=tag,
                )
            )
    _wait_p2p_ops(operations, receive_copies)
    return result


def scatter_ciphertexts(
    values_or_none: Sequence[Ciphertext] | None,
    *,
    src: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> Ciphertext:
    """Scatter independent ciphertext workload items from one source.

    This is representation-preserving transport, not an arithmetic operation.
    Sequence position is process-group-rank order even though ``src`` is a
    global rank.  The operation is synchronous, accepts no ``async_op``
    argument, and returns no :class:`torch.distributed.Work`.

    Args:
        values_or_none: On ``src``, one independent ciphertext per group rank
            in process-group-rank order.  Every non-source rank must pass
            ``None``.
        src: Global rank of the source process, which must belong to ``group``.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        The ciphertext assigned to the caller's process-group rank.  The
        source receives its existing sequence element; other ranks receive
        newly allocated ciphertexts.  For world size one, ``values_or_none[0]``
        is returned unchanged without communication or allocation.

    Raises:
        ValueError: If ``src`` is outside ``group``; the source does not
            provide exactly one ciphertext per group rank; a non-source rank
            supplies values; or rank-local arguments disagree.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    return _scatter_values(
        values_or_none,
        src=src,
        expected_type=Ciphertext,
        operation="scatter_ciphertexts",
        group=group,
    )


def _gather_values(
    value: _WorkloadT,
    *,
    dst: int,
    expected_type: type[_WorkloadT],
    operation: str,
    group: torch.distributed.ProcessGroup | None,
) -> list[_WorkloadT] | None:
    info = _group_info(group)
    _check_global_rank(dst, info, f"{operation} dst")
    local_error = (
        None
        if isinstance(value, expected_type)
        else f"expected {expected_type.__name__}, got {type(value).__name__}"
    )
    _collect_argument_errors(operation, local_error, info)
    if info.world_size == 1:
        return [value]

    descriptors = _all_gather_descriptors(describe_value(value), info=info)
    destination_group_rank = info.global_ranks.index(dst)
    results: list[_WorkloadT] | None = None
    if info.global_rank == dst:
        results = [
            value
            if group_rank == destination_group_rank
            else cast(
                _WorkloadT,
                allocate_value(
                    descriptors[group_rank],
                    local_device=local_device(),
                ),
            )
            for group_rank in range(info.world_size)
        ]

    operations: list[torch.distributed.P2POp] = []
    receive_copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    if info.global_rank == dst:
        if results is None:
            raise RuntimeError(f"{operation} lost destination buffers")
        for source_group_rank, source_value in enumerate(results):
            if source_group_rank == destination_group_rank:
                continue
            source = info.global_ranks[source_group_rank]
            for tag, tensor in enumerate(_workload_tensors(source_value)):
                transfer = _p2p_transfer_tensor(
                    tensor,
                    receive_copies,
                    receiving=True,
                    info=info,
                )
                operations.append(
                    torch.distributed.P2POp(
                        torch.distributed.irecv,
                        transfer,
                        source,
                        group=group,
                        tag=tag,
                    )
                )
    else:
        for tag, tensor in enumerate(_workload_tensors(value)):
            transfer = _p2p_transfer_tensor(
                tensor,
                receive_copies,
                receiving=False,
                info=info,
            )
            operations.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    transfer,
                    dst,
                    group=group,
                    tag=tag,
                )
            )
    _wait_p2p_ops(operations, receive_copies)
    return results


def gather_ciphertexts(
    value: Ciphertext,
    *,
    dst: int = 0,
    group: torch.distributed.ProcessGroup | None = None,
) -> list[Ciphertext] | None:
    """Gather independent ciphertexts in process-group rank order.

    This is representation-preserving transport, not an algebraic reduction.
    Use ``reduce_ciphertext`` when the inputs are additive partials of one
    logical encrypted value.  The operation is synchronous, accepts no
    ``async_op`` argument, and returns no :class:`torch.distributed.Work`.

    Args:
        value: One independent rank-local ciphertext.  It is read but not
            mutated.
        dst: Global rank that receives the list, which must belong to ``group``.
            This is not a process-group-relative rank.
        group: Participating process group.  ``None`` selects the default
            process group.

    Returns:
        On ``dst``, one ciphertext per group rank in process-group-rank order;
        the destination's list entry is its original object and other entries
        are newly allocated.  Other ranks return ``None``.  For world size one,
        the sole rank returns ``[value]`` with the original object.

    Raises:
        ValueError: If ``dst`` is outside ``group``, a rank does not provide a
            ciphertext, or rank-local arguments disagree.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    return cast(
        list[Ciphertext] | None,
        _gather_values(
            value,
            dst=dst,
            expected_type=Ciphertext,
            operation="gather_ciphertexts",
            group=group,
        ),
    )


def all_gather_ciphertexts(
    value: Ciphertext,
    *,
    group: torch.distributed.ProcessGroup | None = None,
) -> list[Ciphertext]:
    """All-gather independent ciphertexts with one identical layout.

    Equality covers context, level, scale, components, prime IDs, domain,
    basis, Montgomery state, degree, shape, and dtype.  Rank-local CUDA device
    indices may differ because descriptors intentionally preserve device type
    rather than one global device index.  This is transport of independent
    values, not algebraic reduction.  The operation is synchronous, accepts no
    ``async_op`` argument, and returns no :class:`torch.distributed.Work`.

    Args:
        value: One independent rank-local ciphertext.  All group ranks must
            provide ciphertexts with identical layout metadata.  The input is
            read but not mutated.
        group: Participating process group.  ``None`` selects the default
            process group.  No root rank is used.

    Returns:
        A newly allocated list on every rank containing one ciphertext per
        group rank in process-group-rank order.  Entries, including the
        caller's entry, are new ciphertext objects whose tensors use the same
        device as the caller's input.  For world size one, a one-element copied
        result is returned.

    Raises:
        TypeError: If ``value`` is not a ciphertext.
        ValueError: If ciphertext shapes or arithmetic metadata differ across
            group ranks.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    if not isinstance(value, Ciphertext):
        raise TypeError("all_gather_ciphertexts expects a Ciphertext")
    info = _group_info(group)
    _check_identical_layout(value, "all_gather_ciphertexts", info)
    return [
        value.with_data(data)
        for data in _all_gather_tensor(value.data, info=info)
    ]


def all_gather_plaintexts(
    value: Plaintext,
    *,
    group: torch.distributed.ProcessGroup | None = None,
) -> list[Plaintext]:
    """All-gather independent plaintexts with one arithmetic state.

    Every rank must provide the same message/encoded representation kind,
    shape, dtype, context, level, scale, layout, domain, basis, Montgomery
    state, and prime IDs.  This is representation-preserving transport, not an
    arithmetic operation.  It is synchronous, accepts no ``async_op``
    argument, and returns no :class:`torch.distributed.Work`.

    Args:
        value: One independent rank-local plaintext.  It is read but not
            mutated.
        group: Participating process group.  ``None`` selects the default
            process group.  No root rank is used.

    Returns:
        A newly allocated list on every rank containing one plaintext per
        group rank in process-group-rank order.  Entries, including the
        caller's entry, are new plaintext objects whose tensors use the same
        device as the caller's input.  For world size one, a one-element copied
        result is returned.

    Raises:
        TypeError: If ``value`` is not a plaintext.
        ValueError: If plaintext representations, shapes, or arithmetic
            metadata differ across group ranks.
        RuntimeError: If distributed communication is uninitialized or the
            caller is not a member of ``group``.
    """

    if not isinstance(value, Plaintext):
        raise TypeError("all_gather_plaintexts expects a Plaintext")
    info = _group_info(group)
    _check_identical_layout(value, "all_gather_plaintexts", info)
    message_parts = (
        None
        if value.message is None
        else _all_gather_tensor(value.message, info=info)
    )
    data_parts = (
        None
        if value.data is None
        else _all_gather_tensor(value.data, info=info)
    )
    return [
        Plaintext(
            message=None if message_parts is None else message_parts[rank],
            level=value.level,
            scale=value.scale,
            data=None if data_parts is None else data_parts[rank],
            context_id=value.context_id,
            representation=value.representation,
            polynomial_domain=value.polynomial_domain,
            modulus_basis=value.modulus_basis,
            residue_representation=value.residue_representation,
            prime_ids=value.prime_ids,
        )
        for rank in range(info.world_size)
    ]


def all_gather_compressed_plaintexts(
    value: CompressedPlaintext,
    *,
    group: torch.distributed.ProcessGroup | None = None,
) -> list[CompressedPlaintext]:
    """All-gather independent compressed plaintexts sharing one layout."""

    if not isinstance(value, CompressedPlaintext):
        raise TypeError(
            "all_gather_compressed_plaintexts expects a CompressedPlaintext"
        )
    info = _group_info(group)
    _check_identical_layout(value, "all_gather_compressed_plaintexts", info)
    data_parts = _all_gather_tensor(value.data, info=info)
    implicit_parts = (
        None
        if value.implicit_data is None
        else _all_gather_tensor(value.implicit_data, info=info)
    )
    return [
        value.with_storage(
            data_parts[rank],
            None if implicit_parts is None else implicit_parts[rank],
        )
        for rank in range(info.world_size)
    ]
