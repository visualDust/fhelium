"""Process-group resolution and descriptor/payload transport primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, TypeVar, cast

import torch

from fhelium.core import (
    Ciphertext,
    CompressedPlaintext,
    ConjugationKey,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.distributed._state import local_device
from fhelium.distributed._transfer import (
    TransferDescriptor,
    _transfer_tensors,
    describe_value,
)

_WorkloadValue: TypeAlias = (
    torch.Tensor | Ciphertext | Plaintext | CompressedPlaintext
)
_KeyValue: TypeAlias = (
    SecretKey
    | PublicKey
    | KeySwitchKey
    | RotationKey
    | RelinearizationKey
    | ConjugationKey
)
_WorkloadT = TypeVar("_WorkloadT", bound=_WorkloadValue)
_KeyT = TypeVar("_KeyT", bound=_KeyValue)


@dataclass(frozen=True)
class _GroupInfo:
    """Resolved ranks for one ProcessGroup; internal, never placement state."""

    group: torch.distributed.ProcessGroup | None
    global_rank: int
    group_rank: int
    world_size: int
    global_ranks: tuple[int, ...]


def _group_info(
    group: torch.distributed.ProcessGroup | None,
) -> _GroupInfo:
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed collective requested before dist.init or "
            "init_process_group"
        )
    group_rank = torch.distributed.get_rank(group)
    if group_rank < 0:
        raise RuntimeError("The current process is not a member of the group")
    # Older supported PyTorch releases require the WORLD group here,
    # while newer releases also accept ``None`` as the default-group sentinel.
    rank_group = group if group is not None else torch.distributed.group.WORLD
    global_ranks = tuple(torch.distributed.get_process_group_ranks(rank_group))
    return _GroupInfo(
        group=group,
        global_rank=torch.distributed.get_rank(),
        group_rank=group_rank,
        world_size=torch.distributed.get_world_size(group),
        global_ranks=global_ranks,
    )


def _check_global_rank(rank: int, info: _GroupInfo, name: str) -> None:
    if rank not in info.global_ranks:
        raise ValueError(
            f"{name}={rank} is not a member of process group "
            f"{info.global_ranks}"
        )


def _collect_argument_errors(
    operation: str,
    local_error: str | None,
    info: _GroupInfo,
) -> None:
    """Make local argument failures visible on every rank before payload I/O."""

    if info.world_size == 1:
        if local_error is not None:
            raise ValueError(local_error)
        return
    errors: list[str | None] = [None] * info.world_size
    torch.distributed.all_gather_object(
        errors,
        local_error,
        group=info.group,
    )
    failures = [
        f"group_rank={rank}: {error}"
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        raise ValueError(
            f"{operation} argument validation failed: " + "; ".join(failures)
        )


def _broadcast_descriptor(
    descriptor: TransferDescriptor | None,
    *,
    src: int,
    info: _GroupInfo,
) -> TransferDescriptor:
    objects = [descriptor]
    backend = str(torch.distributed.get_backend(info.group)).lower()
    if backend == "nccl":
        torch.distributed.broadcast_object_list(
            objects,
            src=src,
            group=info.group,
            device=local_device(),
        )
    else:
        torch.distributed.broadcast_object_list(
            objects,
            src=src,
            group=info.group,
        )
    result = objects[0]
    if not isinstance(result, dict):
        raise RuntimeError("Typed broadcast received no valid descriptor")
    return cast(TransferDescriptor, result)


def _broadcast_descriptor_list(
    descriptors: list[TransferDescriptor] | None,
    *,
    src: int,
    info: _GroupInfo,
) -> list[TransferDescriptor]:
    objects = [descriptors]
    backend = str(torch.distributed.get_backend(info.group)).lower()
    if backend == "nccl":
        torch.distributed.broadcast_object_list(
            objects,
            src=src,
            group=info.group,
            device=local_device(),
        )
    else:
        torch.distributed.broadcast_object_list(
            objects,
            src=src,
            group=info.group,
        )
    result = objects[0]
    if not isinstance(result, list) or len(result) != info.world_size:
        raise RuntimeError("Typed scatter received an invalid descriptor list")
    return cast(list[TransferDescriptor], result)


def _all_gather_descriptors(
    descriptor: TransferDescriptor,
    *,
    info: _GroupInfo,
) -> list[TransferDescriptor]:
    descriptors: list[TransferDescriptor | None] = [None] * info.world_size
    torch.distributed.all_gather_object(
        descriptors,
        descriptor,
        group=info.group,
    )
    if any(item is None for item in descriptors):
        raise RuntimeError("Typed gather received an empty descriptor")
    return cast(list[TransferDescriptor], descriptors)


def _p2p_transfer_tensor(
    tensor: torch.Tensor,
    receive_copies: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    receiving: bool,
    info: _GroupInfo,
) -> torch.Tensor:
    if not receiving and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    backend = str(torch.distributed.get_backend(info.group)).lower()
    if backend != "nccl" or tensor.device.type == "cuda":
        return tensor
    if receiving:
        staging = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device=local_device(),
        )
        receive_copies.append((tensor, staging))
        return staging
    return tensor.to(local_device())


def _wait_p2p_ops(
    operations: list[torch.distributed.P2POp],
    receive_copies: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    requests = torch.distributed.batch_isend_irecv(operations)
    for request in requests:
        request.wait()
    for target, staging in receive_copies:
        target.copy_(staging.cpu())


def _broadcast_transfer_tensor(
    tensor: torch.Tensor,
    *,
    src: int,
    info: _GroupInfo,
) -> None:
    """Broadcast one payload, staging CPU tensors for an NCCL group."""

    backend = str(torch.distributed.get_backend(info.group)).lower()
    if backend != "nccl" or tensor.device.type == "cuda":
        torch.distributed.broadcast(tensor, src=src, group=info.group)
        return

    staging = (
        tensor.to(local_device())
        if info.global_rank == src
        else torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device=local_device(),
        )
    )
    torch.distributed.broadcast(staging, src=src, group=info.group)
    if info.global_rank != src:
        tensor.copy_(staging.cpu())


def _workload_tensors(
    value: _WorkloadValue | _KeyValue,
) -> tuple[torch.Tensor, ...]:
    return _transfer_tensors(value)


def _check_identical_layout(
    value: _WorkloadValue,
    operation: str,
    info: _GroupInfo,
) -> None:
    descriptors = _all_gather_descriptors(describe_value(value), info=info)
    if any(descriptor != descriptors[0] for descriptor in descriptors[1:]):
        raise ValueError(
            f"{operation} requires identical shapes and value metadata on "
            "every process-group rank"
        )


def _all_gather_tensor(
    value: torch.Tensor,
    *,
    info: _GroupInfo,
) -> list[torch.Tensor]:
    backend = str(torch.distributed.get_backend(info.group)).lower()
    if backend == "nccl" and value.device.type == "cpu":
        staging = value.to(local_device())
        gathered_staging = [
            torch.empty_like(staging) for _ in range(info.world_size)
        ]
        torch.distributed.all_gather(
            gathered_staging,
            staging,
            group=info.group,
        )
        return [tensor.cpu() for tensor in gathered_staging]
    gathered = [torch.empty_like(value) for _ in range(info.world_size)]
    torch.distributed.all_gather(gathered, value, group=info.group)
    return gathered
