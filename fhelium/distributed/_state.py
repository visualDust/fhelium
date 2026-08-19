"""Process-group initialization and rank-local device helpers."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch


def _environment_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _environment_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _environment_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def local_device() -> torch.device:
    """Return rank-local device metadata selected for the current process.

    ``init`` selects the CUDA device identified by ``LOCAL_RANK``. The helper
    reads the current CUDA device when CUDA is available and otherwise returns
    CPU; it is available before process-group initialization.

    Returns:
        The current CUDA device when CUDA is available, otherwise the CPU
        device.  World size does not affect the result.
    """

    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def init(
    *,
    backend: str | None = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Any | None = None,
    pg_options: Any | None = None,
    device_id: torch.device | int | None = None,
) -> None:
    """Initialize the PyTorch default process group and local CUDA device.

    The helper uses PyTorch's standard process group and the ``torchrun`` rank
    environment. A direct world-size-one program uses a local ``HashStore`` while
    retaining ordinary PyTorch collective and asynchronous ``Work`` behavior.
    ``rank`` is the global rank in the newly created default process group;
    initialization completes synchronously and returns ``None``.

    Args:
        backend: Process-group backend.  Defaults to ``"nccl"`` when CUDA is
            available and ``"gloo"`` otherwise.
        init_method: Optional PyTorch rendezvous URL.  When omitted for a
            direct world-size-one launch, a local ``HashStore`` is used unless
            ``store`` is supplied.
        timeout: Optional process-group operation timeout forwarded to
            :func:`torch.distributed.init_process_group`.
        world_size: Number of processes in the default group.  ``-1`` resolves
            ``WORLD_SIZE`` from the environment.
        rank: Global rank in the default group.  ``-1`` resolves ``RANK`` from
            the environment.
        store: Optional rendezvous key-value store forwarded to PyTorch.
        pg_options: Optional backend-specific process-group options forwarded
            to PyTorch.
        device_id: Rank-local device.  When omitted with CUDA available,
            ``cuda:LOCAL_RANK`` is selected.  A CUDA device is required when
            CUDA is available; only a CPU device is accepted otherwise.

    Returns:
        None.  If a process group is already initialized, the selected CUDA
        device is still applied and process-group initialization is skipped.

    Raises:
        ValueError: If ``device_id`` is incompatible with CUDA availability.
        RuntimeError: If ``torch.distributed`` is unavailable or PyTorch
            cannot initialize the requested process group.
    """

    local_rank = _environment_local_rank()
    selected_device: torch.device | None
    if torch.cuda.is_available():
        selected_device = (
            torch.device(device_id)
            if device_id is not None
            else torch.device("cuda", local_rank)
        )
        if selected_device.type != "cuda":
            raise ValueError(
                "CUDA is available, so dist.init device_id must identify a "
                f"CUDA device; got {selected_device}"
            )
        torch.cuda.set_device(selected_device)
    else:
        selected_device = None
        if device_id is not None and torch.device(device_id).type != "cpu":
            raise ValueError(
                "CUDA is unavailable, so dist.init device_id must be CPU"
            )

    if torch.distributed.is_initialized():
        return
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")

    selected_backend = backend or (
        "nccl" if torch.cuda.is_available() else "gloo"
    )
    resolved_rank = _environment_rank() if rank == -1 else rank
    resolved_world_size = (
        _environment_world_size() if world_size == -1 else world_size
    )

    kwargs: dict[str, Any] = {
        "backend": selected_backend,
        "rank": resolved_rank,
        "world_size": resolved_world_size,
    }
    if init_method is not None:
        kwargs["init_method"] = init_method
    if timeout is not None:
        kwargs["timeout"] = timeout
    if pg_options is not None:
        kwargs["pg_options"] = pg_options
    if selected_device is not None and selected_backend == "nccl":
        kwargs["device_id"] = selected_device

    if store is not None:
        kwargs["store"] = store
    elif resolved_world_size == 1 and init_method is None:
        # A local store avoids requiring MASTER_ADDR/MASTER_PORT for an
        # ordinary single-process program while still creating a real
        # ProcessGroup and real completed Work objects for async collectives.
        kwargs["store"] = torch.distributed.HashStore()  # pyright: ignore[reportPrivateImportUsage]

    torch.distributed.init_process_group(**kwargs)


def shutdown(
    group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """Destroy an initialized process group.

    Args:
        group: Process group to destroy.  ``None`` selects the PyTorch default
            process group; this is a group object, not a rank identifier.

    Returns:
        None.  An unavailable or uninitialized distributed runtime, including
        a direct process before ``init``, is a no-op.  A world-size-one group
        is destroyed in the same way as any other initialized group.  No
        asynchronous :class:`torch.distributed.Work` handle is returned.
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group(group)
