"""Cross-platform snapshots of host and CUDA memory availability."""

from __future__ import annotations

import time
from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True)
class MemorySnapshot:
    """Record memory capacity and availability at one point in time.

    CPU snapshots use :func:`psutil.virtual_memory`, whose ``available`` value
    estimates memory that the operating system can provide without swapping.
    CUDA snapshots use :func:`torch.cuda.mem_get_info`, whose free value covers
    the whole selected device, including use by other processes and non-PyTorch
    allocators. CUDA snapshots additionally report the current process's
    PyTorch caching-allocator allocation and reservation counters.
    ``observed_at_ns`` is a Unix timestamp in nanoseconds recorded after the
    counters are read.
    """

    device: torch.device
    observed_at_ns: int
    capacity_bytes: int
    available_bytes: int
    torch_allocated_bytes: int | None = None
    torch_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        try:
            device = torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"MemorySnapshot device is invalid: {self.device!r}"
            ) from error
        if device.type == "cpu":
            device = torch.device("cpu")
            if (
                self.torch_allocated_bytes is not None
                or self.torch_reserved_bytes is not None
            ):
                raise ValueError(
                    "CPU MemorySnapshot does not carry CUDA allocator counters"
                )
        elif device.type == "cuda":
            if device.index is None:
                raise ValueError(
                    "MemorySnapshot CUDA device must have a concrete index; "
                    "use MemorySnapshot.read('cuda') to resolve the current device"
                )
            if (
                self.torch_allocated_bytes is None
                or self.torch_reserved_bytes is None
            ):
                raise ValueError(
                    "CUDA MemorySnapshot requires PyTorch allocation and reservation counters"
                )
        else:
            raise ValueError(
                "MemorySnapshot supports CPU and CUDA devices, got "
                f"{device.type!r}"
            )
        object.__setattr__(self, "device", device)

        if type(self.observed_at_ns) is not int or self.observed_at_ns <= 0:
            raise ValueError("MemorySnapshot observed_at_ns must be positive")
        if type(self.capacity_bytes) is not int or self.capacity_bytes <= 0:
            raise ValueError("MemorySnapshot capacity_bytes must be positive")
        if (
            type(self.available_bytes) is not int
            or not 0 <= self.available_bytes <= self.capacity_bytes
        ):
            raise ValueError(
                "MemorySnapshot available_bytes must be between zero and capacity_bytes"
            )
        for name in ("torch_allocated_bytes", "torch_reserved_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    f"MemorySnapshot {name} must be a non-negative integer"
                )

    @classmethod
    def read(
        cls,
        device: str | torch.device = "cpu",
    ) -> MemorySnapshot:
        """Read current host or CUDA memory counters for ``device``."""

        selected = torch.device(device)
        if selected.type == "cpu":
            memory = psutil.virtual_memory()
            return cls(
                device=torch.device("cpu"),
                observed_at_ns=time.time_ns(),
                capacity_bytes=int(memory.total),
                available_bytes=int(memory.available),
            )
        if selected.type != "cuda":
            raise ValueError(
                "MemorySnapshot.read supports CPU and CUDA devices, got "
                f"{selected.type!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA memory read requires an available CUDA device"
            )
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        selected = torch.device("cuda", index)
        with torch.cuda.device(selected):
            available, capacity = torch.cuda.mem_get_info(selected)
            allocated = torch.cuda.memory_allocated(selected)
            reserved = torch.cuda.memory_reserved(selected)
        return cls(
            device=selected,
            observed_at_ns=time.time_ns(),
            capacity_bytes=int(capacity),
            available_bytes=int(available),
            torch_allocated_bytes=int(allocated),
            torch_reserved_bytes=int(reserved),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation of the snapshot."""

        return {
            "device": str(self.device),
            "observed_at_ns": self.observed_at_ns,
            "capacity_bytes": self.capacity_bytes,
            "available_bytes": self.available_bytes,
            "torch_allocated_bytes": self.torch_allocated_bytes,
            "torch_reserved_bytes": self.torch_reserved_bytes,
        }


__all__ = ["MemorySnapshot"]
