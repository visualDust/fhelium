"""Canonical memory-tier locations for managed value residency.

A residency location identifies one storage class and, for CUDA storage, one
physical device index. This immutable pair is the location key used by plans,
snapshots, and accounting records.
"""

from __future__ import annotations

from dataclasses import dataclass
from re import fullmatch
from typing import Literal

import torch

ResidencyLocationKind = Literal["pageable-host", "pinned-host", "cuda"]
_MAX_CUDA_DEVICE_INDEX = 127


def _normalize_cuda_device(device: torch.device | str) -> torch.device:
    if isinstance(device, str):
        match = fullmatch(r"cuda:(0|[1-9][0-9]*)", device)
        if match is None:
            raise ValueError(
                "CUDA residency location requires an indexed "
                "canonical device such as cuda:0"
            )
        index = int(match.group(1))
        if index > _MAX_CUDA_DEVICE_INDEX:
            raise ValueError(
                "CUDA residency location index must be between 0 and 127"
            )
        normalized = torch.device("cuda", index)
    else:
        normalized = torch.device(device)
    if normalized.type != "cuda" or normalized.index is None:
        raise ValueError(
            "CUDA residency location requires an indexed device such as cuda:0"
        )
    if not 0 <= normalized.index <= _MAX_CUDA_DEVICE_INDEX:
        raise ValueError(
            "CUDA residency location index must be between 0 and 127"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ResidencyLocation:
    """Canonical identity of one managed memory tier.

    Host locations always store the canonical unindexed ``cpu`` device.  An
    indexed CPU spelling such as ``cpu:0`` is accepted but normalized, which
    prevents two identities for the same host tier.  CUDA locations require an
    device index because an ambient current device is not a stable
    placement identity.

    Args:
        kind: Storage class represented by the location.
        device: CPU for a host tier, or an indexed CUDA device.
    """

    kind: ResidencyLocationKind
    device: torch.device

    def __post_init__(self) -> None:
        if self.kind in ("pageable-host", "pinned-host"):
            device = torch.device(self.device)
            if device.type != "cpu":
                raise ValueError(f"{self.kind} location requires a CPU device")
            object.__setattr__(self, "device", torch.device("cpu"))
            return
        if self.kind == "cuda":
            device = _normalize_cuda_device(self.device)
            object.__setattr__(self, "device", device)
            return
        raise ValueError(f"Unsupported residency location kind: {self.kind!r}")

    @property
    def name(self) -> str:
        """Return the stable diagnostic name of this location."""

        if self.kind == "cuda":
            return str(self.device)
        return self.kind

    def __str__(self) -> str:
        return self.name


PAGEABLE_HOST = ResidencyLocation("pageable-host", torch.device("cpu"))
PINNED_HOST = ResidencyLocation("pinned-host", torch.device("cpu"))


def cuda_location(device: torch.device | str) -> ResidencyLocation:
    """Return the canonical location for one indexed CUDA device.

    Args:
        device: CUDA device such as ``"cuda:0"``.  Unindexed ``"cuda"`` is
            rejected rather than resolved through process-global device state.

    Returns:
        Immutable CUDA residency location.
    """

    return ResidencyLocation("cuda", _normalize_cuda_device(device))


__all__ = [
    "PAGEABLE_HOST",
    "PINNED_HOST",
    "ResidencyLocation",
    "ResidencyLocationKind",
    "cuda_location",
]
