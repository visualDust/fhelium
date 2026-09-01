"""Observe host CPU and process-visible CUDA topology."""

from __future__ import annotations

import platform as platform_module
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch


@dataclass(frozen=True)
class CpuTopology:
    """Describe the host CPU model, processor counts, and package count."""

    architecture: str
    model: str
    logical_processor_count: int | None
    physical_core_count: int | None
    package_count: int | None

    def __post_init__(self) -> None:
        for name in ("architecture", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CpuTopology {name} must be non-empty")
        for name in (
            "logical_processor_count",
            "physical_core_count",
            "package_count",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    f"CpuTopology {name} must be a positive integer when available"
                )

    @classmethod
    def probe(cls) -> CpuTopology:
        """Observe CPU topology through psutil and platform facilities."""

        model = platform_module.processor()
        cpuinfo_model = ""
        package_ids: set[str] = set()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for record in (
                cpuinfo.read_text(encoding="utf-8", errors="replace")
                .strip()
                .split("\n\n")
            ):
                fields = {
                    line.partition(":")[0].strip().lower(): line.partition(":")[
                        2
                    ].strip()
                    for line in record.splitlines()
                    if ":" in line
                }
                cpuinfo_model = cpuinfo_model or fields.get("model name", "")
                package_id = fields.get("physical id")
                if package_id is not None:
                    package_ids.add(package_id)

        model = cpuinfo_model or model
        package_count = len(package_ids) or None
        if platform_module.system() == "Darwin":

            def sysctl(name: str) -> str:
                try:
                    completed = subprocess.run(
                        ["sysctl", "-n", name],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                except (OSError, subprocess.SubprocessError):
                    return ""
                if completed.returncode != 0:
                    return ""
                return completed.stdout.strip()

            model = (
                sysctl("machdep.cpu.brand_string")
                or sysctl("hw.model")
                or model
            )
            packages = sysctl("hw.packages")
            package_count = int(packages) if packages.isdigit() else None

        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        return cls(
            architecture=platform_module.machine() or "unknown",
            model=model or platform_module.machine() or "unknown",
            logical_processor_count=(None if logical is None else int(logical)),
            physical_core_count=(None if physical is None else int(physical)),
            package_count=package_count,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation of the CPU topology."""

        return {
            "architecture": self.architecture,
            "model": self.model,
            "logical_processor_count": self.logical_processor_count,
            "physical_core_count": self.physical_core_count,
            "package_count": self.package_count,
        }


@dataclass(frozen=True)
class CudaDeviceInfo:
    """Describe one process-visible CUDA device and its hardware properties."""

    device: torch.device
    name: str
    uuid: str
    pci_domain_id: int
    pci_bus_id: int
    pci_device_id: int
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    multiprocessor_count: int
    warp_size: int
    l2_cache_bytes: int
    memory_bus_width_bits: int
    core_clock_khz: int
    memory_clock_khz: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int
    shared_memory_per_block_bytes: int
    shared_memory_per_multiprocessor_bytes: int

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                "CudaDeviceInfo device must be an indexed CUDA device"
            )
        object.__setattr__(self, "device", device)
        for name in ("name", "uuid"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CudaDeviceInfo {name} must be non-empty")
        if (
            not isinstance(self.compute_capability, tuple)
            or len(self.compute_capability) != 2
            or any(
                type(component) is not int or component < 0
                for component in self.compute_capability
            )
        ):
            raise ValueError(
                "CudaDeviceInfo compute_capability must contain two "
                "non-negative integers"
            )
        for name in (
            "pci_domain_id",
            "pci_bus_id",
            "pci_device_id",
            "total_memory_bytes",
            "multiprocessor_count",
            "warp_size",
            "l2_cache_bytes",
            "memory_bus_width_bits",
            "core_clock_khz",
            "memory_clock_khz",
            "max_threads_per_block",
            "max_threads_per_multiprocessor",
            "shared_memory_per_block_bytes",
            "shared_memory_per_multiprocessor_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"CudaDeviceInfo {name} must be a non-negative integer"
                )
        for name in (
            "total_memory_bytes",
            "multiprocessor_count",
            "warp_size",
        ):
            if getattr(self, name) == 0:
                raise ValueError(f"CudaDeviceInfo {name} must be positive")

    @classmethod
    def probe(cls, index: int) -> CudaDeviceInfo:
        """Observe one CUDA device by its process-visible index."""

        properties = torch.cuda.get_device_properties(index)
        return cls(
            device=torch.device("cuda", index),
            name=str(properties.name),
            uuid=str(properties.uuid),
            pci_domain_id=int(properties.pci_domain_id),
            pci_bus_id=int(properties.pci_bus_id),
            pci_device_id=int(properties.pci_device_id),
            compute_capability=(
                int(properties.major),
                int(properties.minor),
            ),
            total_memory_bytes=int(properties.total_memory),
            multiprocessor_count=int(properties.multi_processor_count),
            warp_size=int(properties.warp_size),
            l2_cache_bytes=int(properties.L2_cache_size),
            memory_bus_width_bits=int(properties.memory_bus_width),
            core_clock_khz=int(properties.clock_rate),
            memory_clock_khz=int(properties.memory_clock_rate),
            max_threads_per_block=int(properties.max_threads_per_block),
            max_threads_per_multiprocessor=int(
                properties.max_threads_per_multi_processor
            ),
            shared_memory_per_block_bytes=int(
                properties.shared_memory_per_block
            ),
            shared_memory_per_multiprocessor_bytes=int(
                properties.shared_memory_per_multiprocessor
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation of the CUDA device."""

        return {
            "device": str(self.device),
            "index": self.device.index,
            "name": self.name,
            "uuid": self.uuid,
            "pci_domain_id": self.pci_domain_id,
            "pci_bus_id": self.pci_bus_id,
            "pci_device_id": self.pci_device_id,
            "compute_capability": self.compute_capability,
            "total_memory_bytes": self.total_memory_bytes,
            "multiprocessor_count": self.multiprocessor_count,
            "warp_size": self.warp_size,
            "l2_cache_bytes": self.l2_cache_bytes,
            "memory_bus_width_bits": self.memory_bus_width_bits,
            "core_clock_khz": self.core_clock_khz,
            "memory_clock_khz": self.memory_clock_khz,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_per_multiprocessor": (
                self.max_threads_per_multiprocessor
            ),
            "shared_memory_per_block_bytes": (
                self.shared_memory_per_block_bytes
            ),
            "shared_memory_per_multiprocessor_bytes": (
                self.shared_memory_per_multiprocessor_bytes
            ),
        }


@dataclass(frozen=True)
class CudaTopology:
    """Describe process-visible CUDA devices and directional peer access."""

    devices: tuple[CudaDeviceInfo, ...]
    peer_access: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        devices = tuple(self.devices)
        peer_access = tuple(tuple(row) for row in self.peer_access)
        if not all(isinstance(device, CudaDeviceInfo) for device in devices):
            raise TypeError(
                "CudaTopology devices must contain CudaDeviceInfo values"
            )
        for index, device in enumerate(devices):
            if device.device != torch.device("cuda", index):
                raise ValueError(
                    "CudaTopology devices must follow process-visible CUDA "
                    "index order"
                )
        if len(peer_access) != len(devices):
            raise ValueError(
                "CudaTopology peer_access must have one row per device"
            )
        for source, row in enumerate(peer_access):
            if len(row) != len(devices):
                raise ValueError(
                    "CudaTopology peer_access must be a square matrix"
                )
            if any(type(value) is not bool for value in row):
                raise TypeError(
                    "CudaTopology peer_access entries must be bool values"
                )
            if not row[source]:
                raise ValueError(
                    "CudaTopology peer_access diagonal entries must be True"
                )
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "peer_access", peer_access)

    @classmethod
    def probe(cls) -> CudaTopology:
        """Observe CUDA inventory and peer access without bandwidth tests."""

        if not torch.cuda.is_available():
            return cls(devices=(), peer_access=())
        devices = tuple(
            CudaDeviceInfo.probe(index)
            for index in range(torch.cuda.device_count())
        )
        peer_access = tuple(
            tuple(
                True
                if source == destination
                else bool(
                    torch.cuda.can_device_access_peer(source, destination)
                )
                for destination in range(len(devices))
            )
            for source in range(len(devices))
        )
        return cls(devices=devices, peer_access=peer_access)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation of the CUDA topology."""

        return {
            "devices": [device.as_dict() for device in self.devices],
            "peer_access": [list(row) for row in self.peer_access],
        }


__all__ = [
    "CpuTopology",
    "CudaDeviceInfo",
    "CudaTopology",
]
