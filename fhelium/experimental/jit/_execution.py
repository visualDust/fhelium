"""Inputs supplied by a JIT Session to Backend coverage and build calls."""

from __future__ import annotations

import platform as platform_module
from dataclasses import dataclass

import torch

from fhelium.runtime import CudaTopology


@dataclass(frozen=True)
class ExecutionInputs:
    """Present one Session device and its observed execution inventory."""

    device: torch.device
    platform: str
    machine: str
    python_implementation: str
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_topology: CudaTopology | None

    def __post_init__(self) -> None:
        device = torch.device(self.device)
        if device.type == "cpu":
            device = torch.device("cpu")
        elif device.type == "cuda":
            if device.index is None:
                raise ValueError("JIT CUDA device must have a concrete index")
            topology = self.cuda_topology
            if topology is None or device.index >= len(topology.devices):
                raise ValueError(
                    "JIT CUDA device is absent from the observed CUDA topology"
                )
        else:
            raise ValueError(
                f"JIT supports CPU and CUDA devices, got {device.type!r}"
            )
        object.__setattr__(self, "device", device)

    def as_dict(self) -> dict[str, object]:
        """Return the selected device and observed build inventory."""

        device_info = None
        if (
            self.device.type == "cuda"
            and self.cuda_topology is not None
            and self.device.index is not None
        ):
            device_info = self.cuda_topology.devices[self.device.index]
        return {
            "platform": self.platform,
            "machine": self.machine,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_topology": (
                None
                if self.cuda_topology is None
                else self.cuda_topology.as_dict()
            ),
            "device": str(self.device),
            "device_type": self.device.type,
            "device_name": (
                self.machine if device_info is None else device_info.name
            ),
            "compute_capability": (
                None if device_info is None else device_info.compute_capability
            ),
            "multiprocessor_count": (
                None
                if device_info is None
                else device_info.multiprocessor_count
            ),
            "warp_size": (
                None if device_info is None else device_info.warp_size
            ),
        }


def _observe_execution_inputs(
    device: str | torch.device,
) -> ExecutionInputs:
    """Read the inventory needed to build for one Session device."""

    selected = torch.device(device)
    if selected.type == "cuda" and selected.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "JIT CUDA execution requires an available CUDA device"
            )
        selected = torch.device("cuda", torch.cuda.current_device())
    cuda_topology = CudaTopology.probe() if selected.type == "cuda" else None
    return ExecutionInputs(
        device=selected,
        platform=platform_module.platform(),
        machine=platform_module.machine() or "unknown",
        python_implementation=platform_module.python_implementation(),
        python_version=platform_module.python_version(),
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        cuda_topology=cuda_topology,
    )
