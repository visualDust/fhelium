"""Registered Tensor execution for value placement transfers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.backend.implementation import OperationInvocation
from fhelium.backend.resources import BoundResource, ResourceRequirement
from fhelium.ir.dialects import memory

from .resources import DEVICE_RESOURCE_KIND


@dataclass(frozen=True)
class TorchMemoryTransferImplementation:
    """Move one Tensor payload to a caller-bound device and memory space."""

    name: str = "torch-memory-transfer"
    operation_types: tuple[type[memory.TransferOp], ...] = (memory.TransferOp,)
    supports_in_place: bool = False

    def resource_requirements(
        self, invocation: OperationInvocation
    ) -> tuple[ResourceRequirement, ...]:
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        /,
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        if in_place:
            raise ValueError(
                "Memory transfer does not support in-place execution"
            )
        if len(inputs) != 1 or len(resources) != 1:
            raise ValueError(
                "Memory transfer requires one Tensor and one target-device resource"
            )
        target = resources[0]
        if target.kind != DEVICE_RESOURCE_KIND or not isinstance(
            target.value, torch.device
        ):
            raise TypeError(
                "Memory transfer target must be an execution-device resource"
            )
        memory_space = invocation.attributes.get("memory_space", "default")
        if memory_space not in {"default", "pageable_host", "pinned_host"}:
            raise ValueError(
                f"Memory transfer has unsupported memory space {memory_space!r}"
            )
        target_device = target.value
        if memory_space != "default" and target_device.type != "cpu":
            raise ValueError(
                f"Memory space {memory_space!r} requires a CPU target device"
            )

        source = inputs[0]
        if memory_space == "pinned_host":
            host = source if source.device.type == "cpu" else source.to("cpu")
            result = host if host.is_pinned() else host.pin_memory()
        elif memory_space == "pageable_host":
            host = source if source.device.type == "cpu" else source.to("cpu")
            result = (
                torch.empty_like(host, device="cpu", pin_memory=False).copy_(
                    host
                )
                if host.is_pinned()
                else host
            )
        else:
            result = source.to(target_device)
        return (result,)


__all__ = ["TorchMemoryTransferImplementation"]
