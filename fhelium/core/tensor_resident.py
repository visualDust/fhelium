"""Tensor storage, device transfer, and byte accounting for exact values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch


class TensorResident(ABC):
    """An exact FHElium value whose declared tensor fields move together.

    Subclasses enumerate their direct tensor fields and reconstruct the same
    exact value state around replacement tensors. The capability exposes one
    common device, logical payload bytes, unique backing-storage bytes, and
    functional movement for one value.

    ``TensorResident`` values expose ordinary PyTorch tensors. Functional
    movement creates independent storage when ``copy=True`` and leaves the
    source object accessible to its caller.
    """

    @property
    @abstractmethod
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the direct tensor fields owned by this value."""

    @abstractmethod
    def _with_resident_tensors(self, tensors: tuple[torch.Tensor, ...]) -> Self:
        """Reconstruct this exact value around replacement tensors."""

    @property
    def device(self) -> torch.device:
        """Common device of every declared tensor field."""

        tensors = self._resident_tensors
        if not tensors:
            raise RuntimeError(
                f"{type(self).__name__} has no resident tensor storage"
            )
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors[1:]):
            raise RuntimeError(
                f"{type(self).__name__} tensors do not share one device"
            )
        return device

    @property
    def nbytes(self) -> int:
        """Logical tensor payload bytes, counting every declared tensor field."""

        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._resident_tensors
        )

    @property
    def storage_nbytes(self) -> int:
        """Bytes in unique backing storages referenced by declared tensors.

        This differs from :attr:`nbytes` when tensor fields share storage or a
        view references a backing allocation larger than its logical payload.
        It remains a tensor-storage measure, not CUDA allocator reservation or
        process memory reported by NVML.
        """

        storages: dict[tuple[torch.device, int], int] = {}
        for tensor in self._resident_tensors:
            storage = tensor.untyped_storage()
            storage_nbytes = storage.nbytes()
            if storage_nbytes == 0:
                continue
            key = (tensor.device, storage.data_ptr())
            existing = storages.get(key)
            if existing is not None and existing != storage_nbytes:
                raise RuntimeError(
                    "TensorResident fields report inconsistent sizes for one "
                    "backing storage"
                )
            storages[key] = storage_nbytes
        return sum(storages.values())

    @property
    def is_cpu(self) -> bool:
        """Whether all declared tensors reside on CPU."""

        return self.device.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        """Whether all declared tensors reside on one CUDA device."""

        return self.device.type == "cuda"

    @property
    def is_pinned(self) -> bool:
        """Whether every declared tensor uses pinned CPU storage.

        CUDA values return ``False``. A CPU value whose fields mix pageable and
        pinned storage also returns ``False``; managed residency validates and
        rejects such mixed materializations rather than treating them as
        pageable.
        """

        tensors = self._resident_tensors
        return self.device.type == "cpu" and all(
            tensor.is_pinned() for tensor in tensors
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self:
        """Functionally move all declared tensors to one PyTorch device."""

        target = torch.device(device)
        source = self._resident_tensors
        moved = tuple(
            tensor.to(target, non_blocking=non_blocking, copy=copy)
            for tensor in source
        )
        if all(after is before for before, after in zip(source, moved)):
            return self
        return self._with_resident_tensors(moved)

    def cpu(self, *, copy: bool = False) -> Self:
        """Return this exact value in ordinary pageable CPU storage."""

        return self.to("cpu", copy=copy or self.is_pinned)

    def pin_memory(self, *, copy: bool = False) -> Self:
        """Return this exact value backed by pinned CPU tensor storage.

        Args:
            copy: Create independent pinned storage even when every source
                tensor is already pinned. With ``False``, an already uniformly
                pinned CPU value is returned unchanged.
        """

        source = self._resident_tensors
        if (
            self.device.type == "cpu"
            and all(tensor.is_pinned() for tensor in source)
            and not copy
        ):
            return self

        pinned = tuple(_pin_tensor(tensor) for tensor in source)
        return self._with_resident_tensors(pinned)


def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Clone one tensor into independent pinned CPU storage."""

    with torch.no_grad():
        try:
            pinned = torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=True,
            )
            if pinned.device.type != "cpu" or not pinned.is_pinned():
                raise RuntimeError("PyTorch did not create pinned CPU storage")
        except RuntimeError as error:
            raise RuntimeError(
                "Pinned host storage is unavailable in the current PyTorch "
                "runtime. Use pageable CPU storage for CPU-only execution, "
                "or install an accelerator-enabled PyTorch build when pinned "
                "host staging is required."
            ) from error
        pinned.copy_(tensor.detach(), non_blocking=False)
        pinned.requires_grad_(tensor.requires_grad)
    return pinned


def _storage_keys(value: TensorResident) -> frozenset[tuple[torch.device, int]]:
    """Return non-empty backing-storage identities for manager alias checks."""

    return frozenset(
        (tensor.device, tensor.untyped_storage().data_ptr())
        for tensor in value._resident_tensors
        if tensor.untyped_storage().nbytes() > 0
    )


def _cpu_pinning_is_uniform(value: TensorResident) -> bool:
    """Whether CPU tensor fields are uniformly pageable or uniformly pinned."""

    if value.device.type != "cpu":
        return True
    pin_states = {tensor.is_pinned() for tensor in value._resident_tensors}
    return len(pin_states) <= 1


__all__ = ["TensorResident"]
