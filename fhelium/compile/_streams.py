"""Order prepared resources across caller and preparation CUDA streams."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import torch

from fhelium.serialization import ValueEnvelope
from fhelium.values import TensorResident


def _tensors(value: object) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            yield value
    elif isinstance(value, TensorResident):
        for tensor in ValueEnvelope.from_value(value).tensors.values():
            yield from _tensors(tensor)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _collect(values: Iterable[object]) -> tuple[torch.Tensor, ...]:
    return tuple(
        {
            id(tensor): tensor for value in values for tensor in _tensors(value)
        }.values()
    )


def _record(
    tensors: Iterable[torch.Tensor],
) -> dict[torch.device, torch.cuda.Event]:
    events = {}
    for device in dict.fromkeys(tensor.device for tensor in tensors):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        events[device] = event
    return events


@dataclass
class PreparedStreams:
    """Publish prepared material readiness and track its consumer streams.

    Record this object after preparation has queued its device work. Each new consumer stream waits for readiness before execution and registers material storage with the CUDA allocator. Stream references remain live with this object; allocator stream tracking protects queued material uses when the specialization is cleared. Calls to use are serialized by the owning callable.
    """

    _events: dict[torch.device, torch.cuda.Event]
    _materials: tuple[torch.Tensor, ...]
    _consumers: dict[tuple[torch.device, int], torch.cuda.Stream] = field(
        default_factory=dict
    )

    @classmethod
    def record(
        cls,
        arguments: Iterable[object],
        material_bindings: Iterable[object],
    ) -> PreparedStreams:
        """Record preparation completion on argument and material devices."""
        materials = _collect(material_bindings)
        return cls(_record((*_collect(arguments), *materials)), materials)

    def fork(self) -> PreparedStreams:
        """Retain readiness and materials for an independently linked callable."""
        return PreparedStreams(dict(self._events), self._materials)

    def use(self) -> None:
        """Order first use on each consumer stream and track material lifetimes."""
        for device, event in self._events.items():
            stream = torch.cuda.current_stream(device)
            identity = (device, stream.cuda_stream)
            if identity in self._consumers:
                continue
            stream.wait_event(event)
            for tensor in self._materials:
                if tensor.device == device:
                    tensor.record_stream(stream)
            self._consumers[identity] = stream
