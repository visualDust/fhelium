"""Measure dispatch and completed execution without synchronizing CUDA."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from fhelium.serialization import ValueEnvelope
from fhelium.values import TensorResident


@dataclass(frozen=True)
class _Token:
    host_start: float
    event: torch.cuda.Event | None
    device: torch.device | None
    host_complete: bool


@dataclass
class _Totals:
    host_dispatch_seconds: float = 0.0
    host_dispatch_count: int = 0
    completed_seconds: float = 0.0
    completed_count: int = 0
    completion_unavailable_count: int = 0


@dataclass(frozen=True)
class _Pending:
    start: torch.cuda.Event
    end: torch.cuda.Event
    groups: tuple[str, ...]


class ExecutionTiming:
    """Accumulate input-device execution timings for serialized caller use.

    CPU calls complete when Python returns. Calls whose inputs contain one CUDA device use timing events on that device's caller-selected stream; event durations become available after a nonblocking completion query. Inputs spanning multiple CUDA devices or other accelerators receive host-dispatch timings only. The measurement covers the caller stream, including work joined onto it; independent streams and devices held only by closures are outside the measured scope. Completed events are released on finish or stats. No timing operation synchronizes a device or changes thread settings.
    """

    def __init__(self) -> None:
        self._totals = {
            name: _Totals() for name in ("compiled", "compiled_steady")
        }
        self._pending: list[_Pending] = []

    def start(self, values: tuple[object, ...]) -> _Token:
        """Record a start event when input residency identifies one CUDA device."""
        devices: set[torch.device] = set()
        for value in values:
            if isinstance(value, torch.Tensor):
                devices.add(value.device)
            elif isinstance(value, TensorResident):
                devices.update(
                    tensor.device
                    for tensor in ValueEnvelope.from_value(
                        value, compact_keys=False
                    ).tensors.values()
                )
        cuda_devices = {device for device in devices if device.type == "cuda"}
        device = next(iter(cuda_devices)) if len(cuda_devices) == 1 else None
        event = None
        if device is not None:
            with torch.cuda.device(device):
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(device))
        return _Token(
            time.perf_counter(),
            event,
            device,
            all(device.type == "cpu" for device in devices),
        )

    def finish(
        self,
        token: _Token,
        *,
        steady: bool,
    ) -> None:
        """Record dispatch cost and queue CUDA completion accounting."""
        seconds = time.perf_counter() - token.host_start
        groups = ("compiled", "compiled_steady") if steady else ("compiled",)
        if token.event is not None:
            with torch.cuda.device(token.device):
                end = torch.cuda.Event(enable_timing=True)
                end.record(torch.cuda.current_stream(token.device))
            self._pending.append(_Pending(token.event, end, groups))
        for name in groups:
            totals = self._totals[name]
            totals.host_dispatch_seconds += seconds
            totals.host_dispatch_count += 1
            if token.host_complete:
                totals.completed_seconds += seconds
                totals.completed_count += 1
            elif token.event is None:
                totals.completion_unavailable_count += 1
        self._collect()

    def _collect(self) -> None:
        pending = []
        for record in self._pending:
            if not record.end.query():
                pending.append(record)
                continue
            seconds = record.start.elapsed_time(record.end) / 1000.0
            for name in record.groups:
                totals = self._totals[name]
                totals.completed_seconds += seconds
                totals.completed_count += 1
        self._pending = pending

    def stats(self) -> dict[str, dict[str, float | int]]:
        """Return fresh counters, collecting events already completed on-device."""
        self._collect()
        return {
            name: {
                "host_dispatch_seconds": totals.host_dispatch_seconds,
                "host_dispatch_count": totals.host_dispatch_count,
                "completed_seconds": totals.completed_seconds,
                "completed_count": totals.completed_count,
                "pending_count": sum(
                    name in record.groups for record in self._pending
                ),
                "completion_unavailable_count": totals.completion_unavailable_count,
            }
            for name, totals in self._totals.items()
        }

    def clear(self) -> None:
        """Discard counters and pending timing events without waiting for work."""
        self._pending.clear()
        self._totals = {name: _Totals() for name in self._totals}
