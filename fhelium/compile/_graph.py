"""Route prepared callable specializations through CUDA Graph segments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch

from fhelium.serialization import ValueEnvelope
from fhelium.values import TensorResident

if TYPE_CHECKING:
    from ._preparation import Specialization
    from ._specialization import CallSignature


def _context() -> tuple[object, ...]:
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.get_autocast_dtype("cuda"),
        torch.is_autocast_cache_enabled(),
    )


def _tensors(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, TensorResident):
        return tuple(
            ValueEnvelope.from_value(value, compact_keys=False).tensors.values()
        )
    return ()


@dataclass
class _Entry:
    specialization: Specialization
    context: tuple[object, ...]
    state: str = "new"
    device: torch.device | None = None
    invoke: Any = None
    prepared_calls: int = 0
    first_execute_seconds: float | None = None
    capture_attempt_seconds: float = 0.0
    fallback_reason: str | None = None

    def fallback(self, reason: str) -> None:
        self.state = "fallback"
        self.fallback_reason = reason


class GraphStats(TypedDict):
    """Observations of one prepared specialization in one execution context."""

    signature: CallSignature
    device: str | None
    context: dict[str, object]
    state: str
    prepared_calls: int
    replays: int
    first_execute_seconds: float | None
    capture_attempt_seconds: float
    fallback_reason: str | None
    capture_stats: dict[str, object] | None
    segments: list[dict[str, float | str]]


class _GraphRoute:
    """Select Graph-segment or prepared execution for each specialization and context.

    The owning callable serializes execute and clear. With Graphs enabled, a specialization whose Tensor inputs share one CUDA device runs on Runners that capture the schedule's Graph segments: the first execution of a segment runs it, the next captures it, and later executions replay it (see :mod:`fhelium.backend._planning`). With ``"auto"``, each segment compares the device time of its first execution with the median of the three replays after its capture and keeps its graph only when the replays are faster; otherwise it releases the graph and runs directly from then on. Each grad, inference and CUDA autocast context has its own Runners and captures. CPU inputs, inputs that require gradients, and a failure before the first capture continue as prepared calls. Graph results are the Runners' output storage, returned without a copy; a result the caller still references when the next execution of the same Runner would overwrite it is first moved to fresh storage, so results stay valid for as long as the caller keeps them.
    """

    def __init__(
        self,
        graph: Literal["auto"] | bool = "auto",
    ) -> None:
        if graph != "auto" and type(graph) is not bool:
            raise ValueError("graph must be 'auto', True, or False")
        self.graph: Literal["auto"] | bool = graph
        self._entries: dict[tuple[int, tuple[object, ...]], _Entry] = {}

    def execute(
        self, specialization: Specialization, values: tuple[object, ...]
    ) -> object:
        context = _context()
        key = (id(specialization), context)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = _Entry(specialization, context)
            self._admit(entry, values)
        if entry.state == "captured":
            return specialization._execute(values, entry.invoke)
        if entry.state == "warming":
            started = time.perf_counter()
            try:
                result = specialization._execute(values, entry.invoke)
            except Exception as error:
                entry.fallback(f"{type(error).__name__}: {error}")
                self._release(entry)
            else:
                entry.capture_attempt_seconds += time.perf_counter() - started
                if self._capture_stats(entry)["captures"]:
                    entry.state = "captured"
                return result
        started = time.perf_counter()
        result = specialization._execute(values)
        entry.prepared_calls += 1
        if entry.first_execute_seconds is None:
            entry.first_execute_seconds = time.perf_counter() - started
        return result

    def _admit(self, entry: _Entry, values: tuple[object, ...]) -> None:
        if self.graph is False:
            entry.state = "disabled"
            return
        tensors = [t for value in values for t in _tensors(value)]
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            entry.fallback(
                "Graph execution requires Tensor inputs on one CUDA device"
            )
            return
        entry.device = next(iter(devices))
        if entry.device.type != "cuda":
            entry.fallback("Prepared execution on a non-CUDA device")
            return
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            entry.fallback("Autograd outputs require prepared execution")
            return
        entry.invoke = entry.specialization.executable._graph_invoke(
            entry.context, adaptive=self.graph == "auto"
        )
        entry.state = "warming"

    def _capture_stats(self, entry: _Entry) -> dict[str, int]:
        return entry.specialization.executable.graph_stats(entry.context)

    def _release(self, entry: _Entry) -> None:
        entry.specialization.executable.release_graphs(entry.context)
        entry.invoke = None

    def invalidate(self, specialization: Specialization) -> None:
        """Release the Graph routes of a specialization whose materials were rebound."""
        for key in tuple(self._entries):
            if key[0] == id(specialization):
                self._release(self._entries.pop(key))

    def clear(self) -> None:
        """Release retained Graphs and reset statistics."""
        for entry in self._entries.values():
            self._release(entry)
        self._entries.clear()

    def stats(self) -> tuple[GraphStats, ...]:
        """Return per-context counts and host-observed preparation costs.

        ``capture_attempt_seconds`` is the host time of the Graph executions up to and including the first capture. ``capture_stats`` reports the Runners' segment, capture, replay and variant counts, their staged input copies and detached results, their Graph pool reservation, arena requirement and output region bytes. ``segments`` lists each Graph segment's node count, capture costs and execution choice (see :meth:`~fhelium.backend.execution.ProgramExecutable.segment_stats`).
        """
        records = []
        for entry in self._entries.values():
            capture = (
                self._capture_stats(entry) if entry.invoke is not None else None
            )
            records.append(
                {
                    "signature": entry.specialization.signature,
                    "device": str(entry.device)
                    if entry.device is not None
                    else None,
                    "context": {
                        "grad_enabled": entry.context[0],
                        "inference_mode": entry.context[1],
                        "autocast_enabled": entry.context[2],
                        "autocast_dtype": str(entry.context[3]),
                        "autocast_cache_enabled": entry.context[4],
                    },
                    "state": entry.state,
                    "prepared_calls": entry.prepared_calls,
                    "replays": capture["replays"] if capture else 0,
                    "first_execute_seconds": entry.first_execute_seconds,
                    "capture_attempt_seconds": entry.capture_attempt_seconds,
                    "fallback_reason": entry.fallback_reason,
                    "capture_stats": dict(capture) if capture else None,
                    "segments": entry.specialization.executable.segment_stats(
                        entry.context
                    )
                    if entry.invoke is not None
                    else [],
                }
            )
        return tuple(records)  # type: ignore[return-value]
