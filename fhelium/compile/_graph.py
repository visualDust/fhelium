"""Reuse CUDA Graphs for prepared callable specializations."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

from fhelium.runtime.cuda_graph import CudaGraphPool, CudaGraphProgram
from fhelium.serialization import ValueEnvelope
from fhelium.values import TensorResident

if TYPE_CHECKING:
    from ._preparation import Specialization


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
        return tuple(ValueEnvelope.from_value(value).tensors.values())
    return ()


def _input_device(inputs: tuple[object, ...]) -> torch.device:
    devices = {tensor.device for value in inputs for tensor in _tensors(value)}
    if not devices:
        raise ValueError("Graph execution requires tensor inputs")
    if len(devices) != 1:
        raise ValueError("Graph inputs must share one CUDA device")
    return next(iter(devices))


@dataclass(frozen=True)
class _OutputTree:
    kind: str
    value: object = None
    children: tuple[_OutputTree, ...] = ()

    @classmethod
    def split(
        cls,
        value: object,
        leaves: list[object],
        positions: dict[int, int] | None = None,
    ) -> _OutputTree:
        if positions is None:
            positions = {}
        if isinstance(value, (torch.Tensor, TensorResident)):
            identity = id(value)
            if identity not in positions:
                positions[identity] = len(leaves)
                leaves.append(value)
            return cls("leaf", positions[identity])
        if isinstance(value, (list, tuple)):
            return cls(
                "list" if isinstance(value, list) else "tuple",
                children=tuple(
                    cls.split(item, leaves, positions) for item in value
                ),
            )
        if isinstance(value, dict):
            return cls(
                "dict",
                tuple(value),
                tuple(
                    cls.split(item, leaves, positions)
                    for item in value.values()
                ),
            )
        if value is None or type(value) in (bool, int, float, complex, str):
            return cls("literal", value)
        raise TypeError(f"Unsupported Graph output {type(value).__name__}")

    def restore(self, leaves: tuple[object, ...]) -> object:
        if self.kind == "leaf":
            return leaves[cast(int, self.value)]
        if self.kind == "literal":
            return self.value
        values = [child.restore(leaves) for child in self.children]
        if self.kind == "dict":
            return dict(
                zip(cast(tuple[object, ...], self.value), values, strict=True)
            )
        return values if self.kind == "list" else tuple(values)

    def flatten(self, value: object) -> tuple[object, ...]:
        leaves: list[object] = []
        actual = self.split(value, leaves)
        # repr also compares immutable NaN literals without NaN != NaN.
        if repr(actual) != repr(self):
            raise ValueError(
                "Output structure or literals changed during capture"
            )
        return tuple(leaves)


@dataclass
class _Entry:
    specialization: Specialization
    context: tuple[object, ...]
    state: str = "new"
    device: torch.device | None = None
    positions: tuple[int, ...] = ()
    output_tree: _OutputTree | None = None
    program: CudaGraphProgram | None = None
    prepared_calls: int = 0
    replays: int = 0
    first_execute_seconds: float | None = None
    capture_attempt_seconds: float = 0.0
    fallback_reason: str | None = None

    def fallback(self, reason: str) -> None:
        self.state = "fallback"
        self.fallback_reason = reason


class _GraphExecutor:
    """Manage caller-thread capture and device-local shared graph pools.

    The owning callable serializes execute and clear. A specialization first runs its prepared executable; its next invocation captures in the caller's grad, inference, and CUDA autocast context. Separate contexts retain separate graphs. Prepared execution preserves autograd when outputs require gradients. Replays return owned output storage by default; borrowed storage remains valid until that graph replays or the executor is cleared.
    """

    def __init__(
        self,
        graph: Literal["auto"] | bool = "auto",
        copy_output: bool = True,
    ) -> None:
        if graph != "auto" and type(graph) is not bool:
            raise ValueError("graph must be 'auto', True, or False")
        self.graph = graph
        self.copy_output = copy_output
        self._entries: dict[tuple[int, tuple[object, ...]], _Entry] = {}
        self._pools: dict[torch.device, CudaGraphPool] = {}

    def execute(
        self, specialization: Specialization, values: tuple[object, ...]
    ) -> object:
        context = _context()
        key = (id(specialization), context)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(specialization, context)
            self._entries[key] = entry
        if entry.state == "captured":
            return self._replay(entry, values)
        if entry.state == "warmed":
            started = time.perf_counter()
            try:
                self._capture(entry, values)
            except Exception as error:
                entry.fallback(f"{type(error).__name__}: {error}")
            finally:
                entry.capture_attempt_seconds = time.perf_counter() - started
            if entry.state == "captured":
                return self._replay(entry, values)
        started = time.perf_counter()
        result = specialization._execute(values)
        entry.prepared_calls += 1
        if entry.first_execute_seconds is None:
            entry.first_execute_seconds = time.perf_counter() - started
        if entry.state == "new":
            self._warm(entry, values, result)
        return result

    def _warm(
        self, entry: _Entry, values: tuple[object, ...], result: object
    ) -> None:
        if self.graph is False:
            entry.state = "disabled"
            return
        try:
            entry.positions = tuple(
                index
                for index, value in enumerate(values)
                if isinstance(value, (torch.Tensor, TensorResident))
            )
            inputs = tuple(values[index] for index in entry.positions)
            entry.device = _input_device(inputs)
            if entry.device.type != "cuda":
                entry.fallback("Prepared execution on a non-CUDA device")
                return
            leaves: list[object] = []
            entry.output_tree = _OutputTree.split(result, leaves)
            if any(t.requires_grad for leaf in leaves for t in _tensors(leaf)):
                entry.fallback("Autograd outputs require prepared execution")
                return
            entry.state = "warmed"
        except Exception as error:
            entry.fallback(f"{type(error).__name__}: {error}")

    def _capture(self, entry: _Entry, values: tuple[object, ...]) -> None:
        assert entry.device is not None and entry.output_tree is not None
        pool = self._pools.get(entry.device)
        if pool is None:
            pool = CudaGraphPool(entry.device)
            self._pools[entry.device] = pool
        # Retain scalar constants, but not the caller's dynamic input storage.
        constants = tuple(
            None if index in entry.positions else value
            for index, value in enumerate(values)
        )
        execute = entry.specialization._execute
        output_tree = entry.output_tree
        positions = entry.positions

        def evaluate(*inputs: object) -> tuple[object, ...]:
            arguments = list(constants)
            for index, value in zip(positions, inputs, strict=True):
                arguments[index] = value
            return output_tree.flatten(execute(tuple(arguments)))

        entry.program = CudaGraphProgram.capture(
            evaluate,
            example_inputs=tuple(values[index] for index in positions),
            warmup=1,
            pool=pool,
        )
        entry.state = "captured"

    def _replay(self, entry: _Entry, values: tuple[object, ...]) -> object:
        assert entry.program is not None and entry.output_tree is not None
        leaves = entry.program.replay(
            *(values[index] for index in entry.positions),
            copy_output=self.copy_output,
        )
        entry.replays += 1
        return entry.output_tree.restore(leaves)

    def invalidate(self, specialization: Specialization) -> None:
        """Release graphs whose material pointers belong to an older binding."""
        for key in tuple(self._entries):
            if key[0] == id(specialization):
                del self._entries[key]

    def clear(self) -> None:
        """Close retained graphs before their shared pools and reset statistics."""
        for entry in self._entries.values():
            if entry.program is not None:
                entry.program.close()
        for pool in self._pools.values():
            pool.close()
        self._entries.clear()
        self._pools.clear()

    def stats(self) -> tuple[dict[str, object], ...]:
        """Return per-context counts and host-observed preparation costs.

        Prepared-call counts cover user invocations; capture statistics report the runtime's additional warmup and validation replay. First-execution and capture-attempt durations are host wall times. Runtime capture memory fields are device allocator snapshots.
        """
        return tuple(
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
                "replays": entry.replays,
                "first_execute_seconds": entry.first_execute_seconds,
                "capture_attempt_seconds": entry.capture_attempt_seconds,
                "fallback_reason": entry.fallback_reason,
                "capture_stats": (
                    asdict(entry.program.stats)
                    if entry.program is not None
                    else None
                ),
            }
            for entry in self._entries.values()
        )
