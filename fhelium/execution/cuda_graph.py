"""CUDA Graph capture and replay for fixed-schedule FHElium evaluators.

``CudaGraphProgram`` stages dynamic positional tensor and exact-value inputs
through reusable fixed-address buffers. Callers bind encryption, keys,
resources, and schedule before capture; captured callables use fixed Python
control flow.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar, cast, overload

import torch

from fhelium.core import TensorResident
from fhelium.errors import (
    CudaGraphCaptureError,
    CudaGraphInputError,
    ExecutionError,
    ExecutionInputError,
)
from fhelium.execution.buffer import CopyHandle, ReusableValueBuffer
from fhelium.execution.signature import (
    TensorSignature,
    ValueSignature,
    ValueTreeSignature,
)
from fhelium.serialization import ValueEnvelope

_OutputT = TypeVar("_OutputT")


@dataclass(frozen=True)
class CudaGraphCaptureStats:
    """One-time construction costs and device memory after capture.

    Args:
        warmup_iterations: Number of uncaptured warmup evaluations requested.
        warmup_seconds: Host-observed total warmup duration.
        capture_seconds: Host-observed CUDA Graph capture duration.
        first_replay_seconds: Host-observed duration of the synchronized first
            native graph replay used for validation.
        memory_allocated_bytes: CUDA allocator bytes live after capture.
        memory_reserved_bytes: CUDA allocator reserved bytes after capture.
    """

    warmup_iterations: int
    warmup_seconds: float
    capture_seconds: float
    first_replay_seconds: float
    memory_allocated_bytes: int
    memory_reserved_bytes: int


class CudaGraphProgram(Generic[_OutputT]):
    """CUDA Graph adapter for an eager callable with reusable dynamic inputs.

    The original callable remains independently valid for eager execution.
    Internally, dynamic arguments use a
    :class:`~fhelium.execution.ReusableValueBuffer`; CUDA Graph kernels capture
    the buffer's fixed target addresses while replay changes only its payload.

    Construct programs with :meth:`capture`; direct construction is not public.
    Parameters bound by a closure, :class:`functools.partial`, or a callable
    object are static program state. Positional ``example_inputs`` define the
    reusable dynamic argument tree. Supported leaves are tensors and exact
    serializable FHElium values nested in lists, tuples, and dictionaries.

    Dynamic source values may later reside on CPU or CUDA as long as their
    device-independent structure and exact metadata match capture. CPU-to-CUDA
    overlap requires pinned sources and the advanced :meth:`copy_inputs_from`
    plus :meth:`replay_prepared` path. Ordinary :meth:`replay` remains a
    same-call convenience.

    Replay returns a borrowed output tree whose storage is overwritten by the
    next replay. Pass ``copy_output=True`` when the result must outlive that
    replay. One program instance owns one input/output storage set and does not
    execute replays concurrently; create independent instances for concurrent
    workers.

    Examples:
        A normal evaluator is still callable eagerly::

            schedule = partial(
                matrix_vector,
                engine=engine,
                diagonals=diagonals,
                rotation_keys=rotation_keys,
            )
            eager_output = schedule(source)

        Capture it for repeated execution::

            program = CudaGraphProgram.capture(
                schedule,
                example_inputs=(prototype_source,),
            )
            borrowed = program.replay(next_source)
            owned = program.replay(next_source, copy_output=True)

        Separate transfer from replay with a caller-owned stream::

            copied = program.copy_inputs_from(
                pinned_cpu_source,
                stream=transfer_stream,
                non_blocking=True,
            )
            with torch.cuda.stream(compute_stream):
                output = program.replay_prepared(copy_handle=copied)
    """

    def __init__(
        self,
        *,
        function: Callable[..., _OutputT],
        device: torch.device,
        graph: torch.cuda.CUDAGraph,
        input_buffer: ReusableValueBuffer[tuple[object, ...]],
        static_inputs: tuple[object, ...],
        output: _OutputT,
        stats: CudaGraphCaptureStats,
    ) -> None:
        self._function: Callable[..., _OutputT] | None = function
        self._device = device
        self._graph: torch.cuda.CUDAGraph | None = graph
        self._input_buffer: ReusableValueBuffer[tuple[object, ...]] | None = (
            input_buffer
        )
        self._static_inputs: tuple[object, ...] | None = static_inputs
        self._output: _OutputT | None = output
        self._stats = stats
        self._latest_copy: CopyHandle | None = None
        self._last_replay_event: torch.cuda.Event | None = None
        self._closed = False

    @overload
    @classmethod
    def capture(
        cls,
        function: Callable[..., _OutputT],
        *,
        example_inputs: Sequence[object],
        warmup: int = 3,
        check_input_liveness: bool = True,
    ) -> CudaGraphProgram[_OutputT]: ...

    @overload
    @classmethod
    def capture(
        cls,
        function: None = None,
        *,
        example_inputs: Sequence[object],
        warmup: int = 3,
        check_input_liveness: bool = True,
    ) -> Callable[[Callable[..., _OutputT]], CudaGraphProgram[_OutputT]]: ...

    @classmethod
    def capture(
        cls,
        function: Callable[..., _OutputT] | None = None,
        *,
        example_inputs: Sequence[object],
        warmup: int = 3,
        check_input_liveness: bool = True,
    ) -> (
        CudaGraphProgram[_OutputT]
        | Callable[[Callable[..., _OutputT]], CudaGraphProgram[_OutputT]]
    ):
        """Warm up and capture one deterministic evaluator schedule.

        ``example_inputs`` defines both the dynamic positional argument tree and
        the fixed CUDA target residency. Warmup uses fresh reusable buffers, so
        backend JIT and lazy materialization happen without mutating the buffer
        retained by the captured graph.

        Bind static parameters through a closure, ``functools.partial``, or a
        callable object. This method intentionally does not accept dynamic
        keyword arguments or arbitrary Python control objects. Use a positional
        adapter for a dynamic keyword-only tensor or exact value.

        Passing ``function`` directly is the canonical API. Omitting it returns
        a decorator shorthand that captures at function-definition time::

            @CudaGraphProgram.capture(example_inputs=(prototype,))
            def captured(source):
                return evaluator(source)

        The decorated name is the ready :class:`CudaGraphProgram`, not the
        original Python callable. All CUDA state, static resources, and example
        inputs must therefore already exist when the definition executes.

        Args:
            function: Optional callable evaluator whose Python control flow and
                CUDA work are fixed at capture. Output must be a CUDA tensor,
                exact FHElium value, nested list/tuple/dict of those leaves, or
                ``None``. Omit it only for the decorator shorthand.
            example_inputs: CUDA-resident positional prototypes. Their
                structure and exact metadata specialize this program.
            warmup: Number of fresh-buffer side-stream evaluations before
                capture.
            check_input_liveness: Forwarded to ``torch.cuda.graph``.

        Returns:
            A ready CUDA Graph program with one reusable input buffer, or a
            decorator producing one when ``function`` is omitted.

        Raises:
            TypeError: If ``function`` or an input/output leaf is unsupported.
            ValueError: If warmup is negative, examples have no tensor leaves,
                are not on one CUDA device, or representative storage aliases.
            CudaGraphCaptureError: If warmup, capture, output validation, or the
                first native replay fails.
        """

        if function is None:

            def decorate(
                decorated_function: Callable[..., _OutputT],
            ) -> CudaGraphProgram[_OutputT]:
                result = cls.capture(
                    decorated_function,
                    example_inputs=example_inputs,
                    warmup=warmup,
                    check_input_liveness=check_input_liveness,
                )
                assert isinstance(result, CudaGraphProgram)
                return result

            return decorate
        if not callable(function):
            raise TypeError("function must be callable")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        inputs = tuple(example_inputs)
        function_name = getattr(function, "__qualname__", repr(function))
        input_buffer = ReusableValueBuffer.like(inputs)
        device = input_buffer.device
        if device.type != "cuda":
            input_buffer.close()
            raise ValueError(
                "CUDA Graph example inputs must target one CUDA device; "
                f"got device={device}"
            )
        static_inputs = input_buffer.value

        warmup_start = time.perf_counter()
        try:
            with torch.cuda.device(device):
                warmup_stream = torch.cuda.Stream(device=device)
                warmup_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(warmup_stream):
                    for _ in range(warmup):
                        warm_buffer = ReusableValueBuffer.like(
                            inputs,
                            device=device,
                        )
                        warm_inputs = warm_buffer.value
                        warm_output = function(*warm_inputs)
                        _validate_output_tree(warm_output, device=device)
                        warmup_stream.synchronize()
                        warm_buffer.close()
                        del warm_output, warm_inputs, warm_buffer
                torch.cuda.current_stream(device).wait_stream(warmup_stream)
                torch.cuda.synchronize(device)
        except Exception as error:
            input_buffer.close()
            raise CudaGraphCaptureError(
                stage="warmup",
                function_name=function_name,
                detail=str(error),
            ) from error
        warmup_seconds = time.perf_counter() - warmup_start

        graph = torch.cuda.CUDAGraph()
        capture_start = time.perf_counter()
        try:
            with torch.cuda.device(device):
                cuda_graph_context: Any = torch.cuda.graph
                with cuda_graph_context(
                    graph,
                    check_input_liveness=check_input_liveness,
                ):
                    output = function(*static_inputs)
                _validate_output_tree(output, device=device)
                torch.cuda.synchronize(device)
        except Exception as error:
            graph.reset()
            input_buffer.close()
            raise CudaGraphCaptureError(
                stage="capture",
                function_name=function_name,
                detail=str(error),
            ) from error
        capture_seconds = time.perf_counter() - capture_start

        first_replay_start = time.perf_counter()
        try:
            with torch.cuda.device(device):
                graph.replay()
                torch.cuda.synchronize(device)
        except Exception as error:
            graph.reset()
            input_buffer.close()
            raise CudaGraphCaptureError(
                stage="first replay",
                function_name=function_name,
                detail=str(error),
            ) from error
        first_replay_seconds = time.perf_counter() - first_replay_start

        stats = CudaGraphCaptureStats(
            warmup_iterations=warmup,
            warmup_seconds=warmup_seconds,
            capture_seconds=capture_seconds,
            first_replay_seconds=first_replay_seconds,
            memory_allocated_bytes=torch.cuda.memory_allocated(device),
            memory_reserved_bytes=torch.cuda.memory_reserved(device),
        )
        return cls(
            function=function,
            device=device,
            graph=graph,
            input_buffer=input_buffer,
            static_inputs=static_inputs,
            output=output,
            stats=stats,
        )

    @property
    def device(self) -> torch.device:
        """CUDA device on which this program was captured."""

        return self._device

    @property
    def input_signature(self) -> ValueTreeSignature:
        """Device-independent signature of the positional argument tuple."""

        return self._require_input_buffer().signature

    @property
    def input_nbytes(self) -> int:
        """Logical bytes in the program's fixed dynamic-input storage."""

        return self._require_input_buffer().nbytes

    @property
    def stats(self) -> CudaGraphCaptureStats:
        """One-time construction timings and post-capture memory snapshots.

        Memory fields are process allocator snapshots, not graph-exclusive
        incremental footprints.
        """

        return self._stats

    @property
    def output(self) -> _OutputT:
        """Borrowed output tree retained by the captured graph.

        Its storage is overwritten by the next replay. Use
        ``replay(..., copy_output=True)`` when independent ownership is needed.
        """

        self._require_open()
        return cast(_OutputT, self._output)

    @property
    def cuda_graph(self) -> torch.cuda.CUDAGraph:
        """Underlying PyTorch CUDA Graph for low-level inspection.

        Calling its ``replay()`` directly bypasses input validation, payload
        copying, transfer-event waits, and overwrite protection. It reuses the
        payload currently held by the input buffer.
        """

        self._require_open()
        assert self._graph is not None
        return self._graph

    def copy_inputs_from(
        self,
        *inputs: object,
        stream: torch.cuda.Stream | None = None,
        non_blocking: bool = True,
        wait_for: torch.cuda.Event | Sequence[torch.cuda.Event] | None = None,
    ) -> CopyHandle:
        """Copy matching positional inputs without replaying the graph.

        This is the advanced prefetch mechanism. The caller decides when and on
        which stream to submit the copy. If a previous replay was launched, its
        completion event is automatically added as an overwrite dependency, so
        a transfer stream cannot replace input storage still read by the graph.

        Source values may live on pinned CPU or CUDA. Actual asynchronous H2D
        overlap requires pinned CPU tensors.

        Returns:
            A :class:`CopyHandle` accepted by :meth:`replay_prepared`.

        Raises:
            CudaGraphInputError: If structure or exact state differs from
                capture.
            ExecutionError: If the program is closed.
        """

        buffer = self._require_input_buffer()
        dependencies = _normalize_events(wait_for)
        if self._last_replay_event is not None:
            dependencies += (self._last_replay_event,)
        try:
            copied = buffer.copy_from(
                tuple(inputs),
                stream=stream,
                non_blocking=non_blocking,
                wait_for=dependencies,
            )
        except ExecutionInputError as error:
            raise CudaGraphInputError(str(error)) from error
        self._latest_copy = copied
        return copied

    def replay_prepared(
        self,
        *,
        copy_handle: CopyHandle | None = None,
        copy_output: bool = False,
        synchronize: bool = False,
    ) -> _OutputT:
        """Replay the payload already held by the input buffer.

        Args:
            copy_handle: Optional handle returned by the latest
                :meth:`copy_inputs_from`. The current compute stream waits on
                its completion event without blocking the CPU. Omit this only
                to intentionally replay the buffer's current payload again.
            copy_output: Clone the output after replay instead of returning
                borrowed graph storage.
            synchronize: Synchronize the capture device before returning.

        Returns:
            Borrowed retained output, or an independently owned clone.

        Raises:
            CudaGraphInputError: If ``copy_handle`` belongs to another buffer
                or is not the most recent prepared payload.
            ExecutionError: If the program is closed.
        """

        self._require_open()
        buffer = self._require_input_buffer()
        if copy_handle is None and self._latest_copy is not None:
            raise CudaGraphInputError(
                "replay_prepared must receive the latest CopyHandle while a "
                "prepared input copy is outstanding"
            )
        if copy_handle is not None and copy_handle is not self._latest_copy:
            raise CudaGraphInputError(
                "replay_prepared requires the latest CopyHandle returned by "
                "this CudaGraphProgram"
            )
        assert self._graph is not None
        with torch.cuda.device(self._device):
            compute_stream = torch.cuda.current_stream(self._device)
            if self._last_replay_event is not None:
                wait_event: Any = compute_stream.wait_event
                wait_event(self._last_replay_event)
            if copy_handle is not None:
                try:
                    buffer.wait_for(copy_handle, compute_stream)
                except ExecutionInputError as error:
                    raise CudaGraphInputError(str(error)) from error
            self._graph.replay()
            result = cast(
                _OutputT,
                _clone_output_tree(self._output)
                if copy_output
                else self._output,
            )
            replay_complete = torch.cuda.Event()
            replay_complete.record(compute_stream)
            self._last_replay_event = replay_complete
            self._latest_copy = None
            if synchronize:
                torch.cuda.synchronize(self._device)
        return result

    def replay(
        self,
        *inputs: object,
        copy_output: bool = False,
        synchronize: bool = False,
    ) -> _OutputT:
        """Copy matching inputs, replay, and return the graph output.

        This convenience path performs validation, payload copy, and replay on
        the caller's current CUDA stream. For transfer/compute overlap, use
        :meth:`copy_inputs_from` on a transfer stream followed by
        :meth:`replay_prepared` on a compute stream.

        Tensor source devices may differ from capture, but structure, tensor
        topology, and exact FHElium metadata must match. The default output is
        borrowed; ``copy_output=True`` returns independent storage.
        """

        self._require_open()
        with torch.cuda.device(self._device):
            current_stream = torch.cuda.current_stream(self._device)
            copied = self.copy_inputs_from(
                *inputs,
                stream=current_stream,
                non_blocking=True,
            )
            return self.replay_prepared(
                copy_handle=copied,
                copy_output=copy_output,
                synchronize=synchronize,
            )

    def close(self) -> None:
        """Synchronize outstanding work and release graph-owned references.

        Closing is idempotent. The program, borrowed output, input buffer, and
        direct ``cuda_graph`` access must not be used afterward. CUDA caching
        allocator reserved memory may remain available for process reuse.
        """

        if self._closed:
            return
        if self._last_replay_event is not None:
            self._last_replay_event.synchronize()
        if self._graph is not None:
            self._graph.reset()
        if self._input_buffer is not None:
            self._input_buffer.close()
        self._graph = None
        self._output = None
        self._static_inputs = None
        self._input_buffer = None
        self._function = None
        self._latest_copy = None
        self._last_replay_event = None
        self._closed = True

    def __enter__(self) -> CudaGraphProgram[_OutputT]:
        self._require_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise ExecutionError("CUDA Graph program is closed")

    def _require_input_buffer(
        self,
    ) -> ReusableValueBuffer[tuple[object, ...]]:
        self._require_open()
        if self._input_buffer is None:
            raise ExecutionError("CUDA Graph input buffer is unavailable")
        return self._input_buffer


def _normalize_events(
    events: torch.cuda.Event | Sequence[torch.cuda.Event] | None,
) -> tuple[torch.cuda.Event, ...]:
    if events is None:
        return ()
    if isinstance(events, (list, tuple)):
        return tuple(events)
    return (cast(torch.cuda.Event, events),)


def _validate_output_tree(value: object, *, device: torch.device) -> None:
    if value is None:
        return
    if isinstance(value, TensorResident):
        envelope = ValueEnvelope.from_value(value)
        if not envelope.tensors:
            raise TypeError(
                f"CUDA Graph output {type(value).__name__} has no tensors"
            )
        for tensor in envelope.tensors.values():
            if tensor.device != device:
                raise TypeError(
                    "CUDA Graph output value is not on the capture device"
                )
        return
    if isinstance(value, torch.Tensor):
        if value.device != device:
            raise TypeError("CUDA Graph output tensor is not on capture device")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_output_tree(item, device=device)
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_output_tree(item, device=device)
        return
    raise TypeError(
        "CUDA Graph outputs must be tensors, TensorResident values, nested "
        f"containers, or None (actual={type(value).__name__})"
    )


def _clone_output_tree(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, TensorResident):
        envelope = ValueEnvelope.from_value(value)
        tensors = {
            name: envelope.tensors[name].clone(
                memory_format=torch.preserve_format
            )
            for name in sorted(envelope.tensors)
        }
        return replace(envelope, tensors=tensors).to_value()
    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, list):
        return [_clone_output_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_output_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_output_tree(item) for key, item in value.items()}
    raise TypeError(f"Unsupported CUDA Graph output {type(value).__name__}")


__all__ = [
    "CudaGraphCaptureStats",
    "CudaGraphProgram",
    "TensorSignature",
    "ValueSignature",
    "ValueTreeSignature",
]
