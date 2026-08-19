"""Reusable fixed-address buffers for exact execution values.

Buffers provide exact fixed-storage data movement. Applications choose payload
identity, movement timing, caching, and prefetch policy.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar, cast

import torch

from fhelium.errors import ExecutionError, ExecutionInputError
from fhelium.execution.signature import (
    ValueSignature,
    ValueTreeSignature,
    _build_value_tree_signature,
    _collect_matching_tensors,
)
from fhelium.serialization import ValueEnvelope

_ValueT = TypeVar("_ValueT")


@dataclass
class _BufferNode:
    kind: str
    tensors: tuple[torch.Tensor, ...] = ()
    tensor_names: tuple[str, ...] = ()
    envelope: ValueEnvelope | None = None
    children: tuple[_BufferNode, ...] = ()
    keys: tuple[object, ...] = ()

    def build_value(self) -> object:
        if self.kind == "tensor":
            return self.tensors[0]
        if self.kind == "value":
            assert self.envelope is not None
            tensors = dict(zip(self.tensor_names, self.tensors, strict=True))
            return replace(self.envelope, tensors=tensors).to_value()
        if self.kind == "list":
            return [child.build_value() for child in self.children]
        if self.kind == "tuple":
            return tuple(child.build_value() for child in self.children)
        if self.kind == "dict":
            return {
                key: child.build_value()
                for key, child in zip(self.keys, self.children, strict=True)
            }
        raise RuntimeError(f"Unknown reusable-buffer node kind {self.kind!r}")

    def tensor_leaves(self) -> tuple[torch.Tensor, ...]:
        if self.kind in {"tensor", "value"}:
            return self.tensors
        return tuple(
            tensor
            for child in self.children
            for tensor in child.tensor_leaves()
        )

    def value_envelopes(self) -> tuple[ValueEnvelope, ...]:
        if self.kind == "value":
            assert self.envelope is not None
            return (self.envelope,)
        if self.kind == "tensor":
            return ()
        return tuple(
            envelope
            for child in self.children
            for envelope in child.value_envelopes()
        )


class CopyHandle:
    """Future-like handle for one copy enqueued into a reusable buffer.

    A handle owns the exact submitted tensor leaves until the CUDA event reports
    completion. This prevents a mutable source tree from releasing or replacing
    pinned host or device storage while an asynchronous copy may still read it.
    ``wait_on`` inserts a stream dependency without blocking the CPU;
    ``synchronize`` blocks the caller.

    ``CopyHandle`` is intentionally not an :mod:`asyncio` Future and is not
    awaitable. A future extension may bridge CUDA events to an async scheduler,
    but the core object currently exposes CUDA stream/event ordering only.

    Instances are returned by :meth:`ReusableValueBuffer.copy_from`; direct
    construction is not part of the public API. Internally, ``event`` tracks
    completion, ``device`` identifies the target, ``source_tensors`` keep the
    submitted storage alive, ``bytes_copied`` records payload size, and
    ``target_token`` binds the handle to its originating buffer.
    """

    def __init__(
        self,
        *,
        event: torch.cuda.Event | None,
        device: torch.device,
        source_tensors: tuple[torch.Tensor, ...],
        bytes_copied: int,
        target_token: object,
    ) -> None:
        self._event = event
        self._device = device
        self._source_tensors = source_tensors
        self._bytes_copied = int(bytes_copied)
        self._target_token = target_token

    @property
    def event(self) -> torch.cuda.Event | None:
        """CUDA completion event, or ``None`` for a synchronous CPU copy."""

        return self._event

    @property
    def device(self) -> torch.device:
        """Target device of the copy."""

        return self._device

    @property
    def bytes_copied(self) -> int:
        """Logical tensor payload bytes submitted by this copy."""

        return self._bytes_copied

    def done(self) -> bool:
        """Return whether the copy has completed without blocking."""

        if self._event is None:
            self._source_tensors = ()
            return True
        complete = bool(self._event.query())
        if complete:
            self._source_tensors = ()
        return complete

    def wait_on(self, stream: torch.cuda.Stream | None = None) -> CopyHandle:
        """Make ``stream`` wait for this copy and return ``self``.

        The default is the current stream on the copy target device. This
        method enqueues an event wait and does not synchronize the CPU.

        Args:
            stream: Consumer CUDA stream. Defaults to the current stream on
                this handle's target device.

        Returns:
            This handle, allowing fluent ordering before consumption.

        Raises:
            ValueError: If the supplied stream belongs to another device.
        """

        if self._event is None:
            return self
        with torch.cuda.device(self._device):
            consumer = (
                torch.cuda.current_stream(self._device)
                if stream is None
                else stream
            )
            if torch.device(consumer.device) != self._device:
                raise ValueError(
                    "CopyHandle consumer stream is on the wrong device: "
                    f"expected={self._device}, actual={consumer.device}"
                )
            wait_event: Any = consumer.wait_event
            wait_event(self._event)
        return self

    def synchronize(self) -> None:
        """Block the CPU until the copy completes, then release the source."""

        if self._event is not None:
            self._event.synchronize()
        self._source_tensors = ()

    def _belongs_to(self, target_token: object) -> bool:
        return self._target_token is target_token


class ReusableValueBuffer(Generic[_ValueT]):
    """Fixed-structure storage whose exact-value payload may be replaced.

    Construct a buffer from a representative tensor/exact-value tree with
    :meth:`like`. The representative defines container structure, cryptographic
    metadata, and tensor topology. The buffer separately owns tensors on one
    target device. :meth:`copy_from` accepts a matching tree on CPU or CUDA,
    validates the complete source before copying anything, and preserves every
    target tensor address.

    The buffer identifies payloads solely by exact structure. Applications or
    extensions associate it with model weights, user keys, request ciphertexts,
    and cache/prefetch/eviction policy.

    The :attr:`value` property reconstructs ordinary tensors and exact FHElium
    values around the fixed storage, so an eager callable can consume it without
    a buffer-specific evaluator API. Do not retain a reconstructed tree as an
    ownership or readiness signal; use the buffer and the :class:`CopyHandle`
    returned by :meth:`copy_from` for lifetime and stream ordering.

    Examples:
        Reuse one CUDA allocation for changing plaintext tiles::

            buffer = ReusableValueBuffer.like(
                prototype_tile,
                device="cuda:0",
            )
            copied = buffer.copy_from(
                pinned_cpu_tile,
                stream=transfer_stream,
                non_blocking=True,
            )

            with torch.cuda.stream(compute_stream):
                copied.wait_on(compute_stream)
                output = evaluate_tile(ciphertext, buffer.value)

        Double buffering is built by creating two independent buffers and
        copying into one while an eager program reads the other. Choosing the
        next tile and recording the prior consumer event remain application
        policy.

    Construct buffers with :meth:`like`; the direct constructor accepts
    internal allocation-tree state and is not part of the public API.
    """

    def __init__(
        self,
        *,
        node: _BufferNode,
        signature: ValueTreeSignature,
        device: torch.device,
        pinned: bool,
    ) -> None:
        self._node: _BufferNode | None = node
        self._signature = signature
        self._device = device
        self._pinned = pinned
        self._target_tensors = node.tensor_leaves()
        self._value_envelopes = node.value_envelopes()
        self._nbytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._target_tensors
        )
        self._target_token = object()
        self._inflight_copies: list[CopyHandle] = []
        self._closed = False

    @classmethod
    def like(
        cls,
        example: _ValueT,
        *,
        device: torch.device | str | None = None,
        pin_memory: bool = False,
    ) -> ReusableValueBuffer[_ValueT]:
        """Allocate independent storage with the structure of ``example``.

        Args:
            example: Tensor/exact-value tree defining structure, metadata, and
                initial payload. Supported containers are lists, tuples, and
                dictionaries.
            device: Target residency. If omitted, all example tensors must
                already share one device, which becomes the target.
            pin_memory: Allocate pinned host tensors. This requires a CPU target
                and is useful before non-blocking host-to-device copies.

        Returns:
            A buffer initialized with a copy of ``example``.

        Raises:
            TypeError: If the tree contains an unsupported leaf.
            ValueError: If there are no tensor leaves, the implicit target is
                ambiguous, ``pin_memory`` is requested for a non-CPU target, or
                representative leaves alias the same storage.
        """

        source_tensors: list[torch.Tensor] = []
        value_envelopes: list[ValueEnvelope] = []
        signature = _build_value_tree_signature(
            example,
            source_tensors,
            value_envelopes,
        )
        if not source_tensors:
            raise ValueError("ReusableValueBuffer requires at least one tensor")
        source_devices = {tensor.device for tensor in source_tensors}
        if device is None:
            if len(source_devices) != 1:
                raise ValueError(
                    "ReusableValueBuffer requires an indexed target device "
                    "when example tensors span multiple devices: "
                    f"devices={sorted(map(str, source_devices))}"
                )
            target = next(iter(source_devices))
        else:
            target = torch.device(device)
        if pin_memory and target.type != "cpu":
            raise ValueError("pin_memory=True requires a CPU target device")
        _reject_aliased_storages(source_tensors)
        node = _allocate_node(
            example,
            signature=signature,
            value_envelopes=iter(value_envelopes),
            target=target,
            pin_memory=pin_memory,
        )
        return cls(
            node=node,
            signature=signature,
            device=target,
            pinned=pin_memory,
        )

    @property
    def signature(self) -> ValueTreeSignature:
        """Device-independent structure and exact-value state of this buffer."""

        return self._signature

    @property
    def device(self) -> torch.device:
        """Target residency of the owned storage."""

        return self._device

    @property
    def is_pinned(self) -> bool:
        """Whether all owned CPU tensors use pinned host memory."""

        return self._pinned

    @property
    def nbytes(self) -> int:
        """Logical bytes in the fixed target tensors."""

        self._require_open()
        return self._nbytes

    @property
    def value(self) -> _ValueT:
        """A fresh ordinary value tree around the fixed target tensors.

        Container and exact-value wrapper objects may be newly reconstructed on
        each access; their tensor addresses remain fixed for the buffer
        lifetime. Payload readiness is governed by the relevant
        :class:`CopyHandle` or caller stream ordering.
        """

        return cast(_ValueT, self._require_open().build_value())

    def copy_from(
        self,
        source: _ValueT,
        *,
        stream: torch.cuda.Stream | None = None,
        non_blocking: bool = True,
        wait_for: torch.cuda.Event | Sequence[torch.cuda.Event] | None = None,
    ) -> CopyHandle:
        """Copy a matching source tree into the fixed target storage.

        Full structural and exact-state validation happens before the first
        ``Tensor.copy_``. Tensor source devices may differ from the target and
        from the representative used by :meth:`like`.

        Args:
            source: Matching tensor/exact-value tree whose payload replaces the
                buffer contents.
            stream: Caller-owned CUDA stream on which target writes are
                enqueued. Defaults to the current target-device stream. CPU
                buffers do not accept a stream.
            non_blocking: Forwarded to ``Tensor.copy_``. Actual asynchronous
                host-to-device overlap requires pinned CPU source tensors.
            wait_for: One event or sequence of events that the copy stream must
                wait for before overwriting this buffer, for example an event
                recorded after the prior eager consumer finished reading it.

        Returns:
            A future-like :class:`CopyHandle`. Keep or wait on this handle
            before consuming the new payload. The buffer also retains in-flight
            handles so dropping the returned Python object cannot prematurely
            release source storage.

        Raises:
            ExecutionInputError: If source structure, metadata, or tensor
                topology differs from the buffer.
            ValueError: If stream/event arguments are incompatible with a CPU
                target or the stream belongs to another CUDA device.
            ExecutionError: If the buffer is closed.

        Notes:
            This method orders writes but cannot infer when an arbitrary eager
            consumer has finished reading the previous payload. Record a CUDA
            event after that consumer and pass it as ``wait_for`` before reusing
            the buffer.
        """

        self._require_open()
        source_tensors: list[torch.Tensor] = []
        _collect_matching_tensors(
            source,
            self._signature,
            source_tensors,
            path="source",
            expected_envelopes=iter(self._value_envelopes),
        )
        if len(source_tensors) != len(self._target_tensors):
            raise RuntimeError("Internal reusable-buffer leaf mismatch")
        self._prune_completed_copies()
        events = _normalize_events(wait_for) + tuple(
            handle.event
            for handle in self._inflight_copies
            if handle.event is not None
        )
        bytes_copied = self._nbytes

        if self._device.type == "cpu":
            if stream is not None:
                raise ValueError(
                    "CPU ReusableValueBuffer does not accept a CUDA stream"
                )
            if events:
                raise ValueError(
                    "CPU ReusableValueBuffer does not accept CUDA events"
                )
            with torch.no_grad():
                for target, actual in zip(
                    self._target_tensors,
                    source_tensors,
                    strict=True,
                ):
                    target.copy_(actual, non_blocking=False)
            return CopyHandle(
                event=None,
                device=self._device,
                source_tensors=(),
                bytes_copied=bytes_copied,
                target_token=self._target_token,
            )

        with torch.cuda.device(self._device):
            copy_stream = (
                torch.cuda.current_stream(self._device)
                if stream is None
                else stream
            )
            if torch.device(copy_stream.device) != self._device:
                raise ValueError(
                    "ReusableValueBuffer copy stream is on the wrong device: "
                    f"expected={self._device}, actual={copy_stream.device}"
                )
            for event in events:
                wait_event: Any = copy_stream.wait_event
                wait_event(event)
            with torch.cuda.stream(copy_stream), torch.no_grad():
                for target, actual in zip(
                    self._target_tensors,
                    source_tensors,
                    strict=True,
                ):
                    target.copy_(actual, non_blocking=non_blocking)
                completion = torch.cuda.Event()
                completion.record(copy_stream)
        handle = CopyHandle(
            event=completion,
            device=self._device,
            source_tensors=tuple(source_tensors),
            bytes_copied=bytes_copied,
            target_token=self._target_token,
        )
        self._inflight_copies.append(handle)
        return handle

    def wait_for(
        self,
        handle: CopyHandle,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Wait on a copy produced by this buffer without blocking the CPU.

        Args:
            handle: Copy handle previously returned by this buffer.
            stream: Consumer CUDA stream, or the current target stream when
                omitted.

        Raises:
            ExecutionInputError: If ``handle`` belongs to another buffer.
        """

        self._require_open()
        if not handle._belongs_to(self._target_token):
            raise ExecutionInputError(
                "CopyHandle belongs to another ReusableValueBuffer"
            )
        handle.wait_on(stream)

    def close(self) -> None:
        """Wait for outstanding copies and release target-storage references.

        Closing is idempotent. The caller remains responsible for ensuring that
        arbitrary eager or captured consumers have finished reading the target
        tensors before closing; the buffer can track its own writes but cannot
        infer external read completion.
        """

        if self._closed:
            return
        for handle in self._inflight_copies:
            handle.synchronize()
        self._inflight_copies.clear()
        self._node = None
        self._target_tensors = ()
        self._value_envelopes = ()
        self._closed = True

    def __enter__(self) -> ReusableValueBuffer[_ValueT]:
        self._require_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _prune_completed_copies(self) -> None:
        self._inflight_copies = [
            handle for handle in self._inflight_copies if not handle.done()
        ]

    def _require_open(self) -> _BufferNode:
        if self._closed or self._node is None:
            raise ExecutionError("ReusableValueBuffer is closed")
        return self._node


def pin_value_tree(value: _ValueT) -> _ValueT:
    """Clone a supported value tree into pinned CPU storage.

    The returned ordinary tree owns pinned tensors and has the same
    device-independent signature as ``value``. This helper chooses no cache or
    persistence policy; callers retain and release the returned tree normally.

    Args:
        value: Supported tensor/exact-value tree to clone. Source leaves may be
            on CPU or CUDA.

    Returns:
        Structurally equivalent ordinary value tree backed by pinned CPU
        tensors.
    """

    buffer = ReusableValueBuffer.like(
        value,
        device="cpu",
        pin_memory=True,
    )
    return buffer.value


def value_tree_nbytes(value: object) -> int:
    """Return logical tensor bytes in one supported execution value tree.

    Args:
        value: Tensor/exact-value tree whose tensor payload is counted.

    Returns:
        Sum of ``numel * element_size`` over every tensor leaf.
    """

    tensors: list[torch.Tensor] = []
    _build_value_tree_signature(value, tensors)
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _allocate_node(
    value: object,
    *,
    signature: ValueTreeSignature,
    value_envelopes: Iterator[ValueEnvelope],
    target: torch.device,
    pin_memory: bool,
) -> _BufferNode:
    if signature.kind == "value":
        envelope = next(value_envelopes)
        leaf = signature.leaf
        if not isinstance(leaf, ValueSignature):
            raise RuntimeError("Value tree has an invalid value signature")
        names = tuple(name for name, _ in leaf.tensors)
        tensors = tuple(
            _clone_tensor(
                envelope.tensors[name],
                target=target,
                pin_memory=pin_memory,
            )
            for name in names
        )
        return _BufferNode(
            kind="value",
            tensors=tensors,
            tensor_names=names,
            envelope=replace(envelope, tensors={}),
        )
    if signature.kind == "tensor":
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Compiled tensor tree changed during allocation")
        return _BufferNode(
            kind="tensor",
            tensors=(
                _clone_tensor(
                    value,
                    target=target,
                    pin_memory=pin_memory,
                ),
            ),
        )
    if signature.kind == "list":
        if not isinstance(value, list):
            raise RuntimeError("Compiled list tree changed during allocation")
        return _BufferNode(
            kind="list",
            children=tuple(
                _allocate_node(
                    item,
                    signature=child,
                    value_envelopes=value_envelopes,
                    target=target,
                    pin_memory=pin_memory,
                )
                for item, child in zip(value, signature.children, strict=True)
            ),
        )
    if signature.kind == "tuple":
        if not isinstance(value, tuple):
            raise RuntimeError("Compiled tuple tree changed during allocation")
        return _BufferNode(
            kind="tuple",
            children=tuple(
                _allocate_node(
                    item,
                    signature=child,
                    value_envelopes=value_envelopes,
                    target=target,
                    pin_memory=pin_memory,
                )
                for item, child in zip(value, signature.children, strict=True)
            ),
        )
    if signature.kind == "dict":
        if not isinstance(value, dict):
            raise RuntimeError(
                "Compiled dictionary tree changed during allocation"
            )
        keys = signature.keys
        return _BufferNode(
            kind="dict",
            children=tuple(
                _allocate_node(
                    value[key],
                    signature=child,
                    value_envelopes=value_envelopes,
                    target=target,
                    pin_memory=pin_memory,
                )
                for key, child in zip(keys, signature.children, strict=True)
            ),
            keys=keys,
        )
    raise RuntimeError(f"Unknown compiled buffer node kind {signature.kind!r}")


def _clone_tensor(
    tensor: torch.Tensor,
    *,
    target: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    with torch.no_grad():
        cloned = tensor.detach().to(
            target,
            copy=True,
            non_blocking=False,
            memory_format=torch.preserve_format,
        )
        if pin_memory and not cloned.is_pinned():
            cloned = cloned.pin_memory()
        cloned.requires_grad_(tensor.requires_grad)
    return cloned


def _normalize_events(
    events: torch.cuda.Event | Sequence[torch.cuda.Event] | None,
) -> tuple[torch.cuda.Event, ...]:
    if events is None:
        return ()
    if isinstance(events, (list, tuple)):
        return tuple(events)
    return (cast(torch.cuda.Event, events),)


def _reject_aliased_storages(tensors: Sequence[torch.Tensor]) -> None:
    identities = [
        (tensor.device, tensor.untyped_storage().data_ptr())
        for tensor in tensors
        if tensor.untyped_storage().nbytes() > 0
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Aliased representative storage is not supported by "
            "ReusableValueBuffer; pass independent values or keep the shared "
            "object outside the reusable payload"
        )


__all__ = [
    "CopyHandle",
    "ReusableValueBuffer",
    "pin_value_tree",
    "value_tree_nbytes",
]
