"""Lower registered operations to exec schedules.

An implementation's ``lower`` receives its Tensor operands as layout-carrying placeholder Tensors and writes ordinary PyTorch and native operator calls on them. :class:`ExecBuilder` records each call as an ``exec`` operation: operators that read placeholders become ``exec.call_op``, view operators become static ``exec.view`` operations, allocation operators become ``exec.alloc``/``exec.zero``, and operators whose Tensor inputs are all link-time constants run once while lowering and become bound materials. Kernel launches, Python calls and mutable resource state are recorded through the builder's methods.

Lowering runs once per operation and operand layout: link lowers a Program, and Eager caches one fragment per operation, attributes and operand layouts. :mod:`fhelium.backend.runner` builds the schedules into libfhelium Runners.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    NoneAttr,
    StringAttr,
    f64,
    i64,
)
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from fhelium.ir.dialects import exec as ex

aten = torch.ops.aten

_ALLOCATIONS: dict[object, bool] = {
    aten.empty.memory_format: False,
    aten.empty_strided.default: False,
    aten.empty_like.default: False,
    aten.new_empty.default: False,
    aten.new_empty_strided.default: False,
    aten.zeros.default: True,
    aten.zeros_like.default: True,
    aten.new_zeros.default: True,
}
"""Allocation operators and whether they zero their result."""

_COPIES = {aten.copy_.default}
_ZEROS = {aten.zero_.default}


class HostScalar(float):
    """Real scalar supplied when a lowered fragment runs.

    Lowering passes it to operators like a Python ``float``; the recorded call reads the fragment input at ``index``.
    """

    index: int

    def __new__(cls, value: float, index: int) -> HostScalar:
        scalar = super().__new__(cls, value)
        scalar.index = index
        return scalar


class HostInt(int):
    """Integer supplied when a lowered fragment runs; see :class:`HostScalar`."""

    index: int

    def __new__(cls, value: int, index: int) -> HostInt:
        scalar = super().__new__(cls, value)
        scalar.index = index
        return scalar


_HOST = (HostScalar, HostInt)


@dataclass(frozen=True)
class Layout:
    """Static Tensor layout: sizes, element strides, dtype and device."""

    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, tensor: torch.Tensor) -> Layout:
        return cls(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )


def layout_key(tensor: torch.Tensor) -> tuple[object, ...]:
    """Return the hashable layout of one runtime Tensor."""
    # Metadata reads bypass Torch function modes such as a default-device
    # context, which otherwise dominate the cost of a per-call cache key.
    with torch._C.DisableTorchFunction():
        return (tensor.shape, tensor.stride(), tensor.dtype, tensor.device)


def _buffer_type(storage: str, tensor: torch.Tensor | None) -> ex.BufferType:
    if tensor is None:
        return ex.buffer_type(storage)
    return _cached_buffer_type(
        storage,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


_BUFFER_TYPES: dict[tuple[object, ...], ex.BufferType] = {}


def _cached_buffer_type(
    storage: str,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: str,
    device: str,
) -> ex.BufferType:
    key = (storage, sizes, strides, dtype, device)
    cached = _BUFFER_TYPES.get(key)
    if cached is None:
        cached = _BUFFER_TYPES[key] = ex.buffer_type(
            storage, sizes, strides, dtype, device
        )
    return cached


TorchConstant = ex.TorchConstant


def _torch_constant(value: object) -> Attribute | None:
    if isinstance(value, torch.dtype):
        return TorchConstant(StringAttr("dtype"), StringAttr(str(value)))
    if isinstance(value, torch.device):
        return TorchConstant(StringAttr("device"), StringAttr(str(value)))
    if isinstance(value, torch.layout):
        return TorchConstant(StringAttr("layout"), StringAttr(str(value)))
    if isinstance(value, torch.memory_format):
        return TorchConstant(
            StringAttr("memory_format"), StringAttr(str(value))
        )
    return None


def torch_constant_value(attribute: TorchConstant) -> object:
    """Decode a PyTorch constant argument."""
    kind, value = attribute.kind.data, attribute.value.data
    if kind == "device":
        return torch.device(value)
    return getattr(torch, value.removeprefix("torch."))


class LoweringError(RuntimeError):
    """A lowered operation reads Tensor data or writes a link-time constant."""


_ACTIVE: ContextVar[ExecBuilder | None] = ContextVar(
    "fhelium_exec_builder", default=None
)


def active_builder() -> ExecBuilder | None:
    """Return the builder recording the current lowering, if any."""
    return _ACTIVE.get()


@contextmanager
def isolated() -> Iterator[None]:
    """Suspend recording of an enclosing lowering, for link-time work or a nested schedule."""
    from torch.utils._python_dispatch import _disable_current_modes

    token = _ACTIVE.set(None)
    try:
        with _disable_current_modes():
            yield
    finally:
        _ACTIVE.reset(token)


class _TraceMode(TorchDispatchMode):
    def __init__(self, builder: ExecBuilder) -> None:
        super().__init__()
        self.builder = builder

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.builder._dispatch(func, args, kwargs or {})


class _HostArgumentMode(TorchFunctionMode):
    """Carry host-scalar markers across PyTorch's argument conversion.

    Operator arguments reach the dispatch mode as plain Python numbers. This mode records which schema arguments held :class:`HostScalar` or :class:`HostInt` values so the dispatched call can restore them.
    """

    def __init__(self, builder: ExecBuilder) -> None:
        super().__init__()
        self.builder = builder

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _ACTIVE.get() is not self.builder or not any(
            isinstance(leaf, _HOST) for leaf in tree_flatten((args, kwargs))[0]
        ):
            return func(*args, **kwargs)
        overload = func
        if isinstance(
            func, torch._ops.OpOverloadPacket
        ) and func.overloads() == ["default"]:
            overload = func.default
        if not isinstance(overload, torch._ops.OpOverload):
            raise LoweringError(
                f"Pass runtime host scalars to operator overloads, not {func}"
            )
        schema = overload._schema
        marked = {
            argument.name: value
            for argument in schema.arguments
            for value in (
                _schema_argument(schema, args, kwargs, argument.name),
            )
            if any(isinstance(leaf, _HOST) for leaf in tree_flatten(value)[0])
        }
        self.builder._pending.append((overload, marked))
        try:
            return overload(*args, **kwargs)
        finally:
            self.builder._pending.pop()


def _schema_argument(schema, args, kwargs, name: str) -> object:
    for index, argument in enumerate(schema.arguments):
        if argument.name != name:
            continue
        if not argument.kwarg_only and index < len(args):
            return args[index]
        if name in kwargs:
            return kwargs[name]
        return argument.default_value
    raise KeyError(name)


def _with_arguments(schema, args, kwargs, values: dict[str, object]):
    args = list(args)
    kwargs = dict(kwargs)
    for index, argument in enumerate(schema.arguments):
        if argument.name not in values:
            continue
        if not argument.kwarg_only and index < len(args):
            args[index] = values[argument.name]
        else:
            kwargs[argument.name] = values[argument.name]
    return tuple(args), kwargs


@dataclass
class ExecBuilder:
    """Record exec operations for operations lowered at one insertion point.

    ``buffers`` maps placeholder Tensors to their SSA values. ``constants`` binds link-time constant symbols to Tensors; ``objects`` binds kernel, callable, resource and operator symbols to Python objects.
    """

    operations: list[Operation] = field(default_factory=list)
    constants: dict[str, torch.Tensor] = field(default_factory=dict)
    objects: dict[str, object] = field(default_factory=dict)
    fake_mode: FakeTensorMode = field(
        default_factory=lambda: FakeTensorMode(allow_non_fake_inputs=False)
    )
    _buffers: dict[int, tuple[torch.Tensor, SSAValue]] = field(
        default_factory=dict
    )
    _constant_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    _host: dict[int, SSAValue] = field(default_factory=dict)
    _symbols: dict[int, str] = field(default_factory=dict)
    _ordinal: int = 0
    regions: tuple[Region, ...] = ()
    """Regions of the operation being lowered."""
    _bodies: list[tuple[Region, tuple[object, ...]]] = field(
        default_factory=list
    )
    _memo: dict[Hashable, Any] = field(default_factory=dict)
    _inline: (
        Callable[[Region, tuple[object, ...]], tuple[object, ...]] | None
    ) = None
    _pending: list[tuple[Any, dict[str, object]]] = field(default_factory=list)
    launches: list[tuple[Any, tuple[object, ...]]] = field(default_factory=list)
    """Recorded kernels with warm signatures: dtypes for Tensor arguments, values otherwise."""
    launch_signatures: dict[Operation, tuple[object, ...]] = field(
        default_factory=dict
    )
    """Warm signature of each recorded ``exec.launch``."""
    destinations: bool = False
    """Record functional operator calls that have a destination-passing overload as an allocation and a call of that overload, so that a planned schedule can place their results in its arena."""

    # Placeholders -------------------------------------------------------

    def placeholder(
        self,
        value: SSAValue,
        layout: Layout | torch.Tensor,
    ) -> torch.Tensor:
        """Bind one SSA value to a placeholder with a static layout."""
        if isinstance(layout, torch.Tensor):
            layout = Layout.of(layout)
        with self.fake_mode:
            fake = torch.empty_strided(
                layout.sizes,
                layout.strides,
                dtype=layout.dtype,
                device=layout.device,
            )
        self._buffers[id(fake)] = (fake, value)
        return fake

    def fake(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a placeholder Tensor with the layout of ``tensor``."""
        return self.fake_mode.from_tensor(tensor, static_shapes=True)

    def like(
        self, tensor: torch.Tensor, *, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return an unbound placeholder with the layout of ``tensor``."""
        with self.fake_mode:
            return torch.empty_strided(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device if device is None else device,
            )

    def host_scalar(self, value: SSAValue, sample: float) -> HostScalar:
        """Bind one host scalar SSA value for use as an operator argument."""
        index = len(self._host)
        self._host[index] = value
        return HostScalar(sample, index)

    def host_integer(self, value: SSAValue, sample: int) -> HostInt:
        """Bind one host integer SSA value for use as an operator argument."""
        index = len(self._host)
        self._host[index] = value
        return HostInt(sample, index)

    def drain(self) -> list[Operation]:
        """Return and forget the operations recorded since the last drain."""
        operations, self.operations = self.operations, []
        return operations

    def drain_bodies(self) -> list[tuple[Region, tuple[object, ...]]]:
        """Return Python-call bodies with the sample values of their arguments."""
        bodies, self._bodies = self._bodies, []
        return bodies

    def value(self, tensor: torch.Tensor | HostScalar | HostInt) -> SSAValue:
        """Return the SSA value of a placeholder, host scalar or constant."""
        if isinstance(tensor, _HOST):
            return self._host[tensor.index]
        entry = self._buffers.get(id(tensor))
        if entry is not None:
            return entry[1]
        if isinstance(tensor, FakeTensor):
            raise LoweringError("Placeholder Tensor is not bound to a buffer")
        return self._constant(tensor)

    def is_placeholder(self, tensor: object) -> bool:
        return id(tensor) in self._buffers

    @contextmanager
    def tracing(self) -> Iterator[None]:
        """Record PyTorch operator calls and kernel launches made inside this context."""
        token = _ACTIVE.set(self)
        try:
            with _TraceMode(self), _HostArgumentMode(self):
                yield
        finally:
            _ACTIVE.reset(token)

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Run link-time preparation on placeholders without recording it."""
        with isolated():
            yield

    def memo(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """Return the value built once for ``key`` while lowering this schedule."""
        if key not in self._memo:
            with self.suspended():
                self._memo[key] = build()
        return self._memo[key]

    # Recorded primitives -----------------------------------------------

    def _append(self, operation: Operation) -> Operation:
        self.operations.append(operation)
        return operation

    def _symbol(self, prefix: str, value: object) -> str:
        symbol = self._symbols.get(id(value))
        if symbol is None:
            symbol = f"{prefix}{self._ordinal}"
            self._ordinal += 1
            self.objects[symbol] = value
            self._symbols[id(value)] = symbol
        return symbol

    def _result(
        self,
        operation: Operation,
        index: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._buffers[id(tensor)] = (tensor, operation.results[index])
        return tensor

    def _constant(self, tensor: torch.Tensor) -> SSAValue:
        entry = self._constant_buffers.get(id(tensor))
        if entry is not None:
            return self._buffers[id(entry[1])][1]
        symbol = f"constant{self._ordinal}"
        self._ordinal += 1
        self.constants[symbol] = tensor
        fake = self.fake(tensor)
        operation = self._append(
            ex.MaterialOp(symbol, _buffer_type("material", tensor))
        )
        self._constant_buffers[id(tensor)] = (tensor, fake)
        self._result(operation, 0, fake)
        return operation.results[0]

    def alloc(
        self,
        sizes: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        strides: Sequence[int] | None = None,
        zero: bool = False,
    ) -> torch.Tensor:
        """Allocate one schedule-owned buffer."""
        with self.fake_mode:
            fake = (
                torch.empty(tuple(sizes), dtype=dtype, device=device)
                if strides is None
                else torch.empty_strided(
                    tuple(sizes), tuple(strides), dtype=dtype, device=device
                )
            )
        operation = self._append(ex.AllocOp(_buffer_type("arena", fake)))
        self._result(operation, 0, fake)
        if zero:
            self._append(ex.ZeroOp(operation.results[0]))
        return fake

    def copy(self, target: torch.Tensor, source: torch.Tensor) -> None:
        """Copy ``source`` into the storage of ``target``."""
        self._check_writable(target)
        self._append(ex.CopyOp(self.value(target), self.value(source)))

    def state(self, resource: object, field_name: str) -> torch.Tensor:
        """Bind one mutable Tensor field of a live resource."""
        tensor = getattr(resource, field_name)
        symbol = self._symbol("resource", resource)
        fake = self.fake(tensor)
        operation = self._append(
            ex.StateOp(
                symbol,
                type(resource).__name__,
                field_name,
                _buffer_type("state", tensor),
            )
        )
        return self._result(operation, 0, fake)

    def _group(self, resource: object) -> SSAValue:
        symbol = self._symbol("group", resource)
        return self._append(ex.ResourceOp(symbol, "process-group")).results[0]

    def group_query(self, kind: str, resource: object, sample: int) -> HostInt:
        """Read the ``rank`` or ``size`` of a bound process group when the schedule runs; ``sample`` is its value at link time."""
        operation = self._append(ex.GroupQueryOp(kind, self._group(resource)))
        return self.host_integer(operation.results[0], sample)

    def collective(
        self,
        kind: str,
        resource: object,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        options: Sequence[int] = (),
    ) -> None:
        """Run a collective of a bound process group that reads ``inputs`` and writes ``outputs``."""
        for output in outputs:
            self._check_writable(output)
        self._append(
            ex.CollectiveOp(
                kind,
                self._group(resource),
                [self.value(item) for item in inputs],
                [self.value(item) for item in outputs],
                options,
            )
        )

    def inline(
        self, region: Region, arguments: Sequence[object]
    ) -> tuple[object, ...]:
        """Record the operations of the single-block ``region`` of the operation being lowered at this point of the schedule, with ``arguments`` for its block arguments, and return the values it yields."""
        if self._inline is None:
            raise LoweringError("This lowering cannot inline regions")
        return self._inline(region, tuple(arguments))

    def launch(
        self,
        kernel: object,
        arguments: Sequence[object],
        *,
        device: torch.device,
        grid: Sequence[int] = (),
    ) -> None:
        """Launch a prepared kernel whose Tensor arguments are buffers."""
        inputs: list[SSAValue] = []
        encoded = ArrayAttr(
            self._encode(argument, inputs) for argument in arguments
        )
        symbol = self._symbol("kernel", kernel)
        self.objects.setdefault(f"{symbol}_device", device)
        signature = tuple(
            a.dtype if isinstance(a, torch.Tensor) else a for a in arguments
        )
        self.launches.append((kernel, signature))
        operation = self._append(
            ex.LaunchOp(symbol, tuple(grid), encoded, inputs)
        )
        self.launch_signatures[operation] = signature

    def python_call(
        self,
        function: Callable[..., Sequence[torch.Tensor]],
        inputs: Sequence[object],
        *,
        results: Sequence[torch.Tensor] | None = None,
        bodies: Sequence[tuple[Region, Sequence[object]]] = (),
    ) -> tuple[torch.Tensor, ...]:
        """Call a Python function from the schedule.

        Without ``bodies`` the function receives the operands as positional arguments; with them it receives the operand tuple and one callable per body, each taking and returning a Tensor tuple. Each body pairs a Region, which this call takes from its operation, with placeholder samples of its arguments; the bodies are lowered with the rest of the schedule. ``results`` gives placeholders with the layouts of the returned Tensors; when omitted, a function without bodies runs once on the placeholders to determine them.
        """
        values = [self.value(item) for item in inputs]  # type: ignore[arg-type]
        if results is None:
            if bodies:
                raise LoweringError(
                    "A Python call with bodies requires results"
                )
            with self.fake_mode:
                results = tuple(
                    function(*(self._as_fake(item) for item in inputs))
                )
        regions = []
        for region, samples in bodies:
            if region.parent is not None:
                region.parent.detach_region(region)
            regions.append(region)
            self._bodies.append((region, tuple(samples)))
        symbol = self._symbol("python", function)
        operation = self._append(
            ex.PythonCallOp(
                symbol,
                values,
                [_buffer_type("internal", tensor) for tensor in results],
                regions,
            )
        )
        return tuple(
            self._result(operation, index, self.like(tensor))
            for index, tensor in enumerate(results)
        )

    def call_schedule(
        self,
        function: Callable[..., Sequence[object]],
        inputs: Sequence[object],
        results: Sequence[Layout],
        destinations: Sequence[int] = (),
    ) -> tuple[torch.Tensor, ...]:
        """Call another lowered schedule with buffer operands.

        The results at ``destinations`` are allocated here and passed after the inputs for the callee to write; the call returns the others.
        """
        values = [self.value(item) for item in inputs]  # type: ignore[arg-type]
        written = {
            index: self.alloc(
                results[index].sizes,
                strides=results[index].strides,
                dtype=results[index].dtype,
                device=results[index].device,
            )
            for index in destinations
        }
        values.extend(self.value(written[index]) for index in destinations)
        placeholders = []
        for index, layout in enumerate(results):
            if index in written:
                continue
            with self.fake_mode:
                placeholders.append(
                    torch.empty_strided(
                        layout.sizes,
                        layout.strides,
                        dtype=layout.dtype,
                        device=layout.device,
                    )
                )
        symbol = self._symbol("schedule", function)
        operation = self._append(
            ex.CallScheduleOp(
                symbol,
                values,
                [_buffer_type("internal", tensor) for tensor in placeholders],
            )
        )
        returned = iter(
            self._result(operation, index, tensor)
            for index, tensor in enumerate(placeholders)
        )
        return tuple(
            written[index] if index in written else next(returned)
            for index in range(len(results))
        )

    def _check_writable(
        self, tensor: torch.Tensor | Sequence[torch.Tensor] | None
    ) -> None:
        if tensor is None:
            # An unset optional written argument.
            return
        if isinstance(tensor, (list, tuple)):
            for item in tensor:
                self._check_writable(item)
            return
        if id(tensor) in self._buffers:
            return
        raise LoweringError(
            "A lowered operation writes a link-time constant Tensor"
        )

    # Operator recording --------------------------------------------------

    def _encode(self, value: object, inputs: list[SSAValue]) -> Attribute:
        if isinstance(value, (torch.Tensor, *_HOST)):
            inputs.append(self.value(value))
            return ex.OperandRef(IntAttr(len(inputs) - 1))
        if value is None:
            return NoneAttr()
        if isinstance(value, bool):
            return IntegerAttr(int(value), 1)
        if isinstance(value, int):
            return IntegerAttr(value, 64)
        if isinstance(value, float):
            return FloatAttr(value, f64)
        if isinstance(value, str):
            return StringAttr(value)
        if isinstance(value, (list, tuple)):
            return ArrayAttr(self._encode(item, inputs) for item in value)
        constant = _torch_constant(value)
        if constant is None:
            raise LoweringError(
                f"Unsupported operator argument of type {type(value).__name__}"
            )
        return constant

    def _dispatch(self, func, args, kwargs):
        if self._pending and self._pending[-1][0] is func:
            args, kwargs = _with_arguments(
                func._schema, args, kwargs, self._pending[-1][1]
            )
        tensors = [
            leaf
            for leaf in tree_flatten((args, kwargs))[0]
            if isinstance(leaf, torch.Tensor)
        ]
        traced = any(id(tensor) in self._buffers for tensor in tensors) or any(
            isinstance(leaf, _HOST) for leaf in tree_flatten((args, kwargs))[0]
        )
        if func in _ALLOCATIONS:
            return self._allocation(func, args, kwargs)
        if not traced:
            if any(isinstance(tensor, FakeTensor) for tensor in tensors):
                raise LoweringError(
                    "Placeholder Tensor is not bound to a buffer"
                )
            return func(*args, **kwargs)
        if func in _COPIES:
            self.copy(args[0], args[1])
            return args[0]
        if func in _ZEROS:
            self._check_writable(args[0])
            self._append(ex.ZeroOp(self.value(args[0])))
            return args[0]
        if self.destinations:
            if func is aten.clone.default:
                fake = cast(torch.Tensor, self._as_fake(args[0]))
                with self.fake_mode:
                    output = func(fake, **kwargs)
                operation = self._append(
                    ex.AllocOp(_buffer_type("arena", output))
                )
                result = self._result(operation, 0, output)
                self.copy(result, args[0])
                return result
            destination = _destination_overload(func, _dispatch_key(tensors))
            if destination is not None:
                return self._into_destination(func, destination, args, kwargs)
            if (
                any(a.name == "workspace" for a in func._schema.arguments)
                and _schema_argument(func._schema, args, kwargs, "workspace")
                is None
            ):
                kwargs = {
                    **kwargs,
                    "workspace": self._workspace(func, args, kwargs, tensors),
                }
        fake_args, fake_kwargs = tree_map(self._as_fake, (args, kwargs))
        with self.fake_mode:
            output = func(*fake_args, **fake_kwargs)
        schema = func._schema
        returns = schema.returns
        written = [
            argument
            for argument in schema.arguments
            if argument.alias_info is not None and argument.alias_info.is_write
        ]
        views = bool(returns) and all(
            item.alias_info is not None and not item.alias_info.is_write
            for item in returns
        )
        if views and isinstance(output, torch.Tensor):
            return self._view(
                self._alias_source(schema, args, kwargs, returns[0]),
                output,
            )
        if views and isinstance(output, (list, tuple)):
            source = self._alias_source(schema, args, kwargs, returns[0])
            return type(output)(self._view(source, item) for item in output)
        for argument in written:
            self._check_writable(
                cast(
                    torch.Tensor,
                    self._argument(schema, args, kwargs, argument.name),
                )
            )
        inputs: list[SSAValue] = []
        encoded = []
        for index, argument in enumerate(schema.arguments):
            value = self._argument(schema, args, kwargs, argument.name)
            encoded.append(self._encode(value, inputs))
        if not written and not any(
            isinstance(item, torch.Tensor) for item in tree_flatten(output)[0]
        ):
            # Metadata queries return host values and record nothing.
            return output
        # Pair each output with its schema return: aliased returns are written
        # arguments, unaliased returns are fresh results of this call.
        multiple = isinstance(output, (list, tuple))
        items = list(output) if multiple else [output]
        aliases = (
            [item.alias_info for item in returns]
            if len(returns) == len(items)
            else [returns[0].alias_info if returns else None] * len(items)
        )
        fresh = [
            item
            for item, alias in zip(items, aliases, strict=True)
            if isinstance(item, torch.Tensor) and alias is None
        ]
        operation = self._append(
            ex.CallOperatorOp(
                _schema_name(func),
                ArrayAttr(encoded),
                inputs,
                [_buffer_type("internal", tensor) for tensor in fresh],
            )
        )
        for index, tensor in enumerate(fresh):
            self._result(operation, index, tensor)
        bound = [
            item
            if alias is None
            else tree_map(
                lambda leaf: self._written_result(leaf, args, kwargs), item
            )
            for item, alias in zip(items, aliases, strict=True)
        ]
        return type(output)(bound) if multiple else bound[0]

    def record_call(self, func, args, kwargs, output: object) -> object:
        """Record a call of ``func`` whose ``output`` was computed outside tracing and return its placeholders.

        Placeholder arguments read their buffers and other Tensors become link-time constants. Every Tensor of ``output`` becomes a fresh result of the recorded call, in flattening order; the returned structure mirrors ``output`` with placeholders in their place.
        """
        inputs: list[SSAValue] = []
        encoded = [
            self._encode(
                self._argument(func._schema, args, kwargs, a.name), inputs
            )
            for a in func._schema.arguments
        ]
        leaves, spec = tree_flatten(output)
        fresh = [
            self.fake(leaf) for leaf in leaves if isinstance(leaf, torch.Tensor)
        ]
        operation = self._append(
            ex.CallOperatorOp(
                _schema_name(func),
                ArrayAttr(encoded),
                inputs,
                [_buffer_type("internal", tensor) for tensor in fresh],
            )
        )
        for index, tensor in enumerate(fresh):
            self._result(operation, index, tensor)
        placed = iter(fresh)
        return tree_unflatten(
            [
                next(placed) if isinstance(leaf, torch.Tensor) else leaf
                for leaf in leaves
            ],
            spec,
        )

    def view_of(
        self, source: torch.Tensor, layout: torch.Tensor
    ) -> torch.Tensor:
        """Record a view of placeholder ``source`` with the shape, strides and storage offset of ``layout``."""
        with self.fake_mode:
            output = torch.as_strided(
                source, layout.shape, layout.stride(), layout.storage_offset()
            )
        return self._view(source, output)

    def _into_destination(self, func, destination, args, kwargs):
        """Allocate the results of ``func`` and record ``destination`` writing them."""
        fake_args, fake_kwargs = tree_map(self._as_fake, (args, kwargs))
        with self.fake_mode:
            output = func(*fake_args, **fake_kwargs)
        items = list(output) if isinstance(output, (list, tuple)) else [output]
        allocated = []
        for item in items:
            operation = self._append(ex.AllocOp(_buffer_type("arena", item)))
            allocated.append(self._result(operation, 0, item))
        out = allocated if isinstance(output, (list, tuple)) else allocated[0]
        # The destination overload declares no defaults: pass every argument.
        values: dict[str, object] = {
            argument.name: _schema_argument(
                func._schema, args, kwargs, argument.name
            )
            for argument in func._schema.arguments
        }
        values["out"] = out
        for argument in destination._schema.arguments:
            if argument.name not in values and argument.name != "workspace":
                values[argument.name] = argument.default_value
        if any(a.name == "workspace" for a in destination._schema.arguments):
            values["workspace"] = self._workspace(func, args, kwargs, items)
        self._dispatch(destination, (), values)
        return (
            type(output)(allocated)
            if isinstance(output, (list, tuple))
            else out
        )

    def _workspace(self, func, args, kwargs, items) -> torch.Tensor | None:
        """Allocate the workspace of one call of ``func`` or of its destination overload.

        ``namespace::name_workspace_bytes`` (without the trailing underscore of an in-place ``name_``) takes the arguments of ``func`` other than ``workspace`` and returns the bytes the call's temporaries need; the workspace is a byte buffer that the planner places in the arena for the duration of the call.
        """
        schema = func._schema
        namespace, name = schema.name.split("::", 1)
        query = getattr(
            getattr(
                getattr(torch.ops, namespace),
                f"{name.rstrip('_')}_workspace_bytes",
            ),
            schema.overload_name or "default",
        )
        arguments = [
            tree_map(
                self._as_fake, _schema_argument(schema, args, kwargs, a.name)
            )
            for a in schema.arguments
            if a.name != "workspace"
        ]
        with self.fake_mode:
            size = int(query(*arguments))
        if size == 0:
            return None
        device = next(
            item.device for item in items if isinstance(item, torch.Tensor)
        )
        return self.alloc((size,), dtype=torch.uint8, device=device)

    def _written_result(self, item, args, kwargs):
        if not isinstance(item, torch.Tensor):
            return item
        for candidate in tree_flatten((args, kwargs))[0]:
            if (
                isinstance(candidate, torch.Tensor)
                and id(candidate) in self._buffers
                and self._as_fake(candidate) is item
            ):
                return candidate
        return item

    def _argument(self, schema, args, kwargs, name: str) -> object:
        return _schema_argument(schema, args, kwargs, name)

    def _alias_source(self, schema, args, kwargs, returned) -> torch.Tensor:
        sets = returned.alias_info.before_set
        for argument in schema.arguments:
            # A view list such as unbind's Tensor(a)[] reports no alias set.
            if argument.alias_info is not None and (
                not sets or argument.alias_info.before_set & sets
            ):
                source = self._argument(schema, args, kwargs, argument.name)
                return cast(torch.Tensor, source)
        raise LoweringError(f"View operator {schema.name} has no source")

    def _as_fake(self, value: object) -> object:
        if isinstance(value, HostScalar):
            return float(value)
        if isinstance(value, HostInt):
            return int(value)
        if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
            return value
        entry = self._constant_buffers.get(id(value))
        if entry is None:
            self._constant(value)
            entry = self._constant_buffers[id(value)]
        return entry[1]

    def _view(self, source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        source_fake = cast(torch.Tensor, self._as_fake(source))
        if (
            output.shape == source_fake.shape
            and output.stride() == source_fake.stride()
            and output.storage_offset() == source_fake.storage_offset()
        ):
            # A view with the source layout is the source buffer itself.
            self._buffers[id(output)] = (output, self.value(source_fake))
            return output
        operation = self._append(
            ex.ViewOp(
                self.value(source_fake),
                tuple(output.shape),
                tuple(output.stride()),
                int(output.storage_offset() - source_fake.storage_offset()),
                _buffer_type("view", output),
            )
        )
        return self._result(operation, 0, output)

    def _allocation(self, func, args, kwargs) -> torch.Tensor:
        fake_args, fake_kwargs = tree_map(self._as_fake, (args, kwargs))
        with self.fake_mode:
            output = func(*fake_args, **fake_kwargs)
        # Page-locked host storage keeps its own storage class: the Runner
        # allocates it from pinned memory outside the arena.
        storage = "pinned" if kwargs.get("pin_memory") else "arena"
        operation = self._append(ex.AllocOp(_buffer_type(storage, output)))
        self._result(operation, 0, output)
        if _ALLOCATIONS[func]:
            self._append(ex.ZeroOp(operation.results[0]))
        return output


_DESTINATIONS: dict[tuple[object, str], object] = {}


def _dispatch_key(tensors: Sequence[torch.Tensor]) -> str:
    return "CUDA" if any(t.device.type == "cuda" for t in tensors) else "CPU"


def _destination_overload(func, dispatch_key: str) -> object | None:
    """Return the destination-passing overload of a functional operator, if any.

    The overload takes the arguments of ``func``, including those an in-place FHElium ``name_`` updates, followed by one written argument ``out`` that receives every result (``name_out_`` for such an operator), and for FHElium operators optionally a written byte ``workspace`` for the call's temporaries and optional written scratch arguments that lowering leaves unset: ``aten::name.out`` (``.<overload>_out``) for ATen and ``namespace::name_out`` for FHElium operators. An ATen overload must have a kernel for ``dispatch_key``; its generated wrappers compute the functional result and copy it into ``out``. FHElium ``_out`` operators write their results into ``out``, including composite ones.
    """
    if (func, dispatch_key) in _DESTINATIONS:
        return _DESTINATIONS[func, dispatch_key]
    schema = func._schema
    found = None
    returns = schema.returns
    namespace, name = schema.name.split("::", 1)
    updated = [
        a.name
        for a in schema.arguments
        if a.alias_info is not None and a.alias_info.is_write
    ]
    if (
        returns
        and all(item.alias_info is None for item in returns)
        and (len(returns) == 1 or str(returns[0].type).startswith("List"))
        and all(
            a.alias_info is None or a.alias_info.is_write
            for a in schema.arguments
        )
        and (namespace != "aten" or not updated)
    ):
        packet = getattr(torch.ops, namespace)
        overload = schema.overload_name
        if namespace == "aten":
            candidate = getattr(
                getattr(packet, name, None),
                f"{overload}_out" if overload else "out",
                None,
            )
        else:
            # An in-place ``name_`` that also returns a result has ``name_out_``.
            candidate = getattr(
                getattr(
                    packet,
                    f"{name[:-1]}_out_"
                    if name.endswith("_")
                    else f"{name}_out",
                    None,
                ),
                overload or "default",
                None,
            )
        if candidate is not None:
            arguments = candidate._schema.arguments
            written = [
                a
                for a in arguments
                if a.alias_info is not None and a.alias_info.is_write
            ]
            # After the arguments ``func`` updates and ``out``, FHElium destination
            # overloads may add a written ``workspace`` and optional written
            # scratch arguments, which stay unset.
            names = [a.name for a in written]
            count = len(updated) + 1
            if (
                names[:count] == [*updated, "out"]
                and all(
                    a.name == "workspace" or a.has_default_value()
                    for a in written[count:]
                )
                and [a.name for a in arguments if a.name not in names]
                == [a.name for a in schema.arguments if a.name not in updated]
            ):
                found = candidate
    if (
        found is not None
        and schema.name.startswith("aten::")
        and not torch._C._dispatch_has_kernel_for_dispatch_key(
            found._schema.name
            + (
                f".{found._schema.overload_name}"
                if found._schema.overload_name
                else ""
            ),
            dispatch_key,
        )
    ):
        found = None
    _DESTINATIONS[func, dispatch_key] = found
    return found


def _schema_name(func) -> str:
    schema = func._schema
    return f"{schema.name}.{schema.overload_name or 'default'}"


def resolve_operator(schema: str):
    """Return the dispatcher operator named ``namespace::name.overload``."""
    qualified, overload = schema.rsplit(".", 1)
    namespace, name = qualified.split("::", 1)
    return getattr(getattr(getattr(torch.ops, namespace), name), overload)


def _pure(operation: Operation) -> bool:
    if isinstance(
        operation, (ex.ViewOp, ex.MaterialOp, ex.AllocOp, ex.StateOp)
    ):
        return True
    if isinstance(operation, ex.CallOperatorOp) and operation.results:
        schema = resolve_operator(operation.schema.data)._schema
        return (
            not any(
                argument.alias_info is not None and argument.alias_info.is_write
                for argument in schema.arguments
            )
            and not schema.is_mutable
        )
    return False


def remove_dead_operations(block: Block) -> None:
    """Erase side-effect-free exec operations whose results are unused."""
    for operation in reversed(tuple(block.ops)):
        for region in operation.regions:
            for nested in region.blocks:
                remove_dead_operations(nested)
        if _pure(operation) and not any(
            result.uses for result in operation.results
        ):
            operation.detach()
            operation.erase()


__all__ = [
    "ExecBuilder",
    "Fragment",
    "HostInt",
    "HostScalar",
    "Layout",
    "LoweredCall",
    "LoweringError",
    "TorchConstant",
    "active_builder",
    "compile_fragment",
    "layout_key",
    "remove_dead_operations",
    "resolve_operator",
    "torch_constant_value",
]


@dataclass(frozen=True)
class Fragment:
    """Lowered exec block of one operation call and its Runner function."""

    block: Block
    function: Callable[..., tuple[object, ...]]


def compile_fragment(
    lower: Callable[[ExecBuilder, tuple[object, ...]], Sequence[torch.Tensor]],
    samples: Sequence[object],
) -> Fragment:
    """Lower one call into a fragment.

    ``samples`` lists the fragment inputs in order: Tensors fix the layouts of runtime Tensors, Python floats and integers become :class:`HostScalar` and :class:`HostInt` inputs. ``lower`` receives the builder and the corresponding placeholders and returns the results; a :class:`HostInt` result is returned as a 0-dim int64 CPU Tensor.
    """
    with isolated():
        return _compile_fragment(lower, samples)


def _compile_fragment(
    lower: Callable[[ExecBuilder, tuple[object, ...]], Sequence[torch.Tensor]],
    samples: Sequence[object],
) -> Fragment:
    from .runner import fragment_function, translate_block

    builder = ExecBuilder()
    block = Block(
        arg_types=tuple(
            _buffer_type("input", sample)
            if isinstance(sample, torch.Tensor)
            else f64
            if isinstance(sample, float)
            else i64
            for sample in samples
        )
    )
    values = tuple(
        builder.placeholder(argument, sample)
        if isinstance(sample, torch.Tensor)
        else builder.host_scalar(argument, sample)
        if isinstance(sample, float)
        else builder.host_integer(argument, cast(int, sample))
        for argument, sample in zip(block.args, samples)
    )
    with builder.tracing():
        # A host integer result is returned as a 0-dim int64 CPU Tensor.
        results = tuple(
            torch.ops.aten.full.default(
                [], result, dtype=torch.int64, device=torch.device("cpu")
            )
            if isinstance(result, HostInt)
            else result
            for result in lower(builder, values)
        )
    outputs = [builder.value(result) for result in results]
    block.add_ops(builder.operations)
    block.add_op(ex.YieldOp(*outputs))
    remove_dead_operations(block)
    schedule = translate_block(
        block,
        arguments=tuple(
            ("buffer", argument.type)
            if isinstance(sample, torch.Tensor)
            else ("float" if isinstance(sample, float) else "int", None)
            for argument, sample in zip(block.args, samples, strict=True)
        ),
        objects=builder.objects,
        constants=builder.constants,
        signatures=builder.launch_signatures,
    )
    return Fragment(block, fragment_function(schedule))


class LoweredCall:
    """Call one implementation through fragments lowered per payload layout and resources.

    Generated Backend Python and prepared combine callables use it for operation calls; the first call with a payload layout and resource selection lowers and compiles its fragment.
    """

    def __init__(self, implementation, invocation) -> None:
        self.implementation = implementation
        self.invocation = invocation
        self._fragments: dict[tuple[object, ...], Callable] = {}

    def __call__(
        self, payloads: tuple[object, ...], resources: tuple[object, ...] = ()
    ) -> tuple[torch.Tensor, ...]:
        key = (
            tuple(
                layout_key(value)
                if isinstance(value, torch.Tensor)
                else type(value)
                for value in payloads
            ),
            tuple(id(getattr(r, "value", r)) for r in resources),
        )
        fragment = self._fragments.get(key)
        if fragment is None:
            fragment = compile_fragment(
                lambda builder, values: self.implementation.lower(
                    self.invocation, values, resources, builder
                ),
                payloads,
            ).function
            self._fragments[key] = fragment
        return cast(tuple[torch.Tensor, ...], fragment(*payloads))
