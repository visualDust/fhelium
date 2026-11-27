"""Build exec schedules into libfhelium Runners and execute them.

A :class:`Schedule` is the Runner construction of one exec block: its buffers with storage classes and layouts, its host integer and float registers, and the ordered builder calls that append operations and structured control flow. Translating a block resolves operator arguments, kernel launch parameters, loop-carried values and results once; realizing the schedule replays the builder calls on a ``torch.classes.fhelium.Runner`` and binds materials, link-time constants and mutable resource state to their buffers. Execution passes the Tensor, integer and float inputs to the Runner and returns its output buffers. Python implementations run between Runner instructions: an execution pauses at each ``exec.python_call``, the driver calls the function with the instruction's operands and resumes the execution with its results.
"""

from __future__ import annotations

import atexit
import collections
import weakref

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
    UnrealizedConversionCastOp,
    _FloatType,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Operation, SSAValue

from fhelium.ir.dialects import core, distributed, fusion
from fhelium.ir.dialects import exec as ex

from .lowering import (
    LoweringError,
    TorchConstant,
    resolve_operator,
    torch_constant_value,
)

BUFFER = 1
BUFFER_LIST = 2
INT_REGISTER = 3
FLOAT_REGISTER = 4
CONSTANT = 0
EMPTY_LIST = 5
"""Argument kinds of ``Runner.call_op``, ``Runner.launch`` and their values.

An empty list argument has its own kind because PyTorch cannot infer the element type of an empty ``Any`` list; the Runner takes the type from the schema.
"""


# Bound buffer descriptions ------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """A buffer bound to the material of ``symbol``."""

    symbol: str


@dataclass(frozen=True)
class Constant:
    """A buffer bound to a link-time constant Tensor."""

    tensor: torch.Tensor

    def __reduce__(self):
        buffer = io.BytesIO()
        torch.save(self.tensor.detach().cpu(), buffer)
        return (_constant, (buffer.getvalue(), str(self.tensor.device)))


def _constant(data: bytes, device: str) -> Constant:
    return Constant(torch.load(io.BytesIO(data)).to(device))


@dataclass(frozen=True)
class Resources:
    """Linked resources selected by requirement position."""

    indices: tuple[int, ...]
    single: bool = False
    value: bool = False
    """Select the resource's live value instead of the BoundResource."""


@dataclass(frozen=True)
class State:
    """A buffer bound to the Tensor ``field`` of a live resource.

    ``resource`` is a :class:`Resources` selection or the resource object itself.
    """

    resource: object
    field: str


@dataclass(frozen=True)
class Group:
    """A process group of a schedule: its Program resource symbol, the group size the schedule was lowered for, and the bound resource whose group the Runner uses."""

    symbol: str
    size: int
    resource: object = field(compare=False)


def group_name(resource: object) -> str:
    """Return the c10d registry name of a bound process-group resource."""
    group = getattr(getattr(resource, "value", resource), "group", None)
    if group is None:
        group = torch.distributed.group.WORLD
    return cast(Any, group).group_name


_GROUP_HOLDERS: weakref.WeakKeyDictionary[Any, frozenset[str]] = (
    weakref.WeakKeyDictionary()
)
"""Schedule functions and artifacts whose Runners are bound to process groups, with the c10d names of those groups."""


def release_groups(name: str | None = None) -> None:
    """Release the process groups held by compiled schedules.

    Each Runner bound to the group registered as ``name``, or to any group for ``None``, drops its captured graphs and group references; a released schedule function binds its groups again at its next execution. A process group whose collectives a live graph captured cannot be destroyed, so :func:`fhelium.distributed.shutdown` and interpreter exit call this first.
    """
    for holder, names in list(_GROUP_HOLDERS.items()):
        if name is None or name in names:
            holder.release_groups()
            _GROUP_HOLDERS.pop(holder, None)


atexit.register(release_groups)


@dataclass(frozen=True)
class ObjectRef:
    """Builder-call argument naming ``Schedule.objects[index]``."""

    index: int


# Schedule ---------------------------------------------------------------------


@dataclass(frozen=True)
class PythonCallee:
    """Python function of one ``exec.python_call`` and its operand order.

    ``kinds`` lists the operand kinds in call order (``BUFFER``, ``INT_REGISTER`` or ``FLOAT_REGISTER``); the Runner passes each kind as a separate list. With ``bodies`` the function receives the operand tuple and one callable per body; each body is a schedule whose inputs are its region arguments followed by the values it reads from the enclosing schedule. ``captured`` gives the kinds of those values for each body; they follow the operands in call order.
    """

    function: Callable[..., Sequence[torch.Tensor]]
    kinds: tuple[int, ...]
    bodies: tuple[Schedule, ...] = ()
    captured: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class CalledSchedule:
    """Callee of ``exec.call``: a linked executable lowered for fixed input layouts."""

    executable: Any
    layouts: tuple[object, ...]
    destinations: tuple[int, ...] = ()
    """Positions of the results whose storage the caller passes as trailing Tensor arguments (:func:`~fhelium.backend._planning.with_destinations`)."""

    @property
    def function(self) -> ScheduleFunction:
        function = self.executable._lowered[self.layouts].function
        if not self.destinations:
            return function
        forms = _DESTINATION_FORMS.setdefault(function, {})
        form = forms.get(self.destinations)
        if form is None:
            from ._planning import with_destinations

            form = forms[self.destinations] = ScheduleFunction(
                with_destinations(function.schedule, self.destinations),
                function.materials,
                function.resources,
                arena=True,
                shared=True,
            )
        return form

    @property
    def schedule(self) -> Schedule:
        return self.function.schedule

    def runner(self) -> Any:
        # The caller's arena buffers stay live across the call, so the callee
        # keeps its temporaries in a private arena.
        return self.function.runner(private=True)


_DESTINATION_FORMS: weakref.WeakKeyDictionary[
    ScheduleFunction, dict[tuple[int, ...], ScheduleFunction]
] = weakref.WeakKeyDictionary()
"""Called schedules in destination form, by the lowered function and result positions."""


@dataclass(frozen=True)
class Schedule:
    """Runner construction of one exec block.

    ``buffers`` gives each buffer's storage class and optional sizes, strides, dtype and device. ``instructions`` are builder calls ``(method, *arguments)`` in block order; :class:`ObjectRef` arguments name kernels, Python callees and called schedules in ``objects``. ``inputs`` lists the kind and buffer or register of each argument, ``outputs`` the result of each schedule result (``("buffer", position)`` among the Runner outputs, ``("input", index)`` or ``("value", object)``), and ``bindings`` the bound value of material and state buffers.
    """

    buffers: tuple[
        tuple[
            str,
            tuple[int, ...] | None,
            tuple[int, ...] | None,
            torch.dtype | None,
            torch.device | None,
        ],
        ...,
    ]
    int_registers: int
    float_registers: int
    instructions: tuple[tuple[object, ...], ...]
    inputs: tuple[tuple[str, int], ...]
    output_buffers: tuple[int, ...]
    outputs: tuple[tuple[str, object], ...]
    bindings: tuple[tuple[int, object], ...]
    objects: tuple[Any, ...]
    graphs: bool = False
    """Capture and replay the schedule's ``begin_segment``/``end_segment`` regions as CUDA Graphs."""
    groups: tuple[Group, ...] = ()
    """Process groups by Runner group identifier."""
    fixed_rank: bool = False
    """The structured control flow of the schedule, or of a schedule it calls, was expanded for the process rank it was lowered on."""
    read_in_place: tuple[int, ...] = ()
    """Input buffers whose contents stay at their address across calls, such as keys and plaintexts; Graph segments read them in place rather than through staging storage."""

    @property
    def python_calls(self) -> bool:
        return any(
            instruction[0] == "python_call" for instruction in self.instructions
        )

    @property
    def launches(self) -> tuple[tuple[Any, tuple[object, ...]], ...]:
        """Kernels launched by this schedule and its bodies with their warm signatures."""
        found = []
        for instruction in self.instructions:
            if instruction[0] == "launch":
                kernel = self.objects[cast(ObjectRef, instruction[1]).index]
                found.append((kernel, cast(tuple, instruction[-1])))
        for item in self.objects:
            if isinstance(item, PythonCallee):
                for body in item.bodies:
                    found.extend(body.launches)
        return tuple(found)


# Translation ------------------------------------------------------------------

_BINARY = {
    arith.AddiOp: "add",
    arith.SubiOp: "sub",
    arith.MuliOp: "mul",
    arith.DivUIOp: "div",
    arith.DivSIOp: "div",
    arith.FloorDivSIOp: "div",
    arith.RemUIOp: "rem",
    arith.RemSIOp: "rem",
    arith.MinUIOp: "min",
    arith.MaxUIOp: "max",
    arith.MinSIOp: "min",
    arith.MaxSIOp: "max",
}
_PREDICATES = ("eq", "ne", "lt", "le", "gt", "ge", "lt", "le", "gt", "ge")
_TERMINATORS = (
    ReturnOp,
    scf.YieldOp,
    ex.YieldOp,
    distributed.YieldOp,
    fusion.YieldOp,
)


def _torch_dtype(name: str) -> torch.dtype | None:
    return getattr(torch, name.removeprefix("torch.")) if name else None


def _integers(values: Attribute) -> tuple[int, ...] | None:
    if not isinstance(values, ArrayAttr):
        return None
    return tuple(int(cast(IntegerAttr, v).value.data) for v in values)


def _constant_value(attribute: Attribute) -> object:
    if isinstance(attribute, NoneAttr):
        return None
    if isinstance(attribute, IntegerAttr):
        value = int(attribute.value.data)
        return bool(value) if attribute.type.width.data == 1 else value
    if isinstance(attribute, FloatAttr):
        return float(attribute.value.data)
    if isinstance(attribute, StringAttr):
        return attribute.data
    if isinstance(attribute, TorchConstant):
        return torch_constant_value(attribute)
    if isinstance(attribute, ArrayAttr):
        return [_constant_value(item) for item in attribute]
    raise LoweringError(f"Unsupported exec argument {attribute}")


@dataclass
class _Translation:
    """Mutable state of one block translation."""

    objects_by_symbol: Mapping[str, Any]
    constants: Mapping[str, torch.Tensor]
    reference: Callable[[Operation], tuple[str, object]]
    """Return ``("material", (symbol, value))`` or ``("object", value)`` for a Program reference operation."""
    describe: Callable[[object], object]
    """Return the recorded form of a live resource bound to state buffers."""
    signatures: Mapping[Operation, tuple[object, ...]]
    buffers: list = field(default_factory=list)
    int_registers: int = 0
    float_registers: int = 0
    instructions: list = field(default_factory=list)
    bindings: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    _object_ids: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    values: dict = field(default_factory=dict)

    # Allocation -------------------------------------------------------------

    def buffer(
        self, buffer_type: Attribute | None, storage: str | None = None
    ) -> int:
        if isinstance(buffer_type, ex.BufferType):
            spec = (
                storage or buffer_type.storage.data,
                _integers(buffer_type.sizes),
                _integers(buffer_type.strides),
                _torch_dtype(buffer_type.dtype.data),
                torch.device(buffer_type.device.data)
                if buffer_type.device.data
                else None,
            )
        else:
            spec = (storage or "internal", None, None, None, None)
        self.buffers.append(spec)
        return len(self.buffers) - 1

    def register(self, kind: str) -> int:
        if kind == "int":
            self.int_registers += 1
            return self.int_registers - 1
        self.float_registers += 1
        return self.float_registers - 1

    def object(self, value: object) -> ObjectRef:
        index = self._object_ids.get(id(value))
        if index is None:
            index = len(self.objects)
            self.objects.append(value)
            self._object_ids[id(value)] = index
        return ObjectRef(index)

    def group(self, resource: object) -> int:
        """Return the Runner group identifier of a process-group resource, declaring it on first use."""
        for index, group in enumerate(self.groups):
            if group.resource is resource:
                return index
        value = getattr(resource, "value", resource)
        self.groups.append(
            Group(
                cast(str, getattr(resource, "symbol", "world")),
                cast(Any, value).size,
                resource,
            )
        )
        return len(self.groups) - 1

    def group_of(self, value: SSAValue) -> int:
        kind, item = self.get(value)
        if kind != "group":
            raise LoweringError("exec operand is not a process group")
        return cast(int, item)

    def emit(self, *instruction: object) -> None:
        self.instructions.append(instruction)

    # Values ------------------------------------------------------------------

    def get(self, value: SSAValue) -> tuple[str, object]:
        try:
            return self.values[value]
        except KeyError:
            raise LoweringError(
                f"exec value of {value.owner} has no Runner buffer"
            ) from None

    def buffer_of(self, value: SSAValue) -> int:
        kind, item = self.get(value)
        if kind != "buffer":
            raise LoweringError(f"exec operand of kind {kind} is not a buffer")
        return cast(int, item)

    def scalar(self, value: SSAValue) -> tuple[int, int]:
        """Return the argument kind and register of a host scalar."""
        kind, item = self.get(value)
        if kind == "int":
            return INT_REGISTER, cast(int, item)
        if kind == "float":
            return FLOAT_REGISTER, cast(int, item)
        if kind == "value" and isinstance(item, (int, float)):
            register = self.register(
                "float" if isinstance(item, float) else "int"
            )
            self.emit(
                "float_constant" if isinstance(item, float) else "int_constant",
                register,
                item,
            )
            self.values[value] = (
                "float" if isinstance(item, float) else "int",
                register,
            )
            return (
                FLOAT_REGISTER if isinstance(item, float) else INT_REGISTER,
                register,
            )
        raise LoweringError(f"exec operand of kind {kind} is not a host scalar")

    def operand(self, value: SSAValue) -> tuple[int, int]:
        kind, item = self.get(value)
        if kind == "buffer":
            return BUFFER, cast(int, item)
        return self.scalar(value)

    def bind_value(self, value: SSAValue, host: object) -> None:
        """Bind a Program value with a host constant."""
        if isinstance(host, bool) or not isinstance(host, (int, float)):
            self.values[value] = ("value", host)
            return
        kind = "float" if isinstance(host, float) else "int"
        register = self.register(kind)
        self.emit(f"{kind}_constant", register, host)
        self.values[value] = (kind, register)

    def like(self, source: tuple[str, object]) -> tuple[str, object]:
        """Return a new buffer or register of the same kind."""
        kind, item = source
        if kind == "buffer":
            return ("buffer", self.buffer(None))
        if kind in ("int", "float"):
            return (kind, self.register(kind))
        raise LoweringError(
            f"Structured control flow cannot carry a host {type(item).__name__}"
        )

    def assign(self, target: tuple[str, object], source: SSAValue) -> None:
        kind, item = target
        if kind == "buffer":
            self.emit("assign", item, self.buffer_of(source))
            return
        source_kind, register = self.scalar(source)
        if (source_kind == INT_REGISTER) != (kind == "int"):
            raise LoweringError(
                "Host register kinds differ across control flow"
            )
        self.emit("assign_register", item, register)

    def assign_all(
        self, targets: Sequence[tuple[str, object]], sources: Sequence[SSAValue]
    ) -> None:
        # Stage through temporaries so permuted carried values do not overwrite each other.
        staged = []
        for target, source in zip(targets, sources, strict=True):
            temporary = self.like(target)
            self.assign(temporary, source)
            staged.append(temporary)
        for target, temporary in zip(targets, staged, strict=True):
            kind, item = target
            if kind == "buffer":
                self.emit("assign", item, temporary[1])
            else:
                self.emit("assign_register", item, temporary[1])

    # Blocks -------------------------------------------------------------------

    def block(self, block: Block) -> Operation | None:
        """Translate the operations of ``block`` and return its terminator.

        A buffer created by an operation of this block is released after the last operation of the block that reads it, counting reads inside nested regions at their enclosing operation, so temporaries do not live until the schedule returns.
        """
        operations = tuple(block.ops)
        last_use: dict[SSAValue, int] = {}
        for index, operation in enumerate(operations):
            for nested in operation.walk():
                for value in nested.operands:
                    last_use[value] = index
        owned: dict[int, list[SSAValue]] = {}
        released: set[int] = set()
        for index, operation in enumerate(operations):
            if isinstance(operation, _TERMINATORS):
                return operation
            first = len(self.buffers)
            self.operation(operation)
            for result in operation.results:
                kind, item = self.values.get(result, ("value", None))
                if kind == "buffer" and cast(int, item) >= first:
                    owned.setdefault(cast(int, item), []).append(result)
            for value in operation.operands:
                kind, item = self.values.get(value, ("value", None))
                if kind == "buffer" and cast(int, item) in owned:
                    # A cast result shares the buffer of its source.
                    owned[cast(int, item)].extend(operation.results)
            for buffer, values in owned.items():
                if buffer in released or self.buffers[buffer][0] in (
                    "material",
                    "state",
                ):
                    continue
                if all(last_use.get(v, index) <= index for v in values):
                    self.emit("release", buffer)
                    released.add(buffer)
        return None

    def operation(self, operation: Operation) -> None:
        if isinstance(operation, ex.CallOperatorOp):
            self.call_op(operation)
        elif isinstance(operation, ex.LaunchOp):
            self.launch(operation)
        elif isinstance(operation, ex.AllocOp):
            buffer = self.buffer(operation.result.type)
            self.values[operation.result] = ("buffer", buffer)
            self.emit("alloc", buffer)
        elif isinstance(operation, ex.ViewOp):
            buffer = self.buffer(operation.result.type)
            self.values[operation.result] = ("buffer", buffer)
            self.emit(
                "view",
                buffer,
                self.buffer_of(operation.source),
                list(_integers(operation.sizes) or ()),
                list(_integers(operation.strides) or ()),
                int(cast(IntegerAttr, operation.offset).value.data),
            )
        elif isinstance(operation, ex.CopyOp):
            self.emit(
                "copy",
                self.buffer_of(operation.target),
                self.buffer_of(operation.source),
            )
        elif isinstance(operation, ex.ZeroOp):
            self.emit("zero", self.buffer_of(operation.target))
        elif isinstance(operation, ex.MaterialOp):
            buffer = self.buffer(operation.result.type, "material")
            self.values[operation.result] = ("buffer", buffer)
            self.bindings.append(
                (buffer, Constant(self.constants[operation.symbol.data]))
            )
        elif isinstance(operation, ex.StateOp):
            buffer = self.buffer(operation.result.type, "state")
            self.values[operation.result] = ("buffer", buffer)
            resource = self.objects_by_symbol[operation.resource.data]
            self.bindings.append(
                (buffer, State(self.describe(resource), operation.field.data))
            )
        elif isinstance(operation, ex.ResourceOp):
            resource = self.objects_by_symbol[operation.symbol.data]
            if operation.result.type.kind.data == "process-group":
                self.values[operation.result] = ("group", self.group(resource))
            else:
                self.values[operation.result] = ("value", resource)
        elif isinstance(operation, ex.GroupQueryOp):
            register = self.register("int")
            self.emit(
                "group_query",
                operation.kind.data,
                register,
                self.group_of(operation.group),
            )
            self.values[operation.result] = ("int", register)
        elif isinstance(operation, ex.CollectiveOp):
            buffers = [self.buffer_of(value) for value in operation.buffers]
            count = int(operation.input_count.value.data)
            self.emit(
                "collective",
                operation.kind.data,
                self.group_of(operation.group),
                buffers[:count],
                buffers[count:],
                list(_integers(operation.options) or ()),
            )
        elif isinstance(operation, ex.CallScheduleOp):
            self.call_schedule(operation)
        elif isinstance(operation, ex.PythonCallOp):
            self.python_call(operation)
        elif isinstance(operation, UnrealizedConversionCastOp):
            for source, result in zip(
                operation.inputs, operation.outputs, strict=True
            ):
                self.values[result] = self.values.get(source, ("value", None))
        elif isinstance(operation, (core.MaterialRefOp, core.ResourceRefOp)):
            kind, item = self.reference(operation)
            result = operation.results[0]
            if kind == "material":
                symbol, value = cast(tuple[str, object], item)
                if isinstance(value, torch.Tensor):
                    buffer = self.buffer(None, "material")
                    self.buffers[buffer] = (
                        "material",
                        tuple(value.shape),
                        tuple(value.stride()),
                        value.dtype,
                        value.device,
                    )
                    self.values[result] = ("buffer", buffer)
                    self.bindings.append((buffer, Material(symbol)))
                else:
                    self.values[result] = ("value", value)
            else:
                self.values[result] = ("value", item)
        elif isinstance(operation, core.ConstantOp):
            from fhelium.compile.passes.codegen._common import constant_literal

            self.bind_value(operation.results[0], constant_literal(operation))
        elif isinstance(operation, arith.ConstantOp):
            self.bind_value(
                operation.results[0],
                cast(IntegerAttr | FloatAttr, operation.value).value.data,
            )
        elif type(operation) in _BINARY:
            self.binary(_BINARY[type(operation)], operation)
        elif isinstance(operation, arith.CmpiOp):
            self.binary(_PREDICATES[operation.predicate.value.data], operation)
        elif isinstance(operation, arith.IndexCastOp):
            kind, register = self.scalar(operation.input)
            target = self.register("int")
            self.emit("assign_register", target, register)
            self.values[operation.result] = ("int", target)
        elif isinstance(operation, arith.SelectOp):
            self.select(operation)
        elif isinstance(operation, scf.ForOp):
            self.for_(operation)
        elif isinstance(operation, scf.IfOp):
            self.if_(operation)
        else:
            raise LoweringError(f"Runner cannot execute {operation.name}")

    def binary(self, name: str, operation: Operation) -> None:
        kinds = [self.scalar(value) for value in operation.operands]
        if any(kind != INT_REGISTER for kind, _ in kinds):
            raise LoweringError(f"{operation.name} requires integer registers")
        target = self.register("int")
        self.emit("int_binary", name, target, kinds[0][1], kinds[1][1])
        self.values[operation.results[0]] = ("int", target)

    def select(self, operation: arith.SelectOp) -> None:
        _, condition = self.scalar(operation.cond)
        if self.get(operation.lhs)[0] == "buffer":
            raise LoweringError("Runner selects host registers only")
        kind, lhs = self.scalar(operation.lhs)
        _, rhs = self.scalar(operation.rhs)
        target = self.register("int" if kind == INT_REGISTER else "float")
        self.emit("select", target, condition, lhs, rhs)
        self.values[operation.result] = (
            "int" if kind == INT_REGISTER else "float",
            target,
        )

    def for_(self, operation: scf.ForOp) -> None:
        bounds = [
            self.scalar(value)[1]
            for value in (operation.lb, operation.ub, operation.step)
        ]
        body = operation.body.block
        induction = self.register("int")
        self.values[body.args[0]] = ("int", induction)
        carried = [self.like(self.get(value)) for value in operation.iter_args]
        self.assign_all(carried, tuple(operation.iter_args))
        for argument, target in zip(body.args[1:], carried, strict=True):
            self.values[argument] = target
        self.emit("begin_for", induction, *bounds)
        terminator = self.block(body)
        assert terminator is not None
        self.assign_all(carried, tuple(terminator.operands))
        self.emit("end_for")
        for result, target in zip(operation.results, carried, strict=True):
            self.values[result] = target

    def if_(self, operation: scf.IfOp) -> None:
        _, condition = self.scalar(operation.cond)
        results: list[tuple[str, object]] | None = None
        self.emit("begin_if", condition)
        for index, region in enumerate(
            (operation.true_region, operation.false_region)
        ):
            if index:
                self.emit("begin_else")
            if not region.blocks:
                continue
            terminator = self.block(region.block)
            operands = tuple(terminator.operands) if terminator else ()
            if results is None:
                results = [self.like(self.get(value)) for value in operands]
            self.assign_all(results, operands)
        self.emit("end_if")
        for result, target in zip(
            operation.results, results or (), strict=True
        ):
            self.values[result] = target

    # exec operations ------------------------------------------------------

    def _arguments(
        self, attributes: ArrayAttr, operands: Sequence[SSAValue]
    ) -> tuple[list[int], list[object]]:
        kinds: list[int] = []
        values: list[object] = []
        for attribute in attributes:
            if isinstance(attribute, ex.OperandRef):
                kind, item = self.operand(operands[attribute.index.data])
                kinds.append(kind)
                values.append(item)
            elif isinstance(attribute, ArrayAttr) and any(
                isinstance(item, ex.OperandRef) for item in attribute
            ):
                buffers = []
                for item in attribute:
                    if isinstance(item, NoneAttr):
                        buffers.append(-1)
                    elif isinstance(item, ex.OperandRef):
                        buffers.append(
                            self.buffer_of(operands[item.index.data])
                        )
                    else:
                        raise LoweringError(
                            "Runner operator lists mix buffers with constants"
                        )
                kinds.append(BUFFER_LIST)
                values.append(buffers)
            else:
                value = _constant_value(attribute)
                if value == []:
                    kinds.append(EMPTY_LIST)
                    values.append(0)
                else:
                    kinds.append(CONSTANT)
                    values.append(value)
        return kinds, values

    def call_op(self, operation: ex.CallOperatorOp) -> None:
        kinds, values = self._arguments(
            operation.arguments, tuple(operation.inputs)
        )
        results = []
        for result in operation.outputs:
            buffer = self.buffer(result.type)
            self.values[result] = ("buffer", buffer)
            results.append(buffer)
        self.emit("call_op", operation.schema.data, kinds, values, results)

    def launch(self, operation: ex.LaunchOp) -> None:
        symbol = operation.kernel.data
        kernel = self.objects_by_symbol[symbol]
        device = cast(torch.device, self.objects_by_symbol[f"{symbol}_device"])
        kinds, values = self._arguments(
            operation.arguments, tuple(operation.inputs)
        )
        grid = list(_integers(operation.grid) or ()) or list(kernel.grid)
        self.emit(
            "launch",
            self.object(kernel),
            device.index if device.index is not None else 0,
            grid,
            kinds,
            values,
            self.signatures[operation],
        )

    def call_schedule(self, operation: ex.CallScheduleOp) -> None:
        callee = self.objects_by_symbol[operation.callee.data]
        groups: dict[int, list[int]] = {
            BUFFER: [],
            INT_REGISTER: [],
            FLOAT_REGISTER: [],
        }
        for value in operation.inputs:
            kind, item = self.operand(value)
            groups[kind].append(item)
        results = []
        for result in operation.outputs:
            buffer = self.buffer(result.type)
            self.values[result] = ("buffer", buffer)
            results.append(buffer)
        self.emit(
            "call",
            self.object(callee),
            groups[BUFFER],
            groups[INT_REGISTER],
            groups[FLOAT_REGISTER],
            results,
        )

    def python_call(self, operation: ex.PythonCallOp) -> None:
        groups: dict[int, list[int]] = {
            BUFFER: [],
            INT_REGISTER: [],
            FLOAT_REGISTER: [],
        }

        def pass_operand(value: SSAValue) -> int:
            kind, item = self.operand(value)
            groups[kind].append(item)
            return kind

        kinds = tuple(pass_operand(value) for value in operation.inputs)
        bodies = []
        captured = []
        for region in operation.bodies:
            block = region.block
            defined: set[SSAValue] = set(block.args)
            for nested in block.walk():
                defined.update(nested.results)
                for inner in nested.regions:
                    for inner_block in inner.blocks:
                        defined.update(inner_block.args)
            # Values read from the enclosing schedule: Runner buffers and
            # registers become body inputs, other bound values stay constant.
            free: list[SSAValue] = []
            for nested in block.walk():
                for value in nested.operands:
                    if value not in defined and value not in free:
                        free.append(value)
            passed = [
                v for v in free if self.get(v)[0] in ("buffer", "int", "float")
            ]
            fixed = {v: self.get(v) for v in free if v not in passed}
            schedule = translate_block(
                block,
                arguments=(
                    *(("buffer", argument.type) for argument in block.args),
                    *((self.get(v)[0], None) for v in passed),
                ),
                objects=self.objects_by_symbol,
                constants=self.constants,
                reference=self.reference,
                describe=self.describe,
                signatures=self.signatures,
                block_arguments=(*block.args, *passed),
                bound=fixed,
            )
            captured.append(tuple(pass_operand(v) for v in passed))
            bodies.append(schedule)
        callee = PythonCallee(
            self.objects_by_symbol[operation.callee.data],
            kinds,
            tuple(bodies),
            tuple(captured),
        )
        results = []
        for result in operation.outputs:
            buffer = self.buffer(result.type)
            self.values[result] = ("buffer", buffer)
            results.append(buffer)
        self.emit(
            "python_call",
            self.object(callee),
            groups[BUFFER],
            groups[INT_REGISTER],
            groups[FLOAT_REGISTER],
            results,
        )


def translate_block(
    block: Block,
    *,
    arguments: Sequence[tuple[str, object]],
    objects: Mapping[str, object],
    constants: Mapping[str, torch.Tensor],
    reference: Callable[[Operation], tuple[str, object]] | None = None,
    describe: Callable[[object], object] = lambda value: value,
    signatures: Mapping[Operation, tuple[object, ...]] = {},
    block_arguments: Sequence[SSAValue] | None = None,
    bound: Mapping[SSAValue, tuple[str, object]] = {},
    read_in_place: Sequence[bool] = (),
) -> Schedule:
    """Translate one exec block into a Runner schedule.

    ``arguments`` gives the kind of each block argument: ``("buffer", type)`` for Tensors, ``("int", None)`` and ``("float", None)`` for host scalars and ``("value", None)`` for objects that the schedule passes through. ``read_in_place`` marks the Tensor arguments that Graph segments read at their address (:attr:`Schedule.read_in_place`). ``objects`` and ``constants`` resolve the builder's kernel, callable, resource and constant symbols; ``reference`` resolves Program material and resource references.
    """

    def no_reference(operation: Operation) -> tuple[str, object]:
        raise LoweringError(f"{operation.name} has no bound value")

    state = _Translation(
        objects,
        constants,
        reference or no_reference,
        describe,
        signatures,
    )
    state.values.update(bound)
    inputs = []
    in_place: list[int] = []
    for index, (value, (kind, value_type)) in enumerate(
        zip(
            block.args if block_arguments is None else block_arguments,
            arguments,
            strict=True,
        )
    ):
        if kind == "buffer":
            buffer = state.buffer(cast(Attribute, value_type), "input")
            state.values[value] = ("buffer", buffer)
            inputs.append(("buffer", buffer))
            if index < len(read_in_place) and read_in_place[index]:
                in_place.append(buffer)
        elif kind in ("int", "float"):
            register = state.register(kind)
            state.values[value] = (kind, register)
            inputs.append((kind, register))
        else:
            state.values[value] = ("input", index)
            inputs.append(("value", index))
    terminator = state.block(block)
    output_buffers: list[int] = []
    outputs: list[tuple[str, object]] = []
    for value in terminator.operands if terminator is not None else ():
        kind, item = state.get(value)
        if kind == "buffer":
            outputs.append(("buffer", len(output_buffers)))
            output_buffers.append(cast(int, item))
        elif kind == "input":
            outputs.append(("input", item))
        elif kind == "value":
            outputs.append(("value", item))
        else:
            raise LoweringError("Runner schedules return Tensor buffers only")
    return Schedule(
        tuple(state.buffers),
        state.int_registers,
        state.float_registers,
        # Groups are declared before the instructions that use them.
        (*(("group",) for _ in state.groups), *state.instructions),
        tuple(inputs),
        tuple(output_buffers),
        tuple(outputs),
        tuple(state.bindings),
        tuple(state.objects),
        groups=tuple(state.groups),
        read_in_place=tuple(in_place),
    )


def argument_kind(value_type: Attribute) -> str:
    """Return the Runner argument kind of an exec block argument type."""
    if isinstance(value_type, ex.BufferType):
        return "buffer"
    if isinstance(value_type, (IndexType, IntegerType)):
        return "int"
    if isinstance(value_type, _FloatType):
        return "float"
    return "value"


# Realization --------------------------------------------------------------------


def runner_class() -> Any:
    """Return the libfhelium Runner class registered with PyTorch."""
    # Importing the wrapper loads libfhelium_torch, which registers the class.
    import fhelium.native.wrapper  # noqa: F401

    return torch.classes.fhelium.Runner


def _index_value(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return int(value.item())
    if isinstance(value, int):
        return int(value)
    raise TypeError("Host integer inputs must be integer values")


class ScheduleFunction:
    """Execute one schedule on a Runner built at the first call.

    ``materials`` supplies material Tensors by symbol and ``resources`` the linked resources that :class:`Resources` selections name. Kernels compile or load from Triton's cache when the Runner is built, so a caller that compiles them concurrently beforehand only loads them.

    With ``arena`` the Runner keeps the schedule's non-escaping allocations in one region per device, planned when the Runner is built (:func:`~fhelium.backend._planning.plan_schedule`); the buffer stays allocated while the Runner lives. With ``shared`` as well, that buffer is the device's :func:`shared_arena`, which every shared function on the device uses for its temporaries one execution at a time. An execution that starts while another shared execution on the device has not returned, such as one made by a Python implementation inside a schedule or by another thread, runs on a Runner with a private arena instead. Eager fragments leave ``arena`` off, so their temporaries return to the caching allocator after each call. :meth:`graphed` returns the function that runs the same schedule with CUDA Graph segments.
    """

    def __init__(
        self,
        schedule: Schedule,
        materials: Mapping[str, object] = {},
        resources: Sequence[object] = (),
        *,
        arena: bool = False,
        graphs: bool = False,
        adaptive: bool = False,
        shared: bool = False,
    ) -> None:
        self.schedule = schedule
        self.materials = materials
        self.resources = tuple(resources)
        self.arena = arena or graphs
        self.graphs = graphs
        self.adaptive = adaptive
        """With ``graphs``, each Graph segment keeps its graph only when its replays are faster than its direct execution."""
        self.shared = shared and self.arena
        self.plan: Any = None
        """The :class:`~fhelium.backend._planning.PlanReport` once the schedule is planned."""
        self._planned: Schedule | None = None
        self._device: int | None = None
        """CUDA device index of the shared arena, once planned."""
        self._graphed: dict[object, ScheduleFunction] = {}
        self._runner: Any = None
        self._arena: Any = None
        self._private: Any = None
        self._bodies: dict[int, tuple[ScheduleFunction, ...]] = {}
        kinds = [kind for kind, _ in schedule.inputs]
        self._tensors = tuple(i for i, k in enumerate(kinds) if k == "buffer")
        self._ints = tuple(i for i, k in enumerate(kinds) if k == "int")
        self._floats = tuple(i for i, k in enumerate(kinds) if k == "float")
        self._direct = all(kind == "buffer" for kind, _ in schedule.outputs)
        self._python = schedule.python_calls

    def with_materials(
        self, materials: Mapping[str, object]
    ) -> ScheduleFunction:
        """Return a function of the same schedule with other material bindings."""
        return ScheduleFunction(
            self.schedule,
            {**self.materials, **materials},
            self.resources,
            arena=self.arena,
            graphs=self.graphs,
            adaptive=self.adaptive,
            shared=self.shared,
        )

    def graphed(
        self, key: object = None, *, adaptive: bool = False
    ) -> ScheduleFunction:
        """Return a function of this schedule whose Runner captures Graph segments.

        Each ``key`` selects its own Runner and captures, for callers that execute the schedule under different conditions such as grad or autocast modes. With ``adaptive``, which the first request for a ``key`` fixes, each segment compares the device time of its first direct execution with the median of three replays and keeps its graph only when the replays are faster; otherwise it releases the graph and executes directly from then on. Results the Runner keeps in its output region are returned without a copy; before its next execution overwrites them, the Runner moves the results the caller still references to fresh storage.
        """
        function = self._graphed.get(key)
        if function is None:
            function = self._graphed[key] = ScheduleFunction(
                self.schedule,
                self.materials,
                self.resources,
                graphs=True,
                adaptive=adaptive,
                shared=self.shared,
            )
        return function

    def _schedule(self) -> Schedule:
        planned = self._planned
        if planned is None:
            planned = self.schedule
            if self.arena:
                from ._planning import plan_schedule

                planned, self.plan = plan_schedule(planned, graphs=self.graphs)
                devices = [
                    torch.device(name)
                    for name, size in self.plan.arena_bytes
                    if size
                ]
                if (
                    self.shared
                    and len(devices) == 1
                    and devices[0].type == "cuda"
                ):
                    self._device = devices[0].index or 0
            self._planned = planned
        return planned

    def runner(self, *, private: bool = False) -> Any:
        """Return the Runner, building it at the first request.

        ``private`` selects a Runner whose arena no other function shares, for executions nested in another shared execution and for schedule calls.
        """
        schedule = self._schedule()
        if private and self._device is not None:
            if self._private is None:
                self._private = _realize(
                    schedule,
                    self.materials,
                    self.resources,
                    adaptive=self.adaptive,
                )
                if schedule.groups:
                    _GROUP_HOLDERS[self] = frozenset(
                        group_name(group.resource) for group in schedule.groups
                    )
            return self._private
        runner = self._runner
        if runner is None:
            if self._device is not None:
                self._arena = shared_arena(self._device)
            runner = self._runner = _realize(
                schedule,
                self.materials,
                self.resources,
                self._arena,
                adaptive=self.adaptive,
            )
            if schedule.groups and not isinstance(self, ArtifactFunction):
                _GROUP_HOLDERS[self] = frozenset(
                    group_name(group.resource) for group in schedule.groups
                )
        return runner

    def release_groups(self) -> None:
        """Release the group references and captured graphs of this function's Runners; the next execution builds them again."""
        for runner in (self._runner, self._private):
            if runner is not None:
                runner.release_groups()
        self._runner = self._private = None

    def __call__(self, *payloads: object) -> tuple[object, ...]:
        runner = self._runner
        if runner is None:
            runner = self.runner()
        tensors = [payloads[i] for i in self._tensors]
        ints = [_index_value(payloads[i]) for i in self._ints]
        floats = [float(cast(float, payloads[i])) for i in self._floats]
        device = self._device
        if device is None:
            outputs = self._execute(runner, tensors, ints, floats)
        elif _ACTIVE_SHARED[device]:
            outputs = self._execute(
                self.runner(private=True), tensors, ints, floats
            )
        else:
            _ACTIVE_SHARED[device] += 1
            try:
                outputs = self._execute(runner, tensors, ints, floats)
            finally:
                _ACTIVE_SHARED[device] -= 1
        if self._direct:
            return tuple(outputs)
        return tuple(
            outputs[cast(int, item)]
            if kind == "buffer"
            else payloads[cast(int, item)]
            if kind == "input"
            else item
            for kind, item in self.schedule.outputs
        )

    def _execute(
        self, runner: Any, tensors, ints, floats
    ) -> list[torch.Tensor]:
        if self._python:
            return self._drive(runner, tensors, ints, floats)
        return runner.run(tensors, ints, floats)

    def _drive(self, runner: Any, tensors, ints, floats) -> list[torch.Tensor]:
        """Run a schedule that pauses at Python calls."""
        execution = runner.start(tensors, ints, floats)
        while (index := execution.pending()) >= 0:
            results = self._python_call(
                index,
                list(execution.arguments()),
                list(execution.integer_arguments()),
                list(execution.float_arguments()),
            )
            execution.resume(list(results))
        return execution.outputs()

    def _python_call(
        self,
        index: int,
        tensors: list[torch.Tensor],
        ints: list[int],
        floats: list[float],
    ) -> Sequence[torch.Tensor]:
        callee = cast(PythonCallee, self.schedule.objects[index])
        iterators = {
            BUFFER: iter(tensors),
            INT_REGISTER: iter(ints),
            FLOAT_REGISTER: iter(floats),
        }
        operands = tuple(next(iterators[kind]) for kind in callee.kinds)
        if not callee.bodies:
            return tuple(callee.function(*operands))
        bodies = self._bodies.get(index)
        if bodies is None:
            bodies = self._bodies[index] = tuple(
                ScheduleFunction(body, self.materials, self.resources)
                for body in callee.bodies
            )
        functions = tuple(
            _BodyCall(body, tuple(next(iterators[kind]) for kind in kinds))
            for body, kinds in zip(bodies, callee.captured, strict=True)
        )
        return tuple(callee.function(operands, functions))


class LoadedArtifact:
    """A loaded artifact (``torch.classes.fhelium.Artifact``) with its decoded manifest."""

    def __init__(self, path: str, device: torch.device) -> None:
        import json

        from fhelium.native.artifact import load_artifact

        self.path = path
        self.device = device
        self.artifact = load_artifact(path, device)
        self.manifest: dict[str, Any] = json.loads(self.artifact.manifest())
        self.materials: dict[str, Mapping[str, Any]] = {
            record["symbol"]: record
            for records in self.manifest.get("materials", {}).values()
            for record in records
        }

    def bind(self, materials: Mapping[str, object]) -> None:
        """Bind the Tensors of the declared material symbols in ``materials``."""
        for symbol, value in materials.items():
            if symbol in self.materials:
                self.artifact.bind(symbol, value)

    def bind_group(self, symbol: str, name: str) -> None:
        """Bind the group ``symbol`` of the artifact's entries to the c10d group registered as ``name``."""
        self.artifact.bind_group(symbol, name)
        _GROUP_HOLDERS[self] = _GROUP_HOLDERS.get(self, frozenset()) | {name}

    def release_groups(self) -> None:
        """Release the group references and captured graphs of the Runners built from the artifact; the groups are bound again with :meth:`bind_group`."""
        self.artifact.release_groups()

    def function(self, name: str) -> ArtifactFunction:
        return ArtifactFunction(self, name)


class ArtifactFunction(ScheduleFunction):
    """Execute one entry of a :class:`LoadedArtifact`.

    The entry's planned schedule, kernels, constants and materials come from the artifact, whose Runners share them; ``schedule`` carries only the entry's inputs and outputs. The entry named ``<name>.graphs``, when present, holds the same schedule planned with Graph segments, which :meth:`graphed` runs.
    """

    def __init__(
        self,
        loaded: LoadedArtifact,
        name: str,
        *,
        graphs: bool = False,
        adaptive: bool = False,
    ) -> None:
        record = loaded.manifest["entries"][name]
        super().__init__(
            Schedule(
                buffers=(),
                int_registers=0,
                float_registers=0,
                instructions=(),
                inputs=tuple((kind, item) for kind, item in record["inputs"]),
                output_buffers=tuple(record["output_buffers"]),
                outputs=tuple((kind, item) for kind, item in record["outputs"]),
                bindings=(),
                objects=(),
                graphs=graphs,
            ),
            arena=True,
            graphs=graphs,
            adaptive=adaptive,
            shared=True,
        )
        self.loaded = loaded
        self.name = name
        devices = [
            torch.device(device)
            for device, size in (record.get("arena_bytes") or {}).items()
            if size
        ]
        if len(devices) == 1 and devices[0].type == "cuda":
            self._device = devices[0].index or 0
        self._planned = self.schedule

    def with_materials(
        self, materials: Mapping[str, object]
    ) -> ArtifactFunction:
        """Bind ``materials`` in the artifact and return a function whose Runners read them.

        Runners built before keep the materials they were built with.
        """
        self.loaded.bind(materials)
        return ArtifactFunction(
            self.loaded, self.name, graphs=self.graphs, adaptive=self.adaptive
        )

    def graphed(
        self, key: object = None, *, adaptive: bool = False
    ) -> ScheduleFunction:
        function = self._graphed.get(key)
        if function is None:
            name = f"{self.name}.graphs"
            function = self._graphed[key] = (
                ArtifactFunction(
                    self.loaded, name, graphs=True, adaptive=adaptive
                )
                if name in self.loaded.manifest["entries"]
                else self
            )
        return function

    def runner(self, *, private: bool = False) -> Any:
        mode = ("auto" if self.adaptive else "on") if self.graphs else "off"
        artifact = self.loaded.artifact
        if private and self._device is not None:
            if self._private is None:
                self._private = artifact.runner(
                    self.name, None, mode, PROFILE_SEGMENTS
                )
            return self._private
        if self._runner is None:
            if self._device is not None:
                self._arena = shared_arena(self._device)
            self._runner = artifact.runner(
                self.name, self._arena, mode, PROFILE_SEGMENTS
            )
        return self._runner


class _BodyCall:
    """Call one Python-call body schedule with its captured values."""

    def __init__(self, function: ScheduleFunction, captured: tuple) -> None:
        self.function = function
        self.captured = captured

    def __call__(
        self, values: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.function(*values, *self.captured)
        if any(not isinstance(output, torch.Tensor) for output in outputs):
            raise TypeError("Operation combine regions must return Tensors")
        return cast(tuple[torch.Tensor, ...], outputs)


def fragment_function(schedule: Schedule) -> Callable[..., tuple[object, ...]]:
    """Return the function of a lowered Eager fragment.

    A fragment that consists of one operator call on its inputs and link-time constants and returns that call's results calls the operator directly: for such fragments the dispatcher call costs less host time than a Runner execution. Every other fragment runs on a Runner.
    """
    direct = _direct_call(schedule)
    return ScheduleFunction(schedule) if direct is None else direct


def _direct_call(
    schedule: Schedule,
) -> Callable[..., tuple[object, ...]] | None:
    if (
        len(schedule.instructions) != 1
        or schedule.instructions[0][0] != "call_op"
    ):
        return None
    _, name, kinds, values, results = schedule.instructions[0]
    if list(schedule.output_buffers) != list(cast(list, results)) or any(
        kind != "buffer" for kind, _ in schedule.outputs
    ):
        return None
    operator = resolve_operator(cast(str, name))
    returns = operator._schema.returns
    if any(item.alias_info is not None for item in returns):
        return None
    positions: dict[tuple[str, int], int] = {
        (kind, item): index
        for index, (kind, item) in enumerate(schedule.inputs)
    }
    constants = {
        buffer: binding.tensor
        for buffer, binding in schedule.bindings
        if isinstance(binding, Constant)
    }
    if len(constants) != len(schedule.bindings):
        return None

    def source(kind: int, item: object) -> tuple[int, object] | None:
        """Return (payload index, None) or (-1, constant) for one argument."""
        if kind == CONSTANT:
            return (-1, item)
        if kind == EMPTY_LIST:
            return (-1, [])
        if kind == BUFFER:
            buffer = cast(int, item)
            if ("buffer", buffer) in positions:
                return (positions[("buffer", buffer)], None)
            if buffer in constants:
                return (-1, constants[buffer])
            return None
        register = ("int" if kind == INT_REGISTER else "float", cast(int, item))
        return (positions[register], None) if register in positions else None

    arguments = []
    for kind, item in zip(cast(list, kinds), cast(list, values), strict=True):
        if kind == BUFFER_LIST:
            entries = [
                (-1, None) if buffer == -1 else source(BUFFER, buffer)
                for buffer in cast(list, item)
            ]
            if any(entry is None for entry in entries):
                return None
            arguments.append(("list", entries))
        else:
            entry = source(kind, item)
            if entry is None:
                return None
            arguments.append(("value", entry))
    unpack = len(returns) == 1 and str(returns[0].type).startswith("List")
    single = len(returns) == 1 and not unpack
    keywords = tuple(
        argument.name
        for argument in operator._schema.arguments
        if argument.kwarg_only
    )
    if not keywords and all(
        kind == "value" and entry[0] >= 0 for kind, entry in arguments
    ):
        indices = tuple(entry[0] for _, entry in arguments)
        if indices == tuple(range(len(schedule.inputs))):
            if single:
                return lambda *payloads: (operator(*payloads),)
            return lambda *payloads: tuple(operator(*payloads))

    def call(*payloads: object) -> tuple[object, ...]:
        args = [
            (payloads[entry[0]] if entry[0] >= 0 else entry[1])
            if kind == "value"
            else [
                payloads[index] if index >= 0 else value
                for index, value in entry
            ]
            for kind, entry in arguments
        ]
        if keywords:
            count = len(args) - len(keywords)
            output = operator(
                *args[:count], **dict(zip(keywords, args[count:], strict=True))
            )
        else:
            output = operator(*args)
        return (output,) if single else tuple(output)

    return call


PROFILE_SEGMENTS = False
"""Whether Graph Runners built from now on record device times of their segments' direct runs and replays for ``segment_stats``."""

_SHARED_ARENAS: dict[int, weakref.ref[Any]] = {}
_ACTIVE_SHARED: collections.Counter[int] = collections.Counter()
"""Shared executions in progress per CUDA device."""


def shared_arena(device: int) -> Any:
    """Return the ``torch.classes.fhelium.Arena`` shared by the compiled schedules on one CUDA device.

    It grows to the largest arena requirement of the Runners that use it; a growth makes each of them derive its arena views again and capture its Graph segments again at its next execution. The functions whose Runners use it keep it alive, so its memory returns to the caching allocator once they are all released.
    """
    reference = _SHARED_ARENAS.get(device)
    arena = None if reference is None else reference()
    if arena is None:
        runner_class()
        arena = torch.classes.fhelium.Arena(device)
        _SHARED_ARENAS[device] = weakref.ref(arena)
    return arena


def compiled_execution_bytes(device: int) -> int:
    """Return the device memory that compiled schedules hold on one CUDA device: the shared arena and the CUDA Graph pool of its Runners."""
    reference = _SHARED_ARENAS.get(device)
    arena = None if reference is None else reference()
    if arena is None:
        return 0
    return int(arena.bytes()) + int(arena.graph_pool_bytes())


def _realize(
    schedule: Schedule,
    materials: Mapping[str, object],
    resources: Sequence[object],
    arena: Any = None,
    *,
    adaptive: bool = False,
) -> Any:
    runner = runner_class()()
    if schedule.graphs:
        runner.graphs("auto" if adaptive else "on")
        if PROFILE_SEGMENTS:
            runner.profile_segments(True)
    if arena is not None:
        runner.share_arena(arena)
    for storage, sizes, strides, dtype, device in schedule.buffers:
        runner.buffer(
            storage,
            None if sizes is None else list(sizes),
            None if strides is None else list(strides),
            dtype,
            device,
        )
    for _ in range(schedule.int_registers):
        runner.int_register()
    for _ in range(schedule.float_registers):
        runner.float_register()
    kernels: dict[
        tuple[int, tuple[object, ...]], tuple[int, tuple[int, ...]]
    ] = {}
    callees: dict[int, Any] = {}
    for instruction in schedule.instructions:
        name = instruction[0]
        if name == "launch":
            reference, device, grid, kinds, values, signature = instruction[1:]
            kernel = schedule.objects[cast(ObjectRef, reference).index]
            key = (cast(ObjectRef, reference).index, cast(tuple, signature))
            entry = kernels.get(key)
            if entry is None:
                record = kernel.record(*cast(tuple, signature))
                entry = kernels[key] = (
                    runner.kernel(*record.arguments()),
                    record.runtime_positions,
                )
            identifier, positions = entry
            runner.launch(
                identifier,
                device,
                grid,
                [cast(list, kinds)[i] for i in positions],
                [cast(list, values)[i] for i in positions],
            )
        elif name == "call":
            reference = cast(ObjectRef, instruction[1])
            callee = callees.get(reference.index)
            if callee is None:
                callee = callees[reference.index] = cast(
                    CalledSchedule, schedule.objects[reference.index]
                ).runner()
            runner.call(callee, *instruction[2:])
        elif name == "python_call":
            runner.python_call(
                cast(ObjectRef, instruction[1]).index, *instruction[2:]
            )
        else:
            getattr(runner, cast(str, name))(*instruction[1:])
    runner.inputs(
        [item for kind, item in schedule.inputs if kind == "buffer"],
        [item for kind, item in schedule.inputs if kind == "int"],
        [item for kind, item in schedule.inputs if kind == "float"],
    )
    if schedule.read_in_place:
        runner.read_in_place(list(schedule.read_in_place))
    runner.outputs(list(schedule.output_buffers))
    for buffer, binding in schedule.bindings:
        runner.bind(buffer, _bound_value(binding, materials, resources))
    for index, group in enumerate(schedule.groups):
        runner.bind_group(index, group_name(group.resource))
    runner.finalize()
    return runner


def _bound_value(
    binding: object,
    materials: Mapping[str, object],
    resources: Sequence[object],
) -> torch.Tensor:
    if isinstance(binding, Material):
        return cast(torch.Tensor, materials[binding.symbol])
    if isinstance(binding, Constant):
        return binding.tensor
    if isinstance(binding, State):
        resource = binding.resource
        if isinstance(resource, Resources):
            selected = resources[resource.indices[0]]
            resource = selected.value if resource.value else selected  # type: ignore[attr-defined]
        return cast(torch.Tensor, getattr(resource, binding.field))
    raise TypeError(f"Unsupported buffer binding {binding!r}")


__all__ = [
    "release_groups",
    "ArtifactFunction",
    "CalledSchedule",
    "Constant",
    "LoadedArtifact",
    "Material",
    "ObjectRef",
    "PythonCallee",
    "Resources",
    "Schedule",
    "ScheduleFunction",
    "State",
    "argument_kind",
    "compiled_execution_bytes",
    "fragment_function",
    "runner_class",
    "translate_block",
]
