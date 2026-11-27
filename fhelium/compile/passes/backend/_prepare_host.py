"""Lower a linked Program to an exec schedule and build its Runner function."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING, cast

import torch
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    _FloatType,
    IndexType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Operation, SSAValue

from fhelium.backend.lowering import (
    ExecBuilder,
    HostInt,
    HostScalar,
    Layout,
    LoweringError,
    isolated,
    remove_dead_operations,
)
from fhelium.backend.runner import (
    CalledSchedule,
    Resources,
    Schedule,
    ScheduleFunction,
    translate_block,
)
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core
from fhelium.ir.dialects import exec as ex

from ..codegen._common import constant_literal

if TYPE_CHECKING:
    from fhelium.backend.execution import ProgramExecutable
    from fhelium.backend.resources import BoundResource

HOST_INTEGER = "int"
HOST_FLOAT = "float"
OPAQUE = "object"
"""Input kinds of Program values that are not Tensors."""

InputLayout = Layout | str


@dataclass(frozen=True)
class LoweredProgram:
    """One exec schedule of a linked Program for fixed input layouts."""

    program: Program | None
    """The exec Program; None for a schedule restored from an artifact."""
    function: ScheduleFunction
    result_layouts: tuple[Layout | None, ...]

    @property
    def schedule(self) -> Schedule:
        return self.function.schedule

    @property
    def launches(self) -> tuple[tuple[Any, tuple[object, ...]], ...]:
        """Kernels launched by the schedule with their warm signatures."""
        return self.schedule.launches


def static_input_layout(value_type: Attribute) -> InputLayout | None:
    """Return the input layout declared by a Program argument type, or None when it is unknown."""
    if isinstance(value_type, IndexType):
        return HOST_INTEGER
    if isinstance(value_type, IntegerType):
        return HOST_INTEGER
    state = getattr(getattr(value_type, "state", None), "data", None)
    if not isinstance(state, Mapping):
        return None
    fields = []
    for name in ("shape", "strides"):
        attribute = state.get(name)
        if not isinstance(attribute, ArrayAttr) or not all(
            isinstance(item, IntegerAttr) for item in attribute
        ):
            return None
        fields.append(
            tuple(int(cast(IntegerAttr, item).value.data) for item in attribute)
        )
    dtype, device = state.get("dtype"), state.get("device")
    if not isinstance(dtype, StringAttr) or not isinstance(device, StringAttr):
        return None
    return Layout(
        fields[0],
        fields[1],
        getattr(torch, dtype.data.removeprefix("torch.")),
        torch.device(device.data),
    )


def runtime_input_layout(value: object) -> InputLayout:
    """Return the input layout of one runtime payload."""
    if isinstance(value, torch.Tensor):
        return Layout.of(value)
    if isinstance(value, bool) or isinstance(value, int):
        return HOST_INTEGER
    if isinstance(value, float):
        return HOST_FLOAT
    return OPAQUE


class _Opaque:
    """Program value that lowering may pass along but not read."""


def lower_program(
    executable: ProgramExecutable,
    input_layouts: Sequence[InputLayout],
) -> LoweredProgram:
    """Lower every dispatched operation of a linked Program for fixed input layouts and build its Runner function."""
    with isolated():
        return _lower_program(executable, input_layouts)


def _lower_program(
    executable: ProgramExecutable,
    input_layouts: Sequence[InputLayout],
) -> LoweredProgram:
    original = executable.program
    program = Program(original.module.clone())
    correspondence = dict(
        zip(program.module.walk(), original.module.walk(), strict=True)
    )
    block = program.single_block("main")
    builder = ExecBuilder(destinations=True)
    environment: dict[SSAValue, object] = {}
    fixed_rank: list[bool] = []
    operations = executable.dispatch_table.operations

    def bind_input(value: SSAValue, layout: InputLayout) -> None:
        if isinstance(layout, Layout):
            environment[value] = builder.placeholder(value, layout)
        elif layout == HOST_INTEGER:
            environment[value] = builder.host_integer(value, 0)
        elif layout == HOST_FLOAT:
            environment[value] = builder.host_scalar(value, 0.0)
        else:
            environment[value] = _Opaque()

    def bind_like(value: SSAValue, sample: object) -> None:
        if isinstance(sample, torch.Tensor):
            environment[value] = builder.placeholder(value, Layout.of(sample))
        elif isinstance(sample, HostInt):
            environment[value] = builder.host_integer(value, int(sample))
        elif isinstance(sample, (HostScalar, float)):
            environment[value] = builder.host_scalar(value, float(sample))
        elif isinstance(sample, int):
            environment[value] = builder.host_integer(value, sample)
        else:
            environment[value] = _Opaque()

    def normalize(
        yielded: Sequence[SSAValue],
        targets: Sequence[object],
        before: Operation,
    ) -> list[SSAValue]:
        """Copy yielded Tensors whose layout differs from the carried layout."""
        values = []
        for value, target in zip(yielded, targets, strict=True):
            current = environment.get(value)
            if (
                isinstance(target, torch.Tensor)
                and isinstance(current, torch.Tensor)
                and Layout.of(current) != Layout.of(target)
            ):
                copy = builder.alloc(
                    target.shape,
                    strides=target.stride(),
                    dtype=target.dtype,
                    device=target.device,
                )
                builder.copy(copy, current)
                parent = before.parent_block()
                assert parent is not None
                parent.insert_ops_before(builder.drain(), before)
                value = builder.value(copy)
            values.append(value)
        return values

    def lower_block(current: Block) -> None:
        lower_operations(list(current.ops))

    def lower_operations(sequence: Sequence[Operation]) -> None:
        for operation in sequence:
            source = correspondence.get(operation)
            dispatch = operations.get(source) if source is not None else None
            if dispatch is not None:
                lower_operation(operation, dispatch)
            elif isinstance(operation, UnrealizedConversionCastOp):
                for value, result in zip(
                    operation.inputs, operation.outputs, strict=True
                ):
                    environment[result] = environment.get(value, _Opaque())
            elif isinstance(operation, core.MaterialRefOp):
                material = executable.bound_materials[
                    cast(core.MaterialRefOp, source)
                ]
                if isinstance(material, torch.Tensor):
                    environment[operation.results[0]] = builder.placeholder(
                        operation.results[0], material
                    )
                else:
                    environment[operation.results[0]] = material
            elif isinstance(operation, core.ResourceRefOp):
                environment[operation.results[0]] = executable.bound_resources[
                    cast(Operation, source)
                ]
            elif isinstance(operation, core.ConstantOp):
                environment[operation.results[0]] = constant_literal(operation)
            elif isinstance(operation, arith.ConstantOp):
                value = cast(
                    IntegerAttr | FloatAttr, operation.value
                ).value.data
                environment[operation.results[0]] = value
            elif isinstance(operation, scf.ForOp):
                lower_for(operation)
            elif isinstance(operation, scf.IfOp):
                lower_if(operation)
            elif isinstance(operation, (ReturnOp, scf.YieldOp)):
                continue
            elif operation.dialect_name() == "arith":
                for result in operation.results:
                    bind_like(
                        result,
                        0.0 if isinstance(result.type, _FloatType) else 0,
                    )
            elif operation.regions or operation.results:
                raise LoweringError(
                    f"Program operation {operation.name} has no Backend dispatch"
                )

    def lower_for(operation: scf.ForOp) -> None:
        body = operation.body.block
        environment[body.args[0]] = builder.host_integer(body.args[0], 0)
        initial = [environment.get(value) for value in operation.iter_args]
        for argument, sample in zip(body.args[1:], initial, strict=True):
            bind_like(argument, sample)
        lower_block(body)
        terminator = cast(scf.YieldOp, body.last_op)
        carried = [environment.get(argument) for argument in body.args[1:]]
        replacement = normalize(tuple(terminator.operands), carried, terminator)
        for index, value in enumerate(replacement):
            terminator.operands[index] = value
        for result, sample in zip(operation.results, initial, strict=True):
            bind_like(result, sample)

    def lower_if(operation: scf.IfOp) -> None:
        lower_block(operation.true_region.block)
        true_yield = cast(scf.YieldOp, operation.true_region.block.last_op)
        targets = [environment.get(value) for value in true_yield.operands]
        lower_block(operation.false_region.block)
        false_yield = cast(scf.YieldOp, operation.false_region.block.last_op)
        replacement = normalize(
            tuple(false_yield.operands), targets, false_yield
        )
        for index, value in enumerate(replacement):
            false_yield.operands[index] = value
        for result, sample in zip(operation.results, targets, strict=True):
            bind_like(result, sample)

    def inline(region, arguments: tuple[object, ...]) -> tuple[object, ...]:
        # Record the region's operations into the lowering that is running,
        # with ``arguments`` for its block arguments, so they become ordinary
        # steps of the schedule.
        block = region.block
        values: dict[SSAValue, object] = dict(
            zip(block.args, arguments, strict=True)
        )
        for operation in block.ops:
            if operation is block.last_op:
                return tuple(values[value] for value in operation.operands)
            if isinstance(operation, UnrealizedConversionCastOp):
                for value, result in zip(
                    operation.inputs, operation.outputs, strict=True
                ):
                    values[result] = values.get(value, environment.get(value))
                continue
            source = correspondence.get(operation)
            dispatch = operations.get(source) if source is not None else None
            if dispatch is None or operation.regions:
                raise LoweringError(
                    f"Region operation {operation.name} has no Backend dispatch"
                )
            items = [
                values[value] if value in values else environment[value]
                for value in operation.operands
            ]
            assert source is not None
            resources = (
                *(
                    cast("BoundResource", item)
                    for item in items
                    if _is_resource(item)
                ),
                *(
                    executable.resources[index]
                    for index in executable.resource_indices[source]
                ),
            )
            results = dispatch.implementation.lower(
                dispatch.invocation,
                tuple(
                    cast(torch.Tensor, item)
                    for item in items
                    if not _is_resource(item)
                ),
                resources,
                builder,
            )
            values.update(zip(operation.results, results, strict=True))
        raise LoweringError("Region has no terminator")

    builder._inline = inline

    def lower_operation(operation: Operation, dispatch) -> None:
        # References and constants in the regions an implementation inlines
        # are lowered before the operation.
        hoisted = [
            inner
            for region in operation.regions
            for inner in region.block.ops
            if isinstance(
                inner,
                (
                    core.MaterialRefOp,
                    core.ResourceRefOp,
                    core.ConstantOp,
                    arith.ConstantOp,
                ),
            )
        ]
        if hoisted:
            owner = operation.parent_block()
            assert owner is not None
            for inner in hoisted:
                inner.detach()
                owner.insert_op_before(inner, operation)
            lower_operations(hoisted)
        resource_operands = []
        inputs = []
        for value in operation.operands:
            item = environment.get(value)
            if item is None:
                raise LoweringError(
                    f"Operand of {operation.name} has no lowered value"
                )
            if _is_resource(item):
                resource_operands.append(item)
            else:
                inputs.append(item)
        source = correspondence[operation]
        indices = tuple(executable.resource_indices[source])
        resources = (
            *resource_operands,
            *(executable.resources[index] for index in indices),
        )
        builder.regions = tuple(operation.regions)
        with builder.tracing():
            results = tuple(
                dispatch.implementation.lower(
                    dispatch.invocation, tuple(inputs), resources, builder
                )
            )
        builder.regions = ()
        if len(results) != len(operation.results):
            raise LoweringError(
                f"Implementation {dispatch.implementation.name!r} returned "
                f"{len(results)} values for {len(operation.results)} results"
            )
        parent = operation.parent_block()
        assert parent is not None
        indices = []
        lowered = []
        for result, value in zip(operation.results, results, strict=True):
            if isinstance(value, HostInt) and _drives_control_flow(result):
                # Structured control flow is expanded when the schedule is
                # lowered, so it uses the value the group has now.
                query = builder._host[value.index].owner
                if (
                    isinstance(query, ex.GroupQueryOp)
                    and query.kind.data == "rank"
                ):
                    fixed_rank.append(True)
                index = arith.ConstantOp(
                    IntegerAttr(int(value), cast(Any, result.type))
                )
                indices.append(index)
                lowered.append(index.result)
            elif isinstance(value, HostInt):
                # A host integer the schedule reads when it runs.
                with builder.tracing():
                    tensor = torch.ops.aten.full.default(
                        [], value, dtype=torch.int64, device=torch.device("cpu")
                    )
                lowered.append(builder.value(tensor))
            elif isinstance(result.type, (IndexType, IntegerType)):
                # Other host indices drive structured control flow, so their
                # value must be known when the schedule is lowered.
                if builder.is_placeholder(value) or not (
                    isinstance(value, torch.Tensor) and value.numel() == 1
                ):
                    raise LoweringError(
                        f"Implementation {dispatch.implementation.name!r} "
                        "computes a host index at execution time"
                    )
                index = arith.ConstantOp(
                    IntegerAttr(int(value.item()), result.type)
                )
                indices.append(index)
                lowered.append(index.result)
            else:
                lowered.append(builder.value(value))
        parent.insert_ops_before((*builder.drain(), *indices), operation)
        for result, value, item in zip(
            operation.results, lowered, results, strict=True
        ):
            if isinstance(result.type, (IndexType, IntegerType)) and not (
                isinstance(item, HostInt)
            ):
                item = int(cast(torch.Tensor, item).item())
            environment[value] = item
            result.replace_all_uses_with(value)
        for body, arguments in builder.drain_bodies():
            for argument, sample in zip(
                body.block.args, arguments, strict=True
            ):
                bind_like(argument, sample)
            lower_block(body.block)
        operation.detach()
        operation.erase(safe_erase=False)

    for argument, layout in zip(block.args, input_layouts, strict=True):
        bind_input(argument, layout)
    try:
        lower_block(block)
    finally:
        # The mutually recursive closures form a reference cycle that would
        # otherwise retain the executable and its materials until collection.
        lower_block = lower_operations = lower_for = lower_if = None  # type: ignore[assignment]  # noqa: F841
        lower_operation = inline = builder._inline = None  # type: ignore[assignment]  # noqa: F841
    remove_dead_operations(block)
    terminator = block.last_op
    result_layouts = tuple(
        Layout.of(item) if isinstance(item, torch.Tensor) else None
        for item in (
            environment.get(value)
            for value in (terminator.operands if terminator is not None else ())
        )
    )
    schedule = _translate(
        executable, block, builder, correspondence, input_layouts
    )
    if fixed_rank or any(
        isinstance(item, CalledSchedule) and item.function.schedule.fixed_rank
        for item in schedule.objects
    ):
        schedule = replace(schedule, fixed_rank=True)
    materials = {
        cast(StringAttr, operation.symbol).data: value
        for operation, value in executable.bound_materials.items()
        if operation.symbol is not None
    }
    return LoweredProgram(
        program,
        ScheduleFunction(
            schedule, materials, executable.resources, arena=True, shared=True
        ),
        result_layouts,
    )


def _is_resource(value: object) -> bool:
    from fhelium.backend.resources import BoundResource

    return isinstance(value, BoundResource)


def _translate(
    executable: ProgramExecutable,
    block: Block,
    builder: ExecBuilder,
    correspondence: dict[Operation, Operation],
    input_layouts: Sequence[InputLayout],
) -> Schedule:
    live: dict[int, tuple[int, bool]] = {}
    for index, resource in enumerate(executable.resources):
        live[id(resource)] = (index, False)
        live.setdefault(id(resource.value), (index, True))

    def reference(operation: Operation) -> tuple[str, object]:
        source = correspondence[operation]
        if isinstance(operation, core.MaterialRefOp):
            assert operation.symbol is not None
            return (
                "material",
                (
                    operation.symbol.data,
                    executable.bound_materials[
                        cast(core.MaterialRefOp, source)
                    ],
                ),
            )
        return ("object", executable.bound_resources[source])

    def describe(resource: object) -> object:
        # Artifacts record linked resources by requirement position.
        entry = live.get(id(resource))
        if entry is None:
            return resource
        return Resources((entry[0],), True, entry[1])

    arguments = []
    for argument, layout in zip(block.args, input_layouts, strict=True):
        if isinstance(layout, Layout):
            arguments.append(
                (
                    "buffer",
                    ex.buffer_type(
                        "input",
                        layout.sizes,
                        layout.strides,
                        str(layout.dtype),
                        str(layout.device),
                    ),
                )
            )
        elif layout in (HOST_INTEGER, HOST_FLOAT):
            arguments.append((layout, None))
        else:
            arguments.append(("value", None))
    return translate_block(
        block,
        arguments=arguments,
        objects=builder.objects,
        constants=builder.constants,
        reference=reference,
        describe=describe,
        signatures=builder.launch_signatures,
        # Keys and plaintexts keep their storage across calls; ciphertexts and
        # other per-call values are staged by Graph segments.
        read_in_place=[
            isinstance(argument.type, _READ_IN_PLACE) for argument in block.args
        ],
    )


_READ_IN_PLACE = (
    ckks.EvaluationKeyType,
    ckks.PlaintextType,
    ckks.CompressedPlaintextType,
)


def _drives_control_flow(value: SSAValue) -> bool:
    """Whether ``value`` reaches an ``scf`` operation through integer arithmetic."""
    pending = [value]
    while pending:
        for use in pending.pop().uses:
            owner = use.operation
            if owner.name.startswith("scf."):
                return True
            if owner.name.startswith("arith."):
                pending.extend(owner.results)
    return False


__all__ = [
    "HOST_FLOAT",
    "HOST_INTEGER",
    "LoweredProgram",
    "OPAQUE",
    "lower_program",
    "runtime_input_layout",
    "static_input_layout",
]
