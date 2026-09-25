"""Bind resolved Program calls into a directly executable Python function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import Operation

from fhelium.backend.implementation import RegionOperationImplementation
from fhelium.ir.dialects import core

from ..codegen._common import SourceNames
from ..codegen._host import emit_host_body, tuple_source

if TYPE_CHECKING:
    from fhelium.backend.execution import ProgramExecutable


def prepare_host_call(executable: ProgramExecutable):
    """Resolve references, resource operands and callable targets before execution."""
    from fhelium.backend.execution import _index_value, _tensor_region_results

    block = executable.program.single_block("main")
    names = SourceNames()
    arguments = tuple(
        names.input(value, i) for i, value in enumerate(block.args)
    )
    namespace = {
        "__fhelium_codegen_index": _index_value,
        "__fhelium_codegen_tensor_results": _tensor_region_results,
    }
    resource_values = set()
    for operation in block.walk():
        if isinstance(operation, core.ResourceRefOp):
            resource_values.add(operation.value)
        elif (
            isinstance(operation, UnrealizedConversionCastOp)
            and operation.inputs[0] in resource_values
        ):
            resource_values.add(operation.outputs[0])
    ordinal = 0

    def bind(value: object) -> str:
        nonlocal ordinal
        symbol = f"__fhelium_codegen_bound_{ordinal}"
        ordinal += 1
        namespace[symbol] = value
        return symbol

    def reference(operation: Operation) -> str:
        if isinstance(operation, core.MaterialRefOp):
            return bind(executable.bound_materials[operation])
        return bind(executable.bound_resources[operation])

    def call(operation: Operation, regions: tuple[str, ...]) -> str:
        dispatch = executable.dispatch_table.operations[operation]
        implementation = dispatch.implementation
        payloads = tuple(
            names.operand(value)
            for value in operation.operands
            if value not in resource_values
        )
        operand_resources = tuple(
            names.operand(value)
            for value in operation.operands
            if value in resource_values
        )
        required = tuple(
            executable.resources[index]
            for index in executable.resource_indices[operation]
        )
        resources = (
            f"{tuple_source(operand_resources)} + {bind(required)}"
            if operand_resources
            else bind(required)
        )
        if regions:
            if not isinstance(implementation, RegionOperationImplementation):
                raise TypeError(
                    f"Implementation {implementation.name!r} cannot execute regions"
                )
            return f"{bind(implementation.execute_regions)}({bind(dispatch.invocation)}, {tuple_source(payloads)}, {resources}, {tuple_source(regions)}, in_place={dispatch.in_place!r})"
        return f"{bind(implementation.execute)}({bind(dispatch.invocation)}, {tuple_source(payloads)}, {resources}, in_place={dispatch.in_place!r})"

    body = emit_host_body(
        block,
        names,
        call,
        reference,
        prepared_regions=lambda operation: (
            operation in executable.dispatch_table.operations
            and executable.dispatch_table.operations[operation].prepared_regions
        ),
    )
    source = (
        f"def __fhelium_codegen_execute({', '.join(arguments)}):\n"
        + "\n".join(body)
        + "\n"
    )
    exec(compile(source, "<fhelium-prepared-program>", "exec"), namespace)
    return namespace["__fhelium_codegen_execute"], source
