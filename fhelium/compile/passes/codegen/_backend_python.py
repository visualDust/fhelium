"""Emit direct Backend implementation calls from a flat Program."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    PassResult,
)

import inspect
import importlib
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.func import ReturnOp

from fhelium.backend.execution import OperationBackend
from fhelium.backend.implementation import (
    OperationImplementationRegistry,
    operation_invocation,
    requested_implementation,
)
from fhelium.compile.codegen import BackendPythonSource, PythonCodegenError
from fhelium.ir import Program
from fhelium.ir.dialects import core

from ._common import (
    SourceNames,
    append_unique,
    entry_point_name,
    program_block,
    string,
)


@dataclass(frozen=True)
class EmitBackendPythonPass:
    """Publish direct Backend Python for the Program at this pass position."""

    registry: OperationImplementationRegistry | None = None
    entry_point: str = "generated_backend"
    name: str = "emit-backend-python"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        registry = (
            OperationBackend().registry
            if self.registry is None
            else self.registry
        )
        artifact = emit_backend_python(
            program,
            registry=registry,
            entry_point=self.entry_point,
        )
        shared_data[BackendPythonSource] = artifact
        return PassResult.unchanged(
            program,
            matched=artifact.operation_count,
            diagnostics=(
                f"emitted Backend Python entry point {artifact.entry_point!r}",
            ),
        )


def emit_backend_python(
    program: Program,
    *,
    registry: OperationImplementationRegistry,
    entry_point: str = "generated_backend",
) -> BackendPythonSource:
    """Export the host control flow used by linked execution."""
    from xdsl.ir import Block
    from fhelium.backend.implementation import PreparingOperationImplementation
    from ._host import emit_host_body, tuple_source

    entry_point = entry_point_name(entry_point)
    block = program_block(program, target="Backend")
    names = SourceNames()
    input_names = tuple(
        names.input(value, i) for i, value in enumerate(block.args)
    )
    material_symbols: list[str] = []
    resource_symbols: list[str] = []
    resource_requirements: list[tuple[str, str]] = []
    resource_values = set()
    for operation in block.walk():
        if isinstance(operation, core.ResourceRefOp):
            resource_values.add(operation.value)
        elif (
            isinstance(operation, UnrealizedConversionCastOp)
            and operation.inputs[0] in resource_values
        ):
            resource_values.add(operation.outputs[0])
    imports = [
        "from collections.abc import Mapping",
        "from fhelium.backend.implementation import OperationInvocation as __fhelium_codegen_OperationInvocation",
        "from fhelium.backend.resources import BoundResource as __fhelium_codegen_BoundResource",
        "from fhelium.backend.execution import _index_value as __fhelium_codegen_index, _tensor_region_results as __fhelium_codegen_tensor_results",
        "from fhelium.ir import Program as __fhelium_codegen_Program",
    ]
    declarations: list[str] = []
    call_count = 0
    resolved = {}

    def implementation_for(operation):
        if operation not in resolved:
            resolved[operation] = registry.resolve(
                operation,
                requested=requested_implementation(operation),
                in_place=False,
            )
        return resolved[operation]

    def reference(operation):
        if isinstance(operation, core.MaterialRefOp):
            symbol = string(operation.symbol, label="material symbol")
            append_unique(material_symbols, (symbol,))
            return f"materials[{symbol!r}]"
        symbol = string(operation.symbol, label="resource symbol")
        kind = string(operation.kind, label="resource kind")
        append_unique(resource_symbols, (symbol,))
        _record_requirement(resource_requirements, symbol, kind)
        return f"__fhelium_codegen_BoundResource({symbol!r}, {kind!r}, resources[{symbol!r}])"

    def call(operation, regions):
        nonlocal call_count
        implementation = implementation_for(operation)
        invocation = operation_invocation(operation)
        requirements = implementation.resource_requirements(invocation)
        for requirement in requirements:
            append_unique(resource_symbols, (requirement.symbol,))
            _record_requirement(
                resource_requirements, requirement.symbol, requirement.kind
            )
        _require_importable_class(type(operation), label="operation")
        _require_importable_class(type(implementation), label="implementation")
        operation_alias = f"__fhelium_codegen_Operation{call_count}"
        implementation_alias = f"__fhelium_codegen_Implementation{call_count}"
        variable = f"__fhelium_codegen_implementation_{call_count}"
        invocation_name = f"__fhelium_codegen_invocation_{call_count}"
        imports.extend(
            (
                f"from {type(operation).__module__} import {type(operation).__name__} as {operation_alias}",
                f"from {type(implementation).__module__} import {type(implementation).__name__} as {implementation_alias}",
            )
        )
        declarations.append(
            f"{variable} = {implementation_alias}({_constructor_arguments(implementation)})"
        )
        if isinstance(implementation, PreparingOperationImplementation):
            source_block = Block(
                arg_types=[value.type for value in operation.operands]
            )
            copy = operation.clone(
                value_mapper=dict(zip(operation.operands, source_block.args))
            )
            source_block.add_ops((copy, ReturnOp(*copy.results)))
            description = str(
                Program.from_function(
                    source_block, [value.type for value in operation.results]
                )
            )
            declarations.append(
                f"{variable} = {variable}.prepare_operation(__fhelium_codegen_Program.parse({description!r}).single_block().first_op)"
            )
        declarations.append(
            f"{invocation_name} = __fhelium_codegen_OperationInvocation("
            f"operation_type={operation_alias}, operand_count={invocation.operand_count}, result_count={invocation.result_count}, "
            f"attributes={_python_literal(dict(invocation.attributes))}, operand_prime_ids={_python_literal(invocation.operand_prime_ids)}, "
            f"operand_bases={_python_literal(invocation.operand_bases)}, operand_components={_python_literal(invocation.operand_components)}, "
            f"result_prime_ids={_python_literal(invocation.result_prime_ids)})"
        )
        payloads = tuple(
            names.operand(value)
            for value in operation.operands
            if value not in resource_values
        )
        bound = [
            names.operand(value)
            for value in operation.operands
            if value in resource_values
        ]
        bound.extend(
            f"__fhelium_codegen_BoundResource({r.symbol!r}, {r.kind!r}, resources[{r.symbol!r}])"
            for r in requirements
        )
        method = "execute_regions" if regions else "execute"
        arguments = f"{invocation_name}, {tuple_source(payloads)}, {tuple_source(bound)}"
        if regions:
            arguments += f", {tuple_source(regions)}"
        call_count += 1
        return f"{variable}.{method}({arguments}, in_place=False)"

    def prepared_regions(operation):
        if not operation.regions or operation.name.startswith("scf."):
            return False
        return isinstance(
            implementation_for(operation), PreparingOperationImplementation
        )

    body = emit_host_body(
        block,
        names,
        call,
        reference,
        prepared_regions=prepared_regions,
        tuple_return=False,
    )
    parameters = ", ".join(
        (
            *input_names,
            "*",
            "materials: Mapping[str, object]",
            "resources: Mapping[str, object]",
        )
    )
    # No positional inputs still permits keyword-only binding maps.
    source = [
        *dict.fromkeys(imports),
        "",
        *declarations,
        "",
        f"def {entry_point}({parameters}) -> object:",
        *body,
    ]
    return BackendPythonSource(
        "\n".join(source) + "\n",
        entry_point,
        input_names,
        tuple(material_symbols),
        tuple(resource_symbols),
        sum(1 for _ in block.walk()),
        tuple(resource_requirements),
    )


def _constructor_arguments(implementation: object) -> str:
    """Serialize literal constructor differences for one resolved implementation."""

    implementation_type = type(implementation)
    _require_importable_class(implementation_type, label="implementation")
    try:
        parameters = inspect.signature(implementation_type).parameters
    except (TypeError, ValueError) as error:
        raise PythonCodegenError(
            f"Backend implementation {implementation_type.__qualname__!r} "
            "has no serializable constructor signature"
        ) from error
    arguments: list[str] = []
    for name, parameter in parameters.items():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise PythonCodegenError(
                f"Backend implementation {type(implementation).__name__} has "
                "an unsupported constructor"
            )
        try:
            value = getattr(implementation, name)
        except AttributeError as error:
            raise PythonCodegenError(
                f"Backend implementation {implementation_type.__qualname__!r} "
                f"does not expose constructor field {name!r}"
            ) from error
        if (
            parameter.default is not inspect.Parameter.empty
            and value == parameter.default
        ):
            continue
        if not implementation_type.__module__.startswith("fhelium.backend."):
            raise PythonCodegenError(
                f"Stateful custom Backend implementation "
                f"{implementation_type.__qualname__!r} cannot be embedded in "
                "generated source"
            )
        arguments.append(f"{name}={_python_literal(value)}")
    return ", ".join(arguments)


def _require_importable_class(value: type[object], *, label: str) -> None:
    """Require the exact class object emitted by a from-import statement."""

    try:
        module = importlib.import_module(value.__module__)
    except (ImportError, ValueError) as error:
        raise PythonCodegenError(
            f"Backend {label} class {value.__qualname__!r} is not importable"
        ) from error
    if (
        "." in value.__qualname__
        or getattr(module, value.__name__, None) is not value
    ):
        raise PythonCodegenError(
            f"Backend {label} class {value.__qualname__!r} is not an "
            "importable top-level class"
        )


def _python_literal(value: Any) -> str:
    """Serialize the literal subset used by Backend invocations and constructors."""

    if value is None or isinstance(value, bool | int | float | str):
        return repr(value)
    if isinstance(value, tuple):
        return _tuple_source(tuple(_python_literal(item) for item in value))
    if isinstance(value, list):
        return f"[{', '.join(_python_literal(item) for item in value)}]"
    if isinstance(value, dict):
        entries = ", ".join(
            f"{_python_literal(key)}: {_python_literal(item)}"
            for key, item in value.items()
        )
        return f"{{{entries}}}"
    raise PythonCodegenError(
        f"Backend Python emission cannot serialize {type(value).__name__} "
        "constructor or invocation data"
    )


def _tuple_source(values: tuple[str, ...]) -> str:
    if not values:
        return "()"
    if len(values) == 1:
        return f"({values[0]},)"
    return f"({', '.join(values)})"


def _resource_tuple_source(values: tuple[str, ...]) -> str:
    if len(values) <= 1:
        return _tuple_source(values)
    entries = "\n".join(f"            {value}," for value in values)
    return f"(\n{entries}\n        )"


def _record_requirement(
    requirements: list[tuple[str, str]],
    symbol: str,
    kind: str,
) -> None:
    for existing_symbol, existing_kind in requirements:
        if existing_symbol != symbol:
            continue
        if existing_kind != kind:
            raise PythonCodegenError(
                f"Backend resource {symbol!r} has conflicting kinds "
                f"{existing_kind!r} and {kind!r}"
            )
        return
    requirements.append((symbol, kind))


__all__ = ["EmitBackendPythonPass", "emit_backend_python"]
