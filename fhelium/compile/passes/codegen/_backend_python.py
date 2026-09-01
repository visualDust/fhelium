"""Emit direct Backend implementation calls from a flat Program."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
)

import inspect
import importlib
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.func import ReturnOp
from xdsl.ir import SSAValue

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
    assignment,
    constant_literal,
    entry_point_name,
    program_block,
    return_statement,
    string,
    unsupported,
)


@dataclass(frozen=True)
class EmitBackendPythonPass:
    """Publish direct Backend Python for the Program at this pass position."""

    registry: OperationImplementationRegistry | None = None
    entry_point: str = "generated_backend"
    name: str = "emit-backend-python"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
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
    """Emit direct implementation calls or fail on unsupported Program content."""

    entry_point = entry_point_name(entry_point)
    block = program_block(program, target="Backend")
    names = SourceNames()
    input_names = tuple(
        names.input(argument, index)
        for index, argument in enumerate(block.args)
    )
    resource_values: set[SSAValue] = set()
    material_symbols: list[str] = []
    resource_symbols: list[str] = []
    resource_requirements: list[tuple[str, str]] = []
    imports: list[str] = [
        "from collections.abc import Mapping",
        "from fhelium.backend.implementation import "
        "OperationInvocation as __fhelium_codegen_OperationInvocation",
        "from fhelium.backend.resources import "
        "BoundResource as __fhelium_codegen_BoundResource",
    ]
    declarations: list[str] = []
    body: list[str] = []
    call_count = 0
    operation_count = 0
    returned = False

    for index, operation in enumerate(block.ops):
        if operation.regions:
            raise unsupported("Backend", index, operation)
        operation_count += 1

        if isinstance(operation, UnrealizedConversionCastOp):
            if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                raise PythonCodegenError(
                    "Backend Python emission supports only one-to-one boundary casts"
                )
            result = names.results(operation)
            source = operation.inputs[0]
            body.append(assignment(result, names.operand(source)))
            if source in resource_values:
                resource_values.add(operation.outputs[0])
            continue

        if isinstance(operation, core.MaterialRefOp):
            symbol = string(operation.symbol, label="material symbol")
            append_unique(material_symbols, (symbol,))
            body.append(
                assignment(names.results(operation), f"materials[{symbol!r}]")
            )
            continue

        if isinstance(operation, core.ResourceRefOp):
            symbol = string(operation.symbol, label="resource symbol")
            kind = string(operation.kind, label="resource kind")
            append_unique(resource_symbols, (symbol,))
            _record_requirement(resource_requirements, symbol, kind)
            results = names.results(operation)
            body.append(
                assignment(
                    results,
                    "__fhelium_codegen_BoundResource(\n"
                    f"        {symbol!r},\n"
                    f"        {kind!r},\n"
                    f"        resources[{symbol!r}],\n"
                    "    )",
                )
            )
            resource_values.add(operation.value)
            continue

        if isinstance(operation, core.ConstantOp):
            body.append(
                assignment(
                    names.results(operation), repr(constant_literal(operation))
                )
            )
            continue

        if isinstance(operation, ReturnOp):
            body.append(
                return_statement(
                    tuple(names.operand(value) for value in operation.arguments)
                )
            )
            returned = True
            continue

        invocation = operation_invocation(operation)
        try:
            implementation = registry.resolve(
                operation,
                requested=requested_implementation(operation),
                in_place=False,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PythonCodegenError(
                f"Backend Python emission cannot convert operation {index} "
                f"{operation.name!r}: {error}"
            ) from error
        requirements = implementation.resource_requirements(invocation)
        append_unique(
            resource_symbols,
            (requirement.symbol for requirement in requirements),
        )
        for requirement in requirements:
            _record_requirement(
                resource_requirements,
                requirement.symbol,
                requirement.kind,
            )

        _require_importable_class(type(operation), label="operation")
        _require_importable_class(type(implementation), label="implementation")
        operation_alias = f"__fhelium_codegen_Operation{call_count}"
        implementation_alias = f"__fhelium_codegen_Implementation{call_count}"
        imports.extend(
            (
                f"from {type(operation).__module__} import "
                f"{type(operation).__name__} as {operation_alias}",
                f"from {type(implementation).__module__} import "
                f"{type(implementation).__name__} as {implementation_alias}",
            )
        )
        declarations.extend(
            (
                f"__fhelium_codegen_implementation_{call_count} = "
                f"{implementation_alias}({_constructor_arguments(implementation)})",
                f"__fhelium_codegen_invocation_{call_count} = "
                "__fhelium_codegen_OperationInvocation(",
                f"    operation_type={operation_alias},",
                f"    operand_count={invocation.operand_count},",
                f"    result_count={invocation.result_count},",
                f"    attributes={_python_literal(dict(invocation.attributes))},",
                f"    operand_prime_ids={_python_literal(invocation.operand_prime_ids)},",
                f"    operand_bases={_python_literal(invocation.operand_bases)},",
                f"    operand_components={_python_literal(invocation.operand_components)},",
                f"    result_prime_ids={_python_literal(invocation.result_prime_ids)},",
                ")",
            )
        )

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
        requirement_resources: list[str] = []
        for requirement_index, requirement in enumerate(requirements):
            resource_name = (
                f"__fhelium_codegen_resource_{call_count}_{requirement_index}"
            )
            body.extend(
                (
                    f"    {resource_name} = __fhelium_codegen_BoundResource(",
                    f"        {requirement.symbol!r},",
                    f"        {requirement.kind!r},",
                    f"        resources[{requirement.symbol!r}],",
                    "    )",
                )
            )
            requirement_resources.append(resource_name)
        resource_arguments = (*operand_resources, *requirement_resources)
        expression = (
            f"__fhelium_codegen_implementation_{call_count}.execute(\n"
            f"        __fhelium_codegen_invocation_{call_count},\n"
            f"        {_tuple_source(payloads)},\n"
            f"        {_resource_tuple_source(resource_arguments)},\n"
            "        in_place=False,\n"
            "    )"
        )
        result_names = names.results(operation)
        if not result_names:
            body.append(f"    {expression}")
        else:
            lhs = (
                f"{result_names[0]},"
                if len(result_names) == 1
                else ", ".join(result_names)
            )
            body.append(f"    {lhs} = {expression}")
        call_count += 1

    if not returned:
        raise PythonCodegenError("Backend Python emission requires func.return")

    source = [*dict.fromkeys(imports), "", *declarations, ""]
    source.extend(
        (
            f"def {entry_point}(",
            *(f"    {name}: object," for name in input_names),
            "    *,",
            "    materials: Mapping[str, object],",
            "    resources: Mapping[str, object],",
            ") -> object:",
            *body,
        )
    )
    return BackendPythonSource(
        "\n".join(source) + "\n",
        entry_point,
        input_names,
        tuple(material_symbols),
        tuple(resource_symbols),
        operation_count,
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
