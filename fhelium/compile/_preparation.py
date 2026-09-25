"""Organize input specialization, Compile passes, and Backend linking."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import torch

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, StringAttr, f64
from xdsl.ir import Attribute
from xdsl.rewriter import Rewriter

from fhelium.ir import Program
from fhelium.ir.dialects import ckks
from fhelium.ir.dialects._common import OpenStateType
from fhelium.values import Ciphertext, CompressedPlaintext, Plaintext

from ._compilation import Compilation
from ._lower_and_fuse import default_lower_and_fuse_pipeline
from ._pipeline import Pipeline
from ._specialization import CallSignature, prepare_match
from ._workspace import CompileWorkspace
from .frontend._eager_values import state_attributes

if TYPE_CHECKING:
    from fhelium.backend import OperationBackend, ProgramExecutable


@dataclass(frozen=True)
class _CompiledVariant:
    signature: CallSignature
    source: Compilation
    compilation: Compilation
    input_names: tuple[str, ...]
    output_structure: object | None
    argument_types: tuple[type, ...]


@dataclass(frozen=True)
class Specialization:
    """Expose one prepared input variant, its Programs, and linked execution.

    ``source`` precedes input-state specialization and optimization.
    ``compilation`` includes the selected Compile passes and their reports.
    ``backend`` is the selected implementation registry and resource workspace;
    it can also link the optimized Compilation directly.
    ``executable.manifest`` identifies the bound implementations/resources.
    Programs and workspace materials are inspectable; mutating them after
    preparation is not a supported way to update an existing callable.
    """

    _variant: _CompiledVariant
    backend: OperationBackend
    executable: ProgramExecutable

    @property
    def signature(self) -> CallSignature:
        return self._variant.signature

    @property
    def source(self) -> Compilation:
        return self._variant.source

    @property
    def compilation(self) -> Compilation:
        return self._variant.compilation

    @cached_property
    def _result_count(self) -> int:
        return len(self.executable.program.function().function_type.outputs)

    @cached_property
    def _matches(self):
        return prepare_match(self.signature, self._variant.argument_types)

    @cached_property
    def _execute(self):
        """Adapt matched arguments and restore a fixed output structure."""
        from fhelium.backend.execution import (
            _boundary_kind,
            _prepare_compressed_output,
        )
        from fhelium.values import KeySwitchKey
        from operator import attrgetter

        positions = {
            argument.name: i
            for i, argument in enumerate(self.signature.arguments)
        }
        namespace: dict[str, Any] = {"invoke": self.executable._invoke}
        operands = []
        fields = input_fields(self.compilation.program)
        for ordinal, (name, argument, adapter) in enumerate(
            zip(
                self._variant.input_names,
                self.executable._entry_block.args,
                self.executable._input_adapters,
                strict=True,
            )
        ):
            source_name, field = fields.get(name, (name, None))
            position = positions[source_name]
            typ = self._variant.argument_types[position]
            role = _boundary_kind(argument.type)
            if field is not None:
                pass
            elif isinstance(
                argument.type, ckks.CompressedPlaintextType
            ) and issubclass(typ, CompressedPlaintext):
                adapter = attrgetter("data")
            elif role == "ciphertext" and issubclass(typ, Ciphertext):
                adapter = attrgetter("data")
            elif role == "plaintext" and issubclass(typ, Plaintext):
                representation = self.signature.arguments[position].get(
                    "representation"
                )
                adapter = attrgetter(
                    "message" if representation == "slots" else "data"
                )
            elif isinstance(
                argument.type, ckks.EvaluationKeyType
            ) and issubclass(typ, KeySwitchKey):
                adapter = attrgetter("data")
            namespace[f"adapt{ordinal}"] = adapter
            namespace[f"field{ordinal}"] = field
            actual = (
                f"values[{position}]"
                if field is None
                else f"getattr(values[{position}], field{ordinal})"
            )
            operands.append(f"adapt{ordinal}({actual})")

        def literal(value):
            name = f"literal{len(namespace)}"
            namespace[name] = _literal(value)
            return name

        def output(descriptor):
            kind = descriptor["kind"]
            if kind == "ssa":
                return (
                    "result"
                    if self._result_count == 1
                    else f"result[{descriptor['result']}]"
                )
            if kind == "compressed_plaintext":
                fields = dict(descriptor["fields"])
                fields["prime_ids"] = tuple(fields["prime_ids"])
                name = f"compressed_output{len(namespace)}"
                namespace[name] = _prepare_compressed_output(fields)
                return f"{name}({output(descriptor['data'])}, {output(descriptor['implicit'])})"
            if kind == "literal":
                return literal(descriptor["value"])
            if kind in {"tuple", "list"}:
                items = [output(item) for item in descriptor["items"]]
                if kind == "list":
                    return "[" + ", ".join(items) + "]"
                return (
                    "("
                    + ", ".join(items)
                    + ("," if len(items) == 1 else "")
                    + ")"
                )
            if kind == "mapping":
                return (
                    "{"
                    + ", ".join(
                        f"{literal(key)}: {output(item)}"
                        for key, item in descriptor["entries"]
                    )
                    + "}"
                )
            raise ValueError(f"Unsupported callable output descriptor {kind!r}")

        expression = (
            "result"
            if self._variant.output_structure is None
            else output(self._variant.output_structure)
        )
        source = f"def execute(values):\n    result = invoke({', '.join(operands)})\n    return {expression}\n"
        exec(compile(source, "<fhelium-matched-call>", "exec"), namespace)
        return namespace["execute"]


def fresh_workspace(workspace: Mapping[object, object]) -> CompileWorkspace:
    """Construct a workspace with a shallow copy of the caller's entries.

    Material payloads and other caller-provided mutable workspace entries remain
    shared.
    """

    result = CompileWorkspace(workspace)
    return result


def runtime_names(program: Program) -> tuple[str, ...]:
    arguments = tuple(program.single_block().args)
    represented = program.module.attributes.get("fhelium.input_names")
    if (
        isinstance(represented, ArrayAttr)
        and len(represented) == len(arguments)
        and all(isinstance(item, StringAttr) for item in represented)
    ):
        return tuple(
            item.data for item in represented if isinstance(item, StringAttr)
        )
    return tuple(
        argument.name_hint or f"arg{index}"
        for index, argument in enumerate(arguments)
    )


def input_fields(program: Program) -> dict[str, tuple[str, str]]:
    """Map additional Tensor inputs to their public argument storage field."""
    encoded = program.module.attributes.get("fhelium.input_fields")
    return json.loads(encoded.data) if isinstance(encoded, StringAttr) else {}


def input_values(
    program: Program, arguments: Mapping[str, object]
) -> dict[str, object]:
    values = dict(arguments)
    for name, (source, field) in input_fields(program).items():
        values[name] = getattr(arguments[source], field)
    return values


def _input_specialization(
    compilation: Compilation, arguments: Mapping[str, object]
) -> Compilation:
    program = compilation.program.clone()
    block = program.single_block()
    names = runtime_names(program)
    arguments = input_values(program, arguments)
    for name, argument in zip(names, tuple(block.args), strict=True):
        represented = argument.type
        value = arguments[name]
        if isinstance(represented, ckks.CiphertextType) and not isinstance(
            value, Ciphertext
        ):
            raise TypeError(f"Program input {name!r} requires a Ciphertext")
        if isinstance(represented, ckks.PlaintextType) and not isinstance(
            value, Plaintext
        ):
            raise TypeError(f"Program input {name!r} requires a Plaintext")
        if not isinstance(represented, OpenStateType):
            continue
        actual = state_attributes(value)
        updates: dict[str, Attribute] = {}
        for field, value in actual.items():
            prior = represented.state.data.get(field)
            if (
                field == "scale"
                and isinstance(prior, StringAttr)
                and prior.data != "unknown"
            ):
                prior = FloatAttr(float(prior.data), f64)
                updates[field] = prior
            if prior is None or prior == StringAttr("unknown"):
                updates[field] = value
            elif prior != value:
                raise ValueError(
                    f"Program input {name!r} requires {field}={prior}, "
                    f"but this invocation has {value}"
                )
        if updates:
            Rewriter.replace_value_with_new_type(
                argument, represented.with_state(updates)
            )
    program.function().update_function_type()
    return Compilation(
        program,
        fresh_workspace(compilation.workspace),
        compilation.reports,
        dict(compilation.material_bindings),
    )


def prepare_variant(
    source: Compilation,
    arguments: Mapping[str, object],
    signature: CallSignature,
    *,
    pipeline: Pipeline | Callable[[CallSignature], Pipeline] | None,
    backend: OperationBackend,
) -> tuple[_CompiledVariant, OperationBackend]:
    """Prepare a Program and select the Backend used to bind its execution."""

    request = _input_specialization(source, arguments)
    if pipeline is None:
        optimized = default_lower_and_fuse_pipeline(backend).run(request)
    else:
        if not isinstance(pipeline, Pipeline):
            pipeline = pipeline(signature)
        optimized = pipeline.run(request)
    input_names = runtime_names(optimized.program)
    if any(
        name not in input_values(optimized.program, arguments)
        for name in input_names
    ):
        raise ValueError(
            "The preparation pipeline changed the callable's input interface; "
            "transform the Compilation first and create a callable for its new interface"
        )
    output = optimized.program.module.attributes.get("fhelium.output_structure")
    return (
        _CompiledVariant(
            signature,
            source,
            optimized,
            input_names,
            json.loads(output.data) if isinstance(output, StringAttr) else None,
            tuple(
                type(arguments[argument.name])
                for argument in signature.arguments
            ),
        ),
        backend,
    )


def bind_variant(
    variant: _CompiledVariant, backend: OperationBackend
) -> Specialization:
    return Specialization(variant, backend, backend.link(variant.compilation))


def _literal(value: Any) -> Any:
    if isinstance(value, dict) and value.get("kind") == "complex":
        return complex(value["real"], value["imag"])
    return value


def restore_output(descriptor: Any, results: tuple[object, ...]) -> object:
    """Restore the callable's tuple/list/mapping output from Program results."""

    kind = descriptor["kind"]
    if kind == "compressed_plaintext":
        fields = dict(descriptor["fields"])
        fields["prime_ids"] = tuple(fields["prime_ids"])
        return CompressedPlaintext(
            data=cast(
                torch.Tensor, restore_output(descriptor["data"], results)
            ),
            implicit_data=cast(
                torch.Tensor, restore_output(descriptor["implicit"], results)
            ),
            **fields,
        )
    if kind == "ssa":
        return results[descriptor["result"]]
    if kind == "literal":
        return _literal(descriptor["value"])
    if kind == "tuple":
        return tuple(
            restore_output(item, results) for item in descriptor["items"]
        )
    if kind == "list":
        return [restore_output(item, results) for item in descriptor["items"]]
    if kind == "mapping":
        return {
            _literal(key): restore_output(item, results)
            for key, item in descriptor["entries"]
        }
    raise ValueError(f"Unsupported callable output descriptor {kind!r}")


__all__ = ["Specialization"]
