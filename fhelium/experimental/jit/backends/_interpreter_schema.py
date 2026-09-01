"""Validate the operations selected for interpreter execution."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from typing import NoReturn

from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import BlockArgument, Operation

from .._analysis import RUNTIME_OPERATION_NAMES, RUNTIME_OPERATION_TYPES
from fhelium.ir import (
    DEFAULT_OPERATION_SPECS,
    Program,
    operation_name,
    value_role,
)
from fhelium.ir.dialects import torch as torch_dialect
from .._errors import JitInterpreterError


def display_name(operation: Operation) -> str:
    """Return an SSA hint or operation name for interpreter diagnostics."""

    if operation.results and operation.results[0].name_hint:
        return operation.results[0].name_hint
    return operation_name(operation)


def _fail(message: str) -> NoReturn:
    raise JitInterpreterError(message)


def _validate_literal(value: object) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if not isinstance(value, dict):
        _fail(f"Unsupported executable literal {value!r}")
    kind = value.get("kind")
    if (
        kind == "complex"
        and isinstance(value.get("real"), (int, float))
        and isinstance(value.get("imag"), (int, float))
    ):
        return
    if kind == "ellipsis":
        return
    if kind in {"torch.dtype", "torch.device", "torch.layout"} and isinstance(
        value.get("value"), str
    ):
        return
    _fail(f"Unsupported executable literal descriptor {value!r}")


def _validate_output(value: object, result_count: int) -> None:
    if not isinstance(value, dict):
        _validate_literal(value)
        return
    kind = value.get("kind")
    if kind == "ssa":
        index = value.get("result")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= result_count
        ):
            _fail("Output SSA descriptor has an invalid result index")
        return
    if kind in {"tuple", "list"}:
        items = value.get("items")
        if not isinstance(items, list):
            _fail(f"Output {kind} descriptor lacks an items list")
        for item in items:
            _validate_output(item, result_count)
        return
    if kind == "mapping":
        entries = value.get("entries")
        if not isinstance(entries, list):
            _fail("Output mapping descriptor lacks an entries list")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                _fail("Output mapping descriptor has a malformed entry")
            _validate_literal(entry[0])
            _validate_output(entry[1], result_count)
        return
    _validate_literal(value)


def _validate_entry_argument(argument: BlockArgument) -> None:
    state = getattr(argument.type, "state", None)
    data = getattr(state, "data", None)
    if not isinstance(data, Mapping) or "input_spec" not in data:
        return
    encoded = data["input_spec"]
    if not isinstance(encoded, StringAttr):
        _fail("Program input_spec must be a valid JSON string")
    try:
        spec = json.loads(encoded.data)
    except json.JSONDecodeError as error:
        raise JitInterpreterError("Program input_spec is malformed") from error
    if not isinstance(spec, dict) or spec.get("role") != value_role(argument):
        _fail("Program input_spec role differs from its value type")
    if value_role(argument) != "encrypted":
        return
    level = spec.get("level")
    scale = spec.get("scale")
    slots = spec.get("slots")
    batch_mode = spec.get("batch_mode")
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        _fail("Encrypted input_spec level must be a nonnegative integer")
    if scale is not None and (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0
    ):
        _fail("Encrypted input_spec scale must be positive finite or null")
    if slots != "full" and (
        isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0
    ):
        _fail("Encrypted input_spec slots must be 'full' or positive integer")
    if batch_mode not in {"none", "any"}:
        _fail("Encrypted input_spec batch_mode must be 'none' or 'any'")


def _validate_builtin(
    operation: Operation,
    *,
    handled_torch_targets: Collection[str],
) -> None:
    """Apply shared semantics, then interpreter-only authorization checks."""

    name = operation_name(operation)
    specification = DEFAULT_OPERATION_SPECS.get(name)
    if specification is None:
        _fail(f"No semantic specification for operation {name!r}")
    diagnostics = specification.diagnostics(operation)
    if diagnostics:
        _fail(
            f"Executable operation {display_name(operation)!r} violates "
            f"its semantic specification: {'; '.join(diagnostics)}"
        )
    if not isinstance(operation, torch_dialect.CallOp):
        return
    target_attribute = operation.attributes["fhelium.call.target"]
    assert isinstance(target_attribute, StringAttr)
    roles = {
        value_role(value) for value in (*operation.operands, *operation.results)
    }
    if (
        roles & {"encrypted", "plaintext"}
        and target_attribute.data not in handled_torch_targets
    ):
        _fail(
            f"Torch target {target_attribute.data!r} touches FHE values and "
            "requires a torch_handlers binding"
        )


def validate_interpreter_graph(
    program: Program,
    *,
    entry: str = "main",
    handled_operations: Collection[str] = (),
    handled_torch_targets: Collection[str] = (),
) -> None:
    """Validate structural coverage expected by the IR interpreter.

    Validation requires structural module integrity, one selected single-block
    entry, valid entry input metadata, one final return, built-in operation
    arities, roles, and attributes, authorized Torch targets, authorized
    extension operations, and valid captured output metadata. Built-in runtime
    names are reserved: ``handled_operations`` can authorize extension names
    but cannot replace a built-in schema.

    This entry-scoped executable-schema check complements permissive structural import
    and module-wide ordinary rewriting passes.
    """

    try:
        program.verify_structure()
        block = program.single_block(entry)
    except Exception as error:
        raise JitInterpreterError(
            f"Entry {entry!r} is not a structurally executable single block: {error}"
        ) from error
    operations = tuple(block.ops)
    for argument in block.args:
        _validate_entry_argument(argument)
    returns = tuple(op for op in operations if isinstance(op, ReturnOp))
    if len(returns) != 1 or not operations or operations[-1] is not returns[0]:
        _fail(f"Entry {entry!r} requires one final func.return terminator")

    handled = frozenset(handled_operations) - RUNTIME_OPERATION_NAMES
    handled_torch = frozenset(handled_torch_targets)
    for operation in operations[:-1]:
        name = operation_name(operation)
        if isinstance(operation, RUNTIME_OPERATION_TYPES):
            _validate_builtin(
                operation,
                handled_torch_targets=handled_torch,
            )
        elif name not in handled:
            _fail(f"Operation {name!r} has no handler")

    output_structure = program.module.attributes.get("fhelium.output_structure")
    if output_structure is not None:
        if not isinstance(output_structure, StringAttr):
            _fail("Program output_structure must be a valid JSON string")
        try:
            output_descriptor = json.loads(output_structure.data)
        except json.JSONDecodeError as error:
            raise JitInterpreterError(
                "Program output_structure is malformed"
            ) from error
        _validate_output(output_descriptor, len(returns[0].arguments))


__all__ = ["validate_interpreter_graph"]
