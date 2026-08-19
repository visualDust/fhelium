"""Validate the exact operation surface selected for execution."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, NoReturn

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import BlockArgument, Operation

from .._analysis import RUNTIME_OPERATION_NAMES
from .._dialect import operation_name, value_role
from .._errors import JitPassError
from .._program import Program
from ._base import PassResult
from ._utils import display_name, obligations


def _fail(message: str) -> NoReturn:
    raise JitPassError(message)


def _require_arity(
    operation: Operation,
    *,
    operands: int,
    results: int = 1,
) -> None:
    label = display_name(operation)
    if len(operation.operands) != operands or len(operation.results) != results:
        _fail(
            f"Executable operation {label!r} requires {operands} operands and "
            f"{results} results, got {len(operation.operands)} and "
            f"{len(operation.results)}"
        )


def _require_roles(
    operation: Operation,
    expected_operands: tuple[str, ...],
    expected_result: str,
) -> None:
    label = display_name(operation)
    actual_operands = tuple(value_role(value) for value in operation.operands)
    actual_result = (
        value_role(operation.results[0])
        if len(operation.results) == 1
        else None
    )
    if actual_operands != expected_operands or actual_result != expected_result:
        _fail(
            f"Executable operation {label!r} expects roles "
            f"{expected_operands} -> {expected_result!r}, got "
            f"{actual_operands} -> {actual_result!r}"
        )


def _string_attribute(operation: Operation, name: str) -> str:
    value = operation.attributes.get(name)
    if not isinstance(value, StringAttr) or not value.data:
        _fail(
            f"Executable operation {display_name(operation)!r} requires "
            f"canonical string attribute {name!r}"
        )
    return value.data


def _integer_attribute(operation: Operation, name: str) -> int:
    value = operation.attributes.get(name)
    if not isinstance(value, IntegerAttr):
        _fail(
            f"Executable operation {display_name(operation)!r} requires "
            f"canonical integer attribute {name!r}"
        )
    return int(value.value.data)


def _json_attribute(operation: Operation, name: str) -> object:
    encoded = _string_attribute(operation, name)
    try:
        return json.loads(encoded)
    except json.JSONDecodeError as error:
        raise JitPassError(
            f"Executable operation {display_name(operation)!r} has malformed "
            f"JSON attribute {name!r}"
        ) from error


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


def _validate_argument(value: object, operand_count: int) -> None:
    if not isinstance(value, dict):
        _validate_literal(value)
        return
    kind = value.get("kind")
    if kind == "ssa":
        index = value.get("operand")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= operand_count
        ):
            _fail("SSA argument descriptor has an invalid operand index")
        return
    if kind == "literal":
        _validate_literal(value.get("value"))
        return
    if kind in {"tuple", "list"}:
        items = value.get("items")
        if not isinstance(items, list):
            _fail(f"{kind} argument descriptor lacks an items list")
        for item in items:
            _validate_argument(item, operand_count)
        return
    if kind == "mapping":
        entries = value.get("entries")
        if not isinstance(entries, list):
            _fail("mapping argument descriptor lacks an entries list")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                _fail("mapping argument descriptor has a malformed entry")
            _validate_argument(entry[0], operand_count)
            _validate_argument(entry[1], operand_count)
        return
    if kind == "slice":
        for field in ("start", "stop", "step"):
            _validate_argument(value.get(field), operand_count)
        return
    _fail(f"Unknown executable argument descriptor kind {kind!r}")


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
        _fail("Program input_spec must be a canonical JSON string")
    try:
        spec = json.loads(encoded.data)
    except json.JSONDecodeError as error:
        raise JitPassError("Program input_spec is malformed") from error
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
    name = operation_name(operation)
    if operation.properties or operation.regions or operation.successors:
        _fail(
            f"Interpreter operation {display_name(operation)!r} cannot own "
            "properties, regions, or successors"
        )

    if name in {"fhelium.material.ref", "fhelium.resource.ref"}:
        _require_arity(operation, operands=0)
        _string_attribute(operation, "symbol")
        return
    if name == "fhelium.constant":
        _require_arity(operation, operands=0)
        _validate_literal(_json_attribute(operation, "fhelium.literal"))
        return
    if name == "torch.call":
        _require_arity(operation, operands=len(operation.operands))
        target = _string_attribute(operation, "fhelium.call.target")
        kind = _string_attribute(operation, "fhelium.call.kind")
        if kind not in {"function", "method", "module"}:
            _fail(f"torch.call has unsupported call kind {kind!r}")
        descriptor = _json_attribute(operation, "fhelium.call.arguments")
        if not isinstance(descriptor, dict):
            _fail("torch.call argument metadata must be a JSON object")
        _validate_argument(descriptor.get("args"), len(operation.operands))
        _validate_argument(descriptor.get("kwargs"), len(operation.operands))
        roles = {
            value_role(value)
            for value in (*operation.operands, *operation.results)
        }
        if (
            roles & {"encrypted", "plaintext"}
            and target not in handled_torch_targets
        ):
            _fail(
                f"Torch target {target!r} touches FHE values and requires an "
                "torch_handlers binding"
            )
        return

    if name.startswith("fhelium.ckks.prepare."):
        parts = name.split(".")
        if len(parts) != 5:
            _fail(f"Prepared plaintext operation {name!r} has malformed name")
        action, source_role = parts[3], parts[4]
        if action not in {"add", "multiply"} or source_role not in {
            "message",
            "plaintext",
            "static",
        }:
            _fail(f"Prepared plaintext operation {name!r} is unsupported")
        _require_arity(operation, operands=2)
        _require_roles(operation, (source_role, "encrypted"), "plaintext")
        if _string_attribute(operation, "operation") != action:
            _fail(
                f"Prepared plaintext operation {name!r} has inconsistent action"
            )
        if _string_attribute(operation, "source_role") != source_role:
            _fail(
                f"Prepared plaintext operation {name!r} has inconsistent role"
            )
        scale_mode = _string_attribute(operation, "scale_mode")
        allowed_modes = (
            {"ciphertext_scale"}
            if action == "add"
            else (
                {"runtime_plaintext_scale"}
                if source_role == "plaintext"
                else {"default_scale"}
            )
        )
        if scale_mode not in allowed_modes:
            _fail(
                f"Prepared plaintext operation {name!r} has invalid scale mode"
            )
        return

    if name in {
        "fhelium.ckks.negate",
        "fhelium.ckks.to_ntt",
        "fhelium.ckks.from_ntt",
        "fhelium.ckks.relinearize",
    }:
        _require_arity(operation, operands=1)
        _require_roles(operation, ("encrypted",), "encrypted")
        return
    if name == "fhelium.ckks.rotate":
        _require_arity(operation, operands=1)
        _require_roles(operation, ("encrypted",), "encrypted")
        _integer_attribute(operation, "shift")
        return
    if name in {
        "fhelium.ckks.add",
        "fhelium.ckks.subtract",
        "fhelium.ckks.multiply",
    }:
        _require_arity(operation, operands=2)
        _require_roles(operation, ("encrypted", "encrypted"), "encrypted")
        return
    if name in {
        "fhelium.ckks.add_plaintext",
        "fhelium.ckks.multiply_plaintext",
    }:
        _require_arity(operation, operands=2)
        _require_roles(operation, ("encrypted", "plaintext"), "encrypted")
        return
    if name == "fhelium.ckks.rescale":
        condition = _string_attribute(operation, "condition")
        expected = 2 if condition == "plaintext_scale_not_one" else 1
        if condition not in {"always", "plaintext_scale_not_one"}:
            _fail(f"rescale has unsupported condition {condition!r}")
        _require_arity(operation, operands=expected)
        expected_roles = (
            ("encrypted", "plaintext") if expected == 2 else ("encrypted",)
        )
        _require_roles(operation, expected_roles, "encrypted")
        return
    _fail(f"No built-in executable schema for operation {name!r}")


def validate_executable_graph(
    program: Program,
    *,
    entry: str = "main",
    handled_operations: Collection[str] = (),
    handled_torch_targets: Collection[str] = (),
) -> None:
    """Validate the exact selected-entry schema consumed by the interpreter.

    Validation requires structural module integrity, one selected single-block
    entry, valid entry input metadata, one final return, cleared scheduling
    obligations, exact built-in operation arities/roles/attributes, authorized
    Torch targets, authorized extension operations, and valid captured output
    metadata. Built-in runtime names are reserved: ``handled_operations`` can
    authorize extension names but cannot replace a built-in schema.

    This entry-scoped execution gate complements permissive structural import
    and module-wide ordinary rewriting passes.
    """

    try:
        program.module.verify()
        block = program.entry_block(entry)
    except Exception as error:
        raise JitPassError(
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
        scheduled = obligations(operation)
        if scheduled:
            _fail(
                f"Operation {display_name(operation)!r} retains unresolved "
                f"scheduling obligations {sorted(scheduled)}"
            )
        if name in RUNTIME_OPERATION_NAMES:
            _validate_builtin(
                operation,
                handled_torch_targets=handled_torch,
            )
        elif name not in handled:
            _fail(f"Operation {name!r} has no handler")

    output_structure = program.module.attributes.get("fhelium.output_structure")
    if output_structure is not None:
        if not isinstance(output_structure, StringAttr):
            _fail("Program output_structure must be a canonical JSON string")
        try:
            output_descriptor = json.loads(output_structure.data)
        except json.JSONDecodeError as error:
            raise JitPassError(
                "Program output_structure is malformed"
            ) from error
        _validate_output(output_descriptor, len(returns[0].arguments))


@dataclass(frozen=True)
class ValidateExecutableGraphPass:
    """Apply the selected-entry operation/schema/obligation execution gate."""

    entry: str = "main"
    name: str = "validate-executable-graph"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Validate ``entry`` and return an unchanged Program report."""

        handlers = workspace.get("handlers", {})
        torch_handlers = workspace.get("torch_handlers", {})
        if not isinstance(handlers, Mapping):
            raise TypeError("workspace['handlers'] must be a mapping")
        if not isinstance(torch_handlers, Mapping):
            raise TypeError("workspace['torch_handlers'] must be a mapping")
        validate_executable_graph(
            program,
            entry=self.entry,
            handled_operations=tuple(
                str(name)
                for name, handler in handlers.items()
                if callable(handler)
            ),
            handled_torch_targets=tuple(
                str(name)
                for name, handler in torch_handlers.items()
                if callable(handler)
            ),
        )
        block = program.entry_block(self.entry)
        return PassResult.unchanged(program, matched=max(0, len(block.ops) - 1))


__all__ = ["ValidateExecutableGraphPass", "validate_executable_graph"]
