"""Shared source-emission helpers for optional Python tool passes."""

from __future__ import annotations

import json
import keyword
import re
from collections.abc import Iterable

from xdsl.dialects.builtin import FloatAttr, IntegerAttr, StringAttr
from xdsl.ir import Block, Operation, SSAValue

from fhelium.compile.codegen import PythonCodegenError
from fhelium.ir import Program
from fhelium.ir.dialects import core

_IDENTIFIER = re.compile(r"\W")


def program_block(program: Program, *, target: str) -> Block:
    """Return the flat main block accepted by one Python emitter."""

    functions = program.functions
    if (
        len(functions) != 1
        or len(tuple(program.module.ops)) != 1
        or functions[0].sym_name.data != "main"
    ):
        raise PythonCodegenError(
            f"{target} Python emission requires one top-level main function"
        )
    try:
        block = program.single_block("main")
    except (KeyError, ValueError) as error:
        raise PythonCodegenError(
            f"{target} Python emission requires one main function with one block: "
            f"{error}"
        ) from error
    return block


def entry_point_name(value: str) -> str:
    """Require one callable Python identifier for emitted source."""

    if (
        not isinstance(value, str)
        or not value.isidentifier()
        or keyword.iskeyword(value)
    ):
        raise PythonCodegenError(
            "Python codegen entry point must be a non-keyword identifier"
        )
    return value


class SourceNames:
    """Assign readable unique Python identifiers to SSA values."""

    def __init__(self) -> None:
        self.values: dict[SSAValue, str] = {}
        self._used: set[str] = {
            "engine",
            "executable",
            "runtime",
            "materials",
            "resources",
        }
        self._next_value = 0

    def _unique(self, suggested: str, *, fallback: str) -> str:
        name = _IDENTIFIER.sub("_", suggested).strip("_") or fallback
        if name[0].isdigit():
            name = f"value_{name}"
        if keyword.iskeyword(name):
            name = f"{name}_value"
        if name.startswith("fhelium_codegen_"):
            name = f"value_{name}"
        candidate = name
        suffix = 1
        while candidate in self._used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        self._used.add(candidate)
        return candidate

    def input(self, value: SSAValue, index: int) -> str:
        suggested = value.name_hint or f"input_{index}"
        name = self._unique(suggested, fallback=f"input_{index}")
        self.values[value] = name
        return name

    def results(self, operation: Operation) -> tuple[str, ...]:
        names: list[str] = []
        for result in operation.results:
            suggested = result.name_hint or f"value_{self._next_value}"
            name = self._unique(
                suggested,
                fallback=f"value_{self._next_value}",
            )
            self._next_value += 1
            self.values[result] = name
            names.append(name)
        return tuple(names)

    def operand(self, value: SSAValue) -> str:
        try:
            return self.values[value]
        except KeyError:
            raise PythonCodegenError(
                "Python emission encountered an operand before its SSA value "
                "was defined"
            ) from None


def assignment(results: tuple[str, ...], expression: str) -> str:
    """Return one indented assignment or expression statement."""

    if not results:
        return f"    {expression}"
    if len(results) == 1:
        return f"    {results[0]} = {expression}"
    return f"    {', '.join(results)} = {expression}"


def return_statement(values: tuple[str, ...]) -> str:
    """Return Python source matching ProgramExecutable's result convention."""

    if not values:
        return "    return None"
    if len(values) == 1:
        return f"    return {values[0]}"
    return f"    return {', '.join(values)}"


def integer(attribute: object, *, label: str) -> int:
    if not isinstance(attribute, IntegerAttr):
        raise PythonCodegenError(f"Python emission requires integer {label}")
    return int(attribute.value.data)


def floating(attribute: object, *, label: str) -> float:
    if not isinstance(attribute, FloatAttr):
        raise PythonCodegenError(f"Python emission requires floating {label}")
    return float(attribute.value.data)


def string(attribute: object, *, label: str) -> str:
    if not isinstance(attribute, StringAttr) or not attribute.data:
        raise PythonCodegenError(
            f"Python emission requires non-empty string {label}"
        )
    return attribute.data


def constant_literal(operation: core.ConstantOp) -> object:
    """Decode one portable constant descriptor for safe Python repr emission."""

    literal = operation.literal
    if not isinstance(literal, StringAttr):
        raise PythonCodegenError("fhelium.constant requires a JSON literal")
    try:
        return json.loads(literal.data)
    except json.JSONDecodeError as error:
        raise PythonCodegenError(
            f"fhelium.constant has invalid JSON: {error}"
        ) from error


def append_unique(items: list[str], values: Iterable[str]) -> None:
    """Append strings once while preserving first Program order."""

    seen = set(items)
    for value in values:
        if value in seen:
            continue
        items.append(value)
        seen.add(value)


def unsupported(
    target: str, index: int, operation: Operation
) -> PythonCodegenError:
    """Describe the first operation outside one emitter's supported surface."""

    return PythonCodegenError(
        f"{target} Python emission cannot convert operation {index} "
        f"{operation.name!r}"
    )
