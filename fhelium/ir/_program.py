"""Neutral mixed-dialect xDSL program representation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp
from xdsl.ir import Attribute, Block, Operation, Region
from xdsl.parser import Parser
from xdsl.printer import Printer

from ._dialect import (
    DIALECT_VERSION,
    DIALECT_VERSION_ATTRIBUTE,
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTRIBUTE,
    create_dialect_context,
)

if TYPE_CHECKING:
    from os import PathLike


class Program:
    """Own one structurally valid, mixed-level xDSL module.

    The module may contain registered FHElium operations, unknown extension
    dialects, partial state, and operations from several abstraction levels.
    Numerical analysis, transformation, backend coverage, and execution are
    responsibilities of consumers rather than this representation object.
    """

    def __init__(self, module: ModuleOp) -> None:
        if not isinstance(module, ModuleOp):
            raise TypeError("Program requires an xDSL ModuleOp")
        module.verify()
        self.module = module

    @classmethod
    def empty(
        cls,
        operations: Iterable[Operation] = (),
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> Program:
        """Construct a structurally valid module with default version marks."""

        return cls(
            ModuleOp(operations, attributes=_module_attributes(attributes))
        )

    @classmethod
    def from_function(
        cls,
        block: Block,
        result_types: Sequence[Attribute] = (),
        *,
        name: str = "main",
        visibility: str | None = None,
        module_attributes: Mapping[str, Attribute] | None = None,
    ) -> Program:
        """Wrap one caller-built block in a top-level function."""

        if not isinstance(block, Block):
            raise TypeError("Program.from_function block must be an xDSL Block")
        if not isinstance(name, str):
            raise TypeError("Program function name must be a string")
        if not name.strip():
            raise ValueError("Program function name must be non-empty")
        function = FuncOp(
            name,
            (
                tuple(argument.type for argument in block.args),
                tuple(result_types),
            ),
            Region(block),
            visibility=visibility,
        )
        return cls.empty((function,), attributes=module_attributes)

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        source_name: str = "<unknown>",
    ) -> Program:
        """Parse structural IR while preserving unknown dialect content."""

        if not isinstance(text, str):
            raise TypeError("Program.parse text must be a string")
        if not isinstance(source_name, str):
            raise TypeError("Program.parse source_name must be a string")
        module = Parser(
            create_dialect_context(), text, source_name
        ).parse_module()
        return cls(module)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Program:
        """Load one UTF-8 textual module."""

        file_path = Path(path)
        return cls.parse(
            file_path.read_text(encoding="utf-8"),
            source_name=str(file_path),
        )

    def to_text(
        self,
        *,
        generic: bool = False,
        include_locations: bool = False,
    ) -> str:
        """Return xDSL text for the current module."""

        stream = StringIO()
        Printer(
            stream=stream,
            print_generic_format=generic,
            print_debuginfo=include_locations,
        ).print_op(self.module)
        return stream.getvalue()

    def save(self, path: str | PathLike[str]) -> None:
        """Write the current module and symbolic references as UTF-8 text."""

        Path(path).write_text(
            self.to_text(include_locations=True), encoding="utf-8"
        )

    def clone(self) -> Program:
        """Return a structurally independent copy."""

        return Program(self.module.clone())

    def verify_structure(self) -> None:
        """Run xDSL structural verification on the current module."""

        self.module.verify()

    def walk(self, *, include_module: bool = False) -> Iterator[Operation]:
        """Traverse operations in structural preorder."""

        operations = self.module.walk()
        if include_module:
            return operations
        next(operations)
        return operations

    @property
    def functions(self) -> tuple[FuncOp, ...]:
        """Return registered top-level functions in module order."""

        return tuple(
            operation
            for operation in self.module.ops
            if isinstance(operation, FuncOp)
        )

    def function(self, name: str = "main") -> FuncOp:
        """Return the uniquely named top-level registered function."""

        matches = tuple(
            function
            for function in self.functions
            if function.sym_name.data == name
        )
        if len(matches) != 1:
            raise KeyError(
                f"Program function {name!r} matched {len(matches)} "
                "top-level functions"
            )
        return matches[0]

    def single_block(self, name: str = "main") -> Block:
        """Return one function's block when a consumer needs that form."""

        blocks = tuple(self.function(name).body.blocks)
        if len(blocks) != 1:
            raise ValueError(
                f"Program function {name!r} has {len(blocks)} blocks"
            )
        return blocks[0]

    def __str__(self) -> str:
        return self.to_text()


def _module_attributes(
    attributes: Mapping[str, Attribute] | None,
) -> dict[str, Attribute]:
    result: dict[str, Attribute] = {
        SCHEMA_VERSION_ATTRIBUTE: StringAttr(SCHEMA_VERSION),
        DIALECT_VERSION_ATTRIBUTE: StringAttr(DIALECT_VERSION),
    }
    if attributes is not None:
        result.update(attributes)
    return result


__all__ = ["Program"]
