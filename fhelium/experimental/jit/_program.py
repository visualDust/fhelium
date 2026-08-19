"""Canonical source-independent xDSL ``Program`` representation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    create_ir_context,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from os import PathLike

    from ._analysis import ProgramRequirements
    from ._execution import ReadinessReport
    from .passes._base import Pass, PassPipeline, PipelineResult


class Program:
    """Wrap one mixed-dialect xDSL module as the canonical JIT graph.

    Capture, textual import, direct construction, and extension frontends all
    produce this graph abstraction. Construction verifies the supplied xDSL
    module. Passes may subsequently mutate the exposed module in place and own
    the validity of their result. Readiness re-verifies structure and checks
    parameter/key capabilities, material/resource bindings, operation support,
    and the selected entry's executable schema; execution enforces runtime
    input and CKKS numerical requirements.
    """

    def __init__(self, module: ModuleOp) -> None:
        if not isinstance(module, ModuleOp):
            raise TypeError(
                "Program requires an xdsl.dialects.builtin.ModuleOp"
            )
        module.verify()
        self.module = module

    @classmethod
    def empty(
        cls,
        operations: Iterable[Operation] = (),
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> Program:
        """Construct a structurally verified, versioned canonical module."""

        module_attributes = _module_attributes(attributes)
        return cls(ModuleOp(operations, attributes=module_attributes))

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
        """Wrap one caller-built block in a versioned top-level function.

        ``block`` supplies the arguments, operations, and terminator; xDSL
        verification establishes structural integrity. CKKS state, parameters,
        graph-external bindings, scheduling obligations, and executable-schema
        validation are evaluated by selected passes or readiness checks.
        """

        if not isinstance(block, Block):
            raise TypeError("Program.from_function block must be an xDSL Block")
        if not isinstance(name, str):
            raise TypeError("Program function name must be a string")
        if not name.strip():
            raise ValueError("Program function name must be non-empty")
        argument_types = tuple(argument.type for argument in block.args)
        function = FuncOp(
            name,
            (argument_types, tuple(result_types)),
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
        """Parse mixed-dialect xDSL and verify its structural integrity.

        The parser registers builtin, func, and FHElium structural vocabulary
        and preserves other dialects as unregistered xDSL objects. Numerical
        and execution readiness remains a later operation.
        """

        if not isinstance(text, str):
            raise TypeError("Program.parse text must be a string")
        if not isinstance(source_name, str):
            raise TypeError("Program.parse source_name must be a string")
        module = Parser(create_ir_context(), text, source_name).parse_module()
        return cls(module)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Program:
        """Load and structurally verify one UTF-8 textual module from ``path``."""

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
        """Serialize the current module state as xDSL text."""

        stream = StringIO()
        Printer(
            stream=stream,
            print_generic_format=generic,
            print_debuginfo=include_locations,
        ).print_op(self.module)
        return stream.getvalue()

    def save(self, path: str | PathLike[str]) -> None:
        """Write Program IR and symbolic references as UTF-8 xDSL text.

        Live material and resource bindings remain in the graph-external
        ``Workspace`` and are therefore absent from the serialized module.
        """

        Path(path).write_text(
            self.to_text(include_locations=True), encoding="utf-8"
        )

    def clone(self) -> Program:
        """Return a structurally independent clone of the canonical module."""

        return Program(self.module.clone())

    def walk(self, *, include_module: bool = False) -> Iterator[Operation]:
        """Traverse operations in structural preorder."""

        operations = self.module.walk()
        if include_module:
            return operations
        next(operations)
        return operations

    @property
    def functions(self) -> tuple[FuncOp, ...]:
        """Return top-level registered func operations in module order."""

        return tuple(
            operation
            for operation in self.module.ops
            if isinstance(operation, FuncOp)
        )

    def entry_function(self, name: str = "main") -> FuncOp:
        """Return the uniquely named top-level registered function.

        A Program is a structurally valid interchange module with any legal
        top-level contents. Entry-oriented consumers select a callable function
        and receive a precise lookup failure for zero or multiple
        matches.
        """

        matches = tuple(
            function
            for function in self.functions
            if function.sym_name.data == name
        )
        if len(matches) != 1:
            raise KeyError(
                f"Program entry function {name!r} matched {len(matches)} "
                "top-level functions"
            )
        return matches[0]

    def entry_block(self, name: str = "main") -> Block:
        """Return the unique block of one selected entry function."""

        function = self.entry_function(name)
        blocks = tuple(function.body.blocks)
        if len(blocks) != 1:
            raise ValueError(
                f"Program entry function {name!r} has {len(blocks)} blocks; "
                "a single-block consumer cannot select one implicitly"
            )
        return blocks[0]

    def requirements(self, *, entry: str = "main") -> ProgramRequirements:
        """Scan ``entry`` for symbolic bindings and runtime capabilities."""

        from ._analysis import analyze_requirements

        return analyze_requirements(self, entry=entry)

    def readiness(
        self,
        workspace: Mapping[Any, Any] | None = None,
        *,
        entry: str = "main",
    ) -> ReadinessReport:
        """Compare ``entry`` with graph-external execution capabilities."""

        from ._execution import check_readiness

        return check_readiness(self, workspace, entry=entry)

    def transform(
        self,
        program_pass: Pass | PassPipeline,
        *additional_passes: Pass,
        workspace: MutableMapping[Any, Any] | None = None,
    ) -> PipelineResult:
        """Run selected passes over one Program clone and retain ``workspace``.

        The pipeline clones this Program exactly once, then passes the same
        mutable Workspace object to every pass and returns that identical
        object in ``PipelineResult``. Passes may report a legal no-op, and
        pipeline completion alone makes no claim about execution readiness.
        """

        from .passes._base import PassPipeline

        if isinstance(program_pass, PassPipeline):
            if additional_passes:
                raise TypeError(
                    "A PassPipeline cannot be combined with additional passes"
                )
            pipeline = program_pass
        else:
            pipeline = PassPipeline((program_pass, *additional_passes))
        return pipeline.run(self, workspace)

    def run(
        self,
        *args: object,
        workspace: MutableMapping[Any, Any] | None = None,
        entry: str = "main",
        **kwargs: object,
    ) -> Any:
        """Readiness-check and execute ``entry`` with dynamic result typing.

        Runtime argument binding, role-specific input materialization, symbolic
        reference resolution, and interpretation occur for the selected entry.
        The return type is ``Any`` because imported and directly constructed
        Programs carry runtime IR structure rather than a Python callable type.
        """

        from ._execution import run_program

        return run_program(
            self,
            *args,
            workspace=workspace,
            entry=entry,
            **kwargs,
        )

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
