"""Link called Program functions once and execute calls through their executables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.func import CallOp

from fhelium.backend.execution import OperationBackend, ProgramExecutable
from fhelium.backend._planning import destination_results
from fhelium.backend.runner import CalledSchedule
from fhelium.ir import Program

from ..._pipeline import PassResult

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


@dataclass(frozen=True)
class CalledProgram:
    """Lower ``func.call`` operations to calls of their callee's lowered schedule."""

    executable: ProgramExecutable
    symbol: str | None = None
    name: str = "called-program"
    operation_types: tuple[type, ...] = (CallOp,)
    supports_in_place: bool = False

    @property
    def effect(self) -> str:
        effects = {
            dispatch.effect
            for dispatch in self.executable.dispatch_table.operations.values()
        }
        return "pure" if effects <= {"pure"} else "opaque"

    def resource_requirements(self, invocation) -> tuple[()]:
        return ()

    def lower(self, invocation, inputs, resources, builder, /):
        from ._prepare_host import runtime_input_layout

        layouts = tuple(runtime_input_layout(value) for value in inputs)
        lowered = self.executable._lowered.get(layouts)
        if lowered is None:
            lowered = self.executable._lower(layouts)
        # The callee writes results it allocates alone into storage the
        # caller places, such as its arena, instead of allocating them.
        destinations = destination_results(lowered.schedule)
        return builder.call_schedule(
            CalledSchedule(self.executable, layouts, destinations),
            inputs,
            lowered.result_layouts,
            destinations,
        )


class LinkedFunctions(dict):
    """Map callee symbols of the linked ``main`` function to their executables."""


@dataclass(frozen=True)
class LinkCalledFunctionsPass:
    """Link each function called from ``main`` as its own executable.

    A callee is linked once with the caller's Backend workspace and Tensor table, and every call site shares the result. Material references inside the callee bind by symbol.
    """

    backend: OperationBackend
    name: str = "link-called-functions"

    def run(self, compilation: "Compilation") -> PassResult:
        from fhelium.compile import Compilation, CompileWorkspace

        program = compilation.program
        functions = {
            function.sym_name.data: function for function in program.functions
        }
        linked = LinkedFunctions()
        for operation in program.single_block("main").ops:
            if not isinstance(operation, CallOp):
                continue
            callee = operation.callee.root_reference.data
            if callee in linked:
                continue
            function = functions[callee].clone()
            function.properties["sym_name"] = StringAttr("main")
            function.properties.pop("sym_visibility", None)
            callee_program = Program(
                ModuleOp([function], attributes=dict(program.module.attributes))
            )
            workspace = CompileWorkspace(compilation.workspace)
            workspace.pop(LinkedFunctions, None)
            executable = self.backend.link(
                Compilation(
                    callee_program,
                    workspace,
                    material_bindings=compilation.material_bindings,
                ),
                clone=False,
            )
            object.__setattr__(executable, "_callee", callee)
            linked[callee] = CalledProgram(executable, callee)
        compilation.workspace[LinkedFunctions] = linked
        return PassResult.unchanged(program, matched=len(linked))


__all__ = ["CalledProgram", "LinkCalledFunctionsPass", "LinkedFunctions"]
