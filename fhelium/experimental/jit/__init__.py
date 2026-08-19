"""Capture, import, transform, and execute mixed-dialect xDSL programs.

``Program`` is the package's single source-independent graph abstraction.
PyTorch capture, textual import, and direct xDSL construction all produce this
same representation. Program construction and import perform structural
verification while registered and unregistered dialect content remains
available for interchange. ``check_readiness`` and ``run`` perform the separate
numerical and execution checks for one selected entry.
Pass implementations may use interim states while mutating a module, but each
returned Program must be structurally valid: ``PassPipeline`` verifies every
pass result, and the execution gate verifies structure again.

Programs contain serializable IR and symbolic references. ``Workspace`` retains
live materials, resources, handlers, cryptographic services, and pass analyses
outside that IR. Built-in transformation passes normally inspect every direct
operation in every top-level function block, whereas requirement analysis,
readiness, execution, and entry-oriented utilities operate on the selected
selected entry function.
"""

from collections.abc import MutableMapping
from os import PathLike
from typing import Any

from ._analysis import (
    InferredValueState,
    ProgramRequirements,
    analyze_evaluation_key_requirements,
    analyze_requirements,
    analyze_value_states,
)
from ._capture import CaptureResult, capture as trace
from ._errors import (
    JitError,
    JitInputError,
    JitPassError,
    JitPlanningError,
    JitTraceError,
)
from ._execution import (
    BindingResolver,
    OperationHandler,
    ProgramNotReadyError,
    ReadinessDiagnostic,
    ReadinessReport,
    check_readiness,
)
from ._program import Program
from ._specs import (
    BatchMode,
    InputSpec,
    SlotExtent,
    StaticValue,
    encrypted,
    message,
    plaintext,
    static,
)
from ._workspace import Workspace
from .passes import (
    EliminateDeadValuesPass,
    InsertMultiplyNttTransitionsPass,
    InsertPlaintextPreparationPass,
    InsertRelinearizationPass,
    InsertRescalePass,
    LateRelinearizationPass,
    LateRescalePass,
    LowerLogicalToCkksPass,
    LowerSemanticToLogicalPass,
    Pass,
    PassPipeline,
    PassReport,
    PassResult,
    PassStats,
    PipelineResult,
    StateValidator,
    SvgGraphVisualizationPass,
    ValidateCipherStatesPass,
    ValidateExecutableGraphPass,
    default_pipeline,
    validate_executable_graph,
)


def load(path: str | PathLike[str]) -> Program:
    """Load and structurally verify one textual mixed-dialect ``Program``."""

    return Program.load(path)


def parse(text: str, *, source_name: str = "<unknown>") -> Program:
    """Parse and structurally verify one textual mixed-dialect ``Program``."""

    return Program.parse(text, source_name=source_name)


def run(
    program: Program,
    *args: object,
    workspace: MutableMapping[Any, Any] | None = None,
    entry: str = "main",
    **kwargs: object,
) -> Any:
    """Readiness-check and execute ``entry`` from the supplied ``Program``.

    The result type is dynamic because textual and directly constructed
    Programs have no associated Python callable return annotation. A captured
    callable retains its static return type on ``CaptureResult.reference``;
    execution reconstructs the runtime value described by Program output IR.
    """

    return program.run(
        *args,
        workspace=workspace,
        entry=entry,
        **kwargs,
    )


__all__ = [
    "BatchMode",
    "BindingResolver",
    "CaptureResult",
    "EliminateDeadValuesPass",
    "InferredValueState",
    "InputSpec",
    "InsertMultiplyNttTransitionsPass",
    "InsertPlaintextPreparationPass",
    "InsertRelinearizationPass",
    "InsertRescalePass",
    "JitError",
    "JitInputError",
    "JitPassError",
    "JitPlanningError",
    "JitTraceError",
    "LateRelinearizationPass",
    "LateRescalePass",
    "LowerLogicalToCkksPass",
    "LowerSemanticToLogicalPass",
    "OperationHandler",
    "Pass",
    "PassPipeline",
    "PassReport",
    "PassResult",
    "PassStats",
    "PipelineResult",
    "Program",
    "ProgramNotReadyError",
    "ProgramRequirements",
    "ReadinessDiagnostic",
    "ReadinessReport",
    "SlotExtent",
    "StateValidator",
    "StaticValue",
    "SvgGraphVisualizationPass",
    "ValidateCipherStatesPass",
    "ValidateExecutableGraphPass",
    "Workspace",
    "analyze_evaluation_key_requirements",
    "analyze_requirements",
    "analyze_value_states",
    "check_readiness",
    "default_pipeline",
    "encrypted",
    "load",
    "message",
    "parse",
    "plaintext",
    "run",
    "static",
    "trace",
    "validate_executable_graph",
]
