"""Capture, transform, and execute reusable FHE computations.

The package carries one Program, its Tensor material bindings, caller-extensible
`CompileWorkspace`, and ordered pass reports in a `Compilation`. The `passes.lowering` package maps CKKS
operations to logical RNS/NTT composition in caller-composed pipelines.
Frontend, CKKS, and lowering passes can stop at any represented IR abstraction level;
callers may inspect or export that Program, continue through an external
xDSL/MLIR pipeline, or bind backend resources. Low-level capture and Program
transformation do not require live execution resources. The high-level
``compile`` function produces a lazily prepared ``CompiledCallable`` whose
Backend supplies implementations and resources.
"""

from ._callable import CompiledCallable
from ._compilation import Compilation
from ._driver import compile
from ._errors import (
    CaptureError,
    CompileError,
    CompileInputError,
    PlanningError,
)
from ._materials import prepare_material_bindings
from ._pipeline import (
    DecisionRecord,
    Pass,
    PassReport,
    PassResult,
    PassStats,
    Pipeline,
    TransformError,
)
from ._preparation import Specialization
from ._specialization import (
    ArgumentSignature,
    CallSignature,
    SpecializationMiss,
)
from ._workspace import CompileWorkspace
from .codegen import (
    BackendPythonSource,
    EagerPythonSource,
    GeneratedPythonSource,
    PythonCodegenError,
)
from .frontend._capture import capture
from .frontend._captured_callable import CapturedCallable
from .frontend._eager_capture import capture_eager
from .frontend._specs import (
    BatchMode,
    InputSpec,
    SlotExtent,
    StaticValue,
    encrypted,
    message,
    plaintext,
    static,
)
from .passes import (
    AssignCkksDepthsPass,
    AssignCkksScalesPass,
    AssignImplementationsPass,
    AssignNttImplementationPass,
    EliminateDeadValuesPass,
    EmitBackendPythonPass,
    EmitEagerPythonPass,
    FuseOperationsPass,
    RotationHoistingPass,
    InitializeResourceBindingsPass,
    InsertMultiplyNttTransitionsPass,
    InsertPlaintextPreparationPass,
    InsertRelinearizationPass,
    InsertRescalePass,
    LateRelinearizationPass,
    LateRescalePass,
    LinkProgramPass,
    LowerCkksToRnsNttPass,
    LowerLogicalToCkksPass,
    LowerMessagePlaintextPreparationPass,
    LowerSemanticToLogicalPass,
    MaterializeResourcesPass,
    ResolveBackendOperationsPass,
    ResolveRotationKeyOperandsPass,
    ResolveTensorPlaceholdersPass,
    ReuseIntermediatesPass,
    SelectExecutionLoweringsPass,
    SelectNttImplementationsPass,
    SvgGraphDirection,
    SvgGraphError,
    SvgGraphField,
    SvgGraphOutput,
    SvgGraphPresentation,
    SvgGraphTheme,
    SvgGraphVisualizationPass,
    SvgNodeSection,
    SvgOperationContext,
    ValidateExecutionRepresentationsPass,
    backend_linking_pipeline,
    default_svg_operation_color_key,
)
from .passes.lowering import (
    DEFAULT_CKKS_LOWERINGS,
    CkksLoweringDefinition,
    CkksLoweringRegistry,
    LoweredCkksOperation,
    lower_ckks_program,
)

from .passes.backend import PrepareOperationOperandsPass

__all__ = [
    "PrepareOperationOperandsPass",
    "ArgumentSignature",
    "CallSignature",
    "CompiledCallable",
    "Specialization",
    "SpecializationMiss",
    "FuseOperationsPass",
    "SelectExecutionLoweringsPass",
    "InitializeResourceBindingsPass",
    "AssignImplementationsPass",
    "AssignNttImplementationPass",
    "SelectNttImplementationsPass",
    "AssignCkksDepthsPass",
    "AssignCkksScalesPass",
    "BatchMode",
    "CaptureError",
    "CapturedCallable",
    "Compilation",
    "prepare_material_bindings",
    "CompileWorkspace",
    "CompileError",
    "CompileInputError",
    "CkksLoweringDefinition",
    "CkksLoweringRegistry",
    "DEFAULT_CKKS_LOWERINGS",
    "DecisionRecord",
    "EliminateDeadValuesPass",
    "ReuseIntermediatesPass",
    "EmitBackendPythonPass",
    "EmitEagerPythonPass",
    "BackendPythonSource",
    "EagerPythonSource",
    "GeneratedPythonSource",
    "RotationHoistingPass",
    "InputSpec",
    "InsertMultiplyNttTransitionsPass",
    "InsertPlaintextPreparationPass",
    "InsertRelinearizationPass",
    "InsertRescalePass",
    "LateRelinearizationPass",
    "LateRescalePass",
    "LowerCkksToRnsNttPass",
    "LoweredCkksOperation",
    "LowerLogicalToCkksPass",
    "LowerMessagePlaintextPreparationPass",
    "LowerSemanticToLogicalPass",
    "LinkProgramPass",
    "PlanningError",
    "Pass",
    "PassReport",
    "PassResult",
    "PassStats",
    "Pipeline",
    "PythonCodegenError",
    "MaterializeResourcesPass",
    "ResolveBackendOperationsPass",
    "ResolveTensorPlaceholdersPass",
    "ValidateExecutionRepresentationsPass",
    "ResolveRotationKeyOperandsPass",
    "SlotExtent",
    "StaticValue",
    "SvgGraphDirection",
    "SvgGraphError",
    "SvgGraphField",
    "SvgGraphOutput",
    "SvgGraphPresentation",
    "SvgGraphTheme",
    "SvgGraphVisualizationPass",
    "SvgNodeSection",
    "SvgOperationContext",
    "TransformError",
    "backend_linking_pipeline",
    "capture",
    "capture_eager",
    "compile",
    "default_svg_operation_color_key",
    "encrypted",
    "lower_ckks_program",
    "message",
    "plaintext",
    "static",
]

from ._lower_and_fuse import default_lower_and_fuse_pipeline

__all__ += ["default_lower_and_fuse_pipeline"]
