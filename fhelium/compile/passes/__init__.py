"""Built-in partial transforms for captured tensor and CKKS operations."""

from .backend import (
    AssignImplementationsPass,
    AssignNttImplementationPass,
    InitializeResourceBindingsPass,
    LinkProgramPass,
    MaterializeResourcesPass,
    ResolveBackendOperationsPass,
    ResolveTensorPlaceholdersPass,
    SelectNttImplementationsPass,
    ValidateExecutionRepresentationsPass,
    backend_linking_pipeline,
)
from .ckks import (
    AssignCkksDepthsPass,
    AssignCkksScalesPass,
    RotationHoistingPass as RotationHoistingPass,
    InsertMultiplyNttTransitionsPass,
    InsertPlaintextPreparationPass,
    InsertRelinearizationPass,
    InsertRescalePass,
    LateRelinearizationPass,
    LateRescalePass,
    LowerLogicalToCkksPass,
    LowerMessagePlaintextPreparationPass,
    ResolveRotationKeyOperandsPass,
)
from .codegen import EmitBackendPythonPass, EmitEagerPythonPass
from .distributed import LowerSpecializedCollectivesPass
from .frontend import LowerSemanticToLogicalPass
from .fusion import FuseOperationsPass
from .lowering import LowerCkksToRnsNttPass as LowerCkksToRnsNttPass
from .program import (
    EliminateDeadValuesPass,
    ReuseIntermediatesPass,
    SvgGraphDirection,
    SvgGraphError,
    SvgGraphField,
    SvgGraphOutput,
    SvgGraphPresentation,
    SvgGraphTheme,
    SvgGraphVisualizationPass,
    SvgNodeSection,
    SvgOperationContext,
    default_svg_operation_color_key,
)

from .backend import PrepareOperationOperandsPass

__all__ = [
    "PrepareOperationOperandsPass",
    "SelectExecutionLoweringsPass",
    "FuseOperationsPass",
    "InitializeResourceBindingsPass",
    "AssignImplementationsPass",
    "AssignNttImplementationPass",
    "SelectNttImplementationsPass",
    "AssignCkksDepthsPass",
    "AssignCkksScalesPass",
    "EliminateDeadValuesPass",
    "ReuseIntermediatesPass",
    "EmitBackendPythonPass",
    "EmitEagerPythonPass",
    "RotationHoistingPass",
    "InsertMultiplyNttTransitionsPass",
    "InsertPlaintextPreparationPass",
    "InsertRelinearizationPass",
    "InsertRescalePass",
    "LateRelinearizationPass",
    "LateRescalePass",
    "LowerCkksToRnsNttPass",
    "LowerLogicalToCkksPass",
    "LowerMessagePlaintextPreparationPass",
    "LowerSemanticToLogicalPass",
    "LowerSpecializedCollectivesPass",
    "LinkProgramPass",
    "MaterializeResourcesPass",
    "ResolveBackendOperationsPass",
    "ResolveTensorPlaceholdersPass",
    "ValidateExecutionRepresentationsPass",
    "ResolveRotationKeyOperandsPass",
    "SvgGraphDirection",
    "SvgGraphError",
    "SvgGraphField",
    "SvgGraphOutput",
    "SvgGraphPresentation",
    "SvgGraphTheme",
    "SvgGraphVisualizationPass",
    "SvgNodeSection",
    "SvgOperationContext",
    "backend_linking_pipeline",
    "default_svg_operation_color_key",
]

from .lowering import SelectExecutionLoweringsPass
