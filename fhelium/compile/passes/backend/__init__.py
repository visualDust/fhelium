"""Resolve Program operations and link live resources for execution."""

from ._assign_implementations import AssignImplementationsPass
from ._assign_ntt_implementation import AssignNttImplementationPass
from ._initialize_resources import InitializeResourceBindingsPass
from ._link_program import LinkProgramPass
from ._materialize_resources import (
    MaterializeResourcesPass,
)
from ._pipeline import backend_linking_pipeline
from ._resolve_operations import ResolveBackendOperationsPass
from ._resolve_tensor_placeholders import ResolveTensorPlaceholdersPass
from ._select_ntt import SelectNttImplementationsPass
from ._validate_representations import ValidateExecutionRepresentationsPass

from ._prepare_operands import PrepareOperationOperandsPass

__all__ = [
    "PrepareOperationOperandsPass",
    "AssignImplementationsPass",
    "AssignNttImplementationPass",
    "SelectNttImplementationsPass",
    "InitializeResourceBindingsPass",
    "LinkProgramPass",
    "MaterializeResourcesPass",
    "ResolveBackendOperationsPass",
    "ResolveTensorPlaceholdersPass",
    "ValidateExecutionRepresentationsPass",
    "backend_linking_pipeline",
]
