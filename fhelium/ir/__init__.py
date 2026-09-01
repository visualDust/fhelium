"""Represent and inspect permissive mixed-level FHElium programs.

The package owns xDSL module structure, registered multi-level dialects, open
FHElium value/reference types, read-only Program analyses, and formatting.
It does not own frontend capture, numerical correctness, backend coverage, live
runtime objects, or execution.
"""

from . import dialects
from ._analysis import (
    ProgramInventory,
    ValueState,
    analyze_evaluation_key_requirements,
    inventory_program,
    analyze_value_states,
)
from ._dialect import (
    DIALECT_VERSION,
    DIALECT_VERSION_ATTRIBUTE,
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTRIBUTE,
    EncryptedType,
    FHElium,
    MaterialRefOp,
    MaterialType,
    MessageType,
    PlaintextType,
    ResourceRefOp,
    ResourceType,
    ValueRole,
    create_dialect_context,
    create_operation,
    operation_name,
    value_role,
    value_type,
)
from ._program import Program
from ._state import (
    InferredValueState,
    StateFact,
    StateStatus,
    SymbolicExpression,
    analyze_state_flow,
)
from ._format import ProgramFormatDetail, format_program
from ._operation_specs import (
    DEFAULT_OPERATION_SPECS,
    OperationEffect,
    OperationSpec,
    OperationSpecRegistry,
    OperationValidator,
)
from ._operation_catalog import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from .dialects import (
    FHEliumCkks,
    FHEliumLogical,
    FHEliumSemantic,
    REGISTERED_DIALECTS,
    Torch,
)


def load(path: str) -> Program:
    """Load one textual mixed-level Program."""

    return Program.load(path)


def parse(text: str, *, source_name: str = "<unknown>") -> Program:
    """Parse one textual mixed-level Program."""

    return Program.parse(text, source_name=source_name)


__all__ = [
    "DIALECT_VERSION",
    "DIALECT_VERSION_ATTRIBUTE",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_ATTRIBUTE",
    "EncryptedType",
    "EXECUTION_IMPLEMENTATION_ATTRIBUTE",
    "FHElium",
    "FHEliumCkks",
    "FHEliumLogical",
    "FHEliumSemantic",
    "InferredValueState",
    "MaterialRefOp",
    "MaterialType",
    "REGISTERED_DIALECTS",
    "MessageType",
    "DEFAULT_OPERATION_SPECS",
    "OperationEffect",
    "OperationSpec",
    "OperationSpecRegistry",
    "OperationValidator",
    "PlaintextType",
    "Program",
    "ProgramFormatDetail",
    "ProgramInventory",
    "ResourceRefOp",
    "ResourceType",
    "StateFact",
    "StateStatus",
    "SymbolicExpression",
    "Torch",
    "ValueRole",
    "ValueState",
    "analyze_evaluation_key_requirements",
    "analyze_state_flow",
    "inventory_program",
    "analyze_value_states",
    "create_dialect_context",
    "create_operation",
    "dialects",
    "format_program",
    "load",
    "operation_name",
    "parse",
    "value_role",
    "value_type",
]
