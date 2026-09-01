"""Identify operations that require registered Backend implementations."""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Operation

from fhelium.backend.execution import ProgramDispatchTable
from fhelium.backend.resources import ResourceRequirement
from fhelium.ir import Program
from fhelium.ir.dialects import core, distributed

_STRUCTURAL_OPERATION_TYPES = (
    UnrealizedConversionCastOp,
    core.MaterialRefOp,
    core.ResourceRefOp,
    ReturnOp,
    scf.ForOp,
    scf.IfOp,
    scf.YieldOp,
    distributed.YieldOp,
    arith.ConstantOp,
    arith.AddiOp,
    arith.SubiOp,
    arith.MuliOp,
    arith.DivUIOp,
    arith.DivSIOp,
    arith.RemUIOp,
    arith.RemSIOp,
    arith.MinUIOp,
    arith.MaxUIOp,
    arith.IndexCastOp,
    arith.SelectOp,
    arith.CmpiOp,
)


def executable_operations(program: Program) -> tuple[Operation, ...]:
    """Return operations that require registered Backend implementations."""

    return tuple(
        operation
        for operation in program.single_block("main").walk()
        if not isinstance(operation, _STRUCTURAL_OPERATION_TYPES)
    )


def program_resource_requirements(
    program: Program,
    dispatch_table: ProgramDispatchTable,
) -> tuple[
    dict[Operation, ResourceRequirement],
    tuple[ResourceRequirement, ...],
]:
    """Collect Program resource references and dispatch requirements."""

    requirements_by_symbol: dict[str, ResourceRequirement] = {}

    def record(requirement: ResourceRequirement) -> None:
        previous = requirements_by_symbol.get(requirement.symbol)
        if previous is not None and previous != requirement:
            raise ValueError(
                f"Resource symbol {requirement.symbol!r} has conflicting "
                "Program-wide requirements"
            )
        requirements_by_symbol.setdefault(requirement.symbol, requirement)

    resource_references: dict[Operation, ResourceRequirement] = {}
    for operation in program.single_block("main").walk():
        if not isinstance(operation, core.ResourceRefOp):
            continue
        if operation.symbol is None or operation.kind is None:
            raise ValueError("Resource reference requires symbol and kind")
        requirement = ResourceRequirement(
            operation.symbol.data,
            operation.kind.data,
        )
        record(requirement)
        resource_references[operation] = requirement

    for dispatch in dispatch_table.operations.values():
        for requirement in dispatch.requirements:
            record(requirement)

    return resource_references, tuple(requirements_by_symbol.values())


def program_material_values(
    program: Program,
    materials: Mapping[str, object],
) -> dict[Operation, object]:
    """Match every Program material reference to caller-provided data."""

    bound: dict[Operation, object] = {}
    for operation in program.single_block("main").walk():
        if not isinstance(operation, core.MaterialRefOp):
            continue
        if operation.symbol is None:
            raise ValueError("Material reference requires a symbol")
        symbol = operation.symbol.data
        try:
            bound[operation] = materials[symbol]
        except KeyError:
            raise KeyError(
                f"Required material {symbol!r} is not provided"
            ) from None
    return bound
