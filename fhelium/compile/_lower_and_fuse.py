"""Compose semantic lowering, local execution selection, and operation fusion."""

from __future__ import annotations

from typing import cast


from fhelium.backend import OperationBackend
from fhelium.backend.implementation import FusionImplementation
from fhelium.ir.dialects import ckks

from ._pipeline import Pass, Pipeline
from .passes.backend import (
    ResolveTensorPlaceholdersPass,
    SelectNttImplementationsPass,
)
from .passes.ckks import (
    AssignCkksDepthsPass,
    AssignCkksScalesPass,
    RotationHoistingPass,
    InsertMultiplyNttTransitionsPass,
    InsertPlaintextPreparationPass,
    LowerLogicalToCkksPass,
    LowerMessagePlaintextPreparationPass,
    ResolveRotationKeyOperandsPass,
)
from .passes.frontend import LowerSemanticToLogicalPass
from .passes.fusion import FuseOperationsPass
from .passes.lowering._select_execution import SelectExecutionLoweringsPass
from .passes.program import EliminateDeadValuesPass, ReuseIntermediatesPass


def default_lower_and_fuse_pipeline(
    backend: OperationBackend | None = None,
) -> Pipeline:
    """Build an editable lowering-and-fusion pipeline for a mixed-level Program.

    The recipe lowers high-level encrypted arithmetic, propagates represented entry state, removes dead values, hoists compatible rotations, reuses internal pure computations, and selects supported fusion regions. Supplied CKKS transitions are preserved. Rescale and relinearization scheduling require caller-added passes before depth and scale assignment.

    The selected registry is shared through CompileWorkspace. Lowering and implementation selection complete their operation material declarations using that registry and the workspace's optional ``key_layouts`` mapping. Passes read available Tensor facts from the Compilation's material bindings.

    Lowerings are selected per operation from registered implementations and their fusion support. Selection uses represented operation and layout facts. Missing preparation facts may remain in the optimized Program until linking requires them.
    """
    backend = OperationBackend() if backend is None else backend
    generated = tuple(
        cast(FusionImplementation, implementation)
        for implementation in backend.registry.implementations
        if callable(getattr(implementation, "match_fusion", None))
    )
    passes: list[Pass] = [
        ResolveTensorPlaceholdersPass(),
        EliminateDeadValuesPass(),
        ReuseIntermediatesPass(),
        LowerSemanticToLogicalPass(),
        ResolveRotationKeyOperandsPass(),
        InsertMultiplyNttTransitionsPass(),
        InsertPlaintextPreparationPass(),
        LowerLogicalToCkksPass(),
        AssignCkksDepthsPass(),
        AssignCkksScalesPass(),
        LowerMessagePlaintextPreparationPass(),
        ReuseIntermediatesPass(),
        SelectNttImplementationsPass(backend.registry),
    ]
    if ckks.RotateManyOp in backend.registry.operation_types:
        passes.append(RotationHoistingPass())
    passes.extend(
        (
            ResolveTensorPlaceholdersPass(),
            SelectExecutionLoweringsPass(backend.registry, generated),
            SelectNttImplementationsPass(backend.registry),
            ResolveTensorPlaceholdersPass(),
            ReuseIntermediatesPass(),
        )
    )
    if generated:
        passes.append(FuseOperationsPass(implementations=generated))
    passes.append(EliminateDeadValuesPass())
    return Pipeline(tuple(passes))


__all__ = ["default_lower_and_fuse_pipeline"]
