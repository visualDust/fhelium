"""Compose semantic lowering, local execution selection, and operation fusion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast
from dataclasses import replace


from fhelium.backend import OperationBackend
from fhelium.backend.implementation import FusionImplementation
from fhelium.ir.dialects import ckks

from ._pipeline import Pass, Pipeline
from .passes.backend import (
    ResolveTensorPlaceholdersPass,
    PrepareOperationOperandsPass,
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


if TYPE_CHECKING:
    from fhelium.backend.ckks.materialization import CkksDeviceResources
    from fhelium.backend.ntt.context import NttContext
    from fhelium.backend.rns.context import RnsContext
    from fhelium.values import EvaluationKeySet


def default_lower_and_fuse_pipeline(
    backend: OperationBackend | None = None,
    *,
    resources: CkksDeviceResources
    | RnsContext
    | NttContext
    | Iterable[CkksDeviceResources | RnsContext | NttContext] = (),
    keys: Mapping[object, object] | Iterable[object] | EvaluationKeySet = (),
) -> Pipeline:
    """Build an editable lowering-and-fusion pipeline for a mixed-level Program.

    The recipe lowers high-level encrypted arithmetic, propagates represented entry state, removes dead values, hoists compatible rotations, reuses internal pure computations, and selects supported fusion regions. Supplied CKKS transitions are preserved. Rescale and relinearization scheduling require caller-added passes before depth and scale assignment.

    Passes read Tensor facts from the Compilation's shared material bindings. Optional resources and keys supply missing numerical data during operand preparation, before dependent fusion choices. Missing keys remain unresolved for caller provisioning.

    Lowerings are selected per operation from registered implementations and their fusion support. Selection uses represented operation and layout facts. Missing preparation facts may remain in the optimized Program until linking requires them.
    """
    backend = OperationBackend() if backend is None else backend
    generated = tuple(
        cast(FusionImplementation, implementation)
        for implementation in backend.registry.implementations
        if callable(getattr(implementation, "match_fusion", None))
    )
    prepare = PrepareOperationOperandsPass(
        backend.registry, resources=resources, keys=keys
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
    passes.append(replace(prepare, name="prepare-before-hoisting"))
    if ckks.RotateManyOp in backend.registry.operation_types:
        passes.append(RotationHoistingPass())
    passes.extend(
        (
            replace(prepare, name="prepare-before-lowering"),
            ResolveTensorPlaceholdersPass(),
            SelectExecutionLoweringsPass(backend.registry, generated),
            SelectNttImplementationsPass(backend.registry),
            replace(prepare, name="prepare-after-lowering"),
            ResolveTensorPlaceholdersPass(),
            ReuseIntermediatesPass(),
        )
    )
    if generated:
        passes.append(FuseOperationsPass(implementations=generated))
    passes.append(EliminateDeadValuesPass())
    return Pipeline(tuple(passes))


__all__ = ["default_lower_and_fuse_pipeline"]
