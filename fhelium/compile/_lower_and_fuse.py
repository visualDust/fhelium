"""Compose semantic lowering, local execution selection, and operation fusion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast


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
    """Build the default lowering-and-fusion recipe for a mixed-level Program.

    The returned Pipeline is an ordinary editable value. Known Tensor facts are
    read from supplied bindings; no callable signature, Engine, data factory or
    device-wide routing policy participates. Explicit CKKS state transitions
    remain authoritative. High-level encrypted arithmetic uses the existing
    logical/CKKS passes and propagates represented entry state. No rescale or
    relinearization is inserted by this baseline recipe; callers can insert
    those scheduling passes before depth and scale assignment.

    Optional resources and keys supply numerical data through the existing
    operand-preparation stage. Every pass reads the current Compilation's single
    material dictionary; bindings are available before dependent fusion choices.
    Missing keys stay unresolved and are never generated.

    Lowerings are chosen per operation using the registered implementations and
    their fusion support. The policy does not benchmark alternatives or retry
    failed execution through another implementation. Missing preparation facts
    can remain in the optimized Program until execution linking needs them.
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
    if ckks.RotateManyOp in backend.registry.operation_types:
        passes.append(RotationHoistingPass())
    passes.extend(
        (
            prepare,
            ResolveTensorPlaceholdersPass(),
            SelectExecutionLoweringsPass(backend.registry, generated),
            SelectNttImplementationsPass(backend.registry),
            prepare,
            ResolveTensorPlaceholdersPass(),
            ReuseIntermediatesPass(),
        )
    )
    if generated:
        passes.append(FuseOperationsPass(implementations=generated))
    passes.append(EliminateDeadValuesPass())
    return Pipeline(tuple(passes))


__all__ = ["default_lower_and_fuse_pipeline"]
