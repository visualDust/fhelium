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

    Captured repetitions with matching source sites and SSA structure may use CKKS repeat regions. Homogeneous bodies are prepared once. Independent accumulated sums with compatible inverse-transform successors retain one body through fusion and Backend preparation; other loops expand before fusion. Set ``workspace['ckks_repeat_fusion']`` to False to expand all bodies before fusion. Set ``workspace['ckks_repeat']`` to False to retain the expanded capture path. Caller-replaced passes and custom execution-cost or lowering policies receive expanded operations.

    Keyed rotation tails with equal layouts share a carried repeat body, including their ordered accumulation. Each row retains its key SSA operand and slot displacement; Backend linking expands the selected dispatch sequence. Set ``workspace['ckks_repeat_rotation']`` to False to retain independent rotation tails.

    The selected registry is shared through CompileWorkspace. Lowering and implementation selection complete their operation material declarations using that registry and the workspace's optional ``key_layouts`` mapping. Passes read available Tensor facts from the Compilation's material bindings.

    Key switches that differ only in their Galois element share one lowered private function; each call passes the automorphism tables of its own rotation, and Backend linking prepares that function once. Fusion does not cross call boundaries. Set ``workspace['shared_key_switch_bodies']`` to False to lower every key switch in place.

    Lowerings are selected per operation from registered implementations and their fusion support. Selection uses represented operation and layout facts. Missing preparation facts may remain in the optimized Program until linking requires them. After Triton fusion, independent large repeats are planned unless ``workspace['triton_repeat']`` is False; the existing NTT scheduler retains their endpoint partitions and synchronization.
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
    from ._repeat import ExpandCkksRepeatsPass

    passes.extend(
        (
            ResolveTensorPlaceholdersPass(),
            SelectExecutionLoweringsPass(backend.registry, generated),
            SelectNttImplementationsPass(backend.registry),
            ResolveTensorPlaceholdersPass(),
            ExpandCkksRepeatsPass(),
            ReuseIntermediatesPass(),
        )
    )
    if generated:
        passes.append(FuseOperationsPass(implementations=generated))
    passes.append(EliminateDeadValuesPass())
    if generated:
        from fhelium.backend.triton._pointwise import TritonFusionImplementation
        from fhelium.backend.triton._repeat import IdentifyTritonRepeatsPass

        # Repeat liveness must observe the finalized fusion body: lowering may
        # leave unused casts that still appear in the producers' use lists.
        passes.append(
            IdentifyTritonRepeatsPass(
                tuple(
                    implementation.name
                    for implementation in generated
                    if isinstance(implementation, TritonFusionImplementation)
                )
            )
        )
    return Pipeline(tuple(passes), verify=False)


def instantiated_ckks_pipeline(backend: OperationBackend) -> Pipeline:
    """Select implementations for a state-instantiated captured CKKS Program.

    Template instantiation has already applied the operation state rules and retained concrete CKKS transitions. Reference resolution, state checks, reuse, rotation hoisting, and guarded execution selection remain instance work.
    """
    pipeline = default_lower_and_fuse_pipeline(backend)
    source_lowering = {
        "lower-semantic-to-logical",
        "insert-multiply-ntt-transitions",
        "insert-plaintext-preparation",
        "lower-logical-to-ckks",
    }
    return Pipeline(
        tuple(
            item for item in pipeline.passes if item.name not in source_lowering
        ),
        verify=pipeline.verify,
    )


__all__ = ["default_lower_and_fuse_pipeline"]
