"""Construct the optional standard JIT lowering and scheduling policy."""

from __future__ import annotations

from ._base import PassPipeline
from .eliminate_dead_values import EliminateDeadValuesPass
from .insert_multiply_ntt_transitions import InsertMultiplyNttTransitionsPass
from .insert_plaintext_preparation import InsertPlaintextPreparationPass
from .insert_relinearization import InsertRelinearizationPass
from .insert_rescale import InsertRescalePass
from .late_relinearization import LateRelinearizationPass
from .late_rescale import LateRescalePass
from .lower_logical_to_ckks import LowerLogicalToCkksPass
from .lower_semantic_to_logical import LowerSemanticToLogicalPass


def default_pipeline() -> PassPipeline:
    """Return the standard semantic-to-CKKS lowering and scheduling policy.

    The ordered passes scan all top-level function blocks, remove dead
    known-pure values, lower recognized semantic operations, and materialize
    plaintext preparation, NTT transitions, relinearization, and
    rescale obligations. The late-rescale and late-relinearization steps are
    reporting-only legal no-ops that count existing candidates and retain the
    Program unchanged. Callers add backend-specific movement optimizers and an
    validation pass according to their execution policy; ``run`` also
    performs its independent selected-entry readiness gate.
    """

    return PassPipeline(
        (
            EliminateDeadValuesPass(),
            LowerSemanticToLogicalPass(),
            InsertPlaintextPreparationPass(),
            InsertMultiplyNttTransitionsPass(),
            LowerLogicalToCkksPass(),
            InsertRelinearizationPass(),
            InsertRescalePass(),
            LateRescalePass(),
            LateRelinearizationPass(),
        )
    )
