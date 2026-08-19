"""Managed local value placement, admission, lifetimes, and automation."""

from fhelium.residency.controller import (
    ResidencyController,
    ResidencyDecision,
    ResidencyEviction,
    ResidencyUse,
)
from fhelium.residency.lease import (
    BorrowedValues,
    ResidencyHold,
    ResidencyLease,
    ResidencyReservation,
)
from fhelium.residency.location import (
    PAGEABLE_HOST,
    PINNED_HOST,
    ResidencyLocation,
    ResidencyLocationKind,
    cuda_location,
)
from fhelium.residency.manager import (
    ResidencyManager,
    ResidencyScope,
)
from fhelium.residency.model import (
    Recoverability,
    ReplicaMode,
    ResidencyHandle,
    ResidencySource,
    ResidencyValueSpec,
)
from fhelium.residency.plan import (
    DiscardValue,
    DropResident,
    EnsureResident,
    MemoryReservation,
    MoveResident,
    ResidencyAction,
    ResidencyPlan,
)
from fhelium.residency.policy import (
    DeterministicTieredLRU,
    ResidencyEvictionCandidate,
    ResidencyPolicy,
    ResidencyPolicyMetadata,
)
from fhelium.residency.request import (
    ResidencyRequest,
    ResidencyRequirement,
)
from fhelium.residency.snapshot import (
    MaterializationSnapshot,
    ResidencyActionExplanation,
    ResidencyLocationSnapshot,
    ResidencyPlanExplanation,
    ResidencyPlanReport,
    ResidencyReservationSnapshot,
    ResidencySnapshot,
    ResidencyTransitionReport,
    ResidencyValueSnapshot,
)

__all__ = [
    "PAGEABLE_HOST",
    "PINNED_HOST",
    "BorrowedValues",
    "DeterministicTieredLRU",
    "DiscardValue",
    "DropResident",
    "EnsureResident",
    "MaterializationSnapshot",
    "MemoryReservation",
    "MoveResident",
    "Recoverability",
    "ReplicaMode",
    "ResidencyAction",
    "ResidencyActionExplanation",
    "ResidencyController",
    "ResidencyDecision",
    "ResidencyEviction",
    "ResidencyEvictionCandidate",
    "ResidencyHandle",
    "ResidencyHold",
    "ResidencyLease",
    "ResidencyLocation",
    "ResidencyLocationKind",
    "ResidencyLocationSnapshot",
    "ResidencyManager",
    "ResidencyPlan",
    "ResidencyPlanExplanation",
    "ResidencyPlanReport",
    "ResidencyPolicy",
    "ResidencyPolicyMetadata",
    "ResidencyRequest",
    "ResidencyRequirement",
    "ResidencyReservation",
    "ResidencyReservationSnapshot",
    "ResidencyScope",
    "ResidencySnapshot",
    "ResidencySource",
    "ResidencyTransitionReport",
    "ResidencyUse",
    "ResidencyValueSnapshot",
    "ResidencyValueSpec",
    "cuda_location",
]
