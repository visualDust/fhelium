"""Immutable observations and reports for managed residency state.

Snapshots expose logical identities, materialization protection, and strict
per-location accounting without retaining tensor-bearing values or mutable
manager internals.  Explanation records describe a dry-run decision, while
transition and plan reports describe completed synchronous manager work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from fhelium.core import TensorResident
from fhelium.residency.location import ResidencyLocation
from fhelium.residency.model import (
    ResidencyHandle,
    ResidencyValueSpec,
)
from fhelium.residency.plan import (
    DiscardValue,
    DropResident,
    EnsureResident,
    MemoryReservation,
    MoveResident,
    ResidencyAction,
)

ValueT_co = TypeVar("ValueT_co", bound=TensorResident, covariant=True)
_ACTION_TYPES = (EnsureResident, MoveResident, DropResident, DiscardValue)


def _require_nonnegative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_optional_nonnegative_int(
    value: object | None,
    *,
    field_name: str,
) -> None:
    if value is not None:
        _require_nonnegative_int(value, field_name=field_name)


def _require_bool(value: object, *, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_optional_reason(value: object | None, *, field_name: str) -> None:
    if value is not None:
        _require_nonempty_string(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class MaterializationSnapshot:
    """Protection and byte state of one value at one residency location.

    ``storage_nbytes`` is the current materialization's actual unique backing
    storage. ``charged_nbytes`` is the fixed conservative admission charge and
    can be larger after functional movement compacts a view-backed allocation.
    ``pending_event_count`` records completed lease closures whose CUDA work
    has not yet reached its recorded event.  Such a materialization can have
    ``use_count == 0`` while remaining protected from removal.
    """

    location: ResidencyLocation
    logical_nbytes: int
    storage_nbytes: int
    charged_nbytes: int
    use_count: int
    hold_count: int
    pending_event_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.location, ResidencyLocation):
            raise TypeError(
                "MaterializationSnapshot location must be a ResidencyLocation"
            )
        for field_name in (
            "logical_nbytes",
            "storage_nbytes",
            "charged_nbytes",
            "use_count",
            "hold_count",
            "pending_event_count",
        ):
            _require_nonnegative_int(
                getattr(self, field_name),
                field_name=f"MaterializationSnapshot {field_name}",
            )
        if self.charged_nbytes < max(
            self.logical_nbytes,
            self.storage_nbytes,
        ):
            raise ValueError(
                "MaterializationSnapshot charged_nbytes must cover logical "
                "and actual storage bytes"
            )


@dataclass(frozen=True, slots=True)
class ResidencyValueSnapshot(Generic[ValueT_co]):
    """Public state of one registered or discarded managed value.

    ``source_location`` identifies reconstruction placement when ``has_source``
    is true. Discarded handles remain visible so diagnostics can explain why
    they are rejected; they have no source or live materializations.
    """

    handle: ResidencyHandle[ValueT_co]
    spec: ResidencyValueSpec[ValueT_co]
    materializations: tuple[MaterializationSnapshot, ...]
    has_source: bool
    source_location: ResidencyLocation | None
    discarded: bool

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ResidencyHandle):
            raise TypeError(
                "ResidencyValueSnapshot handle must be a ResidencyHandle"
            )
        if not isinstance(self.spec, ResidencyValueSpec):
            raise TypeError(
                "ResidencyValueSnapshot spec must be a ResidencyValueSpec"
            )
        materializations = tuple(self.materializations)
        if any(
            not isinstance(item, MaterializationSnapshot)
            for item in materializations
        ):
            raise TypeError(
                "ResidencyValueSnapshot materializations must contain "
                "MaterializationSnapshot objects"
            )
        if len({item.location for item in materializations}) != len(
            materializations
        ):
            raise ValueError(
                "ResidencyValueSnapshot materialization locations must be "
                "unique"
            )
        _require_bool(
            self.has_source,
            field_name="ResidencyValueSnapshot has_source",
        )
        if self.source_location is not None and not isinstance(
            self.source_location,
            ResidencyLocation,
        ):
            raise TypeError(
                "ResidencyValueSnapshot source_location must be a "
                "ResidencyLocation"
            )
        if self.has_source != (self.source_location is not None):
            raise ValueError(
                "ResidencyValueSnapshot source presence and location must agree"
            )
        _require_bool(
            self.discarded,
            field_name="ResidencyValueSnapshot discarded",
        )
        if self.discarded and (
            self.has_source
            or self.source_location is not None
            or materializations
        ):
            raise ValueError(
                "A discarded ResidencyValueSnapshot cannot retain a source "
                "or materializations"
            )
        object.__setattr__(self, "materializations", materializations)


@dataclass(frozen=True, slots=True)
class ResidencyReservationSnapshot:
    """One active named accounting reservation without an allocated tensor."""

    location: ResidencyLocation
    nbytes: int
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.location, ResidencyLocation):
            raise TypeError(
                "ResidencyReservationSnapshot location must be a "
                "ResidencyLocation"
            )
        _require_nonnegative_int(
            self.nbytes,
            field_name="ResidencyReservationSnapshot nbytes",
        )
        _require_nonempty_string(
            self.label,
            field_name="ResidencyReservationSnapshot label",
        )


@dataclass(frozen=True, slots=True)
class ResidencyLocationSnapshot:
    """Budget, usage, and protection accounting for one location.

    ``budget_bytes`` and ``remaining_budget_bytes`` are both ``None`` when the
    manager applies no admission limit at this location. With a budget,
    ``remaining_budget_bytes`` equals the budget minus current materialization
    and reservation charges. ``peak_used_bytes`` is the maximum materialization
    charge observed by the manager and excludes reservations;
    ``peak_charged_bytes`` includes `MemoryReservation` and temporary charges. CUDA
    locations additionally report process-wide PyTorch allocator allocated and
    reserved bytes at capture time; those metrics include storage outside this
    manager. Counts are aggregates over current materializations and named
    reservations at this location.
    """

    location: ResidencyLocation
    budget_bytes: int | None
    used_bytes: int
    reserved_bytes: int
    remaining_budget_bytes: int | None
    peak_used_bytes: int
    peak_charged_bytes: int
    value_count: int
    reservation_count: int
    use_count: int
    hold_count: int
    pending_event_count: int
    allocator_allocated_bytes: int | None
    allocator_reserved_bytes: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.location, ResidencyLocation):
            raise TypeError(
                "ResidencyLocationSnapshot location must be a ResidencyLocation"
            )
        for field_name in (
            "used_bytes",
            "reserved_bytes",
            "peak_used_bytes",
            "peak_charged_bytes",
            "value_count",
            "reservation_count",
            "use_count",
            "hold_count",
            "pending_event_count",
        ):
            _require_nonnegative_int(
                getattr(self, field_name),
                field_name=f"ResidencyLocationSnapshot {field_name}",
            )
        _require_optional_nonnegative_int(
            self.budget_bytes,
            field_name="ResidencyLocationSnapshot budget_bytes",
        )
        _require_optional_nonnegative_int(
            self.remaining_budget_bytes,
            field_name="ResidencyLocationSnapshot remaining_budget_bytes",
        )
        if self.peak_used_bytes < self.used_bytes:
            raise ValueError(
                "ResidencyLocationSnapshot peak_used_bytes cannot be less "
                "than used_bytes"
            )
        minimum_peak_charged = max(
            self.peak_used_bytes,
            self.used_bytes + self.reserved_bytes,
        )
        if self.peak_charged_bytes < minimum_peak_charged:
            raise ValueError(
                "ResidencyLocationSnapshot peak_charged_bytes cannot be less "
                "than current or peak materialization charges"
            )
        if self.budget_bytes is None:
            if self.remaining_budget_bytes is not None:
                raise ValueError(
                    "An unbudgeted ResidencyLocationSnapshot must have "
                    "remaining_budget_bytes=None"
                )
        else:
            if self.used_bytes + self.reserved_bytes > self.budget_bytes:
                raise ValueError(
                    "ResidencyLocationSnapshot charges cannot exceed "
                    "budget_bytes"
                )
            expected_remaining = (
                self.budget_bytes - self.used_bytes - self.reserved_bytes
            )
            if self.remaining_budget_bytes != expected_remaining:
                raise ValueError(
                    "ResidencyLocationSnapshot remaining_budget_bytes must "
                    "equal budget_bytes - used_bytes - reserved_bytes"
                )
            if self.peak_charged_bytes > self.budget_bytes:
                raise ValueError(
                    "ResidencyLocationSnapshot peak_charged_bytes cannot "
                    "exceed budget_bytes"
                )
        _require_optional_nonnegative_int(
            self.allocator_allocated_bytes,
            field_name=("ResidencyLocationSnapshot allocator_allocated_bytes"),
        )
        _require_optional_nonnegative_int(
            self.allocator_reserved_bytes,
            field_name="ResidencyLocationSnapshot allocator_reserved_bytes",
        )


@dataclass(frozen=True, slots=True)
class ResidencySnapshot(Generic[ValueT_co]):
    """Hierarchical tensor-free observation of one manager at one instant.

    ``state_version`` is the manager's monotonic mutation version at capture.
    ``captured_at_ns`` is a Unix timestamp in nanoseconds recorded while the
    manager holds the state lock used to construct both value and location
    observations.
    """

    manager_id: str
    state_version: int
    values: tuple[ResidencyValueSnapshot[ValueT_co], ...]
    locations: tuple[ResidencyLocationSnapshot, ...]
    reservations: tuple[ResidencyReservationSnapshot, ...]
    captured_at_ns: int

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.manager_id,
            field_name="ResidencySnapshot manager_id",
        )
        _require_nonnegative_int(
            self.state_version,
            field_name="ResidencySnapshot state_version",
        )
        values = tuple(self.values)
        locations = tuple(self.locations)
        reservations = tuple(self.reservations)
        if any(not isinstance(item, ResidencyValueSnapshot) for item in values):
            raise TypeError(
                "ResidencySnapshot values must contain "
                "ResidencyValueSnapshot objects"
            )
        if any(
            not isinstance(item, ResidencyLocationSnapshot)
            for item in locations
        ):
            raise TypeError(
                "ResidencySnapshot locations must contain "
                "ResidencyLocationSnapshot objects"
            )
        if len({item.handle for item in values}) != len(values):
            raise ValueError("ResidencySnapshot value handles must be unique")
        if len({item.location for item in locations}) != len(locations):
            raise ValueError(
                "ResidencySnapshot location identities must be unique"
            )
        if any(
            not isinstance(item, ResidencyReservationSnapshot)
            for item in reservations
        ):
            raise TypeError(
                "ResidencySnapshot reservations must contain "
                "ResidencyReservationSnapshot objects"
            )
        _require_nonnegative_int(
            self.captured_at_ns,
            field_name="ResidencySnapshot captured_at_ns",
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "reservations", reservations)


@dataclass(frozen=True, slots=True)
class ResidencyTransitionReport:
    """Measured result of one completed residency action.

    ``action`` is the exact requested action.  ``source`` records the resolved
    source when an action left it implicit, and ``destination`` records the
    target when applicable.  CUDA allocator metrics are optional because host
    transitions and managers without allocator instrumentation cannot provide
    them. ``allocator_device`` identifies the optional allocator sample; a
    cross-device move samples its destination. Timestamps are Unix nanoseconds.
    """

    action: ResidencyAction
    no_op: bool
    source: ResidencyLocation | None
    destination: ResidencyLocation | None
    logical_nbytes: int
    storage_nbytes: int
    started_at_ns: int
    completed_at_ns: int
    allocator_device: torch.device | None = None
    allocator_allocated_bytes_before: int | None = None
    allocator_reserved_bytes_before: int | None = None
    allocator_allocated_bytes_after: int | None = None
    allocator_reserved_bytes_after: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, _ACTION_TYPES):
            raise TypeError(
                "ResidencyTransitionReport action must be a residency action"
            )
        _require_bool(
            self.no_op,
            field_name="ResidencyTransitionReport no_op",
        )
        for field_name in (
            "logical_nbytes",
            "storage_nbytes",
            "started_at_ns",
            "completed_at_ns",
        ):
            _require_nonnegative_int(
                getattr(self, field_name),
                field_name=f"ResidencyTransitionReport {field_name}",
            )
        if self.allocator_device is not None:
            allocator_device = torch.device(self.allocator_device)
            if (
                allocator_device.type != "cuda"
                or allocator_device.index is None
            ):
                raise ValueError(
                    "ResidencyTransitionReport allocator_device must be an "
                    "indexed CUDA device"
                )
            object.__setattr__(self, "allocator_device", allocator_device)
        if self.storage_nbytes < self.logical_nbytes:
            raise ValueError(
                "ResidencyTransitionReport storage_nbytes must be greater "
                "than or equal to logical_nbytes"
            )
        if self.completed_at_ns < self.started_at_ns:
            raise ValueError(
                "ResidencyTransitionReport completed_at_ns cannot precede "
                "started_at_ns"
            )
        for field_name in (
            "allocator_allocated_bytes_before",
            "allocator_reserved_bytes_before",
            "allocator_allocated_bytes_after",
            "allocator_reserved_bytes_after",
        ):
            _require_optional_nonnegative_int(
                getattr(self, field_name),
                field_name=f"ResidencyTransitionReport {field_name}",
            )
        _require_optional_reason(
            self.reason,
            field_name="ResidencyTransitionReport reason",
        )


@dataclass(frozen=True, slots=True)
class ResidencyActionExplanation:
    """Dry-run explanation of how one requested action would be resolved."""

    action: ResidencyAction
    executable: bool
    no_op: bool
    source: ResidencyLocation | None
    destination: ResidencyLocation | None
    logical_nbytes: int
    storage_nbytes: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, _ACTION_TYPES):
            raise TypeError(
                "ResidencyActionExplanation action must be a residency action"
            )
        _require_bool(
            self.executable,
            field_name="ResidencyActionExplanation executable",
        )
        _require_bool(
            self.no_op,
            field_name="ResidencyActionExplanation no_op",
        )
        _require_nonnegative_int(
            self.logical_nbytes,
            field_name="ResidencyActionExplanation logical_nbytes",
        )
        _require_nonnegative_int(
            self.storage_nbytes,
            field_name="ResidencyActionExplanation storage_nbytes",
        )
        if self.storage_nbytes < self.logical_nbytes:
            raise ValueError(
                "ResidencyActionExplanation storage_nbytes must be greater "
                "than or equal to logical_nbytes"
            )
        _require_optional_reason(
            self.reason,
            field_name="ResidencyActionExplanation reason",
        )


@dataclass(frozen=True, slots=True)
class ResidencyPlanExplanation:
    """Immutable dry-run feasibility and predicted peak for one plan.

    ``predicted_peak_bytes`` starts with the manager's tracked locations, then
    appends locations first referenced by the plan in reservation/action order.
    It represents the predicted peak of managed backing storage, `MemoryReservation` plan
    charges, and temporary source loading without mutating manager state.
    It is not CUDA caching-allocator reservation.
    """

    plan_name: str
    actions: tuple[ResidencyActionExplanation, ...]
    reservations: tuple[MemoryReservation, ...]
    predicted_peak_bytes: tuple[tuple[ResidencyLocation, int], ...]
    feasible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.plan_name,
            field_name="ResidencyPlanExplanation plan_name",
        )
        actions = tuple(self.actions)
        reservations = tuple(self.reservations)
        predicted_peak_bytes = tuple(
            (location, nbytes) for location, nbytes in self.predicted_peak_bytes
        )
        if any(
            not isinstance(item, ResidencyActionExplanation) for item in actions
        ):
            raise TypeError(
                "ResidencyPlanExplanation actions must contain "
                "ResidencyActionExplanation objects"
            )
        if any(
            not isinstance(item, MemoryReservation) for item in reservations
        ):
            raise TypeError(
                "ResidencyPlanExplanation reservations must contain "
                "MemoryReservation objects"
            )
        for location, nbytes in predicted_peak_bytes:
            if not isinstance(location, ResidencyLocation):
                raise TypeError(
                    "ResidencyPlanExplanation predicted peak locations must "
                    "be ResidencyLocation objects"
                )
            _require_nonnegative_int(
                nbytes,
                field_name=(
                    "ResidencyPlanExplanation predicted peak byte count"
                ),
            )
        if len({item[0] for item in predicted_peak_bytes}) != len(
            predicted_peak_bytes
        ):
            raise ValueError(
                "ResidencyPlanExplanation predicted peak locations must be "
                "unique"
            )
        _require_bool(
            self.feasible,
            field_name="ResidencyPlanExplanation feasible",
        )
        _require_optional_reason(
            self.reason,
            field_name="ResidencyPlanExplanation reason",
        )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "reservations", reservations)
        object.__setattr__(
            self,
            "predicted_peak_bytes",
            predicted_peak_bytes,
        )


@dataclass(frozen=True, slots=True)
class ResidencyPlanReport:
    """Completed transition sequence for one plan operation.

    A successful ``execute_actions`` call or closed scope returns its complete
    report. A ``ResidencyPlanExecutionError`` carries a partial report containing
    only transitions committed before the runtime failure.
    """

    plan_name: str
    transitions: tuple[ResidencyTransitionReport, ...]
    started_at_ns: int
    completed_at_ns: int

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.plan_name,
            field_name="ResidencyPlanReport plan_name",
        )
        transitions = tuple(self.transitions)
        if any(
            not isinstance(item, ResidencyTransitionReport)
            for item in transitions
        ):
            raise TypeError(
                "ResidencyPlanReport transitions must contain "
                "ResidencyTransitionReport objects"
            )
        _require_nonnegative_int(
            self.started_at_ns,
            field_name="ResidencyPlanReport started_at_ns",
        )
        _require_nonnegative_int(
            self.completed_at_ns,
            field_name="ResidencyPlanReport completed_at_ns",
        )
        if self.completed_at_ns < self.started_at_ns:
            raise ValueError(
                "ResidencyPlanReport completed_at_ns cannot precede "
                "started_at_ns"
            )
        object.__setattr__(self, "transitions", transitions)


__all__ = [
    "MaterializationSnapshot",
    "ResidencyActionExplanation",
    "ResidencyLocationSnapshot",
    "ResidencyPlanExplanation",
    "ResidencyPlanReport",
    "ResidencyReservationSnapshot",
    "ResidencySnapshot",
    "ResidencyTransitionReport",
    "ResidencyValueSnapshot",
]
