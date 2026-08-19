"""Inspectable low-level plans for residency transitions.

A residency plan is an immutable intermediate representation (IR).  It records
ordered reclaim, entry, and exit transition requests plus temporary accounting
reservations. A manager resolves any omitted move source against current state
and reports each completed transition separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from fhelium.core import TensorResident
from fhelium.residency.location import ResidencyLocation
from fhelium.residency.model import ResidencyHandle

ValueT_co = TypeVar("ValueT_co", bound=TensorResident, covariant=True)


@dataclass(frozen=True, slots=True)
class EnsureResident(Generic[ValueT_co]):
    """Require a managed value to be resident at ``location``."""

    handle: ResidencyHandle[ValueT_co]
    location: ResidencyLocation

    def __post_init__(self) -> None:
        _validate_action_endpoint(self.handle, self.location)


@dataclass(frozen=True, slots=True)
class MoveResident(Generic[ValueT_co]):
    """Move a managed value to ``to`` from an optional explicit source.

    ``from_location=None`` delegates source resolution to the manager while
    preserving move semantics: successful completion does not retain the
    selected source materialization.
    """

    handle: ResidencyHandle[ValueT_co]
    to: ResidencyLocation
    from_location: ResidencyLocation | None = None

    def __post_init__(self) -> None:
        _validate_action_endpoint(self.handle, self.to)
        if self.from_location is not None and not isinstance(
            self.from_location, ResidencyLocation
        ):
            raise TypeError(
                "MoveResident from_location must be a ResidencyLocation"
            )


@dataclass(frozen=True, slots=True)
class DropResident(Generic[ValueT_co]):
    """Remove one materialization without ending the managed value."""

    handle: ResidencyHandle[ValueT_co]
    location: ResidencyLocation

    def __post_init__(self) -> None:
        _validate_action_endpoint(self.handle, self.location)


@dataclass(frozen=True, slots=True)
class DiscardValue(Generic[ValueT_co]):
    """End a managed value and remove all of its materializations."""

    handle: ResidencyHandle[ValueT_co]

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ResidencyHandle):
            raise TypeError("DiscardValue handle must be a ResidencyHandle")


ResidencyAction: TypeAlias = (
    EnsureResident[TensorResident]
    | MoveResident[TensorResident]
    | DropResident[TensorResident]
    | DiscardValue[TensorResident]
)

_ACTION_TYPES = (EnsureResident, MoveResident, DropResident, DiscardValue)


def _validate_action_endpoint(
    handle: object,
    location: object,
) -> None:
    if not isinstance(handle, ResidencyHandle):
        raise TypeError("Residency action handle must be a ResidencyHandle")
    if not isinstance(location, ResidencyLocation):
        raise TypeError("Residency action location must be a ResidencyLocation")


def _require_nonnegative_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class MemoryReservation:
    """Named accounting headroom held for the lifetime of a plan scope.

    Reservations account for future storage but do not allocate tensors or
    imply any value placement.
    """

    location: ResidencyLocation
    nbytes: int
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.location, ResidencyLocation):
            raise TypeError(
                "MemoryReservation location must be a ResidencyLocation"
            )
        _require_nonnegative_int(
            self.nbytes,
            field_name="MemoryReservation nbytes",
        )
        _require_nonempty_string(
            self.label,
            field_name="MemoryReservation label",
        )


@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    """Immutable ordered residency plan executed at scope entry and exit.

    ``reclaim`` actions execute first and may free capacity needed by the
    scope. Reservations are admitted only after reclaim completes, then
    ``enter`` actions execute in tuple order. ``exit`` actions execute in tuple
    order when the plan scope closes. The tuples are normalized at
    construction so callers cannot mutate a plan by retaining an input list.
    Each action remains a manager request.

    Args:
        name: Non-empty diagnostic plan identity.
        reclaim: Ordered actions that establish capacity before reservations.
        enter: Ordered actions for plan-scope entry.
        exit: Ordered actions for plan-scope exit.
        reservations: Capacity reservations held across the plan scope.
    """

    name: str
    enter: tuple[ResidencyAction, ...] = ()
    exit: tuple[ResidencyAction, ...] = ()
    reservations: tuple[MemoryReservation, ...] = ()
    reclaim: tuple[ResidencyAction, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, field_name="ResidencyPlan name")
        reclaim = tuple(self.reclaim)
        enter = tuple(self.enter)
        exit_actions = tuple(self.exit)
        reservations = tuple(self.reservations)
        for action in (*reclaim, *enter, *exit_actions):
            if not isinstance(action, _ACTION_TYPES):
                raise TypeError(
                    "ResidencyPlan actions must be residency action objects"
                )
        for reservation in reservations:
            if not isinstance(reservation, MemoryReservation):
                raise TypeError(
                    "ResidencyPlan reservations must be MemoryReservation "
                    "objects"
                )
        object.__setattr__(self, "reclaim", reclaim)
        object.__setattr__(self, "enter", enter)
        object.__setattr__(self, "exit", exit_actions)
        object.__setattr__(self, "reservations", reservations)


__all__ = [
    "DiscardValue",
    "DropResident",
    "EnsureResident",
    "MemoryReservation",
    "MoveResident",
    "ResidencyAction",
    "ResidencyPlan",
]
