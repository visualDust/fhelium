"""Declarative residency working-set requirements.

Requests describe which manager-owned logical values must be resident at which
local locations and how much named accounting headroom the operation requires.
They contain no placement path, eviction choice, stream, or concrete tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fhelium.residency.location import ResidencyLocation
from fhelium.residency.model import ResidencyHandle
from fhelium.residency.plan import MemoryReservation


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class ResidencyRequirement:
    """Require one managed logical value at one local location.

    The requirement states a post-admission condition. It deliberately does
    not select a source materialization, transfer path, eviction victim, or
    execution stream; those choices belong to a controller decision.
    """

    handle: ResidencyHandle[Any]
    location: ResidencyLocation

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ResidencyHandle):
            raise TypeError(
                "ResidencyRequirement handle must be a ResidencyHandle"
            )
        if not isinstance(self.location, ResidencyLocation):
            raise TypeError(
                "ResidencyRequirement location must be a ResidencyLocation"
            )


@dataclass(frozen=True, slots=True)
class ResidencyRequest:
    """Immutable declarative working-set and headroom request.

    Each ``(handle, location)`` pair appears exactly once. A replicable value
    may therefore be required at several locations in one request. Whether a
    handle's replica mode permits those simultaneous postconditions is
    validated against manager state during controller decision-making.

    Reservations are accounting headroom held while the admitted request is
    active. They neither allocate tensors nor name eviction victims.
    """

    name: str
    requirements: tuple[ResidencyRequirement, ...]
    reservations: tuple[MemoryReservation, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, field_name="ResidencyRequest name")
        requirements = tuple(self.requirements)
        reservations = tuple(self.reservations)
        if not requirements:
            raise ValueError(
                "ResidencyRequest requires at least one ResidencyRequirement"
            )
        if any(
            not isinstance(requirement, ResidencyRequirement)
            for requirement in requirements
        ):
            raise TypeError(
                "ResidencyRequest requirements must contain "
                "ResidencyRequirement objects"
            )
        endpoints = tuple(
            (requirement.handle, requirement.location)
            for requirement in requirements
        )
        if len(set(endpoints)) != len(endpoints):
            raise ValueError(
                "ResidencyRequest requirements must use unique "
                "(handle, location) pairs"
            )
        if any(
            not isinstance(reservation, MemoryReservation)
            for reservation in reservations
        ):
            raise TypeError(
                "ResidencyRequest reservations must contain "
                "MemoryReservation objects"
            )
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "reservations", reservations)


__all__ = ["ResidencyRequest", "ResidencyRequirement"]
