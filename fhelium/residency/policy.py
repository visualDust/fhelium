"""Pure deterministic policy inputs and built-in eviction ordering.

Policies receive only frozen tensor-free candidates that have already passed
manager-invariant filtering. They rank candidates and expose configured
fallback tiers; they never inspect concrete values or execute
residency actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fhelium.residency.location import ResidencyLocation
from fhelium.residency.model import ResidencyHandle


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class ResidencyPolicyMetadata:
    """Controller-owned eviction hints for one manager handle.

    Higher ``priority`` values retain a value before lower-priority values.
    ``stable_key`` is an optional application-stable tie breaker and diagnostic
    identity. Neither field changes logical value identity or manager
    ownership.
    """

    priority: int = 0
    stable_key: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.priority, bool) or not isinstance(
            self.priority, int
        ):
            raise TypeError(
                "ResidencyPolicyMetadata priority must be an integer"
            )
        if self.stable_key is not None:
            _require_nonempty_string(
                self.stable_key,
                field_name="ResidencyPolicyMetadata stable_key",
            )


@dataclass(frozen=True, slots=True)
class ResidencyEvictionCandidate:
    """One invariant-filtered materialization eligible for policy ranking."""

    handle: ResidencyHandle[Any]
    location: ResidencyLocation
    charged_nbytes: int
    registration_index: int
    last_access_epoch: int | None
    metadata: ResidencyPolicyMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ResidencyHandle):
            raise TypeError(
                "ResidencyEvictionCandidate handle must be a ResidencyHandle"
            )
        if not isinstance(self.location, ResidencyLocation):
            raise TypeError(
                "ResidencyEvictionCandidate location must be a "
                "ResidencyLocation"
            )
        for field_name in ("charged_nbytes", "registration_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"ResidencyEvictionCandidate {field_name} must be an integer"
                )
            if value < 0:
                raise ValueError(
                    f"ResidencyEvictionCandidate {field_name} must be "
                    "non-negative"
                )
        if self.last_access_epoch is not None:
            if isinstance(self.last_access_epoch, bool) or not isinstance(
                self.last_access_epoch, int
            ):
                raise TypeError(
                    "ResidencyEvictionCandidate last_access_epoch must be an "
                    "integer or None"
                )
            if self.last_access_epoch < 0:
                raise ValueError(
                    "ResidencyEvictionCandidate last_access_epoch must be "
                    "non-negative"
                )
        if not isinstance(self.metadata, ResidencyPolicyMetadata):
            raise TypeError(
                "ResidencyEvictionCandidate metadata must be "
                "ResidencyPolicyMetadata"
            )


@runtime_checkable
class ResidencyPolicy(Protocol):
    """Pure deterministic policy interface used by a residency controller."""

    @property
    def name(self) -> str:
        """Stable diagnostic policy name."""

        ...

    @property
    def config_identity(self) -> tuple[tuple[str, object], ...]:
        """Frozen tensor-free configuration evidence for decisions."""

        ...

    def fallback_locations(
        self,
        location: ResidencyLocation,
    ) -> tuple[ResidencyLocation, ...]:
        """Return configured spill tiers for ``location``."""

        ...

    def order_candidates(
        self,
        candidates: Sequence[ResidencyEvictionCandidate],
    ) -> tuple[ResidencyEvictionCandidate, ...]:
        """Return every supplied candidate in deterministic eviction order."""

        ...


class DeterministicTieredLRU:
    """Deterministic priority-aware LRU ordering over configured tier edges.

    Fallback locations are never inferred. An omitted source location has no
    spill path. Candidate order is lower priority, then older access epoch,
    then application stable key or manager registration order. Wall-clock time,
    UUID text, allocator free-memory readings, and random state are not inputs.
    """

    def __init__(
        self,
        fallback_tiers: Mapping[
            ResidencyLocation,
            Sequence[ResidencyLocation],
        ]
        | None = None,
    ) -> None:
        if fallback_tiers is not None and not isinstance(
            fallback_tiers, Mapping
        ):
            raise TypeError(
                "DeterministicTieredLRU fallback_tiers must be a mapping"
            )
        normalized: list[
            tuple[ResidencyLocation, tuple[ResidencyLocation, ...]]
        ] = []
        for source, destinations_input in (
            {} if fallback_tiers is None else fallback_tiers
        ).items():
            if not isinstance(source, ResidencyLocation):
                raise TypeError(
                    "DeterministicTieredLRU fallback source must be a "
                    "ResidencyLocation"
                )
            destinations = tuple(destinations_input)
            if any(
                not isinstance(destination, ResidencyLocation)
                for destination in destinations
            ):
                raise TypeError(
                    "DeterministicTieredLRU fallback destinations must be "
                    "ResidencyLocation objects"
                )
            if source in destinations:
                raise ValueError(
                    "DeterministicTieredLRU fallback tiers cannot contain "
                    "their source location"
                )
            if len(set(destinations)) != len(destinations):
                raise ValueError(
                    "DeterministicTieredLRU fallback destinations must be unique"
                )
            normalized.append((source, destinations))
        self._fallback_tiers = tuple(normalized)

    @property
    def name(self) -> str:
        """Stable diagnostic policy name."""

        return "deterministic-tiered-lru"

    @property
    def config_identity(self) -> tuple[tuple[str, object], ...]:
        """Frozen fallback-tier configuration recorded in decisions."""

        return (
            (
                "fallback_tiers",
                tuple(
                    (
                        source.name,
                        tuple(destination.name for destination in destinations),
                    )
                    for source, destinations in self._fallback_tiers
                ),
            ),
        )

    def fallback_locations(
        self,
        location: ResidencyLocation,
    ) -> tuple[ResidencyLocation, ...]:
        """Return configured destinations in caller-specified order."""

        if not isinstance(location, ResidencyLocation):
            raise TypeError(
                "Policy fallback location must be a ResidencyLocation"
            )
        for source, destinations in self._fallback_tiers:
            if source == location:
                return destinations
        return ()

    def order_candidates(
        self,
        candidates: Sequence[ResidencyEvictionCandidate],
    ) -> tuple[ResidencyEvictionCandidate, ...]:
        """Rank candidates without observing or mutating manager state."""

        normalized = tuple(candidates)
        if any(
            not isinstance(candidate, ResidencyEvictionCandidate)
            for candidate in normalized
        ):
            raise TypeError(
                "ResidencyPolicy candidates must be "
                "ResidencyEvictionCandidate objects"
            )
        return tuple(sorted(normalized, key=_candidate_order_key))


def _candidate_order_key(
    candidate: ResidencyEvictionCandidate,
) -> tuple[int, int, int, str, int]:
    epoch = (
        -1
        if candidate.last_access_epoch is None
        else candidate.last_access_epoch
    )
    stable_key = candidate.metadata.stable_key
    return (
        candidate.metadata.priority,
        epoch,
        0 if stable_key is not None else 1,
        "" if stable_key is None else stable_key,
        candidate.registration_index,
    )


__all__ = [
    "DeterministicTieredLRU",
    "ResidencyEvictionCandidate",
    "ResidencyPolicy",
    "ResidencyPolicyMetadata",
]
