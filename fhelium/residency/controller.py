"""Inspectable deterministic automation over explicit residency mechanisms.

A controller owns no tensor materializations. It decides state-bound placement
from declarative requests and delegates all validation, copying, accounting,
and lifetime enforcement to one :class:`ResidencyManager`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from threading import RLock
from types import TracebackType
from typing import Any

import torch

from fhelium.core import TensorResident
from fhelium.errors import (
    ResidencyHandleError,
    ResidencyOwnershipError,
    ResidencyPlanError,
    ResidencySearchLimitError,
    ResidencyStaleStateError,
    ResidencyUnavailableError,
)
from fhelium.residency.lease import BorrowedValues
from fhelium.residency.location import (
    PAGEABLE_HOST,
    PINNED_HOST,
    ResidencyLocation,
)
from fhelium.residency.manager import (
    ResidencyManager,
    ResidencyScope,
)
from fhelium.residency.model import (
    ReplicaMode,
    ResidencyHandle,
)
from fhelium.residency.plan import (
    DropResident,
    EnsureResident,
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
    ResidencyPlanExplanation,
    ResidencyPlanReport,
    ResidencySnapshot,
    ResidencyValueSnapshot,
)

TransferStreams = Mapping[ResidencyLocation, torch.cuda.Stream]
ConsumerStreams = Mapping[ResidencyLocation, torch.cuda.Stream]


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class ResidencyEviction:
    """One controller-selected reclaim action and its policy evidence."""

    action: DropResident[TensorResident] | MoveResident[TensorResident]
    rank: int
    released_location: ResidencyLocation
    released_nbytes: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, (DropResident, MoveResident)):
            raise TypeError(
                "ResidencyEviction action must be DropResident or MoveResident"
            )
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("ResidencyEviction rank must be an integer")
        if self.rank < 0:
            raise ValueError("ResidencyEviction rank must be non-negative")
        if not isinstance(self.released_location, ResidencyLocation):
            raise TypeError(
                "ResidencyEviction released_location must be a "
                "ResidencyLocation"
            )
        if isinstance(self.released_nbytes, bool) or not isinstance(
            self.released_nbytes, int
        ):
            raise TypeError(
                "ResidencyEviction released_nbytes must be an integer"
            )
        if self.released_nbytes < 0:
            raise ValueError(
                "ResidencyEviction released_nbytes must be non-negative"
            )
        _require_nonempty_string(
            self.reason,
            field_name="ResidencyEviction reason",
        )


@dataclass(frozen=True, slots=True)
class ResidencyDecision:
    """Immutable state-bound result of automatic residency decision-making.

    A decision is valid only for the issuing manager at
    ``expected_state_version``. It is process-local evidence, not a serialized
    or distributed execution artifact. Entering the associated manager scope
    atomically rechecks the version before any reclaim action or reservation.
    """

    manager_id: str
    expected_state_version: int
    explored_states: int
    request: ResidencyRequest
    plan: ResidencyPlan
    policy_name: str
    policy_config: tuple[tuple[str, object], ...]
    evictions: tuple[ResidencyEviction, ...]
    explanation: ResidencyPlanExplanation

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.manager_id,
            field_name="ResidencyDecision manager_id",
        )
        if isinstance(self.expected_state_version, bool) or not isinstance(
            self.expected_state_version, int
        ):
            raise TypeError(
                "ResidencyDecision expected_state_version must be an integer"
            )
        if self.expected_state_version < 0:
            raise ValueError(
                "ResidencyDecision expected_state_version must be non-negative"
            )
        if isinstance(self.explored_states, bool) or not isinstance(
            self.explored_states, int
        ):
            raise TypeError(
                "ResidencyDecision explored_states must be an integer"
            )
        if self.explored_states <= 0:
            raise ValueError(
                "ResidencyDecision explored_states must be positive"
            )
        if not isinstance(self.request, ResidencyRequest):
            raise TypeError(
                "ResidencyDecision request must be a ResidencyRequest"
            )
        if not isinstance(self.plan, ResidencyPlan):
            raise TypeError("ResidencyDecision plan must be a ResidencyPlan")
        _require_nonempty_string(
            self.policy_name,
            field_name="ResidencyDecision policy_name",
        )
        policy_config = tuple(self.policy_config)
        for item in policy_config:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "ResidencyDecision policy_config must contain key/value "
                    "pairs"
                )
            _require_nonempty_string(
                item[0],
                field_name="ResidencyDecision policy config key",
            )
        evictions = tuple(self.evictions)
        if any(not isinstance(item, ResidencyEviction) for item in evictions):
            raise TypeError(
                "ResidencyDecision evictions must contain ResidencyEviction "
                "objects"
            )
        if not isinstance(self.explanation, ResidencyPlanExplanation):
            raise TypeError(
                "ResidencyDecision explanation must be a "
                "ResidencyPlanExplanation"
            )
        if self.plan.name != self.request.name:
            raise ValueError(
                "ResidencyDecision request and plan names must agree"
            )
        if self.explanation.plan_name != self.plan.name:
            raise ValueError(
                "ResidencyDecision explanation and plan names must agree"
            )
        if not self.explanation.feasible:
            raise ValueError(
                "ResidencyDecision explanation must describe a feasible plan"
            )
        object.__setattr__(self, "policy_config", policy_config)
        object.__setattr__(self, "evictions", evictions)


@dataclass
class _PlanningValue:
    snapshot: ResidencyValueSnapshot[Any]
    materializations: dict[ResidencyLocation, MaterializationSnapshot]


@dataclass
class _PlanningState:
    values: dict[ResidencyHandle[Any], _PlanningValue]
    used: dict[ResidencyLocation, int]
    reserved: dict[ResidencyLocation, int]
    budgets: dict[ResidencyLocation, int | None]
    registration_order: dict[ResidencyHandle[Any], int]
    reclaim: list[ResidencyAction]
    evictions: list[ResidencyEviction]

    def clone(self) -> _PlanningState:
        return _PlanningState(
            values={
                handle: _PlanningValue(
                    item.snapshot,
                    dict(item.materializations),
                )
                for handle, item in self.values.items()
            },
            used=dict(self.used),
            reserved=dict(self.reserved),
            budgets=dict(self.budgets),
            registration_order=dict(self.registration_order),
            reclaim=list(self.reclaim),
            evictions=list(self.evictions),
        )

    def restore(self, other: _PlanningState) -> None:
        self.values = other.values
        self.used = other.used
        self.reserved = other.reserved
        self.budgets = other.budgets
        self.registration_order = other.registration_order
        self.reclaim = other.reclaim
        self.evictions = other.evictions


class _PlanningFailure(Exception):
    pass


class _SearchLimitFailure(_PlanningFailure):
    pass


class _CapacityFailure(_PlanningFailure):
    def __init__(
        self,
        *,
        location: ResidencyLocation,
        additional: int,
        stage: str,
    ) -> None:
        self.location = location
        self.additional = additional
        self.stage = stage
        super().__init__(
            f"location {location} cannot admit {additional} bytes during "
            f"{stage}"
        )


@dataclass
class _SearchMemo:
    limit: int
    explored_states: int = 0
    failed_requests: set[tuple[object, ...]] = field(default_factory=set)
    failed_capacities: set[tuple[object, ...]] = field(default_factory=set)

    def count(self, *, objective: str) -> None:
        if self.explored_states >= self.limit:
            raise _SearchLimitFailure(
                "automatic residency search exceeded its deterministic state "
                f"limit during {objective}: limit={self.limit}"
            )
        self.explored_states += 1


class ResidencyController:
    """Deterministic optional automation bound to one residency manager.

    The manager remains the only materialization owner and transition executor.
    This controller stores only policy metadata and logical access epochs. It
    decides only when called, emits an inspectable decision, and never
    runs background eviction, waits for protected victims, retries a failed
    source, rolls back completed actions, or silently replans a stale decision.
    """

    def __init__(
        self,
        manager: ResidencyManager,
        *,
        policy: ResidencyPolicy | None = None,
        search_state_limit: int = 100_000,
    ) -> None:
        if not isinstance(manager, ResidencyManager):
            raise TypeError(
                "ResidencyController manager must be a ResidencyManager"
            )
        selected_policy: ResidencyPolicy = (
            DeterministicTieredLRU() if policy is None else policy
        )
        if not isinstance(selected_policy, ResidencyPolicy):
            raise TypeError(
                "ResidencyController policy must implement ResidencyPolicy"
            )
        _require_nonempty_string(
            selected_policy.name,
            field_name="ResidencyPolicy name",
        )
        if isinstance(search_state_limit, bool) or not isinstance(
            search_state_limit, int
        ):
            raise TypeError(
                "ResidencyController search_state_limit must be an integer"
            )
        if search_state_limit <= 0:
            raise ValueError(
                "ResidencyController search_state_limit must be positive"
            )
        self._manager = manager
        self._policy = selected_policy
        self._search_state_limit = search_state_limit
        self._metadata: dict[ResidencyHandle[Any], ResidencyPolicyMetadata] = {}
        self._last_access: dict[ResidencyHandle[Any], int] = {}
        self._access_epoch = 0
        self._lock = RLock()

    @property
    def manager(self) -> ResidencyManager:
        """The sole state and materialization authority used by this controller."""

        return self._manager

    @property
    def policy(self) -> ResidencyPolicy:
        """Pure policy used to rank invariant-filtered candidates."""

        return self._policy

    def set_policy_metadata(
        self,
        handle: ResidencyHandle[Any],
        *,
        priority: int = 0,
        stable_key: str | None = None,
    ) -> None:
        """Set controller-local eviction metadata for one known handle."""

        metadata = ResidencyPolicyMetadata(
            priority=priority,
            stable_key=stable_key,
        )
        with self._lock:
            snapshot = self._manager.snapshot()
            known = {item.handle: item for item in snapshot.values}
            item = known.get(handle)
            if (
                item is None
                or item.discarded
                or handle.manager_id != snapshot.manager_id
            ):
                raise ResidencyHandleError(
                    "Policy metadata requires a live handle from this manager"
                )
            if stable_key is not None:
                for existing_handle, existing in self._metadata.items():
                    if (
                        existing_handle != handle
                        and existing.stable_key == stable_key
                    ):
                        raise ValueError(
                            "Residency policy stable_key values must be unique "
                            "within one controller"
                        )
            self._metadata[handle] = metadata

    def decide(self, request: ResidencyRequest) -> ResidencyDecision:
        """Derive an inspectable decision without executing residency actions."""

        if not isinstance(request, ResidencyRequest):
            raise TypeError(
                "ResidencyController decide request must be a ResidencyRequest"
            )
        with self._lock:
            snapshot = self._manager.snapshot()
            state_version = snapshot.state_version
            self._validate_request(snapshot, request)
            planning = self._planning_state(snapshot)
            memo = _SearchMemo(self._search_state_limit)
            try:
                planning, enter = self._search_request_plan(
                    planning,
                    request,
                    required_endpoints={
                        (requirement.handle, requirement.location)
                        for requirement in request.requirements
                    },
                    memo=memo,
                )
            except _SearchLimitFailure as error:
                raise ResidencySearchLimitError(
                    request_name=request.name,
                    state_limit=self._search_state_limit,
                    explored_states=memo.explored_states,
                    detail=str(error),
                ) from None
            except _PlanningFailure as error:
                raise ResidencyPlanError(
                    f"Automatic residency request {request.name!r} is "
                    f"infeasible: {error}"
                ) from None
            plan = ResidencyPlan(
                request.name,
                reclaim=tuple(planning.reclaim),
                enter=enter,
                reservations=request.reservations,
            )
            explanation = self._manager.explain(
                plan,
                expected_state_version=state_version,
            )
            actual_version = self._manager.state_version
            if actual_version != state_version:
                raise ResidencyStaleStateError(
                    expected_version=state_version,
                    actual_version=actual_version,
                )
            if not explanation.feasible:
                raise ResidencyPlanError(
                    explanation.reason
                    or f"Automatic residency request {request.name!r} is infeasible"
                )
            return ResidencyDecision(
                manager_id=snapshot.manager_id,
                expected_state_version=state_version,
                explored_states=memo.explored_states,
                request=request,
                plan=plan,
                policy_name=self._policy.name,
                policy_config=tuple(self._policy.config_identity),
                evictions=tuple(planning.evictions),
                explanation=explanation,
            )

    def scope(
        self,
        decision: ResidencyDecision,
        *,
        transfer_streams: TransferStreams | None = None,
    ) -> ResidencyScope:
        """Return a scope that version-checks and commits ``decision`` on entry."""

        self._validate_decision(decision)
        return self._manager.scope(
            decision.plan,
            expected_state_version=decision.expected_state_version,
            transfer_streams=transfer_streams,
        )

    def use(
        self,
        request: ResidencyRequest,
        *,
        consumer_streams: ConsumerStreams | None = None,
        transfer_streams: TransferStreams | None = None,
    ) -> ResidencyUse:
        """Return a context that derives, admits, and borrows one decision.

        Values remain cached after the use scope. Later admission may
        reclaim them under policy. No exit eviction, hidden synchronization, or
        background work is performed.
        """

        return ResidencyUse(
            controller=self,
            request=request,
            consumer_streams=consumer_streams,
            transfer_streams=transfer_streams,
        )

    def _validate_decision(self, decision: ResidencyDecision) -> None:
        if not isinstance(decision, ResidencyDecision):
            raise TypeError(
                "ResidencyController decision must be a ResidencyDecision"
            )
        if decision.manager_id != self._manager.manager_id:
            raise ResidencyHandleError(
                "ResidencyDecision belongs to another manager"
            )
        if decision.policy_name != self._policy.name or (
            decision.policy_config != tuple(self._policy.config_identity)
        ):
            raise ResidencyPlanError(
                "ResidencyDecision was issued under another policy "
                "configuration"
            )

    @staticmethod
    def _validate_request(
        snapshot: ResidencySnapshot[Any],
        request: ResidencyRequest,
    ) -> None:
        values = {item.handle: item for item in snapshot.values}
        locations_by_handle: dict[
            ResidencyHandle[Any], set[ResidencyLocation]
        ] = {}
        for requirement in request.requirements:
            if requirement.handle.manager_id != snapshot.manager_id:
                raise ResidencyHandleError(
                    "ResidencyRequest contains a handle from another manager"
                )
            value = values.get(requirement.handle)
            if value is None:
                raise ResidencyHandleError(
                    "ResidencyRequest contains an unknown handle"
                )
            if value.discarded:
                raise ResidencyHandleError(
                    "ResidencyRequest contains a discarded handle"
                )
            locations_by_handle.setdefault(requirement.handle, set()).add(
                requirement.location
            )
        for handle, locations in locations_by_handle.items():
            if (
                len(locations) > 1
                and values[handle].spec.replica_mode is ReplicaMode.EXCLUSIVE
            ):
                raise ResidencyOwnershipError(
                    "An EXCLUSIVE residency value cannot satisfy simultaneous "
                    "requirements at multiple locations"
                )

    @staticmethod
    def _planning_state(
        snapshot: ResidencySnapshot[Any],
    ) -> _PlanningState:
        return _PlanningState(
            values={
                item.handle: _PlanningValue(
                    item,
                    {
                        materialization.location: materialization
                        for materialization in item.materializations
                    },
                )
                for item in snapshot.values
                if not item.discarded
            },
            used={
                item.location: item.used_bytes for item in snapshot.locations
            },
            reserved={
                item.location: item.reserved_bytes
                for item in snapshot.locations
            },
            budgets={
                item.location: item.budget_bytes for item in snapshot.locations
            },
            registration_order={
                item.handle: position
                for position, item in enumerate(snapshot.values)
            },
            reclaim=[],
            evictions=[],
        )

    def _search_request_plan(
        self,
        planning: _PlanningState,
        request: ResidencyRequest,
        *,
        required_endpoints: set[tuple[ResidencyHandle[Any], ResidencyLocation]],
        memo: _SearchMemo,
    ) -> tuple[_PlanningState, tuple[ResidencyAction, ...]]:
        signature = self._planning_signature(planning)
        key = ("request", *signature)
        if key in memo.failed_requests:
            raise _PlanningFailure(
                "equivalent ordered-request state is already exhausted"
            )
        memo.count(objective="ordered request")
        try:
            enter = self._simulate_request(planning.clone(), request)
            return planning, enter
        except _CapacityFailure as capacity:
            candidates = self._ordered_candidates(
                self._candidates(
                    planning,
                    capacity.location,
                    blocked_handles=frozenset(),
                ),
                required_endpoints=required_endpoints,
            )
            failures: list[str] = []
            for rank, candidate in enumerate(candidates):
                produced = False
                for trial in self._reclaim_states(
                    planning,
                    candidate,
                    rank=rank,
                    required_endpoints=required_endpoints,
                    blocked_handles=frozenset(),
                    active_pairs=frozenset(),
                    memo=memo,
                ):
                    produced = True
                    try:
                        return self._search_request_plan(
                            trial,
                            request,
                            required_endpoints=required_endpoints,
                            memo=memo,
                        )
                    except _SearchLimitFailure:
                        raise
                    except _PlanningFailure as error:
                        failures.append(
                            f"candidate {candidate.registration_index}: {error}"
                        )
                if not produced:
                    failures.append(
                        f"candidate {candidate.registration_index}: no legal "
                        "drop or fallback transition"
                    )
            current = planning.used.get(capacity.location, 0)
            reserved = planning.reserved.get(capacity.location, 0)
            budget = planning.budgets.get(capacity.location)
            detail = "; ".join(failures)
            memo.failed_requests.add(key)
            raise _PlanningFailure(
                f"{capacity}; budget={budget}, used={current}, "
                f"reserved={reserved}, but no invariant-valid victim sequence "
                "satisfies the complete ordered request"
                + (f" ({detail})" if detail else "")
            ) from None

    def _simulate_request(
        self,
        planning: _PlanningState,
        request: ResidencyRequest,
    ) -> tuple[ResidencyAction, ...]:
        for reservation in request.reservations:
            self._require_capacity(
                planning,
                reservation.location,
                reservation.nbytes,
                stage=f"reservation {reservation.label!r}",
            )
            planning.reserved[reservation.location] = (
                planning.reserved.get(reservation.location, 0)
                + reservation.nbytes
            )

        actions: list[ResidencyAction] = []
        for requirement in request.requirements:
            value = planning.values[requirement.handle]
            destination = requirement.location
            if destination in value.materializations:
                continue
            spec = value.snapshot.spec
            charge = spec.storage_nbytes
            source_locations = self._ordered_locations(value.materializations)
            source = source_locations[0] if source_locations else None
            temporary_source = source is None
            if temporary_source:
                source = value.snapshot.source_location
                if source is None:
                    raise ResidencyUnavailableError(
                        "Required residency value has no materialization or "
                        "reconstruction source"
                    )
                self._require_capacity(
                    planning,
                    source,
                    charge,
                    stage=(
                        f"source reconstruction for "
                        f"{requirement.handle.handle_id!r}"
                    ),
                )
                planning.reserved[source] = (
                    planning.reserved.get(source, 0) + charge
                )
            assert source is not None

            if temporary_source and source == destination:
                # Loading and installation reuse one charge at this location.
                planning.reserved[source] -= charge
            self._require_capacity(
                planning,
                destination,
                charge,
                stage=(f"materialization of {requirement.handle.handle_id!r}"),
            )
            planning.used[destination] = (
                planning.used.get(destination, 0) + charge
            )
            source_materialization = (
                None if temporary_source else value.materializations[source]
            )
            value.materializations[destination] = self._project_materialization(
                value.snapshot,
                destination,
                source_materialization,
            )
            if temporary_source and source != destination:
                planning.reserved[source] -= charge

            if spec.replica_mode is ReplicaMode.EXCLUSIVE:
                actions.append(
                    MoveResident(
                        requirement.handle,
                        destination,
                        from_location=(None if temporary_source else source),
                    )
                )
                if not temporary_source:
                    del value.materializations[source]
                    planning.used[source] = (
                        planning.used.get(source, 0) - charge
                    )
            else:
                actions.append(EnsureResident(requirement.handle, destination))
        return tuple(actions)

    @staticmethod
    def _require_capacity(
        planning: _PlanningState,
        location: ResidencyLocation,
        additional: int,
        *,
        stage: str,
    ) -> None:
        budget = planning.budgets.get(location)
        if budget is None:
            return
        if (
            planning.used.get(location, 0)
            + planning.reserved.get(location, 0)
            + additional
            > budget
        ):
            raise _CapacityFailure(
                location=location,
                additional=additional,
                stage=stage,
            )

    def _capacity_states(
        self,
        planning: _PlanningState,
        location: ResidencyLocation,
        additional: int,
        *,
        required_endpoints: set[tuple[ResidencyHandle[Any], ResidencyLocation]],
        blocked_handles: frozenset[ResidencyHandle[Any]],
        active_pairs: frozenset[tuple[ResidencyHandle[Any], ResidencyLocation]],
        memo: _SearchMemo,
    ) -> Iterator[_PlanningState]:
        budget = planning.budgets.get(location)
        if budget is None or (
            planning.used.get(location, 0)
            + planning.reserved.get(location, 0)
            + additional
            <= budget
        ):
            yield planning
            return
        signature = self._planning_signature(planning)
        blocked_key = tuple(
            sorted(
                planning.registration_order[handle]
                for handle in blocked_handles
            )
        )
        active_key = tuple(
            sorted(
                (
                    planning.registration_order[handle],
                    source.name,
                )
                for handle, source in active_pairs
            )
        )
        key = (
            "capacity",
            *signature,
            location.name,
            additional,
            blocked_key,
            active_key,
        )
        if key in memo.failed_capacities:
            return
        memo.count(objective=f"capacity at {location}")
        candidates = self._ordered_candidates(
            self._candidates(
                planning,
                location,
                blocked_handles=blocked_handles,
            ),
            required_endpoints=required_endpoints,
        )
        produced = False
        for rank, candidate in enumerate(candidates):
            pair = (candidate.handle, candidate.location)
            if pair in active_pairs:
                continue
            for reclaimed in self._reclaim_states(
                planning,
                candidate,
                rank=rank,
                required_endpoints=required_endpoints,
                blocked_handles=blocked_handles,
                active_pairs=active_pairs,
                memo=memo,
            ):
                for result in self._capacity_states(
                    reclaimed,
                    location,
                    additional,
                    required_endpoints=required_endpoints,
                    blocked_handles=blocked_handles,
                    active_pairs=active_pairs,
                    memo=memo,
                ):
                    produced = True
                    yield result
        if not produced:
            memo.failed_capacities.add(key)

    def _candidates(
        self,
        planning: _PlanningState,
        location: ResidencyLocation,
        *,
        blocked_handles: frozenset[ResidencyHandle[Any]],
    ) -> tuple[ResidencyEvictionCandidate, ...]:
        candidates: list[ResidencyEvictionCandidate] = []
        for handle, value in planning.values.items():
            if handle in blocked_handles:
                continue
            materialization = value.materializations.get(location)
            if materialization is None:
                continue
            if (
                materialization.use_count
                or materialization.hold_count
                or materialization.pending_event_count
            ):
                continue
            candidates.append(
                ResidencyEvictionCandidate(
                    handle=handle,
                    location=location,
                    charged_nbytes=materialization.charged_nbytes,
                    registration_index=planning.registration_order[handle],
                    last_access_epoch=self._last_access.get(handle),
                    metadata=self._metadata.get(
                        handle,
                        ResidencyPolicyMetadata(),
                    ),
                )
            )
        return tuple(candidates)

    def _ordered_candidates(
        self,
        candidates: Sequence[ResidencyEvictionCandidate],
        *,
        required_endpoints: set[tuple[ResidencyHandle[Any], ResidencyLocation]],
    ) -> tuple[ResidencyEvictionCandidate, ...]:
        ordinary = tuple(
            candidate
            for candidate in candidates
            if (candidate.handle, candidate.location) not in required_endpoints
        )
        required = tuple(
            candidate
            for candidate in candidates
            if (candidate.handle, candidate.location) in required_endpoints
        )
        ordered = (
            *self._policy.order_candidates(ordinary),
            *self._policy.order_candidates(required),
        )
        if len(ordered) != len(candidates) or set(ordered) != set(candidates):
            raise RuntimeError(
                "ResidencyPolicy must return every candidate exactly once"
            )
        return ordered

    def _reclaim_states(
        self,
        planning: _PlanningState,
        candidate: ResidencyEvictionCandidate,
        *,
        rank: int,
        required_endpoints: set[tuple[ResidencyHandle[Any], ResidencyLocation]],
        blocked_handles: frozenset[ResidencyHandle[Any]],
        active_pairs: frozenset[tuple[ResidencyHandle[Any], ResidencyLocation]],
        memo: _SearchMemo,
    ) -> Iterator[_PlanningState]:
        value = planning.values[candidate.handle]
        source = candidate.location
        pair = (candidate.handle, source)
        if pair in active_pairs:
            return
        charge = value.snapshot.spec.storage_nbytes
        alternatives = tuple(
            location
            for location in value.materializations
            if location != source
        )
        if alternatives or value.snapshot.has_source:
            trial = planning.clone()
            trial_value = trial.values[candidate.handle]
            action: DropResident[Any] | MoveResident[Any] = DropResident(
                candidate.handle,
                source,
            )
            self._append_reclaim(
                trial,
                action,
                candidate,
                rank=rank,
                reason=(
                    "another materialization preserves the logical value"
                    if alternatives
                    else "registered source can reconstruct the logical value"
                ),
            )
            del trial_value.materializations[source]
            trial.used[source] = trial.used.get(source, 0) - charge
            yield trial
            if alternatives:
                return

        requested_destinations = tuple(
            requirement_location
            for requirement_handle, requirement_location in required_endpoints
            if requirement_handle == candidate.handle
            and requirement_location != source
        )
        fallbacks = tuple(
            dict.fromkeys(
                (
                    *self._ordered_locations(requested_destinations),
                    *self._policy.fallback_locations(source),
                )
            )
        )
        for destination in fallbacks:
            for capacitated in self._capacity_states(
                planning.clone(),
                destination,
                (0 if destination in value.materializations else charge),
                required_endpoints=required_endpoints,
                blocked_handles=blocked_handles | {candidate.handle},
                active_pairs=active_pairs | {pair},
                memo=memo,
            ):
                trial = capacitated.clone()
                trial_value = trial.values[candidate.handle]
                if source not in trial_value.materializations:
                    continue
                if destination not in trial_value.materializations:
                    trial_value.materializations[destination] = (
                        self._project_materialization(
                            trial_value.snapshot,
                            destination,
                            trial_value.materializations[source],
                        )
                    )
                    trial.used[destination] = (
                        trial.used.get(destination, 0) + charge
                    )
                action = MoveResident(
                    candidate.handle,
                    destination,
                    from_location=source,
                )
                self._append_reclaim(
                    trial,
                    action,
                    candidate,
                    rank=rank,
                    reason=(
                        "sole materialization moved to requested destination "
                        f"{destination} instead of reconstructing it later"
                    ),
                )
                del trial_value.materializations[source]
                trial.used[source] = trial.used.get(source, 0) - charge
                yield trial

    @staticmethod
    def _append_reclaim(
        planning: _PlanningState,
        action: DropResident[Any] | MoveResident[Any],
        candidate: ResidencyEvictionCandidate,
        *,
        rank: int,
        reason: str,
    ) -> None:
        planning.reclaim.append(action)
        planning.evictions.append(
            ResidencyEviction(
                action=action,
                rank=rank,
                released_location=candidate.location,
                released_nbytes=candidate.charged_nbytes,
                reason=reason,
            )
        )

    @staticmethod
    def _project_materialization(
        value: ResidencyValueSnapshot[Any],
        destination: ResidencyLocation,
        source: MaterializationSnapshot | None,
    ) -> MaterializationSnapshot:
        if source is not None:
            return replace(source, location=destination)
        return MaterializationSnapshot(
            location=destination,
            logical_nbytes=value.spec.logical_nbytes,
            storage_nbytes=value.spec.storage_nbytes,
            charged_nbytes=value.spec.storage_nbytes,
            use_count=0,
            hold_count=0,
            pending_event_count=0,
        )

    @staticmethod
    def _planning_signature(
        planning: _PlanningState,
    ) -> tuple[object, ...]:
        values = tuple(
            (
                planning.registration_order[handle],
                tuple(
                    location.name
                    for location in ResidencyController._ordered_locations(
                        value.materializations
                    )
                ),
            )
            for handle, value in sorted(
                planning.values.items(),
                key=lambda item: planning.registration_order[item[0]],
            )
        )
        locations = tuple(
            (
                location.name,
                planning.used.get(location, 0),
                planning.reserved.get(location, 0),
            )
            for location in ResidencyController._ordered_locations(
                tuple(set(planning.used) | set(planning.reserved))
            )
        )
        return values, locations

    @staticmethod
    def _ordered_locations(
        locations: Mapping[ResidencyLocation, object]
        | Sequence[ResidencyLocation],
    ) -> tuple[ResidencyLocation, ...]:
        values = tuple(locations)
        host_order = {PAGEABLE_HOST: 0, PINNED_HOST: 1}
        return tuple(
            sorted(
                values,
                key=lambda location: (
                    host_order.get(location, 2),
                    -1
                    if location.device.index is None
                    else location.device.index,
                ),
            )
        )

    def _record_accesses(
        self,
        handles: Sequence[ResidencyHandle[Any]],
    ) -> None:
        with self._lock:
            self._access_epoch += 1
            for handle in handles:
                self._last_access[handle] = self._access_epoch


class _ResidencyUseValues(Mapping[ResidencyRequirement, TensorResident]):
    def __init__(
        self,
        sources: Mapping[ResidencyRequirement, BorrowedValues],
    ) -> None:
        self._sources = dict(sources)

    def __getitem__(
        self,
        requirement: ResidencyRequirement,
    ) -> TensorResident:
        return self._sources[requirement][requirement.handle]

    def __iter__(self) -> Iterator[ResidencyRequirement]:
        for requirement, values in self._sources.items():
            # Force active-lifetime validation before yielding each key.
            values[requirement.handle]
            yield requirement

    def __len__(self) -> int:
        for requirement, values in self._sources.items():
            values[requirement.handle]
        return len(self._sources)


class ResidencyUse:
    """Single-use automatic admission, borrow, and reservation lifetime."""

    def __init__(
        self,
        *,
        controller: ResidencyController,
        request: ResidencyRequest,
        consumer_streams: ConsumerStreams | None,
        transfer_streams: TransferStreams | None,
    ) -> None:
        if not isinstance(request, ResidencyRequest):
            raise TypeError("ResidencyUse request must be a ResidencyRequest")
        self._controller = controller
        self.request = request
        self._consumer_streams = _normalize_stream_mapping(
            consumer_streams,
            what="consumer_streams",
        )
        required_cuda_locations = {
            requirement.location
            for requirement in request.requirements
            if requirement.location.kind == "cuda"
        }
        supplied_cuda_locations = set(self._consumer_streams)
        if supplied_cuda_locations != required_cuda_locations:
            missing = required_cuda_locations - supplied_cuda_locations
            unexpected = supplied_cuda_locations - required_cuda_locations
            raise ValueError(
                "ResidencyUse consumer_streams must exactly cover required "
                "CUDA locations before automatic admission: "
                f"missing={tuple(map(str, missing))}, "
                f"unexpected={tuple(map(str, unexpected))}"
            )
        self._transfer_streams = _normalize_stream_mapping(
            transfer_streams,
            what="transfer_streams",
        )
        self._stack: ExitStack | None = None
        self._scope: ResidencyScope | None = None
        self._decision: ResidencyDecision | None = None
        self._values: _ResidencyUseValues | None = None
        self._entered = False
        self._closed = False

    @property
    def decision(self) -> ResidencyDecision:
        """State-bound decision after successful context entry."""

        if self._decision is None:
            raise RuntimeError(
                "ResidencyUse decision is available after context entry"
            )
        return self._decision

    @property
    def values(self) -> Mapping[ResidencyRequirement, TensorResident]:
        """Borrowed values keyed by requirement during this context."""

        if self._values is None:
            raise RuntimeError(
                "ResidencyUse values are available during context"
            )
        return self._values

    def value(
        self,
        handle: ResidencyHandle[Any],
        *,
        at: ResidencyLocation,
    ) -> TensorResident:
        """Return the borrow for one ``(handle, location)`` endpoint."""

        return self.values[ResidencyRequirement(handle, at)]

    @property
    def report(self) -> ResidencyPlanReport | None:
        """Completed plan report after successful scope close."""

        return None if self._scope is None else self._scope.report

    @property
    def exit_error(self) -> BaseException | None:
        """Structured plan cleanup failure retained by the underlying scope."""

        return None if self._scope is None else self._scope.exit_error

    def __enter__(self) -> ResidencyUse:
        if self._entered:
            raise RuntimeError("ResidencyUse cannot be entered twice")
        self._entered = True
        stack = ExitStack()
        self._stack = stack
        try:
            decision = self._controller.decide(self.request)
            self._decision = decision
            scope = self._controller.scope(
                decision,
                transfer_streams=self._transfer_streams,
            )
            self._scope = scope
            stack.enter_context(scope)
            sources: dict[ResidencyRequirement, BorrowedValues] = {}
            grouped: dict[ResidencyLocation, list[ResidencyRequirement]] = {}
            for requirement in self.request.requirements:
                grouped.setdefault(requirement.location, []).append(requirement)
            for location, requirements in grouped.items():
                stream = self._consumer_streams.get(location)
                handles = tuple(
                    requirement.handle for requirement in requirements
                )
                lease = self._controller.manager.acquire(
                    handles,
                    at=location,
                    consumer_stream=stream,
                )
                borrowed = stack.enter_context(lease)
                for requirement in requirements:
                    sources[requirement] = borrowed
            self._values = _ResidencyUseValues(sources)
            self._controller._record_accesses(
                tuple(
                    dict.fromkeys(requirement.handle for requirement in sources)
                )
            )
            return self
        except BaseException as error:
            self._closed = True
            stack.__exit__(type(error), error, error.__traceback__)
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._closed:
            return False
        self._closed = True
        assert self._stack is not None
        try:
            return bool(self._stack.__exit__(exc_type, exc_value, traceback))
        finally:
            self._values = None


def _normalize_stream_mapping(
    streams: Mapping[ResidencyLocation, torch.cuda.Stream] | None,
    *,
    what: str,
) -> dict[ResidencyLocation, torch.cuda.Stream]:
    if streams is None:
        return {}
    if not isinstance(streams, Mapping):
        raise TypeError(f"ResidencyUse {what} must be a mapping")
    normalized: dict[ResidencyLocation, torch.cuda.Stream] = {}
    for location, stream in streams.items():
        if not isinstance(location, ResidencyLocation):
            raise TypeError(
                f"ResidencyUse {what} keys must be ResidencyLocation objects"
            )
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError(
                f"ResidencyUse {what} values must be torch.cuda.Stream objects"
            )
        if location.kind != "cuda":
            raise ValueError(
                f"ResidencyUse {what} cannot assign a CUDA stream to {location}"
            )
        if torch.device(stream.device) != location.device:
            raise ValueError(
                f"ResidencyUse {what} stream device {stream.device} does not "
                f"match {location.device}"
            )
        normalized[location] = stream
    return normalized


__all__ = [
    "ResidencyController",
    "ResidencyDecision",
    "ResidencyEviction",
    "ResidencyUse",
]
