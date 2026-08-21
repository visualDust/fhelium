"""Managed value identity, transitions, plans, and accounting."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from operator import index
from threading import Event, Lock, RLock, get_ident
from time import time_ns
from types import TracebackType
from typing import Any, Literal, NoReturn, TypeVar, cast
from uuid import uuid4
from warnings import warn
from weakref import finalize

import torch

from fhelium.core import TensorResident
from fhelium.core.tensor_resident import (
    _cpu_pinning_is_uniform,
    _storage_keys,
)
from fhelium.errors import (
    ResidencyBudgetError,
    ResidencyClosedError,
    ResidencyHandleError,
    ResidencyInUseError,
    ResidencyMaterializationError,
    ResidencyOwnershipError,
    ResidencyPlanError,
    ResidencyPlanExecutionError,
    ResidencyReentrancyError,
    ResidencyStaleStateError,
    ResidencyUnavailableError,
)
from fhelium.residency.lease import (
    ResidencyHold,
    ResidencyLease,
    ResidencyReservation,
)
from fhelium.residency.location import (
    PAGEABLE_HOST,
    PINNED_HOST,
    ResidencyLocation,
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

ValueT = TypeVar("ValueT", bound=TensorResident)
_MANAGER_ID_LOCK = Lock()
_ISSUED_MANAGER_IDS: set[str] = set()


@dataclass
class _Materialization:
    value: TensorResident
    logical_nbytes: int
    storage_nbytes: int
    charged_nbytes: int
    use_tokens: set[object] = field(default_factory=set)
    hold_tokens: set[object] = field(default_factory=set)
    pending_tokens: set[object] = field(default_factory=set)


@dataclass
class _ValueRecord:
    spec: ResidencyValueSpec[Any]
    source: ResidencySource[Any] | None
    source_location: ResidencyLocation | None
    materializations: dict[ResidencyLocation, _Materialization]
    discarded: bool = False


@dataclass
class _LocationState:
    budget_bytes: int | None
    used_bytes: int = 0
    reserved_bytes: int = 0
    peak_used_bytes: int = 0
    peak_charged_bytes: int = 0


@dataclass
class _LeaseRecord:
    handles: tuple[ResidencyHandle[Any], ...]
    location: ResidencyLocation
    state: str = "active"
    events: tuple[torch.cuda.Event, ...] = ()


@dataclass(frozen=True)
class _HoldRecord:
    handles: tuple[ResidencyHandle[Any], ...]
    location: ResidencyLocation


@dataclass(frozen=True)
class _ReservationRecord:
    location: ResidencyLocation
    nbytes: int
    label: str


@dataclass
class _SimulationState:
    locations: dict[ResidencyHandle[Any], set[ResidencyLocation]]
    discarded: set[ResidencyHandle[Any]]
    budgets: dict[ResidencyLocation, int | None]
    used: dict[ResidencyLocation, int]
    reserved: dict[ResidencyLocation, int]
    peaks: dict[ResidencyLocation, int]


class ResidencyManager:
    """Own managed values and their local memory materializations.

    The manager provides ``ensure``, ``move``, ``drop``, and
    ``discard`` transitions across pageable, pinned, or indexed CUDA locations.
    Locations are recorded when first budgeted or used. Values registered
    through :meth:`adopt` transfer logical
    alias ownership to this manager under caller-enforced rules; callers must
    not retain or mutate the concrete value outside a
    :class:`ResidencyLease`. The runtime cannot enforce Python alias
    destruction, so accounting covers only manager-owned storage while those
    rules are followed.

    Materialization is synchronous. CUDA read lifetimes are not: lease release
    records consumer-stream events and retains protection until they complete.
    Plans and scopes compose the same primitive transitions and reserve
    headroom for unmanaged outputs or native workspace.

    Registered source callbacks execute under the manager's transition lock.
    While a callback is active, the callback access guard rejects reentrant and
    concurrent public state access to that manager from every process thread;
    a callback must not call the manager or wait for work that does so.

    Args:
        budgets: Optional strict managed-byte limits by location. Omitted
            locations remain unbudgeted and are recorded lazily when used.
        trace_capacity: Maximum completed transition reports retained in the
            in-memory trace. Zero disables retention.
    """

    def __init__(
        self,
        budgets: Mapping[ResidencyLocation, int] | None = None,
        *,
        trace_capacity: int = 4096,
    ) -> None:
        if budgets is not None and not isinstance(budgets, Mapping):
            raise TypeError("ResidencyManager budgets must be a mapping")
        self._manager_id = _new_manager_id()
        self._state_version = 0
        trace_capacity = _nonnegative_count(
            trace_capacity,
            what="ResidencyManager trace_capacity",
        )
        self._locations: dict[ResidencyLocation, _LocationState] = {}
        for location, budget in ({} if budgets is None else budgets).items():
            self._validate_location(location)
            self._locations[location] = _LocationState(
                budget_bytes=_nonnegative_count(
                    budget,
                    what=f"Budget for {location}",
                )
            )
        self._records: dict[ResidencyHandle[Any], _ValueRecord] = {}
        self._issued_handle_ids: set[str] = set()
        self._storage_owners: dict[
            tuple[torch.device, int],
            tuple[ResidencyHandle[Any], ResidencyLocation],
        ] = {}
        self._leases: dict[object, _LeaseRecord] = {}
        self._holds: dict[object, _HoldRecord] = {}
        self._reservations: dict[object, _ReservationRecord] = {}
        self._trace: deque[ResidencyTransitionReport] = deque(
            maxlen=trace_capacity or None
        )
        self._trace_enabled = trace_capacity > 0
        self._lock = RLock()
        self._source_callback_active = Event()
        self._transition_thread: int | None = None
        self._closed = False

    @property
    def manager_id(self) -> str:
        """Identity embedded in every handle issued by this manager."""

        return self._manager_id

    @property
    def state_version(self) -> int:
        """Monotonic version of placement, ownership, and lifetime state."""

        with self._state_access():
            self._require_available_locked()
            self._require_not_reentrant_locked()
            self._reap_completed_locked()
            return self._state_version

    @property
    def locations(self) -> tuple[ResidencyLocation, ...]:
        """Budgeted or observed locations in registration order."""

        with self._state_access():
            return tuple(self._locations)

    def adopt(
        self,
        value: ValueT,
        *,
        at: ResidencyLocation | None = None,
        replica_mode: ReplicaMode = ReplicaMode.EXCLUSIVE,
    ) -> ResidencyHandle[ValueT]:
        """Transfer logical ownership of one live value into managed state.

        The caller must stop retaining and using ``value`` after this call.
        Python cannot revoke existing aliases; violating this rule weakens
        accounting and immutability guarantees.
        """

        if not isinstance(value, TensorResident):
            raise TypeError(
                "ResidencyManager adopt value must be a TensorResident"
            )
        with self._state_access():
            self._begin_public_operation_locked()
            try:
                location = self._infer_location(value) if at is None else at
                self._validate_location(location)
                self._validate_value_location(value, location)
                spec: ResidencyValueSpec[ValueT] = ResidencyValueSpec(
                    value_type=type(value),
                    logical_nbytes=value.nbytes,
                    storage_nbytes=max(value.nbytes, value.storage_nbytes),
                    replica_mode=replica_mode,
                    recoverability=Recoverability.MUST_PRESERVE,
                )
                handle = self._new_handle(value_type=type(value))
                record = _ValueRecord(
                    spec=spec,
                    source=None,
                    source_location=None,
                    materializations={},
                )
                self._records[handle] = record
                try:
                    self._install_locked(handle, record, location, value)
                except BaseException:
                    del self._records[handle]
                    raise
                return handle
            finally:
                self._end_public_operation_locked()

    def register_source(
        self,
        spec: ResidencyValueSpec[ValueT],
        source: ResidencySource[ValueT],
        *,
        source_location: ResidencyLocation = PAGEABLE_HOST,
    ) -> ResidencyHandle[ValueT]:
        """Register a trusted reconstruction source without loading a value.

        The source must reproduce the contents and CKKS state identified
        by the new handle. The manager later validates runtime type, declared
        location, logical bytes, and storage ceiling; the source owns
        content-identity correctness.

        Args:
            spec: Immutable reconstructible value specification and accounting
                limits.
            source: Synchronous trusted loader returning independent storage.
            source_location: Valid local location charged during reconstruction.

        Raises:
            ValueError: If location or recoverability is invalid.
            TypeError: If ``source`` does not provide ``load()``.
        """

        if not isinstance(spec, ResidencyValueSpec):
            raise TypeError(
                "ResidencyManager source spec must be a ResidencyValueSpec"
            )
        if not callable(getattr(source, "load", None)):
            raise TypeError("Residency source must define load() as a callable")
        with self._state_access():
            self._begin_public_operation_locked()
            try:
                self._validate_location(source_location)
                if spec.recoverability is not Recoverability.RECONSTRUCTIBLE:
                    raise ValueError(
                        "A registered ResidencySource requires RECONSTRUCTIBLE"
                    )
                self._ensure_location_locked(source_location)
                handle = self._new_handle(value_type=spec.value_type)
                self._records[handle] = _ValueRecord(
                    spec=spec,
                    source=source,
                    source_location=source_location,
                    materializations={},
                )
                self._bump_state_locked()
                return handle
            finally:
                self._end_public_operation_locked()

    def ensure(
        self,
        handle: ResidencyHandle[ValueT],
        at: ResidencyLocation,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> ResidencyTransitionReport:
        """Ensure one replica at ``at`` while retaining existing replicas."""

        return self._run_action(EnsureResident(handle, at), stream=stream)

    def copy(
        self,
        handle: ResidencyHandle[ValueT],
        to: ResidencyLocation,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> ResidencyTransitionReport:
        """Create or retain a replica at ``to`` without removing its source."""

        return self.ensure(handle, to, stream=stream)

    def move(
        self,
        handle: ResidencyHandle[ValueT],
        to: ResidencyLocation,
        *,
        from_location: ResidencyLocation | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> ResidencyTransitionReport:
        """Materialize at ``to`` and remove one selected source replica."""

        return self._run_action(
            MoveResident(handle, to, from_location),
            stream=stream,
        )

    def drop(
        self,
        handle: ResidencyHandle[Any],
        at: ResidencyLocation,
    ) -> ResidencyTransitionReport:
        """Remove one unprotected replica without ending its logical value."""

        return self._run_action(DropResident(handle, at))

    def discard(
        self,
        handle: ResidencyHandle[Any],
    ) -> ResidencyTransitionReport:
        """End one managed value and remove all unprotected replicas."""

        return self._run_action(DiscardValue(handle))

    def execute_actions(
        self,
        actions: Iterable[ResidencyAction],
        *,
        name: str = "residency-actions",
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream]
        | None = None,
        expected_state_version: int | None = None,
    ) -> ResidencyPlanReport:
        """Preflight and execute an ordered primitive transition sequence.

        Preflight is atomic with respect to manager state, but execution is not
        transactional. If a runtime source, allocation, copy, or later action
        fails, completed actions remain valid. ``ResidencyPlanExecutionError``
        identifies the failed phase and action and carries their structured
        partial ``ResidencyPlanReport``. Full success returns the complete
        report.

        Args:
            actions: Ordered low-level actions to simulate and execute.
            name: Non-empty diagnostic plan name.
            transfer_streams: Optional destination-location to CUDA copy-stream
                mapping. Actions without a configured destination stream use
                the current stream on that destination device.
            expected_state_version: Optional state-version precondition checked
                atomically before preflight or mutation.

        Raises:
            ResidencyPlanError: If current-state preflight is infeasible.
            ResidencyPlanExecutionError: If execution begins and a runtime
                source, allocation, copy, or residency invariant fails.
        """

        _require_nonempty_string(name, what="Residency plan name")
        action_tuple = _normalize_actions(actions)
        started = time_ns()
        with self._state_access():
            self._begin_public_operation_locked()
            transitions: list[ResidencyTransitionReport] = []
            try:
                self._require_expected_state_version_locked(
                    expected_state_version
                )
                streams = self._normalize_transfer_streams_locked(
                    transfer_streams
                )
                explanation = self._explain_actions_locked(
                    name,
                    action_tuple,
                    (),
                )
                if not explanation.feasible:
                    raise ResidencyPlanError(
                        explanation.reason or f"Plan {name!r} is infeasible"
                    )
                failed_action: ResidencyAction | None = None
                failed_action_index: int | None = None
                try:
                    for action in action_tuple:
                        failed_action = action
                        failed_action_index = len(transitions)
                        transitions.append(
                            self._execute_action_locked(
                                action,
                                stream=self._stream_for_action(action, streams),
                            )
                        )
                        failed_action = None
                        failed_action_index = None
                except BaseException as error:
                    _raise_plan_execution_failure(
                        error,
                        plan_name=name,
                        phase="execute",
                        transitions=tuple(transitions),
                        started_at_ns=started,
                        failed_action=failed_action,
                        failed_action_index=failed_action_index,
                    )
            finally:
                self._end_public_operation_locked()
        return ResidencyPlanReport(
            plan_name=name,
            transitions=tuple(transitions),
            started_at_ns=started,
            completed_at_ns=time_ns(),
        )

    def explain(
        self,
        plan: ResidencyPlan,
        *,
        expected_state_version: int | None = None,
    ) -> ResidencyPlanExplanation:
        """Dry-run one complete scope plan without changing manager state.

        Simulation covers reservations, ordered entry and exit actions,
        deterministic source resolution, protection, budgets, and predicted
        charged peaks. The result is point-in-time evidence, not an admission
        lock; scope entry repeats preflight.
        """

        if not isinstance(plan, ResidencyPlan):
            raise TypeError(
                "ResidencyManager explain plan must be a ResidencyPlan"
            )
        with self._state_access():
            self._require_available_locked()
            self._require_not_reentrant_locked()
            self._require_expected_state_version_locked(expected_state_version)
            return self._explain_actions_locked(
                plan.name,
                (*plan.reclaim, *plan.enter, *plan.exit),
                plan.reservations,
                reservation_after_actions=len(plan.reclaim),
            )

    def scope(
        self,
        plan: ResidencyPlan,
        *,
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream]
        | None = None,
        expected_state_version: int | None = None,
    ) -> ResidencyScope:
        """Create a single-use scope for ordered plan entry, body, and exit.

        Entering executes reclaim actions, admits reservations, then executes
        entry actions. Closing runs exit actions and releases reservations.
        Completed transitions are never rolled back after a later failure.

        ``expected_state_version`` is an optional state-version precondition
        checked atomically at entry before any plan mutation. Destination copy
        streams are selected from ``transfer_streams`` independently for each
        indexed CUDA location.
        """

        if not isinstance(plan, ResidencyPlan):
            raise TypeError(
                "ResidencyManager scope plan must be a ResidencyPlan"
            )
        return ResidencyScope(
            self,
            plan,
            transfer_streams=transfer_streams,
            expected_state_version=expected_state_version,
        )

    def acquire(
        self,
        handles: Iterable[ResidencyHandle[Any]],
        *,
        at: ResidencyLocation,
        consumer_stream: torch.cuda.Stream | None = None,
    ) -> ResidencyLease:
        """Borrow already-ready immutable materializations for evaluator reads.

        CUDA leases require an explicit ``consumer_stream`` so release and
        finalization remain correct across Python threads. Additional consumer
        streams must be registered on the returned lease before release. This
        method is atomic across all handles and never materializes a missing
        value implicitly.

        Args:
            handles: Non-empty managed values to borrow; duplicates collapse.
            at: One valid local location shared by all requested values.
            consumer_stream: Required initial stream for CUDA; omitted for CPU.

        Raises:
            ValueError: If no handles are supplied or the location or CUDA
                stream requirements are not satisfied.
            ResidencyUnavailableError: If any value is not already resident.
            ResidencyHandleError: If any handle is foreign, stale, or unknown.
        """

        normalized = _unique_handles(handles)
        with self._state_access():
            self._require_available_locked()
            self._require_not_reentrant_locked()
            self._validate_location(at)
            if at.kind == "cuda" and consumer_stream is None:
                raise ValueError(
                    "CUDA residency leases require an explicit consumer_stream"
                )
            self._reap_completed_locked()
            streams: tuple[torch.cuda.Stream, ...] = ()
            if consumer_stream is not None:
                self._validate_stream(at, consumer_stream)
                streams = (consumer_stream,)
            token = object()
            installed: list[_Materialization] = []
            for handle in normalized:
                record = self._record_for_locked(handle)
                materialization = record.materializations.get(at)
                if materialization is None:
                    raise ResidencyUnavailableError(
                        f"Value {handle.handle_id!r} is not resident at {at}"
                    )
                installed.append(materialization)
            self._leases[token] = _LeaseRecord(normalized, at)
            for materialization in installed:
                materialization.use_tokens.add(token)
            self._bump_state_locked()
            return ResidencyLease(
                manager=self,
                token=token,
                handles=normalized,
                location=at,
                consumer_streams=streams,
            )

    def hold(
        self,
        handles: Iterable[ResidencyHandle[Any]],
        *,
        at: ResidencyLocation,
    ) -> ResidencyHold:
        """Retain a non-empty set of ready materializations outside active use."""

        normalized = _unique_handles(handles)
        with self._state_access():
            self._require_available_locked()
            self._require_not_reentrant_locked()
            self._validate_location(at)
            self._reap_completed_locked()
            token = object()
            installed: list[_Materialization] = []
            for handle in normalized:
                record = self._record_for_locked(handle)
                materialization = record.materializations.get(at)
                if materialization is None:
                    raise ResidencyUnavailableError(
                        f"Value {handle.handle_id!r} is not resident at {at}"
                    )
                installed.append(materialization)
            self._holds[token] = _HoldRecord(normalized, at)
            for materialization in installed:
                materialization.hold_tokens.add(token)
            self._bump_state_locked()
            return ResidencyHold(
                manager=self,
                token=token,
                handles=normalized,
                location=at,
            )

    def reserve(
        self,
        location: ResidencyLocation,
        nbytes: int,
        *,
        label: str,
    ) -> ResidencyReservation:
        """Record headroom and enforce its location budget when present."""

        nbytes = _nonnegative_count(nbytes, what="Residency reservation nbytes")
        if not isinstance(label, str):
            raise TypeError("Residency reservation label must be a string")
        if not label.strip():
            raise ValueError("Residency reservation label must be non-empty")
        with self._state_access():
            self._require_available_locked()
            self._require_not_reentrant_locked()
            token = self._reserve_locked(location, nbytes, label)
            return ResidencyReservation(
                manager=self,
                token=token,
                location=location,
                nbytes=nbytes,
                label=label,
            )

    def snapshot(self) -> ResidencySnapshot[TensorResident]:
        """Return one atomic tensor-free view of values and location budgets."""

        with self._state_access():
            self._require_available_locked()
            self._require_not_reentrant_locked()
            self._reap_completed_locked()
            value_snapshots: list[ResidencyValueSnapshot[TensorResident]] = []
            for handle, record in self._records.items():
                materializations = tuple(
                    MaterializationSnapshot(
                        location=location,
                        logical_nbytes=item.logical_nbytes,
                        storage_nbytes=item.storage_nbytes,
                        charged_nbytes=item.charged_nbytes,
                        use_count=len(item.use_tokens),
                        hold_count=len(item.hold_tokens),
                        pending_event_count=len(item.pending_tokens),
                    )
                    for location, item in record.materializations.items()
                )
                value_snapshots.append(
                    ResidencyValueSnapshot(
                        handle=handle,
                        spec=record.spec,
                        materializations=materializations,
                        has_source=record.source is not None,
                        source_location=record.source_location,
                        discarded=record.discarded,
                    )
                )
            location_snapshots = []
            for location, state in self._locations.items():
                allocator_allocated, allocator_reserved = (
                    self._best_effort_allocator_metrics(
                        location.device if location.kind == "cuda" else None
                    )
                )
                items = [
                    item
                    for record in self._records.values()
                    for item_location, item in record.materializations.items()
                    if item_location == location
                ]
                location_snapshots.append(
                    ResidencyLocationSnapshot(
                        location=location,
                        budget_bytes=state.budget_bytes,
                        used_bytes=state.used_bytes,
                        reserved_bytes=state.reserved_bytes,
                        remaining_budget_bytes=(
                            None
                            if state.budget_bytes is None
                            else state.budget_bytes
                            - state.used_bytes
                            - state.reserved_bytes
                        ),
                        peak_used_bytes=state.peak_used_bytes,
                        peak_charged_bytes=state.peak_charged_bytes,
                        value_count=len(items),
                        reservation_count=sum(
                            item.location == location
                            for item in self._reservations.values()
                        ),
                        use_count=sum(len(item.use_tokens) for item in items),
                        hold_count=sum(len(item.hold_tokens) for item in items),
                        pending_event_count=sum(
                            len(item.pending_tokens) for item in items
                        ),
                        allocator_allocated_bytes=allocator_allocated,
                        allocator_reserved_bytes=allocator_reserved,
                    )
                )
            reservation_snapshots = tuple(
                ResidencyReservationSnapshot(
                    location=item.location,
                    nbytes=item.nbytes,
                    label=item.label,
                )
                for item in self._reservations.values()
            )
            return ResidencySnapshot(
                manager_id=self.manager_id,
                state_version=self._state_version,
                values=tuple(value_snapshots),
                locations=tuple(location_snapshots),
                reservations=reservation_snapshots,
                captured_at_ns=time_ns(),
            )

    def trace(self) -> tuple[ResidencyTransitionReport, ...]:
        """Return retained completed transitions in completion order."""

        with self._state_access():
            return tuple(self._trace)

    def clear_trace(self) -> None:
        """Remove retained transition reports without changing residency."""

        with self._state_access():
            self._trace.clear()

    def close(self, *, wait: bool = True, force: bool = False) -> None:
        """Release manager-owned state after checking active lifetimes.

        Args:
            wait: Synchronize already pending CUDA lease events before checking
                whether lifetime state remains.
            force: Synchronize configured CUDA devices and clear active
                protections. This can invalidate public lifetime objects.

        ``force=True`` synchronizes configured CUDA devices and clears even
        leaked active protections. It is an explicit unsafe escape hatch for
        tests and process shutdown, not normal request cleanup.

        Raises:
            ResidencyInUseError: If active or pending lifetimes remain without
                ``force=True``.
            ResidencyReentrancyError: If called from a source callback.
        """

        with self._state_access():
            if self._closed:
                return
            self._require_not_reentrant_locked()
            if wait:
                self._synchronize_pending_locked()
            else:
                self._reap_completed_locked()
            active_leases = [
                record
                for record in self._leases.values()
                if record.state == "active"
            ]
            if (
                active_leases
                or self._holds
                or self._reservations
                or self._leases
            ) and not force:
                raise ResidencyInUseError(
                    "ResidencyManager cannot close with active or pending "
                    f"lifetimes: active_leases={len(active_leases)}, "
                    f"pending_leases={len(self._leases) - len(active_leases)}, "
                    f"holds={len(self._holds)}, "
                    f"reservations={len(self._reservations)}"
                )
            if force:
                for location in self._locations:
                    if location.kind == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(location.device)
            self._leases.clear()
            self._holds.clear()
            self._reservations.clear()
            self._storage_owners.clear()
            self._records.clear()
            for state in self._locations.values():
                state.used_bytes = 0
                state.reserved_bytes = 0
            self._closed = True
            self._bump_state_locked()

    def __enter__(self) -> ResidencyManager:
        self._require_available()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # Lifetime methods called only by public objects in lease.py.

    def _leased_value(
        self,
        token: object,
        handle: ResidencyHandle[ValueT],
    ) -> TensorResident:
        with self._state_access():
            self._require_not_reentrant_locked()
            record = self._leases.get(token)
            if (
                record is None
                or record.state != "active"
                or handle not in record.handles
            ):
                raise ResidencyUnavailableError(
                    "Handle is not protected by this active residency lease"
                )
            value_record = self._record_for_locked(handle)
            materialization = value_record.materializations.get(record.location)
            if (
                materialization is None
                or token not in materialization.use_tokens
            ):
                raise RuntimeError("Residency lease accounting is inconsistent")
            return materialization.value

    def _release_lease(
        self,
        token: object,
        *,
        consumer_streams: tuple[torch.cuda.Stream, ...],
        wait: bool,
    ) -> None:
        with self._state_access():
            self._require_not_reentrant_locked()
            record = self._leases.get(token)
            if record is None:
                return
            if record.state != "active":
                return
            if record.location.kind != "cuda":
                self._remove_active_use_locked(token, record)
                del self._leases[token]
                return
            streams = consumer_streams
            if not streams:
                # Defensive fail-safe for an internally malformed or legacy
                # lease. Never resolve a thread-local current stream here.
                with torch.cuda.device(record.location.device):
                    torch.cuda.synchronize(record.location.device)
                self._remove_active_use_locked(token, record)
                del self._leases[token]
                warn(
                    "CUDA residency lease had no consumer stream; "
                    "release used a full-device synchronization fallback",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            for stream in streams:
                self._validate_stream(record.location, stream)
            events: list[torch.cuda.Event] = []
            try:
                with torch.cuda.device(record.location.device):
                    for stream in streams:
                        event = torch.cuda.Event()
                        event.record(stream)
                        events.append(event)
            except BaseException:
                # A conservative synchronous fallback prevents early release
                # after a partially recorded multi-stream completion set.
                for stream in streams:
                    stream.synchronize()
                self._remove_active_use_locked(token, record)
                del self._leases[token]
                warn(
                    "CUDA event recording failed; residency lease release used "
                    "a synchronous stream fallback",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            self._move_use_to_pending_locked(token, record)
            record.state = "pending"
            record.events = tuple(events)
            if wait:
                for event in events:
                    event.synchronize()
                self._release_pending_locked(token, record)

    def _release_hold(self, token: object, *, missing_ok: bool = False) -> None:
        with self._state_access():
            self._require_not_reentrant_locked()
            record = self._holds.get(token)
            if record is None:
                if missing_ok or self._closed:
                    return
                raise RuntimeError("Unknown or released ResidencyHold token")
            for handle in record.handles:
                value_record = self._record_for_locked(handle)
                materialization = value_record.materializations[record.location]
                materialization.hold_tokens.remove(token)
            del self._holds[token]
            self._bump_state_locked()

    def _release_reservation(
        self,
        token: object,
        *,
        missing_ok: bool = False,
    ) -> None:
        with self._state_access():
            self._require_not_reentrant_locked()
            self._release_reservation_locked(token, missing_ok=missing_ok)

    def _release_reservation_locked(
        self,
        token: object,
        *,
        missing_ok: bool = False,
    ) -> None:
        record = self._reservations.get(token)
        if record is None:
            if missing_ok or self._closed:
                return
            raise RuntimeError("Unknown or released reservation token")
        self._locations[record.location].reserved_bytes -= record.nbytes
        del self._reservations[token]
        self._bump_state_locked()

    # Plan-scope methods.

    def _enter_scope(
        self,
        plan: ResidencyPlan,
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream],
        expected_state_version: int | None,
    ) -> tuple[tuple[object, ...], tuple[ResidencyTransitionReport, ...], int]:
        started = time_ns()
        with self._state_access():
            self._begin_public_operation_locked()
            tokens: list[object] = []
            transitions: list[ResidencyTransitionReport] = []
            try:
                self._require_expected_state_version_locked(
                    expected_state_version
                )
                streams = self._normalize_transfer_streams_locked(
                    transfer_streams
                )
                explanation = self._explain_actions_locked(
                    plan.name,
                    (*plan.reclaim, *plan.enter, *plan.exit),
                    plan.reservations,
                    reservation_after_actions=len(plan.reclaim),
                )
                if not explanation.feasible:
                    raise ResidencyPlanError(
                        explanation.reason
                        or f"Plan {plan.name!r} is infeasible"
                    )
                failed_action: ResidencyAction | None = None
                failed_action_index: int | None = None
                try:
                    for action_index, action in enumerate(plan.reclaim):
                        failed_action = action
                        failed_action_index = action_index
                        try:
                            transitions.append(
                                self._execute_action_locked(
                                    action,
                                    stream=self._stream_for_action(
                                        action, streams
                                    ),
                                )
                            )
                        except BaseException as error:
                            _raise_plan_execution_failure(
                                error,
                                plan_name=plan.name,
                                phase="reclaim",
                                transitions=tuple(transitions),
                                started_at_ns=started,
                                failed_action=failed_action,
                                failed_action_index=failed_action_index,
                            )
                        failed_action = None
                        failed_action_index = None
                    for reservation_index, reservation in enumerate(
                        plan.reservations
                    ):
                        try:
                            tokens.append(
                                self._reserve_locked(
                                    reservation.location,
                                    reservation.nbytes,
                                    reservation.label,
                                )
                            )
                        except BaseException as error:
                            _raise_plan_execution_failure(
                                error,
                                plan_name=plan.name,
                                phase="reserve",
                                transitions=tuple(transitions),
                                started_at_ns=started,
                                failed_action=None,
                                failed_action_index=None,
                                failed_reservation=reservation,
                                failed_reservation_index=reservation_index,
                            )
                    for action_index, action in enumerate(plan.enter):
                        failed_action = action
                        failed_action_index = action_index
                        transitions.append(
                            self._execute_action_locked(
                                action,
                                stream=self._stream_for_action(action, streams),
                            )
                        )
                        failed_action = None
                        failed_action_index = None
                except ResidencyPlanExecutionError:
                    for token in reversed(tokens):
                        self._release_reservation_locked(token, missing_ok=True)
                    raise
                except BaseException as error:
                    for token in reversed(tokens):
                        self._release_reservation_locked(token, missing_ok=True)
                    _raise_plan_execution_failure(
                        error,
                        plan_name=plan.name,
                        phase="enter",
                        transitions=tuple(transitions),
                        started_at_ns=started,
                        failed_action=failed_action,
                        failed_action_index=failed_action_index,
                    )
                return tuple(tokens), tuple(transitions), started
            finally:
                self._end_public_operation_locked()

    def _exit_scope(
        self,
        plan: ResidencyPlan,
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream],
        reservation_tokens: tuple[object, ...],
        enter_transitions: tuple[ResidencyTransitionReport, ...],
        started_at_ns: int,
    ) -> ResidencyPlanReport:
        with self._state_access():
            self._begin_public_operation_locked()
            transitions = list(enter_transitions)
            failed_action: ResidencyAction | None = None
            failed_action_index: int | None = None
            try:
                for action_index, action in enumerate(plan.exit):
                    failed_action = action
                    failed_action_index = action_index
                    transitions.append(
                        self._execute_action_locked(
                            action,
                            stream=self._stream_for_action(
                                action, transfer_streams
                            ),
                        )
                    )
                    failed_action = None
                    failed_action_index = None
            except BaseException as error:
                _raise_plan_execution_failure(
                    error,
                    plan_name=plan.name,
                    phase="exit",
                    transitions=tuple(transitions),
                    started_at_ns=started_at_ns,
                    failed_action=failed_action,
                    failed_action_index=failed_action_index,
                )
            finally:
                for token in reversed(reservation_tokens):
                    self._release_reservation_locked(token, missing_ok=True)
                self._end_public_operation_locked()
        return ResidencyPlanReport(
            plan_name=plan.name,
            transitions=tuple(transitions),
            started_at_ns=started_at_ns,
            completed_at_ns=time_ns(),
        )

    # Primitive transition execution.

    def _run_action(
        self,
        action: ResidencyAction,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> ResidencyTransitionReport:
        with self._state_access():
            self._begin_public_operation_locked()
            try:
                return self._execute_action_locked(action, stream=stream)
            finally:
                self._end_public_operation_locked()

    def _execute_action_locked(
        self,
        action: ResidencyAction,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> ResidencyTransitionReport:
        self._reap_completed_locked()
        started = time_ns()
        record = self._record_for_locked(action.handle)
        self._validate_action_locations(action)
        if stream is not None:
            if isinstance(action, EnsureResident):
                self._validate_stream(action.location, stream)
            elif isinstance(action, MoveResident):
                self._validate_stream(action.to, stream)
            else:
                raise ValueError(
                    "Only EnsureResident and MoveResident accept a CUDA "
                    "transfer stream"
                )
        spec = record.spec
        allocator_device = self._allocator_device_for_action(action, record)
        before = self._best_effort_allocator_metrics(allocator_device)
        source_location: ResidencyLocation | None = None
        destination: ResidencyLocation | None = None
        no_op = False
        reason: str | None = None
        source_value: TensorResident | None = None
        candidate: TensorResident | None = None
        source_item: _Materialization | None = None
        item: _Materialization | None = None

        if isinstance(action, EnsureResident):
            destination = action.location
            self._validate_location(destination)
            if destination in record.materializations:
                no_op = True
                reason = "destination already resident"
            else:
                if (
                    spec.replica_mode is ReplicaMode.EXCLUSIVE
                    and record.materializations
                ):
                    raise ResidencyOwnershipError(
                        "EnsureResident would replicate an EXCLUSIVE value; "
                        "use MoveResident"
                    )
                source_location, source_value, temporary = (
                    self._resolve_transfer_source_locked(record)
                )
                try:
                    if temporary and source_location == destination:
                        candidate = source_value
                        self._release_temporary_source_locked(
                            source_location,
                            spec.storage_nbytes,
                        )
                        temporary = False
                    else:
                        self._ensure_budget_locked(
                            destination,
                            spec.storage_nbytes,
                        )
                        candidate = self._copy_value(
                            source_value,
                            destination,
                            stream=stream,
                        )
                    self._install_locked(
                        action.handle,
                        record,
                        destination,
                        candidate,
                    )
                finally:
                    if temporary:
                        self._release_temporary_source_locked(
                            source_location,
                            spec.storage_nbytes,
                        )
        elif isinstance(action, MoveResident):
            destination = action.to
            self._validate_location(destination)
            source_location = self._select_source_location_locked(
                record,
                explicit=action.from_location,
                exclude=(destination if action.from_location is None else None),
            )
            if destination in record.materializations:
                if source_location is None or source_location == destination:
                    no_op = True
                    reason = "destination already owns the selected replica"
                else:
                    self._require_removable_locked(
                        action.handle,
                        source_location,
                        record.materializations[source_location],
                    )
                    self._remove_materialization_locked(
                        action.handle,
                        record,
                        source_location,
                    )
            else:
                if source_location is not None:
                    source_item = record.materializations[source_location]
                    self._require_removable_locked(
                        action.handle,
                        source_location,
                        source_item,
                    )
                    source_value = source_item.value
                    temporary = False
                    remove_source_materialization = True
                else:
                    source_location, source_value, temporary = (
                        self._resolve_transfer_source_locked(record)
                    )
                    remove_source_materialization = False
                try:
                    if temporary and source_location == destination:
                        candidate = source_value
                        self._release_temporary_source_locked(
                            source_location,
                            spec.storage_nbytes,
                        )
                        temporary = False
                    else:
                        self._ensure_budget_locked(
                            destination,
                            spec.storage_nbytes,
                        )
                        candidate = self._copy_value(
                            source_value,
                            destination,
                            stream=stream,
                        )
                    self._install_locked(
                        action.handle,
                        record,
                        destination,
                        candidate,
                        allow_exclusive_transition=True,
                    )
                    if remove_source_materialization:
                        self._remove_materialization_locked(
                            action.handle,
                            record,
                            cast(ResidencyLocation, source_location),
                        )
                finally:
                    if temporary:
                        self._release_temporary_source_locked(
                            source_location,
                            spec.storage_nbytes,
                        )
        elif isinstance(action, DropResident):
            source_location = action.location
            self._validate_location(source_location)
            item = record.materializations.get(source_location)
            if item is None:
                no_op = True
                reason = "materialization already absent"
            else:
                self._require_removable_locked(
                    action.handle,
                    source_location,
                    item,
                )
                if len(record.materializations) == 1 and record.source is None:
                    raise ResidencyOwnershipError(
                        "Cannot drop the final MUST_PRESERVE materialization; "
                        "discard the logical value"
                    )
                self._remove_materialization_locked(
                    action.handle,
                    record,
                    source_location,
                )
        elif isinstance(action, DiscardValue):
            for location, item in record.materializations.items():
                self._require_removable_locked(action.handle, location, item)
            source_location = None
            for location in tuple(record.materializations):
                self._remove_materialization_locked(
                    action.handle,
                    record,
                    location,
                )
            record.source = None
            record.source_location = None
            record.discarded = True
            self._bump_state_locked()
        else:
            raise TypeError(
                f"Unsupported residency action {type(action).__name__}"
            )

        # Do not let transition-local references make the post-action PyTorch
        # allocator sample report storage that the manager has already removed.
        source_value = None
        candidate = None
        source_item = None
        item = None
        after = self._best_effort_allocator_metrics(allocator_device)
        report = ResidencyTransitionReport(
            action=action,
            no_op=no_op,
            source=source_location,
            destination=destination,
            logical_nbytes=spec.logical_nbytes,
            storage_nbytes=spec.storage_nbytes,
            started_at_ns=started,
            completed_at_ns=time_ns(),
            allocator_device=allocator_device,
            allocator_allocated_bytes_before=before[0],
            allocator_reserved_bytes_before=before[1],
            allocator_allocated_bytes_after=after[0],
            allocator_reserved_bytes_after=after[1],
            reason=reason,
        )
        if self._trace_enabled:
            self._trace.append(report)
        return report

    # State mutation and transfer helpers.

    def _install_locked(
        self,
        handle: ResidencyHandle[Any],
        record: _ValueRecord,
        location: ResidencyLocation,
        value: TensorResident,
        *,
        allow_exclusive_transition: bool = False,
    ) -> None:
        if location in record.materializations:
            raise RuntimeError("Materialization is already installed")
        if (
            record.spec.replica_mode is ReplicaMode.EXCLUSIVE
            and record.materializations
            and not allow_exclusive_transition
        ):
            raise ResidencyOwnershipError(
                "An EXCLUSIVE value cannot have simultaneous steady replicas"
            )
        self._validate_materialization(record.spec, value, location)
        self._ensure_budget_locked(location, record.spec.storage_nbytes)
        keys = _storage_keys(value)
        collisions = {
            key: self._storage_owners[key]
            for key in keys
            if key in self._storage_owners
        }
        if collisions:
            raise ResidencyOwnershipError(
                "Managed materializations must own independent backing storage; "
                f"aliases collide with {tuple(collisions.values())!r}"
            )
        state = self._ensure_location_locked(location)
        record.materializations[location] = _Materialization(
            value=value,
            logical_nbytes=record.spec.logical_nbytes,
            storage_nbytes=value.storage_nbytes,
            charged_nbytes=record.spec.storage_nbytes,
        )
        for key in keys:
            self._storage_owners[key] = (handle, location)
        state.used_bytes += record.spec.storage_nbytes
        state.peak_used_bytes = max(state.peak_used_bytes, state.used_bytes)
        state.peak_charged_bytes = max(
            state.peak_charged_bytes,
            state.used_bytes + state.reserved_bytes,
        )
        self._bump_state_locked()

    def _remove_materialization_locked(
        self,
        handle: ResidencyHandle[Any],
        record: _ValueRecord,
        location: ResidencyLocation,
    ) -> None:
        item = record.materializations.pop(location)
        for key in _storage_keys(item.value):
            owner = self._storage_owners.get(key)
            if owner == (handle, location):
                del self._storage_owners[key]
        self._locations[location].used_bytes -= item.charged_nbytes
        self._bump_state_locked()

    def _require_removable_locked(
        self,
        handle: ResidencyHandle[Any],
        location: ResidencyLocation,
        item: _Materialization,
    ) -> None:
        if item.use_tokens or item.hold_tokens or item.pending_tokens:
            raise ResidencyInUseError(
                f"Value {handle.handle_id!r} at {location} is "
                "protected: "
                f"active_reads={len(item.use_tokens)}, "
                f"holds={len(item.hold_tokens)}, "
                f"pending_events={len(item.pending_tokens)}"
            )

    def _resolve_transfer_source_locked(
        self,
        record: _ValueRecord,
    ) -> tuple[ResidencyLocation, TensorResident, bool]:
        source_location = self._select_source_location_locked(record)
        if source_location is not None:
            return (
                source_location,
                record.materializations[source_location].value,
                False,
            )
        if record.source is None or record.source_location is None:
            raise ResidencyUnavailableError(
                "Value has no materialization or reconstruction source"
            )
        source_location = record.source_location
        self._ensure_budget_locked(source_location, record.spec.storage_nbytes)
        source_state = self._locations[source_location]
        source_state.reserved_bytes += record.spec.storage_nbytes
        source_state.peak_charged_bytes = max(
            source_state.peak_charged_bytes,
            source_state.used_bytes + source_state.reserved_bytes,
        )
        try:
            self._source_callback_active.set()
            try:
                value = record.source.load()
            finally:
                self._source_callback_active.clear()
            self._validate_materialization(record.spec, value, source_location)
        except BaseException:
            self._locations[
                source_location
            ].reserved_bytes -= record.spec.storage_nbytes
            raise
        return source_location, value, True

    def _release_temporary_source_locked(
        self,
        location: ResidencyLocation,
        nbytes: int,
    ) -> None:
        self._locations[location].reserved_bytes -= nbytes

    def _select_source_location_locked(
        self,
        record: _ValueRecord,
        *,
        explicit: ResidencyLocation | None = None,
        exclude: ResidencyLocation | None = None,
    ) -> ResidencyLocation | None:
        if explicit is not None:
            self._validate_location(explicit)
            if explicit not in record.materializations:
                raise ResidencyUnavailableError(
                    f"Explicit source {explicit} is not resident"
                )
            return explicit
        candidates = [
            location
            for location in record.materializations
            if location != exclude
        ]
        if not candidates:
            return None
        order = {PAGEABLE_HOST: 0, PINNED_HOST: 1}
        return min(
            candidates,
            key=lambda location: (
                order.get(location, 2),
                -1 if location.device.index is None else location.device.index,
            ),
        )

    def _copy_value(
        self,
        value: TensorResident,
        destination: ResidencyLocation,
        *,
        stream: torch.cuda.Stream | None,
    ) -> TensorResident:
        if destination.kind == "pageable-host":
            if stream is not None:
                raise ValueError("Host transitions do not accept a CUDA stream")
            return value.to("cpu", copy=True, non_blocking=False)
        if destination.kind == "pinned-host":
            if stream is not None:
                raise ValueError("Host transitions do not accept a CUDA stream")
            return value.pin_memory(copy=True)
        if stream is not None:
            self._validate_stream(destination, stream)
        with torch.cuda.device(destination.device):
            copy_stream = (
                torch.cuda.current_stream(destination.device)
                if stream is None
                else stream
            )
            non_blocking = value.device.type == "cuda" or value.is_pinned
            try:
                with torch.cuda.stream(copy_stream):
                    candidate = value.to(
                        destination.device,
                        copy=True,
                        non_blocking=non_blocking,
                    )
                    completion = torch.cuda.Event()
                    completion.record(copy_stream)
                completion.synchronize()
            except BaseException as error:
                try:
                    copy_stream.synchronize()
                except BaseException as cleanup_error:
                    error.add_note(
                        "CUDA copy cleanup synchronization also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise
        return candidate

    def _validate_materialization(
        self,
        spec: ResidencyValueSpec[Any],
        value: TensorResident,
        location: ResidencyLocation,
    ) -> None:
        if not isinstance(value, spec.value_type):
            raise ResidencyMaterializationError(
                f"Expected {spec.value_type.__name__}, got {type(value).__name__}"
            )
        self._validate_value_location(value, location)
        if value.nbytes != spec.logical_nbytes:
            raise ResidencyMaterializationError(
                "Logical payload size mismatch: "
                f"expected={spec.logical_nbytes}, actual={value.nbytes}"
            )
        actual_storage = max(value.nbytes, value.storage_nbytes)
        if actual_storage > spec.storage_nbytes:
            raise ResidencyMaterializationError(
                "Backing-storage charge mismatch: registered charge exceeded; "
                f"charge={spec.storage_nbytes}, actual={actual_storage}"
            )

    @staticmethod
    def _validate_value_location(
        value: TensorResident,
        location: ResidencyLocation,
    ) -> None:
        if value.device != location.device:
            raise ResidencyMaterializationError(
                f"Value device {value.device} does not match {location}"
            )
        if value.device.type == "cpu":
            if not _cpu_pinning_is_uniform(value):
                raise ResidencyMaterializationError(
                    "CPU materialization mixes pageable and pinned tensor storage"
                )
            if location.kind == "pinned-host" and not value.is_pinned:
                raise ResidencyMaterializationError(
                    "Pinned-host materialization is not fully pinned"
                )
            if location.kind == "pageable-host" and value.is_pinned:
                raise ResidencyMaterializationError(
                    "Pageable-host materialization unexpectedly uses pinned storage"
                )

    def _infer_location(self, value: TensorResident) -> ResidencyLocation:
        if value.device.type == "cuda":
            from fhelium.residency.location import cuda_location

            return cuda_location(value.device)
        if not _cpu_pinning_is_uniform(value):
            raise ResidencyMaterializationError(
                "Cannot infer location for mixed pageable/pinned CPU tensors"
            )
        return PINNED_HOST if value.is_pinned else PAGEABLE_HOST

    # Capacity, snapshots, and event accounting.

    def _ensure_budget_locked(
        self,
        location: ResidencyLocation,
        incoming_bytes: int,
    ) -> None:
        self._validate_location(location)
        state = self._locations.get(location)
        if state is None or state.budget_bytes is None:
            return
        if incoming_bytes > (
            state.budget_bytes - state.used_bytes - state.reserved_bytes
        ):
            raise ResidencyBudgetError(
                location=location.name,
                budget_bytes=state.budget_bytes,
                used_bytes=state.used_bytes,
                reserved_bytes=state.reserved_bytes,
                requested_bytes=incoming_bytes,
            )

    def _reserve_locked(
        self,
        location: ResidencyLocation,
        nbytes: int,
        label: str,
    ) -> object:
        self._ensure_budget_locked(location, nbytes)
        state = self._ensure_location_locked(location)
        token = object()
        self._reservations[token] = _ReservationRecord(location, nbytes, label)
        state.reserved_bytes += nbytes
        state.peak_charged_bytes = max(
            state.peak_charged_bytes,
            state.used_bytes + state.reserved_bytes,
        )
        self._bump_state_locked()
        return token

    def _move_use_to_pending_locked(
        self,
        token: object,
        record: _LeaseRecord,
    ) -> None:
        for handle in record.handles:
            item = self._record_for_locked(handle).materializations[
                record.location
            ]
            item.use_tokens.remove(token)
            item.pending_tokens.add(token)
        self._bump_state_locked()

    def _remove_active_use_locked(
        self,
        token: object,
        record: _LeaseRecord,
    ) -> None:
        for handle in record.handles:
            item = self._record_for_locked(handle).materializations[
                record.location
            ]
            item.use_tokens.remove(token)
        self._bump_state_locked()

    def _release_pending_locked(
        self,
        token: object,
        record: _LeaseRecord,
    ) -> None:
        for handle in record.handles:
            item = self._record_for_locked(handle).materializations[
                record.location
            ]
            item.pending_tokens.remove(token)
        del self._leases[token]
        self._bump_state_locked()

    def _reap_completed_locked(self) -> None:
        completed = [
            (token, record)
            for token, record in self._leases.items()
            if record.state == "pending"
            and all(event.query() for event in record.events)
        ]
        for token, record in completed:
            self._release_pending_locked(token, record)

    def _synchronize_pending_locked(self) -> None:
        for token, record in tuple(self._leases.items()):
            if record.state != "pending":
                continue
            for event in record.events:
                event.synchronize()
            self._release_pending_locked(token, record)

    def _allocator_device_for_action(
        self,
        action: ResidencyAction,
        record: _ValueRecord,
    ) -> torch.device | None:
        locations: list[ResidencyLocation] = []
        if isinstance(action, EnsureResident):
            locations.append(action.location)
        elif isinstance(action, MoveResident):
            locations.append(action.to)
            if action.from_location is not None:
                locations.append(action.from_location)
            else:
                locations.extend(record.materializations)
        elif isinstance(action, DropResident):
            locations.append(action.location)
        elif isinstance(action, DiscardValue):
            locations.extend(record.materializations)
        cuda_locations = [item for item in locations if item.kind == "cuda"]
        return None if not cuda_locations else cuda_locations[0].device

    @staticmethod
    def _allocator_metrics(
        device: torch.device | None,
    ) -> tuple[int | None, int | None]:
        if device is None or not torch.cuda.is_available():
            return None, None
        return (
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
        )

    def _best_effort_allocator_metrics(
        self,
        device: torch.device | None,
    ) -> tuple[int | None, int | None]:
        """Sample optional allocator telemetry without failing residency work."""

        try:
            return self._allocator_metrics(device)
        except Exception:
            return None, None

    # Dry-run planning.

    def _explain_actions_locked(
        self,
        name: str,
        actions: Sequence[ResidencyAction],
        reservations: Sequence[MemoryReservation],
        *,
        reservation_after_actions: int = 0,
    ) -> ResidencyPlanExplanation:
        simulation = _SimulationState(
            locations={
                handle: set(record.materializations)
                for handle, record in self._records.items()
            },
            discarded={
                handle
                for handle, record in self._records.items()
                if record.discarded
            },
            budgets={
                location: state.budget_bytes
                for location, state in self._locations.items()
            },
            used={
                location: state.used_bytes
                for location, state in self._locations.items()
            },
            reserved={
                location: state.reserved_bytes
                for location, state in self._locations.items()
            },
            peaks={
                location: state.used_bytes + state.reserved_bytes
                for location, state in self._locations.items()
            },
        )
        explanations: list[ResidencyActionExplanation] = []
        feasible = True
        failure: str | None = None
        if not 0 <= reservation_after_actions <= len(actions):
            raise RuntimeError("Invalid simulated reservation insertion index")

        def admit_reservations() -> None:
            nonlocal feasible, failure
            for reservation in reservations:
                location_error = self._prepare_simulated_location(
                    simulation,
                    reservation.location,
                )
                if location_error is not None:
                    feasible = False
                    failure = location_error
                    return
                simulation.reserved[reservation.location] += reservation.nbytes
                total = (
                    simulation.used[reservation.location]
                    + simulation.reserved[reservation.location]
                )
                simulation.peaks[reservation.location] = max(
                    simulation.peaks[reservation.location], total
                )
                budget = simulation.budgets[reservation.location]
                if budget is not None and total > budget:
                    feasible = False
                    failure = (
                        f"Reservation {reservation.label!r} exceeds budget at "
                        f"{reservation.location}"
                    )
                    return

        for action_index in range(len(actions) + 1):
            if action_index == reservation_after_actions:
                admit_reservations()
                if not feasible:
                    break
            if action_index < len(actions):
                action = actions[action_index]
                explanation, action_failure = self._simulate_action_locked(
                    action,
                    simulation,
                )
                explanations.append(explanation)
                if action_failure is not None:
                    feasible = False
                    failure = action_failure
                    break
        return ResidencyPlanExplanation(
            plan_name=name,
            actions=tuple(explanations),
            reservations=tuple(reservations),
            predicted_peak_bytes=tuple(simulation.peaks.items()),
            feasible=feasible,
            reason=failure,
        )

    def _simulate_action_locked(
        self,
        action: ResidencyAction,
        simulation: _SimulationState,
    ) -> tuple[ResidencyActionExplanation, str | None]:
        try:
            record = self._record_for_locked(action.handle)
        except ResidencyHandleError as error:
            return self._failed_explanation(action, str(error)), str(error)
        spec = record.spec
        locations = simulation.locations[action.handle]
        source: ResidencyLocation | None = None
        destination: ResidencyLocation | None = None
        no_op = False
        reason: str | None = None

        def add(
            location: ResidencyLocation, *, temporary: bool = False
        ) -> str | None:
            location_error = self._prepare_simulated_location(
                simulation,
                location,
            )
            if location_error is not None:
                return location_error
            total = (
                simulation.used[location]
                + simulation.reserved[location]
                + spec.storage_nbytes
            )
            simulation.peaks[location] = max(simulation.peaks[location], total)
            budget = simulation.budgets[location]
            if budget is not None and total > budget:
                return f"Transition exceeds budget at {location}"
            if not temporary:
                simulation.used[location] += spec.storage_nbytes
                locations.add(location)
            return None

        def remove(location: ResidencyLocation) -> None:
            simulation.used[location] -= spec.storage_nbytes
            locations.remove(location)

        if action.handle in simulation.discarded:
            reason = f"Value {action.handle.handle_id!r} is discarded"
            return self._failed_explanation(action, reason), reason
        if isinstance(action, EnsureResident):
            destination = action.location
            reason = self._prepare_simulated_location(
                simulation,
                destination,
            )
            if reason is None and destination in locations:
                no_op = True
                reason = "destination already resident"
            elif (
                reason is None
                and spec.replica_mode is ReplicaMode.EXCLUSIVE
                and locations
            ):
                reason = "EnsureResident would replicate an EXCLUSIVE value"
            elif reason is None:
                source = self._simulated_source(record, locations)
                if source is None:
                    reason = "value has no materialization or source"
                else:
                    if not locations:
                        failure = add(source, temporary=True)
                        if failure:
                            reason = failure
                    if reason is None:
                        reason = add(destination)
        elif isinstance(action, MoveResident):
            destination = action.to
            reason = self._prepare_simulated_location(
                simulation,
                destination,
            )
            if reason is None and action.from_location is not None:
                source = action.from_location
                reason = self._prepare_simulated_location(simulation, source)
                if reason is None and source not in locations:
                    reason = f"explicit source {source} is not resident"
            elif reason is None and locations:
                alternatives = locations - {destination}
                source = (
                    self._sort_locations(alternatives)[0]
                    if alternatives
                    else None
                )
            else:
                source = (
                    record.source_location
                    if record.source is not None
                    else None
                )
            if reason is None and destination in locations:
                if source is None or source == destination:
                    no_op = True
                    reason = "destination already owns the selected replica"
                elif source is not None:
                    if self._simulated_protected(action.handle, source):
                        reason = f"source {source} is protected"
                    else:
                        remove(source)
            elif reason is None and source is None:
                reason = "value has no materialization or source"
            elif reason is None and source is not None:
                source_was_resident = source in locations
                if source_was_resident and self._simulated_protected(
                    action.handle, source
                ):
                    reason = f"source {source} is protected"
                else:
                    if not source_was_resident:
                        failure = add(source, temporary=True)
                        if failure:
                            reason = failure
                    if reason is None:
                        failure = add(destination)
                        if failure:
                            reason = failure
                    if reason is None and source_was_resident:
                        remove(source)
        elif isinstance(action, DropResident):
            source = action.location
            reason = self._prepare_simulated_location(simulation, source)
            if reason is None and source not in locations:
                no_op = True
                reason = "materialization already absent"
            elif reason is None and self._simulated_protected(
                action.handle,
                source,
            ):
                reason = f"materialization at {source} is protected"
            elif (
                reason is None and len(locations) == 1 and record.source is None
            ):
                reason = "cannot drop final MUST_PRESERVE materialization"
            elif reason is None:
                remove(source)
        else:
            protected = [
                location
                for location in locations
                if self._simulated_protected(action.handle, location)
            ]
            if protected:
                reason = (
                    f"discarded value has protected locations {protected!r}"
                )
            else:
                for location in tuple(locations):
                    remove(location)
                simulation.discarded.add(action.handle)
        executable = reason is None or no_op
        explanation = ResidencyActionExplanation(
            action=action,
            executable=executable,
            no_op=no_op,
            source=source,
            destination=destination,
            logical_nbytes=spec.logical_nbytes,
            storage_nbytes=spec.storage_nbytes,
            reason=reason,
        )
        return explanation, None if executable else reason

    def _prepare_simulated_location(
        self,
        simulation: _SimulationState,
        location: ResidencyLocation,
    ) -> str | None:
        """Add a location to dry-run state without mutating manager state."""

        try:
            self._validate_location(location)
        except (TypeError, ValueError) as error:
            return str(error)
        if location in simulation.used:
            return None
        state = self._locations.get(location)
        simulation.budgets[location] = (
            None if state is None else state.budget_bytes
        )
        simulation.used[location] = 0 if state is None else state.used_bytes
        simulation.reserved[location] = (
            0 if state is None else state.reserved_bytes
        )
        simulation.peaks[location] = (
            0 if state is None else state.used_bytes + state.reserved_bytes
        )
        return None

    def _simulated_source(
        self,
        record: _ValueRecord,
        locations: set[ResidencyLocation],
    ) -> ResidencyLocation | None:
        if locations:
            return self._sort_locations(locations)[0]
        return record.source_location if record.source is not None else None

    def _simulated_protected(
        self,
        handle: ResidencyHandle[Any],
        location: ResidencyLocation,
    ) -> bool:
        record = self._records[handle]
        item = record.materializations.get(location)
        if item is None:
            return False
        if item.use_tokens or item.hold_tokens:
            return True
        for token in item.pending_tokens:
            lease = self._leases.get(token)
            if lease is None or lease.state != "pending":
                return True
            try:
                if not all(event.query() for event in lease.events):
                    return True
            except BaseException:
                return True
        return False

    def _failed_explanation(
        self,
        action: ResidencyAction,
        reason: str,
    ) -> ResidencyActionExplanation:
        return ResidencyActionExplanation(
            action=action,
            executable=False,
            no_op=False,
            source=None,
            destination=None,
            logical_nbytes=0,
            storage_nbytes=0,
            reason=reason,
        )

    # Handle and control-state validation.

    def _new_handle(
        self,
        *,
        value_type: type[ValueT],
    ) -> ResidencyHandle[ValueT]:
        handle_id = str(uuid4())
        while handle_id in self._issued_handle_ids:
            handle_id = str(uuid4())
        self._issued_handle_ids.add(handle_id)
        return ResidencyHandle(
            manager_id=self.manager_id,
            handle_id=handle_id,
            value_type=value_type,
        )

    def _record_for_locked(
        self,
        handle: ResidencyHandle[ValueT],
    ) -> _ValueRecord:
        if not isinstance(handle, ResidencyHandle):
            raise ResidencyHandleError("Expected a ResidencyHandle")
        if handle.manager_id != self.manager_id:
            raise ResidencyHandleError(
                f"Handle belongs to manager {handle.manager_id!r}, not "
                f"{self.manager_id!r}"
            )
        record = self._records.get(handle)
        if record is None:
            raise ResidencyHandleError("Residency handle is unknown")
        if record.discarded:
            raise ResidencyHandleError(
                f"Residency value {handle.handle_id!r} has been discarded"
            )
        return record

    def _validate_location(self, location: ResidencyLocation) -> None:
        if not isinstance(location, ResidencyLocation):
            raise TypeError(
                "ResidencyManager locations must be ResidencyLocation values"
            )
        if location.kind != "cuda":
            return
        if not torch.cuda.is_available():
            raise ValueError("CUDA residency locations require available CUDA")
        index = cast(int, location.device.index)
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(
                f"Residency location {location} is not available in this process"
            )

    def _validate_action_locations(self, action: ResidencyAction) -> None:
        """Validate action endpoints before allocator observation or mutation."""

        if isinstance(action, EnsureResident):
            self._validate_location(action.location)
        elif isinstance(action, MoveResident):
            self._validate_location(action.to)
            if action.from_location is not None:
                self._validate_location(action.from_location)
        elif isinstance(action, DropResident):
            self._validate_location(action.location)

    def _require_expected_state_version_locked(
        self,
        expected_state_version: int | None,
    ) -> None:
        expected = _optional_nonnegative_count(
            expected_state_version,
            what="Residency expected_state_version",
        )
        if expected is not None and expected != self._state_version:
            raise ResidencyStaleStateError(
                expected_version=expected,
                actual_version=self._state_version,
            )

    def _normalize_transfer_streams_locked(
        self,
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream] | None,
    ) -> dict[ResidencyLocation, torch.cuda.Stream]:
        if transfer_streams is None:
            return {}
        if not isinstance(transfer_streams, Mapping):
            raise TypeError("Residency transfer_streams must be a mapping")
        normalized: dict[ResidencyLocation, torch.cuda.Stream] = {}
        for location, stream in transfer_streams.items():
            self._validate_location(location)
            self._validate_stream(location, stream)
            normalized[location] = stream
        return normalized

    @staticmethod
    def _stream_for_action(
        action: ResidencyAction,
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream],
    ) -> torch.cuda.Stream | None:
        if isinstance(action, EnsureResident):
            return transfer_streams.get(action.location)
        if isinstance(action, MoveResident):
            return transfer_streams.get(action.to)
        return None

    def _bump_state_locked(self) -> None:
        self._state_version += 1

    def _ensure_location_locked(
        self,
        location: ResidencyLocation,
    ) -> _LocationState:
        self._validate_location(location)
        state = self._locations.get(location)
        if state is None:
            state = _LocationState(budget_bytes=None)
            self._locations[location] = state
            self._bump_state_locked()
        return state

    @contextmanager
    def _state_access(self) -> Iterator[None]:
        """Acquire state while excluding access during source callbacks.

        A timed acquisition loop is deliberate: a different thread may begin
        waiting before a source callback becomes active. It must observe the
        manager-wide callback access guard and fail rather than deadlock when
        the callback joins it. The guard applies to every thread accessing this
        manager, not only to direct same-thread reentry.
        """

        while True:
            if self._source_callback_active.is_set():
                raise ResidencyReentrancyError(
                    "Residency manager state access is rejected while a "
                    "source callback is active"
                )
            if self._lock.acquire(timeout=0.01):
                break
        try:
            yield
        finally:
            self._lock.release()

    def _require_available(self) -> None:
        with self._state_access():
            self._require_available_locked()

    def _require_available_locked(self) -> None:
        if self._closed:
            raise ResidencyClosedError("ResidencyManager is closed")

    def _require_not_reentrant_locked(self) -> None:
        if self._transition_thread == get_ident():
            raise ResidencyReentrancyError(
                "Residency source or user callback re-entered an active transition"
            )

    def _begin_public_operation_locked(self) -> None:
        self._require_available_locked()
        self._require_not_reentrant_locked()
        self._transition_thread = get_ident()
        self._reap_completed_locked()

    def _end_public_operation_locked(self) -> None:
        self._transition_thread = None

    @staticmethod
    def _validate_stream(
        location: ResidencyLocation,
        stream: torch.cuda.Stream,
    ) -> None:
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError(
                "Residency consumer stream must be a torch.cuda.Stream"
            )
        if location.kind != "cuda":
            raise ValueError("CUDA streams require a CUDA residency location")
        if torch.device(stream.device) != location.device:
            raise ValueError(
                f"Stream device {stream.device} does not match {location.device}"
            )

    @staticmethod
    def _sort_locations(
        locations: Iterable[ResidencyLocation],
    ) -> tuple[ResidencyLocation, ...]:
        order = {PAGEABLE_HOST: 0, PINNED_HOST: 1}
        return tuple(
            sorted(
                locations,
                key=lambda location: (
                    order.get(location, 2),
                    -1
                    if location.device.index is None
                    else location.device.index,
                ),
            )
        )


class ResidencyScope:
    """Single-use execution scope for one immutable residency plan.

    ``exit_error`` retains a structured exit failure when a body exception
    remains the primary propagated exception.
    """

    def __init__(
        self,
        manager: ResidencyManager,
        plan: ResidencyPlan,
        *,
        transfer_streams: Mapping[ResidencyLocation, torch.cuda.Stream]
        | None = None,
        expected_state_version: int | None = None,
    ) -> None:
        if not isinstance(manager, ResidencyManager):
            raise TypeError("ResidencyScope manager must be a ResidencyManager")
        if not isinstance(plan, ResidencyPlan):
            raise TypeError("ResidencyScope plan must be a ResidencyPlan")
        if transfer_streams is not None and not isinstance(
            transfer_streams, Mapping
        ):
            raise TypeError("Residency transfer_streams must be a mapping")
        self._manager = manager
        self._plan = plan
        self._transfer_streams = dict(transfer_streams or {})
        self._expected_state_version = _optional_nonnegative_count(
            expected_state_version,
            what="Residency expected_state_version",
        )
        self.report: ResidencyPlanReport | None = None
        self.exit_error: ResidencyPlanExecutionError | None = None
        self._reservation_tokens: tuple[object, ...] = ()
        self._enter_transitions: tuple[ResidencyTransitionReport, ...] = ()
        self._started_at_ns = 0
        self._entered = False
        self._closed = False
        self._state_lock = RLock()
        self._finalizer: Any | None = None

    @property
    def manager(self) -> ResidencyManager:
        """Manager authority captured when this scope was constructed."""

        return self._manager

    @property
    def plan(self) -> ResidencyPlan:
        """Immutable plan captured when this scope was constructed."""

        return self._plan

    @property
    def expected_state_version(self) -> int | None:
        """State-version precondition captured at construction."""

        return self._expected_state_version

    def __enter__(self) -> ResidencyScope:
        with self._state_lock:
            if self._entered:
                raise RuntimeError("ResidencyScope cannot be entered twice")
            self._entered = True
            try:
                (
                    self._reservation_tokens,
                    self._enter_transitions,
                    self._started_at_ns,
                ) = self._manager._enter_scope(
                    self._plan,
                    self._transfer_streams,
                    self._expected_state_version,
                )
            except BaseException:
                self._closed = True
                raise
            self._finalizer = finalize(
                self,
                _abandon_scope_lifetime,
                self._manager,
                self._reservation_tokens,
                self._plan.name,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is not None:
                cleanup_notes = " ".join(
                    getattr(cleanup_error, "__notes__", ())
                )
                exc_value.add_note(
                    f"Residency plan {self._plan.name!r} exit failed: "
                    f"{cleanup_error}"
                    + (f" {cleanup_notes}" if cleanup_notes else "")
                )
                return
            raise

    def close(self) -> None:
        """Execute exit actions and release reservations idempotently."""

        with self._state_lock:
            if self._closed:
                return
            if not self._entered:
                raise RuntimeError(
                    "ResidencyScope must be entered before close"
                )
            try:
                self.report = self._manager._exit_scope(
                    self._plan,
                    self._transfer_streams,
                    self._reservation_tokens,
                    self._enter_transitions,
                    self._started_at_ns,
                )
            except ResidencyPlanExecutionError as error:
                self.exit_error = error
                raise
            finally:
                self._closed = True
                if self._finalizer is not None:
                    self._finalizer.detach()


def _abandon_scope_lifetime(
    manager: ResidencyManager,
    reservation_tokens: tuple[object, ...],
    plan_name: str,
) -> None:
    """Release headroom but never guess abandoned scope exit transitions."""

    for token in reversed(reservation_tokens):
        try:
            manager._release_reservation(token, missing_ok=True)
        except BaseException:
            pass
    warn(
        f"An active ResidencyScope for plan {plan_name!r} was abandoned. "
        "Its reservations were released where possible; exit transitions "
        "were not executed.",
        ResourceWarning,
        stacklevel=2,
    )


def _nonnegative_count(value: int, *, what: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, not bool")
    try:
        normalized = index(value)
    except TypeError as error:
        raise TypeError(f"{what} must be an integer") from error
    if normalized < 0:
        raise ValueError(f"{what} must be non-negative")
    return normalized


def _optional_nonnegative_count(
    value: int | None,
    *,
    what: str,
) -> int | None:
    return None if value is None else _nonnegative_count(value, what=what)


def _new_manager_id() -> str:
    """Issue a process-unique opaque manager authority identity."""

    with _MANAGER_ID_LOCK:
        manager_id = str(uuid4())
        while manager_id in _ISSUED_MANAGER_IDS:
            manager_id = str(uuid4())
        _ISSUED_MANAGER_IDS.add(manager_id)
        return manager_id


def _unique_handles(
    handles: Iterable[ResidencyHandle[Any]],
) -> tuple[ResidencyHandle[Any], ...]:
    try:
        iterator = iter(handles)
    except TypeError as error:
        raise TypeError(
            "Residency lifetimes require an iterable of ResidencyHandle values"
        ) from error
    unique: list[ResidencyHandle[Any]] = []
    seen: set[ResidencyHandle[Any]] = set()
    for handle in iterator:
        if not isinstance(handle, ResidencyHandle):
            raise TypeError(
                "Residency lifetimes require ResidencyHandle values"
            )
        if handle not in seen:
            unique.append(handle)
            seen.add(handle)
    if not unique:
        raise ValueError("Residency lifetimes require at least one handle")
    return tuple(unique)


def _normalize_actions(
    actions: Iterable[ResidencyAction],
) -> tuple[ResidencyAction, ...]:
    try:
        normalized = tuple(actions)
    except TypeError as error:
        raise TypeError(
            "ResidencyManager actions must be an iterable of residency actions"
        ) from error
    action_types = (EnsureResident, MoveResident, DropResident, DiscardValue)
    if any(not isinstance(action, action_types) for action in normalized):
        raise TypeError(
            "ResidencyManager actions must contain residency action objects"
        )
    return normalized


def _require_nonempty_string(value: object, *, what: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if not value.strip():
        raise ValueError(f"{what} must be non-empty")


def _raise_plan_execution_failure(
    error: BaseException,
    *,
    plan_name: str,
    phase: Literal["execute", "reclaim", "reserve", "enter", "exit"],
    transitions: tuple[ResidencyTransitionReport, ...],
    started_at_ns: int,
    failed_action: ResidencyAction | None,
    failed_action_index: int | None,
    failed_reservation: MemoryReservation | None = None,
    failed_reservation_index: int | None = None,
) -> NoReturn:
    """Raise one structured plan failure without rolling back transitions."""

    if not isinstance(error, Exception):
        if transitions:
            error.add_note(
                f"Residency plan {plan_name!r} completed {len(transitions)} "
                "transition(s) before interruption; completed transitions "
                "remain valid and are not rolled back."
            )
        raise error
    partial_report = ResidencyPlanReport(
        plan_name=plan_name,
        transitions=transitions,
        started_at_ns=started_at_ns,
        completed_at_ns=time_ns(),
    )
    raise ResidencyPlanExecutionError(
        plan_name=plan_name,
        phase=phase,
        partial_report=partial_report,
        failed_action=failed_action,
        failed_action_index=failed_action_index,
        failed_reservation=failed_reservation,
        failed_reservation_index=failed_reservation_index,
        detail=str(error),
    ) from error


__all__ = ["ResidencyManager", "ResidencyScope"]
