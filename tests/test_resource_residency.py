from __future__ import annotations

from collections.abc import Callable
from gc import collect
from threading import Event, Thread
from typing import Any, Self, assert_type, get_type_hints
from weakref import ref

import pytest
import torch

from fhelium import Plaintext, TensorResident, errors as errors_module
from fhelium.errors import (
    ResidencyBudgetError,
    ResidencyClosedError,
    ResidencyHandleError,
    ResidencyInUseError,
    ResidencyLifetimeClosedError,
    ResidencyMaterializationError,
    ResidencyOwnershipError,
    ResidencyPlanError,
    ResidencyPlanExecutionError,
    ResidencyReentrancyError,
    ResidencyStaleStateError,
    ResidencyUnavailableError,
)
from fhelium.residency import (
    PAGEABLE_HOST,
    PINNED_HOST,
    DiscardValue,
    DropResident,
    EnsureResident,
    MemoryReservation,
    MoveResident,
    Recoverability,
    ReplicaMode,
    ResidencyHandle,
    ResidencyLocation,
    ResidencyLocationSnapshot,
    ResidencyManager,
    ResidencyPlan,
    ResidencyScope,
    ResidencyValueSpec,
    cuda_location,
)


def _plaintext(
    values: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    *,
    device: torch.device | str = "cpu",
) -> Plaintext:
    return Plaintext(
        message=torch.tensor(values, dtype=torch.float64, device=device),
        level=0,
        scale=16.0,
    )


def _spec(
    value: TensorResident,
    *,
    replica_mode: ReplicaMode = ReplicaMode.REPLICABLE,
) -> ResidencyValueSpec[Any]:
    return ResidencyValueSpec(
        value_type=type(value),
        logical_nbytes=value.nbytes,
        storage_nbytes=max(value.nbytes, value.storage_nbytes),
        replica_mode=replica_mode,
        recoverability=Recoverability.RECONSTRUCTIBLE,
    )


def _value_snapshot(
    manager: ResidencyManager,
    handle: ResidencyHandle[Any],
) -> Any:
    return next(
        item for item in manager.snapshot().values if item.handle == handle
    )


def _location_snapshot(
    manager: ResidencyManager,
    location: ResidencyLocation,
) -> Any:
    return next(
        item
        for item in manager.snapshot().locations
        if item.location == location
    )


def test_pin_memory_rejects_a_non_cpu_or_unpinned_allocator_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_empty_strided = torch.empty_strided

    def fake_empty_strided(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return original_empty_strided(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_strided", fake_empty_strided)

    with pytest.raises(
        RuntimeError, match="Pinned host storage is unavailable"
    ):
        _plaintext().pin_memory()


class _Source:
    def __init__(self, load: Callable[[], TensorResident]) -> None:
        self._load = load
        self.load_count = 0

    def load(self) -> TensorResident:
        self.load_count += 1
        return self._load()


class _PairResident(TensorResident):
    def __init__(self, first: torch.Tensor, second: torch.Tensor) -> None:
        self.first = first
        self.second = second

    @property
    def _resident_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.first, self.second)

    def _with_resident_tensors(self, tensors: tuple[torch.Tensor, ...]) -> Self:
        return type(self)(*tensors)


def test_locations_are_canonical_and_strictly_validated() -> None:
    assert (
        ResidencyLocation("pageable-host", torch.device("cpu:0"))
        == PAGEABLE_HOST
    )
    assert (
        ResidencyLocation("pinned-host", torch.device("cpu:7")) == PINNED_HOST
    )
    assert PAGEABLE_HOST.device == torch.device("cpu")
    assert PAGEABLE_HOST.name == "pageable-host"
    assert str(PINNED_HOST) == "pinned-host"

    device_zero = cuda_location("cuda:0")
    assert device_zero == ResidencyLocation("cuda", torch.device("cuda:0"))
    assert device_zero.name == "cuda:0"
    assert cuda_location("cuda:127").device.index == 127

    with pytest.raises(ValueError, match="indexed"):
        cuda_location("cuda")
    with pytest.raises(ValueError, match="canonical"):
        cuda_location("cuda:00")
    with pytest.raises(ValueError, match="between 0 and 127"):
        cuda_location("cuda:128")
    with pytest.raises(ValueError, match="between 0 and 127"):
        cuda_location("cuda:256")
    with pytest.raises(ValueError, match="between 0 and 127"):
        cuda_location("cuda:999")
    with pytest.raises(ValueError, match="between 0 and 127"):
        ResidencyLocation("cuda", "cuda:256")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between 0 and 127"):
        ResidencyLocation("cuda", "cuda:999")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical"):
        ResidencyLocation("cuda", "cuda:01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical"):
        cuda_location("cuda:-1")
    with pytest.raises(ValueError, match="CPU device"):
        ResidencyLocation("pageable-host", torch.device("cuda:0"))
    with pytest.raises(ValueError, match="CPU device"):
        ResidencyLocation("pinned-host", torch.device("cuda:0"))
    with pytest.raises(ValueError, match="indexed"):
        ResidencyLocation("cuda", torch.device("cpu"))
    with pytest.raises(ValueError, match="Unsupported"):
        ResidencyLocation("disk", "cpu")  # type: ignore[arg-type]


def test_handle_value_spec_reservation_and_plan_validation_is_strict() -> None:
    handle = ResidencyHandle(
        manager_id="manager",
        handle_id="opaque-handle",
        value_type=Plaintext,
    )
    assert hash(handle) == hash(handle)

    for handle_field, handle_replacement, handle_error in (
        ("manager_id", " ", ValueError),
        ("manager_id", 1, TypeError),
        ("handle_id", "", ValueError),
        ("handle_id", 1, TypeError),
        ("value_type", str, TypeError),
    ):
        arguments: dict[str, Any] = {
            "manager_id": "manager",
            "handle_id": "opaque-handle",
            "value_type": Plaintext,
        }
        arguments[handle_field] = handle_replacement
        with pytest.raises(handle_error):
            ResidencyHandle(**arguments)

    spec_arguments: dict[str, Any] = {
        "value_type": Plaintext,
        "logical_nbytes": 16,
        "storage_nbytes": 24,
        "replica_mode": ReplicaMode.REPLICABLE,
        "recoverability": Recoverability.RECONSTRUCTIBLE,
    }
    for spec_field, spec_replacement, spec_error in (
        ("value_type", object, TypeError),
        ("logical_nbytes", -1, ValueError),
        ("logical_nbytes", True, TypeError),
        ("storage_nbytes", 8, ValueError),
        ("storage_nbytes", False, TypeError),
        ("replica_mode", "replicable", TypeError),
        ("recoverability", "reconstructible", TypeError),
    ):
        arguments = dict(spec_arguments)
        arguments[spec_field] = spec_replacement
        with pytest.raises(spec_error):
            ResidencyValueSpec(**arguments)

    with pytest.raises(TypeError, match="integer, not bool"):
        MemoryReservation(PAGEABLE_HOST, True, "workspace")
    with pytest.raises(ValueError, match="non-negative"):
        MemoryReservation(PAGEABLE_HOST, -1, "workspace")
    with pytest.raises(TypeError, match="ResidencyLocation"):
        MemoryReservation("cpu", 1, "workspace")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        MemoryReservation(PAGEABLE_HOST, 1, " ")
    with pytest.raises(TypeError, match="handle"):
        EnsureResident(object(), PAGEABLE_HOST)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="location"):
        DropResident(handle, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="from_location"):
        MoveResident(
            handle,
            PAGEABLE_HOST,
            from_location=object(),  # type: ignore[arg-type]
        )

    actions: Any = [EnsureResident(handle, PAGEABLE_HOST)]
    reservations: Any = [MemoryReservation(PAGEABLE_HOST, 8, "workspace")]
    plan = ResidencyPlan(
        "validated-plan",
        enter=actions,
        reservations=reservations,
    )
    actions.clear()
    reservations.clear()
    assert len(plan.enter) == 1
    assert len(plan.reservations) == 1
    assert isinstance(plan.enter, tuple)
    assert isinstance(plan.reservations, tuple)

    with pytest.raises(ValueError, match="non-empty"):
        ResidencyPlan(" ")
    with pytest.raises(TypeError, match="actions"):
        ResidencyPlan("bad-action", enter=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MemoryReservation"):
        ResidencyPlan("bad-reservation", reservations=(object(),))  # type: ignore[arg-type]

    snapshot_arguments: dict[str, Any] = {
        "location": PAGEABLE_HOST,
        "budget_bytes": None,
        "used_bytes": 8,
        "reserved_bytes": 4,
        "remaining_budget_bytes": None,
        "peak_used_bytes": 8,
        "peak_charged_bytes": 12,
        "value_count": 1,
        "reservation_count": 1,
        "use_count": 0,
        "hold_count": 0,
        "pending_event_count": 0,
        "allocator_allocated_bytes": None,
        "allocator_reserved_bytes": None,
    }
    ResidencyLocationSnapshot(**snapshot_arguments)
    with pytest.raises(ValueError, match="unbudgeted"):
        ResidencyLocationSnapshot(
            **{**snapshot_arguments, "remaining_budget_bytes": 0}
        )
    with pytest.raises(ValueError, match="must equal"):
        ResidencyLocationSnapshot(
            **{
                **snapshot_arguments,
                "budget_bytes": 16,
                "remaining_budget_bytes": None,
            }
        )


def test_manager_constructor_and_registration_inputs_are_strict() -> None:
    execution_error_hints = get_type_hints(ResidencyPlanExecutionError.__init__)
    assert execution_error_hints["partial_report"] is Any
    assert execution_error_hints["failed_action"] == Any | None
    assert execution_error_hints["failed_reservation"] == Any | None
    assert "ResidencyStaleStateError" in errors_module.__all__
    ResidencyManager().close()
    ResidencyManager({}).close()
    with pytest.raises(TypeError, match="mapping"):
        ResidencyManager([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="locations"):
        ResidencyManager({"cpu": 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="integer, not bool"):
        ResidencyManager({PAGEABLE_HOST: True})
    value = _plaintext()
    manager = ResidencyManager()
    preserve_spec = ResidencyValueSpec(
        value_type=Plaintext,
        logical_nbytes=value.nbytes,
        storage_nbytes=value.storage_nbytes,
        replica_mode=ReplicaMode.EXCLUSIVE,
        recoverability=Recoverability.MUST_PRESERVE,
    )
    with pytest.raises(ValueError, match="RECONSTRUCTIBLE"):
        manager.register_source(preserve_spec, _Source(value.clone))
    with pytest.raises(TypeError, match="define load"):
        manager.register_source(_spec(value), object())  # type: ignore[arg-type]
    assert manager.locations == ()
    manager.close()


def test_plan_positional_order_and_direct_scope_defaults_remain_unambiguous() -> (
    None
):
    manager = ResidencyManager()
    value = _plaintext()
    handle = manager.register_source(_spec(value), _Source(value.clone))
    enter = EnsureResident(handle, PAGEABLE_HOST)
    exit_action = DropResident(handle, PAGEABLE_HOST)
    reservation = MemoryReservation(PAGEABLE_HOST, 0, "workspace")

    plan = ResidencyPlan(
        "positional-plan",
        (enter,),
        (exit_action,),
        (reservation,),
    )
    assert plan.enter == (enter,)
    assert plan.exit == (exit_action,)
    assert plan.reservations == (reservation,)
    assert plan.reclaim == ()

    scope = ResidencyScope(manager, plan)
    with scope:
        assert len(_value_snapshot(manager, handle).materializations) == 1
    assert _value_snapshot(manager, handle).materializations == ()
    manager.close()


def test_scope_execution_authority_is_read_only_and_cannot_leak_reservations() -> (
    None
):
    manager = ResidencyManager({PAGEABLE_HOST: 64})
    other_manager = ResidencyManager()
    plan = ResidencyPlan(
        "captured-scope-authority",
        reservations=(MemoryReservation(PAGEABLE_HOST, 32, "workspace"),),
    )
    scope = ResidencyScope(manager, plan)
    assert scope.manager is manager
    assert scope.plan is plan
    assert scope.expected_state_version is None

    scope.__enter__()
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 32
    with pytest.raises(AttributeError):
        scope.manager = other_manager
    with pytest.raises(AttributeError):
        scope.plan = ResidencyPlan("replacement")
    with pytest.raises(AttributeError):
        scope.expected_state_version = 0
    scope.close()
    assert manager.snapshot().reservations == ()
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 0

    prepared_version = manager.state_version
    stale_scope = ResidencyScope(
        manager,
        ResidencyPlan("stale-read-only-version"),
        expected_state_version=prepared_version,
    )
    handle = manager.adopt(_plaintext())
    with pytest.raises(AttributeError):
        stale_scope.expected_state_version = None
    with pytest.raises(ResidencyStaleStateError):
        stale_scope.__enter__()

    manager.discard(handle)
    manager.close()
    other_manager.close()


def test_manager_public_entry_points_reject_wrong_input_kinds_intentionally() -> (
    None
):
    value = _plaintext()
    source = _Source(value.clone)
    manager = ResidencyManager()

    with pytest.raises(TypeError, match="adopt value.*TensorResident"):
        manager.adopt(object())  # type: ignore[type-var]
    with pytest.raises(TypeError, match="source spec.*ResidencyValueSpec"):
        manager.register_source(object(), source)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="load.*callable"):
        manager.register_source(_spec(value), object())  # type: ignore[arg-type]
    assert manager.locations == ()

    handle = manager.register_source(_spec(value), source)
    action = MoveResident(handle, PAGEABLE_HOST)
    before = manager.snapshot()
    with pytest.raises(TypeError, match="iterable of residency actions"):
        manager.execute_actions(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contain residency action"):
        manager.execute_actions((object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="plan name must be a string"):
        manager.execute_actions((action,), name=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plan name must be non-empty"):
        manager.execute_actions((action,), name=" ")
    with pytest.raises(TypeError, match="explain plan.*ResidencyPlan"):
        manager.explain(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scope plan.*ResidencyPlan"):
        manager.scope(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="label must be a string"):
        manager.reserve(PAGEABLE_HOST, 0, label=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="iterable of ResidencyHandle"):
        manager.acquire(object(), at=PAGEABLE_HOST)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="require ResidencyHandle values"):
        manager.hold((object(),), at=PAGEABLE_HOST)  # type: ignore[arg-type]

    after = manager.snapshot()
    assert before.values == after.values
    assert before.locations == after.locations
    assert before.reservations == after.reservations
    assert source.load_count == 0
    assert manager.trace() == ()
    manager.close()


def test_unavailable_cuda_location_is_rejected_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ResidencyManager()
    location = cuda_location("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="require available CUDA"):
        manager.reserve(location, 0, label="unavailable")
    assert manager.locations == ()
    assert manager.snapshot().locations == ()
    manager.close()


def test_locations_observation_serializes_with_lazy_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ResidencyManager()
    entered = Event()
    proceed = Event()
    blocked = False
    original_validate = manager._validate_location  # type: ignore[attr-defined]

    def blocking_validate(location: ResidencyLocation) -> None:
        nonlocal blocked
        original_validate(location)
        if not blocked:
            blocked = True
            entered.set()
            assert proceed.wait(timeout=2.0)

    monkeypatch.setattr(manager, "_validate_location", blocking_validate)
    reservations: list[Any] = []
    writer = Thread(
        target=lambda: reservations.append(
            manager.reserve(PAGEABLE_HOST, 0, label="lazy")
        )
    )
    observed: list[tuple[ResidencyLocation, ...]] = []
    reader = Thread(target=lambda: observed.append(manager.locations))

    writer.start()
    assert entered.wait(timeout=2.0)
    reader.start()
    assert reader.is_alive()
    proceed.set()
    writer.join(timeout=2.0)
    reader.join(timeout=2.0)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert observed == [(PAGEABLE_HOST,)]
    reservations[0].release()
    manager.close()


def test_unbudgeted_manager_lazily_accounts_locations_and_plan_peaks() -> None:
    value = _plaintext()
    manager = ResidencyManager()
    assert manager.locations == ()

    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    assert manager.locations == (PAGEABLE_HOST,)
    pageable = _location_snapshot(manager, PAGEABLE_HOST)
    assert pageable.budget_bytes is None
    assert pageable.remaining_budget_bytes is None
    assert pageable.used_bytes == value.storage_nbytes
    assert pageable.peak_used_bytes == value.storage_nbytes

    plan = ResidencyPlan(
        "lazy-pinned-dry-run",
        enter=(EnsureResident(handle, PINNED_HOST),),
    )
    explanation = manager.explain(plan)
    assert explanation.feasible
    assert dict(explanation.predicted_peak_bytes)[PINNED_HOST] == (
        value.storage_nbytes
    )
    assert manager.locations == (PAGEABLE_HOST,)

    no_op = manager.drop(handle, PINNED_HOST)
    assert no_op.no_op
    assert manager.locations == (PAGEABLE_HOST,)
    manager.close()

    exclusive_manager = ResidencyManager()
    exclusive = exclusive_manager.adopt(_plaintext())
    with pytest.raises(ResidencyOwnershipError, match="use MoveResident"):
        exclusive_manager.ensure(exclusive, PINNED_HOST)
    assert exclusive_manager.locations == (PAGEABLE_HOST,)
    exclusive_manager.close()


def test_partial_budgets_leave_omitted_locations_unbudgeted() -> None:
    value_a = _plaintext()
    value_b = _plaintext((4.0, 5.0, 6.0, 7.0))
    nbytes = value_a.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})

    handle_a = manager.adopt(value_a)
    handle_b = manager.adopt(value_b)
    assert manager.locations == (PINNED_HOST, PAGEABLE_HOST)
    assert tuple(item.location for item in manager.snapshot().locations) == (
        PINNED_HOST,
        PAGEABLE_HOST,
    )
    pageable = _location_snapshot(manager, PAGEABLE_HOST)
    assert pageable.budget_bytes is None
    assert pageable.remaining_budget_bytes is None
    assert pageable.used_bytes == 2 * nbytes

    unbudgeted = manager.reserve(
        PAGEABLE_HOST,
        100 * nbytes,
        label="unbudgeted-headroom",
    )
    pinned = manager.reserve(PINNED_HOST, nbytes, label="pinned-headroom")
    with pytest.raises(ResidencyBudgetError) as error:
        manager.reserve(PINNED_HOST, 1, label="over-budget")
    assert error.value.budget_bytes == nbytes
    assert error.value.used_bytes == 0
    assert error.value.reserved_bytes == nbytes
    assert error.value.requested_bytes == 1

    before_locations = manager.locations
    explanation = manager.explain(
        ResidencyPlan(
            "over-budget-plan",
            reservations=(
                MemoryReservation(PINNED_HOST, 1, "planned-headroom"),
            ),
        )
    )
    assert not explanation.feasible
    assert explanation.reason is not None
    assert "exceeds budget" in explanation.reason
    assert manager.locations == before_locations

    pinned_snapshot = _location_snapshot(manager, PINNED_HOST)
    assert pinned_snapshot.budget_bytes == nbytes
    assert pinned_snapshot.remaining_budget_bytes == 0
    unbudgeted.release()
    pinned.release()
    manager.discard(handle_a)
    manager.discard(handle_b)
    assert manager.locations == (PINNED_HOST, PAGEABLE_HOST)
    manager.close()


def test_budget_mapping_is_copied_and_zero_differs_from_unbudgeted() -> None:
    budgets = {PAGEABLE_HOST: 0}
    manager = ResidencyManager(budgets)
    budgets.clear()

    pageable = _location_snapshot(manager, PAGEABLE_HOST)
    assert pageable.budget_bytes == 0
    assert pageable.remaining_budget_bytes == 0
    with pytest.raises(ResidencyBudgetError):
        manager.adopt(_plaintext())
    assert manager.locations == (PAGEABLE_HOST,)

    unbudgeted = manager.reserve(
        PINNED_HOST,
        1 << 40,
        label="accounting-only",
    )
    pinned = _location_snapshot(manager, PINNED_HOST)
    assert pinned.budget_bytes is None
    assert pinned.remaining_budget_bytes is None
    assert pinned.reserved_bytes == 1 << 40
    unbudgeted.release()
    manager.close()


def test_unbudgeted_allocation_failure_preserves_source_and_lazy_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _plaintext()
    manager = ResidencyManager()
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)

    def fail_copy(*args: object, **kwargs: object) -> TensorResident:
        del args, kwargs
        raise torch.cuda.OutOfMemoryError("injected allocator failure")

    monkeypatch.setattr(manager, "_copy_value", fail_copy)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected"):
        manager.ensure(handle, PINNED_HOST)

    assert manager.locations == (PAGEABLE_HOST,)
    state = _value_snapshot(manager, handle)
    assert tuple(item.location for item in state.materializations) == (
        PAGEABLE_HOST,
    )
    assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == (
        value.storage_nbytes
    )
    manager.close()


def test_adopt_preserves_typed_identity_and_accounts_logical_and_storage_bytes() -> (
    None
):
    backing = torch.arange(8, dtype=torch.float64)
    value = Plaintext(message=backing[::2], level=0, scale=16.0)
    assert value.nbytes == 4 * backing.element_size()
    assert value.storage_nbytes == backing.numel() * backing.element_size()

    manager = ResidencyManager({PAGEABLE_HOST: value.storage_nbytes})
    handle = manager.adopt(
        value,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    assert handle == ResidencyHandle(
        manager_id=manager.manager_id,
        handle_id=handle.handle_id,
        value_type=Plaintext,
    )
    assert handle.handle_id
    assert handle.value_type is Plaintext

    value_state = _value_snapshot(manager, handle)
    materialization = value_state.materializations[0]
    assert value_state.spec.value_type is Plaintext
    assert value_state.spec.logical_nbytes == value.nbytes
    assert value_state.spec.storage_nbytes == value.storage_nbytes
    assert value_state.spec.replica_mode is ReplicaMode.REPLICABLE
    assert value_state.spec.recoverability is Recoverability.MUST_PRESERVE
    assert not value_state.has_source
    assert not value_state.discarded
    assert materialization.location == PAGEABLE_HOST
    assert materialization.logical_nbytes == value.nbytes
    assert materialization.storage_nbytes == value.storage_nbytes
    assert materialization.charged_nbytes == value.storage_nbytes
    assert materialization.use_count == 0
    assert materialization.hold_count == 0
    assert materialization.pending_event_count == 0

    location_state = _location_snapshot(manager, PAGEABLE_HOST)
    assert location_state.used_bytes == value.storage_nbytes
    assert location_state.remaining_budget_bytes == 0
    assert location_state.peak_used_bytes == value.storage_nbytes
    assert location_state.peak_charged_bytes == value.storage_nbytes
    assert location_state.value_count == 1
    assert location_state.allocator_allocated_bytes is None
    assert location_state.allocator_reserved_bytes is None

    lease = manager.acquire((handle,), at=PAGEABLE_HOST)
    borrowed = lease.values
    concrete = assert_type(borrowed[handle], Plaintext)
    assert concrete is value
    lease.release()
    manager.close()


def test_failed_adopt_does_not_install_state_or_consume_budget() -> None:
    value = _plaintext()
    manager = ResidencyManager({PAGEABLE_HOST: value.storage_nbytes})

    with pytest.raises(TypeError, match="replica_mode"):
        manager.adopt(
            value.clone(),
            replica_mode="exclusive",  # type: ignore[arg-type]
        )
    assert manager.snapshot().values == ()
    assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == 0

    handle = manager.adopt(
        value.clone(),
        replica_mode=ReplicaMode.EXCLUSIVE,
    )
    assert handle.handle_id
    manager.close()


def test_snapshot_separates_shared_storage_from_conservative_charge() -> None:
    backing = torch.arange(4, dtype=torch.float64)
    value = _PairResident(backing, backing)
    assert value.nbytes == 2 * value.storage_nbytes
    manager = ResidencyManager({PAGEABLE_HOST: value.nbytes})
    handle = manager.adopt(value)

    materialization = _value_snapshot(manager, handle).materializations[0]
    assert materialization.logical_nbytes == value.nbytes
    assert materialization.storage_nbytes == value.storage_nbytes
    assert materialization.charged_nbytes == value.nbytes
    assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == value.nbytes
    manager.close()


def test_adopt_generates_distinct_handles_and_rejects_storage_aliases() -> None:
    batched = _plaintext(tuple(float(index) for index in range(8)))
    batched.message = batched.message.reshape(2, 4)  # type: ignore[union-attr]
    left, right = batched.unbind_batch()
    manager = ResidencyManager({PAGEABLE_HOST: 256})
    first = manager.adopt(left)

    with pytest.raises(ResidencyOwnershipError, match="independent backing"):
        manager.adopt(right)

    second = manager.adopt(right.clone())
    third = manager.adopt(_plaintext())
    assert len({first.handle_id, second.handle_id, third.handle_id}) == 3
    assert len({first, second, third}) == 3
    manager.close()


def test_budget_reservations_admission_and_peak_accounting() -> None:
    value_a = _plaintext()
    value_b = _plaintext((4.0, 5.0, 6.0, 7.0))
    nbytes = value_a.storage_nbytes
    manager = ResidencyManager({PAGEABLE_HOST: 2 * nbytes})

    reservation = manager.reserve(PAGEABLE_HOST, nbytes, label="output")
    reserved = _location_snapshot(manager, PAGEABLE_HOST)
    assert reserved.budget_bytes == 2 * nbytes
    assert reserved.reserved_bytes == nbytes
    assert reserved.used_bytes == 0
    assert reserved.remaining_budget_bytes == nbytes
    assert reserved.peak_used_bytes == 0
    assert reserved.peak_charged_bytes == nbytes
    assert reserved.reservation_count == 1
    reservation_state = manager.snapshot().reservations
    assert len(reservation_state) == 1
    assert reservation_state[0].location == PAGEABLE_HOST
    assert reservation_state[0].nbytes == nbytes
    assert reservation_state[0].label == "output"

    handle_a = manager.adopt(value_a)
    with pytest.raises(ResidencyBudgetError) as budget_error:
        manager.adopt(value_b)
    assert budget_error.value.used_bytes == nbytes
    assert budget_error.value.budget_bytes == 2 * nbytes
    assert budget_error.value.reserved_bytes == nbytes
    assert budget_error.value.requested_bytes == nbytes

    reservation.release()
    reservation.release()
    assert manager.snapshot().reservations == ()
    with pytest.raises(ResidencyLifetimeClosedError), reservation:
        pass
    handle_b = manager.adopt(value_b)
    full = _location_snapshot(manager, PAGEABLE_HOST)
    assert full.used_bytes == 2 * nbytes
    assert full.peak_used_bytes == 2 * nbytes
    assert full.peak_charged_bytes == 2 * nbytes

    manager.discard(handle_a)
    after_discard = _location_snapshot(manager, PAGEABLE_HOST)
    assert after_discard.used_bytes == nbytes
    assert after_discard.peak_used_bytes == 2 * nbytes
    assert after_discard.peak_charged_bytes == 2 * nbytes
    manager.discard(handle_b)
    manager.close()


def test_exclusive_ensure_is_rejected_and_final_preserved_value_requires_discard() -> (
    None
):
    value = _plaintext()
    manager = ResidencyManager(
        {
            PAGEABLE_HOST: 2 * value.storage_nbytes,
            PINNED_HOST: 2 * value.storage_nbytes,
        }
    )
    handle = manager.adopt(value)

    with pytest.raises(ResidencyOwnershipError, match="use MoveResident"):
        manager.ensure(handle, PINNED_HOST)
    with pytest.raises(ResidencyOwnershipError, match="final MUST_PRESERVE"):
        manager.drop(handle, PAGEABLE_HOST)

    report = manager.discard(handle)
    assert isinstance(report.action, DiscardValue)
    state = next(
        item for item in manager.snapshot().values if item.handle == handle
    )
    assert state.discarded
    assert not state.has_source
    assert state.materializations == ()
    with pytest.raises(ResidencyHandleError, match="discarded"):
        manager.ensure(handle, PAGEABLE_HOST)
    manager.close()


def test_source_load_is_exact_and_reconstructible_values_can_drop_last_replica() -> (
    None
):
    expected = _plaintext()
    source = _Source(expected.clone)
    manager = ResidencyManager({PAGEABLE_HOST: expected.storage_nbytes})
    handle = manager.register_source(_spec(expected), source)

    unloaded = _value_snapshot(manager, handle)
    assert unloaded.has_source
    assert unloaded.source_location == PAGEABLE_HOST
    assert unloaded.materializations == ()
    report = manager.ensure(handle, PAGEABLE_HOST)
    assert report.source == PAGEABLE_HOST
    assert report.destination == PAGEABLE_HOST
    assert source.load_count == 1
    with manager.acquire((handle,), at=PAGEABLE_HOST) as values:
        actual = values[handle]
        assert isinstance(actual, Plaintext)
        torch.testing.assert_close(actual.message, expected.message)

    manager.drop(handle, PAGEABLE_HOST)
    dropped = _value_snapshot(manager, handle)
    assert dropped.has_source
    assert dropped.materializations == ()
    manager.ensure(handle, PAGEABLE_HOST)
    assert source.load_count == 2
    manager.close()


@pytest.mark.pinned_memory
@pytest.mark.parametrize("destination", (PAGEABLE_HOST, PINNED_HOST))
def test_unloaded_source_backed_implicit_move_retains_only_destination(
    destination: ResidencyLocation,
) -> None:
    expected = _plaintext()
    nbytes = expected.storage_nbytes
    source = _Source(expected.clone)
    manager = ResidencyManager({PAGEABLE_HOST: nbytes, PINNED_HOST: nbytes})
    handle = manager.register_source(_spec(expected), source)

    report = manager.move(handle, destination)

    assert not report.no_op
    assert report.source == PAGEABLE_HOST
    assert report.destination == destination
    assert source.load_count == 1
    state = _value_snapshot(manager, handle)
    assert tuple(item.location for item in state.materializations) == (
        destination,
    )
    assert _location_snapshot(manager, destination).used_bytes == nbytes
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 0
    if destination != PAGEABLE_HOST:
        assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == 0
    with manager.acquire((handle,), at=destination) as values:
        actual = values[handle]
        assert actual.message is not None
        assert expected.message is not None
        torch.testing.assert_close(actual.message, expected.message)
    manager.close()


@pytest.mark.parametrize("destination", (PAGEABLE_HOST, PINNED_HOST))
def test_unloaded_source_backed_explicit_move_fails_without_false_success(
    destination: ResidencyLocation,
) -> None:
    expected = _plaintext()
    source = _Source(expected.clone)
    manager = ResidencyManager()
    handle = manager.register_source(_spec(expected), source)
    before = manager.snapshot()

    with pytest.raises(ResidencyUnavailableError, match="Explicit source"):
        manager.move(
            handle,
            destination,
            from_location=PAGEABLE_HOST,
        )

    after = manager.snapshot()
    assert before.values == after.values
    assert before.locations == after.locations
    assert before.reservations == after.reservations
    assert source.load_count == 0
    assert manager.trace() == ()
    manager.close()


def test_source_return_type_bytes_storage_and_location_are_validated() -> None:
    expected = _plaintext()
    expected_spec = _spec(expected)
    larger_backing = torch.arange(8, dtype=torch.float64)
    storage_mismatch = Plaintext(
        message=larger_backing[::2], level=0, scale=16.0
    )
    wrong_type = _PairResident(
        torch.arange(2, dtype=torch.float64),
        torch.arange(2, dtype=torch.float64),
    )
    cases = (
        (_Source(lambda: wrong_type), PAGEABLE_HOST, "Expected Plaintext"),
        (
            _Source(lambda: _plaintext((0.0, 1.0, 2.0))),
            PAGEABLE_HOST,
            "Logical payload size mismatch",
        ),
        (
            _Source(lambda: storage_mismatch),
            PAGEABLE_HOST,
            "Backing-storage charge mismatch",
        ),
        (
            _Source(expected.clone),
            PINNED_HOST,
            "Pinned-host materialization is not fully pinned",
        ),
    )

    for index, (source, source_location, message) in enumerate(cases):
        manager = ResidencyManager(
            {
                PAGEABLE_HOST: 2 * storage_mismatch.storage_nbytes,
                PINNED_HOST: 2 * storage_mismatch.storage_nbytes,
            },
        )
        handle = manager.register_source(
            expected_spec,
            source,
            source_location=source_location,
        )
        with pytest.raises(ResidencyMaterializationError, match=message):
            manager.ensure(handle, PAGEABLE_HOST)
        state = _value_snapshot(manager, handle)
        assert state.has_source
        assert state.materializations == ()
        for location in manager.snapshot().locations:
            assert location.used_bytes == 0
            assert location.reserved_bytes == 0
        manager.close()


def test_source_failure_preserves_registration_and_can_be_retried() -> None:
    expected = _plaintext()

    class FailOnceSource:
        load_count = 0

        def load(self) -> Plaintext:
            self.load_count += 1
            if self.load_count == 1:
                raise OSError("injected source failure")
            return expected.clone()

    source = FailOnceSource()
    manager = ResidencyManager({PAGEABLE_HOST: expected.storage_nbytes})
    handle = manager.register_source(_spec(expected), source)

    with pytest.raises(OSError, match="injected source failure"):
        manager.ensure(handle, PAGEABLE_HOST)
    failed = _value_snapshot(manager, handle)
    assert failed.has_source
    assert failed.materializations == ()
    location = _location_snapshot(manager, PAGEABLE_HOST)
    assert location.used_bytes == 0
    assert location.reserved_bytes == 0

    manager.ensure(handle, PAGEABLE_HOST)
    assert source.load_count == 2
    manager.close()


def test_source_callback_cannot_reenter_manager() -> None:
    expected = _plaintext()
    manager = ResidencyManager({PAGEABLE_HOST: expected.storage_nbytes})

    def reentrant_load() -> Plaintext:
        _ = manager.locations
        return expected.clone()

    handle = manager.register_source(_spec(expected), _Source(reentrant_load))
    with pytest.raises(
        ResidencyReentrancyError, match="source callback is active"
    ):
        manager.ensure(handle, PAGEABLE_HOST)
    state = _value_snapshot(manager, handle)
    assert state.has_source
    assert state.materializations == ()
    manager.close()


def test_source_callback_cross_thread_reentry_fails_without_deadlock() -> None:
    expected = _plaintext()
    manager = ResidencyManager({PAGEABLE_HOST: expected.storage_nbytes})
    failures: list[BaseException] = []

    def cross_thread_load() -> Plaintext:
        def reenter() -> None:
            try:
                manager.snapshot()
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=reenter)
        thread.start()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        return expected.clone()

    handle = manager.register_source(
        _spec(expected),
        _Source(cross_thread_load),
    )
    manager.ensure(handle, PAGEABLE_HOST)
    assert len(failures) == 1
    assert isinstance(failures[0], ResidencyReentrancyError)
    manager.close()


def test_multi_item_acquire_and_hold_are_atomic_and_lifetimes_are_idempotent() -> (
    None
):
    value_a = _plaintext()
    value_b = _plaintext((4.0, 5.0, 6.0, 7.0))
    manager = ResidencyManager({PAGEABLE_HOST: 4 * value_a.storage_nbytes})
    handle_a = manager.adopt(value_a)
    handle_b = manager.adopt(value_b)
    absent = manager.register_source(_spec(value_a), _Source(value_a.clone))

    with pytest.raises(ResidencyUnavailableError, match="not resident"):
        manager.acquire((handle_a, absent), at=PAGEABLE_HOST)
    with pytest.raises(ResidencyUnavailableError, match="not resident"):
        manager.hold((handle_a, absent), at=PAGEABLE_HOST)
    assert _value_snapshot(manager, handle_a).materializations[0].use_count == 0
    assert (
        _value_snapshot(manager, handle_a).materializations[0].hold_count == 0
    )

    lease = manager.acquire((handle_a, handle_b, handle_a), at=PAGEABLE_HOST)
    assert lease.handles == (handle_a, handle_b)
    borrowed = lease.values
    assert tuple(borrowed) == (handle_a, handle_b)
    assert borrowed[handle_a] is value_a
    with pytest.raises(ResidencyInUseError, match="active_reads=1"):
        manager.discard(handle_a)
    lease.release()
    lease.release()
    assert not lease.active
    with pytest.raises(ResidencyLifetimeClosedError):
        borrowed[handle_a]
    with pytest.raises(ResidencyLifetimeClosedError):
        len(borrowed)

    hold = manager.hold((handle_a, handle_b, handle_a), at=PAGEABLE_HOST)
    assert hold.handles == (handle_a, handle_b)
    assert _location_snapshot(manager, PAGEABLE_HOST).hold_count == 2
    with pytest.raises(ResidencyInUseError, match="holds=1"):
        manager.discard(handle_b)
    hold.release()
    hold.release()
    assert not hold.active
    with pytest.raises(ResidencyLifetimeClosedError), hold:
        pass
    assert _location_snapshot(manager, PAGEABLE_HOST).hold_count == 0
    manager.close()


def test_acquire_and_hold_require_nonempty_handle_collections() -> None:
    manager = ResidencyManager()

    with pytest.raises(ValueError, match="at least one handle"):
        manager.acquire((), at=PAGEABLE_HOST)
    with pytest.raises(ValueError, match="at least one handle"):
        manager.hold(iter(()), at=PAGEABLE_HOST)

    assert manager.snapshot().values == ()
    manager.close()


def test_foreign_unknown_and_discarded_handles_are_rejected() -> None:
    value = _plaintext()
    first = ResidencyManager({PAGEABLE_HOST: 2 * value.storage_nbytes})
    second = ResidencyManager({PAGEABLE_HOST: 2 * value.storage_nbytes})
    assert first.manager_id != second.manager_id
    foreign = second.adopt(value.clone())

    with pytest.raises(ResidencyHandleError, match="belongs to manager"):
        first.acquire((foreign,), at=PAGEABLE_HOST)
    unknown = ResidencyHandle(
        manager_id=first.manager_id,
        handle_id="unknown-handle",
        value_type=Plaintext,
    )
    with pytest.raises(ResidencyHandleError, match="unknown"):
        first.ensure(unknown, PAGEABLE_HOST)

    local = first.adopt(value)
    first.discard(local)
    with pytest.raises(ResidencyHandleError, match="discarded"):
        first.acquire((local,), at=PAGEABLE_HOST)
    first.close()
    second.close()


def test_trace_snapshot_and_no_op_transition_reports() -> None:
    value = _plaintext()
    manager = ResidencyManager(
        {PAGEABLE_HOST: value.storage_nbytes},
        trace_capacity=2,
    )
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    report = manager.ensure(handle, PAGEABLE_HOST)
    assert report.no_op
    assert report.reason == "destination already resident"
    assert report.logical_nbytes == value.nbytes
    assert report.storage_nbytes == value.storage_nbytes
    assert report.started_at_ns <= report.completed_at_ns
    assert report.allocator_allocated_bytes_before is None
    assert manager.trace() == (report,)

    snapshot = manager.snapshot()
    assert snapshot.manager_id == manager.manager_id
    assert snapshot.captured_at_ns > 0
    assert snapshot.locations[0].use_count == 0
    assert snapshot.locations[0].pending_event_count == 0
    assert snapshot.reservations == ()

    move = MoveResident(handle, PAGEABLE_HOST)
    explanation = manager.explain(
        ResidencyPlan("move-no-op", enter=(move,))
    ).actions[0]
    move_report = manager.move(handle, PAGEABLE_HOST)
    assert explanation.no_op and move_report.no_op
    assert explanation.source is move_report.source is None
    assert explanation.destination == move_report.destination == PAGEABLE_HOST

    drop = manager.drop(handle, PINNED_HOST)
    assert drop.no_op
    lazy = manager.explain(
        ResidencyPlan("lazy-drop", enter=(DropResident(handle, PINNED_HOST),))
    )
    assert lazy.feasible
    assert lazy.actions[0].no_op
    manager.clear_trace()
    assert manager.trace() == ()
    manager.close()


def test_plan_explain_execute_and_scope_preserve_order_and_reservation_lifetime() -> (
    None
):
    value = _plaintext()
    nbytes = value.storage_nbytes
    source = _Source(value.clone)
    manager = ResidencyManager()
    handle = manager.register_source(_spec(value), source)
    ensure = EnsureResident(handle, PAGEABLE_HOST)
    drop = DropResident(handle, PAGEABLE_HOST)
    reservation = MemoryReservation(PAGEABLE_HOST, nbytes, "workspace")
    plan = ResidencyPlan(
        "scope-plan",
        enter=(ensure,),
        exit=(drop,),
        reservations=(reservation,),
    )

    explanation = manager.explain(plan)
    assert explanation.feasible
    assert explanation.reason is None
    assert tuple(item.action for item in explanation.actions) == (ensure, drop)
    assert explanation.reservations == (reservation,)
    assert dict(explanation.predicted_peak_bytes)[PAGEABLE_HOST] == 2 * nbytes
    assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == 0

    scope = manager.scope(plan)
    with scope as active_scope:
        assert active_scope is scope
        during = _location_snapshot(manager, PAGEABLE_HOST)
        assert during.used_bytes == nbytes
        assert during.reserved_bytes == nbytes
        with manager.acquire((handle,), at=PAGEABLE_HOST) as values:
            assert isinstance(values[handle], Plaintext)
    assert scope.report is not None
    assert tuple(item.action for item in scope.report.transitions) == (
        ensure,
        drop,
    )
    after = _location_snapshot(manager, PAGEABLE_HOST)
    assert after.budget_bytes is None
    assert after.remaining_budget_bytes is None
    assert after.used_bytes == 0
    assert after.reserved_bytes == 0
    assert after.peak_used_bytes == nbytes
    assert after.peak_charged_bytes == 2 * nbytes

    execute_report = manager.execute_actions(
        (ensure, drop, ensure), name="ordered-execute"
    )
    assert tuple(item.action for item in execute_report.transitions) == (
        ensure,
        drop,
        ensure,
    )
    assert _value_snapshot(manager, handle).materializations[0].location == (
        PAGEABLE_HOST
    )
    with pytest.raises(RuntimeError, match="entered twice"), scope:
        pass
    manager.close()


def test_unloaded_source_backed_move_is_acquirable_after_plan_execute() -> None:
    expected = _plaintext()
    source = _Source(expected.clone)
    manager = ResidencyManager()
    handle = manager.register_source(_spec(expected), source)
    action = MoveResident(handle, PAGEABLE_HOST)

    explanation = manager.explain(
        ResidencyPlan("unloaded-move-execute", enter=(action,))
    )
    assert explanation.feasible
    assert not explanation.actions[0].no_op
    assert explanation.actions[0].source == PAGEABLE_HOST
    report = manager.execute_actions((action,), name="unloaded-move-execute")

    assert len(report.transitions) == 1
    assert not report.transitions[0].no_op
    assert report.transitions[0].source == PAGEABLE_HOST
    assert report.transitions[0].destination == PAGEABLE_HOST
    assert source.load_count == 1
    with manager.acquire((handle,), at=PAGEABLE_HOST) as values:
        actual = values[handle]
        assert actual.message is not None
        assert expected.message is not None
        torch.testing.assert_close(actual.message, expected.message)
    assert len(_value_snapshot(manager, handle).materializations) == 1
    manager.close()


@pytest.mark.pinned_memory
def test_unloaded_source_move_simulation_preserves_dependent_destination() -> (
    None
):
    expected = _plaintext()
    source = _Source(expected.clone)
    manager = ResidencyManager()
    handle = manager.register_source(_spec(expected), source)
    actions = (
        MoveResident(handle, PAGEABLE_HOST),
        MoveResident(
            handle,
            PINNED_HOST,
            from_location=PAGEABLE_HOST,
        ),
    )
    plan = ResidencyPlan("dependent-unloaded-move", enter=actions)

    explanation = manager.explain(plan)
    assert explanation.feasible
    assert tuple(item.action for item in explanation.actions) == actions
    report = manager.execute_actions(actions, name=plan.name)

    assert tuple(item.action for item in report.transitions) == actions
    assert source.load_count == 1
    assert tuple(
        item.location
        for item in _value_snapshot(manager, handle).materializations
    ) == (PINNED_HOST,)
    manager.drop(handle, PINNED_HOST)
    manager.close()


def test_unloaded_explicit_move_plan_preflight_is_failure_atomic() -> None:
    expected = _plaintext()
    source = _Source(expected.clone)
    manager = ResidencyManager()
    handle = manager.register_source(_spec(expected), source)
    action = MoveResident(
        handle,
        PAGEABLE_HOST,
        from_location=PAGEABLE_HOST,
    )
    plan = ResidencyPlan("unloaded-explicit-move", enter=(action,))

    explanation = manager.explain(plan)
    assert not explanation.feasible
    assert explanation.actions[0].reason == (
        "explicit source pageable-host is not resident"
    )
    before = manager.snapshot()
    with pytest.raises(ResidencyPlanError, match="explicit source"):
        manager.execute_actions((action,), name=plan.name)
    after = manager.snapshot()

    assert before.values == after.values
    assert before.locations == after.locations
    assert before.reservations == after.reservations
    assert source.load_count == 0
    assert manager.trace() == ()

    scope = manager.scope(
        ResidencyPlan(
            "unloaded-explicit-move-scope",
            enter=(action,),
            reservations=(
                MemoryReservation(PAGEABLE_HOST, 1, "unused-headroom"),
            ),
        )
    )
    before_scope = manager.snapshot()
    with pytest.raises(ResidencyPlanError, match="explicit source"):
        scope.__enter__()
    after_scope = manager.snapshot()
    assert before_scope.values == after_scope.values
    assert before_scope.locations == after_scope.locations
    assert before_scope.reservations == after_scope.reservations
    assert source.load_count == 0
    assert manager.trace() == ()
    manager.close()


def test_unloaded_source_backed_move_is_retained_through_plan_scope_body() -> (
    None
):
    expected = _plaintext()
    source = _Source(expected.clone)
    manager = ResidencyManager()
    handle = manager.register_source(_spec(expected), source)
    move = MoveResident(handle, PAGEABLE_HOST)
    drop = DropResident(handle, PAGEABLE_HOST)
    scope = manager.scope(
        ResidencyPlan(
            "unloaded-move-scope",
            enter=(move,),
            exit=(drop,),
        )
    )

    with scope:
        with manager.acquire((handle,), at=PAGEABLE_HOST) as values:
            actual = values[handle]
            assert actual.message is not None
            assert expected.message is not None
            torch.testing.assert_close(actual.message, expected.message)
        assert len(_value_snapshot(manager, handle).materializations) == 1

    assert scope.report is not None
    assert tuple(item.action for item in scope.report.transitions) == (
        move,
        drop,
    )
    assert not scope.report.transitions[0].no_op
    assert _value_snapshot(manager, handle).materializations == ()
    assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == 0
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 0
    assert source.load_count == 1
    manager.close()


def test_nested_plan_scopes_stack_and_release_reservations() -> None:
    manager = ResidencyManager({PAGEABLE_HOST: 64})
    outer = manager.scope(
        ResidencyPlan(
            "outer",
            reservations=(MemoryReservation(PAGEABLE_HOST, 17, "outer"),),
        )
    )
    inner = manager.scope(
        ResidencyPlan(
            "inner",
            reservations=(MemoryReservation(PAGEABLE_HOST, 11, "inner"),),
        )
    )

    with outer:
        assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 17
        with inner:
            assert (
                _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 28
            )
        assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 17
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 0
    assert outer.report is not None
    assert inner.report is not None
    manager.close()


def test_abandoned_scope_releases_reservations_without_running_exit_actions() -> (
    None
):
    manager = ResidencyManager({PAGEABLE_HOST: 64})
    scope = manager.scope(
        ResidencyPlan(
            "abandoned",
            reservations=(MemoryReservation(PAGEABLE_HOST, 32, "workspace"),),
        )
    )
    scope.__enter__()
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 32
    scope_ref = ref(scope)

    with pytest.warns(ResourceWarning, match="ResidencyScope"):
        del scope
        collect()

    assert scope_ref() is None
    assert _location_snapshot(manager, PAGEABLE_HOST).reserved_bytes == 0
    manager.close()


def test_scope_exit_failure_exposes_structured_partial_report() -> None:
    first_value = _plaintext()
    second_value = _plaintext((4.0, 5.0, 6.0, 7.0))
    nbytes = first_value.storage_nbytes
    manager = ResidencyManager({PAGEABLE_HOST: 2 * nbytes})
    first = manager.register_source(
        _spec(first_value), _Source(first_value.clone)
    )
    second = manager.register_source(
        _spec(second_value), _Source(second_value.clone)
    )
    scope = manager.scope(
        ResidencyPlan(
            "partial-exit",
            enter=(
                EnsureResident(first, PAGEABLE_HOST),
                EnsureResident(second, PAGEABLE_HOST),
            ),
            exit=(
                DropResident(first, PAGEABLE_HOST),
                DropResident(second, PAGEABLE_HOST),
            ),
        )
    )
    scope.__enter__()
    hold = manager.hold((second,), at=PAGEABLE_HOST)

    with pytest.raises(ResidencyPlanExecutionError) as error_info:
        scope.close()
    error = error_info.value
    assert not isinstance(error, ResidencyPlanError)
    assert isinstance(error.__cause__, ResidencyInUseError)
    assert error.plan_name == "partial-exit"
    assert error.phase == "exit"
    assert error.failed_action == DropResident(second, PAGEABLE_HOST)
    assert error.failed_action_index == 1
    assert tuple(
        transition.action for transition in error.partial_report.transitions
    ) == (
        EnsureResident(first, PAGEABLE_HOST),
        EnsureResident(second, PAGEABLE_HOST),
        DropResident(first, PAGEABLE_HOST),
    )
    assert error.partial_report.plan_name == error.plan_name
    assert (
        error.partial_report.completed_at_ns
        >= error.partial_report.started_at_ns
    )
    assert _value_snapshot(manager, first).materializations == ()
    assert len(_value_snapshot(manager, second).materializations) == 1
    hold.release()
    manager.drop(second, PAGEABLE_HOST)
    manager.close()


def test_body_and_exit_failure_retains_structured_scope_exit_error() -> None:
    first_value = _plaintext()
    second_value = _plaintext((4.0, 5.0, 6.0, 7.0))
    manager = ResidencyManager()
    first = manager.register_source(
        _spec(first_value), _Source(first_value.clone)
    )
    second = manager.register_source(
        _spec(second_value), _Source(second_value.clone)
    )
    actions = (
        EnsureResident(first, PAGEABLE_HOST),
        EnsureResident(second, PAGEABLE_HOST),
    )
    exit_actions = (
        DropResident(first, PAGEABLE_HOST),
        DropResident(second, PAGEABLE_HOST),
    )
    scope = manager.scope(
        ResidencyPlan("body-and-exit-failure", enter=actions, exit=exit_actions)
    )

    with pytest.raises(ValueError, match="injected body failure") as body_error:
        with scope:
            hold = manager.hold((second,), at=PAGEABLE_HOST)
            raise ValueError("injected body failure")

    assert any("exit failed" in note for note in body_error.value.__notes__)
    assert scope.report is None
    assert scope.exit_error is not None
    assert scope.exit_error.phase == "exit"
    assert scope.exit_error.failed_action == exit_actions[1]
    assert isinstance(scope.exit_error.__cause__, ResidencyInUseError)
    assert tuple(
        transition.action
        for transition in scope.exit_error.partial_report.transitions
    ) == (*actions, exit_actions[0])
    hold.release()
    manager.drop(second, PAGEABLE_HOST)
    manager.close()


def test_execute_runtime_failure_exposes_structured_partial_report() -> None:
    first_value = _plaintext()
    second_value = _plaintext((4.0, 5.0, 6.0, 7.0))

    class FailingSource:
        def load(self) -> Plaintext:
            raise OSError("injected plan source failure")

    manager = ResidencyManager()
    first = manager.register_source(
        _spec(first_value), _Source(first_value.clone)
    )
    second = manager.register_source(_spec(second_value), FailingSource())
    actions = (
        EnsureResident(first, PAGEABLE_HOST),
        EnsureResident(second, PAGEABLE_HOST),
    )

    with pytest.raises(ResidencyPlanExecutionError) as error_info:
        manager.execute_actions(actions, name="partial-execute")

    error = error_info.value
    assert not isinstance(error, ResidencyPlanError)
    assert isinstance(error.__cause__, OSError)
    assert error.plan_name == "partial-execute"
    assert error.phase == "execute"
    assert error.failed_action == actions[1]
    assert error.failed_action_index == 1
    assert tuple(
        transition.action for transition in error.partial_report.transitions
    ) == (actions[0],)
    assert len(_value_snapshot(manager, first).materializations) == 1
    assert _value_snapshot(manager, second).materializations == ()
    manager.drop(first, PAGEABLE_HOST)
    manager.close()


def test_scope_entry_failure_exposes_partial_report_and_releases_reservation() -> (
    None
):
    first_value = _plaintext()
    second_value = _plaintext((4.0, 5.0, 6.0, 7.0))

    class FailingSource:
        def load(self) -> Plaintext:
            raise OSError("injected scope source failure")

    manager = ResidencyManager()
    first = manager.register_source(
        _spec(first_value), _Source(first_value.clone)
    )
    second = manager.register_source(_spec(second_value), FailingSource())
    actions = (
        EnsureResident(first, PAGEABLE_HOST),
        EnsureResident(second, PAGEABLE_HOST),
    )
    scope = manager.scope(
        ResidencyPlan(
            "partial-enter",
            enter=actions,
            reservations=(
                MemoryReservation(PAGEABLE_HOST, 1, "temporary-workspace"),
            ),
        )
    )

    with pytest.raises(ResidencyPlanExecutionError) as error_info:
        scope.__enter__()

    error = error_info.value
    assert isinstance(error.__cause__, OSError)
    assert error.phase == "enter"
    assert error.failed_action == actions[1]
    assert error.failed_action_index == 1
    assert tuple(
        transition.action for transition in error.partial_report.transitions
    ) == (actions[0],)
    assert manager.snapshot().reservations == ()
    assert len(_value_snapshot(manager, first).materializations) == 1
    assert _value_snapshot(manager, second).materializations == ()
    scope.close()
    assert scope.report is None
    assert len(_value_snapshot(manager, first).materializations) == 1
    manager.drop(first, PAGEABLE_HOST)
    manager.close()


def test_allocator_telemetry_failure_cannot_fail_a_committed_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _plaintext()
    manager = ResidencyManager()
    handle = manager.register_source(_spec(value), _Source(value.clone))

    def fail_allocator_observation(
        device: torch.device | None,
    ) -> tuple[int | None, int | None]:
        del device
        raise RuntimeError("injected allocator telemetry failure")

    monkeypatch.setattr(
        manager, "_allocator_metrics", fail_allocator_observation
    )
    report = manager.move(handle, PAGEABLE_HOST)

    assert report.allocator_allocated_bytes_before is None
    assert report.allocator_reserved_bytes_before is None
    assert report.allocator_allocated_bytes_after is None
    assert report.allocator_reserved_bytes_after is None
    assert len(_value_snapshot(manager, handle).materializations) == 1
    manager.drop(handle, PAGEABLE_HOST)
    manager.close()


def test_plan_preflight_is_atomic_and_wraps_specific_direct_action_failure() -> (
    None
):
    value = _plaintext()
    manager = ResidencyManager(
        {
            PAGEABLE_HOST: value.storage_nbytes,
            PINNED_HOST: value.storage_nbytes,
        }
    )
    handle = manager.adopt(value)
    action = EnsureResident(handle, PINNED_HOST)
    plan = ResidencyPlan("infeasible-exclusive", enter=(action,))

    explanation = manager.explain(plan)
    assert not explanation.feasible
    assert "EXCLUSIVE" in (explanation.reason or "")
    before = manager.snapshot()
    with pytest.raises(ResidencyPlanError, match="EXCLUSIVE"):
        manager.execute_actions((action,), name=plan.name)
    after = manager.snapshot()
    assert before.values == after.values
    assert before.locations == after.locations
    assert manager.trace() == ()

    with pytest.raises(ResidencyOwnershipError, match="EXCLUSIVE"):
        manager.ensure(handle, PINNED_HOST)
    assert manager.trace() == ()
    manager.close()


def test_close_rejects_active_lifetimes_then_closes_idempotently() -> None:
    value = _plaintext()
    manager = ResidencyManager({PAGEABLE_HOST: 2 * value.storage_nbytes})
    handle = manager.adopt(value)
    hold = manager.hold((handle,), at=PAGEABLE_HOST)
    reservation = manager.reserve(PAGEABLE_HOST, 0, label="zero")

    with pytest.raises(ResidencyInUseError, match="holds=1"):
        manager.close()
    reservation.release()
    hold.release()
    manager.close()
    manager.close()
    with pytest.raises(ResidencyClosedError):
        manager.snapshot()
    with pytest.raises(ResidencyClosedError):
        manager.adopt(_plaintext())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_out_of_range_cuda_endpoint_is_purely_rejected() -> None:
    value = _plaintext()
    manager = ResidencyManager(trace_capacity=4)
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    invalid = cuda_location(f"cuda:{torch.cuda.device_count()}")
    action = EnsureResident(handle, invalid)
    locations_before = manager.locations
    trace_before = manager.trace()

    with pytest.raises(ValueError, match="not available"):
        manager.ensure(handle, invalid)
    explanation = manager.explain(
        ResidencyPlan("invalid-cuda", enter=(action,))
    )
    assert not explanation.feasible
    assert (
        explanation.reason is not None and "not available" in explanation.reason
    )
    with pytest.raises(ResidencyPlanError, match="not available"):
        manager.execute_actions((action,), name="invalid-cuda")
    with pytest.raises(ValueError, match="not available"):
        manager.reserve(invalid, 0, label="invalid")
    with pytest.raises(ValueError, match="not available"):
        manager.register_source(
            _spec(value),
            _Source(value.clone),
            source_location=invalid,
        )

    assert manager.locations == locations_before
    assert manager.trace() == trace_before
    assert len(manager.snapshot().values) == 1
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pinned_pageable_invariants_and_tensor_resident_conversions() -> None:
    pageable = _plaintext()
    assert pageable.is_cpu
    assert not pageable.is_pinned
    assert pageable.cpu() is pageable
    pageable_copy = pageable.cpu(copy=True)
    assert pageable_copy is not pageable
    assert not pageable_copy.is_pinned

    pinned = pageable.pin_memory()
    assert pinned.is_cpu
    assert pinned.is_pinned
    assert pinned.pin_memory() is pinned
    pinned_copy = pinned.pin_memory(copy=True)
    assert pinned_copy is not pinned
    assert pinned_copy.is_pinned
    converted = pinned.cpu()
    assert converted is not pinned
    assert not converted.is_pinned
    torch.testing.assert_close(converted.message, pageable.message)

    manager = ResidencyManager()
    with pytest.raises(ResidencyMaterializationError, match="not fully pinned"):
        manager.adopt(pageable.clone(), at=PINNED_HOST)
    with pytest.raises(ResidencyMaterializationError, match="unexpectedly"):
        manager.adopt(pinned, at=PAGEABLE_HOST)

    mixed = _PairResident(
        torch.arange(2, dtype=torch.float64),
        torch.arange(2, dtype=torch.float64).pin_memory(),
    )
    with pytest.raises(ResidencyMaterializationError, match="mixed"):
        manager.adopt(mixed)

    exclusive = manager.adopt(pageable)
    move = manager.move(exclusive, PINNED_HOST, from_location=PAGEABLE_HOST)
    assert isinstance(move.action, MoveResident)
    assert move.source == PAGEABLE_HOST
    assert move.destination == PINNED_HOST
    assert _location_snapshot(manager, PAGEABLE_HOST).budget_bytes is None
    assert _location_snapshot(manager, PINNED_HOST).budget_bytes is None
    with manager.acquire((exclusive,), at=PINNED_HOST) as values:
        assert values[exclusive].is_pinned
    with pytest.raises(ResidencyUnavailableError):
        manager.acquire((exclusive,), at=PAGEABLE_HOST)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_view_backed_value_transitions_use_a_fixed_conservative_charge() -> (
    None
):
    backing = torch.arange(8, dtype=torch.float64)
    value = Plaintext(message=backing[::2], level=0, scale=16.0)
    charge = value.storage_nbytes
    manager = ResidencyManager(
        {PAGEABLE_HOST: 2 * charge, PINNED_HOST: 2 * charge}
    )
    handle = manager.adopt(
        value,
        replica_mode=ReplicaMode.REPLICABLE,
    )

    manager.copy(handle, PINNED_HOST)
    state = _value_snapshot(manager, handle)
    materializations = {item.location: item for item in state.materializations}
    assert materializations[PAGEABLE_HOST].storage_nbytes == charge
    assert materializations[PINNED_HOST].storage_nbytes <= charge
    assert materializations[PAGEABLE_HOST].charged_nbytes == charge
    assert materializations[PINNED_HOST].charged_nbytes == charge
    assert _location_snapshot(manager, PAGEABLE_HOST).used_bytes == charge
    assert _location_snapshot(manager, PINNED_HOST).used_bytes == charge
    with manager.acquire((handle,), at=PINNED_HOST) as values:
        torch.testing.assert_close(values[handle].message, backing[::2])
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_replicable_copy_and_move_across_host_and_cuda_locations() -> None:
    pageable = _plaintext(tuple(float(index) for index in range(8)))
    nbytes = pageable.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    budgets = {
        PAGEABLE_HOST: 3 * nbytes,
        PINNED_HOST: 3 * nbytes,
        cuda_zero: 3 * nbytes,
    }
    if torch.cuda.device_count() >= 2:
        budgets[cuda_location("cuda:1")] = 3 * nbytes
    manager = ResidencyManager(budgets)
    handle = manager.adopt(
        pageable,
        replica_mode=ReplicaMode.REPLICABLE,
    )

    manager.copy(handle, PINNED_HOST)
    with manager.acquire((handle,), at=PINNED_HOST) as values:
        pinned = values[handle]
        assert pinned.is_pinned
        torch.testing.assert_close(pinned.message, pageable.message)

    cuda_report = manager.ensure(handle, cuda_zero)
    allocator_metrics = (
        cuda_report.allocator_allocated_bytes_before,
        cuda_report.allocator_reserved_bytes_before,
        cuda_report.allocator_allocated_bytes_after,
        cuda_report.allocator_reserved_bytes_after,
    )
    assert all(isinstance(item, int) for item in allocator_metrics)
    cuda_snapshot = _location_snapshot(manager, cuda_zero)
    assert isinstance(cuda_snapshot.allocator_allocated_bytes, int)
    assert isinstance(cuda_snapshot.allocator_reserved_bytes, int)
    with manager.acquire(
        (handle,),
        at=cuda_zero,
        consumer_stream=torch.cuda.current_stream(0),
    ) as values:
        device_value = values[handle]
        assert device_value.device == torch.device("cuda:0")
        assert device_value.message is not None
        torch.testing.assert_close(device_value.message.cpu(), pageable.message)

    if torch.cuda.device_count() >= 2:
        cuda_one = cuda_location("cuda:1")
        move = manager.move(handle, cuda_one, from_location=cuda_zero)
        assert move.source == cuda_zero
        assert move.destination == cuda_one
        with pytest.raises(ResidencyUnavailableError):
            manager.acquire(
                (handle,),
                at=cuda_zero,
                consumer_stream=torch.cuda.current_stream(0),
            )
        with manager.acquire(
            (handle,),
            at=cuda_one,
            consumer_stream=torch.cuda.current_stream(1),
        ) as values:
            assert values[handle].device == torch.device("cuda:1")

    manager.move(handle, PAGEABLE_HOST, from_location=PINNED_HOST)
    with pytest.raises(ResidencyUnavailableError):
        manager.acquire((handle,), at=PINNED_HOST)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Two GPUs are required"
)
def test_wrong_device_streams_are_rejected_without_mutation() -> None:
    value = _plaintext()
    nbytes = value.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager(
        {PAGEABLE_HOST: 2 * nbytes, cuda_zero: 2 * nbytes}
    )
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    wrong_stream = torch.cuda.Stream(device=1)

    with pytest.raises(ValueError, match="does not match"):
        manager.ensure(handle, cuda_zero, stream=wrong_stream)
    assert _location_snapshot(manager, cuda_zero).used_bytes == 0

    manager.ensure(handle, cuda_zero)
    with pytest.raises(ValueError, match="explicit consumer_stream"):
        manager.acquire((handle,), at=cuda_zero)
    with pytest.raises(ValueError, match="does not match"):
        manager.acquire((handle,), at=cuda_zero, consumer_stream=wrong_stream)
    lease = manager.acquire(
        (handle,),
        at=cuda_zero,
        consumer_stream=torch.cuda.current_stream(0),
    )
    with pytest.raises(ValueError, match="wrong CUDA device"):
        lease.add_consumer_stream(wrong_stream)
    lease.release(wait=True)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_wrong_stream_types_are_rejected_without_mutation() -> None:
    value = _plaintext()
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager()
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)

    with pytest.raises(TypeError, match="torch.cuda.Stream"):
        manager.ensure(handle, cuda_zero, stream=object())  # type: ignore[arg-type]
    assert _value_snapshot(manager, handle).materializations[0].location == (
        PAGEABLE_HOST
    )

    manager.ensure(handle, cuda_zero)
    with pytest.raises(TypeError, match="torch.cuda.Stream"):
        manager.acquire(
            (handle,),
            at=cuda_zero,
            consumer_stream=object(),  # type: ignore[arg-type]
        )
    lease = manager.acquire(
        (handle,),
        at=cuda_zero,
        consumer_stream=torch.cuda.current_stream(0),
    )
    with pytest.raises(TypeError, match="torch.cuda.Stream"):
        lease.add_consumer_stream(object())  # type: ignore[arg-type]
    lease.release(wait=True)
    manager.drop(handle, cuda_zero)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_lease_release_tracks_pending_consumer_event_until_reaped() -> (
    None
):
    value = _plaintext()
    nbytes = value.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager(
        {PAGEABLE_HOST: 2 * nbytes, cuda_zero: 2 * nbytes}
    )
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    manager.ensure(handle, cuda_zero)

    consumer = torch.cuda.Stream(device=0)
    lease = manager.acquire((handle,), at=cuda_zero, consumer_stream=consumer)
    borrowed = lease.values
    with torch.cuda.stream(consumer):
        borrowed_value = borrowed[handle]
        assert borrowed_value.message is not None
        observed = borrowed_value.message + 1.0
        torch.cuda._sleep(500_000_000)
    lease.release(wait=False)
    assert not lease.active

    pending_value = _value_snapshot(manager, handle).materializations
    cuda_materialization = next(
        item for item in pending_value if item.location == cuda_zero
    )
    assert cuda_materialization.use_count == 0
    assert cuda_materialization.pending_event_count == 1
    assert _location_snapshot(manager, cuda_zero).pending_event_count == 1
    with pytest.raises(ResidencyInUseError, match="pending_events=1"):
        manager.drop(handle, cuda_zero)

    consumer.synchronize()
    assert value.message is not None
    torch.testing.assert_close(observed.cpu(), value.message + 1.0)

    leases_before = tuple(manager._leases)  # type: ignore[attr-defined]
    locations_before = manager.locations
    trace_before = manager.trace()
    explanation = manager.explain(
        ResidencyPlan(
            "drop-after-completed-consumer",
            enter=(DropResident(handle, cuda_zero),),
        )
    )
    assert explanation.feasible
    assert tuple(manager._leases) == leases_before  # type: ignore[attr-defined]
    assert manager.locations == locations_before
    assert manager.trace() == trace_before

    manager.drop(handle, cuda_zero)
    assert _location_snapshot(manager, cuda_zero).used_bytes == 0
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_lease_release_uses_captured_stream_across_python_threads() -> (
    None
):
    value = _plaintext()
    nbytes = value.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager(
        {PAGEABLE_HOST: 2 * nbytes, cuda_zero: 2 * nbytes}
    )
    handle = manager.adopt(
        value,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    manager.ensure(handle, cuda_zero)
    consumer = torch.cuda.Stream(device=0)
    lease = manager.acquire((handle,), at=cuda_zero, consumer_stream=consumer)
    with torch.cuda.stream(consumer):
        resident = lease.values[handle]
        assert resident.message is not None
        observed = resident.message + 3.0
        torch.cuda._sleep(50_000_000)
    del resident
    failures: list[BaseException] = []

    def release() -> None:
        try:
            lease.release(wait=True)
        except BaseException as error:
            failures.append(error)

    thread = Thread(target=release)
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert failures == []
    assert consumer.query()
    assert value.message is not None
    torch.testing.assert_close(observed.cpu(), value.message + 3.0)
    manager.drop(handle, cuda_zero)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_abandoned_cuda_lease_synchronizes_before_last_manager_release() -> (
    None
):
    value = _plaintext()
    nbytes = value.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager(
        {PAGEABLE_HOST: 2 * nbytes, cuda_zero: 2 * nbytes}
    )
    handle = manager.adopt(
        value,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    manager.ensure(handle, cuda_zero)
    consumer = torch.cuda.Stream(device=0)
    lease = manager.acquire((handle,), at=cuda_zero, consumer_stream=consumer)
    with torch.cuda.stream(consumer):
        resident = lease.values[handle]
        assert resident.message is not None
        observed = resident.message + 2.0
        torch.cuda._sleep(50_000_000)
    del resident
    manager_ref = ref(manager)
    lease_ref = ref(lease)
    del manager

    with pytest.warns(ResourceWarning, match="active lease"):
        del lease
        collect()

    assert lease_ref() is None
    assert manager_ref() is None
    assert value.message is not None
    torch.testing.assert_close(observed.cpu(), value.message + 2.0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_drop_report_samples_post_transition_allocator_state() -> None:
    value = _plaintext(tuple(float(index) for index in range(4096)))
    nbytes = value.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager(
        {PAGEABLE_HOST: 2 * nbytes, cuda_zero: 2 * nbytes}
    )
    handle = manager.register_source(
        _spec(value),
        _Source(value.clone),
    )
    manager.ensure(handle, cuda_zero)
    torch.cuda.synchronize(0)

    report = manager.drop(handle, cuda_zero)
    actual_after = int(torch.cuda.memory_allocated(0))
    assert report.allocator_device == torch.device("cuda:0")
    assert report.allocator_allocated_bytes_after == actual_after
    assert report.allocator_allocated_bytes_before is not None
    assert report.allocator_allocated_bytes_before >= actual_after
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Two GPUs are required"
)
def test_cuda_devices_have_isolated_location_capacity_and_accounting() -> None:
    value = _plaintext()
    nbytes = value.storage_nbytes
    cuda_zero = cuda_location("cuda:0")
    cuda_one = cuda_location("cuda:1")
    manager = ResidencyManager(
        {
            PAGEABLE_HOST: 2 * nbytes,
            cuda_zero: nbytes,
            cuda_one: nbytes,
        }
    )
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    manager.ensure(handle, cuda_zero)
    manager.ensure(handle, cuda_one)

    zero = _location_snapshot(manager, cuda_zero)
    one = _location_snapshot(manager, cuda_one)
    assert zero.location != one.location
    assert zero.used_bytes == nbytes
    assert one.used_bytes == nbytes
    assert zero.remaining_budget_bytes == 0
    assert one.remaining_budget_bytes == 0
    with manager.acquire(
        (handle,),
        at=cuda_zero,
        consumer_stream=torch.cuda.current_stream(0),
    ) as values:
        assert values[handle].device == torch.device("cuda:0")
    with manager.acquire(
        (handle,),
        at=cuda_one,
        consumer_stream=torch.cuda.current_stream(1),
    ) as values:
        assert values[handle].device == torch.device("cuda:1")
    manager.close()


def test_state_version_tracks_managed_mutations_but_not_observation_or_no_op() -> (
    None
):
    value = _plaintext()
    manager = ResidencyManager()
    assert manager.state_version == 0

    handle = manager.register_source(_spec(value), _Source(value.clone))
    registered_version = manager.state_version
    assert registered_version > 0
    snapshot = manager.snapshot()
    assert snapshot.state_version == registered_version

    plan = ResidencyPlan(
        "version-observation",
        enter=(EnsureResident(handle, PAGEABLE_HOST),),
    )
    manager.explain(plan, expected_state_version=registered_version)
    assert manager.state_version == registered_version

    manager.ensure(handle, PAGEABLE_HOST)
    resident_version = manager.state_version
    assert resident_version > registered_version
    manager.ensure(handle, PAGEABLE_HOST)
    assert manager.state_version == resident_version

    lease = manager.acquire((handle,), at=PAGEABLE_HOST)
    acquired_version = manager.state_version
    assert acquired_version > resident_version
    lease.release()
    assert manager.state_version > acquired_version
    manager.close()


def test_prepared_state_version_is_checked_before_explain_or_commit_mutation() -> (
    None
):
    value = _plaintext()
    manager = ResidencyManager()
    handle = manager.register_source(_spec(value), _Source(value.clone))
    prepared = manager.snapshot().state_version
    plan = ResidencyPlan(
        "stale-decision",
        enter=(EnsureResident(handle, PAGEABLE_HOST),),
    )

    reservation = manager.reserve(PAGEABLE_HOST, 0, label="state-change")
    current = manager.state_version
    assert current > prepared
    with pytest.raises(ResidencyStaleStateError) as explain_error:
        manager.explain(plan, expected_state_version=prepared)
    assert explain_error.value.expected_version == prepared
    assert explain_error.value.actual_version == current
    with pytest.raises(ResidencyStaleStateError):
        with manager.scope(plan, expected_state_version=prepared):
            pass
    with pytest.raises(ResidencyStaleStateError):
        manager.execute_actions(
            plan.enter,
            name=plan.name,
            expected_state_version=prepared,
        )
    assert _value_snapshot(manager, handle).materializations == ()
    reservation.release()
    manager.close()


def test_reclaim_runs_before_reservation_admission_and_enter_actions() -> None:
    first_value = _plaintext()
    second_value = _plaintext((4.0, 5.0, 6.0, 7.0))
    nbytes = first_value.storage_nbytes
    manager = ResidencyManager({PAGEABLE_HOST: 2 * nbytes})
    first = manager.register_source(
        _spec(first_value), _Source(first_value.clone)
    )
    second = manager.register_source(
        _spec(second_value), _Source(second_value.clone)
    )
    manager.ensure(first, PAGEABLE_HOST)
    reclaim = DropResident(first, PAGEABLE_HOST)
    enter = EnsureResident(second, PAGEABLE_HOST)
    plan = ResidencyPlan(
        "reclaim-before-admission",
        reclaim=(reclaim,),
        enter=(enter,),
        exit=(DropResident(second, PAGEABLE_HOST),),
        reservations=(MemoryReservation(PAGEABLE_HOST, nbytes, "workspace"),),
    )

    explanation = manager.explain(plan)
    assert explanation.feasible
    assert tuple(item.action for item in explanation.actions) == (
        reclaim,
        enter,
        plan.exit[0],
    )
    assert dict(explanation.predicted_peak_bytes)[PAGEABLE_HOST] == 2 * nbytes

    scope = manager.scope(plan)
    with scope:
        location = _location_snapshot(manager, PAGEABLE_HOST)
        assert location.used_bytes == nbytes
        assert location.reserved_bytes == nbytes
        assert _value_snapshot(manager, first).materializations == ()
        assert len(_value_snapshot(manager, second).materializations) == 1
    assert scope.report is not None
    assert tuple(item.action for item in scope.report.transitions) == (
        reclaim,
        enter,
        plan.exit[0],
    )
    manager.close()


def test_reclaim_runtime_failure_reports_reclaim_phase_and_committed_prefix() -> (
    None
):
    first_value = _plaintext()
    second_value = _plaintext((4.0, 5.0, 6.0, 7.0))

    class FailingSource:
        def load(self) -> Plaintext:
            raise OSError("injected reclaim source failure")

    manager = ResidencyManager()
    first = manager.register_source(
        _spec(first_value), _Source(first_value.clone)
    )
    second = manager.register_source(_spec(second_value), FailingSource())
    manager.ensure(first, PAGEABLE_HOST)
    completed = DropResident(first, PAGEABLE_HOST)
    failed = MoveResident(second, PINNED_HOST)
    plan = ResidencyPlan(
        "partial-reclaim",
        reclaim=(completed, failed),
    )

    with pytest.raises(ResidencyPlanExecutionError) as error_info:
        with manager.scope(plan):
            pass
    error = error_info.value
    assert error.phase == "reclaim"
    assert error.failed_action == failed
    assert error.failed_action_index == 1
    assert tuple(item.action for item in error.partial_report.transitions) == (
        completed,
    )
    assert _value_snapshot(manager, first).materializations == ()
    manager.close()


def test_reservation_runtime_failure_reports_reserve_phase_and_releases_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _plaintext()
    manager = ResidencyManager({PAGEABLE_HOST: 4 * value.storage_nbytes})
    handle = manager.register_source(_spec(value), _Source(value.clone))
    manager.ensure(handle, PAGEABLE_HOST)
    reclaim = DropResident(handle, PAGEABLE_HOST)
    first = MemoryReservation(PAGEABLE_HOST, 1, "admitted")
    failed = MemoryReservation(PAGEABLE_HOST, 1, "injected-failure")
    original_reserve = manager._reserve_locked

    def reserve_or_fail(
        location: ResidencyLocation,
        nbytes: int,
        label: str,
    ) -> object:
        if label == failed.label:
            raise MemoryError("injected reservation admission failure")
        return original_reserve(location, nbytes, label)

    monkeypatch.setattr(manager, "_reserve_locked", reserve_or_fail)
    plan = ResidencyPlan(
        "partial-reservation-admission",
        reclaim=(reclaim,),
        reservations=(first, failed),
    )

    with pytest.raises(ResidencyPlanExecutionError) as error_info:
        with manager.scope(plan):
            pass
    error = error_info.value
    assert error.phase == "reserve"
    assert error.failed_action is None
    assert error.failed_action_index is None
    assert error.failed_reservation == failed
    assert error.failed_reservation_index == 1
    assert tuple(item.action for item in error.partial_report.transitions) == (
        reclaim,
    )
    assert manager.snapshot().reservations == ()
    assert _value_snapshot(manager, handle).materializations == ()
    manager.close()


def test_transfer_stream_mapping_is_frozen_and_strictly_validated() -> None:
    manager = ResidencyManager()
    plan = ResidencyPlan("stream-validation")
    with pytest.raises(TypeError, match="transfer_streams.*mapping"):
        manager.scope(plan, transfer_streams=[])  # type: ignore[arg-type]
    scope = manager.scope(plan, transfer_streams={})
    with scope:
        pass
    with pytest.raises(TypeError, match="torch.cuda.Stream"):
        manager.execute_actions(
            (),
            transfer_streams={PAGEABLE_HOST: object()},  # type: ignore[dict-item]
        )
    manager.close()


@pytest.mark.pinned_memory
def test_direct_no_copy_paths_validate_stream_before_mutation() -> None:
    value = _plaintext()
    manager = ResidencyManager()
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    manager.ensure(handle, PINNED_HOST)
    before = _value_snapshot(manager, handle)
    before_version = manager.state_version

    with pytest.raises(TypeError, match="torch.cuda.Stream"):
        manager.ensure(
            handle,
            PAGEABLE_HOST,
            stream=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="torch.cuda.Stream"):
        manager.move(
            handle,
            PINNED_HOST,
            from_location=PAGEABLE_HOST,
            stream=object(),  # type: ignore[arg-type]
        )

    after = _value_snapshot(manager, handle)
    assert after.materializations == before.materializations
    assert manager.state_version == before_version
    manager.discard(handle)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_direct_host_no_op_rejects_cuda_stream() -> None:
    value = _plaintext()
    manager = ResidencyManager()
    handle = manager.adopt(value)
    before_version = manager.state_version
    stream = torch.cuda.Stream(device=0)

    with pytest.raises(ValueError, match="CUDA residency location"):
        manager.ensure(handle, PAGEABLE_HOST, stream=stream)

    assert manager.state_version == before_version
    assert tuple(
        item.location
        for item in _value_snapshot(manager, handle).materializations
    ) == (PAGEABLE_HOST,)
    manager.discard(handle)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Two CUDA devices are required",
)
def test_existing_cuda_destination_rejects_wrong_device_stream() -> None:
    value = _plaintext()
    manager = ResidencyManager()
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    cuda_one = cuda_location("cuda:1")
    stream_one = torch.cuda.Stream(device=1)
    manager.ensure(handle, cuda_one, stream=stream_one)
    torch.cuda.synchronize(1)
    manager.snapshot()
    before_version = manager.state_version
    stream_zero = torch.cuda.Stream(device=0)

    with pytest.raises(ValueError, match="does not match"):
        manager.ensure(handle, cuda_one, stream=stream_zero)
    with pytest.raises(ValueError, match="does not match"):
        manager.move(
            handle,
            cuda_one,
            from_location=PAGEABLE_HOST,
            stream=stream_zero,
        )

    assert manager.state_version == before_version
    assert {
        item.location
        for item in _value_snapshot(manager, handle).materializations
    } == {PAGEABLE_HOST, cuda_one}
    manager.discard(handle)
    manager.close()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_scope_uses_explicit_destination_transfer_stream() -> None:
    value = _plaintext()
    cuda_zero = cuda_location("cuda:0")
    manager = ResidencyManager()
    handle = manager.register_source(_spec(value), _Source(value.clone))
    stream = torch.cuda.Stream(device=0)
    plan = ResidencyPlan(
        "explicit-plan-transfer-stream",
        enter=(EnsureResident(handle, cuda_zero),),
        exit=(DropResident(handle, cuda_zero),),
    )

    with manager.scope(plan, transfer_streams={cuda_zero: stream}):
        with manager.acquire(
            (handle,),
            at=cuda_zero,
            consumer_stream=stream,
        ) as values:
            assert values[handle].device == torch.device("cuda:0")
    manager.close()
