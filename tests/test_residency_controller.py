from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.pinned_memory

import torch

from fhelium import Plaintext
from fhelium.errors import (
    ResidencyHandleError,
    ResidencyLifetimeClosedError,
    ResidencyOwnershipError,
    ResidencyPlanError,
    ResidencySearchLimitError,
    ResidencyStaleStateError,
    ResidencyUnavailableError,
)
from fhelium.residency.controller import ResidencyController
from fhelium.residency.location import (
    PAGEABLE_HOST,
    PINNED_HOST,
    cuda_location,
)
from fhelium.residency.manager import ResidencyManager
from fhelium.residency.model import (
    Recoverability,
    ReplicaMode,
    ResidencyHandle,
    ResidencyValueSpec,
)
from fhelium.residency.plan import (
    DiscardValue,
    DropResident,
    EnsureResident,
    MemoryReservation,
    MoveResident,
)
from fhelium.residency.policy import (
    DeterministicTieredLRU,
    ResidencyEvictionCandidate,
    ResidencyPolicyMetadata,
)
from fhelium.residency.request import (
    ResidencyRequest,
    ResidencyRequirement,
)


def _plaintext(offset: float = 0.0) -> Plaintext:
    return Plaintext(
        message=torch.arange(8, dtype=torch.float64) + offset,
        level=0,
        scale=16.0,
    )


def _locations(
    manager: ResidencyManager,
    handle: ResidencyHandle[Any],
) -> tuple[Any, ...]:
    value = next(
        item for item in manager.snapshot().values if item.handle == handle
    )
    return tuple(item.location for item in value.materializations)


def _request(
    name: str,
    handle: ResidencyHandle[Any],
    *,
    at: Any = PINNED_HOST,
    reservation_bytes: int = 0,
) -> ResidencyRequest:
    reservations = (
        ()
        if reservation_bytes == 0
        else (
            MemoryReservation(
                at,
                reservation_bytes,
                f"{name} workspace",
            ),
        )
    )
    return ResidencyRequest(
        name,
        (ResidencyRequirement(handle, at),),
        reservations,
    )


def test_request_and_policy_definitions_are_strict_and_immutable() -> None:
    manager = ResidencyManager()
    handle = manager.adopt(_plaintext())

    requirements: Any = [ResidencyRequirement(handle, PAGEABLE_HOST)]
    reservations: Any = [MemoryReservation(PAGEABLE_HOST, 8, "workspace")]
    request = ResidencyRequest("request", requirements, reservations)
    requirements.clear()
    reservations.clear()
    assert len(request.requirements) == 1
    assert len(request.reservations) == 1
    assert isinstance(request.requirements, tuple)

    with pytest.raises(ValueError, match="at least one"):
        ResidencyRequest("empty", ())
    multi_location = ResidencyRequest(
        "replicas",
        (
            ResidencyRequirement(handle, PAGEABLE_HOST),
            ResidencyRequirement(handle, PINNED_HOST),
        ),
    )
    assert len(multi_location.requirements) == 2
    with pytest.raises(ValueError, match="unique.*handle, location"):
        ResidencyRequest(
            "duplicate",
            (
                ResidencyRequirement(handle, PAGEABLE_HOST),
                ResidencyRequirement(handle, PAGEABLE_HOST),
            ),
        )
    with pytest.raises(TypeError, match="ResidencyHandle"):
        ResidencyRequirement(object(), PAGEABLE_HOST)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ResidencyLocation"):
        ResidencyRequirement(handle, object())  # type: ignore[arg-type]

    policy = DeterministicTieredLRU({PINNED_HOST: (PAGEABLE_HOST,)})
    assert policy.fallback_locations(PINNED_HOST) == (PAGEABLE_HOST,)
    assert policy.fallback_locations(PAGEABLE_HOST) == ()
    with pytest.raises(ValueError, match="source location"):
        DeterministicTieredLRU({PAGEABLE_HOST: (PAGEABLE_HOST,)})

    candidates = (
        ResidencyEvictionCandidate(
            handle=handle,
            location=PAGEABLE_HOST,
            charged_nbytes=8,
            registration_index=0,
            last_access_epoch=2,
            metadata=ResidencyPolicyMetadata(priority=0, stable_key="old"),
        ),
        ResidencyEvictionCandidate(
            handle=ResidencyHandle(
                manager.manager_id,
                "synthetic-candidate",
                Plaintext,
            ),
            location=PAGEABLE_HOST,
            charged_nbytes=8,
            registration_index=1,
            last_access_epoch=1,
            metadata=ResidencyPolicyMetadata(priority=10, stable_key="new"),
        ),
    )
    assert policy.order_candidates(candidates) == candidates
    manager.close()


def test_decide_and_use_materialize_then_cache_values() -> None:
    value = _plaintext()
    nbytes = value.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    del value

    request = _request("cache-one", handle)
    decision = controller.decide(request)
    assert decision.expected_state_version == manager.state_version
    assert decision.explored_states >= 1
    assert decision.plan.reclaim == ()
    assert len(decision.plan.enter) == 1
    assert decision.evictions == ()
    assert decision.explanation.feasible

    use = controller.use(request)
    with use as admitted:
        borrowed = admitted.value(handle, at=PINNED_HOST)
        assert isinstance(borrowed, Plaintext)
        assert PINNED_HOST in _locations(manager, handle)
    assert use.report is not None
    with pytest.raises(RuntimeError, match="during context"):
        _ = use.values

    cached = controller.decide(request)
    assert cached.plan.reclaim == ()
    assert cached.plan.enter == ()
    manager.discard(handle)
    manager.close()


def test_reservation_headroom_is_reclaimed_before_admission() -> None:
    first = _plaintext(1.0)
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: 2 * nbytes})
    controller = ResidencyController(manager)
    first_handle = manager.adopt(
        first,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    second_handle = manager.adopt(
        second,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    del first, second
    manager.ensure(first_handle, PINNED_HOST)

    decision = controller.decide(
        _request(
            "reserved-stage",
            second_handle,
            reservation_bytes=nbytes,
        )
    )
    assert len(decision.plan.reclaim) == 1
    reclaim = decision.plan.reclaim[0]
    assert isinstance(reclaim, DropResident)
    assert reclaim.handle == first_handle
    assert decision.plan.reservations[0].nbytes == nbytes

    with controller.scope(decision):
        assert PINNED_HOST not in _locations(manager, first_handle)
        assert PINNED_HOST in _locations(manager, second_handle)
        location = next(
            item
            for item in manager.snapshot().locations
            if item.location == PINNED_HOST
        )
        assert location.used_bytes == nbytes
        assert location.reserved_bytes == nbytes

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_tiered_lru_evicts_oldest_unprotected_replica() -> None:
    values = tuple(_plaintext(float(index)) for index in range(3))
    nbytes = values[0].storage_nbytes
    manager = ResidencyManager({PINNED_HOST: 2 * nbytes})
    controller = ResidencyController(manager)
    handles = tuple(
        manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
        for value in values
    )
    del values

    with controller.use(_request("first", handles[0])):
        pass
    with controller.use(_request("second", handles[1])):
        pass

    decision = controller.decide(_request("third", handles[2]))
    assert len(decision.evictions) == 1
    action = decision.evictions[0].action
    assert isinstance(action, DropResident)
    assert action.handle == handles[0]
    assert all(
        not isinstance(item, DiscardValue) for item in decision.plan.reclaim
    )

    with controller.scope(decision):
        assert PINNED_HOST not in _locations(manager, handles[0])
        assert PINNED_HOST in _locations(manager, handles[1])
        assert PINNED_HOST in _locations(manager, handles[2])

    for handle in handles:
        manager.discard(handle)
    manager.close()


def test_policy_priority_overrides_lru_epoch() -> None:
    values = tuple(_plaintext(float(index)) for index in range(3))
    nbytes = values[0].storage_nbytes
    manager = ResidencyManager({PINNED_HOST: 2 * nbytes})
    controller = ResidencyController(manager)
    handles = tuple(
        manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
        for value in values
    )
    del values
    controller.set_policy_metadata(
        handles[0],
        priority=10,
        stable_key="retained/high-priority",
    )
    controller.set_policy_metadata(
        handles[1],
        priority=0,
        stable_key="evictable/normal-priority",
    )

    with controller.use(_request("older-high-priority", handles[0])):
        pass
    with controller.use(_request("newer-normal-priority", handles[1])):
        pass

    decision = controller.decide(_request("new-value", handles[2]))
    assert decision.evictions[0].action.handle == handles[1]

    for handle in handles:
        manager.discard(handle)
    manager.close()


def test_requested_handle_extra_replica_can_be_reclaimed() -> None:
    retained = _plaintext(1.0)
    incoming = _plaintext(2.0)
    nbytes = retained.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    retained_handle = manager.adopt(
        retained,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    incoming_handle = manager.adopt(
        incoming,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    del retained, incoming
    manager.ensure(retained_handle, PINNED_HOST)
    request = ResidencyRequest(
        "endpoint-protection",
        (
            ResidencyRequirement(retained_handle, PAGEABLE_HOST),
            ResidencyRequirement(incoming_handle, PINNED_HOST),
        ),
    )

    decision = controller.decide(request)
    assert decision.plan.reclaim == (
        DropResident(retained_handle, PINNED_HOST),
    )
    with controller.scope(decision):
        assert _locations(manager, retained_handle) == (PAGEABLE_HOST,)
        assert PINNED_HOST in _locations(manager, incoming_handle)

    manager.discard(retained_handle)
    manager.discard(incoming_handle)
    manager.close()


def test_ordered_exclusive_moves_release_capacity_for_later_requirement() -> (
    None
):
    first = _plaintext(1.0).pin_memory()
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes
    manager = ResidencyManager(
        {
            PAGEABLE_HOST: 2 * nbytes,
            PINNED_HOST: nbytes,
        }
    )
    controller = ResidencyController(manager)
    first_handle = manager.adopt(first, at=PINNED_HOST)
    second_handle = manager.adopt(second, at=PAGEABLE_HOST)
    del first, second
    request = ResidencyRequest(
        "ordered-exclusive-swap",
        (
            ResidencyRequirement(first_handle, PAGEABLE_HOST),
            ResidencyRequirement(second_handle, PINNED_HOST),
        ),
    )

    decision = controller.decide(request)
    assert decision.plan.reclaim == ()
    assert decision.plan.enter == (
        MoveResident(
            first_handle,
            PAGEABLE_HOST,
            from_location=PINNED_HOST,
        ),
        MoveResident(
            second_handle,
            PINNED_HOST,
            from_location=PAGEABLE_HOST,
        ),
    )
    with controller.scope(decision):
        assert _locations(manager, first_handle) == (PAGEABLE_HOST,)
        assert _locations(manager, second_handle) == (PINNED_HOST,)

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_multi_location_replicable_request_and_requirement_keyed_use() -> None:
    value = _plaintext()
    nbytes = value.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    del value
    pageable = ResidencyRequirement(handle, PAGEABLE_HOST)
    pinned = ResidencyRequirement(handle, PINNED_HOST)
    request = ResidencyRequest("two-replicas", (pageable, pinned))

    use = controller.use(request)
    with use:
        assert set(use.values) == {pageable, pinned}
        assert use.values[pageable].device.type == "cpu"
        assert use.values[pinned].is_pinned
        assert use.value(handle, at=PAGEABLE_HOST) is use.values[pageable]
        assert use.value(handle, at=PINNED_HOST) is use.values[pinned]

    manager.discard(handle)
    manager.close()


def test_multi_location_exclusive_request_is_rejected() -> None:
    manager = ResidencyManager()
    controller = ResidencyController(manager)
    handle = manager.adopt(_plaintext())
    request = ResidencyRequest(
        "impossible-exclusive-replicas",
        (
            ResidencyRequirement(handle, PAGEABLE_HOST),
            ResidencyRequirement(handle, PINNED_HOST),
        ),
    )

    with pytest.raises(ResidencyOwnershipError, match="EXCLUSIVE"):
        controller.decide(request)

    manager.discard(handle)
    manager.close()


def test_sole_must_preserve_value_moves_to_explicit_fallback() -> None:
    first = _plaintext(1.0).pin_memory()
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes
    manager = ResidencyManager(
        {
            PAGEABLE_HOST: 2 * nbytes,
            PINNED_HOST: nbytes,
        }
    )
    policy = DeterministicTieredLRU({PINNED_HOST: (PAGEABLE_HOST,)})
    controller = ResidencyController(manager, policy=policy)
    first_handle = manager.adopt(first, at=PINNED_HOST)
    second_handle = manager.adopt(second, at=PAGEABLE_HOST)
    del first, second

    decision = controller.decide(_request("fallback", second_handle))
    assert len(decision.plan.reclaim) == 1
    action = decision.plan.reclaim[0]
    assert isinstance(action, MoveResident)
    assert action.handle == first_handle
    assert action.from_location == PINNED_HOST
    assert action.to == PAGEABLE_HOST

    with controller.scope(decision):
        assert _locations(manager, first_handle) == (PAGEABLE_HOST,)
        assert _locations(manager, second_handle) == (PINNED_HOST,)

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_sole_must_preserve_without_fallback_fails_without_mutation() -> None:
    first = _plaintext(1.0).pin_memory()
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    first_handle = manager.adopt(first, at=PINNED_HOST)
    second_handle = manager.adopt(second, at=PAGEABLE_HOST)
    del first, second
    before = manager.snapshot()

    with pytest.raises(ResidencyPlanError, match="no legal drop or fallback"):
        controller.decide(_request("no-fallback", second_handle))
    after = manager.snapshot()
    assert after.state_version == before.state_version
    assert _locations(manager, first_handle) == (PINNED_HOST,)
    assert _locations(manager, second_handle) == (PAGEABLE_HOST,)

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_fallback_offload_accounts_for_destination_transfer_peak() -> None:
    first = _plaintext(1.0).pin_memory()
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes
    manager = ResidencyManager(
        {
            PAGEABLE_HOST: nbytes,
            PINNED_HOST: nbytes,
        }
    )
    controller = ResidencyController(
        manager,
        policy=DeterministicTieredLRU({PINNED_HOST: (PAGEABLE_HOST,)}),
    )
    first_handle = manager.adopt(first, at=PINNED_HOST)
    second_handle = manager.adopt(second, at=PAGEABLE_HOST)
    del first, second

    with pytest.raises(ResidencyPlanError, match="fallback"):
        controller.decide(_request("transfer-peak", second_handle))
    assert _locations(manager, first_handle) == (PINNED_HOST,)
    assert _locations(manager, second_handle) == (PAGEABLE_HOST,)

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_source_only_reconstruction_reclaims_source_location_headroom() -> None:
    victim = _plaintext(1.0)
    prototype = _plaintext(2.0)
    nbytes = victim.storage_nbytes

    class Source:
        def __init__(self) -> None:
            self.load_count = 0

        def load(self) -> Plaintext:
            self.load_count += 1
            return _plaintext(2.0)

    manager = ResidencyManager(
        {
            PAGEABLE_HOST: nbytes,
            PINNED_HOST: 2 * nbytes,
        }
    )
    controller = ResidencyController(manager)
    victim_handle = manager.adopt(
        victim,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    manager.ensure(victim_handle, PINNED_HOST)
    source = Source()
    source_handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=prototype.nbytes,
            storage_nbytes=prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        source,
        source_location=PAGEABLE_HOST,
    )
    del victim, prototype

    decision = controller.decide(_request("source-headroom", source_handle))
    assert source.load_count == 0
    assert decision.plan.reclaim == (
        DropResident(victim_handle, PAGEABLE_HOST),
    )
    with controller.scope(decision):
        assert source.load_count == 1
        assert _locations(manager, victim_handle) == (PINNED_HOST,)
        assert _locations(manager, source_handle) == (PINNED_HOST,)

    manager.discard(victim_handle)
    manager.discard(source_handle)
    manager.close()


def test_source_reconstruction_at_destination_uses_one_capacity_charge() -> (
    None
):
    prototype = _plaintext(3.0)
    nbytes = prototype.storage_nbytes

    class Source:
        def load(self) -> Plaintext:
            return _plaintext(3.0)

    manager = ResidencyManager({PAGEABLE_HOST: nbytes})
    controller = ResidencyController(manager)
    handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=prototype.nbytes,
            storage_nbytes=prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        Source(),
        source_location=PAGEABLE_HOST,
    )
    del prototype

    decision = controller.decide(
        _request("source-equals-destination", handle, at=PAGEABLE_HOST)
    )
    assert decision.plan.reclaim == ()
    with controller.scope(decision):
        location = next(
            item
            for item in manager.snapshot().locations
            if item.location == PAGEABLE_HOST
        )
        assert location.used_bytes == nbytes
        assert location.reserved_bytes == 0

    manager.discard(handle)
    manager.close()


def test_source_backed_reclaim_can_move_instead_of_drop() -> None:
    prototype = _plaintext(3.5)
    nbytes = prototype.storage_nbytes

    class Source:
        def load(self) -> Plaintext:
            return _plaintext(3.5)

    manager = ResidencyManager(
        {
            PAGEABLE_HOST: nbytes,
            PINNED_HOST: nbytes,
        }
    )
    controller = ResidencyController(manager)
    handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=prototype.nbytes,
            storage_nbytes=prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        Source(),
        source_location=PAGEABLE_HOST,
    )
    manager.ensure(handle, PAGEABLE_HOST)
    del prototype

    request = ResidencyRequest(
        "move-source-backed-victim",
        (ResidencyRequirement(handle, PINNED_HOST),),
        (
            MemoryReservation(
                PAGEABLE_HOST,
                nbytes,
                "pageable workspace",
            ),
        ),
    )
    decision = controller.decide(request)
    assert decision.plan.reclaim == (
        MoveResident(handle, PINNED_HOST, from_location=PAGEABLE_HOST),
    )
    assert decision.plan.enter == ()
    with controller.scope(decision):
        assert _locations(manager, handle) == (PINNED_HOST,)

    manager.discard(handle)
    manager.close()


def test_reclaim_search_backtracks_after_locally_successful_victim() -> None:
    must_preserve = _plaintext(1.0).pin_memory()
    droppable_prototype = _plaintext(2.0).pin_memory()
    protected = _plaintext(3.0)
    requested_prototype = _plaintext(4.0)
    nbytes = must_preserve.storage_nbytes

    class PinnedSource:
        def load(self) -> Plaintext:
            return _plaintext(2.0).pin_memory()

    class PageableSource:
        def load(self) -> Plaintext:
            return _plaintext(4.0)

    manager = ResidencyManager(
        {
            PINNED_HOST: 2 * nbytes,
            PAGEABLE_HOST: 2 * nbytes,
        }
    )
    controller = ResidencyController(
        manager,
        policy=DeterministicTieredLRU({PINNED_HOST: (PAGEABLE_HOST,)}),
    )
    first_candidate = manager.adopt(must_preserve, at=PINNED_HOST)
    second_candidate = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=droppable_prototype.nbytes,
            storage_nbytes=droppable_prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        PinnedSource(),
        source_location=PINNED_HOST,
    )
    manager.ensure(second_candidate, PINNED_HOST)
    protected_handle = manager.adopt(protected, at=PAGEABLE_HOST)
    requested_handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=requested_prototype.nbytes,
            storage_nbytes=requested_prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        PageableSource(),
        source_location=PAGEABLE_HOST,
    )
    del must_preserve, droppable_prototype, protected, requested_prototype
    request = ResidencyRequest(
        "backtracking",
        (
            ResidencyRequirement(protected_handle, PAGEABLE_HOST),
            ResidencyRequirement(requested_handle, PINNED_HOST),
        ),
    )

    decision = controller.decide(request)
    assert decision.plan.reclaim == (
        DropResident(second_candidate, PINNED_HOST),
    )
    assert all(
        action.handle != first_candidate for action in decision.plan.reclaim
    )
    with controller.scope(decision):
        assert _locations(manager, first_candidate) == (PINNED_HOST,)
        assert _locations(manager, second_candidate) == ()
        assert _locations(manager, protected_handle) == (PAGEABLE_HOST,)
        assert _locations(manager, requested_handle) == (PINNED_HOST,)

    for handle in (
        first_candidate,
        second_candidate,
        protected_handle,
        requested_handle,
    ):
        manager.discard(handle)
    manager.close()


def test_required_endpoint_can_be_scratch_reclaimed_and_restored() -> None:
    a_prototype = _plaintext(1.0).pin_memory()
    b_value = _plaintext(2.0).pin_memory()
    c_value = _plaintext(3.0)
    nbytes = a_prototype.storage_nbytes

    class ASource:
        def load(self) -> Plaintext:
            return _plaintext(1.0).pin_memory()

    manager = ResidencyManager(
        {
            PINNED_HOST: 2 * nbytes,
            PAGEABLE_HOST: nbytes,
        }
    )
    controller = ResidencyController(manager)
    a_handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=a_prototype.nbytes,
            storage_nbytes=a_prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        ASource(),
        source_location=PINNED_HOST,
    )
    manager.ensure(a_handle, PINNED_HOST)
    b_handle = manager.adopt(b_value, at=PINNED_HOST)
    c_handle = manager.adopt(c_value, at=PAGEABLE_HOST)
    del a_prototype, b_value, c_value
    request = ResidencyRequest(
        "scratch-rotation",
        (
            ResidencyRequirement(c_handle, PINNED_HOST),
            ResidencyRequirement(b_handle, PAGEABLE_HOST),
            ResidencyRequirement(a_handle, PINNED_HOST),
        ),
    )

    decision = controller.decide(request)
    assert decision.plan.reclaim == (
        DropResident(a_handle, PINNED_HOST),
        MoveResident(c_handle, PINNED_HOST, from_location=PAGEABLE_HOST),
        MoveResident(b_handle, PAGEABLE_HOST, from_location=PINNED_HOST),
    )
    assert decision.plan.enter == (EnsureResident(a_handle, PINNED_HOST),)
    with controller.scope(decision):
        assert _locations(manager, a_handle) == (PINNED_HOST,)
        assert _locations(manager, b_handle) == (PAGEABLE_HOST,)
        assert _locations(manager, c_handle) == (PINNED_HOST,)

    manager.discard(a_handle)
    manager.discard(b_handle)
    manager.discard(c_handle)
    manager.close()


def test_impossible_nine_candidate_search_memoizes_canonical_states() -> None:
    values = tuple(_plaintext(float(index)) for index in range(9))
    nbytes = values[0].storage_nbytes
    manager = ResidencyManager({PINNED_HOST: 9 * nbytes})
    controller = ResidencyController(manager, search_state_limit=512)
    handles = tuple(
        manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
        for value in values
    )
    del values
    for handle in handles:
        manager.ensure(handle, PINNED_HOST)
    request = ResidencyRequest(
        "impossible-nine-candidate-search",
        (ResidencyRequirement(handles[0], PAGEABLE_HOST),),
        (
            MemoryReservation(
                PINNED_HOST,
                10 * nbytes,
                "intentionally larger than total pinned budget",
            ),
        ),
    )

    with pytest.raises(ResidencyPlanError) as error:
        controller.decide(request)
    assert not isinstance(error.value, ResidencySearchLimitError)

    for handle in handles:
        manager.discard(handle)
    manager.close()


def test_search_limit_reports_inconclusive_with_exact_evidence() -> None:
    victim = _plaintext(1.0)
    requested = _plaintext(2.0)
    nbytes = victim.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    victim_handle = manager.adopt(
        victim,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    manager.ensure(victim_handle, PINNED_HOST)
    requested_handle = manager.adopt(requested)
    del victim, requested
    request = _request("bounded-search", requested_handle)

    bounded = ResidencyController(manager, search_state_limit=1)
    with pytest.raises(ResidencySearchLimitError) as error_info:
        bounded.decide(request)
    assert error_info.value.request_name == "bounded-search"
    assert error_info.value.state_limit == 1
    assert error_info.value.explored_states == 1
    assert "inconclusive" in str(error_info.value)
    assert "infeasible" not in str(error_info.value)

    complete = ResidencyController(manager, search_state_limit=10)
    decision = complete.decide(request)
    assert 1 <= decision.explored_states <= 10
    assert decision.plan.reclaim == (DropResident(victim_handle, PINNED_HOST),)
    with complete.scope(decision):
        assert _locations(manager, requested_handle) == (PINNED_HOST,)

    manager.discard(victim_handle)
    manager.discard(requested_handle)
    manager.close()


def test_protected_materialization_is_not_an_automatic_victim() -> None:
    first = _plaintext(1.0)
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    first_handle = manager.adopt(
        first,
        at=PAGEABLE_HOST,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    second_handle = manager.adopt(
        second,
        at=PAGEABLE_HOST,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    del first, second
    manager.ensure(first_handle, PINNED_HOST)

    with (
        manager.hold((first_handle,), at=PINNED_HOST),
        pytest.raises(ResidencyPlanError, match="no invariant-valid victim"),
    ):
        controller.decide(_request("protected", second_handle))
    assert PINNED_HOST in _locations(manager, first_handle)

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_reconstructible_final_materialization_is_dropped_not_discarded() -> (
    None
):
    first = _plaintext(1.0).pin_memory()
    second = _plaintext(2.0)
    nbytes = first.storage_nbytes

    class Source:
        def load(self) -> Plaintext:
            return _plaintext(1.0).pin_memory()

    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    first_handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=first.nbytes,
            storage_nbytes=first.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        Source(),
        source_location=PINNED_HOST,
    )
    second_handle = manager.adopt(
        second,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    del first, second
    manager.ensure(first_handle, PINNED_HOST)

    decision = controller.decide(_request("reconstructible", second_handle))
    action = decision.plan.reclaim[0]
    assert isinstance(action, DropResident)
    assert action.handle == first_handle
    assert all(
        not isinstance(item, DiscardValue) for item in decision.plan.reclaim
    )

    with controller.scope(decision):
        assert _locations(manager, first_handle) == ()
        assert _locations(manager, second_handle) == (
            PAGEABLE_HOST,
            PINNED_HOST,
        )

    manager.discard(first_handle)
    manager.discard(second_handle)
    manager.close()


def test_decide_is_tensor_free_and_does_not_invoke_reconstruction_source() -> (
    None
):
    prototype = _plaintext(3.0)

    class Source:
        def __init__(self) -> None:
            self.load_count = 0

        def load(self) -> Plaintext:
            self.load_count += 1
            return _plaintext(3.0)

    source = Source()
    manager = ResidencyManager({PINNED_HOST: prototype.storage_nbytes})
    controller = ResidencyController(manager)
    handle = manager.register_source(
        ResidencyValueSpec(
            value_type=Plaintext,
            logical_nbytes=prototype.nbytes,
            storage_nbytes=prototype.storage_nbytes,
            replica_mode=ReplicaMode.REPLICABLE,
            recoverability=Recoverability.RECONSTRUCTIBLE,
        ),
        source,
    )
    del prototype

    decision = controller.decide(_request("lazy-source", handle))
    assert source.load_count == 0
    assert decision.explanation.feasible

    with controller.scope(decision):
        assert source.load_count == 1
        assert _locations(manager, handle) == (PINNED_HOST,)

    manager.discard(handle)
    manager.close()


def test_stale_decision_fails_before_any_plan_mutation() -> None:
    value = _plaintext()
    nbytes = value.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    del value
    decision = controller.decide(_request("stale", handle))

    reservation = manager.reserve(PAGEABLE_HOST, 1, label="invalidate decision")
    with pytest.raises(ResidencyStaleStateError):
        with controller.scope(decision):
            pass
    assert _locations(manager, handle) == (PAGEABLE_HOST,)
    reservation.release()

    manager.discard(handle)
    manager.close()


def test_controller_rejects_foreign_unknown_and_policy_mismatched_decisions() -> (
    None
):
    first = ResidencyManager()
    second = ResidencyManager()
    first_handle = first.adopt(_plaintext())
    second_handle = second.adopt(_plaintext(1.0))
    controller = ResidencyController(first)

    with pytest.raises(ResidencyHandleError, match="another manager"):
        controller.decide(_request("foreign", second_handle, at=PAGEABLE_HOST))

    unknown = ResidencyHandle(first.manager_id, "unknown", Plaintext)
    with pytest.raises(ResidencyHandleError, match="unknown"):
        controller.decide(_request("unknown", unknown, at=PAGEABLE_HOST))

    decision = controller.decide(
        _request("valid", first_handle, at=PAGEABLE_HOST)
    )
    other_policy = ResidencyController(
        first,
        policy=DeterministicTieredLRU({PINNED_HOST: (PAGEABLE_HOST,)}),
    )
    with pytest.raises(ResidencyPlanError, match="another policy"):
        other_policy.scope(decision)

    first.discard(first_handle)
    second.discard(second_handle)
    first.close()
    second.close()


def test_use_rejects_incomplete_cuda_streams_before_decision_or_mutation() -> (
    None
):
    manager = ResidencyManager()
    controller = ResidencyController(manager)
    handle = manager.adopt(_plaintext(), replica_mode=ReplicaMode.REPLICABLE)
    request = _request(
        "missing-consumer-stream",
        handle,
        at=cuda_location("cuda:0"),
    )
    before = manager.snapshot()

    with pytest.raises(ValueError, match="exactly cover required CUDA"):
        controller.use(request)
    after = manager.snapshot()
    assert after.state_version == before.state_version
    assert _locations(manager, handle) == (PAGEABLE_HOST,)

    manager.discard(handle)
    manager.close()


def test_use_retains_decision_scope_and_report_after_acquisition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _plaintext()
    nbytes = value.storage_nbytes
    manager = ResidencyManager({PINNED_HOST: nbytes})
    controller = ResidencyController(manager)
    handle = manager.adopt(value, replica_mode=ReplicaMode.REPLICABLE)
    del value
    use = controller.use(_request("acquire-failure", handle))

    def fail_acquire(*args: object, **kwargs: object) -> None:
        raise ResidencyUnavailableError("injected acquisition failure")

    monkeypatch.setattr(manager, "acquire", fail_acquire)
    with pytest.raises(ResidencyUnavailableError, match="injected"), use:
        pass

    assert use.decision.request.name == "acquire-failure"
    assert use.report is not None
    assert len(use.report.transitions) == 1
    assert use.exit_error is None
    assert PINNED_HOST in _locations(manager, handle)

    manager.discard(handle)
    manager.close()


def test_controller_exposes_only_one_decision_context_factory() -> None:
    manager = ResidencyManager()
    controller = ResidencyController(manager)
    assert callable(controller.decide)
    assert callable(controller.scope)
    assert not hasattr(controller, "prepare")
    assert not hasattr(controller, "last_search_state_count")
    assert not hasattr(controller, "commit")
    manager.close()


def test_use_borrow_mapping_closes_with_lease() -> None:
    manager = ResidencyManager()
    controller = ResidencyController(manager)
    handle = manager.adopt(_plaintext())
    use = controller.use(_request("borrow", handle, at=PAGEABLE_HOST))
    requirement = use.request.requirements[0]
    with use:
        values = use.values
        assert isinstance(values[requirement], Plaintext)
        assert use.value(handle, at=PAGEABLE_HOST) is values[requirement]
    with pytest.raises(ResidencyLifetimeClosedError):
        _ = values[requirement]

    manager.discard(handle)
    manager.close()
