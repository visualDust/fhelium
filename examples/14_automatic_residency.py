#!/usr/bin/env python3

"""Run deterministic automatic Residency admission under managed CUDA pressure.

The example keeps one cold replicable plaintext cached on CUDA, decides a
state-bound placement for a different CKKS working set, and lets a deterministic
controller reclaim the cold replica before admitting CUDA workspace and input
materializations. It then executes a real rotate/multiply/rescale stage through
strict manager leases and verifies that requested endpoints remain cached.
"""

from __future__ import annotations

import argparse

import torch
from common import (
    add_engine_args,
    error_stats,
    format_bytes,
    make_engine,
    print_table,
)

from fhelium import TensorResident
from fhelium.residency import (
    PAGEABLE_HOST,
    PINNED_HOST,
    DeterministicTieredLRU,
    MemoryReservation,
    ReplicaMode,
    ResidencyController,
    ResidencyDecision,
    ResidencyHandle,
    ResidencyLocation,
    ResidencyManager,
    ResidencyRequest,
    ResidencyRequirement,
    ResidencySnapshot,
    cuda_location,
)


def _charge(value: TensorResident) -> int:
    """Return the conservative managed charge of one value."""

    return max(value.nbytes, value.storage_nbytes)


def _materialization_locations(
    snapshot: ResidencySnapshot,
    handle: ResidencyHandle,
) -> tuple[ResidencyLocation, ...]:
    """Return current locations for one handle in snapshot order."""

    value = next(item for item in snapshot.values if item.handle == handle)
    return tuple(item.location for item in value.materializations)


def _decision_rows(decision: ResidencyDecision) -> list[list[str]]:
    """Format controller-selected reclaim evidence without concrete values."""

    return [
        [
            str(item.rank),
            type(item.action).__name__,
            item.action.handle.handle_id[:8],
            item.released_location.name,
            format_bytes(item.released_nbytes),
            item.reason,
        ]
        for item in decision.evictions
    ]


def _location_rows(snapshot: ResidencySnapshot) -> list[list[str]]:
    """Format manager-local admission and accounting state."""

    return [
        [
            item.location.name,
            (
                "unbudgeted"
                if item.budget_bytes is None
                else format_bytes(item.budget_bytes)
            ),
            format_bytes(item.used_bytes),
            format_bytes(item.reserved_bytes),
            format_bytes(item.peak_charged_bytes),
            str(item.value_count),
            str(item.pending_event_count),
        ]
        for item in snapshot.locations
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(
        parser,
        default_preset="slots8192-scale40-levels7-int64",
        default_device="cuda:0",
    )
    args = parser.parse_args()

    engine = make_engine(args)
    if engine.device.type != "cuda":
        parser.error("this automatic residency example requires CUDA")
    cuda = cuda_location(engine.device)
    transfer_stream = torch.cuda.Stream(device=engine.device)
    compute_stream = torch.cuda.Stream(device=engine.device)

    positions = torch.arange(engine.num_slots, dtype=torch.float64)
    message = 0.01 * torch.sin(positions * 0.017)
    weight_message = torch.full_like(message, 0.5)
    cold_message = torch.full_like(message, 0.25)

    weight = engine.prepare_plaintext_for_multiplication(
        engine.encode(weight_message, level=0),
        modulus_basis="Q",
    ).cpu()
    cold = engine.prepare_plaintext_for_multiplication(
        engine.encode(cold_message, level=0),
        modulus_basis="Q",
    ).cpu()
    rotation_key = engine.create_rotation_key(1, engine.secret_key).cpu()
    source = engine.encrypt_message(message).pin_memory()

    weight_charge = _charge(weight)
    cold_charge = _charge(cold)
    key_charge = _charge(rotation_key)
    source_charge = _charge(source)
    workspace_bytes = 3 * source_charge
    cuda_budget = weight_charge + key_charge + source_charge + workspace_bytes

    residency = ResidencyManager(
        budgets={
            PINNED_HOST: source_charge,
            cuda: cuda_budget,
        }
    )
    cold_handle = residency.adopt(
        cold,
        at=PAGEABLE_HOST,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    weight_handle = residency.adopt(
        weight,
        at=PAGEABLE_HOST,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    key_handle = residency.adopt(
        rotation_key,
        at=PAGEABLE_HOST,
        replica_mode=ReplicaMode.REPLICABLE,
    )
    source_handle = residency.adopt(
        source,
        at=PINNED_HOST,
        replica_mode=ReplicaMode.EXCLUSIVE,
    )
    del cold, weight, rotation_key, source

    # Create deliberate pressure: the cold logical value keeps its pageable
    # replica while an otherwise unused managed CUDA replica occupies capacity.
    residency.ensure(cold_handle, cuda, stream=transfer_stream)

    policy = DeterministicTieredLRU(
        fallback_tiers={cuda: (PINNED_HOST, PAGEABLE_HOST)}
    )
    controller = ResidencyController(residency, policy=policy)
    requirements = (
        ResidencyRequirement(source_handle, cuda),
        ResidencyRequirement(key_handle, cuda),
        ResidencyRequirement(weight_handle, cuda),
    )
    request = ResidencyRequest(
        name="automatic/rotate-scale/tile-0",
        requirements=requirements,
        reservations=(
            MemoryReservation(
                cuda,
                workspace_bytes,
                label="rotate/multiply outputs and workspace",
            ),
        ),
    )

    # decide() reads tensor-free snapshots only. The returned decision records
    # ordered actions, policy evidence, predicted peaks, and its state precondition.
    decision = controller.decide(request)
    if decision.expected_state_version != residency.state_version:
        raise RuntimeError(
            "automatic residency decision became unexpectedly stale"
        )
    if not any(
        item.action.handle == cold_handle and item.released_location == cuda
        for item in decision.evictions
    ):
        raise RuntimeError(
            "automatic residency did not select the cold CUDA replica"
        )

    scope = controller.scope(
        decision,
        transfer_streams={cuda: transfer_stream},
    )
    with scope:
        # scope() performs placement and reservation admission. acquire()
        # remains the strict already-ready-only value-access operation.
        with (
            residency.acquire(
                (source_handle, key_handle, weight_handle),
                at=cuda,
                consumer_stream=compute_stream,
            ) as resident,
            torch.cuda.stream(compute_stream),
        ):
            source_value = resident[source_handle]
            key_value = resident[key_handle]
            weight_value = resident[weight_handle]
            rotated = engine.rotate_with_key(source_value, key_value)
            output = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(
                    engine.multiply_plaintext(
                        engine.coefficient_domain_to_ntt_domain(rotated),
                        weight_value,
                    )
                )
            )
            del source_value, key_value, weight_value

        # Lease release records a consumer-stream event. Keep workspace
        # headroom active until the result-producing stream has completed.
        compute_stream.synchronize()
        del rotated

    if scope.report is None:
        raise RuntimeError("automatic residency scope did not produce a report")

    expected = 0.5 * torch.roll(message, shifts=1)
    actual = engine.decrypt_message(output, is_real=True)
    error = error_stats(actual, expected)
    final_snapshot = residency.snapshot()

    cold_locations = _materialization_locations(final_snapshot, cold_handle)
    if cuda in cold_locations or PAGEABLE_HOST not in cold_locations:
        raise RuntimeError("cold value did not retain only its host-side cache")
    for requirement in requirements:
        if requirement.location not in _materialization_locations(
            final_snapshot,
            requirement.handle,
        ):
            raise RuntimeError(
                "automatic request endpoint was not retained after scope exit"
            )

    print(
        "Decision: "
        f"policy={decision.policy_name}; "
        f"state_version={decision.expected_state_version}; "
        f"search_states={decision.explored_states}; "
        f"reclaim={len(decision.plan.reclaim)}; "
        f"enter={len(decision.plan.enter)}"
    )
    print_table(
        ["rank", "action", "handle", "released", "bytes", "reason"],
        _decision_rows(decision),
    )
    print("\nPredicted managed peaks:")
    print_table(
        ["location", "peak charged"],
        [
            [location.name, format_bytes(nbytes)]
            for location, nbytes in decision.explanation.predicted_peak_bytes
        ],
    )
    print("\nFinal cached residency:")
    print_table(
        [
            "location",
            "budget",
            "used",
            "reserved",
            "peak charged",
            "values",
            "pending",
        ],
        _location_rows(final_snapshot),
    )
    print(
        "\nManaged CUDA budget: "
        f"{format_bytes(cuda_budget)}; "
        f"cold charge: {format_bytes(cold_charge)}; "
        f"workspace reservation: {format_bytes(workspace_bytes)}; "
        f"transitions: {len(scope.report.transitions)}; "
        f"max error: {error['max_abs']:.3e}"
    )

    residency.discard(source_handle)
    residency.discard(key_handle)
    residency.discard(weight_handle)
    residency.discard(cold_handle)
    residency.close()

    if error["max_abs"] > 1e-5:
        raise RuntimeError(
            f"automatic residency example error too large: {error}"
        )


if __name__ == "__main__":
    main()
