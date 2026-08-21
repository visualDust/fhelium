#!/usr/bin/env python3

"""Execute a named FHE stage with explicit residency plans and CUDA leases.

The example adopts one replicated plaintext weight, one replicated rotation
key, and one exclusive request ciphertext. It then performs host/CUDA
transitions with optional pinned/CUDA budgets, dry-runs a stage plan with a
CUDA reservation, protects asynchronous CUDA readers with a consumer-stream
event, and reports manager and allocator accounting.
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

from fhelium.residency import (
    PAGEABLE_HOST,
    PINNED_HOST,
    DropResident,
    EnsureResident,
    MemoryReservation,
    MoveResident,
    ReplicaMode,
    ResidencyLocationSnapshot,
    ResidencyManager,
    ResidencyPlan,
    ResidencySnapshot,
    cuda_location,
)


def location_rows(snapshot: ResidencySnapshot) -> list[list[str]]:
    """Format managed storage and optional budgets from one snapshot."""

    return [
        [
            item.location.name,
            format_budget(item),
            format_remaining_budget(item),
            format_bytes(item.used_bytes),
            format_bytes(item.reserved_bytes),
            format_bytes(item.peak_used_bytes),
            format_bytes(item.peak_charged_bytes),
            str(item.use_count),
            str(item.pending_event_count),
            (
                "n/a"
                if item.allocator_allocated_bytes is None
                else format_bytes(item.allocator_allocated_bytes)
            ),
            (
                "n/a"
                if item.allocator_reserved_bytes is None
                else format_bytes(item.allocator_reserved_bytes)
            ),
        ]
        for item in snapshot.locations
    ]


def format_budget(snapshot: ResidencyLocationSnapshot) -> str:
    """Format one optional strict admission budget."""

    if snapshot.budget_bytes is None:
        return "unbudgeted"
    return format_bytes(snapshot.budget_bytes)


def format_remaining_budget(snapshot: ResidencyLocationSnapshot) -> str:
    """Format remaining strict budget or the unbudgeted marker."""

    if snapshot.remaining_budget_bytes is None:
        return "unbudgeted"
    return format_bytes(snapshot.remaining_budget_bytes)


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
        parser.error("this residency example requires CUDA")
    device_location = cuda_location(engine.device)

    positions = torch.arange(engine.num_slots, device="cpu")
    message = 0.01 * torch.sin(positions.to(torch.float64) * 0.017)
    weight_message = torch.full(
        (engine.num_slots,),
        0.5,
        dtype=torch.float64,
    )

    # Construct live FHElium values before transferring their logical
    # ownership to the manager.
    weight = engine.prepare_plaintext_for_multiplication(
        engine.encode(weight_message, level=0),
        modulus_basis="Q",
    ).cpu()
    rotation_key = engine.create_rotation_key(1, engine.secret_key).cpu()
    source = engine.encrypt_message(message)

    weight_bytes = max(weight.nbytes, weight.storage_nbytes)
    key_bytes = max(rotation_key.nbytes, rotation_key.storage_nbytes)
    ciphertext_logical_bytes = source.nbytes
    ciphertext_storage_bytes = source.storage_nbytes
    ciphertext_charge = max(
        ciphertext_logical_bytes,
        ciphertext_storage_bytes,
    )
    workspace_bytes = 3 * ciphertext_charge

    # Pageable host remains unbudgeted. Pinned host and CUDA use strict
    # application-selected admission budgets; the CUDA budget includes the
    # plan's reservation for unmanaged evaluator expansion.
    residency = ResidencyManager(
        budgets={
            PINNED_HOST: weight_bytes + ciphertext_charge,
            device_location: (
                weight_bytes + key_bytes + ciphertext_charge + workspace_bytes
            ),
        },
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
        at=device_location,
        replica_mode=ReplicaMode.EXCLUSIVE,
    )
    # adopt() transfers logical alias ownership under caller-enforced rules. Concrete aliases
    # must not be retained or used outside a residency lease.
    del weight, rotation_key, source

    # Direct primitive transitions are application decisions. ensure() keeps a
    # replica; move() changes the sole location of an EXCLUSIVE ciphertext.
    residency.ensure(weight_handle, PINNED_HOST)
    residency.move(
        source_handle,
        PINNED_HOST,
        from_location=device_location,
    )

    # The plan name expresses this application's stage and tile. The plan is
    # ordered low-level IR, and the reservation is accounted headroom for
    # unmanaged evaluator outputs/workspace rather than a tensor allocation.
    plan = ResidencyPlan(
        name="inference/rotate-scale/tile-0",
        enter=(
            EnsureResident(weight_handle, device_location),
            EnsureResident(key_handle, device_location),
            MoveResident(
                source_handle,
                device_location,
                from_location=PINNED_HOST,
            ),
        ),
        exit=(
            MoveResident(
                source_handle,
                PINNED_HOST,
                from_location=device_location,
            ),
            DropResident(key_handle, device_location),
            DropResident(weight_handle, device_location),
        ),
        reservations=(
            MemoryReservation(
                device_location,
                workspace_bytes,
                label="rotate/multiply outputs and workspace",
            ),
        ),
    )
    explanation = residency.explain(plan)
    if not explanation.feasible:
        raise RuntimeError(
            f"residency plan is infeasible: {explanation.reason}"
        )

    compute_stream = torch.cuda.Stream(device=engine.device)
    scope = residency.scope(plan)
    with scope:
        with (
            residency.acquire(
                (source_handle, key_handle, weight_handle),
                at=device_location,
                consumer_stream=compute_stream,
            ) as resident,
            torch.cuda.stream(compute_stream),
        ):
            rotated = engine.rotate_with_key(
                resident[source_handle],
                resident[key_handle],
            )
            output = engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(rotated),
                resident[weight_handle],
            )
            output = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(output)
            )

        # Lease release records a completion event on compute_stream. The
        # manager retains any unfinished CUDA readers without synchronizing the
        # whole device. This snapshot may already reap a fast completed event.
        released_snapshot = residency.snapshot()

        # Synchronize only the result-producing stream before host decryption.
        # The plan's exit actions are therefore safe when the scope closes.
        compute_stream.synchronize()
        del rotated

    if scope.report is None:
        raise RuntimeError("residency plan scope did not produce a report")

    expected = 0.5 * torch.roll(message, shifts=1)
    actual = engine.decrypt_message(output, is_real=True)
    error = error_stats(actual, expected)
    final_snapshot = residency.snapshot()

    print(f"Plan: {explanation.plan_name}")
    print_table(
        ["location", "predicted managed peak"],
        [
            [location.name, format_bytes(nbytes)]
            for location, nbytes in explanation.predicted_peak_bytes
        ],
    )
    print("\nAccounting after the CUDA lease was released:")
    print_table(
        [
            "location",
            "budget",
            "remaining",
            "used",
            "reserved",
            "peak used",
            "peak charged",
            "uses",
            "pending events",
            "torch allocated",
            "torch reserved",
        ],
        location_rows(released_snapshot),
    )
    print("\nAccounting after stage exit:")
    print_table(
        [
            "location",
            "budget",
            "remaining",
            "used",
            "reserved",
            "peak used",
            "peak charged",
            "uses",
            "pending events",
            "torch allocated",
            "torch reserved",
        ],
        location_rows(final_snapshot),
    )
    print(
        "\nSource logical payload: "
        f"{format_bytes(ciphertext_logical_bytes)}; "
        "source unique storage: "
        f"{format_bytes(ciphertext_storage_bytes)}; "
        "managed source charge: "
        f"{format_bytes(ciphertext_charge)}; "
        f"torch allocated: {format_bytes(torch.cuda.memory_allocated(engine.device))}; "
        f"torch reserved: {format_bytes(torch.cuda.memory_reserved(engine.device))}; "
        f"plan transitions: {len(scope.report.transitions)}; "
        f"max error: {error['max_abs']:.3e}"
    )

    # End the managed logical values, then close the manager. The
    # output is an ordinary unmanaged ciphertext owned by this application.
    residency.discard(source_handle)
    residency.discard(key_handle)
    residency.discard(weight_handle)
    residency.close()

    if error["max_abs"] > 1e-5:
        raise RuntimeError(f"residency example error too large: {error}")


if __name__ == "__main__":
    main()
