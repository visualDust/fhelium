#!/usr/bin/env python3

"""Compare one homogeneous CKKS batch with an explicit per-message loop.

The evaluator computes a packed ``y = A @ x`` with the cyclic-diagonal
method.  FHElium preserves every leading message dimension as a homogeneous
batch prefix; the application still chooses whether to submit that batch or
to evaluate its members one at a time.

Example:
    python examples/15_homogeneous_batching.py \
        --preset slots8192-scale40-levels7-int64 --level 0 --batch-sizes 1,4,8
"""

from __future__ import annotations

import argparse
import gc
import statistics as stats
import time
from collections.abc import Callable
from typing import Any

import torch
from common import (
    add_engine_args,
    error_stats,
    format_bytes,
    make_engine,
    print_table,
    sync_if_cuda,
)

from fhelium import Ciphertext, CkksEngine, Plaintext, RotationKey


def _parse_batch_sizes(text: str) -> list[int]:
    batch_sizes = [
        int(item.strip()) for item in text.split(',') if item.strip()
    ]
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise argparse.ArgumentTypeError(
            "batch sizes must be a comma-separated list of positive integers"
        )
    return batch_sizes


def matrix_and_vectors(
    size: int,
    batch_size: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(size, dtype=torch.float64).view(-1, 1)
    column = torch.arange(size, dtype=torch.float64).view(1, -1)
    matrix = 0.018 * torch.sin((row + 1) * (column + 2) * 0.17)
    matrix += 0.007 * torch.cos((row + column + 1) * 0.23)
    generator = torch.Generator().manual_seed(seed)
    vectors = (
        torch.randn(
            (batch_size, size),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.025
    )
    return matrix, vectors


def periodic_slots(values: torch.Tensor, num_slots: int) -> torch.Tensor:
    return values.repeat(1, num_slots // values.size(-1))


def cyclic_diagonal_slots(
    matrix: torch.Tensor,
    rotation_step: int,
    num_slots: int,
) -> torch.Tensor:
    row = torch.arange(num_slots) % matrix.size(0)
    column = torch.remainder(row - rotation_step, matrix.size(0))
    return matrix[row, column]


def prepare_constants(
    engine: CkksEngine,
    matrix: torch.Tensor,
    *,
    level: int,
) -> tuple[list[Plaintext], dict[int, RotationKey]]:
    diagonals = [
        engine.prepare_plaintext_for_multiplication(
            engine.encode(
                cyclic_diagonal_slots(matrix, step, engine.num_slots),
                level=level,
            )
        )
        for step in range(matrix.size(0))
    ]
    rotation_keys = {
        step: engine.rotation_key(step) for step in range(1, matrix.size(0))
    }
    return diagonals, rotation_keys


def matrix_vector(
    source: Ciphertext,
    *,
    engine: CkksEngine,
    diagonals: list[Plaintext],
    rotation_keys: dict[int, RotationKey],
) -> Ciphertext:
    """Evaluate the same program for an unbatched or batched ciphertext."""

    rotated_values = []
    for step in range(len(diagonals)):
        rotated = (
            source
            if step == 0
            else engine.rotate_with_key(source, rotation_keys[step])
        )
        rotated_values.append(rotated)
    rotated_ntt = engine.coefficient_domain_to_ntt_domain(
        Ciphertext.stack_batch(rotated_values)
    )
    diagonal_batch = Plaintext.stack_batch(diagonals)
    if source.batch_shape:
        assert diagonal_batch.data is not None
        expanded = (
            diagonal_batch.data.reshape(
                diagonal_batch.batch_shape
                + (1,) * len(source.batch_shape)
                + diagonal_batch.data.shape[-2:]
            )
            .expand(
                diagonal_batch.batch_shape
                + source.batch_shape
                + diagonal_batch.data.shape[-2:]
            )
            .contiguous()
        )
        diagonal_batch = Plaintext(
            message=None,
            level=diagonal_batch.level,
            scale=diagonal_batch.scale,
            data=expanded,
            context_id=diagonal_batch.context_id,
            representation=diagonal_batch.representation,
            polynomial_domain=diagonal_batch.polynomial_domain,
            modulus_basis=diagonal_batch.modulus_basis,
            residue_representation=diagonal_batch.residue_representation,
            prime_ids=diagonal_batch.prime_ids,
        )
    weighted = engine.multiply_plaintext(rotated_ntt, diagonal_batch)
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.sum_ciphertext_batch(weighted)
        )
    )


def _time_call(fn: Callable[[], Any], device: torch.device) -> float:
    sync_if_cuda(device)
    start = time.perf_counter()
    result = fn()
    sync_if_cuda(device)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    del result
    return elapsed_ms


def alternating_benchmark(
    batched: Callable[[], Any],
    looped: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return paired medians while alternating first-run order."""

    for _ in range(warmup):
        _time_call(batched, device)
        _time_call(looped, device)

    batched_times = []
    looped_times = []
    for run_index in range(runs):
        if run_index % 2 == 0:
            batched_times.append(_time_call(batched, device))
            looped_times.append(_time_call(looped, device))
        else:
            looped_times.append(_time_call(looped, device))
            batched_times.append(_time_call(batched, device))
    return stats.median(batched_times), stats.median(looped_times)


def peak_allocated_mib(
    fn: Callable[[], Any],
    *,
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    sync_if_cuda(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = fn()
    sync_if_cuda(device)
    peak = torch.cuda.max_memory_allocated(device) - baseline
    del result
    gc.collect()
    torch.cuda.empty_cache()
    return peak / (1 << 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-levels7-int64")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=[1, 4, 8],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    engine = make_engine(args)
    if engine.device.type == "cuda":
        # The shared synchronization helper and allocator statistics follow
        # the process-current CUDA device.
        torch.cuda.set_device(engine.device)
    if args.size <= 0 or engine.num_slots % args.size != 0:
        parser.error(
            f"--size must be positive and divide num_slots={engine.num_slots}"
        )
    if not 0 <= args.level < engine.config.num_q_primes - 1:
        parser.error(
            "--level must leave at least one Q prime for the workload's "
            f"rescale; got {args.level} with "
            f"{engine.config.num_q_primes} Q primes"
        )
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be non-negative and --runs positive")

    matrix, _ = matrix_and_vectors(args.size, 1, seed=700)
    diagonals, rotation_keys = prepare_constants(
        engine,
        matrix,
        level=args.level,
    )

    active_q_rows = engine.config.num_q_primes - args.level
    active_qp_rows = active_q_rows + engine.config.num_p_primes
    qp_digit_bytes = active_qp_rows * engine.config.N * torch.int64.itemsize
    print(
        "Homogeneous batching benchmark\n"
        f"  preset={args.preset}, level={args.level}, "
        f"device={args.device}\n"
        f"  active rows: Q={active_q_rows}, QP={active_qp_rows}\n"
        "  one extended QP digit per message: "
        f"{format_bytes(qp_digit_bytes)}\n"
        "  policy: the application executes and compares and compares both "
        "paths"
    )

    rows = []
    for batch_size in args.batch_sizes:
        _, vectors = matrix_and_vectors(
            args.size,
            batch_size,
            seed=700 + batch_size,
        )
        source = engine.encrypt_message(
            periodic_slots(vectors, engine.num_slots),
            level=args.level,
        )
        # unbind_batch returns views. Clone here so the loop represents
        # independently owned, ordinary unbatched request values.
        individual_sources = tuple(
            value.clone() for value in source.unbind_batch()
        )

        def batched() -> Ciphertext:
            return matrix_vector(
                source,
                engine=engine,
                diagonals=diagonals,
                rotation_keys=rotation_keys,
            )

        def looped() -> list[Ciphertext]:
            return [
                matrix_vector(
                    value,
                    engine=engine,
                    diagonals=diagonals,
                    rotation_keys=rotation_keys,
                )
                for value in individual_sources
            ]

        batched_result = batched()
        looped_result = looped()
        stacked_loop = Ciphertext.stack_batch(looped_result)
        torch.testing.assert_close(
            batched_result.data,
            stacked_loop.data,
            rtol=0,
            atol=0,
        )
        actual = engine.decrypt_message(batched_result, is_real=True)[
            ..., : args.size
        ]
        error = error_stats(actual, vectors @ matrix.T)
        del batched_result, looped_result, stacked_loop, actual

        batched_ms, looped_ms = alternating_benchmark(
            batched,
            looped,
            warmup=args.warmup,
            runs=args.runs,
            device=engine.device,
        )
        speedup = looped_ms / batched_ms
        faster_path = "batch" if speedup >= 1.0 else "loop"
        batch_peak = peak_allocated_mib(batched, device=engine.device)
        loop_peak = peak_allocated_mib(looped, device=engine.device)
        rows.append(
            [
                batch_size,
                format_bytes(qp_digit_bytes * batch_size),
                f"{batched_ms:.4f}",
                f"{looped_ms:.4f}",
                f"{speedup:.3f}x",
                faster_path,
                "n/a" if batch_peak is None else f"{batch_peak:.1f}",
                "n/a" if loop_peak is None else f"{loop_peak:.1f}",
                f"{error['max_abs']:.3e}",
            ]
        )

    print_table(
        [
            "B",
            "B x QP digit",
            "batch ms",
            "loop ms",
            "loop/batch",
            "faster",
            "batch peak MiB",
            "loop peak MiB",
            "max abs error",
        ],
        rows,
    )
    print(
        "\nThe faster column describes only this measured point. FHElium "
        "does not select or cache an execution policy; keep the choice in "
        "application code and remeasure the deployed preset, level, batch "
        "size, device, and complete workload."
    )


if __name__ == "__main__":
    main()
