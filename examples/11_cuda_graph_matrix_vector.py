#!/usr/bin/env python3

"""Capture and replay packed CKKS matrix-vector multiplication.

The fixed evaluator computes ``y = A @ x`` with the cyclic-diagonal method.
The matrix, encoded diagonals, rotation keys, and operation schedule are static;
``functools.partial`` binds them into the captured callable.
each replay stages a newly encrypted vector into the program's dynamic input
buffer. Encryption and decryption remain outside capture.
"""

from __future__ import annotations

import argparse
from functools import partial

import torch
from common import (
    add_engine_args,
    error_stats,
    make_engine,
    print_table,
    time_ms,
)

from fhelium import Ciphertext, CkksEngine, Plaintext, RotationKey
from fhelium.execution import CudaGraphProgram


def matrix_and_vector(
    size: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(size, dtype=torch.float64).view(-1, 1)
    column = torch.arange(size, dtype=torch.float64).view(1, -1)
    matrix = 0.018 * torch.sin((row + 1) * (column + 2) * 0.17)
    matrix += 0.007 * torch.cos((row + column + 1) * 0.23)
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(size, generator=generator, dtype=torch.float64) * 0.025
    return matrix, vector


def periodic_slots(values: torch.Tensor, num_slots: int) -> torch.Tensor:
    return values.repeat(num_slots // values.numel())


def cyclic_diagonal_slots(
    matrix: torch.Tensor,
    rotation_step: int,
    num_slots: int,
) -> torch.Tensor:
    """Return weights aligned with ``torch.roll(x, rotation_step)``."""

    size = matrix.size(0)
    row = torch.arange(num_slots) % size
    column = torch.remainder(row - rotation_step, size)
    return matrix[row, column]


def prepare_constants(
    engine: CkksEngine,
    matrix: torch.Tensor,
) -> tuple[list[Plaintext], dict[int, RotationKey]]:
    diagonals = [
        engine.prepare_plaintext_for_multiplication(
            engine.encode(
                cyclic_diagonal_slots(matrix, step, engine.num_slots), level=0
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
    """Evaluate one vector with statically bound program state."""

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
    weighted = engine.multiply_plaintext(rotated_ntt, diagonal_batch)
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.sum_ciphertext_batch(weighted)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(
        parser,
        default_preset="slots8192-scale40-levels7-int64",
        default_device="cuda:0",
    )
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--capture-warmup", type=int, default=3)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    engine = make_engine(args)
    if engine.device.type != "cuda":
        parser.error("this CUDA Graph example requires CUDA")
    if args.size <= 0 or engine.num_slots % args.size != 0:
        parser.error(
            f"--size must be positive and divide num_slots={engine.num_slots}"
        )
    if args.capture_warmup < 0 or args.benchmark_warmup < 0 or args.runs <= 0:
        parser.error("warmup counts must be non-negative and --runs positive")

    matrix, prototype_vector = matrix_and_vector(args.size, seed=500)
    diagonals, rotation_keys = prepare_constants(engine, matrix)
    schedule = partial(
        matrix_vector,
        engine=engine,
        diagonals=diagonals,
        rotation_keys=rotation_keys,
    )
    prototype = engine.encrypt_message(
        periodic_slots(prototype_vector, engine.num_slots)
    )
    program = CudaGraphProgram.capture(
        schedule,
        example_inputs=(prototype,),
        warmup=args.capture_warmup,
    )

    correctness_rows = []
    borrowed_pointer = None
    for replay_index, seed in enumerate((501, 502, 503)):
        _, vector = matrix_and_vector(args.size, seed=seed)
        encrypted = engine.encrypt_message(
            periodic_slots(vector, engine.num_slots)
        )
        result = program.replay(encrypted, synchronize=True)
        borrowed_pointer = result.data.data_ptr()
        actual = engine.decrypt_message(result, is_real=True)[: args.size]
        error = error_stats(actual, matrix @ vector)
        correctness_rows.append(
            [
                replay_index,
                f"{error['max_abs']:.3e}",
                f"0x{borrowed_pointer:x}",
            ]
        )

    print("Changing-input matrix-vector replay correctness:")
    print_table(
        ["replay", "max abs error", "borrowed output"],
        correctness_rows,
    )

    eager_stats, _ = time_ms(
        lambda: schedule(prototype),
        warmup=args.benchmark_warmup,
        runs=args.runs,
        device=engine.device,
    )
    graph_stats, _ = time_ms(
        lambda: program.replay(prototype),
        warmup=args.benchmark_warmup,
        runs=args.runs,
        device=engine.device,
    )
    print("\nFixed matrix-vector evaluator latency:")
    print_table(
        ["mode", "mean ms", "relative"],
        [
            ["eager", f"{eager_stats['mean_ms']:.4f}", "1.000x"],
            [
                "CUDA Graph",
                f"{graph_stats['mean_ms']:.4f}",
                f"{eager_stats['mean_ms'] / graph_stats['mean_ms']:.3f}x",
            ],
        ],
    )

    stats = program.stats
    print(
        "\nOne-time construction: "
        f"warmup={stats.warmup_seconds:.3f}s, "
        f"capture={stats.capture_seconds:.3f}s, "
        f"first_replay={stats.first_replay_seconds:.6f}s"
    )
    print(
        "Replay returns borrowed storage at "
        f"0x{borrowed_pointer:x}; pass copy_output=True when a result must "
        "survive the next replay."
    )
    program.close()


if __name__ == "__main__":
    main()
