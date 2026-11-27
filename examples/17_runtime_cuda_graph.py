#!/usr/bin/env python3

"""Run compiled CKKS functions through CUDA Graph segments.

Two compiled functions form a pipeline: a packed matrix-vector product ``A x`` with the cyclic-diagonal method and a square activation. For each ``graph`` setting the example reports latency and the execution choice of each Graph segment. It then shows that the activation reads the product's output in place, and that a result the caller keeps across calls is moved to separate storage before the next call overwrites it.
"""

from __future__ import annotations

import argparse

import torch
from common import print_table, time_ms

import fhelium as fh
from fhelium import compile as fc
from fhelium.backend.runner import compiled_execution_bytes
from fhelium.eager import Engine
from fhelium.values import Ciphertext, Plaintext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("CUDA Graph segments require a CUDA device")

    config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7)
    engine = Engine(config, rng_seed=17)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    relinearization_key = engine.create_relinearization_key(
        secret_key, device=device
    )
    size = args.size
    steps = tuple(-k for k in range(1, size))
    for step in steps:
        engine.set_rotation_key(
            engine.create_rotation_key(step, secret_key, device=device)
        )

    row = torch.arange(size, dtype=torch.float64).view(-1, 1)
    column = torch.arange(size, dtype=torch.float64).view(1, -1)
    matrix = 0.3 * torch.sin((row + 1) * (column + 2) * 0.17)
    slots = torch.arange(engine.num_slots) % size
    diagonals = Plaintext.stack_batch(
        [
            engine.prepare_plaintext_for_multiplication(
                engine.encode(matrix[slots, (slots + k) % size], device=device)
            )
            for k in range(size)
        ]
    )

    def matvec(x: Ciphertext) -> Ciphertext:
        rotated = Ciphertext.stack_batch(
            [x, *engine.rotate_many_by_steps(x, steps)]
        )
        products = engine.sum_plaintext_products(
            engine.coefficient_domain_to_ntt_domain(rotated), diagonals
        )
        return engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(products)
        )

    def square(y: Ciphertext) -> Ciphertext:
        ntt = engine.coefficient_domain_to_ntt_domain(y)
        return engine.relinearize(
            engine.multiply(ntt, ntt), relinearization_key, rescale=True
        )

    generator = torch.Generator().manual_seed(17)
    vectors = [
        0.5 * torch.rand(size, generator=generator, dtype=torch.float64)
        for _ in range(3)
    ]
    inputs = [
        engine.encrypt_message(
            v.repeat(engine.num_slots // size), public_key, device=device
        )
        for v in vectors
    ]

    # Latency and segment choices for each graph setting.
    rows = []
    for graph in ("auto", True, False):
        first = fc.compile(matvec, graph=graph)
        second = fc.compile(square, graph=graph)
        timing, _ = time_ms(
            lambda: second(first(inputs[0])),
            warmup=3,
            runs=args.runs,
            device=device,
        )
        choices = [
            [segment["choice"] for segment in record["segments"]]
            for compiled in (first, second)
            for record in compiled.stats()["graphs"]
        ]
        rows.append([str(graph), f"{timing['mean_ms']:.3f}", choices])
        first.clear()
        second.clear()
    print_table(["graph", "mean ms", "segment choices (matvec, square)"], rows)

    # A pipeline of two compiled functions on one device.
    first = fc.compile(matvec, graph=True)
    second = fc.compile(square, graph=True)
    errors = []
    for vector, x in zip(vectors, inputs * 2, strict=False):
        result = second(first(x))
        decoded = engine.decrypt_message(result, secret_key, is_real=True)
        errors.append(
            float((decoded[:size].cpu() - (matrix @ vector) ** 2).abs().max())
        )
        del result
    for _ in range(3):
        result = second(first(inputs[0]))
        del result
    capture = second.stats()["graphs"][0]["capture_stats"]
    print()
    print_table(
        ["activation reading the product in place", "value"],
        [
            ["max abs error over inputs", f"{max(errors):.3e}"],
            ["captured variants", capture["variants"]],
            ["staged input copies", capture["staged_copies"]],
            [
                "compiled execution bytes on the device",
                compiled_execution_bytes(device.index or 0),
            ],
        ],
    )

    # Results the caller keeps are moved out of the output region.
    kept = [second(first(x)) for x in inputs]
    dropped_before = second.stats()["graphs"][0]["capture_stats"][
        "detached_outputs"
    ]
    for x in inputs:
        output = second(first(x))
        del output
    dropped_after = second.stats()["graphs"][0]["capture_stats"][
        "detached_outputs"
    ]
    distinct = len({result.data.data_ptr() for result in kept})
    print()
    print_table(
        ["result ownership", "value"],
        [
            [
                "kept results with distinct storage",
                f"{distinct} of {len(kept)}",
            ],
            ["detached outputs while keeping results", dropped_before],
            [
                "additional detaches after dropping",
                dropped_after - dropped_before,
            ],
        ],
    )
    del kept
    first.clear()
    second.clear()


if __name__ == "__main__":
    main()
