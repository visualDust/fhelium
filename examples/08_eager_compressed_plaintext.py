"""Prepare and evaluate compact coefficient and NTT plaintexts.

One period of a CKKS slot message supplies compact multiplication and addition
operands. The example also demonstrates lossless compression of existing RNS
data, checks arithmetic against dense equivalents, and checks in-place storage
sharing. Example 11 uses compressed diagonals in a compiled matrix product.
"""

from __future__ import annotations

import argparse
import os
from statistics import median
from time import perf_counter

import torch
from common import add_engine_args, make_engine, print_table

import fhelium as fh


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _paired_ms(dense, compact, *, iterations: int, device: torch.device):
    """Measure warmed calls in alternating order, excluding preparation."""
    functions = (dense, compact)
    for _ in range(3):
        for function in functions:
            function()
    samples: tuple[list[float], list[float]] = ([], [])
    for iteration in range(iterations):
        for index in (0, 1) if iteration % 2 == 0 else (1, 0):
            _synchronize(device)
            start = perf_counter()
            functions[index]()
            _synchronize(device)
            samples[index].append((perf_counter() - start) * 1000)
    return median(samples[0]), median(samples[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(
        parser, default_preset=fh.Preset.slots8192_scale40_depth7_int64.value
    )
    parser.add_argument("--period", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    engine = make_engine(args)
    device = torch.get_default_device()
    period = args.period
    if period <= 0 or period & (period - 1) or engine.num_slots % period:
        raise ValueError(
            "--period must be a positive power of two dividing the slot count"
        )

    index = torch.arange(period, dtype=torch.float64)
    unique_slots = torch.complex(
        0.03 * torch.cos(index * 0.07) + 0.001 * index / period,
        0.02 * torch.sin(index * 0.05),
    )
    factor = unique_slots.repeat(engine.num_slots // period)

    # The output domain selects the prepared polynomial and storage layout.
    multiply_weight = engine.prepare_compressed_plaintext(unique_slots)
    add_weight = engine.prepare_compressed_plaintext(
        unique_slots, polynomial_domain="coefficient"
    )
    dense_multiply = multiply_weight.to_plaintext()
    dense_add = add_weight.to_plaintext()

    # Existing encoded data can instead be compressed with bit-for-bit checks.
    encoded = engine.prepare_plaintext_for_multiplication(engine.encode(factor))
    checked = fh.CompressedPlaintext.from_plaintext(
        encoded,
        unique_count=multiply_weight.unique_count,
        compression_layout="contiguous",
    )
    torch.testing.assert_close(
        checked.decompress_data(), encoded.data, rtol=0, atol=0
    )

    print_table(
        ["property", "value"],
        [
            ["device", str(device)],
            ["ring dimension", engine.ring_dimension],
            ["periodic slots", period],
            ["encoded unique count", multiply_weight.unique_count],
            [
                "PyTorch intra-op / inter-op threads",
                f"{torch.get_num_threads()} / {torch.get_num_interop_threads()}",
            ],
            [
                "thread environment",
                str(
                    {
                        key: os.environ[key]
                        for key in (
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                        )
                        if key in os.environ
                    }
                ),
            ],
        ],
    )
    print_table(
        ["prepared form", "layout", "compact bytes", "dense bytes"],
        [
            [
                "NTT / Montgomery",
                multiply_weight.compression_layout,
                multiply_weight.nbytes,
                dense_multiply.nbytes,
            ],
            [
                "coefficient / Montgomery",
                add_weight.compression_layout,
                add_weight.nbytes,
                dense_add.nbytes,
            ],
        ],
    )

    slots = torch.arange(engine.num_slots, dtype=torch.float64)
    message = torch.complex(
        0.01 * torch.sin(slots * 0.013), 0.008 * torch.cos(slots * 0.011)
    )
    ciphertext = engine.encrypt_message(message)
    ciphertext_ntt = engine.coefficient_domain_to_ntt_domain(ciphertext)
    moduli = torch.tensor(
        [engine.config.moduli[i] for i in ciphertext.prime_ids],
        dtype=torch.int64,
    ).view(-1, 1)

    cases = [
        (
            "coefficient addition",
            engine.add_plaintext,
            ciphertext,
            add_weight,
            dense_add,
        ),
        (
            "NTT addition",
            engine.add_plaintext,
            ciphertext_ntt,
            multiply_weight,
            dense_multiply,
        ),
        (
            "NTT multiplication",
            engine.multiply_plaintext,
            ciphertext_ntt,
            multiply_weight,
            dense_multiply,
        ),
    ]
    timings = []
    for label, operation, source, compact, dense in cases:
        expected = operation(source, dense)
        actual = operation(source, compact)
        torch.testing.assert_close(
            actual.data.to(torch.int64) % moduli,
            expected.data.to(torch.int64) % moduli,
            rtol=0,
            atol=0,
        )
        target = source.clone()
        alias = target.data
        operation(target, compact, inplace=True)
        assert target.data is alias
        torch.testing.assert_close(
            alias.to(torch.int64) % moduli,
            expected.data.to(torch.int64) % moduli,
            rtol=0,
            atol=0,
        )
        dense_ms, compact_ms = _paired_ms(
            lambda: operation(source, dense),
            lambda: operation(source, compact),
            iterations=args.iterations,
            device=device,
        )
        timings.append([label, f"{dense_ms:.4f}", f"{compact_ms:.4f}"])

    product = engine.multiply_plaintext(ciphertext_ntt, multiply_weight)
    decoded = engine.decrypt_message(
        engine.ntt_domain_to_coefficient_domain(product)
    )
    print(
        f"Maximum multiplication cleartext error: {(decoded - message * factor).abs().max().item():.3e}"
    )
    zero = engine.zero_plaintext_like(add_weight)
    assert torch.count_nonzero(zero.data) == 0
    assert (
        zero.implicit_data is not None
        and torch.count_nonzero(zero.implicit_data) == 0
    )
    assert add_weight.is_rns and add_weight.representation == "rns"
    print(
        "Dense-equivalent arithmetic, in-place storage, zero value and lossless conversion checks passed."
    )
    print_table(
        ["operation", "dense median (ms)", "compact median (ms)"], timings
    )


if __name__ == "__main__":
    main()
