"""JIT-compile CKKS weighted-product and matrix-multiplication workloads.

Compare each original function with its JIT wrapper after checking correctness.
First-call latency is separate from warmed, alternating paired measurements.
Manual pipelines and persisted Programs have separate examples.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from statistics import median
from time import perf_counter
from typing import TypeVar, cast

import torch
from common import print_table, sync_if_cuda

from fhelium import CkksConfig, Preset, compile as fc
from fhelium.eager import Engine
from fhelium.values import Ciphertext, Plaintext


_T = TypeVar("_T")


def _timed_call(fn: Callable[[], _T], device: torch.device) -> tuple[_T, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (perf_counter() - start) * 1e3


def _latency_row(
    label: str,
    eager: Callable[[], object],
    jit: Callable[[], object],
    device: torch.device,
    *,
    warmup: int,
    runs: int,
) -> list[str]:
    for _ in range(warmup):
        eager()
        sync_if_cuda(device)
        jit()
        sync_if_cuda(device)
    calls = (eager, jit)
    samples: tuple[list[float], list[float]] = ([], [])
    for sample in range(runs):
        for index in (0, 1) if sample % 2 == 0 else (1, 0):
            result, elapsed = _timed_call(calls[index], device)
            samples[index].append(elapsed)
            del result
    eager_ms, jit_ms = (median(values) for values in samples)
    return [
        label,
        f"{eager_ms:.4f}",
        f"{jit_ms:.4f}",
        f"{eager_ms / jit_ms:.3f}x",
        f"{100 * (jit_ms / eager_ms - 1):+.1f}%",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--print-ir", action="store_true")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup calls per path"
    )
    parser.add_argument(
        "--runs", type=int, default=30, help="Timed samples per path"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be nonnegative and --runs must be positive")
    device = torch.device(args.device)

    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    engine = Engine(config, rng_seed=24)
    secret_key = engine.create_secret_key(device=device)
    engine.set_secret_key(secret_key)
    public_key = engine.create_public_key(secret_key, device=device)

    @fc.compile
    def weighted_product(
        lhs: Ciphertext,
        rhs: Ciphertext,
        weight: Plaintext,
        negative: bool = False,
    ) -> Ciphertext:
        a = engine.coefficient_domain_to_ntt_domain(lhs)
        b = engine.coefficient_domain_to_ntt_domain(rhs)
        product = engine.multiply(a, b)
        weighted = engine.multiply_plaintext(product, weight)
        doubled = engine.subtract(weighted, engine.negate(weighted))
        if negative:
            doubled = engine.negate(doubled)
        return engine.ntt_domain_to_coefficient_domain(doubled)

    clear_x = torch.linspace(
        -0.1, 0.1, config.num_slots, dtype=torch.float64, device=device
    )
    clear_y = torch.linspace(
        0.03, 0.07, config.num_slots, dtype=torch.float64, device=device
    )
    clear_weight = torch.linspace(
        0.25, 0.5, config.num_slots, dtype=torch.float64, device=device
    )
    x = engine.encrypt_message(clear_x, public_key, device=device)
    y = engine.encrypt_message(clear_y, public_key, device=device)
    weight = engine.prepare_plaintext_for_multiplication(
        engine.encode(clear_weight, device=device)
    )

    # An ordinary first call prepares its specialization. Startup code can call
    # prepare separately, but the first real execution may compile device code.
    result, weighted_first_ms = _timed_call(
        lambda: weighted_product(x, y, weight), device
    )
    reference = weighted_product.reference
    assert reference is not None
    torch.testing.assert_close(
        result.data, reference(x, y, weight).data, rtol=0, atol=0
    )
    other_x = engine.negate(x)
    torch.testing.assert_close(
        weighted_product(other_x, y, weight).data,
        reference(other_x, y, weight).data,
        rtol=0,
        atol=0,
    )
    assert len(weighted_product.specializations) == 1
    weighted_product.prepare(x, y, weight, negative=True)
    torch.testing.assert_close(
        weighted_product(x, y, weight, negative=True).data,
        reference(x, y, weight, True).data,
        rtol=0,
        atol=0,
    )

    decoded = engine.decrypt_message(result, secret_key)
    weighted_error = float(
        (decoded - 2 * clear_x * clear_y * clear_weight).abs().max()
    )

    matrix_size, columns, baby_step = 128, 8, 8
    indices = torch.arange(matrix_size, device=device)
    row = indices.to(torch.float64)[:, None]
    column = indices.to(torch.float64)[None, :]
    matrix = 0.006 * torch.sin(
        (row + 1) * (column + 2) * 0.013
    ) + 0.003 * torch.cos((row - column) * 0.027)
    channels = torch.arange(columns, dtype=torch.float64, device=device)[
        None, :
    ]
    clear_matrix_input = 0.025 * torch.cos(
        (row + 1) * 0.031 + channels * 0.23
    ) + 0.007 * torch.sin(row * 0.071 - channels * 0.11)
    matrix_expected = matrix @ clear_matrix_input
    # Each batch item stores one column, periodically tiled across CKKS slots.
    matrix_input = engine.encrypt_message(
        clear_matrix_input.T.contiguous().repeat(
            1, config.num_slots // matrix_size
        ),
        public_key,
        device=device,
    )
    baby_steps = tuple(range(1, baby_step))
    giant_steps = tuple(range(baby_step, matrix_size, baby_step))
    rotation_keys = {
        step: engine.create_rotation_key(step, secret_key, device=device)
        for step in (*baby_steps, *giant_steps)
    }
    for key in rotation_keys.values():
        engine.set_rotation_key(key)
    diagonals = []
    for shift in range(matrix_size):
        diagonal = matrix[indices, (indices - shift) % matrix_size]
        adjusted = torch.roll(
            diagonal, shifts=-(shift // baby_step) * baby_step
        )
        diagonals.append(
            engine.prepare_compressed_plaintext(
                adjusted,
                depth=matrix_input.depth,
                device=device,
            )
        )

    @fc.compile
    def matrix_multiply(value: Ciphertext) -> Ciphertext:
        rotated = engine.rotate_many_by_steps(
            value, baby_steps, use_hoisting=True
        )
        babies = [engine.coefficient_domain_to_ntt_domain(value)]
        for baby in rotated:
            babies.append(engine.coefficient_domain_to_ntt_domain(baby))
        groups = []
        for group in range(matrix_size // baby_step):
            partial = engine.multiply_plaintext(
                babies[0], diagonals[group * baby_step]
            )
            for baby in range(1, baby_step):
                term = engine.multiply_plaintext(
                    babies[baby], diagonals[group * baby_step + baby]
                )
                partial = engine.add(partial, term)
            partial = engine.ntt_domain_to_coefficient_domain(partial)
            if group:
                partial = engine.rotate_with_key(
                    partial, rotation_keys[group * baby_step]
                )
            groups.append(partial)
        total = groups[0]
        for partial in groups[1:]:
            total = engine.add(total, partial)
        return engine.rescale_to_next_depth(total)

    matrix_result, matrix_first_ms = _timed_call(
        lambda: matrix_multiply(matrix_input), device
    )
    matrix_reference = matrix_multiply.reference
    assert matrix_reference is not None
    torch.testing.assert_close(
        matrix_result.data, matrix_reference(matrix_input).data, rtol=0, atol=0
    )
    matrix_decoded = engine.decrypt_message(
        matrix_result, secret_key, is_real=True
    )[..., :matrix_size].T
    torch.testing.assert_close(
        matrix_decoded, matrix_expected, rtol=0, atol=1e-5
    )
    matrix_error = float((matrix_decoded - matrix_expected).abs().max())

    print_table(
        ["property", "value"],
        [
            ["device", str(device)],
            [
                "PyTorch intra-op / inter-op threads",
                f"{torch.get_num_threads()} / {torch.get_num_interop_threads()}",
            ],
            [
                "thread environment",
                str(
                    {
                        name: os.environ[name]
                        for name in (
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "OMP_DYNAMIC",
                            "MKL_DYNAMIC",
                        )
                        if name in os.environ
                    }
                ),
            ],
            ["ring dimension N", config.N],
            ["active Q rows", len(x.prime_ids)],
        ],
    )
    print_table(
        [
            "workload",
            "logical size",
            "input Tensor shapes",
            "output depth / components",
            "maximum clear error",
        ],
        [
            [
                "1. CKKS weighted product",
                f"{config.num_slots} slots, batch 1",
                f"CT {tuple(x.data.shape)}, PT {tuple(cast(torch.Tensor, weight.data).shape)}",
                f"{result.depth} / {result.component_count}",
                f"{weighted_error:.6e}",
            ],
            [
                "2. CKKS matrix multiplication",
                f"{matrix_size}x{matrix_size} @ {matrix_size}x{columns}; BSGS baby {baby_step}",
                f"CT {tuple(matrix_input.data.shape)}, {len(diagonals)} compact PT diagonals",
                f"{matrix_result.depth} / {matrix_result.component_count}",
                f"{matrix_error:.6e}",
            ],
        ],
    )
    print(
        "lowering and fusion pipeline:",
        fc.default_lower_and_fuse_pipeline().names,
    )
    for label, compiled in (
        ("1. CKKS weighted product", weighted_product),
        ("2. CKKS matrix multiplication", matrix_multiply),
    ):
        variant = compiled.specializations[0]
        print(f"\n{label}: {len(compiled.specializations)} specialization(s)")
        print_table(
            ["pass", "transformed"],
            [
                [report.name, report.stats.transformed]
                for report in variant.compilation.reports
            ],
        )
        implementations = sorted(
            {
                dispatch.implementation.name
                for dispatch in variant.executable.dispatch_table.operations.values()
            }
        )
        print(f"implementations: {', '.join(implementations)}")
        if args.print_ir:
            print(variant.compilation.program)
    print_table(
        ["workload", "JIT first call (ms)"],
        [
            ["1. CKKS weighted product", f"{weighted_first_ms:.3f}"],
            ["2. CKKS matrix multiplication", f"{matrix_first_ms:.3f}"],
        ],
    )
    rows = [
        _latency_row(
            "1. CKKS weighted product",
            lambda: reference(x, y, weight),
            lambda: weighted_product(x, y, weight),
            device,
            warmup=args.warmup,
            runs=args.runs,
        ),
        _latency_row(
            "2. CKKS matrix multiplication",
            lambda: matrix_reference(matrix_input),
            lambda: matrix_multiply(matrix_input),
            device,
            warmup=args.warmup,
            runs=args.runs,
        ),
    ]
    print(
        f"Warm latency: {args.warmup} warmups, {args.runs} alternating samples per path; median synchronized wall time."
    )
    print_table(
        [
            "workload",
            "Eager (ms)",
            "JIT (ms)",
            "Eager/JIT",
            "JIT latency change",
        ],
        rows,
    )
    print(
        "First call includes preparation and any needed kernel compilation/cache loading. Warm timings exclude setup, encryption and decryption; no CUDA Graph."
    )


if __name__ == "__main__":
    main()
