#!/usr/bin/env python3

"""Benchmark grouped rotation hoisting against independent rotations.

Example:
    python examples/07_rotation_hoisting_benchmark.py --preset slots32768-scale40-levels34-int64 --counts 4,8,16
"""

from __future__ import annotations

import argparse
import gc
import statistics as stats
import time

import torch
from common import add_engine_args, make_engine, print_table, sync_if_cuda


def _time_once(fn, device: torch.device):
    sync_if_cuda(device)
    start = time.perf_counter()
    result = fn()
    sync_if_cuda(device)
    return (time.perf_counter() - start) * 1e3, result


def _summarize(times: list[float]) -> dict[str, float]:
    return {
        "mean_ms": stats.mean(times),
        "median_ms": stats.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": stats.pstdev(times),
    }


def _alternating_bench(
    independent, hoisted, *, warmup: int, runs: int, device: torch.device
):
    for _ in range(warmup):
        independent()
        hoisted()
        sync_if_cuda(device)

    independent_times = []
    hoisted_times = []
    for run_idx in range(runs):
        if run_idx % 2 == 0:
            timing, result = _time_once(independent, device)
            independent_times.append(timing)
            del result
            timing, result = _time_once(hoisted, device)
            hoisted_times.append(timing)
            del result
        else:
            timing, result = _time_once(hoisted, device)
            hoisted_times.append(timing)
            del result
            timing, result = _time_once(independent, device)
            independent_times.append(timing)
            del result

    return _summarize(independent_times), _summarize(hoisted_times)


def _parse_counts(text: str) -> list[int]:
    counts = [int(item.strip()) for item in text.split(',') if item.strip()]
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("counts must be positive integers")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots32768-scale40-levels34-int64")
    parser.add_argument("--counts", type=_parse_counts, default=[4, 8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    engine = make_engine(args)
    max_count = max(args.counts)
    rotation_steps_all = list(range(1, max_count + 1))

    _ = engine.public_key
    for rotation_step in rotation_steps_all:
        _ = engine.rotation_key(rotation_step)

    slots = engine.num_slots
    idx = torch.arange(slots, dtype=torch.float64)
    message = (
        0.01 * torch.sin(idx * 0.001) + 0.005 * torch.cos(idx * 0.003)
    ).to(torch.complex128)
    ct = engine.encrypt_message(message, level=args.level)
    sync_if_cuda(engine.device)

    rows = []
    for count in args.counts:
        rotation_steps = rotation_steps_all[:count]

        def independent():
            return [
                engine.rotate_by_step(ct, rotation_step)
                for rotation_step in rotation_steps
            ]

        def hoisted():
            return engine.rotate_many_by_steps(ct, rotation_steps)

        independent_stats, hoisted_stats = _alternating_bench(
            independent,
            hoisted,
            warmup=args.warmup,
            runs=args.runs,
            device=engine.device,
        )
        speedup = independent_stats["mean_ms"] / hoisted_stats["mean_ms"]
        savings = (
            1.0 - hoisted_stats["mean_ms"] / independent_stats["mean_ms"]
        ) * 100.0
        rows.append(
            [
                count,
                f"{independent_stats['mean_ms']:.4f}",
                f"{hoisted_stats['mean_ms']:.4f}",
                f"{speedup:.4f}x",
                f"{savings:.2f}%",
                f"{independent_stats['median_ms']:.4f}",
                f"{hoisted_stats['median_ms']:.4f}",
            ]
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"Engine: preset={args.preset}, level={args.level}, device={args.device}, runs={args.runs}, warmup={args.warmup}"
    )
    print_table(
        [
            "rotations",
            "one-by-one mean ms",
            "hoisted mean ms",
            "speedup",
            "mean savings",
            "one-by-one median ms",
            "hoisted median ms",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
