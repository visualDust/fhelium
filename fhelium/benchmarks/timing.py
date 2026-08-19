"""Device-aware timing and CUDA allocator helpers for benchmark workloads."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

DeviceLike: TypeAlias = torch.device | str | int | None


@dataclass(frozen=True)
class CudaMemoryBaseline:
    """Allocator counters captured immediately after resetting peak statistics."""

    device: torch.device | None
    allocated_bytes: int
    reserved_bytes: int


def _cuda_device(device: DeviceLike) -> torch.device | None:
    """Resolve an indexed CUDA device while retaining legacy ``None`` behavior."""

    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", device)
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def synchronize(device: DeviceLike = None) -> None:
    """Synchronize ``device`` when it is CUDA.

    Omitting ``device`` preserves the original helper's current-device
    behavior. Benchmark runners should pass the device that owns the measured
    tensors so another current CUDA device cannot leave the measurement unsynchronized.
    """

    cuda_device = _cuda_device(device)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)


def _summary(
    samples_ms: list[float], *, include_samples: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "std_ms": statistics.pstdev(samples_ms),
    }
    if include_samples:
        result["samples_ms"] = list(samples_ms)
    return result


def _ratio_summary(
    samples: list[float], *, include_samples: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "std": statistics.pstdev(samples),
    }
    if include_samples:
        result["samples"] = list(samples)
    return result


def measure(
    function: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
    device: DeviceLike = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Measure synchronized wall latency in milliseconds.

    ``device`` and ``include_samples`` are optional to preserve existing
    callers. Warmup calls are excluded from the returned statistics. When raw
    evidence is requested, ``samples_ms`` retains samples in execution order.
    """

    if runs <= 0 or warmup < 0:
        raise ValueError(
            "runs must be positive and warmup must be non-negative"
        )
    for _ in range(warmup):
        function()
    synchronize(device)

    samples_ms = []
    for _ in range(runs):
        synchronize(device)
        start = time.perf_counter()
        function()
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1e3)

    return _summary(samples_ms, include_samples=include_samples)


def measure_paired(
    first: Callable[[], Any],
    second: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
    repetitions: int = 1,
    device: DeviceLike = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Measure a paired A/B comparison with alternating execution order.

    Each warmup and measured pair invokes both callables exactly once. The
    leading callable alternates for every pair and for every repetition, which
    avoids assigning all first-run or thermal effects to one side. Returned
    ``first`` and ``second`` summaries pool all repetitions. ``paired_ratio``
    summarizes sample-wise ``first_ms / second_ms`` ratios, while the
    repetition records preserve the pairing structure. Raw samples are added
    only when ``include_samples`` is true.
    """

    if runs <= 0 or warmup < 0 or repetitions <= 0:
        raise ValueError(
            "runs and repetitions must be positive and warmup must be "
            "non-negative"
        )

    def invoke_pair(first_leads: bool) -> tuple[float, float]:
        samples: dict[str, float] = {}
        order: tuple[tuple[str, Callable[[], Any]], ...] = (
            ("first", first),
            ("second", second),
        )
        if not first_leads:
            order = tuple(reversed(order))
        for name, function in order:
            synchronize(device)
            start = time.perf_counter()
            function()
            synchronize(device)
            samples[name] = (time.perf_counter() - start) * 1e3
        return samples["first"], samples["second"]

    for index in range(warmup):
        invoke_pair(index % 2 == 0)
    synchronize(device)

    first_samples: list[float] = []
    second_samples: list[float] = []
    ratios: list[float] = []
    repetition_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        repetition_first: list[float] = []
        repetition_second: list[float] = []
        for run in range(runs):
            first_ms, second_ms = invoke_pair((repetition + run) % 2 == 0)
            repetition_first.append(first_ms)
            repetition_second.append(second_ms)
            first_samples.append(first_ms)
            second_samples.append(second_ms)
            ratios.append(first_ms / second_ms)
        repetition_rows.append(
            {
                "repetition": repetition,
                "starting_order": (
                    "first_then_second"
                    if repetition % 2 == 0
                    else "second_then_first"
                ),
                "first": _summary(
                    repetition_first, include_samples=include_samples
                ),
                "second": _summary(
                    repetition_second, include_samples=include_samples
                ),
            }
        )

    return {
        "first": _summary(first_samples, include_samples=include_samples),
        "second": _summary(second_samples, include_samples=include_samples),
        "paired_ratio": _ratio_summary(ratios, include_samples=include_samples),
        "repetitions": repetition_rows,
    }


def reset_peak_memory(device: DeviceLike = None) -> CudaMemoryBaseline:
    """Reset allocator peaks on one CUDA device and return its live baseline."""

    cuda_device = _cuda_device(device)
    if cuda_device is None:
        return CudaMemoryBaseline(None, 0, 0)
    synchronize(cuda_device)
    allocated = int(torch.cuda.memory_allocated(cuda_device))
    reserved = int(torch.cuda.memory_reserved(cuda_device))
    torch.cuda.reset_peak_memory_stats(cuda_device)
    return CudaMemoryBaseline(cuda_device, allocated, reserved)


def read_peak_memory(baseline: CudaMemoryBaseline) -> dict[str, int]:
    """Read device-targeted CUDA allocator peaks relative to ``baseline``."""

    if baseline.device is None:
        return {
            "baseline_allocated_bytes": 0,
            "baseline_reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "peak_allocated_delta_bytes": 0,
            "peak_reserved_delta_bytes": 0,
        }
    synchronize(baseline.device)
    peak_allocated = int(torch.cuda.max_memory_allocated(baseline.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(baseline.device))
    return {
        "baseline_allocated_bytes": baseline.allocated_bytes,
        "baseline_reserved_bytes": baseline.reserved_bytes,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(
            0, peak_allocated - baseline.allocated_bytes
        ),
        "peak_reserved_delta_bytes": max(
            0, peak_reserved - baseline.reserved_bytes
        ),
    }
