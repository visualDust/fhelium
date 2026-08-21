"""Fixed cross-backend indexed radix-2 NTT workload for Benchmark v1."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from typing import Any, Literal, cast

import torch

import fhelium as fh
from fhelium.benchmarks.model import (
    BenchmarkCheck,
    BenchmarkDefinition,
    BenchmarkMetric,
    BenchmarkProfile,
    BenchmarkResult,
    BenchmarkTimedBoundary,
    ProgressCallback,
)
from fhelium.benchmarks.registry import register_benchmark
from fhelium.benchmarks.timing import synchronize
from fhelium.config import CkksConfig

from .model import BenchmarkExecution

WORKLOAD_ID = "indexed-ntt-operations"
PRESET = fh.Preset.slots8192_scale40_levels7_int64
BACKEND = "radix2_indexed"
SEED = 20260807
ModulusBasis = Literal["Q", "QP"]
NttOperation = Literal["forward_ntt", "inverse_ntt", "roundtrip"]
MODULUS_BASES: tuple[ModulusBasis, ...] = ("Q", "QP")
OPERATIONS: tuple[NttOperation, ...] = (
    "forward_ntt",
    "inverse_ntt",
    "roundtrip",
)
_CONFIG = CkksConfig.parse(PRESET)
ENTRY_LEVELS = tuple(range(_CONFIG.num_scale_primes))


def describe_basis_cell(
    config: CkksConfig, *, entry_level: int, modulus_basis: ModulusBasis
) -> dict[str, Any]:
    if not 0 <= entry_level < config.num_scale_primes:
        raise ValueError("entry_level is outside the public CKKS levels")
    q_ids = tuple(range(entry_level, config.num_q_primes))
    p_ids = (
        tuple(range(config.num_q_primes, config.total_num_primes))
        if modulus_basis == "QP"
        else ()
    )
    prime_ids = q_ids + p_ids
    moduli = tuple(config.moduli[index] for index in prime_ids)
    return {
        "entry_level": entry_level,
        "modulus_basis": modulus_basis,
        "active_prime_ids": list(prime_ids),
        "active_prime_count": len(prime_ids),
        "active_q_prime_count": len(q_ids),
        "active_p_prime_count": len(p_ids),
        "active_prime_product_bits": (math.prod(moduli) - 1).bit_length(),
        "parameter_row_start": entry_level,
        "parameter_row_stop": config.total_num_primes
        if modulus_basis == "QP"
        else config.num_q_primes,
    }


def make_standard_residue_rows(
    config: CkksConfig, *, seed: int = SEED
) -> torch.Tensor:
    coefficients = torch.arange(config.N, dtype=torch.int64, device="cpu")
    quadratic = coefficients.square() * 104729
    rows = []
    for prime_id, modulus in enumerate(config.moduli):
        row = torch.remainder(
            quadratic
            + coefficients * (13007 + 2 * prime_id)
            + seed
            + 65537 * prime_id,
            modulus,
        )
        row[1::2] = modulus - 1 - row[1::2]
        rows.append(row.to(dtype=config.torch_dtype))
    return torch.stack(rows).contiguous()


def _run_operation(
    engine: fh.CkksEngine,
    data: torch.Tensor,
    operation: NttOperation,
    *,
    include_p: bool,
) -> None:
    if operation in {"forward_ntt", "roundtrip"}:
        engine.rns_runtime.forward_montgomery_(data, include_p=include_p)
    if operation in {"inverse_ntt", "roundtrip"}:
        engine.rns_runtime.inverse_montgomery_(data, include_p=include_p)


def _prepare(
    engine: fh.CkksEngine,
    rows: torch.Tensor,
    *,
    level: int,
    basis: ModulusBasis,
    operation: NttOperation,
) -> torch.Tensor:
    cell = describe_basis_cell(
        engine.config, entry_level=level, modulus_basis=basis
    )
    data = (
        rows[int(cell["parameter_row_start"]) : int(cell["parameter_row_stop"])]
        .to(engine.device)
        .contiguous()
    )
    include_p = basis == "QP"
    engine.rns_runtime.to_montgomery_(data, include_p=include_p)
    if operation == "inverse_ntt":
        engine.rns_runtime.forward_montgomery_(data, include_p=include_p)
    return data


def _summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": statistics.pstdev(samples),
    }


def _run_indexed_ntt(
    profile: BenchmarkProfile,
    progress: ProgressCallback,
    *,
    execution: BenchmarkExecution,
) -> BenchmarkResult:
    parameters = dict(profile.parameters)
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    engine = fh.CkksEngine(PRESET, device=execution.device, ntt_backend=BACKEND)
    rows = make_standard_residue_rows(engine.config)
    metrics: list[BenchmarkMetric] = []
    checks: list[BenchmarkCheck] = []
    result_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for level in ENTRY_LEVELS:
        for basis in MODULUS_BASES:
            include_p = basis == "QP"
            roundtrip = _prepare(
                engine, rows, level=level, basis=basis, operation="roundtrip"
            )
            expected = roundtrip.clone()
            _run_operation(engine, roundtrip, "roundtrip", include_p=include_p)
            cell = describe_basis_cell(
                engine.config, entry_level=level, modulus_basis=basis
            )
            moduli = torch.tensor(
                [
                    engine.config.moduli[index]
                    for index in cell["active_prime_ids"]
                ],
                dtype=engine.config.torch_dtype,
                device=engine.device,
            )
            mismatch_count = int(
                torch.count_nonzero(
                    torch.remainder(roundtrip - expected, moduli[:, None])
                ).item()
            )
            checks.append(
                BenchmarkCheck(
                    name=f"indexed-ntt-level-{level}-{basis.lower()}-roundtrip",
                    passed=mismatch_count == 0,
                    oracle="Forward followed by inverse NTT preserves every active residue modulo its prime.",
                    metric="residue_mismatch_count",
                    observed=mismatch_count,
                    comparison="==",
                    limit=0,
                    unit="residues",
                    details=cell,
                )
            )
            for operation in OPERATIONS:
                progress(f"Measuring {operation} at level {level} in {basis}")
                base = _prepare(
                    engine, rows, level=level, basis=basis, operation=operation
                )
                for _ in range(warmup):
                    sample = base.clone()
                    _run_operation(
                        engine, sample, operation, include_p=include_p
                    )
                samples: list[float] = []
                for _ in range(runs):
                    sample = base.clone()
                    synchronize(engine.device)
                    started = time.perf_counter()
                    _run_operation(
                        engine, sample, operation, include_p=include_p
                    )
                    synchronize(engine.device)
                    samples.append((time.perf_counter() - started) * 1e3)
                timing = _summary(samples)
                dimensions = {
                    "category": "latency",
                    "operation": operation,
                    "entry_level": level,
                    "modulus_basis": basis,
                    "active_prime_count": int(cell["active_prime_count"]),
                    "backend": BACKEND,
                }
                result_rows.append({**cell, **dimensions, **timing})
                metrics.append(
                    BenchmarkMetric(
                        name="indexed-ntt-latency",
                        value=timing["median_ms"],
                        unit="ms",
                        statistic="median",
                        direction="lower",
                        dimensions=dimensions,
                        samples=tuple(samples),
                    )
                )
                evidence.append(
                    {
                        "kind": "raw_timing_samples",
                        **dimensions,
                        "unit": "ms",
                        "samples": samples,
                    }
                )
    boundary = BenchmarkTimedBoundary(
        id="indexed-radix2-ntt-call",
        description="One fixed indexed radix-2 NTT operation on a prepared active-prime tensor.",
        includes=("native indexed radix-2 transform",),
        excludes=(
            "input construction",
            "host-to-device transfer",
            "input restoration",
            "correctness roundtrip",
        ),
        synchronization="Synchronize the selected engine device before and after every sample; CPU calls complete synchronously.",
    )
    return BenchmarkResult(
        benchmark=WORKLOAD_ID,
        profile=profile.name,
        workload_id=WORKLOAD_ID,
        effective_parameters=parameters,
        timed_boundary=boundary,
        metrics=metrics,
        correctness=checks,
        rows=result_rows,
        scalars={
            "cell_count": len(metrics),
            "roundtrip_check_count": len(checks),
        },
        metadata={
            "workload_id": WORKLOAD_ID,
            "benchmark_context": {
                "ckks_plan": {
                    "preset": PRESET.value,
                    "logN": engine.config.logN,
                    "slot_count": engine.num_slots,
                    "ntt_backend": BACKEND,
                },
                "parameter_selection": {
                    "policy": "one identical CPU/CUDA indexed-radix2 plan",
                    "hardware_adaptive": False,
                },
                "entry_state": {
                    "entry_levels": list(ENTRY_LEVELS),
                    "modulus_bases": list(MODULUS_BASES),
                    "operations": list(OPERATIONS),
                },
            },
            "execution": execution.to_dict(),
        },
        notes=[
            "The backend is part of the fixed workload and is not ranked or selected at runtime."
        ],
        evidence=evidence,
    )


_PARAMETERS = {
    "preset": PRESET.value,
    "ntt_backend": BACKEND,
    "entry_levels": list(ENTRY_LEVELS),
    "modulus_bases": list(MODULUS_BASES),
    "operations": list(OPERATIONS),
    "seed": SEED,
    "warmup": 1,
    "runs": 3,
    "include_raw_samples": True,
}

DEFINITION = register_benchmark(
    BenchmarkDefinition(
        name=WORKLOAD_ID,
        title="Indexed radix-2 NTT operations",
        category="local device",
        description="Measures one fixed indexed radix-2 NTT workload identically on CPU and CUDA.",
        profiles=(
            BenchmarkProfile(
                "core", "Complete fixed cross-backend NTT cells.", _PARAMETERS
            ),
        ),
        runner=cast(Any, _run_indexed_ntt),
        workload_id=WORKLOAD_ID,
    )
)

__all__ = [
    "BACKEND",
    "DEFINITION",
    "ENTRY_LEVELS",
    "MODULUS_BASES",
    "OPERATIONS",
    "PRESET",
    "WORKLOAD_ID",
    "describe_basis_cell",
    "make_standard_residue_rows",
]
