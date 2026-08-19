"""Register a deterministic one-GPU packed matrix-vector workload."""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from typing import Any

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
from fhelium.benchmarks.timing import (
    measure,
    read_peak_memory,
    reset_peak_memory,
    synchronize,
)

CORRECTNESS_ATOL = 3e-5
_MIB = 1024**2

_FIXED_WORKLOADS: dict[str, dict[str, Any]] = {
    "quick": {
        "preset": fh.Preset.slots8192_scale40_levels7_int64.value,
        "ntt_backend": "radix2_compact_group8_smem8",
        "matrix_size": 8,
        "hoist_chunk_size": 7,
        "seed": 20260807,
    },
    "core": {
        "preset": fh.Preset.slots8192_scale40_levels7_int64.value,
        "ntt_backend": "radix2_compact_group8_smem8",
        "matrix_size": 128,
        "hoist_chunk_size": 64,
        "seed": 20260807,
    },
}


def matrix_and_vector(
    size: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the versioned CPU-binary64 matrix and input vector."""

    if size <= 0:
        raise ValueError("matrix size must be positive")
    row = torch.arange(size, dtype=torch.float64).view(-1, 1)
    column = torch.arange(size, dtype=torch.float64).view(1, -1)
    matrix = 0.018 * torch.sin((row + 1) * (column + 2) * 0.17)
    matrix += 0.007 * torch.cos((row + column + 1) * 0.23)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(
        size, generator=generator, dtype=torch.float64, device="cpu"
    )
    vector *= 0.025
    return matrix, vector


def periodic_slots(values: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Repeat one logical vector across the complete CKKS slot vector."""

    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("values must be a non-empty one-dimensional tensor")
    if num_slots % values.numel() != 0:
        raise ValueError(
            f"vector size {values.numel()} must divide num_slots={num_slots}"
        )
    return values.repeat(num_slots // values.numel())


def cyclic_diagonal_slots(
    matrix: torch.Tensor,
    rotation_step: int,
    num_slots: int,
) -> torch.Tensor:
    """Align one cyclic diagonal with ``torch.roll(x, rotation_step)``."""

    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix must be square")
    size = matrix.size(0)
    if size <= 0 or num_slots % size != 0:
        raise ValueError("matrix size must be positive and divide num_slots")
    row = torch.arange(num_slots) % size
    column = torch.remainder(row - rotation_step, size)
    return matrix[row, column]


def prepare_packed_matvec(
    engine: fh.CkksEngine,
    matrix: torch.Tensor,
    source: fh.Ciphertext,
) -> tuple[tuple[fh.Plaintext, ...], dict[int, fh.RotationKey]]:
    """Materialize operation-ready diagonals and exact rotation keys."""

    diagonals = tuple(
        engine.prepare_plaintext_for_multiplication(
            engine.encode(
                cyclic_diagonal_slots(matrix, step, engine.num_slots),
                level=source.level,
            )
        )
        for step in range(matrix.size(0))
    )
    rotation_keys = {
        step: engine.rotation_key(step) for step in range(1, matrix.size(0))
    }
    return diagonals, rotation_keys


def evaluate_packed_matvec(
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    diagonals: Sequence[fh.Plaintext],
    rotation_keys: Mapping[int, fh.RotationKey],
    *,
    hoist_chunk_size: int,
) -> fh.Ciphertext:
    """Evaluate a cyclic-diagonal matrix-vector product in bounded chunks.

    Each chunk batches its rotated ciphertexts, forward NTT, plaintext
    products, and additive reduction. The chunk sum returns to coefficient
    domain and is rescaled once before joining the global accumulator.
    """

    if not diagonals:
        raise ValueError("at least one diagonal is required")
    if hoist_chunk_size <= 0:
        raise ValueError("hoist_chunk_size must be positive")
    expected_steps = set(range(1, len(diagonals)))
    if set(rotation_keys) != expected_steps:
        raise ValueError(
            "rotation_keys must contain exactly steps 1 through diagonal_count-1"
        )

    accumulator = None
    steps = tuple(range(len(diagonals)))
    for start in range(0, len(steps), hoist_chunk_size):
        chunk = steps[start : start + hoist_chunk_size]
        nonzero_steps = tuple(step for step in chunk if step != 0)
        rotated_nonzero = engine.rotate_many_with_keys(
            source,
            [rotation_keys[step] for step in nonzero_steps],
            use_hoisting=len(nonzero_steps) > 1,
        )
        rotated_by_step = dict(zip(nonzero_steps, rotated_nonzero, strict=True))
        rotated_values = tuple(
            source if step == 0 else rotated_by_step[step] for step in chunk
        )
        rotated_batch_ntt = engine.coefficient_domain_to_ntt_domain(
            fh.Ciphertext.stack_batch(rotated_values)
        )
        diagonal_batch = fh.Plaintext.stack_batch(
            tuple(diagonals[step] for step in chunk)
        )
        chunk_sum = engine.sum_ciphertext_batch(
            engine.multiply_plaintext(rotated_batch_ntt, diagonal_batch)
        )
        chunk_sum = engine.rescale_to_next_level(
            engine.ntt_domain_to_coefficient_domain(chunk_sum)
        )
        if accumulator is None:
            accumulator = chunk_sum
        else:
            engine.add_(accumulator, chunk_sum)
    assert accumulator is not None
    return accumulator


def _effective_parameters(profile: BenchmarkProfile) -> dict[str, Any]:
    parameters = dict(profile.parameters)
    try:
        fixed = _FIXED_WORKLOADS[profile.name]
    except KeyError as error:
        raise ValueError(
            f"unsupported packed-matvec profile {profile.name!r}"
        ) from error
    for name, expected in fixed.items():
        if parameters.get(name) != expected:
            raise ValueError(
                f"{name} is fixed at {expected!r} by the packed-matvec workload"
            )
    parameters.update(fixed)
    return parameters


def _error_stats(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    error = actual - expected
    max_abs_error = float(torch.max(torch.abs(error)))
    rms_error = float(torch.sqrt(torch.mean(error.square())))
    return max_abs_error, rms_error


def _run_packed_matvec(
    profile: BenchmarkProfile, progress: ProgressCallback
) -> BenchmarkResult:
    parameters = _effective_parameters(profile)
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    include_raw_samples = bool(parameters.get("include_raw_samples", True))
    size = int(parameters["matrix_size"])
    seed = int(parameters["seed"])
    hoist_chunk_size = int(parameters["hoist_chunk_size"])

    progress("Creating the engine and deterministic packed input")
    engine = fh.CkksEngine(
        fh.Preset(str(parameters["preset"])),
        device=str(parameters.get("device", "cuda:0")),
        ntt_backend=str(parameters["ntt_backend"]),
    )
    if engine.num_slots % size != 0:
        raise ValueError(
            f"matrix_size={size} must divide num_slots={engine.num_slots}"
        )
    matrix, vector = matrix_and_vector(size, seed=seed)
    packed_vector = periodic_slots(vector, engine.num_slots)
    source = engine.encrypt_message(packed_vector)

    progress("Materializing exact rotation keys and operation-ready diagonals")
    diagonals, rotation_keys = prepare_packed_matvec(engine, matrix, source)

    def evaluate() -> fh.Ciphertext:
        return evaluate_packed_matvec(
            engine,
            source,
            diagonals,
            rotation_keys,
            hoist_chunk_size=hoist_chunk_size,
        )

    progress("Warming the encrypted evaluator")
    for _ in range(warmup):
        evaluate()
    synchronize(engine.device)
    gc.collect()

    baseline = reset_peak_memory(engine.device)
    progress("Measuring encrypted matrix-vector evaluation")
    timing = measure(
        evaluate,
        warmup=0,
        runs=runs,
        device=engine.device,
        include_samples=include_raw_samples,
    )
    memory = read_peak_memory(baseline)

    progress("Decrypting one result and checking the cleartext oracle")
    result = evaluate()
    actual = engine.decrypt_message(result, is_real=True)[:size]
    expected = matrix @ vector
    max_abs_error, rms_error = _error_stats(actual, expected)
    if max_abs_error > CORRECTNESS_ATOL:
        raise AssertionError(
            "packed matrix-vector result exceeded the established numerical "
            f"validation limit: max_abs_error={max_abs_error}, "
            f"atol={CORRECTNESS_ATOL}"
        )

    row = {
        "phase": "encrypted packed matrix-vector evaluation",
        "mean_ms": round(timing["mean_ms"], 4),
        "median_ms": round(timing["median_ms"], 4),
        "min_ms": round(timing["min_ms"], 4),
        "max_ms": round(timing["max_ms"], 4),
        "std_ms": round(timing["std_ms"], 4),
        "matrix_vectors_per_second": round(1000.0 / timing["median_ms"], 3),
        "diagonal_terms_per_second": round(
            size * 1000.0 / timing["median_ms"], 3
        ),
    }
    evidence = []
    if include_raw_samples:
        evidence.append(
            {
                "kind": "raw_timing_samples",
                "phase": row["phase"],
                "unit": "ms",
                "samples": timing["samples_ms"],
            }
        )

    effective_parameters = dict(parameters)
    resolved_parameters = {
        "preset": parameters["preset"],
        "device": str(engine.device),
        "ntt_backend": engine.ntt_backend_name,
        "matrix_size": size,
        "matrix_shape": [size, size],
        "vector_shape": [size],
        "packed_slot_count": engine.num_slots,
        "seed": seed,
        "diagonal_count": size,
        "rotation_steps": list(range(1, size)),
        "hoist_chunk_size": hoist_chunk_size,
        "runs": runs,
        "warmup": warmup,
        "include_raw_samples": include_raw_samples,
    }
    timed_boundary = BenchmarkTimedBoundary(
        id="packed-cyclic-diagonal-evaluation-v1",
        description="One encrypted packed matrix-vector evaluation using bounded cyclic-diagonal chunks.",
        includes=(
            "bounded grouped exact-key rotations",
            "batched forward NTT and operation-ready plaintext multiplication",
            "batched ciphertext reduction and one rescale per chunk",
        ),
        excludes=(
            "engine and key construction",
            "matrix and vector generation",
            "diagonal encoding and operation-ready preparation",
            "input encryption",
            "output decryption and oracle comparison",
        ),
        synchronization="Synchronize the engine CUDA device before and after every sample.",
    )
    metrics = [
        BenchmarkMetric(
            name="packed-matrix-vector-latency",
            value=timing["median_ms"],
            unit="ms",
            statistic="median",
            direction="lower",
            dimensions={
                "category": "latency",
                "matrix_size": size,
                "backend": engine.ntt_backend_name,
            },
            samples=tuple(timing.get("samples_ms", ())),
        ),
        BenchmarkMetric(
            name="packed-matrix-vector-throughput",
            value=1000.0 / timing["median_ms"],
            unit="matrix-vectors/s",
            statistic="inverse_median_latency",
            direction="higher",
            dimensions={
                "category": "throughput",
                "matrix_size": size,
                "backend": engine.ntt_backend_name,
            },
        ),
        BenchmarkMetric(
            name="packed-matrix-vector-peak-allocated",
            value=memory["peak_allocated_bytes"] / _MIB,
            unit="MiB",
            statistic="maximum",
            direction="lower",
            dimensions={
                "category": "memory",
                "matrix_size": size,
                "backend": engine.ntt_backend_name,
            },
        ),
        BenchmarkMetric(
            name="packed-matrix-vector-peak-allocated-delta",
            value=memory["peak_allocated_delta_bytes"] / _MIB,
            unit="MiB",
            statistic="maximum",
            direction="lower",
            dimensions={
                "category": "memory",
                "matrix_size": size,
                "backend": engine.ntt_backend_name,
            },
        ),
    ]
    checks = [
        BenchmarkCheck(
            name="packed-matrix-vector-cleartext-oracle",
            passed=max_abs_error <= CORRECTNESS_ATOL,
            oracle="Decrypted first logical vector compared with CPU-binary64 matrix @ vector.",
            metric="max_abs_error",
            observed=max_abs_error,
            comparison="<=",
            limit=CORRECTNESS_ATOL,
            unit="absolute",
            details={"rms_error": rms_error, "output_elements": size},
        )
    ]

    return BenchmarkResult(
        benchmark="packed-matrix-vector",
        profile=profile.name,
        workload_id="packed-matrix-vector",
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=[row],
        scalars={
            "max_abs_error": max_abs_error,
            "rms_error": rms_error,
            "correctness_atol": CORRECTNESS_ATOL,
            "resident_allocated_mib": (
                memory["baseline_allocated_bytes"] / _MIB
            ),
            "peak_allocated_mib": memory["peak_allocated_bytes"] / _MIB,
            "peak_allocated_delta_mib": (
                memory["peak_allocated_delta_bytes"] / _MIB
            ),
            "peak_reserved_mib": memory["peak_reserved_bytes"] / _MIB,
        },
        metadata={
            "workload_id": "packed-matrix-vector",
            "effective_parameters": effective_parameters,
            "resolved_parameters": resolved_parameters,
            "input": {
                "matrix_dtype": str(matrix.dtype),
                "vector_dtype": str(vector.dtype),
                "vector_min": float(vector.min()),
                "vector_max": float(vector.max()),
                "packing": "periodic repetitions of one logical vector",
            },
            "output_state": {
                "level": result.level,
                "scale": result.scale,
                "component_count": result.component_count,
                "polynomial_domain": result.polynomial_domain,
                "modulus_basis": result.modulus_basis,
                "residue_representation": result.residue_representation,
            },
            "timed_boundary": timed_boundary.to_dict(),
            "memory_scope": (
                "resident operation-ready state plus transient measured "
                "evaluation allocations on the selected device"
            ),
        },
        notes=[
            "The cyclic-diagonal packing and deterministic matrix formula follow the maintained FHElium packed-matvec methodology without importing distributed worker internals.",
            "Correctness is enforced at atol=3e-5, the existing public packed-matvec validation limit in example 09; the benchmark does not infer or tune tolerance.",
            "Peak memory is PyTorch CUDA allocator memory on the engine device, not total device memory or external allocator usage.",
        ],
        evidence=evidence,
    )


def _profile(
    name: str,
    description: str,
    *,
    warmup: int,
    runs: int,
) -> BenchmarkProfile:
    return BenchmarkProfile(
        name,
        description,
        {
            **_FIXED_WORKLOADS[name],
            "device": "cuda:0",
            "warmup": warmup,
            "runs": runs,
            "include_raw_samples": True,
        },
    )


register_benchmark(
    BenchmarkDefinition(
        name="packed-matrix-vector",
        title="Packed matrix-vector multiplication",
        category="single GPU workload",
        description=(
            "Evaluates a fixed dense matrix-vector product with cyclic "
            "diagonal packing, bounded grouped rotations, a cleartext oracle, "
            "raw timing samples, and device-targeted allocator peaks."
        ),
        profiles=(
            _profile(
                "quick",
                "Fixed 8x8 packed-matvec correctness and timing smoke.",
                warmup=1,
                runs=3,
            ),
            _profile(
                "core",
                "Fixed 128x128 packed-matvec workload.",
                warmup=3,
                runs=10,
            ),
        ),
        runner=_run_packed_matvec,
        workload_id="packed-matrix-vector",
    )
)
