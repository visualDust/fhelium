"""Register CKKS operator-latency and rotation-hoisting benchmarks."""

from __future__ import annotations

import gc
from typing import Any

import torch

from fhelium import DEFAULT_NTT_BACKEND, CkksEngine, Preset
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
from fhelium.benchmarks.synthetic import ckks_message
from fhelium.benchmarks.timing import (
    measure,
    measure_paired,
    read_peak_memory,
    reset_peak_memory,
)

CKKS_CORRECTNESS_ATOL = 2e-5
_MIB = 1024**2


def _preset(name: str) -> Preset:
    try:
        return Preset(name)
    except ValueError as error:
        choices = ", ".join(item.value for item in Preset)
        raise ValueError(
            f"Unknown preset {name!r}; choices: {choices}"
        ) from error


def _engine(parameters: Any) -> CkksEngine:
    return CkksEngine(
        _preset(str(parameters["preset"])),
        device=str(parameters.get("device", "cuda:0")),
        ntt_backend=str(parameters.get("ntt_backend") or DEFAULT_NTT_BACKEND),
    )


def _error_stats(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    error = torch.abs(actual - expected)
    return {
        "max_abs_error": float(torch.max(error)),
        "rms_error": float(torch.sqrt(torch.mean(error.square()))),
    }


def _enforce_correctness(
    operation: str,
    stats: dict[str, float],
    *,
    atol: float = CKKS_CORRECTNESS_ATOL,
) -> None:
    if stats["max_abs_error"] > atol:
        raise AssertionError(
            f"{operation} exceeded the established CKKS validation limit: "
            f"max_abs_error={stats['max_abs_error']}, atol={atol}"
        )


def _timing_evidence(
    operation: str, timing: dict[str, Any]
) -> dict[str, Any] | None:
    samples = timing.get("samples_ms")
    if samples is None:
        return None
    return {
        "kind": "raw_timing_samples",
        "operation": operation,
        "unit": "ms",
        "samples": list(samples),
    }


def _operator_latency(
    profile: BenchmarkProfile, progress: ProgressCallback
) -> BenchmarkResult:
    parameters = profile.parameters
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    include_raw_samples = bool(parameters.get("include_raw_samples", True))
    rotation_steps = tuple(int(value) for value in parameters["rotation_steps"])
    if not rotation_steps:
        raise ValueError("rotation_steps must contain at least one step")
    if len(set(rotation_steps)) != len(rotation_steps):
        raise ValueError("rotation_steps must not contain duplicates")

    progress("Creating engine and materializing keys")
    engine = _engine(parameters)
    device = engine.device
    x = ckks_message(engine)
    y = ckks_message(engine, phase=0.3)
    ct_x = engine.encrypt_message(x)
    ct_y = engine.encrypt_message(y)
    ct_x_ntt = engine.coefficient_domain_to_ntt_domain(ct_x)
    ct_y_ntt = engine.coefficient_domain_to_ntt_domain(ct_y)
    engine.relinearization_key
    for rotation_step in rotation_steps:
        engine.rotation_key(rotation_step)

    operations = {
        "encrypt_message": lambda: engine.encrypt_message(x),
        "decrypt_message": lambda: engine.decrypt_message(ct_x),
        "add": lambda: engine.add(ct_x, ct_y),
        "multiply": lambda: engine.multiply(ct_x_ntt, ct_y_ntt),
        "rotate_with_key": lambda: engine.rotate_with_key(
            ct_x, engine.rotation_key(rotation_steps[0])
        ),
        f"rotate_many_by_steps[{len(rotation_steps)}]": lambda: (
            engine.rotate_many_by_steps(ct_x, rotation_steps)
        ),
    }

    rows = []
    evidence = []
    metrics = []
    for name, operation in operations.items():
        progress(f"Measuring {name}")
        baseline = reset_peak_memory(device)
        timing = measure(
            operation,
            warmup=warmup,
            runs=runs,
            device=device,
            include_samples=include_raw_samples,
        )
        memory = read_peak_memory(baseline)
        rows.append(
            {
                "operation": name,
                "mean_ms": round(timing["mean_ms"], 4),
                "median_ms": round(timing["median_ms"], 4),
                "min_ms": round(timing["min_ms"], 4),
                "max_ms": round(timing["max_ms"], 4),
                "std_ms": round(timing["std_ms"], 4),
                "ops_per_second": round(1000.0 / timing["mean_ms"], 2),
                "ops_per_s": round(1000.0 / timing["mean_ms"], 2),
                "peak_allocated_delta_mib": round(
                    memory["peak_allocated_delta_bytes"] / _MIB, 3
                ),
                "peak_delta_mib": round(
                    memory["peak_allocated_delta_bytes"] / _MIB, 3
                ),
                "peak_allocated_mib": round(
                    memory["peak_allocated_bytes"] / _MIB, 3
                ),
                "peak_reserved_mib": round(
                    memory["peak_reserved_bytes"] / _MIB, 3
                ),
            }
        )
        timing_row = _timing_evidence(name, timing)
        if timing_row is not None:
            evidence.append(timing_row)
        metric_dimensions = {"category": "latency", "operation": name}
        metrics.append(
            BenchmarkMetric(
                name="ckks-operation-latency",
                value=timing["median_ms"],
                unit="ms",
                statistic="median",
                direction="lower",
                dimensions=metric_dimensions,
                samples=tuple(timing.get("samples_ms", ())),
            )
        )
        metrics.append(
            BenchmarkMetric(
                name="ckks-operation-throughput",
                value=1000.0 / timing["mean_ms"],
                unit="operations/s",
                statistic="inverse_mean_latency",
                direction="higher",
                dimensions={"category": "throughput", "operation": name},
            )
        )
        metrics.append(
            BenchmarkMetric(
                name="ckks-operation-peak-allocated-delta",
                value=memory["peak_allocated_delta_bytes"] / _MIB,
                unit="MiB",
                statistic="maximum",
                direction="lower",
                dimensions={"category": "memory", "operation": name},
            )
        )
        gc.collect()

    progress("Checking decrypted outputs")
    correctness: dict[str, dict[str, float]] = {}
    correctness["encrypt_decrypt"] = _error_stats(
        engine.decrypt_message(engine.encrypt_message(x)), x
    )
    correctness["add"] = _error_stats(
        engine.decrypt_message(engine.add(ct_x, ct_y)), x + y
    )
    product = engine.relinearize(engine.multiply(ct_x_ntt, ct_y_ntt))
    correctness["multiply_relinearize"] = _error_stats(
        engine.decrypt_message(product), x * y
    )
    rotated = engine.rotate_with_key(
        ct_x, engine.rotation_key(rotation_steps[0])
    )
    correctness["rotate_with_key"] = _error_stats(
        engine.decrypt_message(rotated),
        torch.roll(x, shifts=rotation_steps[0]),
    )
    many = engine.rotate_many_by_steps(ct_x, rotation_steps)
    many_errors = [
        _error_stats(engine.decrypt_message(value), torch.roll(x, shifts=step))
        for step, value in zip(rotation_steps, many, strict=True)
    ]
    correctness["rotate_many_by_steps"] = {
        "max_abs_error": max(row["max_abs_error"] for row in many_errors),
        "rms_error": max(row["rms_error"] for row in many_errors),
    }
    for name, stats in correctness.items():
        _enforce_correctness(name, stats)

    effective_parameters = dict(parameters)
    resolved_parameters = {
        "preset": parameters["preset"],
        "device": str(device),
        "ntt_backend": engine.ntt_backend_name,
        "rotation_steps": list(rotation_steps),
        "runs": runs,
        "warmup": warmup,
        "include_raw_samples": include_raw_samples,
    }
    timed_boundary = BenchmarkTimedBoundary(
        id="single-ckks-operation-v1",
        description="One named CKKS operation with per-operation CUDA-device synchronization.",
        includes=("one named CKKS operation",),
        excludes=(
            "engine construction",
            "key generation",
            "input construction",
            "correctness decryption",
        ),
        synchronization="Synchronize the engine CUDA device before and after every sample.",
    )
    checks = [
        BenchmarkCheck(
            name=f"{name}-max-absolute-error",
            passed=stats["max_abs_error"] <= CKKS_CORRECTNESS_ATOL,
            oracle="Decryption compared with the matching cleartext CKKS operation.",
            metric="max_abs_error",
            observed=stats["max_abs_error"],
            comparison="<=",
            limit=CKKS_CORRECTNESS_ATOL,
            unit="absolute",
            details={"rms_error": stats["rms_error"]},
        )
        for name, stats in correctness.items()
    ]

    return BenchmarkResult(
        benchmark="ckks-operator-latency",
        profile=profile.name,
        workload_id="ckks-operator-latency",
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        scalars={
            "correctness_atol": CKKS_CORRECTNESS_ATOL,
            "cadd_max_error": correctness["add"]["max_abs_error"],
            "multiply_max_error": correctness["multiply_relinearize"][
                "max_abs_error"
            ],
            "rotation_max_error": correctness["rotate_with_key"][
                "max_abs_error"
            ],
            **{
                f"{name}_{metric}": value
                for name, stats in correctness.items()
                for metric, value in stats.items()
            },
        },
        metadata={
            "workload_id": "ckks-operator-latency",
            "preset": parameters["preset"],
            "device": str(device),
            "ntt_backend": engine.ntt_backend_name,
            "runs": runs,
            "warmup": warmup,
            "effective_parameters": effective_parameters,
            "resolved_parameters": resolved_parameters,
            "timed_boundary": timed_boundary.to_dict(),
            "memory_scope": (
                "warmup and measured calls after operation-ready state is resident"
            ),
        },
        notes=[
            "Correctness is enforced at atol=2e-5, the existing public CKKS workflow validation limit; tolerances are not inferred from benchmark results.",
            "Multiply latency is the raw ciphertext product; correctness additionally relinearizes before decryption.",
        ],
        evidence=evidence,
    )


def _rotation_workload(
    profile: BenchmarkProfile, progress: ProgressCallback
) -> BenchmarkResult:
    parameters = profile.parameters
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    repetitions = int(parameters.get("repetitions", 1))
    include_raw_samples = bool(parameters.get("include_raw_samples", True))
    counts = tuple(int(value) for value in parameters["counts"])
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("counts must contain positive rotation counts")
    if len(set(counts)) != len(counts):
        raise ValueError("counts must not contain duplicates")

    progress("Creating engine and rotation keys")
    engine = _engine(parameters)
    device = engine.device
    message = ckks_message(engine)
    ct = engine.encrypt_message(message)
    max_count = max(counts)
    rotation_steps = tuple(range(1, max_count + 1))
    for rotation_step in rotation_steps:
        engine.rotation_key(rotation_step)

    rows = []
    evidence = []
    correctness_rows = []
    metrics = []
    for count in counts:
        selected = rotation_steps[:count]
        progress(f"Measuring {count} rotations with paired alternating order")

        def independent_rotation():
            return [
                engine.rotate_with_key(ct, engine.rotation_key(rotation_step))
                for rotation_step in selected
            ]

        def hoisted_rotation():
            return engine.rotate_many_by_steps(ct, selected)

        comparison = measure_paired(
            independent_rotation,
            hoisted_rotation,
            warmup=warmup,
            runs=runs,
            repetitions=repetitions,
            device=device,
            include_samples=include_raw_samples,
        )
        independent = comparison["first"]
        hoisted = comparison["second"]
        mean_speedup = independent["mean_ms"] / hoisted["mean_ms"]
        median_speedup = independent["median_ms"] / hoisted["median_ms"]
        rows.append(
            {
                "rotations": count,
                "independent_mean_ms": round(independent["mean_ms"], 4),
                "independent_median_ms": round(independent["median_ms"], 4),
                "independent_min_ms": round(independent["min_ms"], 4),
                "independent_max_ms": round(independent["max_ms"], 4),
                "independent_std_ms": round(independent["std_ms"], 4),
                "hoisted_mean_ms": round(hoisted["mean_ms"], 4),
                "hoisted_median_ms": round(hoisted["median_ms"], 4),
                "hoisted_min_ms": round(hoisted["min_ms"], 4),
                "hoisted_max_ms": round(hoisted["max_ms"], 4),
                "hoisted_std_ms": round(hoisted["std_ms"], 4),
                "independent_ms": round(independent["mean_ms"], 4),
                "hoisted_ms": round(hoisted["mean_ms"], 4),
                "speedup": round(mean_speedup, 3),
                "median_speedup": round(median_speedup, 3),
                "savings_percent": round((1.0 - 1.0 / mean_speedup) * 100.0, 2),
                "savings_pct": round((1.0 - 1.0 / mean_speedup) * 100.0, 2),
                "hoisted_rotations_per_second": round(
                    count * 1000.0 / hoisted["mean_ms"], 2
                ),
                "hoisted_rotations_per_s": round(
                    count * 1000.0 / hoisted["mean_ms"], 2
                ),
            }
        )
        for mode, timing in (
            ("independent", independent),
            ("hoisted", hoisted),
        ):
            metrics.append(
                BenchmarkMetric(
                    name="rotation-set-latency",
                    value=timing["median_ms"],
                    unit="ms",
                    statistic="median",
                    direction="lower",
                    dimensions={
                        "category": "latency",
                        "mode": mode,
                        "rotation_count": count,
                    },
                    samples=tuple(timing.get("samples_ms", ())),
                )
            )
        metrics.append(
            BenchmarkMetric(
                name="hoisted-rotation-throughput",
                value=count * 1000.0 / hoisted["mean_ms"],
                unit="rotations/s",
                statistic="inverse_mean_latency",
                direction="higher",
                dimensions={
                    "category": "throughput",
                    "mode": "hoisted",
                    "rotation_count": count,
                },
            )
        )
        if include_raw_samples:
            evidence.append(
                {
                    "kind": "paired_raw_timing_samples",
                    "rotation_count": count,
                    "unit": "ms",
                    "independent_samples": independent["samples_ms"],
                    "hoisted_samples": hoisted["samples_ms"],
                    "paired_independent_over_hoisted": comparison[
                        "paired_ratio"
                    ]["samples"],
                    "repetition_order": [
                        row["starting_order"]
                        for row in comparison["repetitions"]
                    ],
                }
            )

        independent_values = independent_rotation()
        hoisted_values = hoisted_rotation()
        if len(independent_values) != count or len(hoisted_values) != count:
            raise AssertionError(
                "rotation workload returned an unexpected number of outputs"
            )
        count_errors: list[dict[str, float]] = []
        for step, independent_value, hoisted_value in zip(
            selected, independent_values, hoisted_values, strict=True
        ):
            expected = torch.roll(message, shifts=step)
            independent_error = _error_stats(
                engine.decrypt_message(independent_value), expected
            )
            hoisted_error = _error_stats(
                engine.decrypt_message(hoisted_value), expected
            )
            _enforce_correctness("independent rotation", independent_error)
            _enforce_correctness("hoisted rotation", hoisted_error)
            count_errors.extend((independent_error, hoisted_error))
        correctness_rows.append(
            {
                "rotations": count,
                "max_abs_error": max(
                    stats["max_abs_error"] for stats in count_errors
                ),
                "max_rms_error": max(
                    stats["rms_error"] for stats in count_errors
                ),
            }
        )

    effective_parameters = dict(parameters)
    resolved_parameters = {
        "preset": parameters["preset"],
        "device": str(device),
        "ntt_backend": engine.ntt_backend_name,
        "counts": list(counts),
        "rotation_steps": list(rotation_steps),
        "runs_per_repetition": runs,
        "repetitions": repetitions,
        "warmup_pairs": warmup,
        "include_raw_samples": include_raw_samples,
    }
    timed_boundary = BenchmarkTimedBoundary(
        id="paired-rotation-schedules-v1",
        description="Paired independent and hoisted exact-key rotation schedules.",
        includes=(
            "one ordered set of exact-key rotations per side of each pair",
            "alternating first-executed schedule",
        ),
        excludes=(
            "engine construction",
            "key generation",
            "input encryption",
            "correctness decryption",
        ),
        synchronization="Synchronize the engine CUDA device around every side of every pair.",
    )
    checks = [
        BenchmarkCheck(
            name=f"rotation-count-{row['rotations']}-max-absolute-error",
            passed=row["max_abs_error"] <= CKKS_CORRECTNESS_ATOL,
            oracle="Every independent and hoisted output decrypted against its ordered torch.roll result.",
            metric="max_abs_error",
            observed=row["max_abs_error"],
            comparison="<=",
            limit=CKKS_CORRECTNESS_ATOL,
            unit="absolute",
            details={"max_rms_error": row["max_rms_error"]},
        )
        for row in correctness_rows
    ]

    return BenchmarkResult(
        benchmark="rotation-hoisting-workload",
        profile=profile.name,
        workload_id="rotation-hoisting-workload",
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        scalars={
            "correctness_atol": CKKS_CORRECTNESS_ATOL,
            "max_abs_error": max(
                row["max_abs_error"] for row in correctness_rows
            ),
            "max_rms_error": max(
                row["max_rms_error"] for row in correctness_rows
            ),
        },
        metadata={
            "workload_id": "rotation-hoisting-workload",
            "preset": parameters["preset"],
            "device": str(device),
            "ntt_backend": engine.ntt_backend_name,
            "runs": runs,
            "warmup": warmup,
            "effective_parameters": effective_parameters,
            "resolved_parameters": resolved_parameters,
            "timed_boundary": timed_boundary.to_dict(),
            "comparison_order": (
                "paired samples alternating which implementation executes first"
            ),
            "correctness_by_count": correctness_rows,
        },
        notes=[
            "Key generation is excluded; all requested rotation keys are materialized before timing.",
            "Every output from both schedules is decrypted and checked against the same torch.roll oracle at the established atol=2e-5 CKKS validation limit.",
        ],
        evidence=evidence,
    )


def _profile(
    name: str,
    description: str,
    *,
    preset: Preset,
    counts_or_steps: list[int],
    warmup: int,
    runs: int,
    repetitions: int | None = None,
) -> BenchmarkProfile:
    parameters: dict[str, Any] = {
        "preset": preset.value,
        "device": "cuda:0",
        "ntt_backend": "radix2_compact_group16_smem8",
        "warmup": warmup,
        "runs": runs,
        "include_raw_samples": True,
    }
    if repetitions is None:
        parameters["rotation_steps"] = counts_or_steps
    else:
        parameters["counts"] = counts_or_steps
        parameters["repetitions"] = repetitions
    return BenchmarkProfile(name, description, parameters)


register_benchmark(
    BenchmarkDefinition(
        name="ckks-operator-latency",
        title="CKKS operator latency",
        category="single GPU",
        description=(
            "Measures dense encode/encrypt/decrypt, ciphertext arithmetic, "
            "scalar rotation, and grouped rotation on one device. Keys are "
            "materialized before operation timing and outputs are validated."
        ),
        profiles=(
            _profile(
                "quick",
                "8,192-slot/40-bit-scale/7-level smoke profile with short timing loops.",
                preset=Preset.slots8192_scale40_levels7_int64,
                counts_or_steps=[1, 2, 4, 8],
                warmup=2,
                runs=5,
            ),
            _profile(
                "core",
                "Versioned 8,192-slot core operator measurement with raw samples.",
                preset=Preset.slots8192_scale40_levels7_int64,
                counts_or_steps=[1, 2, 4, 8],
                warmup=5,
                runs=20,
            ),
            _profile(
                "standard",
                "32,768-slot/40-bit-scale/34-level profile for stable operator comparisons.",
                preset=Preset.slots32768_scale40_levels34_int64,
                counts_or_steps=[1, 2, 4, 8],
                warmup=5,
                runs=20,
            ),
        ),
        runner=_operator_latency,
        workload_id="ckks-operator-latency",
    )
)

register_benchmark(
    BenchmarkDefinition(
        name="rotation-hoisting-workload",
        title="Rotation-hoisting workload",
        category="workload",
        description=(
            "Compares independent key-switched rotations with grouped "
            "NTT-domain rotation hoisting using alternating paired samples."
        ),
        profiles=(
            _profile(
                "quick",
                "Short 8,192-slot/40-bit-scale/7-level paired rotation sweep.",
                preset=Preset.slots8192_scale40_levels7_int64,
                counts_or_steps=[2, 4, 8],
                warmup=2,
                runs=3,
                repetitions=2,
            ),
            _profile(
                "core",
                "Versioned 8,192-slot paired sweep with repeated A/B measurements.",
                preset=Preset.slots8192_scale40_levels7_int64,
                counts_or_steps=[2, 4, 8],
                warmup=4,
                runs=5,
                repetitions=3,
            ),
            _profile(
                "standard",
                "Longer 32,768-slot/40-bit-scale/34-level rotation sweep.",
                preset=Preset.slots32768_scale40_levels34_int64,
                counts_or_steps=[4, 8, 16],
                warmup=5,
                runs=10,
                repetitions=2,
            ),
        ),
        runner=_rotation_workload,
        workload_id="rotation-hoisting-workload",
    )
)
