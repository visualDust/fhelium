"""Evidence-based NTT backend recommendation suites."""

from __future__ import annotations

import gc
import math
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

import fhelium
from fhelium import DEFAULT_NTT_BACKEND, CkksEngine, Preset
from fhelium.benchmarks.model import (
    BenchmarkCheck,
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkTimedBoundary,
    ProgressCallback,
)
from fhelium.benchmarks.standalone.ntt_kernel import (
    assert_ntt_roundtrip,
    make_residue_rows,
    measure_ntt_operation,
    prepare_ntt_operation_inputs,
)
from fhelium.benchmarks.synthetic import ckks_message
from fhelium.benchmarks.timing import measure
from fhelium.config import CkksConfig
from fhelium.config.ntt import compatible_ntt_backends

NttRecommendationSuite = Literal["kernel", "ckks-primitive"]
_SUPPORTED_SUITES: tuple[NttRecommendationSuite, ...] = (
    "kernel",
    "ckks-primitive",
)
_KERNEL_OPERATIONS = ("forward_ntt", "inverse_ntt", "roundtrip")
_PRIMITIVE_OPERATIONS = (
    "encrypt_message",
    "decrypt_message",
    "multiply_relinearize",
    "rotate_with_key",
    "rotate_many_by_steps[4]",
)
_TIE_THRESHOLD = 0.03
_MEDIUM_CONFIDENCE_MARGIN = 0.03
_HIGH_CONFIDENCE_MARGIN = 0.05
_CORRECTNESS_ATOL = 1e-4
WORKLOAD_ID = "ntt-backend-recommendation"


def _preset(name: str) -> Preset:
    try:
        return Preset(name)
    except ValueError as error:
        choices = ", ".join(item.value for item in Preset)
        raise ValueError(
            f"Unknown preset {name!r}; choices: {choices}"
        ) from error


def _resolve_candidates(
    *,
    log_ring_dimension: int,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    compatible = compatible_ntt_backends(log_ring_dimension)
    if requested is None or len(requested) == 0:
        return tuple(name for name in compatible if name != "radix2_indexed")
    candidates = tuple(str(name) for name in requested)
    if len(candidates) < 2:
        raise ValueError("NTT recommendation requires at least two backends")
    if len(set(candidates)) != len(candidates):
        raise ValueError(
            "NTT recommendation backends must not contain duplicates"
        )
    incompatible = tuple(name for name in candidates if name not in compatible)
    if incompatible:
        raise ValueError(
            f"NTT backends {incompatible!r} are incompatible with "
            f"logN={log_ring_dimension}; compatible backends: {compatible!r}"
        )
    return candidates


def _rotated_order(
    backends: tuple[str, ...], repetition: int
) -> tuple[str, ...]:
    offset = repetition % len(backends)
    return backends[offset:] + backends[:offset]


def _kernel_measurements(
    *,
    preset: Preset,
    device: str,
    backends: tuple[str, ...],
    warmup: int,
    runs: int,
    repetitions: int,
    seed: int,
    progress: ProgressCallback,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    measurements: list[dict[str, Any]] = []
    orders: list[list[str]] = []
    for repetition in range(repetitions):
        order = _rotated_order(backends, repetition)
        orders.append(list(order))
        for backend in order:
            progress(
                f"Kernel suite repetition {repetition + 1}/{repetitions}: "
                f"{backend}"
            )
            engine = CkksEngine(preset, device=device, ntt_backend=backend)
            standard_rows = make_residue_rows(engine, seed=seed + repetition)
            operation_inputs = prepare_ntt_operation_inputs(
                engine, standard_rows
            )
            assert_ntt_roundtrip(engine, operation_inputs["roundtrip"])
            for operation in _KERNEL_OPERATIONS:
                with torch.cuda.device(engine.device):
                    timing = measure_ntt_operation(
                        engine,
                        operation_inputs[operation],
                        operation,
                        warmup=warmup,
                        runs=runs,
                    )
                measurements.append(
                    {
                        "backend": backend,
                        "operation": operation,
                        "repetition": repetition,
                        **timing,
                        "max_error": 0.0,
                    }
                )
            del engine, standard_rows, operation_inputs
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return measurements, orders


def _primitive_setup(
    engine: CkksEngine,
) -> tuple[
    dict[str, Callable[[], object]],
    dict[str, float],
]:
    message_x = ckks_message(engine)
    message_y = ckks_message(engine, phase=0.3)
    ciphertext_x = engine.encrypt_message(message_x)
    ciphertext_y = engine.encrypt_message(message_y)
    ciphertext_x_ntt = engine.coefficient_domain_to_ntt_domain(ciphertext_x)
    ciphertext_y_ntt = engine.coefficient_domain_to_ntt_domain(ciphertext_y)
    relinearization_key = engine.relinearization_key
    rotation_steps = (1, 2, 4, 8)
    rotation_keys = tuple(engine.rotation_key(step) for step in rotation_steps)

    operations: dict[str, Callable[[], object]] = {
        "encrypt_message": lambda: engine.encrypt_message(message_x),
        "decrypt_message": lambda: engine.decrypt_message(ciphertext_x),
        "multiply_relinearize": lambda: engine.relinearize(
            engine.multiply(ciphertext_x_ntt, ciphertext_y_ntt),
            relinearization_key,
        ),
        "rotate_with_key": lambda: engine.rotate_with_key(
            ciphertext_x, rotation_keys[0]
        ),
        "rotate_many_by_steps[4]": lambda: engine.rotate_many_by_steps(
            ciphertext_x, rotation_steps
        ),
    }

    decrypted = engine.decrypt_message(ciphertext_x)
    decrypt_error = float(torch.max(torch.abs(decrypted - message_x)).item())
    multiplied = engine.relinearize(
        engine.multiply(ciphertext_x_ntt, ciphertext_y_ntt),
        relinearization_key,
    )
    multiply_error = float(
        torch.max(
            torch.abs(
                engine.decrypt_message(multiplied) - message_x * message_y
            )
        ).item()
    )
    rotated = engine.rotate_with_key(ciphertext_x, rotation_keys[0])
    rotate_error = float(
        torch.max(
            torch.abs(
                engine.decrypt_message(rotated)
                - torch.roll(message_x, shifts=rotation_steps[0])
            )
        ).item()
    )
    rotated_many = engine.rotate_many_by_steps(ciphertext_x, rotation_steps)
    rotate_many_error = max(
        float(
            torch.max(
                torch.abs(
                    engine.decrypt_message(value)
                    - torch.roll(message_x, shifts=step)
                )
            ).item()
        )
        for value, step in zip(rotated_many, rotation_steps, strict=True)
    )
    errors = {
        "decrypt_message": decrypt_error,
        "multiply_relinearize": multiply_error,
        "rotate_with_key": rotate_error,
        "rotate_many_by_steps[4]": rotate_many_error,
    }
    maximum_error = max(errors.values())
    if maximum_error > _CORRECTNESS_ATOL:
        raise RuntimeError(
            f"CKKS primitive correctness failed for {engine.ntt_backend_name}: "
            f"max error {maximum_error:.6g} exceeds {_CORRECTNESS_ATOL:.6g}"
        )
    return operations, errors


def _primitive_measurements(
    *,
    preset: Preset,
    device: str,
    backends: tuple[str, ...],
    warmup: int,
    runs: int,
    repetitions: int,
    progress: ProgressCallback,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    measurements: list[dict[str, Any]] = []
    orders: list[list[str]] = []
    for repetition in range(repetitions):
        order = _rotated_order(backends, repetition)
        orders.append(list(order))
        for backend in order:
            progress(
                f"CKKS primitive suite repetition {repetition + 1}/{repetitions}: "
                f"{backend}; preparing keys"
            )
            engine = CkksEngine(preset, device=device, ntt_backend=backend)
            operations, errors = _primitive_setup(engine)
            maximum_error = max(errors.values())
            for operation_name in _PRIMITIVE_OPERATIONS:
                progress(f"Measuring {backend}: {operation_name}")
                timing = measure(
                    operations[operation_name],
                    warmup=warmup,
                    runs=runs,
                    device=engine.device,
                    include_samples=True,
                )
                measurements.append(
                    {
                        "backend": backend,
                        "operation": operation_name,
                        "repetition": repetition,
                        **timing,
                        "max_error": maximum_error,
                    }
                )
            del engine, operations
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return measurements, orders


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("Geometric-mean inputs must be positive")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _rank_measurements(
    measurements: Sequence[dict[str, Any]],
    *,
    backends: tuple[str, ...],
    operations: tuple[str, ...],
    repetitions: int,
    runs: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped_by_repetition: dict[tuple[str, str], dict[int, float]] = {
        (backend, operation): {}
        for backend in backends
        for operation in operations
    }
    timing_cv: dict[str, list[float]] = {backend: [] for backend in backends}
    for measurement in measurements:
        backend = str(measurement["backend"])
        operation = str(measurement["operation"])
        repetition = int(measurement["repetition"])
        key = (backend, operation)
        if repetition in grouped_by_repetition[key]:
            raise ValueError(
                f"Duplicate measurement for {key} repetition {repetition}"
            )
        grouped_by_repetition[key][repetition] = float(measurement["median_ms"])
        mean_ms = float(measurement["mean_ms"])
        std_ms = float(measurement["std_ms"])
        timing_cv[backend].append(0.0 if mean_ms == 0.0 else std_ms / mean_ms)

    grouped: dict[tuple[str, str], list[float]] = {}
    expected_repetitions = set(range(repetitions))
    for key, values_by_repetition in grouped_by_repetition.items():
        if set(values_by_repetition) != expected_repetitions:
            raise ValueError(
                f"Expected repetitions {sorted(expected_repetitions)} for {key}, "
                f"got {sorted(values_by_repetition)}"
            )
        grouped[key] = [
            values_by_repetition[repetition]
            for repetition in range(repetitions)
        ]

    aggregate_latency = {
        key: statistics.median(values) for key, values in grouped.items()
    }
    operation_best = {
        operation: min(
            aggregate_latency[(backend, operation)] for backend in backends
        )
        for operation in operations
    }
    aggregate_scores = {
        backend: _geometric_mean(
            [
                aggregate_latency[(backend, operation)]
                / operation_best[operation]
                for operation in operations
            ]
        )
        for backend in backends
    }

    repetition_winners: list[str] = []
    for repetition in range(repetitions):
        best_by_operation = {
            operation: min(
                grouped[(backend, operation)][repetition]
                for backend in backends
            )
            for operation in operations
        }
        score_by_backend = {
            backend: _geometric_mean(
                [
                    grouped[(backend, operation)][repetition]
                    / best_by_operation[operation]
                    for operation in operations
                ]
            )
            for backend in backends
        }
        repetition_winners.append(
            min(
                score_by_backend,
                key=lambda backend: score_by_backend[backend],
            )
        )

    ranked = sorted(backends, key=lambda backend: aggregate_scores[backend])
    numerical_winner = ranked[0]
    runner_up = ranked[1]
    best_score = aggregate_scores[numerical_winner]
    margin = aggregate_scores[runner_up] / best_score - 1.0
    tied = tuple(
        backend
        for backend in ranked
        if aggregate_scores[backend] <= best_score * (1.0 + _TIE_THRESHOLD)
    )
    recommended = (
        DEFAULT_NTT_BACKEND if DEFAULT_NTT_BACKEND in tied else numerical_winner
    )
    winner_counts = Counter(repetition_winners)
    consistent = winner_counts[numerical_winner] == repetitions
    winner_cv = statistics.median(timing_cv[numerical_winner])
    if recommended != numerical_winner:
        confidence = "low"
        reason = (
            f"{numerical_winner} leads by less than {_TIE_THRESHOLD * 100:.0f}%; "
            f"retaining stable fallback {DEFAULT_NTT_BACKEND}"
        )
    elif repetitions < 3:
        confidence = "low"
        reason = (
            f"{numerical_winner} leads by {margin * 100:.2f}%, but fewer than "
            "three repetitions were requested"
        )
    elif runs < 5:
        confidence = "low"
        reason = (
            f"{numerical_winner} leads by {margin * 100:.2f}%, but fewer than "
            "five timed runs per operation were requested"
        )
    elif consistent and margin >= _HIGH_CONFIDENCE_MARGIN and winner_cv <= 0.05:
        confidence = "high"
        reason = (
            f"{recommended} won every repetition and leads the runner-up by "
            f"{margin * 100:.2f}%"
        )
    elif (
        consistent and margin >= _MEDIUM_CONFIDENCE_MARGIN and winner_cv <= 0.10
    ):
        confidence = "medium"
        reason = (
            f"{recommended} won every repetition and leads the runner-up by "
            f"{margin * 100:.2f}%"
        )
    else:
        confidence = "low"
        reason = (
            f"aggregate winner {numerical_winner} has a {margin * 100:.2f}% "
            f"runner-up margin and won "
            f"{winner_counts[numerical_winner]}/{repetitions} repetitions"
        )

    rows = []
    for rank, backend in enumerate(ranked, start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "backend": backend,
            "pick": backend == recommended,
            "score": round(aggregate_scores[backend], 6),
            "gap_pct": round(
                (aggregate_scores[backend] / best_score - 1.0) * 100.0,
                3,
            ),
            "wins": winner_counts[backend],
            "cv_pct": round(
                statistics.median(timing_cv[backend]) * 100.0,
                3,
            ),
        }
        rows.append(row)

    summary = {
        "recommended_backend": recommended,
        "numerical_winner": numerical_winner,
        "runner_up": runner_up,
        "runner_up_margin_pct": round(margin * 100.0, 3),
        "confidence": confidence,
        "reason": reason,
        "near_tied_backends": list(tied),
        "repetition_winners": repetition_winners,
    }
    return rows, summary


def recommend_ntt_backend(
    *,
    suite: str,
    preset_name: str,
    device: str,
    warmup: int,
    runs: int,
    repetitions: int,
    requested_backends: Sequence[str] | None,
    seed: int,
    progress: ProgressCallback,
) -> BenchmarkResult:
    """Run one named suite and recommend an exact NTT backend name."""

    if suite not in _SUPPORTED_SUITES:
        raise ValueError(
            f"Unsupported NTT recommendation suite {suite!r}; "
            f"choices: {_SUPPORTED_SUITES!r}"
        )
    if warmup < 0 or runs <= 0 or repetitions <= 0:
        raise ValueError(
            "warmup must be non-negative; runs and repetitions must be positive"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NTT backend recommendation")

    preset = _preset(preset_name)
    config = CkksConfig.parse(preset)
    backends = _resolve_candidates(
        log_ring_dimension=config.logN,
        requested=requested_backends,
    )
    if len(backends) < 2:
        raise ValueError("NTT recommendation requires at least two backends")

    operations: tuple[str, ...]
    if suite == "kernel":
        operations = _KERNEL_OPERATIONS
        measurements, measurement_orders = _kernel_measurements(
            preset=preset,
            device=device,
            backends=backends,
            warmup=warmup,
            runs=runs,
            repetitions=repetitions,
            seed=seed,
            progress=progress,
        )
    else:
        operations = _PRIMITIVE_OPERATIONS
        measurements, measurement_orders = _primitive_measurements(
            preset=preset,
            device=device,
            backends=backends,
            warmup=warmup,
            runs=runs,
            repetitions=repetitions,
            progress=progress,
        )

    rows, summary = _rank_measurements(
        measurements,
        backends=backends,
        operations=operations,
        repetitions=repetitions,
        runs=runs,
    )
    device_properties = torch.cuda.get_device_properties(torch.device(device))
    recommended = str(summary["recommended_backend"])
    effective_parameters = {
        "suite": suite,
        "preset": preset.value,
        "device": device,
        "backends": list(backends),
        "operations": list(operations),
        "warmup": warmup,
        "runs": runs,
        "repetitions": repetitions,
        "seed": seed,
    }
    timed_boundary = BenchmarkTimedBoundary(
        id=f"ntt-recommendation-{suite}-operation-v1",
        description=(
            "One prepared semantic NTT operation."
            if suite == "kernel"
            else "One operation-ready CKKS primitive."
        ),
        includes=("one selected operation for one backend",),
        excludes=(
            "engine construction",
            "key generation and operation-ready input preparation",
            "correctness validation",
            "backend ranking and recommendation logic",
        ),
        synchronization="Synchronize the selected CUDA device before and after every sample; rotate backend order by repetition.",
    )
    metrics = [
        BenchmarkMetric(
            name="ntt-recommendation-operation-latency",
            value=float(measurement["median_ms"]),
            unit="ms",
            statistic="median",
            direction="lower",
            dimensions={
                "category": "latency",
                "suite": suite,
                "backend": str(measurement["backend"]),
                "operation": str(measurement["operation"]),
                "repetition": int(measurement["repetition"]),
            },
            samples=tuple(measurement.get("samples_ms", ())),
        )
        for measurement in measurements
    ]
    checks = []
    for backend in backends:
        observed = max(
            float(measurement["max_error"])
            for measurement in measurements
            if measurement["backend"] == backend
        )
        limit = 0.0 if suite == "kernel" else _CORRECTNESS_ATOL
        checks.append(
            BenchmarkCheck(
                name=f"{backend}-correctness",
                passed=observed <= limit,
                oracle=(
                    "Exact equality modulo every active QP prime after NTT roundtrip."
                    if suite == "kernel"
                    else "Decrypted CKKS primitive outputs compared with deterministic cleartext operations."
                ),
                metric=(
                    "mismatched_residues"
                    if suite == "kernel"
                    else "max_abs_error"
                ),
                observed=observed,
                comparison="<=",
                limit=limit,
                unit="residues" if suite == "kernel" else "absolute",
                details={"backend": backend, "suite": suite},
            )
        )
    return BenchmarkResult(
        benchmark="ntt-backend-recommendation",
        profile=suite,
        workload_id=WORKLOAD_ID,
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        scalars={
            **summary,
            "apply": (
                f'CkksEngine(Preset.{preset.name}, device="{device}", '
                f'ntt_backend="{recommended}")'
            ),
        },
        metadata={
            "suite": suite,
            "preset": preset.value,
            "logN": config.logN,
            "device": device,
            "device_name": device_properties.name,
            "compute_capability": [
                device_properties.major,
                device_properties.minor,
            ],
            "fhelium_version": fhelium.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "backends": list(backends),
            "operations": list(operations),
            "warmup": warmup,
            "runs": runs,
            "repetitions": repetitions,
            "seed": seed,
            "measurement_orders": measurement_orders,
            "score": "equal-weight geometric mean of per-operation latency ratios",
            "tie_threshold_pct": _TIE_THRESHOLD * 100.0,
            "correctness_atol": (
                0.0 if suite == "kernel" else _CORRECTNESS_ATOL
            ),
            "measurement_count": len(measurements),
            "workload_id": WORKLOAD_ID,
            "effective_parameters": effective_parameters,
            "timed_boundary": timed_boundary.to_dict(),
        },
        notes=[
            "Recommendation is specific to this GPU, software environment, preset, and suite.",
            "Kernel is a fast screening suite; confirm production choices with ckks-primitive or an application workload.",
            "A less than 3% aggregate gap is treated as a near tie; the stable fallback is retained when it is near-tied.",
            "No default backend or native shared-memory setting is changed by this command.",
        ],
        evidence=measurements,
    )
