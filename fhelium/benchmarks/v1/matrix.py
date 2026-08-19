"""Fixed local-device dense matrix workloads for Benchmark v1."""

from __future__ import annotations

import statistics
import time
from collections.abc import Mapping, Sequence
from functools import partial
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

from .model import BenchmarkExecution

MATRIX_SIZE = 16
INPUT_SEED = 20260807
CORRECTNESS_ATOL = 3e-5
PRESET = fh.Preset.slots8192_scale40_levels7_int64
NTT_BACKEND = "radix2_indexed"
OperandMode = Literal["ptct", "ctct"]
WORKLOAD_IDS = {
    "ptct": "dense-matrix-multiplication-ptct",
    "ctct": "dense-matrix-multiplication-ctct",
}


def matrix_inputs(
    *, seed: int = INPUT_SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed != INPUT_SEED:
        raise ValueError(f"seed is fixed at {INPUT_SEED} by Benchmark v1")
    row = torch.arange(MATRIX_SIZE, dtype=torch.float64).view(-1, 1)
    column = torch.arange(MATRIX_SIZE, dtype=torch.float64).view(1, -1)
    lhs = 0.018 * torch.sin((row + 1) * (column + 2) * 0.17)
    lhs += 0.007 * torch.cos((row + column + 1) * 0.23)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rhs = 0.025 * torch.randn(
        (MATRIX_SIZE, MATRIX_SIZE), generator=generator, dtype=torch.float64
    )
    return lhs, rhs


def periodic_slots(values: torch.Tensor, num_slots: int = 8192) -> torch.Tensor:
    if values.size(-1) != MATRIX_SIZE or num_slots % MATRIX_SIZE:
        raise ValueError(
            "values and slot count must match the fixed matrix block"
        )
    repeats = [1] * values.ndim
    repeats[-1] = num_slots // MATRIX_SIZE
    return values.repeat(*repeats)


def cyclic_diagonal_slots(
    matrix: torch.Tensor, rotation_step: int, num_slots: int = 8192
) -> torch.Tensor:
    if matrix.shape != (MATRIX_SIZE, MATRIX_SIZE):
        raise ValueError("matrix has the wrong fixed shape")
    row = torch.arange(num_slots) % MATRIX_SIZE
    column = torch.remainder(row - rotation_step, MATRIX_SIZE)
    return matrix[row, column]


def expected_periodic_output(
    lhs: torch.Tensor, rhs: torch.Tensor, *, num_slots: int = 8192
) -> torch.Tensor:
    return periodic_slots((lhs @ rhs).T, num_slots)


def evaluate_ptct_column(
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    diagonal_batch: fh.Plaintext,
    rotation_keys: Mapping[int, fh.RotationKey],
) -> fh.Ciphertext:
    if len(diagonal_batch.batch_shape) != 1:
        raise ValueError(
            "PTxCT matrix column requires exactly one diagonal batch axis; "
            f"got batch_shape={tuple(diagonal_batch.batch_shape)}"
        )
    diagonal_count = diagonal_batch.batch_shape[0]
    if diagonal_count == 0:
        raise ValueError("PTxCT matrix column requires at least one diagonal")
    expected_steps = set(range(1, diagonal_count))
    if set(rotation_keys) != expected_steps:
        raise ValueError(
            "rotation_keys must contain exactly steps 1 through "
            "diagonal_count-1"
        )
    rotated = (
        source,
        *engine.rotate_many_with_keys(
            source,
            [rotation_keys[step] for step in range(1, diagonal_count)],
            use_hoisting=diagonal_count > 2,
        ),
    )
    rotated_ntt = engine.coefficient_domain_to_ntt_domain(
        fh.Ciphertext.stack_batch(rotated)
    )
    products = engine.multiply_plaintext(rotated_ntt, diagonal_batch)
    accumulator = engine.sum_ciphertext_batch(products)
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(accumulator)
    )


def evaluate_ctct_column(
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    diagonal_ciphertexts_ntt: Sequence[fh.Ciphertext],
    rotation_keys: Mapping[int, fh.RotationKey],
    relinearization_key: fh.RelinearizationKey,
) -> fh.Ciphertext:
    accumulator = None
    for step, diagonal in enumerate(diagonal_ciphertexts_ntt):
        rotated = (
            source
            if step == 0
            else engine.rotate_with_key(source, rotation_keys[step])
        )
        triplet = engine.multiply(
            diagonal, engine.coefficient_domain_to_ntt_domain(rotated)
        )
        accumulator = (
            triplet
            if accumulator is None
            else engine.add_(accumulator, triplet)
        )
    assert accumulator is not None
    return engine.rescale_to_next_level(
        engine.relinearize(accumulator, relinearization_key)
    )


def _prepare(
    engine: fh.CkksEngine, operand_mode: OperandMode
) -> tuple[
    Any,
    tuple[fh.Ciphertext, ...],
    dict[int, fh.RotationKey],
    fh.RelinearizationKey | None,
    torch.Tensor,
    torch.Tensor,
]:
    lhs, rhs = matrix_inputs()
    if operand_mode == "ptct":
        left = fh.Plaintext.stack_batch(
            tuple(
                engine.prepare_plaintext_for_multiplication(
                    engine.encode(
                        cyclic_diagonal_slots(lhs, step, engine.num_slots),
                        level=0,
                        scale=engine.config.default_scale,
                    )
                )
                for step in range(MATRIX_SIZE)
            )
        )
    else:
        left = tuple(
            engine.coefficient_domain_to_ntt_domain(
                engine.encrypt_message(
                    cyclic_diagonal_slots(lhs, step, engine.num_slots),
                    level=0,
                    scale=engine.config.default_scale,
                )
            )
            for step in range(MATRIX_SIZE)
        )
    sources = tuple(
        engine.encrypt_message(
            periodic_slots(rhs[:, column], engine.num_slots),
            level=0,
            scale=engine.config.default_scale,
        )
        for column in range(MATRIX_SIZE)
    )
    rotations = {
        step: engine.rotation_key(step) for step in range(1, MATRIX_SIZE)
    }
    relin = engine.relinearization_key if operand_mode == "ctct" else None
    synchronize(engine.device)
    return left, sources, rotations, relin, lhs, rhs


def _run_matrix(
    profile: BenchmarkProfile,
    progress: ProgressCallback,
    *,
    execution: BenchmarkExecution,
    operand_mode: OperandMode,
) -> BenchmarkResult:
    parameters = dict(profile.parameters)
    engine = fh.CkksEngine(
        PRESET,
        device=execution.device,
        ntt_backend=NTT_BACKEND,
        rng_seed=INPUT_SEED,
        rng_nonce=0,
    )
    progress(
        f"Preparing fixed {MATRIX_SIZE}x{MATRIX_SIZE} {operand_mode} workload"
    )
    left, sources, rotations, relin, lhs, rhs = _prepare(engine, operand_mode)

    def evaluate() -> list[fh.Ciphertext]:
        if operand_mode == "ptct":
            diagonal_batch = cast(fh.Plaintext, left)
            return [
                evaluate_ptct_column(engine, source, diagonal_batch, rotations)
                for source in sources
            ]
        ciphertexts = cast(tuple[fh.Ciphertext, ...], left)
        assert relin is not None
        return [
            evaluate_ctct_column(engine, source, ciphertexts, rotations, relin)
            for source in sources
        ]

    for _ in range(int(parameters["warmup"])):
        evaluate()
    synchronize(engine.device)
    samples: list[float] = []
    outputs: list[fh.Ciphertext] = []
    for _ in range(int(parameters["runs"])):
        synchronize(engine.device)
        started = time.perf_counter()
        outputs = evaluate()
        synchronize(engine.device)
        samples.append((time.perf_counter() - started) * 1e3)
    expected = expected_periodic_output(lhs, rhs, num_slots=engine.num_slots)
    actual = torch.stack(
        [engine.decrypt_message(value, is_real=True) for value in outputs]
    )
    error = torch.abs(actual.cpu() - expected)
    max_abs_error = float(error.max())
    rms_error = float(torch.sqrt(torch.mean(error.square())))
    median_ms = statistics.median(samples)
    dimensions = {
        "category": "latency",
        "operand_mode": operand_mode,
        "phase": "end-to-end",
        "matrix_size": MATRIX_SIZE,
    }
    boundary = BenchmarkTimedBoundary(
        id=f"dense-matrix-{operand_mode}-local-device",
        description=f"One complete fixed {MATRIX_SIZE}x{MATRIX_SIZE} encrypted matrix product on one local device.",
        includes=(
            "all output columns",
            "rotations",
            "multiplications",
            "accumulation",
            "relinearization when required",
            "rescale",
        ),
        excludes=(
            "engine and key construction",
            "input preparation and encryption",
            "decryption and correctness oracle",
        ),
        synchronization="Synchronize the selected engine device before and after every sample; CPU calls complete synchronously.",
    )
    return BenchmarkResult(
        benchmark=WORKLOAD_IDS[operand_mode],
        profile=profile.name,
        workload_id=WORKLOAD_IDS[operand_mode],
        effective_parameters=parameters,
        timed_boundary=boundary,
        metrics=[
            BenchmarkMetric(
                "dense-matrix-multiplication-latency",
                median_ms,
                "ms",
                "median",
                "lower",
                dimensions,
                tuple(samples),
            ),
            BenchmarkMetric(
                "dense-matrix-logical-macs-rate",
                MATRIX_SIZE**3 / (median_ms / 1000),
                "logical-macs/s",
                "inverse_median_latency",
                "higher",
                {**dimensions, "category": "throughput"},
            ),
        ],
        correctness=[
            BenchmarkCheck(
                name=f"dense-matrix-{operand_mode}-cleartext-oracle",
                passed=max_abs_error <= CORRECTNESS_ATOL,
                oracle="Every decoded packed output slot is compared with CPU binary64 A @ B.",
                metric="max_abs_error",
                observed=max_abs_error,
                comparison="<=",
                limit=CORRECTNESS_ATOL,
                unit="absolute",
                details={
                    "rms_error": rms_error,
                    "checked_elements": int(actual.numel()),
                },
            )
        ],
        rows=[
            {
                "operand_mode": operand_mode,
                "matrix_size": MATRIX_SIZE,
                "median_ms": median_ms,
                "max_abs_error": max_abs_error,
                "rms_error": rms_error,
            }
        ],
        scalars={
            "matrix_size": MATRIX_SIZE,
            "max_abs_error": max_abs_error,
            "rms_error": rms_error,
        },
        metadata={
            "workload_id": WORKLOAD_IDS[operand_mode],
            "benchmark_context": {
                "ckks_plan": {
                    "preset": PRESET.value,
                    "logN": engine.config.logN,
                    "slot_count": engine.num_slots,
                    "ntt_backend": NTT_BACKEND,
                },
                "parameter_selection": {
                    "policy": "one identical CPU/CUDA local-device plan",
                    "hardware_adaptive": False,
                },
                "entry_state": {
                    "level": 0,
                    "scale": engine.config.default_scale,
                    "matrix_shape": [MATRIX_SIZE, MATRIX_SIZE],
                },
            },
            "execution": execution.to_dict(),
        },
        notes=[
            "The workload is sequential and unbatched on both CPU and CUDA."
        ],
        evidence=[
            {"kind": "raw_timing_samples", "unit": "ms", "samples": samples}
        ],
    )


def _definition(operand_mode: OperandMode) -> BenchmarkDefinition:
    parameters = {
        "preset": PRESET.value,
        "ntt_backend": NTT_BACKEND,
        "matrix_shape": [MATRIX_SIZE, MATRIX_SIZE],
        "input_seed": INPUT_SEED,
        "packing": "cyclic-diagonal-left-periodic-column-right",
        "column_schedule": "sequential-unbatched",
        "correctness_atol": CORRECTNESS_ATOL,
        "warmup": 0,
        "runs": 1,
        "include_raw_samples": True,
    }
    label = (
        "Plaintext x ciphertext"
        if operand_mode == "ptct"
        else "Ciphertext x ciphertext"
    )
    return BenchmarkDefinition(
        name=WORKLOAD_IDS[operand_mode],
        title=f"{label} dense matrix multiplication",
        category="local device",
        description=f"Evaluates one fixed {MATRIX_SIZE}x{MATRIX_SIZE} {label.lower()} workload identically on CPU and CUDA.",
        profiles=(
            BenchmarkProfile(
                "core",
                "Fixed local-device cross-backend matrix workload.",
                parameters,
            ),
        ),
        runner=cast(Any, partial(_run_matrix, operand_mode=operand_mode)),
        workload_id=WORKLOAD_IDS[operand_mode],
    )


PTCT_DEFINITION = register_benchmark(_definition("ptct"))
CTCT_DEFINITION = register_benchmark(_definition("ctct"))

__all__ = [
    "CORRECTNESS_ATOL",
    "CTCT_DEFINITION",
    "INPUT_SEED",
    "MATRIX_SIZE",
    "NTT_BACKEND",
    "PRESET",
    "PTCT_DEFINITION",
    "cyclic_diagonal_slots",
    "evaluate_ctct_column",
    "evaluate_ptct_column",
    "expected_periodic_output",
    "matrix_inputs",
    "periodic_slots",
]
