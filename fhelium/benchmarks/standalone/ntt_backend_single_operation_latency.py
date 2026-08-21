"""Register single-operation latency comparisons across NTT backends."""

from __future__ import annotations

import gc
from collections.abc import Sequence

import torch

from fhelium import CkksEngine, Preset
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
from fhelium.benchmarks.standalone.ntt_kernel import (
    NTT_OPERATIONS,
    assert_ntt_roundtrip,
    make_residue_rows,
    measure_ntt_operation,
    prepare_ntt_operation_inputs,
)
from fhelium.config import CkksConfig
from fhelium.config.ntt import compatible_ntt_backends


def _preset(name: str) -> Preset:
    try:
        return Preset(name)
    except ValueError as error:
        choices = ", ".join(item.value for item in Preset)
        raise ValueError(
            f"Unknown preset {name!r}; choices: {choices}"
        ) from error


def _resolve_backends(
    *,
    log_ring_dimension: int,
    configured_backends: object,
) -> tuple[str, ...]:
    """Resolve the default all-compatible set or validate a named subset."""

    compatible = compatible_ntt_backends(log_ring_dimension)
    if configured_backends is None:
        return compatible
    if not isinstance(configured_backends, Sequence) or isinstance(
        configured_backends, str
    ):
        raise ValueError(
            "backends must be null or a JSON array of backend names"
        )
    requested = tuple(str(value) for value in configured_backends)
    if not requested:
        raise ValueError("backends must contain at least one backend name")
    if len(set(requested)) != len(requested):
        raise ValueError("backends must not contain duplicate names")
    incompatible = tuple(name for name in requested if name not in compatible)
    if incompatible:
        raise ValueError(
            f"NTT backends {incompatible!r} are not compatible with "
            f"logN={log_ring_dimension}; compatible backends: {compatible!r}"
        )
    return requested


def _resolve_operations(configured_operations: object) -> tuple[str, ...]:
    """Validate the semantic operations requested by one profile."""

    if not isinstance(configured_operations, Sequence) or isinstance(
        configured_operations, str
    ):
        raise ValueError("operations must be a JSON array of operation names")
    operations = tuple(str(value) for value in configured_operations)
    if not operations:
        raise ValueError("operations must contain at least one operation name")
    if len(set(operations)) != len(operations):
        raise ValueError("operations must not contain duplicate names")
    unsupported = tuple(
        operation for operation in operations if operation not in NTT_OPERATIONS
    )
    if unsupported:
        raise ValueError(
            f"Unsupported NTT operations {unsupported!r}; "
            f"choices: {NTT_OPERATIONS!r}"
        )
    return operations


def _run_ntt_backend_single_op(
    profile: BenchmarkProfile,
    progress: ProgressCallback,
) -> BenchmarkResult:
    parameters = profile.parameters
    preset_name = str(parameters["preset"])
    preset = _preset(preset_name)
    config = CkksConfig.parse(preset)
    device = str(parameters.get("device", "cuda:0"))
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    seed = int(parameters.get("seed", 20260530))
    configured_backends = parameters.get("backends")
    backends = _resolve_backends(
        log_ring_dimension=config.logN,
        configured_backends=configured_backends,
    )
    operations = _resolve_operations(
        parameters.get("operations", NTT_OPERATIONS)
    )

    rows = []
    metrics = []
    checks = []
    baseline_by_operation: dict[str, float] = {}
    measured_rows: list[tuple[dict[str, object], float]] = []
    for backend in backends:
        progress(f"Creating {preset_name} engine for {backend}")
        engine = CkksEngine(
            preset,
            device=device,
            ntt_backend=backend,
        )
        standard_rows = make_residue_rows(engine, seed=seed)
        operation_inputs = prepare_ntt_operation_inputs(engine, standard_rows)
        assert_ntt_roundtrip(engine, operation_inputs["roundtrip"])
        checks.append(
            BenchmarkCheck(
                name=f"{backend}-ntt-roundtrip-residue-equality",
                passed=True,
                oracle="Residue equality modulo every active QP prime after forward and inverse NTT.",
                metric="mismatched_residues",
                observed=0,
                comparison="==",
                limit=0,
                unit="residues",
                details={"backend": backend},
            )
        )

        for operation in operations:
            progress(f"Measuring {backend}: {operation}")
            # ``measure_ntt_operation`` predates device arguments and
            # synchronizes the current device. Scope it to the operand device
            # so a caller-selected non-current GPU remains correctly bounded.
            with torch.cuda.device(engine.device):
                timing = measure_ntt_operation(
                    engine,
                    operation_inputs[operation],
                    operation,
                    warmup=warmup,
                    runs=runs,
                )
            if backend == "radix2_indexed":
                baseline_by_operation[operation] = timing["mean_ms"]
            row: dict[str, object] = {
                "backend": backend,
                "operation": operation,
                "mean_ms": round(timing["mean_ms"], 4),
                "median_ms": round(timing["median_ms"], 4),
                "min_ms": round(timing["min_ms"], 4),
                "max_ms": round(timing["max_ms"], 4),
                "std_ms": round(timing["std_ms"], 4),
                "speedup": None,
                "rows": standard_rows.size(0),
                "N": standard_rows.size(1),
                "device": str(engine.device),
            }
            rows.append(row)
            measured_rows.append((row, timing["mean_ms"]))
            metrics.append(
                BenchmarkMetric(
                    name="ntt-operation-latency",
                    value=timing["median_ms"],
                    unit="ms",
                    statistic="median",
                    direction="lower",
                    dimensions={
                        "category": "latency",
                        "backend": backend,
                        "operation": operation,
                    },
                )
            )

        del engine, standard_rows, operation_inputs
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    for row, raw_mean_ms in measured_rows:
        baseline = baseline_by_operation.get(str(row["operation"]))
        if baseline is not None:
            row["speedup"] = round(baseline / raw_mean_ms, 3)

    effective_parameters = dict(parameters)
    resolved_parameters = {
        "preset": preset_name,
        "device": device,
        "backends": list(backends),
        "operations": list(operations),
        "warmup": warmup,
        "runs": runs,
        "seed": seed,
    }
    timed_boundary = BenchmarkTimedBoundary(
        id="in-place-semantic-ntt-operation-v1",
        description="One in-place semantic NTT operation on a prepared representation-specific input.",
        includes=("one forward, inverse, or roundtrip in-place NTT operation",),
        excludes=(
            "engine construction",
            "deterministic residue generation",
            "representation-specific input restoration",
            "roundtrip correctness validation",
        ),
        synchronization="Synchronize the operand CUDA device before and after every sample.",
    )

    return BenchmarkResult(
        benchmark="ntt-backend-single-op",
        profile=profile.name,
        workload_id="ntt-backend-single-op",
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        metadata={
            "workload_id": "ntt-backend-single-op",
            "preset": preset_name,
            "backends": list(backends),
            "operations": list(operations),
            "warmup": warmup,
            "runs": runs,
            "seed": seed,
            "timed_boundary": timed_boundary.to_dict(),
            "effective_parameters": effective_parameters,
            "resolved_parameters": resolved_parameters,
            "level": 0,
            "modulus_basis": "QP",
            "residue_dtype": (
                "torch.int32"
                if config.buffer_bit_length == 30
                else "torch.int64"
            ),
            "forward_input": "coefficient/Montgomery",
            "inverse_input": "NTT/Montgomery",
        },
        notes=[
            "Default profiles enumerate only canonical backends compatible with the selected logN.",
            "Each backend is roundtrip-validated in Montgomery representation before timing.",
            "Forward inputs are coefficient/Montgomery; inverse inputs are NTT/Montgomery.",
            "Timing covers only the semantic NTT operation; resetting the input buffer is excluded.",
            "Speedup uses radix2_indexed latency for the same operation as the baseline.",
        ],
    )


def _profile(
    name: str,
    description: str,
    *,
    preset: str,
    warmup: int,
    runs: int,
) -> BenchmarkProfile:
    return BenchmarkProfile(
        name,
        description,
        {
            "preset": preset,
            "device": "cuda:0",
            "backends": None,
            "operations": list(NTT_OPERATIONS),
            "warmup": warmup,
            "runs": runs,
            "seed": 20260530,
        },
    )


register_benchmark(
    BenchmarkDefinition(
        name="ntt-backend-single-op",
        title="NTT backend single-op latency",
        category="single GPU",
        description=(
            "Compares forward NTT, inverse NTT, and NTT+INTT roundtrip latency "
            "across every canonical FHElium NTT backend compatible with the "
            "selected logN, without higher-level CKKS, key-switch, or rotation "
            "work."
        ),
        profiles=(
            _profile(
                "quick",
                "Short 8,192-slot/40-bit-scale/7-level all-compatible-backend smoke comparison.",
                preset=Preset.slots8192_scale40_levels7_int64.value,
                warmup=1,
                runs=3,
            ),
            _profile(
                "core",
                "Versioned 8,192-slot all-compatible-backend core comparison.",
                preset=Preset.slots8192_scale40_levels7_int64.value,
                warmup=5,
                runs=20,
            ),
            _profile(
                Preset.slots8192_scale40_levels7_int64.value,
                "Stable 8,192-slot/40-bit-scale/7-level all-compatible-backend comparison.",
                preset=Preset.slots8192_scale40_levels7_int64.value,
                warmup=5,
                runs=50,
            ),
            _profile(
                Preset.slots16384_scale40_levels16_int64.value,
                "Stable 16,384-slot/40-bit-scale/16-level all-compatible-backend comparison.",
                preset=Preset.slots16384_scale40_levels16_int64.value,
                warmup=5,
                runs=50,
            ),
            _profile(
                Preset.slots32768_scale40_levels34_int64.value,
                "Stable 32,768-slot/40-bit-scale/34-level all-compatible-backend comparison.",
                preset=Preset.slots32768_scale40_levels34_int64.value,
                warmup=5,
                runs=50,
            ),
            _profile(
                Preset.slots65536_scale40_levels72_int64.value,
                "Stable 65,536-slot/40-bit-scale/72-level all-compatible-backend comparison.",
                preset=Preset.slots65536_scale40_levels72_int64.value,
                warmup=5,
                runs=50,
            ),
        ),
        runner=_run_ntt_backend_single_op,
        workload_id="ntt-backend-single-op",
    )
)
