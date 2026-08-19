"""Fixed affine and degree-four polynomial methods for Benchmark v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
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
from fhelium.benchmarks.timing import measure, synchronize
from fhelium.experimental import bootstrap as bs

from .model import BenchmarkExecution

PRESET = fh.Preset.slots8192_scale40_levels7_int64.value
NTT_BACKEND = "radix2_indexed"
INPUT_RANGE = (-1.0, 1.0)
ENTRY_LEVEL = 0
ENTRY_SCALE = float(2**40)
RNG_SEED = 20260807
RNG_NONCE = 0
AFFINE_COEFFICIENTS = (0.125, 0.75)
DENSE_D4_COEFFICIENTS = (0.01, 0.2, -0.3, 0.125, -0.0625)
# These ceilings retain the prior method-specific acceptance values for the
# same affine and degree-four semantics. The new common plan still requires a
# controlled CPU/CUDA calibration before release; failures are not concealed
# by widening these values.
CORRECTNESS_LIMITS = {
    "affine-d1-balanced": 8.0e-6,
    "dense-power-d4-balanced": 1.4e-5,
    "dense-power-d4-horner": 1.2e-5,
    "dense-power-d4-paterson-stockmeyer-k2": 1.2e-5,
}


@dataclass(frozen=True)
class _MethodCase:
    id: str
    polynomial_id: str
    polynomial: bs.PolynomialApproximation
    evaluator: Any
    method: str
    baby_step: int | None = None

    @property
    def required_levels(self) -> int:
        return int(self.evaluator.required_levels(self.polynomial))

    @property
    def operation_inventory(self) -> dict[str, int]:
        return dict(self.evaluator.operation_inventory(self.polynomial))

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "polynomial_id": self.polynomial_id,
            "basis": self.polynomial.basis,
            "degree": self.polynomial.degree,
            "coefficients_ascending": [
                value.real for value in self.polynomial.coefficients
            ],
            "domain": list(self.polynomial.domain),
            "method": self.method,
            "evaluator": type(self.evaluator).__name__,
            "baby_step": self.baby_step,
            "entry_level": ENTRY_LEVEL,
            "required_levels": self.required_levels,
            "predicted_exit_level": ENTRY_LEVEL + self.required_levels,
            "operation_inventory": self.operation_inventory,
        }


def polynomial_definition(
    polynomial_id: str = "dense-power-d4",
) -> bs.PolynomialApproximation:
    if polynomial_id == "affine-d1":
        coefficients = AFFINE_COEFFICIENTS
    elif polynomial_id == "dense-power-d4":
        coefficients = DENSE_D4_COEFFICIENTS
    else:
        raise KeyError(f"unknown polynomial benchmark case {polynomial_id!r}")
    return bs.PolynomialApproximation(
        basis="power",
        coefficients=coefficients,
        domain=INPUT_RANGE,
        name=polynomial_id,
    )


def polynomial_inputs(num_slots: int) -> torch.Tensor:
    return torch.linspace(
        INPUT_RANGE[0], INPUT_RANGE[1], num_slots, dtype=torch.float64
    )


def plaintext_oracle(
    polynomial: bs.PolynomialApproximation, values: torch.Tensor
) -> torch.Tensor:
    coordinates = values.numpy().astype(np.longdouble, copy=False)
    coefficients = tuple(
        np.longdouble(value.real) for value in polynomial.coefficients
    )
    result = np.zeros_like(coordinates, dtype=np.longdouble)
    for coefficient in reversed(coefficients):
        result = result * coordinates + coefficient
    return torch.from_numpy(np.asarray(result, dtype=np.float64))


def polynomial_method_cases() -> tuple[_MethodCase, ...]:
    affine = polynomial_definition("affine-d1")
    degree_four = polynomial_definition("dense-power-d4")
    return (
        _MethodCase(
            "affine-d1-balanced",
            "affine-d1",
            affine,
            bs.BalancedPowerEvaluator(),
            "balanced-power",
        ),
        _MethodCase(
            "dense-power-d4-balanced",
            "dense-power-d4",
            degree_four,
            bs.BalancedPowerEvaluator(),
            "balanced-power",
        ),
        _MethodCase(
            "dense-power-d4-horner",
            "dense-power-d4",
            degree_four,
            bs.HornerPowerEvaluator(),
            "corrected-horner",
        ),
        _MethodCase(
            "dense-power-d4-paterson-stockmeyer-k2",
            "dense-power-d4",
            degree_four,
            bs.PatersonStockmeyerPowerEvaluator(baby_step=2),
            "paterson-stockmeyer",
            2,
        ),
    )


_METHODS = [case.identity() for case in polynomial_method_cases()]
_PARAMETERS = {
    "preset": PRESET,
    "ntt_backend": NTT_BACKEND,
    "input_range": list(INPUT_RANGE),
    "entry_level": ENTRY_LEVEL,
    "entry_scale": ENTRY_SCALE,
    "rng_seed": RNG_SEED,
    "rng_nonce": RNG_NONCE,
    "method_cases": _METHODS,
    "correctness_limits": dict(CORRECTNESS_LIMITS),
    "correctness_limit_policy": "retained-method-specific-ceilings-pending-cross-backend-calibration",
    "warmup": 0,
    "runs": 2,
    "include_raw_samples": True,
}


def _state(engine: fh.CkksEngine, value: fh.Ciphertext) -> dict[str, Any]:
    return {
        "level": value.level,
        "scale": value.scale,
        "component_count": value.component_count,
        "polynomial_domain": value.polynomial_domain,
        "modulus_basis": value.modulus_basis,
        "residue_representation": value.residue_representation,
        "active_q_count": len(value.prime_ids),
    }


def _run_polynomial(
    profile: BenchmarkProfile,
    progress: ProgressCallback,
    *,
    execution: BenchmarkExecution,
) -> BenchmarkResult:
    parameters = dict(profile.parameters)
    engine = fh.CkksEngine(
        fh.Preset(PRESET),
        device=execution.device,
        ntt_backend=NTT_BACKEND,
        rng_seed=RNG_SEED,
        rng_nonce=RNG_NONCE,
    )
    values = polynomial_inputs(engine.num_slots)
    source = engine.encrypt_message(
        values, level=ENTRY_LEVEL, scale=ENTRY_SCALE
    )
    relinearization_key = engine.relinearization_key
    rows: list[dict[str, Any]] = []
    metrics: list[BenchmarkMetric] = []
    checks: list[BenchmarkCheck] = []
    evidence: list[dict[str, Any]] = []
    method_states: list[dict[str, Any]] = []
    for method in polynomial_method_cases():

        def evaluate() -> fh.Ciphertext:
            return method.evaluator.evaluate(
                engine,
                source,
                method.polynomial,
                relinearization_key=relinearization_key,
            )

        progress(f"Measuring polynomial method {method.id}")
        timing = measure(
            evaluate,
            warmup=int(parameters["warmup"]),
            runs=int(parameters["runs"]),
            device=engine.device,
            include_samples=True,
        )
        result = evaluate()
        synchronize(engine.device)
        actual = engine.decrypt_message(result, is_real=True).cpu()
        expected = plaintext_oracle(method.polynomial, values)
        error = torch.abs(actual - expected)
        max_abs_error = float(error.max())
        rms_error = float(torch.sqrt(torch.mean(error.square())))
        limit = CORRECTNESS_LIMITS[method.id]
        observed_state = _state(engine, result)
        predicted_level = method.required_levels
        state_passed = observed_state["level"] == predicted_level
        dimensions = {
            "category": "latency",
            "case_id": method.polynomial_id,
            "method_id": method.id,
            "degree": method.polynomial.degree,
            "required_levels": method.required_levels,
        }
        metrics.append(
            BenchmarkMetric(
                "polynomial-evaluation-latency",
                timing["median_ms"],
                "ms",
                "median",
                "lower",
                dimensions,
                tuple(timing["samples_ms"]),
            )
        )
        rows.append(
            {
                **dimensions,
                **timing,
                "max_abs_error": max_abs_error,
                "rms_error": rms_error,
                "observed_exit_state": observed_state,
            }
        )
        checks.extend(
            (
                BenchmarkCheck(
                    f"{method.id}-independent-polynomial-oracle",
                    max_abs_error <= limit,
                    "All decoded slots are compared with independent CPU long-double Horner evaluation.",
                    "max_abs_error",
                    max_abs_error,
                    "<=",
                    limit,
                    "absolute",
                    {
                        "rms_error": rms_error,
                        "output_elements": engine.num_slots,
                    },
                ),
                BenchmarkCheck(
                    f"{method.id}-exit-level",
                    state_passed,
                    "Observed output level equals the evaluator's declared required level count.",
                    "level",
                    observed_state["level"],
                    "==",
                    predicted_level,
                    "level",
                    {"observed_exit_state": observed_state},
                ),
            )
        )
        evidence.append(
            {
                "kind": "raw_timing_samples",
                "method_id": method.id,
                "unit": "ms",
                "samples": timing["samples_ms"],
            }
        )
        method_states.append(
            {
                "method_id": method.id,
                "predicted_exit_level": predicted_level,
                "observed_exit_state": observed_state,
            }
        )
    boundary = BenchmarkTimedBoundary(
        id="fixed-affine-degree4-polynomial-method",
        description="One public fixed-polynomial evaluator invocation for one method row.",
        includes=(
            "coefficient preparation",
            "ciphertext arithmetic",
            "relinearization",
            "rescale",
        ),
        excludes=(
            "engine and key construction",
            "input encryption",
            "output decryption and oracle",
        ),
        synchronization="Synchronize the selected engine device before and after every sample; CPU calls complete synchronously.",
    )
    return BenchmarkResult(
        benchmark="polynomial-evaluation",
        profile=profile.name,
        workload_id="polynomial-evaluation",
        effective_parameters=parameters,
        timed_boundary=boundary,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        scalars={
            "method_count": len(rows),
            "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
        },
        metadata={
            "workload_id": "polynomial-evaluation",
            "benchmark_context": {
                "ckks_plan": {
                    "preset": PRESET,
                    "logN": engine.config.logN,
                    "slot_count": engine.num_slots,
                    "ntt_backend": NTT_BACKEND,
                },
                "parameter_selection": {
                    "policy": "common CPU/CUDA affine-and-degree4 plan",
                    "hardware_adaptive": False,
                    "calibration_status": "controlled cross-backend evidence required before release",
                },
                "entry_state": {"level": ENTRY_LEVEL, "scale": ENTRY_SCALE},
                "method_states": method_states,
            },
            "execution": execution.to_dict(),
        },
        notes=[
            "The fixed limits were not changed in response to this implementation; controlled CPU/CUDA calibration remains required before release."
        ],
        evidence=evidence,
    )


DEFINITION = register_benchmark(
    BenchmarkDefinition(
        name="polynomial-evaluation",
        title="Affine and degree-four polynomial methods",
        category="local device",
        description="Compares four fixed affine and degree-four methods on one identical CPU/CUDA plan.",
        profiles=(
            BenchmarkProfile(
                "core",
                "Complete fixed cross-backend polynomial method matrix.",
                _PARAMETERS,
            ),
        ),
        runner=cast(Any, _run_polynomial),
        workload_id="polynomial-evaluation",
    )
)

__all__ = [
    "CORRECTNESS_LIMITS",
    "DEFINITION",
    "NTT_BACKEND",
    "PRESET",
    "plaintext_oracle",
    "polynomial_definition",
    "polynomial_inputs",
    "polynomial_method_cases",
]
