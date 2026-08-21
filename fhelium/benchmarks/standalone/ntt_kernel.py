"""NTT kernel measurement helpers with representation-state validation."""

from __future__ import annotations

import statistics
import time

import torch

from fhelium import CkksEngine
from fhelium.benchmarks.timing import synchronize

NTT_OPERATIONS = ("forward_ntt", "inverse_ntt", "roundtrip")


def make_residue_rows(engine: CkksEngine, *, seed: int) -> torch.Tensor:
    """Create deterministic standard-residue QP rows for one engine."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows = []
    moduli = [int(value) for value in engine.rns_runtime.moduli.cpu().tolist()]
    for modulus in moduli:
        row = torch.randint(
            0,
            modulus,
            (engine.config.N,),
            dtype=torch.int64,
            generator=generator,
            device="cpu",
        )
        rows.append(row.to(device=engine.device, non_blocking=False))
    return torch.stack(rows, dim=0).contiguous()


def run_ntt_operation(
    engine: CkksEngine,
    data: torch.Tensor,
    operation: str,
) -> None:
    """Run one in-place semantic NTT operation on a prepared operand."""

    if operation == "forward_ntt":
        engine.rns_runtime.forward_montgomery_(data, include_p=True)
    elif operation == "inverse_ntt":
        engine.rns_runtime.inverse_montgomery_(data, include_p=True)
    elif operation == "roundtrip":
        engine.rns_runtime.forward_montgomery_(data, include_p=True)
        engine.rns_runtime.inverse_montgomery_(data, include_p=True)
    else:
        raise ValueError(f"Unsupported NTT operation: {operation}")


def prepare_ntt_operation_inputs(
    engine: CkksEngine,
    standard_rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build valid representation-specific inputs outside timed regions."""

    coefficient_montgomery = standard_rows.clone()
    engine.rns_runtime.to_montgomery_(coefficient_montgomery, include_p=True)
    ntt_montgomery = coefficient_montgomery.clone()
    engine.rns_runtime.forward_montgomery_(ntt_montgomery, include_p=True)
    return {
        "forward_ntt": coefficient_montgomery,
        "inverse_ntt": ntt_montgomery,
        "roundtrip": coefficient_montgomery,
    }


def measure_ntt_operation(
    engine: CkksEngine,
    base_data: torch.Tensor,
    operation: str,
    *,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    """Measure only an NTT operation, excluding restoration of its input."""

    if runs <= 0 or warmup < 0:
        raise ValueError(
            "runs must be positive and warmup must be non-negative"
        )
    data = base_data.clone()
    for _ in range(warmup):
        data.copy_(base_data)
        run_ntt_operation(engine, data, operation)
    synchronize()

    samples = []
    for _ in range(runs):
        data.copy_(base_data)
        synchronize()
        start = time.perf_counter()
        run_ntt_operation(engine, data, operation)
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": statistics.pstdev(samples),
    }


def assert_ntt_roundtrip(
    engine: CkksEngine,
    base_data: torch.Tensor,
) -> None:
    """Require residue equality modulo each active QP prime after a roundtrip."""

    result = base_data.clone()
    run_ntt_operation(engine, result, "roundtrip")
    if not torch.equal(result, base_data):
        difference = torch.remainder(
            result - base_data, engine.rns_runtime.moduli[:, None]
        )
        if not bool(torch.all(difference == 0).item()):
            raise RuntimeError(
                f"NTT roundtrip validation failed for {engine.ntt_backend_name}"
            )
