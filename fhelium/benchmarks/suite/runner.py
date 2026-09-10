"""Sequential suite execution, correctness qualification and run checkpoints."""

from __future__ import annotations

import gc
import hashlib
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from fhelium import __version__
from fhelium.benchmarks.io import write_json_atomic
from fhelium.benchmarks.timing import synchronize
from fhelium.runtime import CpuTopology

from .specification import (
    ABSOLUTE_ERROR_LIMIT,
    SAMPLES,
    WARMUPS,
    SuiteCell,
    cells,
    configurations,
    specification,
    specification_hash,
)
from .workloads import PreparedWorkload, WorkloadInputs


def collect_platform(device: torch.device) -> dict[str, Any]:
    """Collect hardware and runtime fields used to compare complete runs."""
    cpu = CpuTopology.probe()
    gpu = (
        torch.cuda.get_device_properties(device)
        if device.type == "cuda"
        else None
    )
    return {
        "device": str(device),
        "name": gpu.name if gpu else cpu.model,
        "cpu": cpu.model,
        "logical_processors": cpu.logical_processor_count,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ[key]
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
            )
            if key in os.environ
        },
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda if gpu else None,
        "python": platform.python_version(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "compute_capability": list(torch.cuda.get_device_capability(device))
        if gpu
        else None,
        "device_memory_bytes": gpu.total_memory if gpu else None,
    }


def _source() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    result = {"version": __version__}
    if (root / ".git").exists():
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "fhelium", "csrc"], cwd=root
        )
        result["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    result["suite_code_sha256"] = digest.hexdigest()
    return result


def _qualify(prepared: PreparedWorkload, output: Any) -> dict[str, Any]:
    actual = prepared.decode(output)
    if isinstance(output, torch.Tensor):
        if not torch.equal(actual, prepared.expected):
            raise ArithmeticError("Residue output differs from the integer oracle")
        return {
            "max_absolute_error": 0,
            "rms_error": 0,
            "checked_elements": actual.numel(),
            "qualification": "exact-residues",
            "output_depth": prepared.output_depth,
            "output_scale": None,
            "output_components": None,
            "output_domain": prepared.output_domain,
        }
    error = (actual - prepared.expected).abs()
    maximum = float(error.max())
    if not bool(torch.isfinite(error).all()) or maximum > ABSOLUTE_ERROR_LIMIT:
        raise ArithmeticError(
            f"Decoded max absolute error {maximum:.8g} exceeds {ABSOLUTE_ERROR_LIMIT}"
        )
    value = output[0] if isinstance(output, tuple) else output
    if value.depth != prepared.output_depth:
        raise ArithmeticError(
            f"Output depth {value.depth} != {prepared.output_depth}"
        )
    return {
        "max_absolute_error": maximum,
        "rms_error": float(error.square().mean().sqrt()),
        "checked_elements": actual.numel(),
        "output_depth": value.depth,
        "output_scale": value.scale,
        "output_components": value.component_count,
        "output_domain": value.polynomial_domain,
    }


def measure(prepared: PreparedWorkload, device: torch.device) -> dict[str, Any]:
    """Measure a qualified callable; failed checks never produce latency rows."""
    output = prepared.evaluate()
    _qualify(prepared, output)
    del output
    for _ in range(WARMUPS):
        output = prepared.evaluate()
        del output
    synchronize(device)
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = None
    samples = []
    for _ in range(SAMPLES):
        synchronize(device)
        start = time.perf_counter()
        output = prepared.evaluate()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        del output
    peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    output = prepared.evaluate()
    accuracy = _qualify(prepared, output)
    del output
    quartiles = statistics.quantiles(samples, n=4)
    return {
        "median_ms": statistics.median(samples),
        "q25_ms": quartiles[0],
        "q75_ms": quartiles[2],
        "samples_ms": samples,
        "required_evaluation_key_bytes": prepared.key_bytes,
        "resident_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "temporary_allocated_bytes": peak - baseline
        if peak is not None and baseline is not None
        else None,
        "input_scale": prepared.input_scale,
        **accuracy,
    }


def run_suite(
    device: str,
    output_path: Path,
    *,
    progress: Callable[[str], None] = print,
    selected_cells: Sequence[SuiteCell] | None = None,
) -> dict[str, Any]:
    """Run the suite and checkpoint every result.

    ``selected_cells`` is a diagnostic subset and produces a partial report.
    The public suite command always executes the full specification.
    """
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("The suite executes on CPU or CUDA")
    if selected_device.type == "cuda" and selected_device.index is None:
        selected_device = torch.device("cuda", torch.cuda.current_device())
    inventory = tuple(selected_cells) if selected_cells is not None else cells()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    report: dict[str, Any] = {
        "suite": "fhelium-local",
        "specification_hash": specification_hash(),
        "specification": specification(),
        "id": started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8],
        "started_at": started.isoformat(),
        "finished_at": None,
        "status": "running",
        "coverage": "partial" if selected_cells is not None else "full",
        "platform": collect_platform(selected_device),
        "source": _source(),
        "results": [
            {"id": cell.id, **asdict(cell), "status": "pending"}
            for cell in inventory
        ],
    }
    write_json_atomic(output_path, report)
    configs = configurations()
    owner = None
    current = None
    try:
        for number, (cell, row) in enumerate(
            zip(inventory, report["results"], strict=True), 1
        ):
            progress(f"[{number}/{len(inventory)}] {cell.id}")
            prepared = None
            try:
                identity = (cell.config, cell.workload, cell.size)
                if current != identity:
                    owner = None
                    gc.collect()
                    if selected_device.type == "cuda":
                        torch.cuda.empty_cache()
                    owner = WorkloadInputs(
                        configs[cell.config], selected_device
                    )
                    current = identity
                assert owner is not None
                prepared = owner.prepare(cell)
                row["measurement"] = measure(prepared, selected_device)
                row["status"] = "passed"
                row["measurement"]["tasks_per_second"] = (
                    cell.batch * 1000 / row["measurement"]["median_ms"]
                )
                del prepared
            except torch.cuda.OutOfMemoryError:
                prepared = None
                row["status"] = "capacity"
                owner = None
                current = None
                gc.collect()
                torch.cuda.empty_cache()
            except NotImplementedError as error:
                row["status"] = "unsupported"
                row["reason"] = str(error)
            except Exception as error:
                row["status"] = "failed"
                row["failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                progress(f"  FAILED {type(error).__name__}: {error}")
                if isinstance(error, torch.AcceleratorError):
                    # CUDA execution faults can leave the context unusable.
                    # Keep later cells pending rather than reporting cascades.
                    break
            finally:
                prepared = None
            write_json_atomic(output_path, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    else:
        report["status"] = (
            "failed"
            if any(row["status"] == "failed" for row in report["results"])
            else "completed"
        )
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        report["counts"] = dict(
            Counter(row["status"] for row in report["results"])
        )
        write_json_atomic(output_path, report)
    return report
