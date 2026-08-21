"""Register SPMD collective and rotation-parallel matvec benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from fhelium import Preset
from fhelium.benchmarks.model import (
    BenchmarkDefinition,
    BenchmarkProfile,
    BenchmarkResult,
    ProgressCallback,
)
from fhelium.benchmarks.registry import register_benchmark


def _resolve_world_size(parameters: dict, *, visible: int) -> int:
    requested = parameters.get("world_size", "visible")
    if isinstance(requested, str) and requested.lower() in {
        "all",
        "all-visible",
        "visible",
    }:
        world_size = visible
    else:
        if isinstance(requested, bool):
            raise ValueError("world_size must be an integer or 'visible'")
        try:
            world_size = int(requested)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"world_size must be an integer or 'visible', got {requested!r}"
            ) from error
    if world_size <= 0:
        if visible <= 0:
            raise RuntimeError(
                "Distributed benchmark requires a visible CUDA device"
            )
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size > visible:
        raise RuntimeError(
            f"Benchmark requires {world_size} CUDA devices, but {visible} are visible"
        )
    return world_size


def _run_worker(
    *,
    kind: str,
    profile: BenchmarkProfile,
    progress: ProgressCallback,
) -> BenchmarkResult:
    parameters = dict(profile.parameters)
    visible = torch.cuda.device_count()
    world_size = _resolve_world_size(parameters, visible=visible)

    result_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="fhelium-benchmark-", suffix=".json", delete=False
        ) as result_file:
            result_path = Path(result_file.name)

        launcher_started_at = time.perf_counter()
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            "-m",
            "fhelium.benchmarks.standalone.distributed_worker",
            "--kind",
            kind,
            "--profile",
            profile.name,
            "--parameters",
            json.dumps(parameters),
            "--result",
            str(result_path),
            "--launcher-started-at",
            str(launcher_started_at),
        ]
        progress(f"Launching {world_size} local ranks")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            timeout=float(parameters.get("timeout_seconds", 600)),
        )
        if completed.returncode != 0:
            output = "\n".join(
                part for part in (completed.stdout, completed.stderr) if part
            )
            tail = "\n".join(output.splitlines()[-40:])
            raise RuntimeError(
                f"Distributed benchmark worker failed with exit code "
                f"{completed.returncode}:\n{tail}"
            )
        if not result_path.exists() or result_path.stat().st_size == 0:
            raise RuntimeError("Distributed worker did not write a result file")
        result = BenchmarkResult.from_dict(json.loads(result_path.read_text()))
        result.scalars["torchrun_total_wall_ms"] = (
            time.perf_counter() - launcher_started_at
        ) * 1e3
        result.metadata["visible_cuda_devices"] = visible
        result.metadata["world_size_request"] = parameters.get(
            "world_size", "visible"
        )
        expected_identity = (
            "spmd-collectives"
            if kind == "collectives"
            else "spmd-ckks-rotation-workload"
        )
        actual_identity = result.workload_id
        if actual_identity != expected_identity:
            raise RuntimeError(
                "Distributed worker returned an unexpected workload identity: "
                f"{actual_identity!r} != {expected_identity!r}"
            )
        result.metadata.update(
            {
                "workload_id": result.workload_id,
                "timed_boundary": result.timed_boundary.to_dict(),
                "effective_parameters": dict(result.effective_parameters),
            }
        )
        return result
    finally:
        if result_path is not None:
            result_path.unlink(missing_ok=True)


def _collectives(
    profile: BenchmarkProfile, progress: ProgressCallback
) -> BenchmarkResult:
    return _run_worker(kind="collectives", profile=profile, progress=progress)


def _rotation_matvec(
    profile: BenchmarkProfile, progress: ProgressCallback
) -> BenchmarkResult:
    return _run_worker(
        kind="ckks-rotation-matvec", profile=profile, progress=progress
    )


register_benchmark(
    BenchmarkDefinition(
        name="spmd-collectives",
        title="SPMD collective throughput",
        category="multi GPU",
        description=(
            "Launches local torchrun ranks and measures the named in-place "
            "broadcast and all-reduce over process-local CUDA tensors. "
            "Reported latency is the slowest rank for each iteration."
        ),
        profiles=(
            BenchmarkProfile(
                "quick-2gpu",
                "Two-rank smoke profile over 1 and 16 MiB payloads.",
                {
                    "world_size": 2,
                    "sizes_mib": [1, 16],
                    "warmup": 3,
                    "runs": 10,
                    "timeout_seconds": 180,
                },
            ),
            BenchmarkProfile(
                "standard-2gpu",
                "Two-rank communication profile through 256 MiB.",
                {
                    "world_size": 2,
                    "sizes_mib": [1, 16, 64, 256],
                    "warmup": 10,
                    "runs": 50,
                    "timeout_seconds": 300,
                },
            ),
        ),
        runner=_collectives,
        workload_id="spmd-collectives",
    )
)

register_benchmark(
    BenchmarkDefinition(
        name="spmd-ckks-rotation-workload",
        title="SPMD CKKS rotation-parallel matrix-vector workload",
        category="multi GPU",
        description=(
            "Runs a packed dense matrix-vector product by partitioning cyclic "
            "diagonals and direct rotation keys across local ranks. Profiles use "
            "every CUDA device visible to the launcher by default. Bounded "
            "grouped rotations, homogeneous diagonal-term batching, and "
            "optional rank-local CUDA Graph execution are declared profile "
            "parameters; "
            "CUDA_VISIBLE_DEVICES controls the selected device set. Startup, "
            "rank-local encrypted computation, and ciphertext reduction are "
            "reported separately."
        ),
        profiles=(
            BenchmarkProfile(
                "slots8192-scale40-levels7-int64-matvec128",
                "128x128 8,192-slot/40-bit-scale/7-level packed matvec across every visible GPU.",
                {
                    "world_size": "visible",
                    "preset": Preset.slots8192_scale40_levels7_int64.value,
                    "ntt_backend": "radix2_compact_group8_smem8",
                    "matrix_size": 128,
                    "hoist_chunk_size": 64,
                    "batch_diagonal_terms": True,
                    "diagonal_batch_size": 16,
                    "use_cuda_graph": True,
                    "cuda_graph_warmup": 3,
                    "warmup": 2,
                    "runs": 5,
                    "timeout_seconds": 900,
                },
            ),
            BenchmarkProfile(
                "slots16384-scale40-levels16-int64-matvec128",
                "128x128 16,384-slot/40-bit-scale/16-level packed matvec across every visible GPU.",
                {
                    "world_size": "visible",
                    "preset": Preset.slots16384_scale40_levels16_int64.value,
                    "ntt_backend": "radix2_compact_group16_smem8",
                    "matrix_size": 128,
                    "hoist_chunk_size": 64,
                    "batch_diagonal_terms": True,
                    "diagonal_batch_size": 8,
                    "use_cuda_graph": True,
                    "cuda_graph_warmup": 3,
                    "warmup": 2,
                    "runs": 5,
                    "timeout_seconds": 1200,
                },
            ),
            BenchmarkProfile(
                "slots32768-scale40-levels34-int64-matvec128",
                "128x128 32,768-slot/40-bit-scale/34-level packed matvec across every visible GPU.",
                {
                    "world_size": "visible",
                    "preset": Preset.slots32768_scale40_levels34_int64.value,
                    "ntt_backend": "radix2_compact_group16_smem8",
                    "matrix_size": 128,
                    "hoist_chunk_size": 64,
                    "batch_diagonal_terms": False,
                    "use_cuda_graph": True,
                    "cuda_graph_warmup": 3,
                    "warmup": 2,
                    "runs": 5,
                    "timeout_seconds": 1800,
                },
            ),
            BenchmarkProfile(
                "slots32768-scale40-levels34-int64-matvec64",
                "64x64 32,768-slot/40-bit-scale/34-level packed matvec across every visible GPU.",
                {
                    "world_size": "visible",
                    "preset": Preset.slots32768_scale40_levels34_int64.value,
                    "ntt_backend": "radix2_compact_group8_smem8",
                    "matrix_size": 64,
                    "hoist_chunk_size": 16,
                    "use_cuda_graph": True,
                    "cuda_graph_warmup": 3,
                    "warmup": 2,
                    "runs": 5,
                    "timeout_seconds": 1200,
                },
            ),
            BenchmarkProfile(
                "slots32768-scale40-levels34-int64-matvec256",
                "256x256 32,768-slot/40-bit-scale/34-level packed matvec; two or more GPUs recommended.",
                {
                    "world_size": "visible",
                    "preset": Preset.slots32768_scale40_levels34_int64.value,
                    "ntt_backend": "radix2_compact_group8_smem8",
                    "matrix_size": 256,
                    "minimum_world_size": 2,
                    "hoist_chunk_size": 16,
                    "use_cuda_graph": True,
                    "cuda_graph_warmup": 3,
                    "warmup": 1,
                    "runs": 3,
                    "timeout_seconds": 2400,
                },
            ),
            BenchmarkProfile(
                "slots8192-scale40-levels7-int64-matvec16",
                "16x16 8,192-slot/40-bit-scale/7-level packed matvec correctness and launch smoke.",
                {
                    "world_size": "visible",
                    "preset": Preset.slots8192_scale40_levels7_int64.value,
                    "ntt_backend": "radix2_compact_group8_smem8",
                    "matrix_size": 16,
                    "hoist_chunk_size": 8,
                    "use_cuda_graph": True,
                    "cuda_graph_warmup": 3,
                    "warmup": 1,
                    "runs": 3,
                    "timeout_seconds": 300,
                },
            ),
        ),
        runner=_rotation_matvec,
        workload_id="spmd-ckks-rotation-workload",
    )
)
