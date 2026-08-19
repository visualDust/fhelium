"""Internal ``torchrun`` worker for distributed benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch

import fhelium as fh
import fhelium.distributed as dist
from fhelium.benchmarks.model import (
    BenchmarkCheck,
    BenchmarkMetric,
    BenchmarkResult,
    BenchmarkTimedBoundary,
)
from fhelium.execution import CudaGraphProgram

COLLECTIVES_WORKLOAD_ID = "spmd-collectives"
ROTATION_MATVEC_WORKLOAD_ID = "spmd-ckks-rotation-workload"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _values_across_ranks(
    value: float,
    device: torch.device,
) -> list[float]:
    """Collect one scalar from each rank in rank order."""

    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if not dist.is_initialized():
        return [float(tensor.item())]
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [float(item.item()) for item in gathered]


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "std_ms": statistics.pstdev(samples),
        "samples_ms": list(samples),
    }


def _measure_collective(
    operation,
    *,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        dist.barrier()
        operation()
        _sync()

    samples = []
    for _ in range(runs):
        dist.barrier()
        _sync()
        start = time.perf_counter()
        operation()
        _sync()
        elapsed = (time.perf_counter() - start) * 1e3
        samples.append(_max_across_ranks(elapsed, dist.local_device()))
    return _summary(samples)


def _collective_benchmark(
    profile: str,
    parameters: dict,
) -> BenchmarkResult:
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    sizes_mib = tuple(float(value) for value in parameters["sizes_mib"])
    if not sizes_mib or any(value <= 0.0 for value in sizes_mib):
        raise ValueError("sizes_mib must contain positive payload sizes")
    rows: list[dict[str, Any]] = []
    metrics: list[BenchmarkMetric] = []
    checks: list[BenchmarkCheck] = []
    evidence: list[dict[str, Any]] = []

    for size_mib in sizes_mib:
        numel = max(1, int(float(size_mib) * 1024**2) // 4)
        tensor = torch.empty(
            numel, dtype=torch.float32, device=dist.local_device()
        )
        payload_bytes = tensor.numel() * tensor.element_size()

        def run_broadcast() -> None:
            tensor.fill_(dist.get_rank() + 1)
            dist.broadcast(tensor, src=0)

        def run_all_reduce() -> None:
            tensor.fill_(dist.get_rank() + 1)
            dist.all_reduce(tensor)

        broadcast = _measure_collective(run_broadcast, warmup=warmup, runs=runs)
        all_reduce = _measure_collective(
            run_all_reduce, warmup=warmup, runs=runs
        )
        expected_all_reduce = (
            dist.get_world_size() * (dist.get_world_size() + 1) / 2.0
        )
        for operation, timing, function, expected_value in (
            ("broadcast", broadcast, run_broadcast, 1.0),
            (
                "all_reduce",
                all_reduce,
                run_all_reduce,
                expected_all_reduce,
            ),
        ):
            seconds = timing["mean_ms"] / 1000.0
            rows.append(
                {
                    "operation": operation,
                    "payload_mib": float(size_mib),
                    "mean_ms": round(timing["mean_ms"], 4),
                    "median_ms": round(timing["median_ms"], 4),
                    "min_ms": round(timing["min_ms"], 4),
                    "payload_gib_per_s": round(
                        payload_bytes / seconds / 1024**3, 3
                    ),
                }
            )
            median_seconds = timing["median_ms"] / 1000.0
            dimensions = {
                "operation": operation,
                "payload_mib": float(size_mib),
                "world_size": dist.get_world_size(),
            }
            metrics.extend(
                (
                    BenchmarkMetric(
                        name="collective-latency",
                        value=timing["median_ms"],
                        unit="ms",
                        statistic="slowest_rank_median",
                        direction="lower",
                        dimensions={"category": "latency", **dimensions},
                        samples=tuple(timing["samples_ms"]),
                    ),
                    BenchmarkMetric(
                        name="collective-payload-throughput",
                        value=payload_bytes / median_seconds / 1024**3,
                        unit="GiB/s",
                        statistic="payload_over_slowest_rank_median_latency",
                        direction="higher",
                        dimensions={"category": "throughput", **dimensions},
                    ),
                )
            )
            evidence.append(
                {
                    "kind": "raw_timing_samples",
                    "operation": operation,
                    "payload_mib": float(size_mib),
                    "unit": "ms",
                    "samples": timing["samples_ms"],
                }
            )

            function()
            _sync()
            local_valid = bool(torch.all(tensor == expected_value).item())
            invalid_rank_flags = _values_across_ranks(
                float(not local_valid), dist.local_device()
            )
            invalid_ranks = int(sum(invalid_rank_flags))
            check = BenchmarkCheck(
                name=f"{operation}-{float(size_mib):g}-mib-content",
                passed=invalid_ranks == 0,
                oracle=(
                    "Broadcast output equals rank-0 fill value on every rank."
                    if operation == "broadcast"
                    else "All-reduce output equals the exact sum of rank fill values on every rank."
                ),
                metric="invalid_rank_count",
                observed=invalid_ranks,
                comparison="==",
                limit=0,
                unit="ranks",
                details={
                    "expected_value": expected_value,
                    "payload_elements": tensor.numel(),
                },
            )
            checks.append(check)
            if not check.passed:
                raise AssertionError(
                    f"{operation} content validation failed for "
                    f"{size_mib} MiB on {invalid_ranks} rank(s)"
                )

    names = [
        torch.cuda.get_device_name(index)
        for index in range(dist.get_world_size())
    ]
    effective_parameters = dict(parameters)
    resolved_parameters = {
        **parameters,
        "resolved_world_size": dist.get_world_size(),
        "resolved_backend": dist.get_backend(),
    }
    timed_boundary = BenchmarkTimedBoundary(
        id="barrier-bounded-collective-v1",
        description="One in-place collective measured as the slowest participating rank.",
        includes=("rank-local tensor fill", "one in-place collective"),
        excludes=(
            "torchrun startup",
            "process-group initialization",
            "tensor allocation",
            "post-measurement content validation",
        ),
        synchronization="Barrier before each iteration, CUDA synchronization around the operation, then maximum elapsed time across ranks.",
    )
    return BenchmarkResult(
        benchmark="spmd-collectives",
        profile=profile,
        workload_id=COLLECTIVES_WORKLOAD_ID,
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=rows,
        metadata={
            "world_size": dist.get_world_size(),
            "devices": names,
            "backend": dist.get_backend(),
            "runs": runs,
            "warmup": warmup,
            "workload_id": COLLECTIVES_WORKLOAD_ID,
            "effective_parameters": effective_parameters,
            "resolved_parameters": resolved_parameters,
            "timed_boundary": timed_boundary.to_dict(),
        },
        notes=[
            "Latency is the maximum rank latency per iteration.",
            "Throughput is payload bytes divided by latency, not topology-adjusted bus bandwidth.",
        ],
        evidence=evidence,
    )


def _matrix_and_vector(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(size, dtype=torch.float64).view(-1, 1)
    column = torch.arange(size, dtype=torch.float64).view(1, -1)
    matrix = 0.018 * torch.sin((row + 1) * (column + 2) * 0.17)
    matrix += 0.007 * torch.cos((row + column + 1) * 0.23)
    vector = 0.025 * torch.cos(torch.arange(size, dtype=torch.float64) * 0.31)
    vector -= 0.009 * torch.sin(torch.arange(size, dtype=torch.float64) * 0.19)
    return matrix, vector


def _periodic_slots(values: torch.Tensor, num_slots: int) -> torch.Tensor:
    if num_slots % values.numel() != 0:
        raise ValueError(
            f"matrix size {values.numel()} must divide num_slots={num_slots}"
        )
    return values.repeat(num_slots // values.numel())


def _cyclic_diagonal_slots(
    matrix: torch.Tensor,
    rotation_step: int,
    num_slots: int,
) -> torch.Tensor:
    size = matrix.size(0)
    row = torch.arange(num_slots) % size
    column = torch.remainder(row - rotation_step, size)
    return matrix[row, column]


def _allocate_rotation_key_buffer(
    engine: fh.CkksEngine,
) -> fh.RotationKey:
    """Preallocate a reusable receiver so timing excludes setup transfer."""

    prime_ids = engine.rns_layout.prime_ids(0, include_p=True)
    return fh.RotationKey(
        data=torch.empty(
            (
                engine.rns_layout.key_digit_count,
                2,
                len(prime_ids),
                engine.config.N,
            ),
            device=dist.local_device(),
            dtype=engine.config.torch_dtype,
        ),
        context_id=engine.context.context_id,
        prime_ids=prime_ids,
        rotation_step=1,
    )


def _timed_key_broadcast(
    key: fh.RotationKey,
) -> float:
    dist.barrier()
    _sync()
    start = time.perf_counter()
    dist.broadcast(key.data, src=0)
    _sync()
    elapsed = (time.perf_counter() - start) * 1e3
    return _max_across_ranks(elapsed, dist.local_device())


def _evaluate_rotation_matvec(
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    local_rotation_steps: list[int],
    local_keys: fh.RotationKeySet,
    local_diagonals: dict[int, fh.Plaintext],
    local_diagonal_batches: dict[tuple[int, ...], fh.Plaintext],
    hoist_chunk_size: int,
    batch_diagonal_terms: bool,
    diagonal_batch_size: int,
) -> fh.Ciphertext:
    """Evaluate one rank's terms with bounded grouped-rotation batches."""

    if hoist_chunk_size <= 0:
        raise ValueError("hoist_chunk_size must be positive")

    accumulator = None
    remaining_steps = local_rotation_steps
    if remaining_steps and remaining_steps[0] == 0:
        accumulator = engine.rescale_to_next_level(
            engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(source),
                    local_diagonals[0],
                )
            )
        )
        remaining_steps = remaining_steps[1:]

    for start in range(0, len(remaining_steps), hoist_chunk_size):
        chunk = remaining_steps[start : start + hoist_chunk_size]
        rotated_values = engine.rotate_many_with_keys(
            source,
            [local_keys[step] for step in chunk],
            use_hoisting=len(chunk) > 1,
        )
        if batch_diagonal_terms:
            for batch_start in range(0, len(chunk), diagonal_batch_size):
                batch_end = batch_start + diagonal_batch_size
                term_steps = tuple(chunk[batch_start:batch_end])
                rotated_batch = fh.Ciphertext.stack_batch(
                    rotated_values[batch_start:batch_end]
                )
                product_batch = engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(rotated_batch),
                    local_diagonal_batches[term_steps],
                )
                term = engine.rescale_to_next_level(
                    engine.ntt_domain_to_coefficient_domain(
                        engine.sum_ciphertext_batch(product_batch)
                    )
                )
                if accumulator is None:
                    accumulator = term
                else:
                    engine.add_(accumulator, term)
            continue
        for rotation_step, rotated in zip(chunk, rotated_values, strict=True):
            term = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(
                    engine.multiply_plaintext(
                        engine.coefficient_domain_to_ntt_domain(rotated),
                        local_diagonals[rotation_step],
                    )
                )
            )
            if accumulator is None:
                accumulator = term
            else:
                engine.add_(accumulator, term)
    assert accumulator is not None
    return accumulator


def _run_rotation_matvec(
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    evaluate_local: Callable[[fh.Ciphertext], fh.Ciphertext],
) -> tuple[fh.Ciphertext, float, float, float]:
    dist.barrier()
    _sync()
    total_start = time.perf_counter()

    compute_start = total_start
    partial = evaluate_local(source)
    _sync()
    compute_ms = (time.perf_counter() - compute_start) * 1e3

    reduction_start = time.perf_counter()
    dist.reduce_ciphertext(partial, dst=0, engine=engine)
    _sync()
    reduction_ms = (time.perf_counter() - reduction_start) * 1e3
    total_ms = (time.perf_counter() - total_start) * 1e3
    return partial, compute_ms, reduction_ms, total_ms


def _ckks_rotation_matvec_benchmark(
    profile: str,
    parameters: dict,
    *,
    process_group_init_ms: float,
    launcher_started_at: float | None,
) -> BenchmarkResult:
    preset = fh.Preset(str(parameters["preset"]))
    matrix_size = int(parameters["matrix_size"])
    warmup = int(parameters["warmup"])
    runs = int(parameters["runs"])
    minimum_world_size = int(parameters.get("minimum_world_size", 1))
    tolerance = float(parameters.get("atol", 5e-5))
    hoist_chunk_size = int(parameters.get("hoist_chunk_size", 1))
    batch_diagonal_terms = bool(parameters.get("batch_diagonal_terms", False))
    diagonal_batch_size = int(
        parameters.get("diagonal_batch_size", hoist_chunk_size)
    )
    use_cuda_graph = bool(parameters.get("use_cuda_graph", False))
    cuda_graph_warmup = int(parameters.get("cuda_graph_warmup", 3))
    world_size = dist.get_world_size()
    if matrix_size <= 0:
        raise ValueError(f"matrix_size must be positive, got {matrix_size}")
    if world_size > matrix_size:
        raise ValueError(
            f"world_size={world_size} exceeds matrix_size={matrix_size}"
        )
    if world_size < minimum_world_size:
        raise ValueError(
            f"profile requires at least {minimum_world_size} ranks, got "
            f"world_size={world_size}"
        )
    if hoist_chunk_size <= 0:
        raise ValueError("hoist_chunk_size must be positive")
    if diagonal_batch_size <= 0:
        raise ValueError("diagonal_batch_size must be positive")
    if cuda_graph_warmup < 0:
        raise ValueError("cuda_graph_warmup must be non-negative")

    dist.barrier()
    _sync()
    setup_start = time.perf_counter()
    engine = fh.CkksEngine(
        preset,
        device=dist.local_device(),
        allow_sk_gen=False,
        ntt_backend=str(parameters["ntt_backend"]),
    )
    if engine.num_slots % matrix_size != 0:
        raise ValueError(
            f"matrix_size={matrix_size} must divide num_slots={engine.num_slots}"
        )

    matrix, vector = _matrix_and_vector(matrix_size)
    if dist.get_rank() == 0:
        secret_key = engine.create_secret_key()
        public_key = engine.create_public_key(secret_key)
        root_source = engine.encrypt_message(
            _periodic_slots(vector, engine.num_slots),
            public_key,
        )
    else:
        secret_key = None
        root_source = None
    source = dist.broadcast_ciphertext(root_source, src=0)

    selected_key = _allocate_rotation_key_buffer(engine)
    local_key_table: dict[int, fh.RotationKey] = {}
    key_broadcast_samples = []
    for rotation_step in range(1, matrix_size):
        if dist.get_rank() == 0:
            assert secret_key is not None
            selected_key.data.copy_(
                engine.create_rotation_key(rotation_step, secret_key).data
            )
        key_broadcast_samples.append(_timed_key_broadcast(selected_key))
        selected_key.rotation_step = fh.RotationKey.canonical_step(
            rotation_step,
            ring_dimension=engine.config.N,
        )
        owner = rotation_step % world_size
        if dist.get_rank() == owner:
            local_key_table[rotation_step] = selected_key.clone()

    key_bytes = selected_key.data.nbytes
    del selected_key

    local_rotation_steps = list(range(dist.get_rank(), matrix_size, world_size))
    local_keys = fh.RotationKeySet(local_key_table)
    local_diagonals = {
        rotation_step: engine.prepare_plaintext_for_multiplication(
            engine.encode(
                _cyclic_diagonal_slots(matrix, rotation_step, engine.num_slots),
                level=source.level,
            )
        )
        for rotation_step in local_rotation_steps
    }
    remaining_steps = local_rotation_steps
    if remaining_steps and remaining_steps[0] == 0:
        remaining_steps = remaining_steps[1:]
    rotation_chunks = [
        tuple(remaining_steps[start : start + hoist_chunk_size])
        for start in range(0, len(remaining_steps), hoist_chunk_size)
    ]
    local_diagonal_batches = {}
    if batch_diagonal_terms:
        diagonal_term_chunks = [
            chunk[start : start + diagonal_batch_size]
            for chunk in rotation_chunks
            for start in range(0, len(chunk), diagonal_batch_size)
        ]
        local_diagonal_batches = {
            chunk: fh.Plaintext.stack_batch(
                [local_diagonals[rotation_step] for rotation_step in chunk]
            )
            for chunk in diagonal_term_chunks
        }
        # Keep only the unrotated scalar term in individual form. Batched
        # operation-ready diagonals replace, rather than duplicate, the
        # remaining persistent plaintext storage.
        local_diagonals = (
            {0: local_diagonals[0]} if 0 in local_diagonals else {}
        )

    eager_local = partial(
        _evaluate_rotation_matvec,
        engine,
        local_rotation_steps=local_rotation_steps,
        local_keys=local_keys,
        local_diagonals=local_diagonals,
        local_diagonal_batches=local_diagonal_batches,
        hoist_chunk_size=hoist_chunk_size,
        batch_diagonal_terms=batch_diagonal_terms,
        diagonal_batch_size=diagonal_batch_size,
    )
    graph_program = None
    evaluate_local: Callable[[fh.Ciphertext], fh.Ciphertext] = eager_local
    if use_cuda_graph:
        graph_program = CudaGraphProgram.capture(
            eager_local,
            example_inputs=(source,),
            warmup=cuda_graph_warmup,
        )
        evaluate_local = graph_program.replay

    dist.barrier()
    _sync()
    setup_ms = _max_across_ranks(
        (time.perf_counter() - setup_start) * 1e3,
        dist.local_device(),
    )
    launcher_to_ready_ms = process_group_init_ms + setup_ms
    if launcher_started_at is not None:
        launcher_to_ready_ms = _max_across_ranks(
            (time.perf_counter() - launcher_started_at) * 1e3,
            dist.local_device(),
        )

    last_result = None
    for _ in range(warmup):
        last_result = None
        last_result, _, _, _ = _run_rotation_matvec(
            engine,
            source,
            evaluate_local,
        )

    gc.collect()
    _sync()
    resident_allocated = _values_across_ranks(
        float(torch.cuda.memory_allocated(dist.local_device())),
        dist.local_device(),
    )
    torch.cuda.reset_peak_memory_stats(dist.local_device())

    compute_samples = []
    reduction_samples = []
    total_samples = []
    for _ in range(runs):
        last_result = None
        last_result, compute_ms, reduction_ms, total_ms = _run_rotation_matvec(
            engine,
            source,
            evaluate_local,
        )
        compute_samples.append(
            _max_across_ranks(compute_ms, dist.local_device())
        )
        reduction_samples.append(
            _max_across_ranks(reduction_ms, dist.local_device())
        )
        total_samples.append(_max_across_ranks(total_ms, dist.local_device()))

    _sync()
    peak_allocated = _values_across_ranks(
        float(torch.cuda.max_memory_allocated(dist.local_device())),
        dist.local_device(),
    )
    peak_reserved = _values_across_ranks(
        float(torch.cuda.max_memory_reserved(dist.local_device())),
        dist.local_device(),
    )

    compute = _summary(compute_samples)
    reduction = _summary(reduction_samples)
    total = _summary(total_samples)
    max_error = 0.0
    if dist.get_rank() == 0:
        assert secret_key is not None
        assert last_result is not None
        decoded = engine.decrypt_message(
            last_result,
            secret_key=secret_key,
            is_real=True,
        )[:matrix_size]
        expected = matrix @ vector
        max_error = float(torch.max(torch.abs(decoded - expected)))
    max_error = _max_across_ranks(max_error, dist.local_device())
    if max_error > tolerance:
        raise AssertionError(
            "Rotation-parallel matrix-vector result exceeded tolerance: "
            f"max_abs_error={max_error}, atol={tolerance}"
        )

    process_group_init_ms = _max_across_ranks(
        process_group_init_ms,
        dist.local_device(),
    )
    devices = [torch.cuda.get_device_name(index) for index in range(world_size)]
    local_diagonal_counts = [
        len(range(rank, matrix_size, world_size)) for rank in range(world_size)
    ]
    local_rotation_key_counts = [
        sum(step != 0 for step in range(rank, matrix_size, world_size))
        for rank in range(world_size)
    ]
    gib = 1024**3
    graph_stats = None if graph_program is None else graph_program.stats
    effective_parameters = dict(parameters)
    resolved_parameters = {
        **parameters,
        "resolved_world_size": world_size,
        "resolved_device_names": devices,
        "resolved_ntt_backend": engine.ntt_backend_name,
        "local_diagonal_counts": local_diagonal_counts,
        "local_rotation_key_counts": local_rotation_key_counts,
    }
    timed_boundary = BenchmarkTimedBoundary(
        id="rank-compute-and-ciphertext-reduction-v1",
        description="One distributed packed matrix-vector evaluation through root reduction.",
        includes=(
            "rank-local encrypted cyclic-diagonal evaluation",
            "ciphertext tree reduction to rank zero",
        ),
        excludes=(
            "torchrun and process-group startup",
            "engine, key, and diagonal construction",
            "optional CUDA Graph capture",
            "input encryption",
            "output decryption and oracle comparison",
        ),
        synchronization="Barrier before each iteration; synchronize local CUDA work and report the maximum elapsed time across ranks for each phase.",
    )
    timing_metrics = []
    for phase, timing in (
        ("rank-local-compute", compute),
        ("ciphertext-reduction", reduction),
        ("end-to-end", total),
    ):
        timing_metrics.append(
            BenchmarkMetric(
                name="distributed-packed-matvec-latency",
                value=timing["median_ms"],
                unit="ms",
                statistic="slowest_rank_median",
                direction="lower",
                dimensions={
                    "category": "latency",
                    "phase": phase,
                    "world_size": world_size,
                    "matrix_size": matrix_size,
                },
                samples=tuple(timing["samples_ms"]),
            )
        )
    metrics = [
        *timing_metrics,
        BenchmarkMetric(
            name="distributed-packed-matvec-throughput",
            value=1000.0 / total["median_ms"],
            unit="matrix-vectors/s",
            statistic="inverse_slowest_rank_median_latency",
            direction="higher",
            dimensions={
                "category": "throughput",
                "world_size": world_size,
                "matrix_size": matrix_size,
            },
        ),
        BenchmarkMetric(
            name="distributed-packed-matvec-max-rank-peak-allocated",
            value=max(peak_allocated) / gib,
            unit="GiB",
            statistic="maximum_rank_peak",
            direction="lower",
            dimensions={
                "category": "memory",
                "world_size": world_size,
                "matrix_size": matrix_size,
            },
        ),
        BenchmarkMetric(
            name="distributed-packed-matvec-max-rank-peak-reserved",
            value=max(peak_reserved) / gib,
            unit="GiB",
            statistic="maximum_rank_peak",
            direction="lower",
            dimensions={
                "category": "memory",
                "world_size": world_size,
                "matrix_size": matrix_size,
            },
        ),
    ]
    checks = [
        BenchmarkCheck(
            name="distributed-packed-matvec-cleartext-oracle",
            passed=max_error <= tolerance,
            oracle="Rank-zero decryption compared with the deterministic CPU-binary64 matrix @ vector.",
            metric="max_abs_error",
            observed=max_error,
            comparison="<=",
            limit=tolerance,
            unit="absolute",
            details={
                "matrix_size": matrix_size,
                "world_size": world_size,
            },
        )
    ]
    result = BenchmarkResult(
        benchmark="spmd-ckks-rotation-workload",
        profile=profile,
        workload_id=ROTATION_MATVEC_WORKLOAD_ID,
        effective_parameters=effective_parameters,
        timed_boundary=timed_boundary,
        metrics=metrics,
        correctness=checks,
        rows=[
            {
                "phase": "process-group initialization",
                "scope": "one-time startup",
                "median_ms": round(process_group_init_ms, 4),
                "mean_ms": round(process_group_init_ms, 4),
            },
            {
                "phase": "CKKS resources and execution preparation",
                "scope": "one-time setup",
                "median_ms": round(setup_ms, 4),
                "mean_ms": round(setup_ms, 4),
            },
            {
                "phase": "launcher through workload ready",
                "scope": "one-time startup",
                "median_ms": round(launcher_to_ready_ms, 4),
                "mean_ms": round(launcher_to_ready_ms, 4),
            },
            {
                "phase": "rank-local encrypted matvec",
                "scope": "per iteration",
                "median_ms": round(compute["median_ms"], 4),
                "mean_ms": round(compute["mean_ms"], 4),
            },
            {
                "phase": "ciphertext tree reduction",
                "scope": "per iteration",
                "median_ms": round(reduction["median_ms"], 4),
                "mean_ms": round(reduction["mean_ms"], 4),
            },
            {
                "phase": "parallel matrix-vector end-to-end",
                "scope": "per iteration",
                "median_ms": round(total["median_ms"], 4),
                "mean_ms": round(total["mean_ms"], 4),
            },
        ],
        scalars={
            "matrix_vectors_per_second": 1000.0 / total["median_ms"],
            "diagonal_terms_per_second": matrix_size
            * 1000.0
            / total["median_ms"],
            "max_abs_error": max_error,
            "mean_rotation_key_broadcast_ms": (
                statistics.fmean(key_broadcast_samples)
                if key_broadcast_samples
                else 0.0
            ),
            "rotation_key_mib": key_bytes / 1024**2,
            "aggregate_rotation_key_gib": (matrix_size - 1)
            * key_bytes
            / 1024**3,
            "max_rank_rotation_key_gib": max(local_rotation_key_counts)
            * key_bytes
            / gib,
            "max_rank_resident_allocated_gib": max(resident_allocated) / gib,
            "max_rank_peak_allocated_gib": max(peak_allocated) / gib,
            "max_rank_peak_reserved_gib": max(peak_reserved) / gib,
            "aggregate_resident_allocated_gib": sum(resident_allocated) / gib,
            "aggregate_peak_allocated_gib": sum(peak_allocated) / gib,
        },
        metadata={
            "world_size": world_size,
            "devices": devices,
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", "<not set: all devices visible>"
            ),
            "preset": parameters["preset"],
            "matrix_size": matrix_size,
            "diagonal_count": matrix_size,
            "rotation_count": matrix_size - 1,
            "local_diagonal_counts": local_diagonal_counts,
            "local_rotation_key_counts": local_rotation_key_counts,
            "resident_allocated_gib_per_rank": [
                value / gib for value in resident_allocated
            ],
            "peak_allocated_gib_per_rank": [
                value / gib for value in peak_allocated
            ],
            "peak_reserved_gib_per_rank": [
                value / gib for value in peak_reserved
            ],
            "rotation_step_assignment": "round-robin",
            "rotation_mode": "bounded grouped exact-key rotations",
            "hoist_chunk_size": hoist_chunk_size,
            "batch_diagonal_terms": batch_diagonal_terms,
            "diagonal_batch_size": diagonal_batch_size,
            "use_cuda_graph": use_cuda_graph,
            "cuda_graph_warmup": cuda_graph_warmup,
            "cuda_graph_capture_seconds": (
                None if graph_stats is None else graph_stats.capture_seconds
            ),
            "accumulation_mode": (
                "streaming grouped-rotation chunks with batched diagonal "
                "multiply/rescale and batch-tree modular reduction"
                if batch_diagonal_terms
                else "streaming grouped-rotation chunks"
            ),
            "ntt_backend": engine.ntt_backend_name,
            "runs": runs,
            "warmup": warmup,
            "workload_id": ROTATION_MATVEC_WORKLOAD_ID,
            "effective_parameters": effective_parameters,
            "resolved_parameters": resolved_parameters,
            "timed_boundary": timed_boundary.to_dict(),
        },
        notes=[
            "CUDA_VISIBLE_DEVICES defines the local GPU set; profiles launch one rank per visible device unless world_size is overridden.",
            "Launcher-to-ready startup includes torchrun process creation, Python imports, process-group initialization, engine construction, input encryption, exact-key provisioning, diagonal encoding, and optional graph capture.",
            "Grouped rotations are bounded by the configured hoist chunk size; completed chunks are accumulated immediately.",
            "When batch_diagonal_terms is enabled, each completed rotation chunk is split into homogeneous message batches of at most diagonal_batch_size for matching batched plaintext multiply and rescale before its terms are accumulated.",
            "A batched term group is summed through contiguous-half modular-addition rounds, preserving the zero-copy native RNS batch ABI while avoiding one under-filled addition launch per message.",
            "CUDA Graph capture, when selected, covers only rank-local encrypted evaluation; final ciphertext reduction remains eager.",
            "Per-iteration matrix-vector timing excludes startup, warmup, key creation, and input encryption.",
            "The result is reduced only to rank 0 with CKKS modular addition over a general binomial tree.",
        ],
        evidence=[
            {
                "kind": "raw_timing_samples",
                "phase": phase,
                "unit": "ms",
                "samples": timing["samples_ms"],
            }
            for phase, timing in (
                ("rank-local-compute", compute),
                ("ciphertext-reduction", reduction),
                ("end-to-end", total),
            )
        ],
    )
    if graph_program is not None:
        graph_program.close()
    return result


def _write_result(path: Path, result: BenchmarkResult) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("collectives", "ckks-rotation-matvec"),
        required=True,
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--launcher-started-at", type=float)
    args = parser.parse_args()
    parameters = json.loads(args.parameters)

    init_start = time.perf_counter()
    dist.init()
    try:
        dist.barrier()
        _sync()
        process_group_init_ms = (time.perf_counter() - init_start) * 1e3
        if args.kind == "collectives":
            result = _collective_benchmark(args.profile, parameters)
        else:
            result = _ckks_rotation_matvec_benchmark(
                args.profile,
                parameters,
                process_group_init_ms=process_group_init_ms,
                launcher_started_at=args.launcher_started_at,
            )
        if dist.get_rank() == 0:
            _write_result(args.result, result)
        dist.barrier()
    finally:
        dist.shutdown()


if __name__ == "__main__":
    main()
