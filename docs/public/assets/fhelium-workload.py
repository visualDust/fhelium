"""Measure conventional packed BSGS matrix-vector evaluation in FHElium."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import torch

import fhelium as fh
import fhelium.distributed as dist
from fhelium.execution import CudaGraphProgram

Mode = Literal["pt-ct", "ct-ct"]
PRESETS = {
    7: fh.Preset.slots8192_scale40_levels7_int64,
    16: fh.Preset.slots16384_scale40_levels16_int64,
    34: fh.Preset.slots32768_scale40_levels34_int64,
}
CUDA_NTT_BACKENDS: dict[tuple[int, Mode], str] = {
    (7, "pt-ct"): "radix4_compact",
    (7, "ct-ct"): "radix2_compact_group4_smem8",
    (16, "pt-ct"): "radix2_compact_group16_smem8",
    (16, "ct-ct"): "radix2_compact_group16_smem8",
    (34, "pt-ct"): "radix2_compact_group16_smem8",
    (34, "ct-ct"): "radix2_compact_group16_smem8",
}
CPU_THREAD_COUNTS: dict[tuple[int, Mode], int] = {
    (7, "pt-ct"): 32,
    (7, "ct-ct"): 24,
    (16, "pt-ct"): 32,
    (16, "ct-ct"): 32,
    (34, "pt-ct"): 32,
    (34, "ct-ct"): 24,
}


def _inputs(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(size, dtype=torch.float64).view(-1, 1)
    column = torch.arange(size, dtype=torch.float64).view(1, -1)
    matrix = 0.018 * torch.sin((row + 1) * (column + 2) * 0.17)
    matrix += 0.007 * torch.cos((row + column + 1) * 0.23)
    vector = 0.025 * torch.cos(torch.arange(size, dtype=torch.float64) * 0.31)
    vector -= 0.009 * torch.sin(torch.arange(size, dtype=torch.float64) * 0.19)
    return matrix, vector


def _periodic(values: torch.Tensor, slots: int) -> torch.Tensor:
    if slots % values.numel():
        raise ValueError("matrix size must divide the CKKS slot count")
    return values.repeat(slots // values.numel())


def _diagonal(matrix: torch.Tensor, step: int, slots: int) -> torch.Tensor:
    size = matrix.size(0)
    row = torch.arange(slots) % size
    column = torch.remainder(row - step, size)
    return matrix[row, column]


def _bsgs_diagonal(
    matrix: torch.Tensor,
    *,
    giant_index: int,
    baby_index: int,
    baby_step: int,
    slots: int,
) -> torch.Tensor:
    diagonal = _diagonal(
        matrix,
        giant_index * baby_step + baby_index,
        slots,
    )
    return torch.roll(diagonal, shifts=-giant_index * baby_step)


def _summarize(samples: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
    }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(dist.local_device())


def _max_rank_ms(value: float) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(
        value, dtype=torch.float64, device=dist.local_device()
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _required_steps(
    *, baby_step: int, local_giants: tuple[int, ...]
) -> tuple[int, ...]:
    steps = set(range(1, baby_step))
    steps.update(
        giant_index * baby_step
        for giant_index in local_giants
        if giant_index != 0
    )
    return tuple(sorted(steps))


def _provision_rotation_keys(
    engine: fh.CkksEngine,
    *,
    secret_key: fh.SecretKey | None,
    baby_step: int,
    giant_count: int,
    local_giants: tuple[int, ...],
) -> dict[int, fh.RotationKey]:
    local_steps = set(
        _required_steps(
            baby_step=baby_step,
            local_giants=local_giants,
        )
    )
    all_steps = tuple(
        sorted(
            set(range(1, baby_step))
            | {i * baby_step for i in range(1, giant_count)}
        )
    )
    result: dict[int, fh.RotationKey] = {}
    for step in all_steps:
        root_key = None
        if dist.get_rank() == 0:
            assert secret_key is not None
            root_key = engine.create_rotation_key(step, secret_key)
        key = dist.broadcast_key(root_key, src=0)
        if step in local_steps:
            result[step] = key
    return result


def _prepare_groups(
    engine: fh.CkksEngine,
    *,
    mode: Mode,
    matrix: torch.Tensor,
    public_key: fh.PublicKey | None,
    local_giants: tuple[int, ...],
    baby_step: int,
    input_level: int,
) -> tuple[
    dict[int, fh.Plaintext],
    dict[int, fh.Ciphertext],
]:
    plaintext_groups: dict[int, fh.Plaintext] = {}
    ciphertext_groups: dict[int, fh.Ciphertext] = {}
    for giant_index in range(matrix.size(0) // baby_step):
        messages = [
            _bsgs_diagonal(
                matrix,
                giant_index=giant_index,
                baby_index=baby_index,
                baby_step=baby_step,
                slots=engine.num_slots,
            )
            for baby_index in range(baby_step)
        ]
        if mode == "pt-ct":
            if giant_index in local_giants:
                plaintext_groups[giant_index] = fh.Plaintext.stack_batch(
                    [
                        engine.prepare_plaintext_for_multiplication(
                            engine.encode(message, level=input_level)
                        )
                        for message in messages
                    ]
                )
            continue
        root_ciphertexts = None
        if dist.get_rank() == 0:
            assert public_key is not None
            root_ciphertexts = [
                engine.encrypt_message(message, public_key, level=input_level)
                for message in messages
            ]
        received = [
            dist.broadcast_ciphertext(
                None if root_ciphertexts is None else root_ciphertexts[index],
                src=0,
            )
            for index in range(baby_step)
        ]
        if giant_index in local_giants:
            ciphertext_groups[giant_index] = (
                engine.coefficient_domain_to_ntt_domain(
                    fh.Ciphertext.stack_batch(received)
                )
            )
    return plaintext_groups, ciphertext_groups


def _evaluate_local(
    engine: fh.CkksEngine,
    *,
    mode: Mode,
    source: fh.Ciphertext,
    local_giants: tuple[int, ...],
    baby_step: int,
    rotation_keys: dict[int, fh.RotationKey],
    plaintext_groups: dict[int, fh.Plaintext],
    ciphertext_groups: dict[int, fh.Ciphertext],
    relinearization_key: fh.RelinearizationKey | None,
) -> fh.Ciphertext:
    baby_rotations = engine.rotate_many_with_keys(
        source,
        [rotation_keys[step] for step in range(1, baby_step)],
        use_hoisting=True,
    )
    baby_batch = engine.coefficient_domain_to_ntt_domain(
        fh.Ciphertext.stack_batch([source, *baby_rotations])
    )
    accumulator = None
    for giant_index in local_giants:
        if mode == "pt-ct":
            product_batch = engine.multiply_plaintext(
                baby_batch,
                plaintext_groups[giant_index],
            )
            group_ntt = engine.sum_ciphertext_batch(product_batch)
            group = engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(group_ntt)
            )
        else:
            assert relinearization_key is not None
            product_batch = engine.multiply(
                baby_batch,
                ciphertext_groups[giant_index],
            )
            group_triplet = engine.sum_ciphertext_batch(product_batch)
            group = engine.rescale_to_next_level(
                engine.relinearize(group_triplet, relinearization_key)
            )
        if giant_index:
            group = engine.rotate_with_key(
                group,
                rotation_keys[giant_index * baby_step],
            )
        if accumulator is None:
            accumulator = group
        else:
            engine.add_(accumulator, group)
    if accumulator is None:
        raise RuntimeError("rank owns no BSGS giant group")
    return accumulator


def _run_case(
    *,
    depth: int,
    mode: Mode,
    size: int,
    baby_step: int,
    warmup: int,
    runs: int,
    device: str,
    input_level: int,
) -> dict[str, object] | None:
    if size % baby_step:
        raise ValueError("baby_step must divide matrix size")
    if device == "cpu":
        available = set(os.sched_getaffinity(0))
        selected = available & set(range(48))
        if selected:
            os.sched_setaffinity(0, selected)
        torch.set_num_threads(
            int(
                os.environ.get(
                    "FHE_CPU_THREADS",
                    str(CPU_THREAD_COUNTS[(depth, mode)]),
                )
            )
        )
        torch.set_num_interop_threads(1)
    graph_program: CudaGraphProgram[fh.Ciphertext] | None = None
    dist.init()
    try:
        world_size = dist.get_world_size()
        giant_count = size // baby_step
        if world_size > giant_count:
            raise ValueError("world size exceeds BSGS giant count")
        local_giants = tuple(range(dist.get_rank(), giant_count, world_size))
        setup_started = time.perf_counter()
        engine = fh.CkksEngine(
            PRESETS[depth],
            device=dist.local_device() if device == "cuda" else "cpu",
            ntt_backend=(
                CUDA_NTT_BACKENDS[(depth, mode)]
                if device == "cuda"
                else "radix2_indexed"
            ),
            allow_sk_gen=False,
            rng_seed=20260814,
            rng_nonce=depth + (0 if mode == "pt-ct" else 100),
        )
        matrix, vector = _inputs(size)
        secret_key = None
        public_key = None
        root_source = None
        root_relinearization_key = None
        if dist.get_rank() == 0:
            secret_key = engine.create_secret_key()
            public_key = engine.create_public_key(secret_key)
            root_source = engine.encrypt_message(
                _periodic(vector, engine.num_slots),
                public_key,
                level=input_level,
            )
            if mode == "ct-ct":
                root_relinearization_key = engine.create_relinearization_key(
                    secret_key
                )
        source = dist.broadcast_ciphertext(root_source, src=0)
        relinearization_key = None
        if mode == "ct-ct":
            relinearization_key = dist.broadcast_key(
                root_relinearization_key,
                src=0,
            )
        rotation_keys = _provision_rotation_keys(
            engine,
            secret_key=secret_key,
            baby_step=baby_step,
            giant_count=giant_count,
            local_giants=local_giants,
        )
        plaintext_groups, ciphertext_groups = _prepare_groups(
            engine,
            mode=mode,
            matrix=matrix,
            public_key=public_key,
            local_giants=local_giants,
            baby_step=baby_step,
            input_level=input_level,
        )
        _sync()
        dist.barrier()
        setup_ms = _max_rank_ms((time.perf_counter() - setup_started) * 1e3)

        def evaluate_local(dynamic_source: fh.Ciphertext) -> fh.Ciphertext:
            return _evaluate_local(
                engine,
                mode=mode,
                source=dynamic_source,
                local_giants=local_giants,
                baby_step=baby_step,
                rotation_keys=rotation_keys,
                plaintext_groups=plaintext_groups,
                ciphertext_groups=ciphertext_groups,
                relinearization_key=relinearization_key,
            )

        replay_local = evaluate_local
        if device == "cuda":
            graph_program = CudaGraphProgram.capture(
                evaluate_local,
                example_inputs=(source,),
                warmup=warmup,
            )
            replay_local = graph_program.replay

        def evaluate() -> fh.Ciphertext:
            local = replay_local(source)
            dist.reduce_ciphertext(local, dst=0, engine=engine)
            return local

        last = None
        for _ in range(warmup):
            dist.barrier()
            last = evaluate()
            _sync()
        gc.collect()
        samples = []
        for _ in range(runs):
            dist.barrier()
            _sync()
            started = time.perf_counter()
            last = evaluate()
            _sync()
            samples.append(_max_rank_ms((time.perf_counter() - started) * 1e3))
        assert last is not None
        max_abs_error = 0.0
        rms_error = 0.0
        if dist.get_rank() == 0:
            assert secret_key is not None
            actual = engine.decrypt_message(
                last,
                secret_key,
                is_real=True,
            ).cpu()
            expected = (matrix @ vector).repeat(engine.num_slots // size)
            error = actual - expected
            max_abs_error = float(error.abs().max())
            rms_error = float(torch.sqrt(torch.mean(error.square())))
        max_abs_error = _max_rank_ms(max_abs_error)
        correctness_threshold = 5e-5 * max(1.0, size / 128)
        if max_abs_error > correctness_threshold:
            raise AssertionError(
                f"max_abs_error={max_abs_error} exceeds "
                f"correctness_threshold={correctness_threshold}"
            )
        if dist.get_rank() != 0:
            return None
        config = engine.config
        return {
            "library": "FHElium",
            "version": fh.__version__,
            "mode": mode,
            "algorithm": "baby-step/giant-step cyclic-diagonal packed dense matrix-vector",
            "schedule": "hoisted babies, adjusted diagonals, per-group completion and rescale, giant rotations",
            "cuda_graph": graph_program is not None,
            "cuda_graph_scope": (
                "rank-local BSGS evaluation; ciphertext reduction remains eager"
                if graph_program is not None
                else None
            ),
            "cuda_graph_capture_ms": (
                graph_program.stats.capture_seconds * 1e3
                if graph_program is not None
                else None
            ),
            "matrix_size": size,
            "baby_step": baby_step,
            "giant_count": giant_count,
            "world_size": world_size,
            "device": device,
            "depth_label": depth,
            "preset": PRESETS[depth].value,
            "ntt_backend": engine.ntt_backend_name,
            "ring_dimension": config.N,
            "q_product_bits": math.prod(config.q_moduli).bit_length(),
            "qp_product_bits": math.prod(config.moduli).bit_length(),
            "multiplication_q_bits": math.prod(
                config.q_moduli[input_level:]
            ).bit_length(),
            "input_level": input_level,
            "output_level": input_level + 1,
            "threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "setup_ms": setup_ms,
            "warmup": warmup,
            "runs": runs,
            "max_abs_error": max_abs_error,
            "correctness_threshold": correctness_threshold,
            "rms_error": rms_error,
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices": (
                [torch.cuda.get_device_name(i) for i in range(world_size)]
                if device == "cuda"
                else [platform.processor()]
            ),
            **_summarize(samples),
        }
    finally:
        if graph_program is not None:
            graph_program.close()
        dist.shutdown()


def _launch_cuda(args: argparse.Namespace) -> None:
    with tempfile.NamedTemporaryFile(
        prefix="fhelium-bsgs-", suffix=".json", delete=False
    ) as file:
        worker_output = Path(file.name)
    try:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.gpus}",
            str(Path(__file__).resolve()),
            "--worker",
            "--device",
            "cuda",
            "--gpus",
            str(args.gpus),
            "--depth",
            str(args.depth),
            "--mode",
            args.mode,
            "--size",
            str(args.size),
            "--baby-step",
            str(args.baby_step),
            "--warmup",
            str(args.warmup),
            "--runs",
            str(args.runs),
            "--input-level",
            str(args.input_level),
            "--output",
            str(worker_output),
        ]
        subprocess.run(command, check=True)
        args.output.write_text(worker_output.read_text())
        print(worker_output.read_text(), end="")
    finally:
        worker_output.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--depth", type=int, choices=PRESETS, required=True)
    parser.add_argument("--mode", choices=("pt-ct", "ct-ct"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--baby-step", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--input-level", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not args.worker:
        _launch_cuda(args)
        return
    result = _run_case(
        depth=args.depth,
        mode=args.mode,
        size=args.size,
        baby_step=args.baby_step,
        warmup=args.warmup,
        runs=args.runs,
        device=args.device,
        input_level=args.input_level,
    )
    if result is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not args.worker:
            print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
