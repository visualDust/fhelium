"""Measure conventional packed BSGS matrix-vector evaluation in FHElium.

Eager runs directly. On CUDA, JIT compiles the same rank-local function and
captures it with CudaGraphProgram; cross-rank ciphertext reduction remains
outside capture. CPU JIT uses Compile without CUDA Graph. Both paths use the
same live inputs, keys, diagonal tensors, and CKKS parameters. Timed CUDA
replays include refreshing the graph input buffer. Setup, compilation, graph
capture, encryption, decryption, and correctness checks are not timed.

The homepage uses input depth zero, five warmups and sixty alternating samples
per execution mode. GPU matrices are 256x256: two ranks use baby step 16;
one rank uses 32 except Depth-16 PTxCT, which uses 16. GPU intermediates remain
in NTT. CPU matrices are 16x16: PTxCT uses baby step 4 and coefficient-domain
completion; CTxCT uses baby step 8 and NTT-domain completion.

PTxCT diagonals use CompressedPlaintext with contiguous NTT repetition and
2 * matrix_size stored values per prime row. Conversion checks the ordinary
encoded plaintext bit-for-bit; multiplication reads the compact data directly.
CTxCT diagonals remain encrypted ciphertexts.
"""

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
from fhelium.eager import Engine
from fhelium import compile as fc
from fhelium.runtime import CudaGraphProgram

Mode = Literal["pt-ct", "ct-ct"]
# Fixed benchmark chains: ordered rescale primes, terminal Q prime, then P.
# Ring dimensions and Q/QP bit budgets match the retained references.
# Preset defaults may evolve independently of these benchmark chains.
BENCHMARK_PARAMETERS: dict[
    int, tuple[int, tuple[int, ...], tuple[int, ...]]
] = {
    7: (
        14,
        (
            1099510054913,
            1099515691009,
            1099508121601,
            1099515789313,
            1099507695617,
            1099516280833,
            1099506515969,
            1152921504606748673,
        ),
        (1152921504606683137,),
    ),
    16: (
        15,
        (
            1099510054913,
            1099515691009,
            1099507695617,
            1099516280833,
            1099506515969,
            1099520606209,
            1099504549889,
            1099523555329,
            1099503894529,
            1099527946241,
            1099503370241,
            1099529060353,
            1099498258433,
            1099531223041,
            1099469684737,
            1099532009473,
            1152921504606584833,
        ),
        (1152921504598720513, 1152921504597016577),
    ),
    34: (
        16,
        (
            1099510054913,
            1099515691009,
            1099507695617,
            1099516870657,
            1099506515969,
            1099521458177,
            1099503894529,
            1099522375681,
            1099490000897,
            1099523555329,
            1099489607681,
            1099525128193,
            1099486855169,
            1099526176769,
            1099484889089,
            1099529060353,
            1099480956929,
            1099535220737,
            1099469684737,
            1099536138241,
            1099468767233,
            1099537580033,
            1099461820417,
            1099538104321,
            1099457495041,
            1099540725761,
            1099455004673,
            1099540856833,
            1099454218241,
            1099591974913,
            1099453431809,
            1099629723649,
            1099451465729,
            1099630510081,
            1152921504606584833,
        ),
        (
            1152921504598720513,
            1152921504597016577,
            1152921504595968001,
            1152921504592822273,
        ),
    ),
}


def _configuration(depth: int) -> fh.CkksConfig:
    log_n, q_moduli, p_moduli = BENCHMARK_PARAMETERS[depth]
    return fh.CkksConfig(
        default_scale=float(1 << 40),
        logN=log_n,
        q_depth_groups=tuple((prime,) for prime in q_moduli),
        p_moduli=p_moduli,
    )


CUDA_NTT_BACKENDS: dict[tuple[int, Mode], str] = {
    (7, "pt-ct"): "radix4_compact",
    (7, "ct-ct"): "radix2_compact_group4_smem8",
    (16, "pt-ct"): "radix2_compact_group16_smem8",
    (16, "ct-ct"): "radix2_compact_group16_smem8",
    (34, "pt-ct"): "radix2_compact_group16_smem8",
    (34, "ct-ct"): "radix2_compact_group16_smem8",
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    engine: Engine,
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
    engine: Engine,
    *,
    mode: Mode,
    matrix: torch.Tensor,
    public_key: fh.PublicKey | None,
    local_giants: tuple[int, ...],
    baby_step: int,
    input_depth: int,
    device: torch.device,
) -> tuple[
    dict[int, fh.CompressedPlaintext],
    dict[int, fh.Ciphertext],
]:
    plaintext_groups: dict[int, fh.CompressedPlaintext] = {}
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
                plaintext_groups[giant_index] = (
                    fh.CompressedPlaintext.stack_batch(
                        [
                            fh.CompressedPlaintext.from_plaintext(
                                engine.prepare_plaintext_for_multiplication(
                                    engine.encode(
                                        message,
                                        depth=input_depth,
                                        device=device,
                                    )
                                ),
                                unique_count=2 * matrix.size(0),
                                compression_layout="contiguous",
                            )
                            for message in messages
                        ]
                    )
                )
            continue
        root_ciphertexts = None
        if dist.get_rank() == 0:
            assert public_key is not None
            root_ciphertexts = [
                engine.encrypt_message(
                    message,
                    public_key,
                    depth=input_depth,
                    device=device,
                )
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
    engine: Engine,
    *,
    mode: Mode,
    source: fh.Ciphertext,
    local_giants: tuple[int, ...],
    baby_step: int,
    rotation_keys: dict[int, fh.RotationKey],
    plaintext_groups: dict[int, fh.CompressedPlaintext],
    ciphertext_groups: dict[int, fh.Ciphertext],
    relinearization_key: fh.RelinearizationKey | None,
    retained_domain: Literal["coefficient", "ntt"],
) -> fh.Ciphertext:
    baby_rotations = engine.rotate_many_with_keys(
        source,
        [rotation_keys[step] for step in range(1, baby_step)],
        use_hoisting=True,
        output_domain=retained_domain,
    )
    if retained_domain == "ntt":
        source_for_batch = engine.coefficient_domain_to_ntt_domain(source)
        baby_batch = fh.Ciphertext.stack_batch(
            [source_for_batch, *baby_rotations]
        )
    else:
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
            group = engine.sum_ciphertext_batch(product_batch)
            if retained_domain == "coefficient":
                group = engine.ntt_domain_to_coefficient_domain(group)
            group = engine.rescale_to_next_depth(group)
        else:
            assert relinearization_key is not None
            product_batch = engine.multiply(
                baby_batch,
                ciphertext_groups[giant_index],
            )
            group_triplet = engine.sum_ciphertext_batch(product_batch)
            group = engine.rescale_to_next_depth(
                engine.relinearize(
                    group_triplet,
                    relinearization_key,
                    output_domain=retained_domain,
                )
            )
        if giant_index:
            group = engine.rotate_with_key(
                group,
                rotation_keys[giant_index * baby_step],
                output_domain=retained_domain,
            )
        if accumulator is None:
            accumulator = group
        else:
            accumulator = engine.add(accumulator, group)
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
    input_depth: int,
    retained_domain: Literal["coefficient", "ntt"],
    execution: str,
    cuda_graph: bool,
) -> dict[str, object] | None:
    if size % baby_step:
        raise ValueError("baby_step must divide matrix size")
    graph_programs: dict[str, CudaGraphProgram[fh.Ciphertext]] = {}
    dist.init()
    try:
        world_size = dist.get_world_size()
        execution_device = (
            dist.local_device() if device == "cuda" else torch.device("cpu")
        )
        giant_count = size // baby_step
        if world_size > giant_count:
            raise ValueError("world size exceeds BSGS giant count")
        local_giants = tuple(range(dist.get_rank(), giant_count, world_size))
        setup_started = time.perf_counter()
        engine = Engine(
            _configuration(depth),
            ntt_backend=(
                CUDA_NTT_BACKENDS[(depth, mode)]
                if device == "cuda"
                else "radix2_indexed"
            ),
            allow_automatic_key_generation=False,
            rng_seed=20260814,
            rng_nonce=depth + (0 if mode == "pt-ct" else 100),
        )
        matrix, vector = _inputs(size)
        secret_key = None
        public_key = None
        root_source = None
        root_relinearization_key = None
        if dist.get_rank() == 0:
            secret_key = engine.create_secret_key(device=execution_device)
            public_key = engine.create_public_key(secret_key)
            root_source = engine.encrypt_message(
                _periodic(vector, engine.num_slots),
                public_key,
                depth=input_depth,
                device=execution_device,
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
            input_depth=input_depth,
            device=execution_device,
        )
        _sync(execution_device)
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
                retained_domain=retained_domain,
            )

        names = ("eager", "jit") if execution == "both" else (execution,)
        evaluators = {}
        preparations = {}
        compiled = None
        for name in names:
            prepare_started = time.perf_counter()
            evaluator = evaluate_local
            if name == "jit":
                compiled = fc.compile(evaluate_local)
                compiled.prepare(source)
                evaluator = compiled
            if name == "jit" and execution_device.type == "cuda" and cuda_graph:
                graph = CudaGraphProgram.capture(
                    evaluator, example_inputs=(source,), warmup=warmup
                )
                graph_programs[name] = graph
                evaluator = graph.replay
            evaluators[name] = evaluator
            _sync(execution_device)
            preparations[name] = _max_rank_ms(
                (time.perf_counter() - prepare_started) * 1e3
            )

        def evaluate(name: str) -> fh.Ciphertext:
            local = evaluators[name](source)
            dist.reduce_ciphertext(local, dst=0, engine=engine)
            return local

        last = {}
        for _ in range(warmup):
            for name in names:
                dist.barrier()
                last[name] = evaluate(name)
                _sync(execution_device)
        gc.collect()
        samples: dict[str, list[float]] = {name: [] for name in names}
        for iteration in range(runs):
            order = names if iteration % 2 == 0 else tuple(reversed(names))
            for name in order:
                dist.barrier()
                _sync(execution_device)
                started = time.perf_counter()
                last[name] = evaluate(name)
                _sync(execution_device)
                samples[name].append(
                    _max_rank_ms((time.perf_counter() - started) * 1e3)
                )
        correctness_threshold = 5e-5 * max(1.0, size / 128)
        measurements = {}
        correctness_errors = []
        for name in names:
            max_abs_error = rms_error = 0.0
            if dist.get_rank() == 0:
                assert secret_key is not None
                actual = engine.decrypt_message(
                    last[name], secret_key, is_real=True
                ).cpu()
                expected = (matrix @ vector).repeat(engine.num_slots // size)
                error = actual - expected
                max_abs_error = float(error.abs().max())
                rms_error = float(torch.sqrt(torch.mean(error.square())))
            max_abs_error = _max_rank_ms(max_abs_error)
            rms_error = _max_rank_ms(rms_error)
            if (
                not math.isfinite(max_abs_error)
                or max_abs_error > correctness_threshold
            ):
                correctness_errors.append(
                    f"{name}: max_abs_error={max_abs_error} exceeds "
                    f"correctness_threshold={correctness_threshold}"
                )
            measurements[name] = {
                **_summarize(samples[name]),
                "preparation_ms": preparations[name],
                "cuda_graph": name in graph_programs,
                "max_abs_error": max_abs_error,
                "rms_error": rms_error,
                "correctness_threshold": correctness_threshold,
                "cuda_graph_capture_ms": (
                    graph_programs[name].stats.capture_seconds * 1e3
                    if name in graph_programs
                    else None
                ),
            }
        if len(names) == 2 and dist.get_rank() == 0:
            eager, jit = last["eager"], last["jit"]
            if (
                eager.depth,
                eager.scale,
                eager.prime_ids,
                eager.component_count,
            ) != (jit.depth, jit.scale, jit.prime_ids, jit.component_count):
                correctness_errors.append("Eager and JIT output states differ")
            moduli = torch.tensor(
                [engine.config.moduli[i] for i in eager.prime_ids],
                dtype=eager.data.dtype,
                device=eager.device,
            ).view(-1, 1)
            if not torch.equal(eager.data % moduli, jit.data % moduli):
                correctness_errors.append(
                    "Eager and JIT output residues differ modulo Q"
                )
        if compiled is not None:
            measurements["jit"]["implementations"] = sorted(
                {
                    dispatch.implementation.name
                    for dispatch in compiled.specializations[
                        0
                    ].executable.dispatch_table.operations.values()
                }
            )
        if dist.get_rank() != 0:
            return None
        config = engine.config
        return {
            "library": "FHElium",
            "version": fh.__version__,
            "mode": mode,
            "algorithm": "baby-step/giant-step cyclic-diagonal packed dense matrix-vector",
            "schedule": (
                "hoisted NTT-output babies, adjusted diagonals, NTT group "
                "completion and rescale, NTT-domain giant rotations"
                if retained_domain == "ntt"
                else "hoisted coefficient-output babies, adjusted diagonals, "
                "coefficient group completion and rescale, coefficient-domain "
                "giant rotations"
            ),
            "retained_domain": retained_domain,
            "cuda_graph": bool(graph_programs),
            "cuda_graph_scope": (
                "rank-local BSGS evaluation; ciphertext reduction remains eager"
                if graph_programs
                else None
            ),
            "matrix_size": size,
            "baby_step": baby_step,
            "giant_count": giant_count,
            "world_size": world_size,
            "device": device,
            "depth_label": depth,
            "q_depth_groups": config.q_depth_groups,
            "p_moduli": config.p_moduli,
            "ntt_backend": engine.ntt_backend_name(execution_device),
            "ring_dimension": config.N,
            "q_product_bits": math.prod(config.q_moduli).bit_length(),
            "qp_product_bits": math.prod(config.moduli).bit_length(),
            "multiplication_q_bits": math.prod(
                config.active_q_moduli(input_depth)
            ).bit_length(),
            "input_depth": input_depth,
            "output_depth": input_depth + 1,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "setup_ms": setup_ms,
            "warmup": warmup,
            "runs": runs,
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "devices": (
                [torch.cuda.get_device_name(i) for i in range(world_size)]
                if device == "cuda"
                else [platform.processor()]
            ),
            "executions": measurements,
            "correctness_errors": correctness_errors,
            "plaintext_storage": (
                {
                    "compression_layout": "contiguous",
                    "unique_count": 2 * size,
                    "compact_bytes_per_rank": sum(
                        value.nbytes for value in plaintext_groups.values()
                    ),
                    "dense_bytes_per_rank": sum(
                        value.nbytes
                        * value.ring_dimension
                        // value.unique_count
                        for value in plaintext_groups.values()
                    ),
                }
                if mode == "pt-ct"
                else None
            ),
        }
    finally:
        for graph in graph_programs.values():
            graph.close()
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
            "--input-depth",
            str(args.input_depth),
            "--retained-domain",
            args.retained_domain,
            "--execution",
            args.execution,
            "--cuda-graph" if args.cuda_graph else "--no-cuda-graph",
            "--output",
            str(worker_output),
        ]
        if args.threads is not None:
            command.extend(("--threads", str(args.threads)))
        environment = dict(os.environ)
        # Keep the invoking process's effective thread count in torchrun workers.
        environment.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
        process = subprocess.run(command, check=False, env=environment)
        recorded = worker_output.read_text()
        if recorded:
            args.output.write_text(recorded)
            print(recorded, end="")
        process.check_returncode()
    finally:
        worker_output.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--depth", type=int, choices=BENCHMARK_PARAMETERS, required=True
    )
    parser.add_argument("--mode", choices=("pt-ct", "ct-ct"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--baby-step", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--input-depth", type=int, default=0)
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override PyTorch intra-op threads; otherwise preserve environment settings.",
    )
    parser.add_argument(
        "--retained-domain",
        choices=("coefficient", "ntt"),
        default="ntt",
    )
    parser.add_argument(
        "--execution", choices=("eager", "jit", "both"), default="both"
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture JIT rank-local GPU evaluation; reduction remains outside capture.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
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
        input_depth=args.input_depth,
        retained_domain=args.retained_domain,
        execution=args.execution,
        cuda_graph=args.cuda_graph,
    )
    if result is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not args.worker:
            print(json.dumps(result, indent=2), flush=True)
        if result["correctness_errors"]:
            raise AssertionError(result["correctness_errors"])


if __name__ == "__main__":
    main()
