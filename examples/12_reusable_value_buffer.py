#!/usr/bin/env python3

"""Measure all-resident versus double-buffered CKKS plaintext weights.

This example intentionally uses an ordinary eager evaluator, not CUDA Graphs.
It compares two ways to execute the same sequence of plaintext-weight tiles:

``all-resident``
    Copy every operation-ready Plaintext to CUDA before evaluation and retain
    all tiles until the workload completes.

``double-buffer``
    Retain every tile in pinned CPU memory, own only two fixed-address CUDA
    :class:`fhelium.execution.ReusableValueBuffer` objects, and copy tile
    ``i+1`` while the current CUDA stream evaluates tile ``i``.

The application chooses the number of tiles and Plaintexts per tile. Execution
utilities only validate exact value structure, reuse storage, enqueue copies,
and expose future-like :class:`fhelium.execution.CopyHandle` objects.

The default `slots32768-scale40-levels34-int64` workload at level 20 materializes 16
tiles with 64 Plaintexts per tile. One multiply-ready Plaintext is 7.5 MiB, so
all-resident weight
storage is 7.5 GiB while two reusable buffers own 0.9375 GiB. Peak measurements
also include the shared ciphertext and eager evaluator temporaries.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from common import parse_preset, preset_names

import fhelium as fh
from fhelium.execution import CopyHandle, ReusableValueBuffer


@dataclass(frozen=True)
class ModeResult:
    """Timing, allocator peak, and correctness for one residency strategy."""

    name: str
    setup_seconds: float
    evaluation_seconds: float
    baseline_allocated_bytes: int
    resident_after_setup_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    max_error: float

    @property
    def total_seconds(self) -> float:
        return self.setup_seconds + self.evaluation_seconds

    @property
    def peak_allocated_increment(self) -> int:
        return self.peak_allocated_bytes - self.baseline_allocated_bytes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=preset_names(),
        default=fh.Preset.slots32768_scale40_levels34_int64.value,
    )
    parser.add_argument("--level", type=int, default=20)
    parser.add_argument("--num-tiles", type=int, default=16)
    parser.add_argument("--plaintexts-per-tile", type=int, default=64)
    parser.add_argument("--message-size", type=int, default=256)
    parser.add_argument("--weight-sum", type=float, default=0.125)
    parser.add_argument(
        "--skip-all-resident",
        action="store_true",
        help="Run only double buffering when the all-resident case will not fit.",
    )
    return parser.parse_args()


def _gib(byte_count: int) -> float:
    return byte_count / 2**30


def _pinned_plaintext_copy(prototype: fh.Plaintext) -> fh.Plaintext:
    """Copy one exact Plaintext into an independent pinned CPU allocation."""

    if prototype.data is None or not prototype.is_cpu:
        raise ValueError("prototype must be an encoded CPU Plaintext")
    data = torch.empty_like(
        prototype.data,
        device="cpu",
        pin_memory=True,
    )
    data.copy_(prototype.data)
    return fh.Plaintext(
        message=None,
        level=prototype.level,
        scale=prototype.scale,
        data=data,
        context_id=prototype.context_id,
        representation=prototype.representation,
        polynomial_domain=prototype.polynomial_domain,
        modulus_basis=prototype.modulus_basis,
        residue_representation=prototype.residue_representation,
        prime_ids=prototype.prime_ids,
    )


def _prepare_pinned_tiles(
    prototype: fh.Plaintext,
    *,
    num_tiles: int,
    plaintexts_per_tile: int,
) -> list[list[fh.Plaintext]]:
    """Materialize application-selected tiles in pinned host memory."""

    return [
        [_pinned_plaintext_copy(prototype) for _ in range(plaintexts_per_tile)]
        for _ in range(num_tiles)
    ]


def evaluate_weight_tile(
    source: fh.Ciphertext,
    weights: Sequence[fh.Plaintext],
    *,
    engine: fh.CkksEngine,
) -> fh.Ciphertext:
    """Eagerly multiply one ciphertext by a tile and stream the sum.

    Only one product plus the accumulator is live at a time. All-resident and
    double-buffer modes call this same function for every tile, so their
    allocator difference comes from weight residency rather than different
    arithmetic schedules.
    """

    if not weights:
        raise ValueError("a weight tile must contain at least one Plaintext")
    accumulator = None
    for weight in weights:
        product = engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(source), weight
        )
        if accumulator is None:
            accumulator = product
        else:
            engine.add_(accumulator, product)
    assert accumulator is not None
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(accumulator)
    )


def _measure_all_resident(
    *,
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    host_tiles: Sequence[Sequence[fh.Plaintext]],
    expected: torch.Tensor,
    secret_key: fh.SecretKey,
) -> ModeResult:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(engine.device)
    baseline = torch.cuda.memory_allocated(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)

    setup_start = time.perf_counter()
    cuda_tiles = [
        [
            weight.to(
                engine.device,
                non_blocking=True,
                copy=True,
            )
            for weight in tile
        ]
        for tile in host_tiles
    ]
    torch.cuda.synchronize(engine.device)
    setup_seconds = time.perf_counter() - setup_start
    resident_after_setup = torch.cuda.memory_allocated(engine.device)

    evaluation_start = time.perf_counter()
    output = None
    for tile in cuda_tiles:
        output = evaluate_weight_tile(source, tile, engine=engine)
    torch.cuda.synchronize(engine.device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    peak_allocated = torch.cuda.max_memory_allocated(engine.device)
    peak_reserved = torch.cuda.max_memory_reserved(engine.device)

    assert output is not None
    actual = engine.decrypt_message(
        output,
        secret_key,
        is_real=True,
    )[: expected.numel()]
    max_error = float((actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=5e-5)

    del output, cuda_tiles
    torch.cuda.synchronize(engine.device)
    torch.cuda.empty_cache()
    return ModeResult(
        name="all-resident",
        setup_seconds=setup_seconds,
        evaluation_seconds=evaluation_seconds,
        baseline_allocated_bytes=baseline,
        resident_after_setup_bytes=resident_after_setup,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        max_error=max_error,
    )


def _measure_double_buffer(
    *,
    engine: fh.CkksEngine,
    source: fh.Ciphertext,
    host_tiles: Sequence[Sequence[fh.Plaintext]],
    expected: torch.Tensor,
    secret_key: fh.SecretKey,
) -> ModeResult:
    torch.cuda.empty_cache()
    torch.cuda.synchronize(engine.device)
    baseline = torch.cuda.memory_allocated(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)

    setup_start = time.perf_counter()
    buffers = [
        ReusableValueBuffer.like(
            host_tiles[0],
            device=engine.device,
        )
        for _ in range(2)
    ]
    torch.cuda.synchronize(engine.device)
    setup_seconds = time.perf_counter() - setup_start
    resident_after_setup = torch.cuda.memory_allocated(engine.device)
    initial_pointers = [
        tuple(
            weight.data.data_ptr()
            for weight in buffer.value
            if weight.data is not None
        )
        for buffer in buffers
    ]

    transfer_stream = torch.cuda.Stream(device=engine.device)
    compute_stream = torch.cuda.current_stream(engine.device)
    buffer_read_done_events: list[torch.cuda.Event | None] = [None, None]
    current_copy_handle: CopyHandle | None = None
    output = None

    evaluation_start = time.perf_counter()
    for tile_index in range(len(host_tiles)):
        buffer_index = tile_index % 2
        if tile_index > 0:
            assert current_copy_handle is not None
        next_copy_handle = None
        if tile_index + 1 < len(host_tiles):
            next_buffer_index = (tile_index + 1) % 2
            next_copy_handle = buffers[next_buffer_index].copy_from(
                host_tiles[tile_index + 1],
                stream=transfer_stream,
                non_blocking=True,
                wait_for=buffer_read_done_events[next_buffer_index],
            )

        with torch.cuda.stream(compute_stream):
            if current_copy_handle is not None:
                current_copy_handle.wait_on(compute_stream)
            output = evaluate_weight_tile(
                source,
                buffers[buffer_index].value,
                engine=engine,
            )
            buffer_read_done = torch.cuda.Event()
            buffer_read_done.record(compute_stream)
            buffer_read_done_events[buffer_index] = buffer_read_done
        current_copy_handle = next_copy_handle

    compute_stream.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_start
    peak_allocated = torch.cuda.max_memory_allocated(engine.device)
    peak_reserved = torch.cuda.max_memory_reserved(engine.device)
    final_pointers = [
        tuple(
            weight.data.data_ptr()
            for weight in buffer.value
            if weight.data is not None
        )
        for buffer in buffers
    ]
    if final_pointers != initial_pointers:
        raise RuntimeError(
            "ReusableValueBuffer changed a target tensor address"
        )

    assert output is not None
    actual = engine.decrypt_message(
        output,
        secret_key,
        is_real=True,
    )[: expected.numel()]
    max_error = float((actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=5e-5)

    del output
    for buffer in buffers:
        buffer.close()
    torch.cuda.synchronize(engine.device)
    torch.cuda.empty_cache()
    return ModeResult(
        name="double-buffer",
        setup_seconds=setup_seconds,
        evaluation_seconds=evaluation_seconds,
        baseline_allocated_bytes=baseline,
        resident_after_setup_bytes=resident_after_setup,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        max_error=max_error,
    )


def _print_results(
    results: Sequence[ModeResult],
    *,
    tile_bytes: int,
    host_weight_bytes: int,
) -> None:
    print(
        f"pinned host weights: {_gib(host_weight_bytes):.3f} GiB; "
        f"one CUDA tile: {_gib(tile_bytes):.3f} GiB"
    )
    print()
    print(
        "mode           setup s  eval s  total s  resident-after-setup "
        "peak-allocated-increment  peak-reserved  max error"
    )
    print(
        "-------------  -------  ------  -------  -------------------- "
        "------------------------  -------------  ---------"
    )
    for result in results:
        print(
            f"{result.name:13s}  "
            f"{result.setup_seconds:7.3f}  "
            f"{result.evaluation_seconds:6.3f}  "
            f"{result.total_seconds:7.3f}  "
            f"{_gib(result.resident_after_setup_bytes - result.baseline_allocated_bytes):20.3f}  "
            f"{_gib(result.peak_allocated_increment):24.3f}  "
            f"{_gib(result.peak_reserved_bytes):13.3f}  "
            f"{result.max_error:.3e}"
        )
    if len(results) == 2:
        all_resident, double_buffer = results
        saved = (
            all_resident.peak_allocated_increment
            - double_buffer.peak_allocated_increment
        )
        fraction = saved / all_resident.peak_allocated_increment
        print()
        print(
            "double-buffer peak allocated saving: "
            f"{_gib(saved):.3f} GiB ({fraction:.1%})"
        )


def run(args: argparse.Namespace) -> None:
    if args.num_tiles < 3:
        raise ValueError("--num-tiles must be at least 3 for double buffering")
    if args.plaintexts_per_tile < 1:
        raise ValueError("--plaintexts-per-tile must be positive")
    if args.message_size < 1:
        raise ValueError("--message-size must be positive")

    preset = parse_preset(args.preset)
    engine = fh.CkksEngine(
        preset,
        device="cuda:0",
        allow_sk_gen=False,
    )
    if not 0 <= args.level < engine.final_public_level:
        raise ValueError(
            "--level must leave one rescale available: "
            f"level={args.level}, "
            f"final_public_level={engine.final_public_level}"
        )
    if args.message_size > engine.num_slots:
        raise ValueError(
            f"--message-size exceeds {engine.num_slots} CKKS slots"
        )

    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    message = torch.linspace(
        -0.01,
        0.01,
        args.message_size,
        dtype=torch.float64,
    )
    source = engine.encrypt_message(
        message,
        public_key,
        level=args.level,
    )
    scalar = args.weight_sum / args.plaintexts_per_tile
    prototype_weight = engine.prepare_plaintext_for_multiplication(
        engine.encode(scalar, level=args.level)
    ).cpu()

    host_prepare_start = time.perf_counter()
    host_tiles = _prepare_pinned_tiles(
        prototype_weight,
        num_tiles=args.num_tiles,
        plaintexts_per_tile=args.plaintexts_per_tile,
    )
    host_prepare_seconds = time.perf_counter() - host_prepare_start
    tile_bytes = sum(weight.nbytes for weight in host_tiles[0])
    host_weight_bytes = tile_bytes * len(host_tiles)
    expected = message * args.weight_sum

    print(
        f"preset={args.preset} level={args.level} "
        f"tiles={args.num_tiles} "
        f"plaintexts_per_tile={args.plaintexts_per_tile}"
    )
    print(
        f"one Plaintext={_gib(prototype_weight.nbytes):.6f} GiB; "
        f"host preparation={host_prepare_seconds:.3f} s"
    )

    # Warm kernels and allocator caches before either measured residency mode.
    warm_tile = [
        weight.to(engine.device, non_blocking=True, copy=True)
        for weight in host_tiles[0]
    ]
    warm_output = evaluate_weight_tile(source, warm_tile, engine=engine)
    torch.cuda.synchronize(engine.device)
    del warm_output, warm_tile
    torch.cuda.empty_cache()

    results = []
    if not args.skip_all_resident:
        results.append(
            _measure_all_resident(
                engine=engine,
                source=source,
                host_tiles=host_tiles,
                expected=expected,
                secret_key=secret_key,
            )
        )
    results.append(
        _measure_double_buffer(
            engine=engine,
            source=source,
            host_tiles=host_tiles,
            expected=expected,
            secret_key=secret_key,
        )
    )
    _print_results(
        results,
        tile_bytes=tile_bytes,
        host_weight_bytes=host_weight_bytes,
    )


if __name__ == "__main__":
    run(_parse_args())
