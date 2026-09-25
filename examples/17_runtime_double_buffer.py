#!/usr/bin/env python3

"""Stream pinned-host plaintext tiles through two fixed CUDA value buffers.

The transfer stream prepares tile i+1 while the compute stream evaluates tile i.
A CopyHandle makes computation wait for the incoming data; a CUDA event makes
buffer reuse wait for the preceding reader. No CUDA Graph or admission manager
is needed for this application-owned streaming schedule.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from common import format_bytes, print_table

import fhelium as fh
from fhelium.eager import Engine
from fhelium.runtime import CopyHandle, ReusableValueBuffer

# Preserve the established CKKS configuration and absolute error criterion.
_WORKLOAD_PRESET = fh.Preset.slots32768_scale40_depth34_int64
_WORKLOAD_DEPTH = 20
_VALIDATION_ATOL = 1e-5


def evaluate_weight_tile(
    source_ntt: fh.Ciphertext,
    weights: Sequence[fh.Plaintext],
    *,
    engine: Engine,
) -> fh.Ciphertext:
    """Sum plaintext products in NTT form, then invert and rescale once."""
    accumulator = engine.multiply_plaintext(source_ntt, weights[0])
    for weight in weights[1:]:
        engine.add_(accumulator, engine.multiply_plaintext(source_ntt, weight))
    return engine.rescale_to_next_depth(
        engine.ntt_domain_to_coefficient_domain(accumulator)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-tiles", type=int, default=4)
    parser.add_argument("--plaintexts-per-tile", type=int, default=4)
    parser.add_argument("--message-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("Double-buffered asynchronous transfer requires CUDA")
    if args.num_tiles < 3 or args.plaintexts_per_tile < 1:
        parser.error("Use at least three tiles and one plaintext per tile")
    torch.set_default_device(device)
    engine = Engine(_WORKLOAD_PRESET, allow_automatic_key_generation=False)
    if not 1 <= args.message_size <= engine.num_slots:
        parser.error(f"message-size must be between 1 and {engine.num_slots}")
    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    message = torch.linspace(
        -0.01, 0.01, args.message_size, dtype=torch.float64
    )
    source = engine.encrypt_message(message, public_key, depth=_WORKLOAD_DEPTH)
    source_ntt = engine.coefficient_domain_to_ntt_domain(source)

    # Different tile values make missing or incorrectly ordered copies visible.
    factors = [0.125 * (i + 1) / args.num_tiles for i in range(args.num_tiles)]
    host_tiles = []
    for factor in factors:
        prototype = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                factor / args.plaintexts_per_tile, depth=_WORKLOAD_DEPTH
            )
        ).cpu()
        host_tiles.append(
            [prototype.pin_memory() for _ in range(args.plaintexts_per_tile)]
        )

    buffers = [
        ReusableValueBuffer.like(host_tiles[0], device=device) for _ in range(2)
    ]
    views = [buffer.value for buffer in buffers]
    pointers = [
        [weight.data.data_ptr() for weight in tile if weight.data is not None]
        for tile in views
    ]
    transfer_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device)
    transfer_stream.wait_stream(compute_stream)
    read_done: list[torch.cuda.Event | None] = [None, None]
    current_copy: CopyHandle | None = None
    outputs = []
    try:
        for index in range(args.num_tiles):
            current = index % 2
            next_copy = None
            if index + 1 < args.num_tiles:
                next_buffer = (index + 1) % 2
                next_copy = buffers[next_buffer].copy_from(
                    host_tiles[index + 1],
                    stream=transfer_stream,
                    non_blocking=True,
                    wait_for=read_done[next_buffer],
                )
            if current_copy is not None:
                current_copy.wait_on(compute_stream)
            outputs.append(
                evaluate_weight_tile(source_ntt, views[current], engine=engine)
            )
            event = torch.cuda.Event()
            event.record(compute_stream)
            read_done[current] = event
            current_copy = next_copy
        compute_stream.synchronize()
        assert pointers == [
            [
                weight.data.data_ptr()
                for weight in tile
                if weight.data is not None
            ]
            for tile in views
        ]
        rows = []
        for index, (output, factor) in enumerate(
            zip(outputs, factors, strict=True)
        ):
            actual = engine.decrypt_message(output, secret_key, is_real=True)[
                : args.message_size
            ]
            expected = message * factor
            torch.testing.assert_close(
                actual, expected, atol=_VALIDATION_ATOL, rtol=0
            )
            rows.append([index, factor, float((actual - expected).abs().max())])
        print_table(["tile", "weight sum", "maximum error"], rows)
        print(
            f"Pinned weights: {format_bytes(sum(weight.nbytes for tile in host_tiles for weight in tile))}"
        )
        print(
            f"Two fixed CUDA buffers: {format_bytes(sum(buffer.nbytes for buffer in buffers))}"
        )
    finally:
        transfer_stream.synchronize()
        compute_stream.synchronize()
        for buffer in buffers:
            buffer.close()


if __name__ == "__main__":
    main()
