#!/usr/bin/env python3

"""Serve weight-tile requests with a compiled layer and two device value buffers.

A DoubleBuffer copies the pinned-host plaintext tile of request i+1 on a transfer stream while the compiled layer evaluates request i from the other buffer. The layer captures one CUDA Graph and replays it for later requests; each replay copies its arguments from the current buffer into the Graph's input storage on the device.
"""

from __future__ import annotations

import argparse
import queue
import threading

import torch
from common import format_bytes, print_table

import fhelium as fh
from fhelium import compile as fc
from fhelium.eager import Engine
from fhelium.runtime import DoubleBuffer

# Preserve the established CKKS configuration and absolute error criterion.
_WORKLOAD_PRESET = fh.Preset.slots32768_scale40_depth34
_WORKLOAD_DEPTH = 20
_VALIDATION_ATOL = 1e-5
_PLAINTEXTS_PER_TILE = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--message-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("Double-buffered asynchronous transfer requires CUDA")
    if args.num_requests < 3:
        parser.error("Use at least three requests")
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

    @fc.compile(graph="auto")
    def layer(
        current: fh.Ciphertext,
        w0: fh.Plaintext,
        w1: fh.Plaintext,
        w2: fh.Plaintext,
        w3: fh.Plaintext,
    ) -> fh.Ciphertext:
        """Sum plaintext products in NTT form, then invert and rescale once."""
        accumulator = engine.multiply_plaintext(current, w0)
        for weight in (w1, w2, w3):
            accumulator = engine.add(
                accumulator, engine.multiply_plaintext(current, weight)
            )
        return engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(accumulator)
        )

    # Different tile values make missing or incorrectly ordered copies visible.
    factors = [
        0.125 * (i + 1) / args.num_requests for i in range(args.num_requests)
    ]
    host_tiles = []
    for factor in factors:
        prototype = engine.prepare_plaintext_for_multiplication(
            engine.encode(factor / _PLAINTEXTS_PER_TILE, depth=_WORKLOAD_DEPTH)
        ).cpu()
        host_tiles.append(
            tuple(prototype.pin_memory() for _ in range(_PLAINTEXTS_PER_TILE))
        )

    # A producer thread enqueues requests; None ends the stream.
    requests: queue.Queue = queue.Queue()

    def produce() -> None:
        for index, tile in enumerate(host_tiles):
            requests.put((index, tile))
        requests.put(None)

    producer = threading.Thread(target=produce)
    producer.start()
    rows = []
    with DoubleBuffer(host_tiles[0], device=device) as buffers:
        for (index, _), weights in buffers.feed(
            requests, value=lambda request: request[1]
        ):
            output = layer(source_ntt, *weights)
            # The result lives in the layer's output region until the next call.
            # Decrypting it now and dropping the reference lets the next call
            # reuse that storage without copying the result out.
            actual = engine.decrypt_message(output, secret_key, is_real=True)[
                : args.message_size
            ]
            del output
            expected = message * factors[index]
            torch.testing.assert_close(
                actual, expected, atol=_VALIDATION_ATOL, rtol=0
            )
            rows.append(
                [index, factors[index], float((actual - expected).abs().max())]
            )
        torch.cuda.current_stream(device).synchronize()
        buffer_bytes = sum(buffer.nbytes for buffer in buffers.buffers)
    producer.join()
    print_table(["request", "weight sum", "maximum error"], rows)
    print(
        f"Pinned weights: {format_bytes(sum(weight.nbytes for tile in host_tiles for weight in tile))}"
    )
    print(f"Two device buffers: {format_bytes(buffer_bytes)}")
    for record in layer.stats()["graphs"]:
        capture = record["capture_stats"] or {}
        print(
            "Graph:",
            record["state"],
            "replays:",
            record["replays"],
            "variants:",
            capture.get("variants"),
            "staged:",
            capture.get("staged_copies"),
        )


if __name__ == "__main__":
    main()
