#!/usr/bin/env python3

"""Distribute an encrypted input batch across data-parallel ranks.

Run on one process or one process per GPU:

    python examples/21_distributed_batch_inputs.py --batch-size 7

    torchrun --standalone --nproc-per-node=2 \
        examples/21_distributed_batch_inputs.py --batch-size 7

Rank 0 encrypts one batch, chooses consecutive sample intervals, and scatters
Ciphertext.slice_batch views. Each rank applies the same public affine model
to its entire local batch. Rank 0 gathers and decrypts the sub-batches, then
concatenates their clear results in original sample order. Weights are public;
encryption and decryption keys remain on rank 0.

The partition may be uneven. Neither the model nor sample generation depends
on rank, so changing the partition does not change the intended result.
"""

from __future__ import annotations

import argparse

import torch
from common import parse_preset

import fhelium as fh
import fhelium.distributed as dist
from fhelium.eager import Engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--preset",
        type=parse_preset,
        default=fh.Preset.slots8192_scale40_depth7_int64,
    )
    args = parser.parse_args()
    dist.init()
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if args.batch_size < world_size:
            raise ValueError(
                "batch-size must be at least world size so every rank receives "
                "a nonempty sub-batch"
            )
        torch.set_default_device(dist.local_device())
        engine = Engine(args.preset, allow_automatic_key_generation=False)
        sample_width = 32
        weight_value, bias_value = 1.25, -0.003
        batch_ranges = [
            (
                owner * args.batch_size // world_size,
                (owner + 1) * args.batch_size // world_size,
            )
            for owner in range(world_size)
        ]

        if rank == 0:
            secret_key = engine.create_secret_key()
            public_key = engine.create_public_key(secret_key)
            samples = torch.arange(args.batch_size, dtype=torch.float64)
            messages = torch.linspace(
                -0.015, 0.015, sample_width, dtype=torch.float64
            ).unsqueeze(0) + 0.004 * torch.sin(samples * 0.7).unsqueeze(1)
            encrypted_batch = engine.encrypt_message(messages, public_key)
            chunks = [
                encrypted_batch.slice_batch(start, stop)
                for start, stop in batch_ranges
            ]
            root_weight = engine.prepare_plaintext_for_multiplication(
                engine.encode(
                    torch.full(
                        (sample_width,), weight_value, dtype=torch.float64
                    ),
                    depth=0,
                )
            )
        else:
            secret_key = None
            messages = None
            chunks = None
            root_weight = None

        # Each list entry is a sub-batch, not a single sample. The collective
        # preserves its shape and state; the application chose its sample range.
        local_input = dist.scatter_ciphertexts(chunks, src=0)
        weight = dist.broadcast_plaintext(root_weight, src=0)

        # Engine operations evaluate every item in the local batch. This model
        # is independent of both rank number and the chosen batch partition.
        local_output = engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(local_input), weight
                )
            )
        )
        bias = engine.prepare_plaintext_for_addition(
            engine.encode(
                torch.full((sample_width,), bias_value, dtype=torch.float64),
                depth=local_output.depth,
                scale=local_output.scale,
            )
        )
        local_output = engine.add_plaintext(local_output, bias)
        outputs = dist.gather_ciphertexts(local_output, dst=0)
        start, stop = batch_ranges[rank]
        print(
            f"rank={rank} samples=[{start},{stop}) "
            f"local_batch={local_input.batch_shape[0]} depth={local_output.depth}"
        )

        if rank == 0:
            assert secret_key is not None
            assert messages is not None
            assert outputs is not None
            decoded = torch.cat(
                [
                    engine.decrypt_message(
                        output, secret_key=secret_key, is_real=True
                    ).cpu()[..., :sample_width]
                    for output in outputs
                ],
                dim=0,
            )
            expected = (weight_value * messages + bias_value).cpu()
            torch.testing.assert_close(decoded, expected, atol=3e-5, rtol=0)
            max_error = float(torch.max(torch.abs(decoded - expected)))
            print(
                f"data_parallel_batch_ok batch_size={args.batch_size} "
                f"world_size={world_size} max_abs_error={max_error:.3e}"
            )
    finally:
        dist.shutdown()


if __name__ == "__main__":
    main()
