#!/usr/bin/env python3
"""Rotation-parallel packed matrix-vector multiplication.

Run on one process or one process per GPU:

    python examples/09_spmd_rotation_parallel_mxv.py

    torchrun --standalone --nproc-per-node=2 \
        examples/09_spmd_rotation_parallel_mxv.py --size 8

The packed input is replicated, while cyclic matrix diagonals and their
rotation keys are partitioned by rotation step. Each rank computes an
additive partial ciphertext. ``reduce_ciphertext`` combines those partials
with CKKS modular addition and leaves the final result only on rank 0.

Use this pattern for rotation/diagonal/head partitions whose rank-local
outputs are contributions to one logical encrypted result. The secret key
never leaves rank 0; only selected rotation keys are communicated.
"""

from __future__ import annotations

import argparse

import torch

import fhelium as fh
import fhelium.distributed as dist


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
    """Return weights aligned with ``torch.roll(x, shifts=rotation_step)``."""

    size = matrix.size(0)
    row = torch.arange(num_slots) % size
    column = torch.remainder(row - rotation_step, size)
    return matrix[row, column]


def _provision_owned_rotation_keys(
    engine: fh.CkksEngine,
    secret_key: fh.SecretKey | None,
    size: int,
) -> dict[int, fh.RotationKey]:
    """Create each exact key on rank 0 and retain it only on its owner."""

    local_keys: dict[int, fh.RotationKey] = {}
    for rotation_step in range(1, size):
        owner = rotation_step % dist.get_world_size()
        source_key = None
        if dist.get_rank() == 0:
            assert secret_key is not None
            source_key = engine.create_rotation_key(rotation_step, secret_key)

        if owner == 0:
            if dist.get_rank() == 0:
                assert source_key is not None
                local_keys[rotation_step] = source_key
            continue

        # All ranks participate in the typed key broadcast, but only the
        # designated owner retains the transferred object.
        transferred_key = dist.broadcast_key(source_key, src=0)
        if dist.get_rank() == owner:
            local_keys[rotation_step] = transferred_key
        else:
            del transferred_key
    return local_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()

    dist.init()
    engine = fh.CkksEngine(
        fh.Preset.slots32768_scale40_levels34_int64,
        device=dist.local_device(),
        allow_sk_gen=False,
    )
    if args.size <= 0 or args.size > engine.num_slots:
        raise ValueError(f"size must be in [1, {engine.num_slots}]")
    if engine.num_slots % args.size != 0:
        raise ValueError("size must divide the CKKS slot count")
    if dist.get_world_size() > args.size:
        raise ValueError(
            f"world_size={dist.get_world_size()} exceeds size={args.size}"
        )

    matrix, vector = _matrix_and_vector(args.size)
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

    # One logical encrypted input is intentionally replicated because every
    # rank evaluates a different subset of its rotations.
    source = dist.broadcast_ciphertext(root_source, src=0)
    local_rotation_steps = list(
        range(dist.get_rank(), args.size, dist.get_world_size())
    )
    local_keys = _provision_owned_rotation_keys(
        engine,
        secret_key,
        args.size,
    )

    local_terms = []
    for rotation_step in local_rotation_steps:
        rotated = (
            source.clone()
            if rotation_step == 0
            else engine.rotate_with_key(source, local_keys[rotation_step])
        )
        diagonal = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                _cyclic_diagonal_slots(matrix, rotation_step, engine.num_slots),
                level=rotated.level,
            )
        )
        local_terms.append(
            engine.rescale_to_next_level(
                engine.ntt_domain_to_coefficient_domain(
                    engine.multiply_plaintext(
                        engine.coefficient_domain_to_ntt_domain(rotated),
                        diagonal,
                    )
                )
            )
        )

    local_partial = engine.sum_ciphertexts(local_terms)

    # These are additive contributions to one result, so reduction is the
    # correct operation. Gathering would retain unnecessary per-rank objects.
    dist.reduce_ciphertext(local_partial, dst=0, engine=engine)
    print(
        f"rank={dist.get_rank()} rotation_steps={local_rotation_steps} "
        f"retained_rotation_keys={sorted(local_keys)}"
    )

    if dist.get_rank() == 0:
        assert secret_key is not None
        decoded = engine.decrypt_message(
            local_partial,
            secret_key=secret_key,
            is_real=True,
        )[: args.size]
        expected = matrix @ vector
        max_error = float(torch.max(torch.abs(decoded - expected)))
        torch.testing.assert_close(decoded, expected, atol=3e-5, rtol=0)
        print(
            "rotation_parallel_mxv_ok "
            f"size={args.size} world_size={dist.get_world_size()} "
            f"max_abs_error={max_error:.3e}"
        )

    dist.shutdown()


if __name__ == "__main__":
    main()
