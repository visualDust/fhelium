#!/usr/bin/env python3
"""Limb-parallel arithmetic on one logical ciphertext.

Run on one process or one process per GPU:

    python examples/10_spmd_limb_parallel_pipeline.py

    torchrun --standalone --nproc-per-node=2 \
        examples/10_spmd_limb_parallel_pipeline.py

The program evaluates ``(a + b)^2`` in two limb-local stages:

1. scatter level-0 limb fragments, add locally, and reconstruct ``a + b``;
2. rank 0 performs full-basis rescale/NTT preparation, scatters the remaining
   limbs, ranks run local ciphertext multiplication, and rank 0 reconstructs and relinearizes.

Use this pattern when one large ciphertext is structurally partitioned across
its RNS limb axis. The application chooses every limb range. Operations that
need the complete basis remain gather points, and no relinearization
key is sent to worker ranks.
"""

from __future__ import annotations

import torch

import fhelium as fh
import fhelium.distributed as dist


def _limb_ranges(limb_count: int) -> list[tuple[int, int]]:
    if dist.get_world_size() > limb_count:
        raise ValueError(
            f"world_size={dist.get_world_size()} exceeds limbs={limb_count}"
        )
    return [
        (
            rank * limb_count // dist.get_world_size(),
            (rank + 1) * limb_count // dist.get_world_size(),
        )
        for rank in range(dist.get_world_size())
    ]


def _split_limbs(
    ciphertext: fh.Ciphertext,
    ranges: list[tuple[int, int]],
) -> list[fh.Ciphertext]:
    return [ciphertext.slice_limbs(start, stop) for start, stop in ranges]


def main() -> None:
    dist.init()
    engine = fh.CkksEngine(
        fh.Preset.slots32768_scale40_levels34_int64,
        device=dist.local_device(),
        allow_sk_gen=False,
    )

    message_a = torch.linspace(-0.008, 0.011, 32, dtype=torch.float64)
    message_b = torch.linspace(0.006, -0.004, 32, dtype=torch.float64)
    addition_ranges = _limb_ranges(engine.config.num_q_primes)

    if dist.get_rank() == 0:
        secret_key = engine.create_secret_key()
        public_key = engine.create_public_key(secret_key)
        relinearization_key = engine.create_relinearization_key(secret_key)
        ciphertext_a = engine.encrypt_message(message_a, public_key)
        ciphertext_b = engine.encrypt_message(message_b, public_key)
        shards_a = _split_limbs(ciphertext_a, addition_ranges)
        shards_b = _split_limbs(ciphertext_b, addition_ranges)
    else:
        secret_key = None
        relinearization_key = None
        shards_a = None
        shards_b = None

    # Stage 1: add is independent for every modulus, so every rank can apply
    # the ordinary operation to its selected contiguous prime interval.
    local_a = dist.scatter_ciphertext_limbs(shards_a, src=0)
    local_b = dist.scatter_ciphertext_limbs(shards_b, src=0)
    local_sum = engine.add(local_a, local_b)
    full_sum = dist.gather_ciphertext_limbs(local_sum, dst=0)

    # Stage 2 preparation changes domain and uses the full active basis. It is
    # therefore performed only after reconstruction on rank 0.
    multiplication_ranges = _limb_ranges(engine.config.num_q_primes)
    if dist.get_rank() == 0:
        assert full_sum is not None
        prepared_sum = engine.coefficient_domain_to_ntt_domain(full_sum)
        prepared_shards = _split_limbs(
            prepared_sum,
            multiplication_ranges,
        )
    else:
        prepared_shards = None

    local_operand = dist.scatter_ciphertext_limbs(prepared_shards, src=0)

    # Exact-state ciphertext multiplication is also limb-local. It returns three components;
    # relinearization is deliberately delayed until the limbs are complete.
    local_triplet = engine.multiply(local_operand, local_operand)
    full_triplet = dist.gather_ciphertext_limbs(local_triplet, dst=0)

    add_start, add_stop = addition_ranges[dist.get_rank()]
    mul_start, mul_stop = multiplication_ranges[dist.get_rank()]
    print(
        f"rank={dist.get_rank()} add_limbs=[{add_start},{add_stop}) "
        f"multiply_limbs=[{mul_start},{mul_stop})"
    )

    if dist.get_rank() == 0:
        assert secret_key is not None
        assert relinearization_key is not None
        assert full_triplet is not None
        result = engine.rescale_to_next_level(
            engine.relinearize(full_triplet, relinearization_key)
        )
        decoded = engine.decrypt_message(
            result,
            secret_key=secret_key,
            is_real=True,
        )[: message_a.numel()]
        expected = (message_a + message_b).square()
        max_error = float(torch.max(torch.abs(decoded - expected)))
        torch.testing.assert_close(decoded, expected, atol=3e-6, rtol=0)
        print(
            "limb_parallel_pipeline_ok "
            f"world_size={dist.get_world_size()} "
            f"max_abs_error={max_error:.3e}"
        )

    dist.shutdown()


if __name__ == "__main__":
    main()
