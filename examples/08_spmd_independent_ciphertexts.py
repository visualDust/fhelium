#!/usr/bin/env python3

"""Data parallelism over independent encrypted inputs.

Run on one process or one process per GPU:

    python examples/08_spmd_independent_ciphertexts.py

    torchrun --standalone --nproc-per-node=2 \
        examples/08_spmd_independent_ciphertexts.py

Rank 0 encrypts one independent input per rank and scatters those
ciphertexts. Every rank applies the same public affine transform, then rank 0
gathers the independent outputs for decryption. The model weight is a public
Plaintext replicated with ``broadcast_plaintext``; no key leaves rank 0.

Use this pattern when ranks process different samples or requests. The outputs
must be gathered, not reduced, because they are distinct logical values.
"""

from __future__ import annotations

import torch

import fhelium as fh
import fhelium.distributed as dist


def _message_for_rank(rank: int) -> torch.Tensor:
    base = torch.linspace(-0.015, 0.015, 32, dtype=torch.float64)
    return base + 0.004 * rank


def main() -> None:
    dist.init()
    engine = fh.CkksEngine(
        fh.Preset.slots32768_scale40_levels34_int64,
        device=dist.local_device(),
        allow_sk_gen=False,
    )

    # Only the data owner needs encryption/decryption keys. The worker ranks
    # evaluate plaintext-ciphertext operations without receiving any key.
    if dist.get_rank() == 0:
        secret_key = engine.create_secret_key()
        public_key = engine.create_public_key(secret_key)
        messages = [
            _message_for_rank(rank) for rank in range(dist.get_world_size())
        ]
        encrypted_inputs = [
            engine.encrypt_message(message, public_key) for message in messages
        ]
        root_weight = engine.prepare_plaintext_for_multiplication(
            engine.encode(torch.full((32,), 1.25, dtype=torch.float64), level=0)
        )
    else:
        secret_key = None
        messages = None
        encrypted_inputs = None
        root_weight = None

    # Independent objects use scatter/gather. The collective allocates the
    # receiving Ciphertext from transmitted metadata; it does not infer which
    # application sample belongs to which rank.
    local_input = dist.scatter_ciphertexts(encrypted_inputs, src=0)

    # This public model parameter is one logical Plaintext replicated to every
    # rank. It is distinct from scattering independent Ciphertexts above.
    weight = dist.broadcast_plaintext(root_weight, src=0)

    # multiply_plaintext deliberately does not rescale. The level transition is separate,
    # and each rank creates its public rank-specific bias locally.
    local_output = engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(local_input), weight
            )
        )
    )
    bias_message = torch.full(
        (32,),
        -0.003 + 0.001 * dist.get_rank(),
        dtype=torch.float64,
    )
    bias = engine.prepare_plaintext_for_addition(
        engine.encode(
            bias_message,
            level=local_output.level,
            scale=local_output.scale,
        )
    )
    local_output = engine.add_plaintext(local_output, bias)

    outputs = dist.gather_ciphertexts(local_output, dst=0)
    print(
        f"rank={dist.get_rank()} sample={dist.get_rank()} "
        f"level={local_output.level}"
    )

    if dist.get_rank() == 0:
        assert secret_key is not None
        assert messages is not None
        assert outputs is not None
        errors = []
        for rank, (message, output) in enumerate(zip(messages, outputs)):
            decoded = engine.decrypt_message(
                output,
                secret_key=secret_key,
                is_real=True,
            )[: message.numel()]
            expected = 1.25 * message + (-0.003 + 0.001 * rank)
            torch.testing.assert_close(decoded, expected, atol=3e-5, rtol=0)
            errors.append(float(torch.max(torch.abs(decoded - expected))))
        print(
            "independent_ciphertexts_ok "
            f"world_size={dist.get_world_size()} max_abs_errors={errors}"
        )

    dist.shutdown()


if __name__ == "__main__":
    main()
