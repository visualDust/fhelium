r"""Deplete and refresh one full-slot CKKS ciphertext at $\mathtt{logN}=16$.

The measured profile uses 50-bit scale and structural-base primes, generator 5,
and a periodic-reduction raw `input_bound` of 1024. The example message range is
an empirical end-to-end input, not a proof of the encrypted branch bound.
"""

from __future__ import annotations

import time

import torch

import fhelium as fh
from fhelium.values import EvaluationKeySet
from fhelium.eager import Engine
from fhelium.experimental.bootstrap.presets import (
    cosine_depth_refresh_logn16_v1,
)


def deplete_to_bootstrap_input(
    engine: Engine,
    ciphertext: fh.Ciphertext,
    target_depth: int,
) -> fh.Ciphertext:
    r"""Consume public Q depths with multiplication by semantic $1$.

    Each iteration multiplies a two-component coefficient-domain standard-RNS
    Q ciphertext by an unbatched NTT/Montgomery plaintext at scale $\Delta_0$,
    then drops the leading Q row. The output remains coefficient-domain
    standard RNS with axes `[component, *batch, limb, coefficient]`, unchanged
    batch shape and component count, next-depth `prime_ids`, and actual
    scale $\Delta_{\rm out}=\Delta_{\rm in}\Delta_0/q_{\rm drop}$. The
    returned ciphertext is functional; the argument's storage is not mutated.
    """

    ones = torch.ones(
        engine.num_slots,
        dtype=torch.float64,
        device=torch.get_default_device(),
    )
    while ciphertext.depth < target_depth:
        identity = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                ones, depth=ciphertext.depth, scale=engine.config.default_scale
            )
        )
        ciphertext = engine.rescale_to_next_depth(
            engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(ciphertext),
                    identity,
                )
            )
        )
    return ciphertext


def main() -> None:
    torch.set_default_device("cuda:0")
    config = fh.CkksConfig.parse(
        fh.Preset.slots32768_scale50_depth27_int64,
        galois_generator=5,
    )
    engine = Engine(
        config,
        allow_automatic_key_generation=False,
    )
    bootstrap = cosine_depth_refresh_logn16_v1(engine)

    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    rotation_keys = bootstrap.create_rotation_keys(secret_key)
    relinearization_key = engine.create_relinearization_key(secret_key)
    conjugation_key = engine.create_conjugation_key(secret_key)
    evaluation_keys = EvaluationKeySet(
        rotations=rotation_keys,
        relinearization=relinearization_key,
        conjugation=conjugation_key,
    )

    values = torch.linspace(
        -0.1,
        0.1,
        engine.num_slots,
        dtype=torch.float64,
    )
    depleted = deplete_to_bootstrap_input(
        engine,
        engine.encrypt_message(values, public_key),
        bootstrap.input_depth,
    )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(torch.get_default_device())
    started = time.perf_counter()
    refreshed = bootstrap(
        depleted,
        evaluation_keys=evaluation_keys,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    decoded = engine.decrypt_message(refreshed, secret_key, is_real=True).cpu()
    error = (decoded - values.cpu()).abs()
    print(f'input depth: {depleted.depth}')
    print(f'output depth: {refreshed.depth}')
    print(
        'pipeline depth: '
        f'{bootstrap.output_depth - bootstrap.modulus_raise_target_depth}'
    )
    print(
        f'periodic raw input bound: {bootstrap.modular_reduction.input_bound}'
    )
    print(f'output actual scale: {refreshed.scale:.6g}')
    print(f'rotation keys: {len(bootstrap.key_steps("power_of_two"))}')
    print(f'bootstrap seconds: {elapsed:.3f}')
    print(f'max error: {error.max().item():.6g}')
    print(f'mean error: {error.mean().item():.6g}')
    print(
        'peak allocated GPU GiB: '
        f'{torch.cuda.max_memory_allocated(torch.get_default_device()) / 2**30:.3f}'
    )


if __name__ == '__main__':
    main()
