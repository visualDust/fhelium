r"""Auxiliary moduli, keys and execution of encapsulated modulus raising.

The auxiliary primes are the largest primes of the configured P directory.
The low relation uses the source-depth Q basis with one auxiliary prime
$P_{low}$. The return relation uses the next prime as guard $g$ and the
fewest following primes $P_{ret}$ with $\log_2(gP_{ret})\ge\log_2 q_b+\log_2 g$,
where $q_b$ is the source Q product.
"""

from __future__ import annotations

import math

import torch

from fhelium.values import Ciphertext, EncapsulationKey, EvaluationKeySet


def _auxiliary_moduli(config, source_depth: int) -> dict[str, object]:
    candidates = sorted(config.p_moduli, reverse=True)
    source = math.prod(config.active_q_moduli(source_depth))
    count, product = 2, 1
    while count < len(candidates) and product < source:
        product *= candidates[count]
        count += 1
    if product < source:
        raise ValueError(
            "the P directory cannot hold the encapsulation auxiliary primes"
        )
    return dict(
        low_p_moduli=(candidates[0],),
        return_prime=candidates[1],
        return_p_moduli=tuple(candidates[2:count]),
    )


def _add_encapsulation_keys(engine, plan, secret_key, inventory, *, device=None):
    """Sample the intermediate secret, add both encapsulation keys and drop the secret."""
    method = plan.modulus_raise
    intermediate = engine.create_secret_key(
        hamming_weight=method.intermediate_weight,
        device=secret_key.device if device is None else device,
    )
    keys = engine.create_encapsulation_keys(
        secret_key,
        intermediate,
        source_depth=plan.source_depth,
        **_auxiliary_moduli(engine.config, plan.source_depth),
    )
    del intermediate
    inventory.update(keys)


def _select_keys(engine, plan, keys, device) -> tuple[EncapsulationKey, EncapsulationKey]:
    if not isinstance(keys, EvaluationKeySet):
        raise TypeError("Encapsulated modulus raise requires an EvaluationKeySet")
    config = engine.config
    auxiliary = _auxiliary_moduli(config, plan.source_depth)
    rows = {
        "to_intermediate": len(config.active_q_moduli(plan.source_depth))
        + len(auxiliary["low_p_moduli"]),
        "to_source": config.num_q_primes + 1 + len(auxiliary["return_p_moduli"]),
    }
    selected = []
    for direction, count in rows.items():
        matches = [
            key
            for key in keys.encapsulation_keys
            if key.direction == direction
            and key.device == torch.device(device)
            and key.data.size(-2) == count
            and key.data.size(-1) == engine.ring_dimension
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {direction} encapsulation key for this plan on "
                f"{device}, found {len(matches)}"
            )
        selected.append(matches[0])
    return selected[0], selected[1]


def _encapsulated_raise(
    engine, method, value: Ciphertext, *, plan, target_depth: int, keys
) -> Ciphertext:
    if not isinstance(keys, tuple):
        keys = _select_keys(engine, plan, keys, value.device)
    low_key, return_key = keys
    return engine.encapsulated_modulus_raise(
        value,
        low_key,
        return_key,
        target_depth=target_depth,
        moddown_rounding=method.moddown_rounding,
        **_auxiliary_moduli(engine.config, value.depth),
    )
