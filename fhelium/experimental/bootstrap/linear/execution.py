r"""Apply compiled diagonal transforms with caller-owned keys and caches."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch

from fhelium.eager import Engine
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.values import Ciphertext, Plaintext, RotationKey, RotationKeySet
from fhelium.experimental.bootstrap.linear.transform import (
    DiagonalLinearTransform,
)
from fhelium.experimental.bootstrap.structural import ModRaisedCiphertext
from fhelium.utils.rotation import decompose_signed_power_of_two_rotation


def _rotate_with_key_inventory(
    engine: Engine,
    decomposition_cache: MutableMapping[int, tuple[int, ...]],
    ciphertext: Ciphertext,
    step: int,
    *,
    rotation_keys: RotationKeySet,
) -> Ciphertext:
    r"""Compute $\operatorname{Rot}_r(c)$ with direct or composed keys.

    The function first normalizes `step` as signed slot rotation $r$ and uses a direct key when the
    inventory contains one. Otherwise it caches a signed power-of-two
    decomposition and applies those keyed rotations in sequence. The cache
    contains only integer steps; it never retains ciphertexts or key tensors.

    The input must be a two-component coefficient-domain, standard-RNS
    ciphertext on the engine device. Rotation and its key switches preserve
    data shape, depth, actual scale, Q basis, `prime_ids`, component count,
    polynomial domain, and residue representation. The returned ciphertext has
    new storage; a zero step clones rather than aliases the input.
    """

    normalized = RotationKey.normalize_step(
        step, ring_dimension=engine.config.N
    )
    if normalized == 0:
        return ciphertext.clone()
    direct_key = rotation_keys.get(normalized)
    if direct_key is not None:
        return engine.rotate_with_key(ciphertext, direct_key)
    decomposition = decomposition_cache.get(normalized)
    if decomposition is None:
        decomposition = tuple(
            decompose_signed_power_of_two_rotation(normalized, engine.num_slots)
        )
        decomposition_cache[normalized] = decomposition
    result = ciphertext
    for substep in decomposition:
        key = rotation_keys.get(substep)
        if key is None:
            raise KeyError(
                f'Cannot compose rotation {normalized}; missing key {substep}'
            )
        result = engine.rotate_with_key(result, key)
    return result


def _encode_diagonal(
    engine: Engine,
    cache: MutableMapping[tuple[torch.device, int, int, int, int, float], Plaintext],
    *,
    device: torch.device,
    retain: bool,
    transform: DiagonalLinearTransform,
    offset: int,
    giant: int,
    depth: int,
    scale: float,
    diagonal: np.ndarray,
) -> Plaintext:
    r"""Encode $\operatorname{Rot}_{-g}(d_k)$ for plaintext multiplication.

    `diagonal` has CPU axes `[slot]`. After rolling by `-giant`, encoding and
    preparation produce an unbatched tensor with axes `[limb, ntt_index]`,
    engine integral dtype/device, NTT domain, Montgomery residues, Q basis,
    `prime_ids` for `depth`, and the supplied actual ``scale``. The cached plaintext
    may be returned by identity; callers must treat it as immutable. The input
    NumPy array is not mutated.
    """

    cache_key = (device, id(transform), offset, giant, depth, scale)
    plaintext = cache.get(cache_key)
    if plaintext is not None:
        return plaintext
    plaintext = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.as_tensor(
                np.roll(diagonal, -giant),
                dtype=torch.complex128,
                device=device,
            ),
            depth=depth,
            scale=scale,
            device=device,
        ),
        modulus_basis='Q',
    )
    if retain:
        cache[cache_key] = plaintext
    return plaintext


def _apply_linear_transform(
    arithmetic: BootstrapArithmetic,
    stages: tuple[Any, ...],
    evaluator,
    ciphertext: Ciphertext,
    *,
    rotation_keys: RotationKeySet,
    diagonal_cache: MutableMapping[tuple[torch.device, int, int, int, int, float], Plaintext],
    rotation_cache: MutableMapping[int, tuple[int, ...]],
    retain_diagonals: bool,
) -> Ciphertext:
    r"""Evaluate ordered maps $L_{m-1}\circ\cdots\circ L_0$.

    Each stage removes its Q group with product $M_d$. The supplied
    arithmetic owner chooses plaintext scale $M_d s_{d+1}/\Delta_{in}$,
    so the resulting actual scale is $s_{d+1}$.
    Coefficient-domain output keeps two components and the input batch axes;
    its prime rows are the suffix after the dropped group.
    """

    engine = arithmetic.engine
    if not stages or any(stage.slots != engine.num_slots for stage in stages):
        raise ValueError('linear transform has the wrong slot count')

    def rotate_with_keys(value: Ciphertext, step: int) -> Ciphertext:
        """Use this invocation's key inventory and decomposition cache."""

        return _rotate_with_key_inventory(
            engine,
            rotation_cache,
            value,
            step,
            rotation_keys=rotation_keys,
        )

    def encode_cached_diagonal(**kwargs) -> Plaintext:
        """Encode through this invocation's optional diagonal cache."""

        return _encode_diagonal(
            engine,
            diagonal_cache,
            device=ciphertext.device,
            retain=retain_diagonals,
            scale=arithmetic.plaintext_scale(result.scale, result.depth),
            **kwargs,
        )

    result = ciphertext
    for transform in stages:
        result = evaluator.evaluate(
            arithmetic,
            result,
            transform,
            rotation_keys=rotation_keys,
            rotate=rotate_with_keys,
            encode_diagonal=encode_cached_diagonal,
        )
    return result


def _apply_modraised_linear(
    arithmetic: BootstrapArithmetic,
    raised: ModRaisedCiphertext,
    stages: tuple[Any, ...],
    evaluator,
    *,
    rotation_keys: RotationKeySet,
    **cache_options,
) -> Ciphertext:
    """Apply the first linear transform after validating ModRaise provenance.

    The wrapped ciphertext is centered-raised Q RNS at the target public depth;
    its data axes and stage recurrence are those of :func:`_apply_linear_transform`.
    The private source-basis metadata authorizes this first transform but is
    deliberately discarded from the ordinary ciphertext result.
    """

    engine = arithmetic.engine
    if raised.ciphertext.depth >= raised.source_depth:
        raise ValueError('linear transform requires centered ModRaise history')
    return _apply_linear_transform(
        arithmetic,
        stages,
        evaluator,
        raised.ciphertext,
        rotation_keys=rotation_keys,
        **cache_options,
    )
