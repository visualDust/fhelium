r"""Private CKKS arithmetic compositions shared by bootstrap components.

Unless stated otherwise, ciphertext payloads have axes
`[component, *batch, limb, coefficient]` in coefficient-domain standard RNS.
Each helper is functional and returns two components over the Q basis. The
bootstrap uses the engine's exact active `prime_ids`; the limb axis is never
identified by level alone.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch

from fhelium.core import (
    Ciphertext,
    Plaintext,
    RelinearizationKey,
    RotationKey,
    RotationKeySet,
)
from fhelium.engine.ckks_engine import CkksEngine
from fhelium.experimental.bootstrap._linear import (
    DiagonalLinearTransform,
)
from fhelium.experimental.bootstrap._modraise import ModRaisedCiphertext
from fhelium.core.rotation import (
    decompose_signed_power_of_two_rotation,
)


def _rotate_with_key_inventory(
    engine: CkksEngine,
    decomposition_cache: MutableMapping[int, tuple[int, ...]],
    ciphertext: Ciphertext,
    step: int,
    *,
    rotation_keys: RotationKeySet,
) -> Ciphertext:
    r"""Compute $\operatorname{Rot}_r(c)$ with exact or composed keys.

    The function first canonicalizes `step` as signed slot rotation $r$ and uses an exact key when the
    inventory contains one. Otherwise it caches a signed power-of-two
    decomposition and applies those keyed rotations in sequence. The cache
    contains only integer steps; it never retains ciphertexts or key tensors.

    The input must be a two-component coefficient-domain, standard-RNS
    ciphertext on the engine device. Rotation and its key switches preserve
    data shape, level, actual scale, Q basis, exact `prime_ids`, component count,
    polynomial domain, and residue representation. The returned ciphertext has
    new storage; a zero step clones rather than aliases the input.
    """

    canonical = RotationKey.canonical_step(step, ring_dimension=engine.config.N)
    if canonical == 0:
        return ciphertext.clone()
    exact = rotation_keys.get(canonical)
    if exact is not None:
        return engine.rotate_with_key(ciphertext, exact)
    decomposition = decomposition_cache.get(canonical)
    if decomposition is None:
        decomposition = tuple(
            decompose_signed_power_of_two_rotation(canonical, engine.num_slots)
        )
        decomposition_cache[canonical] = decomposition
    result = ciphertext
    for substep in decomposition:
        key = rotation_keys.get(substep)
        if key is None:
            raise KeyError(
                f'Cannot compose rotation {canonical}; missing key {substep}'
            )
        result = engine.rotate_with_key(result, key)
    return result


def _encode_diagonal(
    engine: CkksEngine,
    cache: MutableMapping[tuple[int, int, int, int], Plaintext],
    *,
    retain: bool,
    transform: DiagonalLinearTransform,
    offset: int,
    giant: int,
    level: int,
    diagonal: np.ndarray,
) -> Plaintext:
    r"""Encode $\operatorname{Rot}_{-g}(d_k)$ for plaintext multiplication.

    `diagonal` has CPU axes `[slot]`. After rolling by `-giant`, encoding and
    preparation produce an unbatched tensor with axes `[limb, ntt_index]`,
    engine integral dtype/device, NTT domain, Montgomery residues, Q basis,
    `prime_ids` for `level`, and actual scale $\Delta_0$. The cached plaintext
    may be returned by identity; callers must treat it as immutable. The input
    NumPy array is not mutated.
    """

    cache_key = (id(transform), offset, giant, level)
    plaintext = cache.get(cache_key)
    if plaintext is not None:
        return plaintext
    plaintext = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.as_tensor(
                np.roll(diagonal, -giant),
                dtype=torch.complex128,
                device=engine.device,
            ),
            level=level,
            scale=engine.config.default_scale,
        ),
        modulus_basis='Q',
    )
    if retain:
        cache[cache_key] = plaintext
    return plaintext


def _apply_linear_transform(
    engine: CkksEngine,
    stages: tuple[Any, ...],
    evaluator,
    ciphertext: Ciphertext,
    *,
    rotation_keys: RotationKeySet,
    diagonal_cache: MutableMapping[tuple[int, int, int, int], Plaintext],
    rotation_cache: MutableMapping[int, tuple[int, ...]],
    retain_diagonals: bool,
) -> Ciphertext:
    r"""Evaluate ordered maps $L_{m-1}\circ\cdots\circ L_0$.

    Each evaluator stage consumes one level. If $q_j$ is the leading Q prime
    at stage $j$, the exact metadata recurrence is

    $$
    \ell_{j+1}=\ell_j+1,\qquad
    \Delta_{j+1}=\frac{\Delta_j\Delta_0}{q_j}.
    $$

    The two-component coefficient-domain standard-RNS state and batch axes are
    preserved while the limb axis loses its leading row at every stage. Input
    storage is unchanged. Diagonal and rotation caches contain no output
    ciphertext.
    """

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
            retain=retain_diagonals,
            **kwargs,
        )

    result = ciphertext
    for transform in stages:
        result = evaluator.evaluate(
            engine,
            result,
            transform,
            rotation_keys=rotation_keys,
            rotate=rotate_with_keys,
            encode_diagonal=encode_cached_diagonal,
        )
    return result


def _apply_modraised_linear(
    engine: CkksEngine,
    raised: ModRaisedCiphertext,
    stages: tuple[Any, ...],
    evaluator,
    *,
    rotation_keys: RotationKeySet,
    **cache_options,
) -> Ciphertext:
    """Apply the first linear transform after validating ModRaise provenance.

    The wrapped ciphertext is centered-raised Q RNS at the target public level;
    its data axes and stage recurrence are those of :func:`_apply_linear_transform`.
    The private source-basis metadata authorizes this first transform but is
    deliberately discarded from the ordinary ciphertext result.
    """

    if raised.ciphertext.level >= raised.source_level:
        raise ValueError('linear transform requires centered ModRaise history')
    return _apply_linear_transform(
        engine,
        stages,
        evaluator,
        raised.ciphertext,
        rotation_keys=rotation_keys,
        **cache_options,
    )


def _multiply_by_monomial(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    exponent: int,
) -> Ciphertext:
    r"""Multiply every component by $X^e$ in $R=\mathbb{Z}[X]/(X^N+1)$.

    For each `[component, *batch, limb]` polynomial and $e=$ `exponent`, the
    last `[coefficient]` axis is negacyclically shifted so that

    $$
    c_{\rm out}(X)=X^e c_{\rm in}(X)\pmod{X^N+1}.
    $$

    The input must use coefficient-domain standard residues on the engine
    device. The operation is functional and preserves shape, component count,
    level, actual scale, basis, exact `prime_ids`, polynomial domain, and
    residue representation. The output is canonicalized into each limb's
    interval $[0,q_i)$ and does not alias the input.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient', residue_representation="standard"
    )
    engine._assert_engine_ciphertext(ciphertext)
    degree = engine.config.N
    exponent %= 2 * degree
    sign = -1 if exponent >= degree else 1
    shift = exponent % degree
    data = ciphertext.data.clone()
    if shift:
        original = data.clone()
        data[..., shift:] = original[..., : degree - shift]
        data[..., :shift] = -original[..., degree - shift :]
    if sign < 0:
        data.neg_()
    moduli = torch.tensor(
        [
            engine.montgomery_parameters.moduli[index]
            for index in ciphertext.prime_ids
        ],
        dtype=data.dtype,
        device=data.device,
    ).view(*([1] * (data.ndim - 2)), -1, 1)
    return ciphertext.with_data(torch.remainder(data, moduli))


def _align_levels(
    engine: CkksEngine,
    lhs: Ciphertext,
    rhs: Ciphertext,
) -> tuple[Ciphertext, Ciphertext]:
    r"""Advance the shallower operand until $\ell_{\rm lhs}=\ell_{\rm rhs}$.

    Each advancement consumes one leading Q row under `_advance_level` and
    returns actual scale $\Delta_0$. An operand already at the deeper level is
    returned by identity, so this helper does not by itself guarantee equal
    scales; callers performing addition still rely on exact engine validation.
    """

    while lhs.level < rhs.level:
        lhs = _advance_level(engine, lhs)
    while rhs.level < lhs.level:
        rhs = _advance_level(engine, rhs)
    return lhs, rhs


def _advance_level(
    engine: CkksEngine,
    ciphertext: Ciphertext,
) -> Ciphertext:
    r"""Consume one Q level and return the private target scale $\Delta_0$.

    A pending-scale input near $\Delta_0^2$ is rescaled directly. An ordinary
    input near $\Delta_0$ is first multiplied by an encoding of $1$ at
    $\Delta_0$. In either case the core arithmetic first records
    $\Delta_{\rm actual}=\Delta_{\rm product}/q_{\rm drop}$; bootstrap then
    reinterprets that unchanged residue tensor at $\Delta_0$. The output is a
    functional two-component coefficient-domain standard-RNS Q value at level
    $\ell+1$ with the leading `prime_ids` row removed.
    """

    pending_ratio = ciphertext.scale / engine.config.default_scale**2
    if 0.5 <= pending_ratio <= 2.0:
        return _rescale_to_default_scale(engine, ciphertext)
    ordinary_ratio = ciphertext.scale / engine.config.default_scale
    if not 0.5 <= ordinary_ratio <= 2.0:
        raise ValueError(
            'Cannot advance ciphertext with unsupported scale/Delta='
            f'{ordinary_ratio:.6g}'
        )
    return _multiply_scalar(engine, ciphertext, 1.0)


def _multiply_relinearize_rescale(
    engine: CkksEngine,
    lhs: Ciphertext,
    rhs: Ciphertext,
    *,
    relinearization_key: RelinearizationKey,
) -> Ciphertext:
    r"""Compute $c_{\rm lhs}c_{\rm rhs}$ under bootstrap's scale policy.

    Inputs are level-aligned, converted from coefficient-domain standard RNS
    to NTT-domain Montgomery RNS, multiplied by component convolution to three
    components at scale $\Delta_{\rm lhs}\Delta_{\rm rhs}$, and relinearized
    back to two coefficient-domain standard components. One leading Q row is
    then dropped and the result is explicitly reinterpreted at $\Delta_0$.
    The result is functional and has level
    $\max(\ell_{\rm lhs},\ell_{\rm rhs})+1$ after any alignment levels, Q basis,
    and the corresponding exact `prime_ids`.
    """

    lhs, rhs = _align_levels(engine, lhs, rhs)
    product = engine.relinearize(
        engine.multiply(
            engine.coefficient_domain_to_ntt_domain(lhs),
            engine.coefficient_domain_to_ntt_domain(rhs),
        ),
        relinearization_key,
    )
    return _rescale_to_default_scale(engine, product)


def _rescale_to_default_scale(
    engine: CkksEngine,
    ciphertext: Ciphertext,
) -> Ciphertext:
    r"""Rescale once, then explicitly reinterpret metadata at $\Delta_0$.

    Public CKKS rescale preserves the actual `input_scale / dropped_q`
    metadata. The current bootstrap polynomial and transform schedules were
    compiled for one default scale, so this private circuit deliberately
    accepts the small prime-ratio approximation and reinterprets every
    internal rescale result at that scale. Keeping this policy here prevents
    the core arithmetic API from silently making the same choice for user
    programs.

    Core rescale first computes

    $$
    c'=\operatorname{Round}(c/q_{\rm drop}),\qquad
    \Delta'=\Delta(c)/q_{\rm drop}.
    $$

    Bootstrap leaves $c'$ unchanged and records $\Delta_0$, so its interpreted
    message is multiplied by $\Delta'/\Delta_0$. The operation advances the
    level, removes the leading Q row from `prime_ids`, preserves two-component
    coefficient-domain standard-RNS state, and does not mutate the input.
    """

    return engine.reinterpret_at_scale(
        engine.rescale_to_next_level(ciphertext),
        engine.config.default_scale,
    )


def _multiply_scalar(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    scalar: complex,
) -> Ciphertext:
    r"""Compute $c_{\rm out}=a c$ and return metadata scale $\Delta_0$.

    The scalar $a$ is broadcast across the semantic `[slot]` axis and every
    homogeneous batch member. It is encoded as an unbatched NTT/Montgomery Q
    plaintext at input level and scale $\Delta_0$. Plaintext multiplication
    preserves two components and coefficient-domain standard RNS at pending
    scale $\Delta(c)\Delta_0$; one rescale then advances the level, drops the
    leading `prime_ids` row, and reinterprets the result at $\Delta_0$.
    Evaluation is functional.
    """

    plaintext = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.full(
                (engine.num_slots,),
                complex(scalar),
                dtype=torch.complex128,
                device=engine.device,
            ),
            level=ciphertext.level,
            scale=engine.config.default_scale,
        ),
        modulus_basis='Q',
    )
    return _rescale_to_default_scale(
        engine,
        engine.ntt_domain_to_coefficient_domain(
            engine.multiply_plaintext(
                engine.coefficient_domain_to_ntt_domain(ciphertext),
                plaintext,
            )
        ),
    )


def _add_scalar(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    scalar: complex,
) -> Ciphertext:
    r"""Compute $c_{\rm out}=c+a$ without changing CKKS state metadata.

    The scalar is broadcast across semantic `[slot]` and batch axes, encoded at
    the ciphertext's exact actual scale, and prepared as coefficient-domain
    Montgomery RNS over the same Q `prime_ids`. Addition updates only $c_0$.
    The functional output preserves data shape, level, scale, two components,
    coefficient-domain standard residues, basis, and exact `prime_ids`.
    """

    plaintext = engine.prepare_plaintext_for_addition(
        engine.encode(
            torch.full(
                (engine.num_slots,),
                complex(scalar),
                dtype=torch.complex128,
                device=engine.device,
            ),
            level=ciphertext.level,
            scale=ciphertext.scale,
        ),
        modulus_basis='Q',
    )
    return engine.add_plaintext(ciphertext, plaintext)
