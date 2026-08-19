"""Centered modulus raising and its exact-RNS reference helpers."""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from fhelium.core import Ciphertext
from fhelium.engine.ckks_engine import CkksEngine
from fhelium.native.wrapper import rns_ops


def _mixed_radix_tables(
    engine: CkksEngine,
    source_prime_ids: tuple[int, ...],
    target_prime_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build device tables for Garner decomposition and basis extension.

    ``source_prime_ids`` names the depleted Q basis whose residues represent
    the input integer.  Garner decomposition expresses that integer as mixed
    radix digits with prefix products $q_0$, $q_0q_1$, and so on. The three
    returned tables serve different native-kernel steps:

    * ``normalizers[i]`` is the inverse of prefix product $M_{i+1}$ modulo the
      next source prime, in Montgomery form;
    * ``propagation[i, j]`` propagates the corresponding digit into later
      source row ``j``;
    * ``extension[i, k]`` contributes that digit times $M_{i+1}$ to target row
      ``k`` and is stored with the target row's Montgomery $R^2$
      factor expected by ``mixed_radix_basis_extend_to_montgomery``.

    A one-row source has no higher mixed-radix digits, so the tables have zero
    prefix rows while direct residue copying remains valid.

    Outputs are separately allocated integral CUDA tensors on ``engine.device``
    with shapes ``[source_limb - 1]``,
    ``[source_limb - 1, source_limb]``, and
    ``[source_limb - 1, destination_limb]``. Source and destination columns map
    exactly to the supplied ``prime_ids`` order.
    """

    moduli = engine.montgomery_parameters.moduli
    source_moduli = [int(moduli[index]) for index in source_prime_ids]
    target_moduli = [int(moduli[index]) for index in target_prime_ids]
    target_r2 = [
        int(engine.montgomery_parameters.montgomery_r2[index])
        for index in target_prime_ids
    ]
    width = len(source_moduli)
    prefix_products: list[int] = []
    product = source_moduli[0]
    for index in range(width - 1):
        if index:
            product *= source_moduli[index]
        prefix_products.append(product)

    normalizers = []
    propagation = torch.zeros(
        (max(width - 1, 0), width),
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    for component_index, prefix in enumerate(prefix_products):
        next_modulus = source_moduli[component_index + 1]
        normalizers.append(
            (pow(prefix, -1, next_modulus) * engine.montgomery_parameters.R)
            % next_modulus
        )
        for target_index in range(component_index + 2, width):
            propagation[component_index, target_index] = (
                prefix * engine.montgomery_parameters.R
            ) % source_moduli[target_index]

    extension = torch.tensor(
        [
            [
                (prefix * r2) % modulus
                for r2, modulus in zip(target_r2, target_moduli, strict=True)
            ]
            for prefix in prefix_products
        ],
        dtype=engine.config.torch_dtype,
        device=engine.device,
    )
    if width == 1:
        extension = torch.empty(
            (0, len(target_prime_ids)),
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
    return (
        torch.tensor(
            normalizers,
            dtype=engine.config.torch_dtype,
            device=engine.device,
        ),
        propagation,
        extension,
    )


def _mixed_radix_half_digits(moduli: Sequence[int]) -> tuple[int, ...]:
    r"""Represent $\lfloor(\prod_i q_i)/2\rfloor$ in mixed-radix form."""

    value = math.prod(int(modulus) for modulus in moduli) // 2
    digits = []
    for modulus in moduli:
        value, digit = divmod(value, int(modulus))
        digits.append(digit)
    return tuple(digits)


def _mixed_radix_is_above_half(
    digits: torch.Tensor,
    moduli: Sequence[int],
) -> torch.Tensor:
    """Identify coefficients represented by the upper half of the source ring.

    Mixed-radix significance increases with the row index, so comparison starts
    at the last digit and proceeds toward row zero.  The result has the batch
    and coefficient dimensions of ``digits`` with the RNS-row dimension
    removed.  Equality with the half value is not considered negative; only
    strictly larger representatives are recentered.
    """

    half_digits = _mixed_radix_half_digits(moduli)
    equal = torch.ones_like(digits[..., 0, :], dtype=torch.bool)
    above = torch.zeros_like(equal)
    for row in range(len(moduli) - 1, -1, -1):
        above |= equal & (digits[..., row, :] > half_digits[row])
        equal &= digits[..., row, :] == half_digits[row]
    return above


def reference_centered_basis_extend(
    residues: torch.Tensor,
    *,
    source_moduli: Sequence[int],
    target_moduli: Sequence[int],
    centered: bool = True,
) -> torch.Tensor:
    """Reconstruct source integers and reduce them into a target RNS basis.

    This deliberately slow implementation is a test oracle for the native
    mixed-radix path.  It applies ordinary CRT with Python integers to every
    coefficient.  When ``centered`` is true, representatives greater than half
    the source modulus product are interpreted as negative before reduction
    into each target modulus.

    Args:
        residues: Tensor shaped ``[..., source_row, coefficient]``.
        source_moduli: Modulus corresponding to each source row.
        target_moduli: Moduli of the desired output rows.
        centered: Choose signed half-interval representatives instead of
            canonical nonnegative representatives.

    Returns:
        Tensor on the original device with shape
        ``[..., target_row, coefficient]`` and the same integer dtype.
    """

    if residues.ndim < 2:
        raise ValueError(
            "residues must have [..., source row, coefficient] shape"
        )
    if residues.size(-2) != len(source_moduli):
        raise ValueError("source row count does not match source_moduli")
    source_moduli = tuple(int(modulus) for modulus in source_moduli)
    target_moduli = tuple(int(modulus) for modulus in target_moduli)
    source_product = math.prod(source_moduli)
    crt_terms = []
    for modulus in source_moduli:
        partial = source_product // modulus
        crt_terms.append(partial * pow(partial, -1, modulus))

    source = (
        residues.detach()
        .cpu()
        .reshape(-1, len(source_moduli), residues.size(-1))
    )
    output = torch.empty(
        (source.size(0), len(target_moduli), source.size(-1)),
        dtype=residues.dtype,
    )
    for batch in range(source.size(0)):
        for coefficient in range(source.size(-1)):
            value = (
                sum(
                    int(source[batch, row, coefficient])
                    % source_moduli[row]
                    * crt_terms[row]
                    for row in range(len(source_moduli))
                )
                % source_product
            )
            if centered and value > source_product // 2:
                value -= source_product
            for row, modulus in enumerate(target_moduli):
                output[batch, row, coefficient] = value % modulus
    return output.reshape(
        *residues.shape[:-2], len(target_moduli), residues.size(-1)
    ).to(residues.device)


@dataclass(frozen=True)
class ModRaisedCiphertext:
    """A ciphertext plus the depleted Q basis extended by ModRaise."""

    ciphertext: Ciphertext
    source_level: int
    source_prime_ids: tuple[int, ...]
    source_modulus_bits: int
    scale: float

    @classmethod
    def from_engine(
        cls,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        *,
        source_level: int,
        scale: float,
    ) -> ModRaisedCiphertext:
        """Attach the source basis needed by the first raised transform."""

        source_prime_ids = engine.rns_layout.prime_ids(source_level)
        source_modulus = math.prod(
            engine.montgomery_parameters.moduli[index]
            for index in source_prime_ids
        )
        return cls(
            ciphertext=ciphertext,
            source_level=source_level,
            source_prime_ids=source_prime_ids,
            source_modulus_bits=source_modulus.bit_length(),
            scale=float(scale),
        )


def _prepare_entry(
    engine: CkksEngine,
    ciphertext: Ciphertext,
) -> Ciphertext:
    r"""Prepare the pending scale required by the structural-base rescale.

    A final-level input may arrive at default scale $\Delta_0$ or already at
    pending-product scale $\Delta_0^2$. The pending input passes through
    unchanged. For an ordinary input, multiplying by a plaintext encoding of
    one changes the represented polynomial from approximately $\Delta_0m$ to
    $\Delta_0^2m$ without changing its decoded message. This multiplication
    consumes no level; the following structural-base rescale consumes it.

    Rescaling a default-scale input directly would leave scale near one, so
    other input scales are rejected rather than silently changing the required
    CKKS scale relation.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        components=2,
    )
    engine._assert_engine_ciphertext(ciphertext)
    ordinary_ratio = ciphertext.scale / engine.config.default_scale
    pending_ratio = ciphertext.scale / engine.config.default_scale**2
    if 0.5 <= pending_ratio <= 2.0:
        return ciphertext
    if not 0.5 <= ordinary_ratio <= 2.0:
        raise ValueError(
            'bootstrap input scale must be close to Delta or Delta**2; '
            f'got scale/Delta={ordinary_ratio:.6g}'
        )
    identity = engine.prepare_plaintext_for_multiplication(
        engine.encode(
            torch.ones(
                engine.num_slots, dtype=torch.float64, device=engine.device
            ),
            level=ciphertext.level,
            scale=engine.config.default_scale,
        ),
        modulus_basis='Q',
    )
    return engine.ntt_domain_to_coefficient_domain(
        engine.multiply_plaintext(
            engine.coefficient_domain_to_ntt_domain(ciphertext), identity
        )
    )


def _rescale_to_structural_base(
    engine: CkksEngine,
    ciphertext: Ciphertext,
) -> Ciphertext:
    r"""Drop the final public prime and enter the structural-base representation.

    The input must be at the final public level with pending scale near
    $\Delta_0^2$. Nearest-integer rescale divides its residues and metadata by
    the final public prime, which is chosen near $\Delta_0$. This bootstrap-only
    transition then explicitly reinterprets the result at the circuit's default
    scale and retains only the structural base prime used as the source basis
    for centered ModRaise.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        components=2,
    )
    engine._assert_engine_ciphertext(ciphertext)
    expected_level = engine.final_public_level
    if ciphertext.level != expected_level:
        raise ValueError(
            'bootstrap entry rescale requires final public level '
            f'{expected_level}, got {ciphertext.level}'
        )
    pending_ratio = ciphertext.scale / engine.config.default_scale**2
    if not 0.5 <= pending_ratio <= 2.0:
        raise ValueError(
            'bootstrap entry requires pending scale near Delta**2; '
            f'got scale/Delta**2={pending_ratio:.6g}'
        )
    structural_value = engine._rescale_final_public_level_to_structural_base(
        ciphertext,
        rounding='nearest',
    )
    # This private structural level intentionally lies beyond the public CKKS
    # level interval and therefore must not re-enter the public facade's
    # imported-value validator.
    structural_value.scale = engine.config.default_scale
    return structural_value


def _modulus_raise(
    engine: CkksEngine,
    cache: MutableMapping[Any, Any],
    ciphertext: Ciphertext,
    *,
    target_level: int,
) -> ModRaisedCiphertext:
    r"""Centered-extend each ciphertext component into a larger Q basis.

    Mixed-radix decomposition reconstructs the centered integer represented by
    the depleted source rows, then emits residues for the Q prefix beginning at
    ``target_level``. The encoded message and scale are preserved; this is a
    basis extension rather than a rescale. The returned private wrapper also
    records the source basis needed by the first coefficient-to-slot transform.

    ``cache`` stores only modulus-dependent mixed-radix tables and may be reused
    across ciphertexts from the same engine configuration.

    Input payload is integral CUDA
    ``[component=2, *batch, source_limb, coefficient]`` in
    coefficient/standard/canonical Q state. Each coefficient is interpreted in
    the centered interval modulo the source product, then reduced exactly into
    every ``target_prime_ids`` row. Output owns newly allocated
    ``[component=2, *batch, destination_limb, coefficient]`` storage in
    coefficient/standard/canonical Q state. Component count, batch axes, exact
    integer polynomial, and actual scale are preserved; no input aliases or is
    mutated by the returned ciphertext.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        modulus_basis='Q',
        components=2,
    )
    engine._assert_structural_base_ciphertext(ciphertext)
    if not 0 <= target_level < ciphertext.level:
        raise ValueError(
            f'target_level must satisfy 0 <= target_level < {ciphertext.level}'
        )
    source_prime_ids: tuple[int, ...] = ciphertext.prime_ids
    target_prime_ids: tuple[int, ...] = engine.rns_layout.prime_ids(
        target_level
    )
    if not source_prime_ids or not target_prime_ids:
        raise ValueError(
            'modulus raise requires nonempty source and target bases'
        )
    if len(source_prime_ids) > 8:
        raise NotImplementedError(
            'native bootstrap modulus raise currently supports at most 8 '
            'source rows'
        )
    cache_key = (source_prime_ids, target_prime_ids)
    runtime = cache.get(cache_key)
    if runtime is None:
        source_parameters = engine.rns_runtime.row_parameters(source_prime_ids)
        normalizers, propagation, extension = _mixed_radix_tables(
            engine, source_prime_ids, target_prime_ids
        )
        target_params = engine.rns_runtime.rns_parameter_tensor[
            :, target_prime_ids[0] : target_prime_ids[-1] + 1
        ]
        source_moduli = tuple(
            int(engine.montgomery_parameters.moduli[index])
            for index in source_prime_ids
        )
        target_moduli = tuple(
            int(engine.montgomery_parameters.moduli[index])
            for index in target_prime_ids
        )
        source_product = math.prod(source_moduli)
        corrections = torch.tensor(
            [source_product % modulus for modulus in target_moduli],
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
        target_moduli_tensor = torch.tensor(
            target_moduli,
            dtype=engine.config.torch_dtype,
            device=engine.device,
        )
        runtime = (
            source_parameters,
            normalizers,
            propagation,
            extension,
            target_params,
            source_moduli,
            corrections,
            target_moduli_tensor,
        )
        cache[cache_key] = runtime
    (
        source_parameters,
        normalizers,
        propagation,
        extension,
        target_params,
        source_moduli,
        corrections_1d,
        target_moduli_1d,
    ) = runtime
    source_start = min(source_prime_ids)
    source_stop = max(source_prime_ids) + 1
    source_params = engine.rns_runtime.rns_parameter_tensor[
        :, source_start:source_stop
    ]
    raised_components = []
    for component in (ciphertext.c0, ciphertext.c1):
        source = component.clone()
        rns_ops.canonicalize_residues_(source, source_params)
        if len(source_prime_ids) == 1:
            digits = source
        else:
            lo, hi, neg_lo, neg_hi = (
                source_parameters.montgomery_reduction_parameters
            )
            digits = rns_ops.mixed_radix_decompose(
                source,
                normalizers,
                propagation,
                lo,
                hi,
                neg_lo,
                neg_hi,
            )
            rns_ops.canonicalize_residues_(digits, source_params)
        raised = rns_ops.mixed_radix_basis_extend_to_montgomery(
            digits,
            extension,
            target_params,
            len(target_prime_ids),
        )
        rns_ops.from_montgomery_(raised, target_params)
        rns_ops.canonicalize_residues_(raised, target_params)
        negative = _mixed_radix_is_above_half(digits, source_moduli)
        corrections = corrections_1d.view(*([1] * (raised.ndim - 2)), -1, 1)
        moduli = target_moduli_1d.view(*([1] * (raised.ndim - 2)), -1, 1)
        raised = torch.where(
            negative.unsqueeze(-2),
            raised - corrections,
            raised,
        )
        raised_components.append(torch.remainder(raised, moduli))

    raised_ct = engine._ciphertext_from_components(
        raised_components,
        level=target_level,
        scale=ciphertext.scale,
        polynomial_domain='coefficient',
        modulus_basis='Q',
        residue_representation="standard",
        prime_ids=target_prime_ids,
    )
    return ModRaisedCiphertext.from_engine(
        engine,
        raised_ct,
        source_level=ciphertext.level,
        scale=ciphertext.scale,
    )
