"""Centered modulus raising and its integer-RNS reference helpers."""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from fhelium.values import Ciphertext
from fhelium.eager import Engine
from fhelium.native.wrapper import rns_ops


def _validate_structural_ciphertext(
    engine: Engine,
    value: Ciphertext,
) -> None:
    """Validate the terminal-basis ciphertext consumed by ModRaise."""

    if not isinstance(value, Ciphertext):
        raise TypeError(f"Expected Ciphertext, got {type(value).__name__}")
    Ciphertext(
        data=value.data,
        depth=value.depth,
        scale=value.scale,
        prime_ids=value.prime_ids,
        polynomial_domain=value.polynomial_domain,
        modulus_basis=value.modulus_basis,
        residue_representation=value.residue_representation,
    )
    if value.depth != engine.max_depth:
        raise ValueError(
            "ModRaise input must use terminal depth "
            f"{engine.max_depth}, got {value.depth}"
        )
    if value.data.dtype != engine.dtype:
        raise TypeError("Ciphertext dtype differs from the engine")
    if value.data.size(-1) != engine.ring_dimension:
        raise ValueError("Ciphertext ring dimension differs from the engine")
    if value.modulus_basis != "Q":
        raise ValueError("Structural-base Ciphertext requires Q basis")
    value.assert_state(
        polynomial_domain="coefficient",
        residue_representation="standard",
        components=2,
    )
    rns_context = engine._rns_context_for(value.device)
    expected_prime_ids = rns_context.rns_layout.prime_ids(
        engine.max_depth
    )
    if value.prime_ids != expected_prime_ids:
        raise ValueError(
            "Structural-base Ciphertext prime IDs differ from the engine: "
            f"{value.prime_ids} != {expected_prime_ids}"
        )


def _mixed_radix_tables(
    engine: Engine,
    device: torch.device,
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

    Outputs are separately allocated integral tensors on ``device``
    with shapes ``[source_limb - 1]``,
    ``[source_limb - 1, source_limb]``, and
    ``[source_limb - 1, destination_limb]``. Source and destination columns map
    exactly to the supplied ``prime_ids`` order.
    """

    rns_context = engine._rns_context_for(device)
    moduli = rns_context.montgomery_parameters.moduli
    source_moduli = [int(moduli[index]) for index in source_prime_ids]
    target_moduli = [int(moduli[index]) for index in target_prime_ids]
    target_r2 = [
        int(rns_context.montgomery_parameters.montgomery_r2[index])
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
        dtype=engine.dtype,
        device=device,
    )
    for component_index, prefix in enumerate(prefix_products):
        next_modulus = source_moduli[component_index + 1]
        normalizers.append(
            (
                pow(prefix, -1, next_modulus)
                * rns_context.montgomery_parameters.R
            )
            % next_modulus
        )
        for target_index in range(component_index + 2, width):
            propagation[component_index, target_index] = (
                prefix * rns_context.montgomery_parameters.R
            ) % source_moduli[target_index]

    extension = torch.tensor(
        [
            [
                (prefix * r2) % modulus
                for r2, modulus in zip(target_r2, target_moduli, strict=True)
            ]
            for prefix in prefix_products
        ],
        dtype=engine.dtype,
        device=device,
    )
    if width == 1:
        extension = torch.empty(
            (0, len(target_prime_ids)),
            dtype=engine.dtype,
            device=device,
        )
    return (
        torch.tensor(
            normalizers,
            dtype=engine.dtype,
            device=device,
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
            least nonnegative representatives.

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
    source_depth: int
    source_prime_ids: tuple[int, ...]
    source_modulus_bits: int
    scale: float

    @classmethod
    def from_engine(
        cls,
        engine: Engine,
        ciphertext: Ciphertext,
        *,
        source_depth: int,
        scale: float,
    ) -> ModRaisedCiphertext:
        """Attach the source basis needed by the first raised transform."""

        rns_context = engine._rns_context_for(ciphertext.device)
        source_prime_ids = rns_context.rns_layout.prime_ids(source_depth)
        source_modulus = math.prod(
            rns_context.montgomery_parameters.moduli[index]
            for index in source_prime_ids
        )
        return cls(
            ciphertext=ciphertext,
            source_depth=source_depth,
            source_prime_ids=source_prime_ids,
            source_modulus_bits=source_modulus.bit_length(),
            scale=float(scale),
        )


def _prepare_entry(
    engine: Engine,
    ciphertext: Ciphertext,
) -> Ciphertext:
    r"""Lift the entry scale by an integer before terminal rescale.

    Choose $k=\max(1,\lceil M_d\Delta_0/\Delta_{in}\rceil)$ and replace
    $(c,\Delta_{in})$ with $(kc,k\Delta_{in})$. The integer multiplication
    preserves the represented message without quantizing an encoded identity.
    When $k=1$ the input is returned unchanged. The quotient scale after the
    following rescale is retained for the full-slot coordinate conversion.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        components=2,
    )
    engine.validate_ciphertext(ciphertext)
    multiplier = max(1, math.ceil(
        engine.config.rescale_divisor(ciphertext.depth)
        * engine.config.default_scale / ciphertext.scale
    ))
    if multiplier == 1:
        return ciphertext
    result = engine.multiply_integer_scalar(ciphertext, multiplier)
    result.scale = ciphertext.scale * multiplier
    return result


def _rescale_to_structural_basis(
    engine: Engine,
    ciphertext: Ciphertext,
) -> Ciphertext:
    r"""Rescale the prepared Bootstrap input into the terminal Q basis.

    The input occupies the penultimate depth. Nearest group rescale preserves
    the represented message and divides the pending scale by the group
    product. The terminal basis is then used for centered ModRaise.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        components=2,
    )
    engine.validate_ciphertext(ciphertext)
    expected_depth = engine.max_depth - 1
    if ciphertext.depth != expected_depth:
        raise ValueError(
            'bootstrap entry rescale requires penultimate depth '
            f'{expected_depth}, got {ciphertext.depth}'
        )
    structural_value = engine.rescale_to_next_depth(
        ciphertext,
        rounding='nearest',
    )
    return structural_value


def _modulus_raise(
    engine: Engine,
    cache: MutableMapping[Any, Any],
    ciphertext: Ciphertext,
    *,
    target_depth: int,
) -> ModRaisedCiphertext:
    r"""Centered-extend each ciphertext component into a larger Q basis.

    Mixed-radix decomposition reconstructs the centered integer represented by
    the depleted source rows, then emits residues for the Q prefix beginning at
    ``target_depth``. The encoded message and scale are preserved; this is a
    basis extension rather than a rescale. The returned private wrapper also
    records the source basis needed by the first coefficient-to-slot transform.

    ``cache`` stores only modulus-dependent mixed-radix tables and may be reused
    across ciphertexts from the same engine configuration.

    Input payload is integral CUDA
    ``[component=2, *batch, source_limb, coefficient]`` in
    coefficient/standard Q state with residues in $[0,q_i)$. Each coefficient is interpreted in
    the centered interval modulo the source product, then reduced exactly into
    every ``target_prime_ids`` row. Output owns newly allocated
    ``[component=2, *batch, destination_limb, coefficient]`` storage in
    coefficient/standard Q state with residues in $[0,q_i)$. Component count, batch axes,
    integer polynomial, and actual scale are preserved; no input aliases or is
    mutated by the returned ciphertext.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation="standard",
        modulus_basis='Q',
        components=2,
    )
    _validate_structural_ciphertext(engine, ciphertext)
    if not 0 <= target_depth < ciphertext.depth:
        raise ValueError(
            f'target_depth must satisfy 0 <= target_depth < {ciphertext.depth}'
        )
    source_prime_ids: tuple[int, ...] = ciphertext.prime_ids
    rns_context = engine._rns_context_for(ciphertext.device)
    target_prime_ids: tuple[int, ...] = rns_context.rns_layout.prime_ids(
        target_depth
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
    cache_key = (ciphertext.device, source_prime_ids, target_prime_ids)
    runtime = cache.get(cache_key)
    if runtime is None:
        source_parameters = rns_context.row_parameters(source_prime_ids)
        normalizers, propagation, extension = _mixed_radix_tables(
            engine, ciphertext.device, source_prime_ids, target_prime_ids
        )
        target_params = rns_context.rns_parameter_tensor[
            :, target_prime_ids[0] : target_prime_ids[-1] + 1
        ]
        source_moduli = tuple(
            int(rns_context.montgomery_parameters.moduli[index])
            for index in source_prime_ids
        )
        target_moduli = tuple(
            int(rns_context.montgomery_parameters.moduli[index])
            for index in target_prime_ids
        )
        source_product = math.prod(source_moduli)
        corrections = torch.tensor(
            [source_product % modulus for modulus in target_moduli],
            dtype=engine.dtype,
            device=ciphertext.device,
        )
        target_moduli_tensor = torch.tensor(
            target_moduli,
            dtype=engine.dtype,
            device=ciphertext.device,
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
    source_params = rns_context.rns_parameter_tensor[
        :, source_start:source_stop
    ]
    raised_components = []
    for component in (ciphertext.c0, ciphertext.c1):
        source = component.clone()
        rns_ops.reduce_to_standard_(source, source_params)
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
            rns_ops.reduce_to_standard_(digits, source_params)
        raised = rns_ops.mixed_radix_basis_extend_to_montgomery(
            digits,
            extension,
            target_params,
            len(target_prime_ids),
        )
        rns_ops.from_montgomery_(raised, target_params)
        rns_ops.reduce_to_standard_(raised, target_params)
        negative = _mixed_radix_is_above_half(digits, source_moduli)
        corrections = corrections_1d.view(*([1] * (raised.ndim - 2)), -1, 1)
        moduli = target_moduli_1d.view(*([1] * (raised.ndim - 2)), -1, 1)
        raised = torch.where(
            negative.unsqueeze(-2),
            raised - corrections,
            raised,
        )
        raised_components.append(torch.remainder(raised, moduli))

    raised_ct = Ciphertext(
        data=torch.stack(raised_components, dim=0),
        depth=target_depth,
        scale=ciphertext.scale,
        prime_ids=target_prime_ids,
        polynomial_domain='coefficient',
        modulus_basis='Q',
        residue_representation="standard",
    )
    engine.validate_ciphertext(raised_ct)
    return ModRaisedCiphertext.from_engine(
        engine,
        raised_ct,
        source_depth=ciphertext.depth,
        scale=ciphertext.scale,
    )
