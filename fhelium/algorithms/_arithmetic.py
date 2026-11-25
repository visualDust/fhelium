r"""CKKS arithmetic with depth-dependent scale targets for algorithm components."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal
import math

import torch

from fhelium.eager import Engine
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    RelinearizationKey,
    EvaluationKeySet,
)


def _cached_constant(
    cache: MutableMapping[object, object] | None,
    key: object,
    *,
    prepare: bool,
) -> object | None:
    """Read a constant from the preparation cache or bound execution table."""
    if prepare:
        return None if cache is None else cache.get(key)
    if cache is None:
        raise ValueError(
            "Prepared Bootstrap arithmetic requires bound constants"
        )
    return cache[key]


def _multiply_by_monomial(
    engine: Engine,
    ciphertext: Ciphertext,
    exponent: int,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    prepare_materials: bool = True,
) -> Ciphertext:
    r"""Multiply every component by $X^e$ in $R=\mathbb{Z}[X]/(X^N+1)$.

    For each `[component, *batch, limb]` polynomial and $e=$ `exponent`, the
    last `[coefficient]` axis is negacyclically shifted so that

    $$
    c_{\rm out}(X)=X^e c_{\rm in}(X)\pmod{X^N+1}.
    $$

    The input must use coefficient-domain standard residues on the engine
    device. The operation is functional and preserves shape, component count,
    depth, actual scale, basis, `prime_ids`, polynomial domain, and
    residue representation. The output is reduced into each limb's
    interval $[0,q_i)$ and does not alias the input.
    """

    ciphertext.assert_state(
        polynomial_domain='coefficient', residue_representation="standard"
    )
    engine.validate_ciphertext(ciphertext)
    degree = engine.config.N
    exponent %= 2 * degree
    sign = -1 if exponent >= degree else 1
    shift = exponent % degree
    data = (
        torch.roll(ciphertext.data, shift, dims=-1)
        if shift
        else ciphertext.data.clone()
    )
    if shift:
        data = torch.cat((-data[..., :shift], data[..., shift:]), dim=-1)
    if sign < 0:
        data = -data
    cache_key = (
        'bootstrap-monomial-moduli',
        engine,
        ciphertext.data.dtype,
        ciphertext.device,
        ciphertext.prime_ids,
    )
    moduli = _cached_constant(
        constant_cache, cache_key, prepare=prepare_materials
    )
    if not isinstance(moduli, torch.Tensor):
        moduli = torch.tensor(
            [engine.config.moduli[index] for index in ciphertext.prime_ids],
            dtype=data.dtype,
            device=data.device,
        )
        if constant_cache is not None:
            constant_cache[cache_key] = moduli
    moduli = moduli.view(*([1] * (data.ndim - 2)), -1, 1)
    return ciphertext.with_data(torch.remainder(data, moduli))


def _align_depths(
    arithmetic: Arithmetic,
    lhs: Ciphertext,
    rhs: Ciphertext,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> tuple[Ciphertext, Ciphertext]:
    r"""Advance operands to the same Q-chain position using encoded one.

    Each advancement performs a plaintext multiplication and one group
    rescale, targeting the arithmetic owner's scale at the next depth.
    Operands already at the selected depth are returned unchanged; addition
    still requires their actual scales to agree.
    """

    while lhs.depth < rhs.depth:
        lhs = _advance_depth(
            arithmetic,
            lhs,
            constant_cache=constant_cache,
            retain_ntt=retain_ntt,
        )
    while rhs.depth < lhs.depth:
        rhs = _advance_depth(
            arithmetic,
            rhs,
            constant_cache=constant_cache,
            retain_ntt=retain_ntt,
        )
    return lhs, rhs


def _advance_depth(
    arithmetic: Arithmetic,
    ciphertext: Ciphertext,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Preserve the message while advancing one depth.

    Multiplication by an encoded one uses plaintext scale
    $\rho_d s_{d+1}/\Delta(c)$. Rescaling therefore returns the message at
    target scale $s_{d+1}$ without a metadata-only adjustment.
    """

    return _multiply_scalar(
        arithmetic,
        ciphertext,
        1.0,
        constant_cache=constant_cache,
        retain_ntt=retain_ntt,
    )


def _multiply_relinearize_rescale(
    arithmetic: Arithmetic,
    lhs: Ciphertext,
    rhs: Ciphertext,
    *,
    relinearization_key: RelinearizationKey
    | EvaluationKeySet
    | Mapping[int, RelinearizationKey],
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Multiply aligned ciphertexts, relinearize, and rescale once.

    Component convolution produces three components at actual scale
    $\Delta_l\Delta_r$. Relinearization restores two components; rescale
    divides the scale by $\rho_d$. When both operands have scheduled scale
    $s_d$, the result has $s_{d+1}=s_d^2/\rho_d$. No scale metadata is replaced.
    An evaluation-key inventory is looked up at the product depth; a supplied
    primitive key is used directly. The result retains NTT form when requested
    by ``retain_ntt`` and the
    input representation; otherwise it is coefficient/standard.
    """

    engine = arithmetic.engine

    lhs, rhs = _align_depths(
        arithmetic,
        lhs,
        rhs,
        constant_cache=constant_cache,
        retain_ntt=retain_ntt,
    )
    if (
        lhs.polynomial_domain != rhs.polynomial_domain
        or lhs.residue_representation != rhs.residue_representation
    ):
        raise ValueError(
            'bootstrap ciphertext multiplication requires matching '
            'represented domains'
        )
    input_domain = lhs.polynomial_domain
    lhs_ntt = (
        lhs
        if input_domain == 'ntt'
        else engine.coefficient_domain_to_ntt_domain(lhs)
    )
    rhs_ntt = (
        rhs
        if input_domain == 'ntt'
        else engine.coefficient_domain_to_ntt_domain(rhs)
    )
    product = engine.multiply(lhs_ntt, rhs_ntt)
    return arithmetic.relinearize(
        product,
        relinearization_key,
        rescale=True,
        output_domain='ntt'
        if retain_ntt and input_domain == 'ntt'
        else 'coefficient',
    )


def _rescale(
    engine: Engine,
    ciphertext: Ciphertext,
    *,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Rescale once and retain the arithmetic result's actual scale.

    The quotient is rounded by the Engine's ordinary rescale operation.
    ``retain_ntt=False`` converts an NTT input to coefficient form first.
    """

    if not retain_ntt and ciphertext.polynomial_domain == 'ntt':
        ciphertext = engine.ntt_domain_to_coefficient_domain(ciphertext)
    return engine.rescale_to_next_depth(ciphertext)


def _multiply_scalar(
    arithmetic: Arithmetic,
    ciphertext: Ciphertext,
    scalar: complex,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Multiply the represented message by $a$ and advance one depth.

    The scalar plaintext scale targets $s_{d+1}$ through
    $p=\rho_d s_{d+1}/\Delta(c)$; execution records the actual binary64
    quotient $(\Delta(c)p)/\rho_d$. It preserves the represented multiplication
    apart from encoding and quotient rounding. It retains the
    input representation unless coefficient output is requested.
    """

    engine = arithmetic.engine

    plaintext = _prepare_multiply_scalar(
        arithmetic,
        ciphertext,
        scalar,
        constant_cache=constant_cache,
    )
    input_domain = ciphertext.polynomial_domain
    ciphertext_ntt = (
        ciphertext
        if input_domain == 'ntt'
        else engine.coefficient_domain_to_ntt_domain(ciphertext)
    )
    result = _rescale(
        engine,
        engine.multiply_plaintext(ciphertext_ntt, plaintext),
        retain_ntt=retain_ntt,
    )
    if input_domain == 'ntt' or not retain_ntt:
        return result
    return engine.ntt_domain_to_coefficient_domain(result)


def _prepare_multiply_scalar(
    arithmetic: Arithmetic,
    ciphertext: Ciphertext,
    scalar: complex,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
) -> CompressedPlaintext:
    """Prepare one scalar plaintext for a pending multiplication."""

    engine = arithmetic.engine

    plaintext_scale = arithmetic.plaintext_scale(
        ciphertext.scale, ciphertext.depth
    )
    return _prepare_scalar(
        engine,
        ciphertext,
        scalar,
        plaintext_scale,
        'ntt',
        constant_cache=constant_cache,
        prepare_materials=arithmetic._prepare_materials,
    )


def _prepare_scalar(
    engine: Engine,
    ciphertext: Ciphertext,
    scalar: complex,
    scale: float,
    domain: Literal["coefficient", "ntt"],
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    prepare_materials: bool = True,
) -> CompressedPlaintext:
    r"""Encode a constant slot message with the target ring's periodic map.

    The period-one message uses the Engine's generator-dependent compact
    polynomial and actual Q rows. The first preparation consumes stochastic
    rounding state. A cache hit reuses that encoded sample without consuming
    further randomness; clearing the caller's cache permits fresh encoding.
    Engine identity separates independently configured material owners.
    """

    coefficient_bound = (
        max(abs(complex(scalar).real), abs(complex(scalar).imag)) * scale
    )
    if coefficient_bound >= 2**63:
        raise ValueError(
            'scalar plaintext coefficients exceed the signed integer encoding range; '
            'advance the input to a suitable arithmetic scale before polynomial evaluation'
        )
    cache_key = (
        'bootstrap-scalar',
        engine,
        ciphertext.device,
        ciphertext.data.dtype,
        ciphertext.depth,
        ciphertext.prime_ids,
        scale,
        domain,
        complex(scalar),
    )
    plaintext = _cached_constant(
        constant_cache, cache_key, prepare=prepare_materials
    )
    if not isinstance(plaintext, CompressedPlaintext):
        plaintext = engine.prepare_compressed_plaintext(
            torch.full(
                (1,),
                complex(scalar),
                dtype=torch.complex128,
                device=ciphertext.device,
            ),
            depth=ciphertext.depth,
            scale=scale,
            device=ciphertext.device,
            polynomial_domain=domain,
        )
        if constant_cache is not None:
            constant_cache[cache_key] = plaintext
    return plaintext


def _weighted_scalar_sum(
    arithmetic: Arithmetic,
    terms: Sequence[tuple[Ciphertext, complex]],
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Compute $\sum_j a_jc_j$ with one rescale for a common-depth group."""

    engine = arithmetic.engine

    if not terms:
        raise ValueError('weighted scalar sum requires at least one term')
    depth = terms[0][0].depth
    input_domain = terms[0][0].polynomial_domain
    pending: Ciphertext | None = None
    for ciphertext, scalar in terms:
        if (
            ciphertext.depth != depth
            or ciphertext.polynomial_domain != input_domain
        ):
            raise ValueError(
                'weighted scalar terms require one depth and polynomial domain'
            )
        ciphertext_ntt = (
            ciphertext
            if input_domain == 'ntt'
            else engine.coefficient_domain_to_ntt_domain(ciphertext)
        )
        term = engine.multiply_plaintext(
            ciphertext_ntt,
            _prepare_multiply_scalar(
                arithmetic,
                ciphertext,
                scalar,
                constant_cache=constant_cache,
            ),
        )
        pending = term if pending is None else engine.add(pending, term)
    assert pending is not None
    result = _rescale(engine, pending, retain_ntt=retain_ntt)
    if input_domain == 'ntt' or not retain_ntt:
        return result
    return engine.ntt_domain_to_coefficient_domain(result)


def _add_scalar(
    engine: Engine,
    ciphertext: Ciphertext,
    scalar: complex,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    prepare_materials: bool = True,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Compute $c_{\rm out}=c+a$ without changing CKKS state metadata.

    The scalar is broadcast across semantic `[slot]` and batch axes, encoded at
    the ciphertext's actual scale, and prepared for the ciphertext's current
    coefficient/standard or NTT/Montgomery representation over the same Q
    `prime_ids`. Addition updates only $c_0$. The functional output preserves
    data shape, depth, scale, two components, represented domain, basis, and
    `prime_ids`.
    """

    if not retain_ntt and ciphertext.polynomial_domain == 'ntt':
        ciphertext = engine.ntt_domain_to_coefficient_domain(ciphertext)
    plaintext = _prepare_scalar(
        engine,
        ciphertext,
        scalar,
        ciphertext.scale,
        ciphertext.polynomial_domain,
        constant_cache=constant_cache,
        prepare_materials=prepare_materials,
    )
    return engine.add_plaintext(ciphertext, plaintext)


@dataclass(frozen=True)
class Arithmetic:
    r"""CKKS operations that evaluation components use to reach per-depth target scales.

    Polynomial plans, linear-transform plans and periodic reductions receive an
    ``Arithmetic`` as the first argument of their ``evaluate`` method and
    perform their scalar, constant, product and depth operations through it.
    ``target_scales[d]`` is the target $s_d$ of depth $d$; without supplied
    targets it follows $s_d=\sqrt{\rho_d s_{d+1}}$ backward from
    ``engine.config.scale_at(engine.max_depth)``, where $\rho_d$ is
    ``engine.config.rescale_divisor(d)``. With these targets:

    - ``multiply_scalar``, ``weighted_scalar_sum`` and ``advance_depth`` move
      depth $d\to d+1$ and return actual scale $s_{d+1}$; the plaintext scale
      is $\rho_d s_{d+1}/\Delta$ for input scale $\Delta$
      (``plaintext_scale``);
    - ``multiply_relinearize_rescale`` and ``relinearize(..., rescale=True)``
      move $d\to d+1$ with actual scale $\Delta_{lhs}\Delta_{rhs}/\rho_d$;
    - ``add_scalar`` and ``multiply_by_monomial`` keep depth, scale, domain and
      basis; ``align_depths`` advances the shallower operand.

    Every rescale records its actual quotient scale. ``for_input`` and
    ``scaled_targets`` derive views with other targets for intermediate
    values. ``engine`` exposes the remaining Eager operations (``add``,
    ``rotate``, ``conjugate`` and so on).

    Keys and encoded constants pass through the object, so the same
    ``evaluate`` call serves both preparation and prepared execution: during
    :func:`fhelium.bootstrap.prepare` and :func:`fhelium.algorithms.linear.prepare`
    a relinearization inventory selects and records one key per depth and
    ``constant_cache`` records each scalar plaintext with its first stochastic
    encoding sample; prepared calls read the recorded entries and encode
    nothing. A component therefore obtains keys and constants only through
    its arguments and these methods. Scalar preparation requires
    ``max(abs(real(c)), abs(imag(c))) * scale < 2**63``; addition uses the
    ciphertext scale and multiplication its selected plaintext scale, and
    values outside this bound raise ``ValueError``. ``retain_ntt=False``
    returns coefficient-domain results, and ``basis_conversion`` is forwarded
    to relinearization.
    """

    engine: Engine
    constant_cache: MutableMapping[object, object] | None = None
    retain_ntt: bool = True
    basis_conversion: Literal["exact", "crt", "normalized_crt"] = "exact"
    _prepare_materials: bool = field(default=True, repr=False)
    _scales: tuple[float, ...] | None = field(default=None, repr=False)
    _relinearization_bindings: (
        MutableMapping[int, RelinearizationKey] | None
    ) = field(default=None, repr=False)
    _explicit_relinearization_keys: (
        MutableMapping[int, RelinearizationKey] | None
    ) = field(default=None, repr=False)

    @cached_property
    def target_scales(self) -> tuple[float, ...]:
        r"""Return the selected per-depth arithmetic scales.

        The default selection is computed backward from the terminal default
        scale using $s_d^2/\rho_d=s_{d+1}$. ``for_input`` follows an actual input
        scale forward over the requested depths. ``scaled_targets`` instead
        describes coefficient accumulators multiplied by unscaled basis nodes.
        """

        if self._scales is not None:
            return self._scales
        scales = [self.engine.config.scale_at(self.engine.max_depth)] * (
            self.engine.max_depth + 1
        )
        for depth in reversed(range(self.engine.max_depth)):
            scales[depth] = math.sqrt(
                self.engine.config.rescale_divisor(depth) * scales[depth + 1]
            )
        return tuple(scales)

    def for_input(
        self, ciphertext: Ciphertext, required_depths: int
    ) -> Arithmetic:
        r"""Follow $s_{d+1}=s_d^2/\rho_d$ from the input's actual scale.

        Polynomial basis nodes use this recurrence through their required
        depths; coefficient products may select a different output scale.
        """

        scales = list(self.target_scales)
        scales[ciphertext.depth] = ciphertext.scale
        for depth in range(
            ciphertext.depth, ciphertext.depth + required_depths
        ):
            scales[depth + 1] = (
                scales[depth]
                * scales[depth]
                / self.engine.config.rescale_divisor(depth)
            )
        return replace(self, _scales=tuple(scales))

    def scaled_targets(self, factor: float) -> Arithmetic:
        """Use a common scale factor for polynomial coefficient accumulators."""

        return replace(
            self, _scales=tuple(scale * factor for scale in self.target_scales)
        )

    def plaintext_scale(self, input_scale: float, depth: int) -> float:
        r"""Choose $s_{d+1}\rho_d/\Delta$ for a plaintext product and rescale."""

        divisor = self.engine.config.rescale_divisor(depth)
        target = self.target_scales[depth + 1]
        if target == (input_scale * input_scale) / divisor:
            return input_scale
        return (
            self.target_scales[depth + 1]
            * self.engine.config.rescale_divisor(depth)
            / input_scale
        )

    def multiply_by_monomial(
        self, ciphertext: Ciphertext, exponent: int
    ) -> Ciphertext:
        r"""Return $X^e c(X)\bmod(X^N+1)$ without changing CKKS metadata."""

        return _multiply_by_monomial(
            self.engine,
            ciphertext,
            exponent,
            constant_cache=self.constant_cache,
            prepare_materials=self._prepare_materials,
        )

    def align_depths(
        self, lhs: Ciphertext, rhs: Ciphertext
    ) -> tuple[Ciphertext, Ciphertext]:
        """Advance the shallower value to the deeper value's Q depth."""

        return _align_depths(
            self,
            lhs,
            rhs,
            constant_cache=self.constant_cache,
            retain_ntt=self.retain_ntt,
        )

    def advance_depth(self, ciphertext: Ciphertext) -> Ciphertext:
        """Advance one depth and return at the next target scale."""

        return _advance_depth(
            self,
            ciphertext,
            constant_cache=self.constant_cache,
            retain_ntt=self.retain_ntt,
        )

    def relinearize(
        self,
        ciphertext: Ciphertext,
        key: RelinearizationKey
        | EvaluationKeySet
        | Mapping[int, RelinearizationKey],
        *,
        output_domain: Literal["coefficient", "ntt"] = "coefficient",
        rescale: bool = False,
    ) -> Ciphertext:
        r"""Reduce $c_0+c_1s+c_2s^2$ to two components, optionally rescaling.

        A primitive key is consumed directly. During material preparation an
        inventory selects a key covering the product's Q rows. Prepared execution
        reads its selected per-depth references. The basis conversion and requested
        polynomial representation are forwarded to the Eager operation. Without
        rescaling, Q depth and actual scale are preserved. With ``rescale=True``,
        the operation advances depth once and returns actual scale
        $\Delta/\rho_d$ using a common quotient with the auxiliary product.
        """
        if isinstance(key, EvaluationKeySet):
            selected = key.select_relinearization(
                q_prime_ids=ciphertext.prime_ids,
                p_prime_ids=None,
                ring_dimension=self.engine.config.N,
                dtype=ciphertext.data.dtype,
                device=ciphertext.device,
            )
            if selected is None:
                raise ValueError(
                    f"Missing relinearization key at depth {ciphertext.depth}"
                )
            key = selected
            if (
                self._prepare_materials
                and self._relinearization_bindings is not None
            ):
                self._relinearization_bindings[ciphertext.depth] = key
        else:
            if isinstance(key, Mapping):
                key = key[ciphertext.depth]
            if (
                self._prepare_materials
                and self._explicit_relinearization_keys is not None
            ):
                self._explicit_relinearization_keys[id(key)] = key
        return self.engine.relinearize(
            ciphertext,
            key,
            output_domain=output_domain,
            rescale=rescale,
            basis_conversion=self.basis_conversion,
        )

    def multiply_relinearize_rescale(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        relinearization_key: RelinearizationKey
        | EvaluationKeySet
        | Mapping[int, RelinearizationKey],
    ) -> Ciphertext:
        r"""Return relinearized $c_{lhs}c_{rhs}$ after one rescale."""

        return _multiply_relinearize_rescale(
            self,
            lhs,
            rhs,
            relinearization_key=relinearization_key,
            constant_cache=self.constant_cache,
            retain_ntt=self.retain_ntt,
        )

    def multiply_scalar(
        self, ciphertext: Ciphertext, scalar: complex
    ) -> Ciphertext:
        """Multiply by one encoded scalar, rescale, and return target scale."""

        return _multiply_scalar(
            self,
            ciphertext,
            scalar,
            constant_cache=self.constant_cache,
            retain_ntt=self.retain_ntt,
        )

    def weighted_scalar_sum(
        self, terms: Sequence[tuple[Ciphertext, complex]]
    ) -> Ciphertext:
        r"""Return $\sum_j a_jc_j$ with one shared coefficient rescale."""

        return _weighted_scalar_sum(
            self,
            terms,
            constant_cache=self.constant_cache,
            retain_ntt=self.retain_ntt,
        )

    def add_scalar(self, ciphertext: Ciphertext, scalar: complex) -> Ciphertext:
        """Add a scalar without changing depth, scale, domain, or RNS basis."""

        return _add_scalar(
            self.engine,
            ciphertext,
            scalar,
            constant_cache=self.constant_cache,
            prepare_materials=self._prepare_materials,
            retain_ntt=self.retain_ntt,
        )
