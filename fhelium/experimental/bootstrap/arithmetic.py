r"""CKKS arithmetic and depth-dependent scale targets for Bootstrap components."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
import math

import torch

from fhelium.eager import Engine
from fhelium.values import Ciphertext, Plaintext, RelinearizationKey


def _multiply_by_monomial(
    engine: Engine,
    ciphertext: Ciphertext,
    exponent: int,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
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
    data = ciphertext.data.clone()
    if shift:
        original = data.clone()
        data[..., shift:] = original[..., : degree - shift]
        data[..., :shift] = -original[..., degree - shift :]
    if sign < 0:
        data.neg_()
    cache_key = (
        'bootstrap-monomial-moduli',
        ciphertext.device,
        ciphertext.prime_ids,
    )
    moduli = None if constant_cache is None else constant_cache.get(cache_key)
    if not isinstance(moduli, torch.Tensor):
        moduli = torch.tensor(
            [
                engine._rns_context_for(
                    ciphertext.device
                ).montgomery_parameters.moduli[index]
                for index in ciphertext.prime_ids
            ],
            dtype=data.dtype,
            device=data.device,
        )
        if constant_cache is not None:
            constant_cache[cache_key] = moduli
    moduli = moduli.view(*([1] * (data.ndim - 2)), -1, 1)
    return ciphertext.with_data(torch.remainder(data, moduli))


def _align_depths(
    arithmetic: BootstrapArithmetic,
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
    arithmetic: BootstrapArithmetic,
    ciphertext: Ciphertext,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Preserve the message while advancing one Q group.

    Multiplication by an encoded one uses plaintext scale
    $M_d s_{d+1}/\Delta(c)$. Rescaling therefore returns the message at
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
    arithmetic: BootstrapArithmetic,
    lhs: Ciphertext,
    rhs: Ciphertext,
    *,
    relinearization_key: RelinearizationKey,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Multiply aligned ciphertexts, relinearize, and remove one Q group.

    Component convolution produces three components at actual scale
    $\Delta_l\Delta_r$. Relinearization restores two components; rescale
    divides the scale by $M_d$. When both operands have scheduled scale
    $s_d$, the result has $s_{d+1}=s_d^2/M_d$. No scale metadata is replaced.
    The result retains NTT form when requested by ``retain_ntt`` and the
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
    if retain_ntt and input_domain == 'ntt':
        product = engine.relinearize(
            product,
            relinearization_key,
            output_domain='ntt',
        )
    else:
        product = engine.relinearize(product, relinearization_key)
    return _rescale(engine, product, retain_ntt=retain_ntt)


def _rescale(
    engine: Engine,
    ciphertext: Ciphertext,
    *,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Remove one Q group and retain the arithmetic result's actual scale.

    The quotient is rounded by the Engine's ordinary rescale operation.
    ``retain_ntt=False`` converts an NTT input to coefficient form first.
    """

    if not retain_ntt and ciphertext.polynomial_domain == 'ntt':
        ciphertext = engine.ntt_domain_to_coefficient_domain(ciphertext)
    return engine.rescale_to_next_depth(ciphertext)


def _multiply_scalar(
    arithmetic: BootstrapArithmetic,
    ciphertext: Ciphertext,
    scalar: complex,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
    retain_ntt: bool = True,
) -> Ciphertext:
    r"""Multiply the represented message by $a$ and advance one Q group.

    The scalar plaintext uses scale $M_d s_{d+1}/\Delta(c)$ so that the
    resulting quotient has scale $s_{d+1}$. The operation preserves the
    message apart from encoding and quotient rounding. It retains the
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
    arithmetic: BootstrapArithmetic,
    ciphertext: Ciphertext,
    scalar: complex,
    *,
    constant_cache: MutableMapping[object, object] | None = None,
) -> Plaintext:
    """Prepare one scalar plaintext for a pending multiplication."""

    engine = arithmetic.engine

    plaintext_scale = arithmetic.plaintext_scale(ciphertext.scale, ciphertext.depth)
    coefficient_bound = max(abs(complex(scalar).real), abs(complex(scalar).imag)) * plaintext_scale
    if coefficient_bound >= 2**63:
        raise ValueError(
            'scalar plaintext coefficients exceed the signed integer encoding range; '
            'advance the input to a suitable arithmetic scale before polynomial evaluation'
        )
    cache_key = (
        'bootstrap-multiply-scalar',
        ciphertext.device,
        ciphertext.depth,
        plaintext_scale,
        complex(scalar),
    )
    plaintext = (
        None if constant_cache is None else constant_cache.get(cache_key)
    )
    if not isinstance(plaintext, Plaintext):
        plaintext = engine.prepare_plaintext_for_multiplication(
            engine.encode(
                torch.full(
                    (engine.num_slots,),
                    complex(scalar),
                    dtype=torch.complex128,
                    device=ciphertext.device,
                ),
                depth=ciphertext.depth,
                scale=plaintext_scale,
                device=ciphertext.device,
            ),
            modulus_basis='Q',
        )
        if constant_cache is not None:
            constant_cache[cache_key] = plaintext
    return plaintext


def _weighted_scalar_sum(
    arithmetic: BootstrapArithmetic,
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
    cache_key = (
        'bootstrap-add-scalar',
        ciphertext.device,
        ciphertext.depth,
        ciphertext.scale,
        ciphertext.polynomial_domain,
        complex(scalar),
    )
    plaintext = (
        None if constant_cache is None else constant_cache.get(cache_key)
    )
    if not isinstance(plaintext, Plaintext):
        encoded = engine.encode(
            torch.full(
                (engine.num_slots,),
                complex(scalar),
                dtype=torch.complex128,
                device=ciphertext.device,
            ),
            depth=ciphertext.depth,
            scale=ciphertext.scale,
            device=ciphertext.device,
        )
        if retain_ntt:
            plaintext = engine.prepare_plaintext_for_addition(
                encoded,
                modulus_basis='Q',
                polynomial_domain=ciphertext.polynomial_domain,
            )
        else:
            plaintext = engine.prepare_plaintext_for_addition(
                encoded,
                modulus_basis='Q',
            )
        if constant_cache is not None:
            constant_cache[cache_key] = plaintext
    return engine.add_plaintext(ciphertext, plaintext)


@dataclass(frozen=True)
class BootstrapArithmetic:
    r"""Execute Bootstrap CKKS arithmetic with depth-dependent scale targets.

    ``target_scales`` gives an inspectable multiplication schedule derived
    from Q group products. Scalar and diagonal products select their plaintext
    scales to reach that schedule; ciphertext products retain actual scales.
    ``constant_cache`` retains prepared scalar and monomial plaintexts for
    reuse across evaluation calls.
    ``retain_ntt=False`` selects coefficient-domain results for callers that
    compose Bootstrap without NTT-retaining Eager operations.
    FullSlotBootstrap creates this owner internally; custom compositions pass
    it to linear, polynomial, and periodic-reduction evaluators.
    """

    engine: Engine
    constant_cache: MutableMapping[object, object] | None = None
    retain_ntt: bool = True
    _scales: tuple[float, ...] | None = field(default=None, repr=False)

    @cached_property
    def target_scales(self) -> tuple[float, ...]:
        r"""Return the selected per-depth arithmetic scales.

        The default selection is computed backward from the terminal default
        scale using $s_d^2/M_d=s_{d+1}$. ``for_input`` follows an actual input
        scale forward over the requested depths. ``scaled_targets`` instead
        describes coefficient accumulators multiplied by unscaled basis nodes.
        """

        if self._scales is not None:
            return self._scales
        scales = [self.engine.config.default_scale] * (self.engine.max_depth + 1)
        for depth in reversed(range(self.engine.max_depth)):
            scales[depth] = math.sqrt(
                self.engine.config.rescale_divisor(depth) * scales[depth + 1]
            )
        return tuple(scales)

    def for_input(
        self, ciphertext: Ciphertext, required_depths: int
    ) -> BootstrapArithmetic:
        r"""Follow $s_{d+1}=s_d^2/M_d$ from the input's actual scale.

        Polynomial basis nodes use this recurrence through their required
        depths; coefficient products may select a different output scale.
        """

        scales = list(self.target_scales)
        scales[ciphertext.depth] = ciphertext.scale
        for depth in range(ciphertext.depth, ciphertext.depth + required_depths):
            scales[depth + 1] = (
                scales[depth] * scales[depth]
                / self.engine.config.rescale_divisor(depth)
            )
        return replace(self, _scales=tuple(scales))

    def scaled_targets(self, factor: float) -> BootstrapArithmetic:
        """Use a common scale factor for polynomial coefficient accumulators."""

        return replace(self, _scales=tuple(scale * factor for scale in self.target_scales))

    def plaintext_scale(self, input_scale: float, depth: int) -> float:
        r"""Choose $s_{d+1}M_d/\Delta$ for a plaintext product and rescale."""

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
        """Advance one Q group and return at the next target scale."""

        return _advance_depth(
            self,
            ciphertext,
            constant_cache=self.constant_cache,
            retain_ntt=self.retain_ntt,
        )

    def multiply_relinearize_rescale(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        relinearization_key: RelinearizationKey,
    ) -> Ciphertext:
        r"""Return relinearized $c_{lhs}c_{rhs}$ after one Q rescale."""

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
            retain_ntt=self.retain_ntt,
        )
