r"""Cosine-seed periodic reduction with double-angle reconstruction."""

from __future__ import annotations

from collections.abc import Mapping

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fhelium.values import (
    EvaluationKeySet,
    Ciphertext,
    RelinearizationKey,
)
from fhelium.algorithms._arithmetic import Arithmetic
from fhelium.algorithms.polynomial import Polynomial

if TYPE_CHECKING:
    pass


def _bessel_first_kind(order: int, x: float) -> np.ndarray:
    r"""Return $J_0(x),\ldots,J_{order}(x)$ by Miller's backward recurrence.

    $J_{n-1}=(2n/x)J_n-J_{n+1}$ runs downward from an order well above
    $\max(order, x)$ and is normalized by $J_0+2\sum_{k\ge1}J_{2k}=1$.
    """

    start = 2 * ((max(order, int(x)) + 40) // 2)
    values = np.zeros(start + 2)
    values[start] = 1e-300
    for n in range(start, 0, -1):
        values[n - 1] = 2.0 * n / x * values[n] - values[n + 1]
        if abs(values[n - 1]) > 1e250:
            values[n - 1 :] *= 1e-250
    values /= values[0] + 2.0 * values[2::2].sum()
    return values[: order + 1]


@dataclass(frozen=True)
class CosineExpansion:
    r"""Truncated Chebyshev expansion of $A\cos(az+\varphi)$ on $[-1,1]$.

    The Jacobi–Anger identities
    $\cos(a\cos\theta)=J_0(a)+2\sum_{k\ge1}(-1)^kJ_{2k}(a)\cos2k\theta$ and
    $\sin(a\cos\theta)=2\sum_{k\ge0}(-1)^kJ_{2k+1}(a)\cos(2k+1)\theta$ give
    the coefficients $c_n=A\epsilon_nJ_n(a)\,\Re\big(i^ne^{i\varphi}\big)$ of
    $T_n(z)$ for $n\le$ ``degree``, with $\epsilon_0=1$ and $\epsilon_n=2$
    otherwise; the truncation error is at most $2A\sum_{n>degree}|J_n(a)|$.
    :class:`CosineDoubleAngle` evaluates this series for its seed.
    """

    degree: int

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError('degree must be nonnegative')

    def cosine(
        self, amplitude: float, frequency: float, phase: float
    ) -> Polynomial:
        bessel = _bessel_first_kind(self.degree, frequency)
        rotation = np.exp(1j * phase) * 1j ** np.arange(self.degree + 1)
        coefficients = amplitude * bessel * rotation.real
        coefficients[1:] *= 2.0
        return Polynomial(
            'chebyshev',
            tuple(float(value) for value in coefficients),
            name='periodic_cosine_seed',
        )


@dataclass(frozen=True)
class CosineDoubleAngle:
    r"""Reduce one real branch with a cosine seed and double-angle chain.

    `input_bound` is the positive integer $B$ bounding the raw coordinate
    $r$. Reduction uses normalized coordinate $x=r/B\in[-1,1]$;
    ``seed_shift`` places its seed polynomial at a shifted coordinate.
    With $R=$ `double_angle_iterations`, the unshifted seed is

    $$
    z_0(x)\mathrel{\approx}
    \pi^{-1/2^R}
    \cos\left(\frac{\pi Bx}{2^R}-\frac{\pi}{2^{R+1}}\right),
    $$

    and iteration $j=1,\ldots,R$ computes

    $$
    z_j=2z_{j-1}^2-\pi^{-1/2^{R-j}}.
    $$

    Thus $z_R\mathrel{\approx}\rho_B(x)=\sin(\pi Bx)/\pi$.
    ``seed_shift=c`` evaluates the seed at z=x+c and subtracts the same shift
    in its fitted phase. The periodic map is unchanged. For c=-1/(2B), the
    fitted seed is even; its odd Chebyshev coefficients are zero.
    Approximation and homomorphic evaluation are separate components.

    `reference(values)` always takes normalized $x$ coordinates and applies no
    input division. `evaluate(...)` takes raw $r$ when
    `fuse_input_normalization=False`, dividing by $B$ homomorphically. When
    fusion is enabled, the caller must already have divided by $B$; the
    built-in full-slot callable folds that factor into CoeffsToSlots.

    The homomorphic input is a two-component Q ciphertext in either
    coefficient/standard or NTT/Montgomery representation, with payload axes
    `[component, *batch, limb, coefficient]`, active `prime_ids`, and
    actual scale matching the arithmetic owner's target at the input depth.
    Evaluation is functional.
    With ``retain_ntt=True``, a coefficient input is transformed once and the
    polynomial and double-angle chain retain NTT/Montgomery form until one
    final inverse transform. Otherwise each multiplication returns to the
    input representation. Rescale removes one Q group and divides actual scale by its product.
    The output uses the input representation, batch and component axes, Q
    basis, and context; its depth advances by `required_depths`, its limb axis
    contains the corresponding suffix of `prime_ids`, and its actual scale follows the arithmetic owner's depth schedule.

    More iterations reduce the seed frequency but each iteration costs one
    ciphertext multiplication, relinearization, and rescale depth. Neither the
    class nor `evaluate` measures the encrypted branch range; the caller must
    establish $|r|\le B$.
    """

    requires_relinearization = True

    input_bound: int
    double_angle_iterations: int
    approximator: Any
    polynomial_plan: Any
    fuse_input_normalization: bool = False
    retain_ntt: bool = False
    maximum_plaintext_error: float = 1e-3
    seed_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.input_bound <= 0:
            raise ValueError('input_bound must be positive')
        if self.double_angle_iterations <= 0:
            raise ValueError('double_angle_iterations must be positive')
        if self.maximum_plaintext_error <= 0:
            raise ValueError('maximum_plaintext_error must be positive')
        if self.polynomial.basis not in {'power', 'chebyshev'}:
            raise ValueError('unsupported approximation basis')
        if self.approximation_error > self.maximum_plaintext_error:
            raise ValueError(
                'modular-reduction approximation is numerically insufficient: '
                f'{self.approximation_error:.3g} > '
                f'{self.maximum_plaintext_error:.3g}'
            )

    @cached_property
    def polynomial(self) -> Polynomial:
        r"""Fit the low-frequency seed in normalized coordinate $x$.

        For $R=$ `double_angle_iterations`, the returned Chebyshev or power
        series in z=x+seed_shift approximates

        $$
        \pi^{-1/2^R}
        \cos\left(\frac{\pi B(z-c)}{2^R}-\frac{\pi}{2^{R+1}}\right)
        \quad\text{on }[-1,1].
        $$

        Its coefficient basis is selected by `approximator`; coefficients use
        that basis's ascending-degree convention. A :class:`CosineExpansion`
        approximator supplies the truncated Chebyshev expansion of this cosine.
        """

        phase_scale = 2**self.double_angle_iterations
        amplitude = math.pi ** (-1.0 / phase_scale)
        frequency = math.pi * self.input_bound / phase_scale
        phase_shift = (
            -math.pi / (2.0 * phase_scale) - frequency * self.seed_shift
        )
        if isinstance(self.approximator, CosineExpansion):
            polynomial = self.approximator.cosine(
                amplitude, frequency, phase_shift
            )
        else:
            polynomial = self.approximator.approximate(
                lambda value: amplitude
                * np.cos(frequency * value + phase_shift),
                domain=(-1.0, 1.0),
                name='periodic_cosine_seed',
            )
        if phase_shift == 0.0:
            polynomial = replace(
                polynomial,
                coefficients=tuple(
                    value if index % 2 == 0 else 0.0
                    for index, value in enumerate(polynomial.coefficients)
                ),
            )
        return polynomial

    @cached_property
    def approximation_error(self) -> float:
        r"""Sample $|\operatorname{reference}(x)-\rho_B(x)|$ on a fixed grid.

        The grid spans normalized $x\in[-1,1]$. This is a fail-fast
        construction check, not a certified supremum bound and not a CKKS
        evaluation-error measurement.
        """

        grid = np.linspace(-1.0, 1.0, 8193)
        actual = self.reference(grid)
        expected = np.sin(math.pi * self.input_bound * grid) / math.pi
        return float(np.max(np.abs(actual - expected)))

    @property
    def fused_input_divisor(self) -> float:
        r"""Return $B$ when normalization is fused into the input transform, or $1$ when evaluation divides by $B$."""

        return float(self.input_bound) if self.fuse_input_normalization else 1.0

    @property
    def required_depths(self) -> int:
        """Count normalization, polynomial-evaluation, and recurrence depths."""

        return (
            int(not self.fuse_input_normalization)
            + self.polynomial_plan.required_depths(self.polynomial)
            + self.double_angle_iterations
        )

    def relinearization_depths(
        self, *, input_depth: int = 0
    ) -> tuple[int, ...] | None:
        """Compose normalization, polynomial products and double-angle products."""
        declare = getattr(self.polynomial_plan, "relinearization_depths", None)
        if not callable(declare):
            return None
        polynomial_start = input_depth + int(not self.fuse_input_normalization)
        depths = cast(
            tuple[int, ...] | None,
            declare(self.polynomial, input_depth=polynomial_start),
        )
        if depths is None:
            return None
        recurrence_start = polynomial_start + self.polynomial_plan.required_depths(
            self.polynomial
        )
        return tuple(
            sorted(
                set(depths)
                | set(
                    range(
                        recurrence_start,
                        recurrence_start + self.double_angle_iterations,
                    )
                )
            )
        )

    def evaluate(
        self,
        arithmetic: Arithmetic,
        ciphertext: Ciphertext,
        *,
        relinearization_key: RelinearizationKey
        | EvaluationKeySet
        | Mapping[int, RelinearizationKey]
        | None,
    ) -> Ciphertext:
        r"""Evaluate $\rho_B$ homomorphically under the class coordinate convention.

        With non-fused normalization the input represents raw $r$ and the first
        depth computes $x=r/B$. With fused normalization it already represents
        $x$. Each recurrence iteration squares and relinearizes the current
        ciphertext, rescales once, doubles it by addition, and subtracts
        the stage constant. The output represents
        $\rho_B(x)=\sin(\pi r)/\pi$ at
        `ciphertext.depth + required_depths`.
        Input storage is not mutated or aliased by the result.
        """

        engine = arithmetic.engine
        ops = arithmetic

        if relinearization_key is None:
            raise ValueError('cosine reduction requires relinearization key')
        input_domain = ciphertext.polynomial_domain
        reduced = ciphertext
        if self.retain_ntt and input_domain == 'coefficient':
            reduced = engine.coefficient_domain_to_ntt_domain(reduced)
        if not self.fuse_input_normalization:
            reduced = ops.multiply_scalar(
                reduced,
                1.0 / self.input_bound,
            )
        if self.seed_shift:
            reduced = ops.add_scalar(reduced, self.seed_shift)
        reduced = self.polynomial_plan.evaluate(
            arithmetic,
            reduced,
            self.polynomial,
            relinearization_key=relinearization_key,
        )
        for iteration in range(1, self.double_angle_iterations + 1):
            reduced = ops.multiply_relinearize_rescale(
                reduced,
                reduced,
                relinearization_key=relinearization_key,
            )
            reduced = engine.add(reduced, reduced)
            recurrence_constant = math.pi ** (
                -1.0 / (2 ** (self.double_angle_iterations - iteration))
            )
            reduced = ops.add_scalar(reduced, -recurrence_constant)
        if self.retain_ntt and input_domain == 'coefficient':
            return engine.ntt_domain_to_coefficient_domain(reduced)
        return reduced

    def reference(self, values: np.ndarray) -> np.ndarray:
        r"""Evaluate the plaintext oracle on normalized coordinates $x$.

        `values` may have any NumPy-broadcastable shape and that shape is
        preserved. Unlike non-fused `evaluate`, this method never divides by
        $B$; pass raw coordinates as `values / input_bound`. The target is
        $\rho_B(x)=\sin(\pi Bx)/\pi$ and CKKS rounding is not modeled.
        """

        reduced = self.polynomial.evaluate_plaintext(
            np.asarray(values) + self.seed_shift
        )
        for iteration in range(1, self.double_angle_iterations + 1):
            reduced = 2.0 * reduced * reduced - math.pi ** (
                -1.0 / (2 ** (self.double_angle_iterations - iteration))
            )
        return reduced
