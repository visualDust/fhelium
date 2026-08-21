r"""Periodic nonlinear algorithms used to remove the approximate CKKS carry.

Let $r$ denote a raw real branch coordinate after CoeffsToSlots and branch
splitting. For a configured `input_bound` $B$, the polynomial coordinate is
$x=r/B$ and must lie in $[-1,1]$. The built-in components approximate

$$
\rho_B(x)=\frac{\sin(\pi Bx)}{\pi}
          =\frac{\sin(\pi r)}{\pi}.
$$

When $r=2k+\epsilon$ for an integer carry $k$ and small residual $\epsilon$,
$\rho_B(x)$ approximates $\epsilon$. Implementations own the approximation,
homomorphic evaluation schedule, and function composition. The full bootstrap
composition consumes their declared scale, range, and level behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from fhelium.core import Ciphertext, ConjugationKey, RelinearizationKey
from fhelium.experimental.bootstrap._polynomial import (
    BalancedPowerEvaluator,
    PolynomialApproximation,
)

if TYPE_CHECKING:
    from fhelium.engine.ckks_engine import CkksEngine


@dataclass(frozen=True)
class CosineDoubleAngleReduction:
    r"""Reduce one real branch with a cosine seed and double-angle chain.

    `input_bound` is $B$, a positive power of two bounding the raw coordinate
    $r$. The fitted polynomial always consumes the normalized coordinate
    $x=r/B\in[-1,1]$. With $R=$ `double_angle_iterations`, the seed is

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
    Approximation and homomorphic evaluation are separate components.

    `reference(values)` always takes normalized $x$ coordinates and applies no
    input division. `evaluate(...)` takes raw $r$ when
    `fuse_input_normalization=False`, dividing by $B$ homomorphically. When
    fusion is enabled, the caller must already have divided by $B$; the
    built-in full-slot callable folds that factor into CoeffsToSlots.

    The homomorphic input is a two-component coefficient-domain, standard-RNS
    Q ciphertext with payload axes
    `[component, *batch, limb, coefficient]`, active `prime_ids`, and
    actual scale near `engine.config.default_scale`. Evaluation is functional.
    Each multiplication returns to coefficient-domain standard RNS, drops one
    leading Q row on rescale, and is explicitly reinterpreted at the default
    scale by the bootstrap's private fixed-scale policy. The output has the
    same batch and component axes, Q basis, and context; its level advances by
    `required_levels`, its limb axis contains the corresponding suffix of
    `prime_ids`, and its actual scale is the default scale.

    More iterations reduce the seed frequency but each iteration costs one
    ciphertext multiplication, relinearization, and rescale level. Neither the
    class nor `evaluate` measures the encrypted branch range; the caller must
    establish $|r|\le B$.
    """

    requires_relinearization = True

    input_bound: int
    double_angle_iterations: int
    approximator: Any
    evaluator: Any
    fuse_input_normalization: bool = False
    maximum_plaintext_error: float = 1e-3

    def __post_init__(self) -> None:
        if self.input_bound <= 0 or (self.input_bound & (self.input_bound - 1)):
            raise ValueError('input_bound must be a positive power of two')
        maximum_iterations = int(math.log2(self.input_bound))
        if not 0 < self.double_angle_iterations <= maximum_iterations:
            raise ValueError(
                f'double_angle_iterations must lie in [1, {maximum_iterations}]'
            )
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
    def polynomial(self) -> PolynomialApproximation:
        r"""Fit the low-frequency seed in normalized coordinate $x$.

        For $R=$ `double_angle_iterations`, the returned Chebyshev or power
        series approximates

        $$
        \pi^{-1/2^R}
        \cos\left(\frac{\pi Bx}{2^R}-\frac{\pi}{2^{R+1}}\right)
        \quad\text{on }[-1,1].
        $$

        Its coefficient basis is selected by `approximator`; coefficients use
        that basis's ascending-degree convention.
        """

        phase_scale = 2**self.double_angle_iterations
        amplitude = math.pi ** (-1.0 / phase_scale)
        frequency = math.pi * self.input_bound / phase_scale
        phase_shift = -math.pi / (2.0 * phase_scale)
        return self.approximator.approximate(
            lambda value: amplitude * np.cos(frequency * value + phase_shift),
            domain=(-1.0, 1.0),
            name='periodic_cosine_seed',
        )

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
        r"""Return $B$ for fused $x=r/B$, or $1$ for explicit division."""

        return float(self.input_bound) if self.fuse_input_normalization else 1.0

    @property
    def required_levels(self) -> int:
        """Count explicit division, polynomial depth, and recurrence depth."""

        return (
            int(not self.fuse_input_normalization)
            + self.evaluator.required_levels(self.polynomial)
            + self.double_angle_iterations
        )

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        *,
        relinearization_key: RelinearizationKey | None,
        conjugation_key: ConjugationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate $\rho_B$ homomorphically under the class coordinate convention.

        With non-fused normalization the input represents raw $r$ and the first
        level computes $x=r/B$. With fused normalization it already represents
        $x$. Each recurrence iteration squares and relinearizes the current
        ciphertext, rescales one Q row, doubles it by addition, and subtracts
        the stage constant. The output represents
        $\rho_B(x)=\sin(\pi r)/\pi$ at
        `ciphertext.level + required_levels`.
        Input storage is not mutated or aliased by the result.
        """

        from fhelium.experimental.bootstrap import _ops

        del conjugation_key
        if relinearization_key is None:
            raise ValueError('cosine reduction requires relinearization key')
        reduced = ciphertext
        if not self.fuse_input_normalization:
            reduced = _ops._multiply_scalar(
                engine,
                reduced,
                1.0 / self.input_bound,
            )
        reduced = self.evaluator.evaluate(
            engine,
            reduced,
            self.polynomial,
            relinearization_key=relinearization_key,
        )
        for iteration in range(1, self.double_angle_iterations + 1):
            reduced = _ops._multiply_relinearize_rescale(
                engine,
                reduced,
                reduced,
                relinearization_key=relinearization_key,
            )
            reduced = engine.add(reduced, reduced)
            recurrence_constant = math.pi ** (
                -1.0 / (2 ** (self.double_angle_iterations - iteration))
            )
            reduced = _ops._add_scalar(engine, reduced, -recurrence_constant)
        return reduced

    def reference(self, values: np.ndarray) -> np.ndarray:
        r"""Evaluate the plaintext oracle on normalized coordinates $x$.

        `values` may have any NumPy-broadcastable shape and that shape is
        preserved. Unlike non-fused `evaluate`, this method never divides by
        $B$; pass raw coordinates as `values / input_bound`. The target is
        $\rho_B(x)=\sin(\pi Bx)/\pi$ and CKKS rounding is not modeled.
        """

        reduced = self.polynomial.evaluate_plaintext(np.asarray(values))
        for iteration in range(1, self.double_angle_iterations + 1):
            reduced = 2.0 * reduced * reduced - math.pi ** (
                -1.0 / (2 ** (self.double_angle_iterations - iteration))
            )
        return reduced


@dataclass(frozen=True)
class ExponentialSquaringReduction:
    r"""Reduce one real branch through a truncated exponential and squaring.

    Let $B=$ `input_bound`, $x=r/B\in[-1,1]$, and $K=\log_2 B$. A power-basis
    polynomial first approximates

    $$
    z_0(x)=\exp(i\pi x).
    $$

    Repeated squaring computes $z_K\mathrel{\approx}\exp(i\pi Bx)$, and

    $$
    \frac{z_K-\overline{z_K}}{2i\pi}
    \mathrel{\approx}\frac{\sin(\pi Bx)}{\pi}=\rho_B(x).
    $$

    `reference(values)` always consumes normalized $x$. `evaluate(...)`
    consumes raw $r$ and divides by $B$ when `fuse_input_normalization=False`;
    with fusion enabled its caller must provide $x$ directly. The raw range
    $|r|\le B$ is a caller precondition and is not inspected in ciphertexts.

    The ciphertext state, axes, functional behavior, level transition, and
    private default-scale reinterpretation are the same as for
    `CosineDoubleAngleReduction`. This strategy additionally requires a
    conjugation key. It returns a two-component coefficient-domain,
    standard-RNS Q ciphertext at `ciphertext.level + required_levels`, with
    the corresponding active `prime_ids` and actual default scale.
    """

    requires_relinearization = True

    input_bound: int
    degree: int
    evaluator: Any = BalancedPowerEvaluator()
    fuse_input_normalization: bool = False

    def __post_init__(self) -> None:
        if self.input_bound <= 0 or (self.input_bound & (self.input_bound - 1)):
            raise ValueError('input_bound must be a positive power of two')
        if self.degree < 1:
            raise ValueError('degree must be positive')
        if self.evaluator.required_levels(self.polynomial) <= 0:
            raise ValueError('polynomial evaluator has an invalid level cost')

    @cached_property
    def polynomial(self) -> PolynomialApproximation:
        r"""Return ascending power coefficients for $\exp(i\pi x)$.

        Entry $n$ is $(i\pi)^n/n!$, so the stored polynomial is
        $\sum_{n=0}^{d}(i\pi)^n x^n/n!$ in normalized coordinate $x$.
        """

        frequency = 1j * math.pi
        return PolynomialApproximation(
            basis='power',
            coefficients=tuple(
                frequency**exponent / math.factorial(exponent)
                for exponent in range(self.degree + 1)
            ),
            name='periodic_exponential_seed',
        )

    @property
    def squaring_iterations(self) -> int:
        r"""Return $\log_2 B$, the repeated-squaring count."""

        return int(math.log2(self.input_bound))

    @property
    def fused_input_divisor(self) -> float:
        r"""Return $B$ for fused $x=r/B$, or $1$ for explicit division."""

        return float(self.input_bound) if self.fuse_input_normalization else 1.0

    @property
    def required_levels(self) -> int:
        """Count normalization, polynomial DAG, squarings, and sine scaling."""

        return (
            int(not self.fuse_input_normalization)
            + self.evaluator.required_levels(self.polynomial)
            + self.squaring_iterations
            + 1
        )

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        *,
        relinearization_key: RelinearizationKey | None,
        conjugation_key: ConjugationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate $\rho_B$ homomorphically under the class coordinate convention.

        Non-fused evaluation first maps raw $r$ to $x=r/B$. After evaluating
        the power series, $\log_2 B$ ciphertext squarings restore frequency.
        Finally $(z-\overline z)/(2i\pi)$ isolates and scales the sine branch.
        The result represents $\sin(\pi r)/\pi$; input storage is unchanged.
        """

        from fhelium.experimental.bootstrap import _ops

        if relinearization_key is None:
            raise ValueError(
                'exponential reduction requires relinearization key'
            )
        if conjugation_key is None:
            raise ValueError('exponential reduction requires conjugation key')
        exponential = ciphertext
        if not self.fuse_input_normalization:
            exponential = _ops._multiply_scalar(
                engine,
                exponential,
                1.0 / self.input_bound,
            )
        exponential = self.evaluator.evaluate(
            engine,
            exponential,
            self.polynomial,
            relinearization_key=relinearization_key,
        )
        for _ in range(self.squaring_iterations):
            exponential = _ops._multiply_relinearize_rescale(
                engine,
                exponential,
                exponential,
                relinearization_key=relinearization_key,
            )
        conjugated = engine.conjugate(exponential, conjugation_key)
        sine_times_two_i = engine.subtract(exponential, conjugated)
        return _ops._multiply_scalar(engine, sine_times_two_i, -0.5j / math.pi)

    def reference(self, values: np.ndarray) -> np.ndarray:
        r"""Evaluate the plaintext oracle on normalized coordinates $x$.

        The input shape is preserved. This method never divides by $B$; pass
        raw coordinates as `values / input_bound`. It models polynomial
        truncation and repeated squaring but not CKKS error.
        """

        exponential = self.polynomial.evaluate_plaintext(np.asarray(values))
        for _ in range(self.squaring_iterations):
            exponential = exponential * exponential
        return np.imag(exponential) / math.pi
