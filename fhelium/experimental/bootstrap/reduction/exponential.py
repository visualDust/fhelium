r"""Exponential-seed periodic reduction with repeated squaring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from fhelium.values import Ciphertext, ConjugationKey, RelinearizationKey
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.polynomial import (
    BalancedPowerEvaluator,
    PolynomialApproximation,
)

if TYPE_CHECKING:
    pass


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

    The ciphertext state, axes, functional behavior, depth transition, and
    depth-dependent scale schedule are the same as for
    `CosineDoubleAngleReduction`. This strategy additionally requires a
    conjugation key. It returns a two-component coefficient-domain,
    standard-RNS Q ciphertext at `ciphertext.depth + required_depths`, with
    the corresponding active `prime_ids` and the arithmetic owner's target scale.
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
        if self.evaluator.required_depths(self.polynomial) <= 0:
            raise ValueError('polynomial evaluator has an invalid depth cost')

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
    def required_depths(self) -> int:
        """Count normalization, polynomial DAG, squarings, and sine scaling."""

        return (
            int(not self.fuse_input_normalization)
            + self.evaluator.required_depths(self.polynomial)
            + self.squaring_iterations
            + 1
        )

    def evaluate(
        self,
        arithmetic: BootstrapArithmetic,
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

        engine = arithmetic.engine
        ops = arithmetic

        if relinearization_key is None:
            raise ValueError(
                'exponential reduction requires relinearization key'
            )
        if conjugation_key is None:
            raise ValueError('exponential reduction requires conjugation key')
        exponential = ciphertext
        if not self.fuse_input_normalization:
            exponential = ops.multiply_scalar(
                exponential,
                1.0 / self.input_bound,
            )
        exponential = self.evaluator.evaluate(
            arithmetic,
            exponential,
            self.polynomial,
            relinearization_key=relinearization_key,
        )
        for _ in range(self.squaring_iterations):
            exponential = ops.multiply_relinearize_rescale(
                exponential,
                exponential,
                relinearization_key=relinearization_key,
            )
        conjugated = engine.conjugate(exponential, conjugation_key)
        sine_times_two_i = engine.subtract(exponential, conjugated)
        return ops.multiply_scalar(sine_times_two_i, -0.5j / math.pi)

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
