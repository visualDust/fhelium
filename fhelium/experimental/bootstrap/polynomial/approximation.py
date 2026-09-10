r"""Polynomial values and numerical Chebyshev interpolation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

PolynomialBasis = Literal['power', 'chebyshev']


@dataclass(frozen=True)
class PolynomialApproximation:
    r"""An immutable polynomial produced independently of its evaluation DAG.

    The coefficient convention is ascending degree. For `basis="power"`,

    $$
    p(x)=\sum_{n=0}^{d}a_nx^n,
    $$

    while `basis="chebyshev"` means

    $$
    p(x)=\sum_{n=0}^{d}a_nT_n(x),\qquad
    T_0(x)=1,\quad T_1(x)=x.
    $$

    `domain=(a, b)` records the physical interval used to design the
    approximation. The evaluator input is nevertheless normalized $x$ unless
    `(a, b) == (-1, 1)`; the caller owns the affine map.

    Attributes:
        basis: Basis in which ``coefficients`` are expressed.
        coefficients: Ascending coefficients: entry ``i`` multiplies either
            $x^i$ or $T_i(x)$.
        domain: Plaintext interval on which the approximation was designed.
        name: Human-readable diagnostic name.
        max_error: Optional sampled or certified approximation error.
    """

    basis: PolynomialBasis
    coefficients: tuple[complex, ...]
    domain: tuple[float, float] = (-1.0, 1.0)
    name: str = 'polynomial'
    max_error: float | None = None

    def __post_init__(self) -> None:
        if self.basis not in {'power', 'chebyshev'}:
            raise ValueError("basis must be 'power' or 'chebyshev'")
        if not self.coefficients:
            raise ValueError('a polynomial needs at least one coefficient')
        lower, upper = self.domain
        if not lower < upper:
            raise ValueError('polynomial domain must have positive width')
        object.__setattr__(
            self,
            'coefficients',
            tuple(complex(value) for value in self.coefficients),
        )
        if self.max_error is not None and self.max_error < 0:
            raise ValueError('max_error cannot be negative')

    @property
    def degree(self) -> int:
        """Return the algebraic degree including trailing zero entries."""

        return len(self.coefficients) - 1

    def evaluate_plaintext(self, values: np.ndarray) -> np.ndarray:
        r"""Evaluate $p(x)$ elementwise without homomorphic arithmetic.

        `values` may have any NumPy-broadcastable shape, which is preserved in
        the output. They are coordinates in the polynomial's basis domain. For a
        Chebyshev approximation created on a physical interval other than
        $[-1,1]$, callers must first apply the same affine normalization
        described by the approximator.  This method is a numerical oracle; it
        does not model CKKS rounding or depth consumption.
        """

        x = np.asarray(values)
        if self.basis == 'power':
            return np.polynomial.polynomial.polyval(x, self.coefficients)
        return np.polynomial.chebyshev.chebval(x, self.coefficients)


@dataclass(frozen=True)
class ChebyshevInterpolator:
    r"""Fit a degree-limited Chebyshev series at first-kind nodes.

    `degree` controls both the number of interpolation nodes and the highest
    returned term $T_d$. `error_samples` controls only the dense grid
    used to report `max_error`; that sampled value is not a proof of the
    uniform error between grid points.
    """

    degree: int
    error_samples: int = 8193

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError('degree must be positive')
        if self.error_samples < 3:
            raise ValueError('error_samples must be at least three')

    def approximate(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        *,
        domain: tuple[float, float] = (-1.0, 1.0),
        name: str = 'polynomial',
    ) -> PolynomialApproximation:
        r"""Interpolate after mapping physical $t\in[a,b]$ to $x\in[-1,1]$.

        If `domain=(a, b)`, first-kind nodes $x_j$ are evaluated physically at

        $$
        t_j=a+(x_j+1)(b-a)/2.
        $$

        Coefficients in the returned object are functions of normalized $x$,
        not directly of physical $t$. The consuming evaluator must therefore
        normalize its ciphertext to $x$ and account for any
        depth spent on that affine map.

        `max_error` is measured on `error_samples` equally spaced normalized
        coordinates after fitting. Approximation runs on CPU binary64/complex128
        arrays and returns no encrypted tensor.
        """

        lower, upper = domain
        if not lower < upper:
            raise ValueError('approximation domain must have positive width')

        def normalized_function(value: np.ndarray) -> np.ndarray:
            physical = lower + (value + 1.0) * (upper - lower) / 2.0
            return np.asarray(function(physical))

        order = self.degree + 1
        nodes = np.polynomial.chebyshev.chebpts1(order)
        samples = normalized_function(nodes)
        vandermonde = np.polynomial.chebyshev.chebvander(nodes, self.degree)
        coefficients = vandermonde.T @ samples
        coefficients[0] /= order
        coefficients[1:] /= 0.5 * order
        grid = np.linspace(-1.0, 1.0, self.error_samples)
        expected = normalized_function(grid)
        actual = np.polynomial.chebyshev.chebval(grid, coefficients)
        return PolynomialApproximation(
            basis='chebyshev',
            coefficients=tuple(complex(value) for value in coefficients),
            domain=domain,
            name=name,
            max_error=float(np.max(np.abs(actual - expected))),
        )
