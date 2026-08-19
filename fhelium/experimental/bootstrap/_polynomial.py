r"""Separate polynomial design from homomorphic polynomial execution.

An approximator answers the numerical question "which coefficients represent
this function on this interval?"  An evaluator answers the evaluation question
"which multiplication DAG evaluates those coefficients, and how many CKKS
levels does it consume?"  Keeping the two objects separate lets the same
approximation be evaluated with different addition chains without refitting it.

The stored basis coordinate is always named $x$. For a fitted physical domain
$[a,b]$, `ChebyshevInterpolator` first maps a physical coordinate $t$ to

$$
x=\frac{2t-(a+b)}{b-a}\in[-1,1].
$$

The returned coefficients are functions of $x$, not $t$. Neither plaintext nor
homomorphic evaluator performs this affine map implicitly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from fhelium.core import Ciphertext, RelinearizationKey

if TYPE_CHECKING:
    from fhelium.engine.ckks_engine import CkksEngine

PolynomialBasis = Literal['power', 'chebyshev']


def _inventory(
    *,
    ciphertext_multiplications: int,
    coefficient_multiplications: int,
    alignment_multiplications: int,
) -> dict[str, int]:
    """Return one JSON-compatible multiplication inventory."""

    total = (
        ciphertext_multiplications
        + coefficient_multiplications
        + alignment_multiplications
    )
    return {
        'ciphertext_multiplications': ciphertext_multiplications,
        'coefficient_multiplications': coefficient_multiplications,
        'alignment_multiplications': alignment_multiplications,
        'relinearizations': ciphertext_multiplications,
        'rescale_operations': total,
        'total_multiplications': total,
    }


def _validate_polynomial_coefficients(
    polynomial: PolynomialApproximation,
) -> None:
    """Reject non-finite coefficients before allocating encrypted temporaries."""

    if any(
        not math.isfinite(coefficient.real)
        or not math.isfinite(coefficient.imag)
        for coefficient in polynomial.coefficients
    ):
        raise ValueError('polynomial coefficients must be finite')


def _validate_encrypted_evaluation(
    engine: CkksEngine,
    ciphertext: Ciphertext,
    polynomial: PolynomialApproximation,
    *,
    required_levels: int,
    relinearization_key: RelinearizationKey | None,
    requires_relinearization: bool,
    evaluator_name: str,
) -> None:
    """Validate all polynomial-evaluator entry requirements."""

    _validate_polynomial_coefficients(polynomial)
    if not isinstance(ciphertext, Ciphertext):
        raise TypeError(
            f'{evaluator_name} expects Ciphertext, '
            f'got {type(ciphertext).__name__}'
        )
    ciphertext.assert_state(
        polynomial_domain='coefficient',
        residue_representation='standard',
        modulus_basis='Q',
        components=2,
    )
    engine._assert_engine_ciphertext(ciphertext)
    if not math.isclose(
        ciphertext.scale,
        engine.config.default_scale,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        raise ValueError(
            f'{evaluator_name} requires input scale equal to the engine '
            f'default scale: {ciphertext.scale} != '
            f'{engine.config.default_scale}'
        )
    output_level = ciphertext.level + required_levels
    if output_level > engine.final_public_level:
        raise ValueError(
            f'{evaluator_name} needs {required_levels} level transitions '
            f'from entry level {ciphertext.level}, but the final public '
            f'level is {engine.final_public_level}'
        )
    if requires_relinearization:
        if relinearization_key is None:
            raise ValueError(f'{evaluator_name} requires a relinearization key')
        engine._assert_engine_key(
            relinearization_key,
            expected_type=RelinearizationKey,
            modulus_basis='QP',
        )


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
        does not model CKKS rounding or level consumption.
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
        level spent on that affine map.

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


@dataclass(frozen=True)
class BalancedPowerEvaluator:
    r"""Evaluate a power series through a shared balanced product tree.

    To construct $x^n$, the evaluator recursively multiplies
    $x^{\lfloor n/2\rfloor}$ by $x^{\lceil n/2\rceil}$. Every exponent is
    cached for one call, so terms share intermediate ciphertexts. Terms are
    level-aligned before addition and each active coefficient multiplication
    consumes one final level.

    `evaluate` consumes a two-component coefficient-domain, standard-RNS Q
    ciphertext representing the basis coordinate $x$. Its dense payload axes
    are `[component, *batch, limb, coefficient]`; all batch members share level,
    scale, and exact `prime_ids`. The method is functional. Each ciphertext
    product converts operands to NTT/Montgomery form, multiplies to three
    components, relinearizes back to two coefficient-domain standard
    components, drops one leading Q row, and applies the bootstrap's explicit
    default-scale reinterpretation. The final result is coefficient-domain
    standard RNS over Q at `ciphertext.level + required_levels(polynomial)`,
    with the corresponding `prime_ids`, unchanged batch shape, two components,
    and actual default scale.
    """

    skip_near_zero: float = 0.0

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return exact multiplying operations executed by :meth:`evaluate`.

        Coefficient multiplication counts only active nonconstant terms.
        Alignment multiplication counts every multiply-by-one level advance,
        including advances used inside the shared power tree and before term
        addition. Additions and plaintext encoding are not multiplications.
        """

        if polynomial.basis != 'power':
            raise ValueError('BalancedPowerEvaluator requires power basis')
        degrees = [
            index
            for index, coefficient in enumerate(polynomial.coefficients)
            if index and abs(coefficient) > self.skip_near_zero
        ]
        if not degrees:
            return _inventory(
                ciphertext_multiplications=0,
                coefficient_multiplications=1,
                alignment_multiplications=0,
            )

        ciphertext_multiplications = 0
        alignment_multiplications = 0
        power_levels: dict[int, int] = {1: 0}

        def power_level(exponent: int) -> int:
            nonlocal ciphertext_multiplications, alignment_multiplications
            cached = power_levels.get(exponent)
            if cached is not None:
                return cached
            left = power_level(exponent // 2)
            right = power_level(exponent - exponent // 2)
            alignment_multiplications += abs(left - right)
            level = max(left, right) + 1
            ciphertext_multiplications += 1
            power_levels[exponent] = level
            return level

        result_level: int | None = None
        for degree in degrees:
            term_level = power_level(degree) + 1
            if result_level is None:
                result_level = term_level
            else:
                alignment_multiplications += abs(result_level - term_level)
                result_level = max(result_level, term_level)
        return _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=len(degrees),
            alignment_multiplications=alignment_multiplications,
        )

    def required_levels(self, polynomial: PolynomialApproximation) -> int:
        """Count the deepest balanced-product path and coefficient product."""

        if polynomial.basis != 'power':
            raise ValueError('BalancedPowerEvaluator requires power basis')
        degrees = [
            index
            for index, coefficient in enumerate(polynomial.coefficients)
            if index and abs(coefficient) > self.skip_near_zero
        ]
        if not degrees:
            return 1

        @cache
        def depth(exponent: int) -> int:
            if exponent <= 1:
                return 0
            left = exponent // 2
            return max(depth(left), depth(exponent - left)) + 1

        return max(depth(degree) for degree in degrees) + 1

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate $p(x)=\sum_n a_nx^n$ at a common CKKS level.

        Coefficients with magnitude at or below `skip_near_zero` are omitted.
        A constant-only polynomial still consumes one level by multiplying the
        input by zero; this keeps the execution behavior equal to the declared
        one-level cost. Inputs are not mutated and output storage does not alias
        an input.
        """

        if polynomial.basis != 'power':
            raise ValueError('BalancedPowerEvaluator requires power basis')
        from fhelium.experimental.bootstrap import _ops

        coefficients = polynomial.coefficients
        degrees = [
            degree
            for degree, value in enumerate(coefficients)
            if degree and abs(value) > self.skip_near_zero
        ]
        if not degrees:
            zero = _ops._multiply_scalar(engine, ciphertext, 0.0)
            return _ops._add_scalar(engine, zero, coefficients[0])
        powers: dict[int, Ciphertext] = {1: ciphertext}

        def power(exponent: int) -> Ciphertext:
            # The same recursively constructed power may feed many terms.
            cached = powers.get(exponent)
            if cached is not None:
                return cached
            if relinearization_key is None:
                raise ValueError(
                    'nonlinear power evaluation requires relinearization key'
                )
            left = exponent // 2
            value = _ops._multiply_relinearize_rescale(
                engine,
                power(left),
                power(exponent - left),
                relinearization_key=relinearization_key,
            )
            powers[exponent] = value
            return value

        result: Ciphertext | None = None
        for degree in degrees:
            term = _ops._multiply_scalar(
                engine, power(degree), coefficients[degree]
            )
            if result is None:
                result = term
            else:
                result, term = _ops._align_levels(engine, result, term)
                if not math.isclose(result.scale, term.scale, rel_tol=1e-9):
                    raise ValueError(
                        'polynomial terms have incompatible scales'
                    )
                result = engine.add(result, term)
        if result is None:
            raise RuntimeError('polynomial evaluation produced no ciphertext')
        if abs(coefficients[0]) > self.skip_near_zero:
            result = _ops._add_scalar(engine, result, coefficients[0])
        return result


@dataclass(frozen=True)
class HornerPowerEvaluator:
    r"""Evaluate a power polynomial by a corrected level-aware Horner chain.

    For degree $d\geq 1$, evaluation starts with

    $$
    r=c_dx+c_{d-1}
    $$

    and then applies $r\leftarrow rx+c_i$ for $i=d-2,\ldots,0$.
    Thus the leading term remains $c_dx^d$. A level-specific copy of $x$ is
    advanced once per ciphertext product and retained for that iteration,
    avoiding repeated advancement from the entry level. The evaluator does
    not omit zero coefficients: the declared algebraic degree and execution
    schedule are stable properties of the coefficient tuple.

    A constant polynomial deliberately consumes one level by multiplying the
    input by zero. A linear polynomial consumes one coefficient-multiplication
    level and requires no relinearization key. Degree $d\geq2$ consumes exactly
    $d$ levels and requires one compatible QP relinearization key.
    """

    def required_levels(self, polynomial: PolynomialApproximation) -> int:
        """Return one level for constants or the declared power degree."""

        if polynomial.basis != 'power':
            raise ValueError('HornerPowerEvaluator requires power basis')
        return max(1, polynomial.degree)

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return the exact corrected-Horner multiplication inventory."""

        required_levels = self.required_levels(polynomial)
        del required_levels
        degree = polynomial.degree
        if degree == 0:
            return _inventory(
                ciphertext_multiplications=0,
                coefficient_multiplications=1,
                alignment_multiplications=0,
            )
        return _inventory(
            ciphertext_multiplications=max(0, degree - 1),
            coefficient_multiplications=1,
            alignment_multiplications=max(0, degree - 1),
        )

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate a power polynomial with a corrected Horner recurrence.

        The input must be a complete two-component coefficient-domain,
        standard-RNS Q ciphertext at the engine default scale. The method
        validates all context, key, scale, and available-depth requirements
        before allocating encrypted temporaries. The functional output is in
        the same arithmetic state at
        ``ciphertext.level + required_levels(polynomial)`` and default scale.
        """

        required_levels = self.required_levels(polynomial)
        degree = polynomial.degree
        _validate_encrypted_evaluation(
            engine,
            ciphertext,
            polynomial,
            required_levels=required_levels,
            relinearization_key=relinearization_key,
            requires_relinearization=degree >= 2,
            evaluator_name=type(self).__name__,
        )
        from fhelium.experimental.bootstrap import _ops

        coefficients = polynomial.coefficients
        if degree == 0:
            result = _ops._multiply_scalar(engine, ciphertext, 0.0)
            return _ops._add_scalar(engine, result, coefficients[0])

        result = _ops._multiply_scalar(engine, ciphertext, coefficients[degree])
        result = _ops._add_scalar(engine, result, coefficients[degree - 1])
        level_x = ciphertext
        for coefficient_index in range(degree - 2, -1, -1):
            assert relinearization_key is not None
            level_x = _ops._advance_level(engine, level_x)
            result = _ops._multiply_relinearize_rescale(
                engine,
                result,
                level_x,
                relinearization_key=relinearization_key,
            )
            result = _ops._add_scalar(
                engine, result, coefficients[coefficient_index]
            )
        expected_level = ciphertext.level + required_levels
        if result.level != expected_level:
            raise RuntimeError(
                'HornerPowerEvaluator produced an unexpected output level: '
                f'{result.level} != {expected_level}'
            )
        return result


@dataclass(frozen=True)
class PatersonStockmeyerPowerEvaluator:
    r"""Evaluate a power polynomial with one fixed baby-step size.

    ``baby_step=k`` is part of the evaluator identity and is never selected at
    runtime. The evaluator writes

    $$
    p(x)=\sum_g q_g(x)(x^k)^g,
    \qquad \deg q_g<k.
    $$

    Balanced shared powers $x,\ldots,x^k$ are constructed once. Baby powers
    are advanced to one common level before coefficient multiplication, then
    the $q_g$ values are combined by Horner evaluation in $x^k$. Level-specific
    copies of shared powers are cached for the invocation. If the highest
    group is a constant, its first giant step is the plaintext product
    $c_dx^k$, not an encrypted-zero seed followed by a ciphertext product.

    The declared coefficient tuple, including zero entries, fixes the schedule.
    ``baby_step`` therefore controls a reproducible DAG rather than an
    unreliable degree-only estimate.
    """

    baby_step: int

    def __post_init__(self) -> None:
        if type(self.baby_step) is not int:
            raise TypeError('baby_step must be an integer')
        if self.baby_step < 2:
            raise ValueError('baby_step must be at least two')

    def _schedule(
        self, polynomial: PolynomialApproximation
    ) -> tuple[int, dict[str, int]]:
        """Simulate the exact relative levels and multiplying operations."""

        if polynomial.basis != 'power':
            raise ValueError(
                'PatersonStockmeyerPowerEvaluator requires power basis'
            )
        degree = polynomial.degree
        if degree == 0:
            return 1, _inventory(
                ciphertext_multiplications=0,
                coefficient_multiplications=1,
                alignment_multiplications=0,
            )

        ciphertext_multiplications = 0
        coefficient_multiplications = 0
        alignment_multiplications = 0
        base_levels: dict[int, int] = {1: 0}
        cached_levels: dict[int, set[int]] = {1: {0}}

        def at_level(exponent: int, target: int) -> int:
            nonlocal alignment_multiplications
            levels = cached_levels[exponent]
            candidates = [level for level in levels if level <= target]
            if not candidates:
                raise RuntimeError('power cache cannot move backwards')
            source = max(candidates)
            alignment_multiplications += target - source
            levels.update(range(source + 1, target + 1))
            return target

        def power(exponent: int) -> int:
            nonlocal ciphertext_multiplications
            cached = base_levels.get(exponent)
            if cached is not None:
                return cached
            left_exponent = exponent // 2
            right_exponent = exponent - left_exponent
            left = power(left_exponent)
            right = power(right_exponent)
            target = max(left, right)
            at_level(left_exponent, target)
            at_level(right_exponent, target)
            result = target + 1
            ciphertext_multiplications += 1
            base_levels[exponent] = result
            cached_levels[exponent] = {result}
            return result

        maximum_power = min(self.baby_step, degree)
        for exponent in range(2, maximum_power + 1):
            power(exponent)

        baby_maximum = min(self.baby_step - 1, degree)
        baby_target = max(
            power(exponent) for exponent in range(1, baby_maximum + 1)
        )
        for exponent in range(1, baby_maximum + 1):
            at_level(exponent, baby_target)
        group_level = baby_target + 1
        group_count = degree // self.baby_step + 1

        def count_group(group: int) -> int:
            nonlocal coefficient_multiplications
            start = group * self.baby_step
            upper = min(self.baby_step - 1, degree - start)
            if upper <= 0:
                raise RuntimeError('constant group must use the top shortcut')
            coefficient_multiplications += upper
            return group_level

        if group_count == 1:
            result_level = count_group(0)
        else:
            xk_level = power(self.baby_step)
            highest_group = group_count - 1
            highest_remainder = degree - highest_group * self.baby_step
            if highest_remainder == 0:
                coefficient_multiplications += 1
                result_level = xk_level + 1
                group = highest_group - 1
                term_level = count_group(group)
                alignment_multiplications += abs(result_level - term_level)
                result_level = max(result_level, term_level)
                group -= 1
            else:
                result_level = count_group(highest_group)
                group = highest_group - 1

            while group >= 0:
                if xk_level < result_level:
                    at_level(self.baby_step, result_level)
                    product_input_level = result_level
                else:
                    alignment_multiplications += xk_level - result_level
                    product_input_level = xk_level
                ciphertext_multiplications += 1
                result_level = product_input_level + 1
                term_level = count_group(group)
                alignment_multiplications += abs(result_level - term_level)
                result_level = max(result_level, term_level)
                group -= 1

        return result_level, _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=coefficient_multiplications,
            alignment_multiplications=alignment_multiplications,
        )

    def required_levels(self, polynomial: PolynomialApproximation) -> int:
        """Return the exact critical-path level cost for this fixed ``k``."""

        return self._schedule(polynomial)[0]

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return exact ciphertext, coefficient, and alignment counts."""

        return self._schedule(polynomial)[1]

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate with the fixed baby/giant schedule and level caches."""

        required_levels = self.required_levels(polynomial)
        degree = polynomial.degree
        _validate_encrypted_evaluation(
            engine,
            ciphertext,
            polynomial,
            required_levels=required_levels,
            relinearization_key=relinearization_key,
            requires_relinearization=degree >= 2,
            evaluator_name=type(self).__name__,
        )
        from fhelium.experimental.bootstrap import _ops

        coefficients = polynomial.coefficients
        if degree == 0:
            result = _ops._multiply_scalar(engine, ciphertext, 0.0)
            return _ops._add_scalar(engine, result, coefficients[0])

        base_powers: dict[int, Ciphertext] = {1: ciphertext}
        level_powers: dict[int, dict[int, Ciphertext]] = {
            1: {ciphertext.level: ciphertext}
        }

        def at_level(exponent: int, target: int) -> Ciphertext:
            levels = level_powers[exponent]
            candidates = [level for level in levels if level <= target]
            if not candidates:
                raise RuntimeError('power cache cannot move backwards')
            source_level = max(candidates)
            value = levels[source_level]
            while value.level < target:
                value = _ops._advance_level(engine, value)
                levels[value.level] = value
            return value

        def power(exponent: int) -> Ciphertext:
            cached = base_powers.get(exponent)
            if cached is not None:
                return cached
            assert relinearization_key is not None
            left_exponent = exponent // 2
            right_exponent = exponent - left_exponent
            left = power(left_exponent)
            right = power(right_exponent)
            target = max(left.level, right.level)
            left = at_level(left_exponent, target)
            right = at_level(right_exponent, target)
            value = _ops._multiply_relinearize_rescale(
                engine,
                left,
                right,
                relinearization_key=relinearization_key,
            )
            base_powers[exponent] = value
            level_powers[exponent] = {value.level: value}
            return value

        maximum_power = min(self.baby_step, degree)
        for exponent in range(2, maximum_power + 1):
            power(exponent)

        baby_maximum = min(self.baby_step - 1, degree)
        baby_target = max(
            power(exponent).level for exponent in range(1, baby_maximum + 1)
        )
        for exponent in range(1, baby_maximum + 1):
            at_level(exponent, baby_target)
        group_count = degree // self.baby_step + 1

        def build_group(group: int) -> Ciphertext:
            start = group * self.baby_step
            upper = min(self.baby_step - 1, degree - start)
            if upper <= 0:
                raise RuntimeError('constant group must use the top shortcut')
            terms = [
                _ops._multiply_scalar(
                    engine,
                    at_level(exponent, baby_target),
                    coefficients[start + exponent],
                )
                for exponent in range(1, upper + 1)
            ]
            result = terms[0]
            for term in terms[1:]:
                result = engine.add(result, term)
            return _ops._add_scalar(engine, result, coefficients[start])

        if group_count == 1:
            result = build_group(0)
        else:
            xk = power(self.baby_step)
            highest_group = group_count - 1
            highest_remainder = degree - highest_group * self.baby_step
            if highest_remainder == 0:
                result = _ops._multiply_scalar(
                    engine,
                    xk,
                    coefficients[highest_group * self.baby_step],
                )
                group = highest_group - 1
                term = build_group(group)
                result, term = _ops._align_levels(engine, result, term)
                result = engine.add(result, term)
                group -= 1
            else:
                result = build_group(highest_group)
                group = highest_group - 1

            while group >= 0:
                assert relinearization_key is not None
                if xk.level < result.level:
                    xk = at_level(self.baby_step, result.level)
                elif result.level < xk.level:
                    result, xk = _ops._align_levels(engine, result, xk)
                result = _ops._multiply_relinearize_rescale(
                    engine,
                    result,
                    xk,
                    relinearization_key=relinearization_key,
                )
                term = build_group(group)
                result, term = _ops._align_levels(engine, result, term)
                result = engine.add(result, term)
                group -= 1

        expected_level = ciphertext.level + required_levels
        if result.level != expected_level:
            raise RuntimeError(
                'PatersonStockmeyerPowerEvaluator produced an unexpected '
                f'output level: {result.level} != {expected_level}'
            )
        return result


@dataclass(frozen=True)
class BinaryDecompositionChebyshevEvaluator:
    r"""Evaluate a Chebyshev series through shared doubling identities.

    The evaluator builds only basis elements required by nonzero terms.  It
    recursively uses

    $$
    T_{2n}(x)=2T_n(x)^2-1,\qquad
    T_{2n+1}(x)=2T_n(x)T_{n+1}(x)-T_1(x),
    $$

    caching every required $T_n$ for one call. This keeps intermediate inputs near the approximation's
    bounded domain and exposes an exact critical-path depth.

    The coordinate, tensor axes, arithmetic-state preconditions, functional
    behavior, per-product transitions, output level, active `prime_ids`, and
    explicit default-scale policy match `BalancedPowerEvaluator`; only the
    polynomial basis and multiplication DAG differ.
    """

    skip_near_zero: float = 0.0

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return exact multiplying operations executed by :meth:`evaluate`.

        Alignment includes operand advancement within odd recurrences,
        advancement of the original $T_1=x$ before subtraction, and term
        advancement before addition.
        """

        if polynomial.basis != 'chebyshev':
            raise ValueError(
                'BinaryDecompositionChebyshevEvaluator requires Chebyshev basis'
            )
        degrees = [
            index
            for index, coefficient in enumerate(polynomial.coefficients)
            if index and abs(coefficient) > self.skip_near_zero
        ]
        if not degrees:
            return _inventory(
                ciphertext_multiplications=0,
                coefficient_multiplications=1,
                alignment_multiplications=0,
            )

        ciphertext_multiplications = 0
        alignment_multiplications = 0
        basis_levels: dict[int, int] = {1: 0}

        def basis_level(degree: int) -> int:
            nonlocal ciphertext_multiplications, alignment_multiplications
            cached = basis_levels.get(degree)
            if cached is not None:
                return cached
            half = degree // 2
            left = basis_level(half)
            right = left if degree % 2 == 0 else basis_level(half + 1)
            alignment_multiplications += abs(left - right)
            level = max(left, right) + 1
            ciphertext_multiplications += 1
            if degree % 2:
                alignment_multiplications += level
            basis_levels[degree] = level
            return level

        result_level: int | None = None
        for degree in degrees:
            term_level = basis_level(degree) + 1
            if result_level is None:
                result_level = term_level
            else:
                alignment_multiplications += abs(result_level - term_level)
                result_level = max(result_level, term_level)
        return _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=len(degrees),
            alignment_multiplications=alignment_multiplications,
        )

    def required_levels(self, polynomial: PolynomialApproximation) -> int:
        """Count the deepest required recurrence plus coefficient product."""

        if polynomial.basis != 'chebyshev':
            raise ValueError(
                'BinaryDecompositionChebyshevEvaluator requires Chebyshev basis'
            )
        degrees = [
            index
            for index, coefficient in enumerate(polynomial.coefficients)
            if index and abs(coefficient) > self.skip_near_zero
        ]
        if not degrees:
            return 1

        @cache
        def depth(index: int) -> int:
            if index <= 1:
                return 0
            half = index // 2
            if index % 2 == 0:
                return depth(half) + 1
            return max(depth(half), depth(half + 1)) + 1

        return max(depth(degree) for degree in degrees) + 1

    def evaluate(
        self,
        engine: CkksEngine,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Build required $T_n(x)$ values, align term levels, and sum them.

        Terms at or below `skip_near_zero` are omitted. As in the power
        evaluator, the constant-only case deliberately consumes one level so
        its execution agrees with `required_levels`. The method is functional
        and returns a two-component coefficient-domain standard-RNS Q value at
        actual default scale.
        """

        if polynomial.basis != 'chebyshev':
            raise ValueError(
                'BinaryDecompositionChebyshevEvaluator requires Chebyshev basis'
            )
        from fhelium.experimental.bootstrap import _ops

        coefficients = polynomial.coefficients
        degrees = [
            degree
            for degree, value in enumerate(coefficients)
            if degree and abs(value) > self.skip_near_zero
        ]
        if not degrees:
            zero = _ops._multiply_scalar(engine, ciphertext, 0.0)
            return _ops._add_scalar(engine, zero, coefficients[0])
        basis_values: dict[int, Ciphertext] = {1: ciphertext}

        def basis(degree: int) -> Ciphertext:
            # Cache recurrence nodes because adjacent terms share most of them.
            cached = basis_values.get(degree)
            if cached is not None:
                return cached
            if relinearization_key is None:
                raise ValueError(
                    'nonlinear Chebyshev evaluation requires '
                    'relinearization key'
                )
            half = degree // 2
            if degree % 2 == 0:
                product = _ops._multiply_relinearize_rescale(
                    engine,
                    basis(half),
                    basis(half),
                    relinearization_key=relinearization_key,
                )
                doubled = engine.add(product, product)
                value = _ops._add_scalar(engine, doubled, -1.0)
            else:
                product = _ops._multiply_relinearize_rescale(
                    engine,
                    basis(half),
                    basis(half + 1),
                    relinearization_key=relinearization_key,
                )
                doubled = engine.add(product, product)
                doubled, linear = _ops._align_levels(
                    engine, doubled, ciphertext
                )
                value = engine.subtract(doubled, linear)
            basis_values[degree] = value
            return value

        result: Ciphertext | None = None
        for degree in degrees:
            term = _ops._multiply_scalar(
                engine, basis(degree), coefficients[degree]
            )
            if result is None:
                result = term
            else:
                result, term = _ops._align_levels(engine, result, term)
                result = engine.add(result, term)
        if result is None:
            raise RuntimeError('Chebyshev evaluation produced no ciphertext')
        if abs(coefficients[0]) > self.skip_near_zero:
            result = _ops._add_scalar(engine, result, coefficients[0])
        return result
