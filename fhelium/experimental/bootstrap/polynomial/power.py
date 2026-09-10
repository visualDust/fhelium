r"""Encrypted evaluators for power-basis polynomials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from fhelium.values import Ciphertext, RelinearizationKey
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.polynomial.evaluation import (
    _inventory,
    _validate_encrypted_evaluation,
)
from fhelium.experimental.bootstrap.polynomial.approximation import (
    PolynomialApproximation,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class BalancedPowerEvaluator:
    r"""Evaluate a power series through a shared balanced product tree.

    To construct $x^n$, the evaluator recursively multiplies
    $x^{\lfloor n/2\rfloor}$ by $x^{\lceil n/2\rceil}$. Every exponent is
    cached for one call, so terms share intermediate ciphertexts. Terms are
    depth-aligned using a basis scale recurrence anchored at the input.
    Each active coefficient multiplication
    consumes one final depth.

    `evaluate` consumes a two-component coefficient-domain, standard-RNS Q
    ciphertext representing the basis coordinate $x$. Its dense payload axes
    are `[component, *batch, limb, coefficient]`; all batch members share depth,
    scale, and `prime_ids`. The method is functional. Each ciphertext
    product converts operands to NTT/Montgomery form, multiplies to three
    components, relinearizes back to two coefficient-domain standard
    components, and rescales by the complete Q group product while retaining
    actual scale. The final result is coefficient-domain
    standard RNS over Q at `ciphertext.depth + required_depths(polynomial)`,
    with the corresponding `prime_ids`, unchanged batch shape, two components,
    and the coefficient-product target scale.
    """

    skip_near_zero: float = 0.0

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return multiplying operations executed by :meth:`evaluate`.

        Coefficient multiplication counts only active nonconstant terms.
        Alignment multiplication counts every multiply-by-one depth advance,
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
        power_depths: dict[int, int] = {1: 0}

        def power_depth(exponent: int) -> int:
            nonlocal ciphertext_multiplications, alignment_multiplications
            cached = power_depths.get(exponent)
            if cached is not None:
                return cached
            left = power_depth(exponent // 2)
            right = power_depth(exponent - exponent // 2)
            alignment_multiplications += abs(left - right)
            depth = max(left, right) + 1
            ciphertext_multiplications += 1
            power_depths[exponent] = depth
            return depth

        result_depth: int | None = None
        for degree in degrees:
            term_depth = power_depth(degree) + 1
            if result_depth is None:
                result_depth = term_depth
            else:
                alignment_multiplications += abs(result_depth - term_depth)
                result_depth = max(result_depth, term_depth)
        return _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=len(degrees),
            alignment_multiplications=alignment_multiplications,
        )

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
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
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate $p(x)=\sum_n a_nx^n$ at a common CKKS depth.

        Coefficients with magnitude at or below `skip_near_zero` are omitted.
        A constant-only polynomial still consumes one depth by multiplying the
        input by zero; this keeps the execution behavior equal to the declared
        one-depth cost. Inputs are not mutated and output storage does not alias
        an input.
        """

        if polynomial.basis != 'power':
            raise ValueError('BalancedPowerEvaluator requires power basis')
        engine = arithmetic.engine
        ops = arithmetic
        basis_ops = arithmetic.for_input(
            ciphertext, self.required_depths(polynomial) - 1
        )

        coefficients = polynomial.coefficients
        degrees = [
            degree
            for degree, value in enumerate(coefficients)
            if degree and abs(value) > self.skip_near_zero
        ]
        if not degrees:
            zero = ops.multiply_scalar(ciphertext, 0.0)
            return ops.add_scalar(zero, coefficients[0])
        powers: dict[int, Ciphertext] = {1: ciphertext}

        def power(exponent: int) -> Ciphertext:
            pending = [exponent]
            while pending:
                current = pending[-1]
                if current in powers:
                    pending.pop()
                    continue
                left = current // 2
                right = current - left
                if left not in powers:
                    pending.append(left)
                    continue
                if right not in powers:
                    pending.append(right)
                    continue
                if relinearization_key is None:
                    raise ValueError(
                        'nonlinear power evaluation requires relinearization key'
                    )
                powers[current] = basis_ops.multiply_relinearize_rescale(
                    powers[left], powers[right],
                    relinearization_key=relinearization_key,
                )
                pending.pop()
            return powers[exponent]

        result: Ciphertext | None = None
        for degree in degrees:
            term = ops.multiply_scalar(power(degree), coefficients[degree])
            if result is None:
                result = term
            else:
                result, term = ops.align_depths(result, term)
                if not math.isclose(result.scale, term.scale, rel_tol=1e-9):
                    raise ValueError(
                        'polynomial terms have incompatible scales'
                    )
                result = engine.add(result, term)
        if result is None:
            raise RuntimeError('polynomial evaluation produced no ciphertext')
        if abs(coefficients[0]) > self.skip_near_zero:
            result = ops.add_scalar(result, coefficients[0])
        return result


@dataclass(frozen=True)
class HornerPowerEvaluator:
    r"""Evaluate a power polynomial by a corrected depth-aware Horner chain.

    For degree $d\geq 1$, evaluation starts with

    $$
    r=c_dx+c_{d-1}
    $$

    and then applies $r\leftarrow rx+c_i$ for $i=d-2,\ldots,0$.
    Thus the leading term remains $c_dx^d$. A depth-specific copy of $x$ is
    advanced once per ciphertext product and retained for that iteration,
    avoiding repeated advancement from the entry depth. The evaluator does
    not omit zero coefficients: the declared algebraic degree and execution
    schedule are stable properties of the coefficient tuple.

    A constant polynomial deliberately consumes one depth by multiplying the
    input by zero. A linear polynomial consumes one coefficient-multiplication
    depth and requires no relinearization key. Degree $d\geq2$ consumes exactly
    $d$ depths and requires one compatible QP relinearization key.
    """

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
        """Return one depth for constants or the declared power degree."""

        if polynomial.basis != 'power':
            raise ValueError('HornerPowerEvaluator requires power basis')
        return max(1, polynomial.degree)

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return the corrected-Horner multiplication inventory."""

        required_depths = self.required_depths(polynomial)
        del required_depths
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
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate a power polynomial with a corrected Horner recurrence.

        The input must be a complete two-component coefficient-domain,
        standard-RNS Q ciphertext at its recorded actual scale. The method
        validates all context, key, scale, and available-depth requirements
        before allocating encrypted temporaries. The functional output is in
        the same arithmetic state at
        ``ciphertext.depth + required_depths(polynomial)`` and the arithmetic owner's target scale.
        """

        engine = arithmetic.engine
        ops = arithmetic
        required_depths = self.required_depths(polynomial)
        degree = polynomial.degree
        _validate_encrypted_evaluation(
            engine,
            ciphertext,
            polynomial,
            required_depths=required_depths,
            relinearization_key=relinearization_key,
            requires_relinearization=degree >= 2,
            evaluator_name=type(self).__name__,
        )
        coefficients = polynomial.coefficients
        if degree == 0:
            result = ops.multiply_scalar(ciphertext, 0.0)
            return ops.add_scalar(result, coefficients[0])

        result = ops.multiply_scalar(ciphertext, coefficients[degree])
        result = ops.add_scalar(result, coefficients[degree - 1])
        depth_x = ciphertext
        for coefficient_index in range(degree - 2, -1, -1):
            assert relinearization_key is not None
            depth_x = ops.advance_depth(depth_x)
            result = ops.multiply_relinearize_rescale(
                result,
                depth_x,
                relinearization_key=relinearization_key,
            )
            result = ops.add_scalar(result, coefficients[coefficient_index])
        expected_depth = ciphertext.depth + required_depths
        if result.depth != expected_depth:
            raise RuntimeError(
                'HornerPowerEvaluator produced an unexpected output depth: '
                f'{result.depth} != {expected_depth}'
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
    are advanced to one common depth before coefficient multiplication, then
    the $q_g$ values are combined by Horner evaluation in $x^k$. Depth-specific
    copies of shared powers are cached for the invocation. If the highest
    group is a constant, its first giant step is the plaintext product
    $c_dx^k$, not an encrypted-zero seed followed by a ciphertext product.

    Basis powers follow the input's actual scale recurrence. Coefficient
    accumulators use a common multiple of those scales chosen so the final
    product reaches the requested output scale. This keeps additions coherent
    without changing the polynomial's coefficients or its input coordinate.

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
        """Simulate relative depths and multiplying operations."""

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
        base_depths: dict[int, int] = {1: 0}
        cached_depths: dict[int, set[int]] = {1: {0}}

        def at_depth(exponent: int, target: int) -> int:
            nonlocal alignment_multiplications
            depths = cached_depths[exponent]
            candidates = [depth for depth in depths if depth <= target]
            if not candidates:
                raise RuntimeError('power cache cannot move backwards')
            source = max(candidates)
            alignment_multiplications += target - source
            depths.update(range(source + 1, target + 1))
            return target

        def power(exponent: int) -> int:
            nonlocal ciphertext_multiplications
            cached = base_depths.get(exponent)
            if cached is not None:
                return cached
            left_exponent = exponent // 2
            right_exponent = exponent - left_exponent
            left = power(left_exponent)
            right = power(right_exponent)
            target = max(left, right)
            at_depth(left_exponent, target)
            at_depth(right_exponent, target)
            result = target + 1
            ciphertext_multiplications += 1
            base_depths[exponent] = result
            cached_depths[exponent] = {result}
            return result

        maximum_power = min(self.baby_step, degree)
        for exponent in range(2, maximum_power + 1):
            power(exponent)

        baby_maximum = min(self.baby_step - 1, degree)
        baby_target = max(
            power(exponent) for exponent in range(1, baby_maximum + 1)
        )
        for exponent in range(1, baby_maximum + 1):
            at_depth(exponent, baby_target)
        group_depth = baby_target + 1
        group_count = degree // self.baby_step + 1

        def count_group(group: int) -> int:
            nonlocal coefficient_multiplications
            start = group * self.baby_step
            upper = min(self.baby_step - 1, degree - start)
            if upper <= 0:
                raise RuntimeError('constant group must use the top shortcut')
            coefficient_multiplications += upper
            return group_depth

        if group_count == 1:
            result_depth = count_group(0)
        else:
            xk_depth = power(self.baby_step)
            highest_group = group_count - 1
            highest_remainder = degree - highest_group * self.baby_step
            if highest_remainder == 0:
                coefficient_multiplications += 1
                result_depth = xk_depth + 1
                group = highest_group - 1
                term_depth = count_group(group)
                alignment_multiplications += abs(result_depth - term_depth)
                result_depth = max(result_depth, term_depth)
                group -= 1
            else:
                result_depth = count_group(highest_group)
                group = highest_group - 1

            while group >= 0:
                if xk_depth < result_depth:
                    at_depth(self.baby_step, result_depth)
                    product_input_depth = result_depth
                else:
                    alignment_multiplications += xk_depth - result_depth
                    product_input_depth = xk_depth
                ciphertext_multiplications += 1
                result_depth = product_input_depth + 1
                term_depth = count_group(group)
                alignment_multiplications += abs(result_depth - term_depth)
                result_depth = max(result_depth, term_depth)
                group -= 1

        return result_depth, _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=coefficient_multiplications,
            alignment_multiplications=alignment_multiplications,
        )

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
        """Return the critical-path depth cost for this fixed ``k``."""

        return self._schedule(polynomial)[0]

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return ciphertext, coefficient, and alignment counts."""

        return self._schedule(polynomial)[1]

    def evaluate(
        self,
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Evaluate with the fixed baby/giant schedule and depth caches."""

        engine = arithmetic.engine
        ops = arithmetic
        required_depths = self.required_depths(polynomial)
        degree = polynomial.degree
        _validate_encrypted_evaluation(
            engine,
            ciphertext,
            polynomial,
            required_depths=required_depths,
            relinearization_key=relinearization_key,
            requires_relinearization=degree >= 2,
            evaluator_name=type(self).__name__,
        )
        power_ops = arithmetic.for_input(ciphertext, required_depths)
        output_depth = ciphertext.depth + required_depths
        ops = power_ops.scaled_targets(
            arithmetic.target_scales[output_depth]
            / power_ops.target_scales[output_depth]
        )
        coefficients = polynomial.coefficients
        if degree == 0:
            result = ops.multiply_scalar(ciphertext, 0.0)
            return ops.add_scalar(result, coefficients[0])

        base_powers: dict[int, Ciphertext] = {1: ciphertext}
        depth_powers: dict[int, dict[int, Ciphertext]] = {
            1: {ciphertext.depth: ciphertext}
        }

        def at_depth(exponent: int, target: int) -> Ciphertext:
            depths = depth_powers[exponent]
            candidates = [depth for depth in depths if depth <= target]
            if not candidates:
                raise RuntimeError('power cache cannot move backwards')
            source_depth = max(candidates)
            value = depths[source_depth]
            while value.depth < target:
                value = power_ops.advance_depth(value)
                depths[value.depth] = value
            return value

        maximum_power = min(self.baby_step, degree)
        for exponent in range(2, maximum_power + 1):
            assert relinearization_key is not None
            left_exponent = exponent // 2
            right_exponent = exponent - left_exponent
            target = max(
                base_powers[left_exponent].depth,
                base_powers[right_exponent].depth,
            )
            value = power_ops.multiply_relinearize_rescale(
                at_depth(left_exponent, target),
                at_depth(right_exponent, target),
                relinearization_key=relinearization_key,
            )
            base_powers[exponent] = value
            depth_powers[exponent] = {value.depth: value}

        baby_maximum = min(self.baby_step - 1, degree)
        baby_target = max(
            base_powers[exponent].depth for exponent in range(1, baby_maximum + 1)
        )
        for exponent in range(1, baby_maximum + 1):
            at_depth(exponent, baby_target)
        group_count = degree // self.baby_step + 1

        def build_group(group: int) -> Ciphertext:
            start = group * self.baby_step
            upper = min(self.baby_step - 1, degree - start)
            if upper <= 0:
                raise RuntimeError('constant group must use the top shortcut')
            terms = [
                ops.multiply_scalar(
                    at_depth(exponent, baby_target),
                    coefficients[start + exponent],
                )
                for exponent in range(1, upper + 1)
            ]
            result = terms[0]
            for term in terms[1:]:
                result = engine.add(result, term)
            return ops.add_scalar(result, coefficients[start])

        if group_count == 1:
            result = build_group(0)
        else:
            xk = base_powers[self.baby_step]
            highest_group = group_count - 1
            highest_remainder = degree - highest_group * self.baby_step
            if highest_remainder == 0:
                result = ops.multiply_scalar(
                    xk,
                    coefficients[highest_group * self.baby_step],
                )
                group = highest_group - 1
                term = build_group(group)
                result, term = ops.align_depths(result, term)
                result = engine.add(result, term)
                group -= 1
            else:
                result = build_group(highest_group)
                group = highest_group - 1

            while group >= 0:
                assert relinearization_key is not None
                if xk.depth < result.depth:
                    xk = at_depth(self.baby_step, result.depth)
                elif result.depth < xk.depth:
                    result, xk = ops.align_depths(result, xk)
                result = ops.multiply_relinearize_rescale(
                    result,
                    xk,
                    relinearization_key=relinearization_key,
                )
                term = build_group(group)
                result, term = ops.align_depths(result, term)
                result = engine.add(result, term)
                group -= 1

        expected_depth = ciphertext.depth + required_depths
        if result.depth != expected_depth:
            raise RuntimeError(
                'PatersonStockmeyerPowerEvaluator produced an unexpected '
                f'output depth: {result.depth} != {expected_depth}'
            )
        return result
