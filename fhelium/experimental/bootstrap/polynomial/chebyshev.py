r"""Binary-decomposition evaluation of encrypted Chebyshev series."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from fhelium.values import Ciphertext, RelinearizationKey
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from fhelium.experimental.bootstrap.polynomial.evaluation import _inventory
from fhelium.experimental.bootstrap.polynomial.approximation import (
    PolynomialApproximation,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class BinaryDecompositionChebyshevEvaluator:
    r"""Evaluate a Chebyshev series through shared doubling identities.

    The evaluator builds only basis elements required by nonzero terms.  It
    recursively uses

    $$
    T_{2n}(x)=2T_n(x)^2-1,\qquad
    T_{2n+1}(x)=2T_n(x)T_{n+1}(x)-T_1(x),
    $$

    caching every required $T_n$ and each depth-specific $T_1$ for one call.
    Basis-node scales follow the input scale through each product and group
    rescale; the advanced copies of $T_1$ use that same recurrence.
    Coefficient products built from basis values at one depth are accumulated
    before one shared rescale; the resulting depth groups are then aligned and
    added. This keeps intermediate inputs near the approximation's bounded
    domain and exposes the critical-path depth.

    The coordinate, tensor axes, arithmetic-state preconditions, functional
    behavior, per-product transitions, output depth, active `prime_ids`, and
    depth-dependent scale schedule match `BalancedPowerEvaluator`; only the
    polynomial basis and multiplication DAG differ.
    """

    skip_near_zero: float = 0.0

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return multiplying operations executed by :meth:`evaluate`.

        Alignment includes operand advancement within odd recurrences, the
        shared depth chain for $T_1=x$, and advancement between coefficient
        depth groups. ``rescale_operations`` counts one coefficient rescale per
        occupied basis depth rather than one per coefficient product.
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
        basis_alignment_multiplications = 0
        maximum_linear_depth = 0
        basis_depths: dict[int, int] = {1: 0}

        def basis_depth(degree: int) -> int:
            nonlocal ciphertext_multiplications
            nonlocal basis_alignment_multiplications
            nonlocal maximum_linear_depth
            cached = basis_depths.get(degree)
            if cached is not None:
                return cached
            half = degree // 2
            left = basis_depth(half)
            right = left if degree % 2 == 0 else basis_depth(half + 1)
            basis_alignment_multiplications += abs(left - right)
            depth = max(left, right) + 1
            ciphertext_multiplications += 1
            if degree % 2:
                maximum_linear_depth = max(maximum_linear_depth, depth)
            basis_depths[degree] = depth
            return depth

        term_depths = {basis_depth(degree) + 1 for degree in degrees}
        grouped_term_alignment = max(term_depths) - min(term_depths)
        alignment_multiplications = (
            basis_alignment_multiplications
            + maximum_linear_depth
            + grouped_term_alignment
        )
        return _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=len(degrees),
            alignment_multiplications=alignment_multiplications,
            coefficient_rescales=len(term_depths),
        )

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
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
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Build required $T_n(x)$ values, align term depths, and sum them.

        Terms at or below `skip_near_zero` are omitted. As in the power
        evaluator, the constant-only case deliberately consumes one depth so
        its execution agrees with `required_depths`. The method is functional
        and returns a two-component coefficient-domain standard-RNS Q value at
        the arithmetic owner's target scale.
        """

        if polynomial.basis != 'chebyshev':
            raise ValueError(
                'BinaryDecompositionChebyshevEvaluator requires Chebyshev basis'
            )
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
        basis_values: dict[int, Ciphertext] = {1: ciphertext}
        linear_depths: dict[int, Ciphertext] = {ciphertext.depth: ciphertext}

        def linear_at_depth(target_depth: int) -> Ciphertext:
            cached = linear_depths.get(target_depth)
            if cached is not None:
                return cached
            source_depth = max(
                depth for depth in linear_depths if depth < target_depth
            )
            value = linear_depths[source_depth]
            while value.depth < target_depth:
                value = basis_ops.advance_depth(value)
                linear_depths[value.depth] = value
            return value

        def basis(degree: int) -> Ciphertext:
            pending = [degree]
            while pending:
                current = pending[-1]
                if current in basis_values:
                    pending.pop()
                    continue
                half = current // 2
                right = half if current % 2 == 0 else half + 1
                if half not in basis_values:
                    pending.append(half)
                    continue
                if right not in basis_values:
                    pending.append(right)
                    continue
                if relinearization_key is None:
                    raise ValueError(
                        'nonlinear Chebyshev evaluation requires '
                        'relinearization key'
                    )
                product = basis_ops.multiply_relinearize_rescale(
                    basis_values[half], basis_values[right],
                    relinearization_key=relinearization_key,
                )
                doubled = engine.add(product, product)
                basis_values[current] = (
                    ops.add_scalar(doubled, -1.0)
                    if current % 2 == 0
                    else engine.subtract(
                        doubled, linear_at_depth(doubled.depth),
                    )
                )
                pending.pop()
            return basis_values[degree]

        terms_by_depth: dict[int, list[tuple[Ciphertext, complex]]] = {}
        for degree in degrees:
            basis_value = basis(degree)
            terms_by_depth.setdefault(basis_value.depth, []).append(
                (basis_value, coefficients[degree])
            )
        if not terms_by_depth:
            raise RuntimeError('Chebyshev evaluation produced no ciphertext')
        ordered_depths = sorted(terms_by_depth)
        grouped_terms = {
            depth: ops.weighted_scalar_sum(terms)
            for depth, terms in terms_by_depth.items()
        }
        result = grouped_terms[ordered_depths[0]]
        for depth in ordered_depths[1:]:
            result, term = ops.align_depths(
                result,
                grouped_terms[depth],
            )
            result = engine.add(result, term)
        if abs(coefficients[0]) > self.skip_near_zero:
            result = ops.add_scalar(result, coefficients[0])
        return result
