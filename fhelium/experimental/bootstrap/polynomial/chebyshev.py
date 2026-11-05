r"""Binary-decomposition evaluation of encrypted Chebyshev series."""

from __future__ import annotations

from collections.abc import Mapping

from collections import Counter
from dataclasses import dataclass

from fhelium.values import EvaluationKeySet, Ciphertext, RelinearizationKey
from fhelium.experimental.bootstrap.arithmetic import (
    BootstrapArithmetic,
    _prepare_multiply_scalar,
    _rescale,
)
from fhelium.experimental.bootstrap.polynomial.evaluation import _inventory
from fhelium.experimental.bootstrap.polynomial.approximation import (
    PolynomialApproximation,
)


@dataclass(frozen=True)
class BinaryDecompositionChebyshevEvaluator:
    r"""Evaluate a Chebyshev series through shared doubling identities.

    The evaluator builds only basis elements required by nonzero terms.  It
    recursively uses

    $$
    T_{2n}(x)=2T_n(x)^2-1,\qquad
    T_{2n+1}(x)=2T_n(x)T_{n+1}(x)-T_1(x),
    $$

    Shared basis nodes are retained until their last polynomial or recurrence
    consumer. Depth-specific $T_1$ copies serve the odd recurrences.
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

    def _schedule(
        self, polynomial: PolynomialApproximation
    ) -> tuple[dict[int, int], dict[str, int]]:
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
            return {1: 0}, _inventory(
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
        return basis_depths, _inventory(
            ciphertext_multiplications=ciphertext_multiplications,
            coefficient_multiplications=len(degrees),
            alignment_multiplications=alignment_multiplications,
            coefficient_rescales=len(term_depths),
        )

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Return multiplying operations in the shared basis schedule."""
        return self._schedule(polynomial)[1]

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
        """Count the deepest shared basis node plus its coefficient rescale."""
        return max(self._schedule(polynomial)[0].values()) + 1

    def relinearization_depths(
        self, polynomial: PolynomialApproximation, *, input_depth: int = 0
    ) -> tuple[int, ...]:
        """Return product-input depths of the required shared basis nodes."""
        return tuple(
            sorted(
                {
                    input_depth + depth - 1
                    for degree, depth in self._schedule(polynomial)[0].items()
                    if degree > 1
                }
            )
        )

    def evaluate(
        self,
        arithmetic: BootstrapArithmetic,
        ciphertext: Ciphertext,
        polynomial: PolynomialApproximation,
        *,
        relinearization_key: RelinearizationKey
        | EvaluationKeySet
        | Mapping[int, RelinearizationKey]
        | None = None,
    ) -> Ciphertext:
        r"""Build required $T_n(x)$ values, align term depths, and sum them.

        Terms at or below `skip_near_zero` are omitted. A constant-only polynomial consumes one depth, as reported by `required_depths`. The method is functional
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
        consumers: Counter[int] = Counter(degrees)
        nodes: set[int] = {1}

        def count_dependencies(degree: int) -> None:
            if degree in nodes:
                return
            nodes.add(degree)
            half = degree // 2
            right = half if degree % 2 == 0 else half + 1
            consumers.update((half, right))
            count_dependencies(half)
            count_dependencies(right)

        for degree in degrees:
            count_dependencies(degree)
        basis_values: dict[int, Ciphertext] = {1: ciphertext}

        def consume(degree: int) -> None:
            consumers[degree] -= 1
            if consumers[degree] == 0:
                del basis_values[degree]

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
                    basis_values[half],
                    basis_values[right],
                    relinearization_key=relinearization_key,
                )
                doubled = engine.add(product, product)
                basis_values[current] = (
                    ops.add_scalar(doubled, -1.0)
                    if current % 2 == 0
                    else engine.subtract(
                        doubled,
                        linear_at_depth(doubled.depth),
                    )
                )
                consume(half)
                consume(right)
                del product, doubled
                pending.pop()
            return basis_values[degree]

        pending_terms: dict[int, Ciphertext] = {}
        domains: dict[int, str] = {}
        for degree in degrees:
            value = basis(degree)
            depth = value.depth
            domains[depth] = value.polynomial_domain
            value_ntt = (
                value
                if value.polynomial_domain == 'ntt'
                else engine.coefficient_domain_to_ntt_domain(value)
            )
            term = engine.multiply_plaintext(
                value_ntt,
                _prepare_multiply_scalar(
                    ops,
                    value,
                    coefficients[degree],
                    constant_cache=ops.constant_cache,
                ),
            )
            accumulator = pending_terms.get(depth)
            pending_terms[depth] = (
                term if accumulator is None else engine.add(accumulator, term)
            )
            consume(degree)
            del value, value_ntt, term, accumulator
        ordered_depths = sorted(pending_terms)
        grouped_terms: dict[int, Ciphertext] = {}
        for depth in ordered_depths:
            term = pending_terms.pop(depth)
            term = _rescale(engine, term, retain_ntt=ops.retain_ntt)
            if domains[depth] == 'coefficient' and ops.retain_ntt:
                term = engine.ntt_domain_to_coefficient_domain(term)
            grouped_terms[depth] = term
        del term
        result = grouped_terms.pop(ordered_depths[0])
        for depth in ordered_depths[1:]:
            result, term = ops.align_depths(
                result,
                grouped_terms.pop(depth),
            )
            result = engine.add(result, term)
            del term
        if abs(coefficients[0]) > self.skip_near_zero:
            result = ops.add_scalar(result, coefficients[0])
        return result
