r"""Block-factorized Chebyshev evaluation with a shared binary basis."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass, field, replace
import math

from fhelium.values import EvaluationKeySet, Ciphertext, RelinearizationKey
from fhelium.experimental.bootstrap.arithmetic import (
    BootstrapArithmetic,
    _prepare_scalar,
)
from fhelium.experimental.bootstrap.polynomial.approximation import (
    PolynomialApproximation,
)
from fhelium.experimental.bootstrap.polynomial.chebyshev import (
    BinaryDecompositionChebyshevEvaluator,
)
from fhelium.experimental.bootstrap.polynomial.evaluation import (
    _validate_encrypted_evaluation,
    _validate_polynomial_coefficients,
)


@dataclass(frozen=True)
class ChebyshevBlockPlan:
    """Coefficient blocks, shared binary nodes and their depth schedule.

    ``coefficients[j][i]`` multiplies ``T_i(x) T_(j*m)(x)``. A baby step of
    zero denotes the direct low-degree schedule. ``basis_depths`` records
    each node's depth relative to the input, in dependency order.
    """

    polynomial: PolynomialApproximation
    baby_step: int
    coefficients: tuple[tuple[complex, ...], ...]
    basis_depths: tuple[tuple[int, int], ...]
    required_depths: int
    inventory: tuple[tuple[str, int], ...]


def _plan(
    polynomial: PolynomialApproximation, baby_step: int | None
) -> ChebyshevBlockPlan:
    degree = polynomial.degree
    direct = BinaryDecompositionChebyshevEvaluator()
    if degree < 4 or (baby_step is not None and baby_step > degree):
        return ChebyshevBlockPlan(
            polynomial,
            0,
            (polynomial.coefficients,),
            (),
            direct.required_depths(polynomial),
            tuple(direct.operation_inventory(polynomial).items()),
        )
    m = (
        baby_step
        if baby_step is not None
        else 1 << ((degree - 1).bit_length() + 1) // 2
    )
    count = degree // m
    work = list(polynomial.coefficients)
    blocks = [[0j] * m for _ in range(count + 1)]
    for n in range(degree, m - 1, -1):
        j, i = divmod(n, m)
        value = work[n]
        work[n] = 0j
        if i:
            blocks[j][i] += 2 * value
            work[j * m - i] -= value
        else:
            blocks[j][0] += value
    blocks[0] = work[:m]
    nodes = {1}
    pending = [*range(2, m), *(j * m for j in range(1, count + 1))]
    while pending:
        n = pending.pop()
        if n in nodes:
            continue
        nodes.add(n)
        pending.extend((n // 2, n // 2 + n % 2))
    depths = {1: 0}
    advances: set[tuple[int, int]] = set()
    for n in sorted(nodes - {1}):
        left, right = n // 2, n // 2 + n % 2
        depths[n] = max(depths[left], depths[right]) + 1
        for operand in (left, right):
            advances.update(
                (operand, depth)
                for depth in range(depths[operand] + 1, depths[n])
            )
        if n % 2:
            advances.update((1, depth) for depth in range(1, depths[n] + 1))
    final = max(
        max(depths[k] for k in range(1, m)) + 2,
        max(depths[j * m] for j in range(1, count + 1)) + 1,
    )
    for j in range(1, count + 1):
        advances.update(
            (j * m, depth) for depth in range(depths[j * m] + 1, final)
        )
    alignment = len(advances)
    basis_products = len(nodes) - 1
    coefficient_products = (count + 1) * (m - 1) + 1
    inventory = {
        'ciphertext_multiplications': basis_products + count,
        'coefficient_multiplications': coefficient_products,
        'alignment_multiplications': alignment,
        'relinearizations': basis_products + 1,
        'rescale_operations': basis_products + alignment + count + 3,
        'total_multiplications': basis_products
        + count
        + coefficient_products
        + alignment,
    }
    return ChebyshevBlockPlan(
        polynomial,
        m,
        tuple(tuple(row) for row in blocks),
        tuple(sorted(depths.items())),
        final,
        tuple(inventory.items()),
    )


def _common_product_scale(scales: tuple[float, ...], preferred: float) -> float:
    r"""Choose a common binary64 product scale for coefficient encoding.

    Retain ``preferred`` when every ``s * (preferred / s)`` rounds back to it.
    Otherwise choose the smallest non-power-of-two operand significand and
    a nearby power-of-two exponent. For significands ``a > m``, division
    error multiplied by the operand is strictly below half an ulp of the
    interior target ``m * 2**e``. Equal significands and power-of-two
    operands give exact exponent shifts. Scales and quotients remain normal
    in the supported plaintext encoding range; no neighboring-float search
    or ciphertext metadata adjustment is performed.
    """
    if all(scale * (preferred / scale) == preferred for scale in scales):
        return preferred
    significands = tuple(math.frexp(scale)[0] for scale in scales)
    significand = min((m for m in significands if m != 0.5), default=0.5)
    exponent = math.frexp(preferred)[1]
    common = math.ldexp(significand, exponent)
    if common < preferred / math.sqrt(2.0):
        common *= 2.0
    elif common > preferred * math.sqrt(2.0):
        common *= 0.5
    if not all(scale * (common / scale) == common for scale in scales):
        raise ValueError(
            'coefficient scales cannot form one common product scale'
        )
    return common


@dataclass(frozen=True)
class BlockedChebyshevEvaluator:
    r"""Evaluate $P(x)=\sum_j A_j(x)T_{jm}(x)$ with shared binary nodes.

    Descending elimination uses $T_{jm+i}=2T_{jm}T_i-T_{jm-i}$ without
    refitting the polynomial. Giants use the original-coordinate binary
    recurrence. With ``baby_step=None``, $m$ is the smallest power of two
    satisfying $m^2\geq\deg P$. The plan determines the actual depth cost;
    constants and low degrees use the direct binary schedule.

    At final depth $f$, babies are restricted to $f-2$ while preserving each
    actual scale $S_k$. Coefficient plaintext scales $p_k=C/S_k$ are selected
    so every binary64 product is the common $C$. One block rescale produces
    $A=C/M_{f-2}$. Giants retain common scale $G$ at $f-1$; nonconstant giant
    products are summed as CT3 before one relinearization/rescale. The $j=0$
    block is multiplied by encoded one at scale $G$. All output terms have
    actual scale $(AG)/M_{f-1}$. Constants are added after block rescale at
    scale $A$. $M_d$ is the complete dropped prime-group product.

    Input and output are CT2 over Q with the same batch axes. The input
    domain is preserved when retaining NTT; otherwise output is coefficient
    form. NTT intermediates follow ``BootstrapArithmetic.retain_ntt``.
    The approximation's domain remains caller-owned; complex inputs are
    evaluated as $P(z)$ and are never projected to their real part.
    """

    baby_step: int | None = None
    skip_near_zero: float = 0.0
    _plans: dict[PolynomialApproximation, ChebyshevBlockPlan] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.baby_step is not None and self.baby_step < 2:
            raise ValueError('baby_step must be at least two')
        if self.skip_near_zero < 0 or not math.isfinite(self.skip_near_zero):
            raise ValueError('skip_near_zero must be finite and nonnegative')

    def plan(self, polynomial: PolynomialApproximation) -> ChebyshevBlockPlan:
        """Return the shared coefficient, basis, depth and operation plan."""
        if polynomial.basis != 'chebyshev':
            raise ValueError(
                'BlockedChebyshevEvaluator requires Chebyshev basis'
            )
        cached = self._plans.get(polynomial)
        if cached is not None:
            return cached
        _validate_polynomial_coefficients(polynomial)
        coefficients = [
            c if i == 0 or abs(c) > self.skip_near_zero else 0j
            for i, c in enumerate(polynomial.coefficients)
        ]
        while len(coefficients) > 1 and coefficients[-1] == 0:
            coefficients.pop()
        effective = replace(polynomial, coefficients=tuple(coefficients))
        result = _plan(effective, self.baby_step)
        self._plans[polynomial] = result
        return result

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
        """Return the depth consumed by this polynomial's selected plan."""
        return self.plan(polynomial).required_depths

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        """Count products, relinearizations and group rescales in the plan."""
        return dict(self.plan(polynomial).inventory)

    def relinearization_depths(
        self, polynomial: PolynomialApproximation, *, input_depth: int = 0
    ) -> tuple[int, ...]:
        """Return basis-product depths and the single delayed giant-sum depth."""
        plan = self.plan(polynomial)
        if plan.baby_step == 0:
            return (
                BinaryDecompositionChebyshevEvaluator().relinearization_depths(
                    plan.polynomial, input_depth=input_depth
                )
            )
        depths = {
            input_depth + depth - 1
            for degree, depth in plan.basis_depths
            if degree > 1
        }
        depths.add(input_depth + plan.required_depths - 1)
        return tuple(sorted(depths))

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
        """Execute the plan, retaining actual scales and the input domain."""
        plan = self.plan(polynomial)
        _validate_encrypted_evaluation(
            arithmetic.engine,
            ciphertext,
            plan.polynomial,
            required_depths=plan.required_depths,
            relinearization_key=relinearization_key,
            requires_relinearization=plan.polynomial.degree > 1,
            evaluator_name=type(self).__name__,
        )
        e = arithmetic.engine
        original_domain = ciphertext.polynomial_domain
        x = ciphertext
        if arithmetic.retain_ntt and original_domain == 'coefficient':
            x = e.coefficient_domain_to_ntt_domain(x)
        elif not arithmetic.retain_ntt and original_domain == 'ntt':
            x = e.ntt_domain_to_coefficient_domain(x)
        if plan.baby_step == 0:
            result = BinaryDecompositionChebyshevEvaluator().evaluate(
                arithmetic,
                x,
                plan.polynomial,
                relinearization_key=relinearization_key,
            )
        else:
            assert relinearization_key is not None
            result = self._evaluate_blocks(
                arithmetic, x, plan, relinearization_key
            )
        if (
            original_domain == 'coefficient'
            and result.polynomial_domain == 'ntt'
        ):
            result = e.ntt_domain_to_coefficient_domain(result)
        elif (
            original_domain == 'ntt'
            and arithmetic.retain_ntt
            and result.polynomial_domain != 'ntt'
        ):
            result = e.coefficient_domain_to_ntt_domain(result)
        return result

    @staticmethod
    def _evaluate_blocks(
        arithmetic: BootstrapArithmetic,
        x: Ciphertext,
        plan: ChebyshevBlockPlan,
        key: RelinearizationKey
        | EvaluationKeySet
        | Mapping[int, RelinearizationKey],
    ) -> Ciphertext:
        e = arithmetic.engine
        ops = arithmetic.for_input(x, plan.required_depths - 1)
        final = x.depth + plan.required_depths
        m = plan.baby_step
        count = len(plan.coefficients) - 1
        values = {1: x}
        advanced: dict[tuple[int, int], Ciphertext] = {}

        def at_depth(index: int, target_depth: int) -> Ciphertext:
            value = values[index]
            while value.depth < target_depth:
                key = (index, value.depth + 1)
                cached = advanced.get(key)
                if cached is None:
                    cached = ops.advance_depth(value)
                    advanced[key] = cached
                value = cached
            return value

        for n, _ in plan.basis_depths:
            if n == 1:
                continue
            left, right = n // 2, n // 2 + n % 2
            depth = max(values[left].depth, values[right].depth)
            product = ops.multiply_relinearize_rescale(
                at_depth(left, depth),
                at_depth(right, depth),
                relinearization_key=key,
            )
            doubled = e.add(product, product)
            if n % 2:
                values[n] = e.subtract(doubled, at_depth(1, doubled.depth))
            else:
                values[n] = ops.add_scalar(doubled, -1.0)
        babies = {
            k: e.mod_switch_to_depth(values[k], final - 2) for k in range(1, m)
        }
        giants = {j: at_depth(j * m, final - 1) for j in range(1, count + 1)}
        values.clear()
        advanced.clear()
        for mapping in (babies, giants):
            for k, value in mapping.items():
                mapping[k] = (
                    value
                    if value.polynomial_domain == 'ntt'
                    else e.coefficient_domain_to_ntt_domain(value)
                )
        giant_scale = giants[1].scale
        block_target = (
            arithmetic.target_scales[final]
            * e.config.rescale_divisor(final - 1)
            / giant_scale
        )
        reference_scale = ops.target_scales[final - 2]
        preferred_plain_scale = (
            block_target * e.config.rescale_divisor(final - 2) / reference_scale
        )
        product_scale = _common_product_scale(
            tuple(baby.scale for baby in babies.values()),
            reference_scale * preferred_plain_scale,
        )
        pending: Ciphertext | None = None
        zero_term: Ciphertext | None = None
        for j, coefficients in enumerate(plan.coefficients):
            accumulator: Ciphertext | None = None
            for k in range(1, m):
                baby = babies[k]
                plaintext = _prepare_scalar(
                    e,
                    baby,
                    coefficients[k],
                    product_scale / baby.scale,
                    'ntt',
                    constant_cache=arithmetic.constant_cache,
                    prepare_materials=arithmetic._prepare_materials,
                )
                term = e.multiply_plaintext(baby, plaintext)
                accumulator = (
                    term if accumulator is None else e.add(accumulator, term)
                )
            assert accumulator is not None
            if not arithmetic.retain_ntt:
                accumulator = e.ntt_domain_to_coefficient_domain(accumulator)
            block = e.rescale_to_next_depth(accumulator)
            block = ops.add_scalar(block, coefficients[0])
            block_ntt = (
                block
                if block.polynomial_domain == 'ntt'
                else e.coefficient_domain_to_ntt_domain(block)
            )
            if j == 0:
                one = _prepare_scalar(
                    e,
                    block,
                    1.0,
                    giant_scale,
                    'ntt',
                    constant_cache=arithmetic.constant_cache,
                    prepare_materials=arithmetic._prepare_materials,
                )
                zero_term = e.multiply_plaintext(block_ntt, one)
            else:
                product = e.multiply(block_ntt, giants.pop(j))
                pending = (
                    product if pending is None else e.add(pending, product)
                )
        assert pending is not None and zero_term is not None
        pending = arithmetic.relinearize(
            pending,
            key,
            rescale=True,
            output_domain='ntt' if arithmetic.retain_ntt else 'coefficient',
        )
        if not arithmetic.retain_ntt:
            zero_term = e.ntt_domain_to_coefficient_domain(zero_term)
        return e.add(pending, e.rescale_to_next_depth(zero_term))
