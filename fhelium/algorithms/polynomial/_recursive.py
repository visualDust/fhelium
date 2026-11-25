r"""Recursive split evaluation of encrypted Chebyshev series."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from fhelium.algorithms._arithmetic import Arithmetic, _prepare_scalar
from fhelium.algorithms.polynomial._approximation import Polynomial
from fhelium.algorithms.polynomial._evaluation import (
    _validate_polynomial_coefficients,
)
from fhelium.values import Ciphertext, EvaluationKeySet, RelinearizationKey


@dataclass(frozen=True)
class _Node:
    """One factor of the split tree, evaluated at ``output_depth``.

    Depths are relative to the input. A node multiplies its terms at
    ``output_depth - rescale`` and rescales once when ``rescale`` is set.
    """

    coefficients: tuple[complex, ...]
    output_depth: int
    rescale: bool
    split_degree: int = 0
    low: _Node | None = None
    high: _Node | None = None
    low_constant: complex = 0j
    high_constant: complex = 0j

    @property
    def working_depth(self) -> int:
        return self.output_depth - int(self.rescale)

    @property
    def product_components(self) -> int:
        three = self.high is not None or (
            self.low is not None and self.low.components == 3
        )
        return 3 if three else 2

    @property
    def components(self) -> int:
        return 2 if self.rescale else self.product_components


@dataclass(frozen=True)
class _Basis:
    """$T_n=2T_{left}T_{right}-T_{subtract}$ produced at relative ``depth``."""

    degree: int
    left: int
    right: int
    subtract: int
    depth: int


@dataclass(frozen=True)
class _Schedule:
    required_depths: int
    root: _Node
    basis: tuple[_Basis, ...]
    inventory: tuple[tuple[str, int], ...]
    relinearization_depths: tuple[int, ...]


def _trim(coefficients) -> tuple[complex, ...]:
    values = list(coefficients)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _tree(coefficients, margin, baby_step, output_depth, rescale) -> _Node:
    degree = len(coefficients) - 1
    leaf_depth = (degree - 1).bit_length() + 1 if degree else 1
    if degree < baby_step and leaf_depth <= margin:
        return _Node(coefficients, output_depth, rescale)
    split = 1 << (degree.bit_length() - 1)
    # T_{m+j} = 2 T_m T_j - T_{m-j} moves a_{m+j} into H and L.
    low, high = list(coefficients[:split]), list(coefficients[split:])
    for j in range(1, len(high)):
        low[split - j] -= high[j]
        high[j] *= 2
    low, high = _trim(low), _trim(high)
    working = output_depth - int(rescale)
    return _Node(
        coefficients,
        output_depth,
        rescale,
        split,
        _tree(low, margin, baby_step, working, False) if len(low) > 1 else None,
        _tree(high, margin - 1, baby_step, working, True)
        if len(high) > 1
        else None,
        low[0] if len(low) == 1 else 0j,
        high[0] if len(high) == 1 else 0j,
    )


def _walk(node: _Node):
    yield node
    for child in (node.low, node.high):
        if child is not None:
            yield from _walk(child)


def _leaf_terms(node: _Node) -> tuple[tuple[int, complex], ...]:
    return tuple(
        (degree, value)
        for degree, value in enumerate(node.coefficients)
        if degree and value != 0
    ) or ((1, 0j),)


@lru_cache(maxsize=64)
def _schedule(coefficients: tuple[complex, ...], baby_step: int | None):
    degree = len(coefficients) - 1
    required = max(1, degree.bit_length())
    baby = baby_step or 1 << ((required + 1) // 2)
    root = _tree(coefficients, required, baby, required, True)
    bases: dict[int, _Basis] = {}

    def basis(n: int) -> int:
        if n == 1:
            return 0
        if n in bases:
            return bases[n].depth
        left = n // 2 if n & (n - 1) == 0 else 1 << (n.bit_length() - 1)
        right, subtract = n - left, 2 * left - n
        depth = max(basis(left), basis(right)) + 1
        if subtract:
            basis(subtract)
        bases[n] = _Basis(n, left, right, subtract, depth)
        return depth

    coefficient_products = ciphertext_products = rescales = 0
    relinearizations = []
    for node in _walk(root):
        if not node.split_degree:
            for n, _ in _leaf_terms(node):
                basis(n)
                coefficient_products += 1
        else:
            basis(node.split_degree)
            if node.high is None:
                coefficient_products += 1
            else:
                ciphertext_products += 1
        if node.rescale:
            rescales += 1
            if node.product_components == 3:
                relinearizations.append(node.working_depth)
    steps = tuple(bases[n] for n in sorted(bases))
    coefficient_products += sum(bool(step.subtract) for step in steps)
    ciphertext_products += len(steps)
    rescales += len(steps)
    relinearizations.extend(step.depth - 1 for step in steps)
    inventory = {
        'ciphertext_multiplications': ciphertext_products,
        'coefficient_multiplications': coefficient_products,
        'alignment_multiplications': 0,
        'relinearizations': len(relinearizations),
        'rescale_operations': rescales,
        'total_multiplications': ciphertext_products + coefficient_products,
    }
    return _Schedule(
        required,
        root,
        steps,
        tuple(inventory.items()),
        tuple(sorted(set(relinearizations))),
    )


def _factor(scale: float, target: float) -> float:
    """Return a plaintext scale $c$ near ``target/scale`` with ``scale*c == target``.

    Binary64 products are searched over a few neighboring values of $c$; when
    none is exact, the nearest value is returned and the caller reinterprets
    the product's scale.
    """

    candidate = target / scale
    below = above = candidate
    for _ in range(8):
        for value in (above, below):
            if scale * value == target:
                return value
        above = math.nextafter(above, math.inf)
        below = math.nextafter(below, 0.0)
    return candidate


@dataclass(frozen=True)
class RecursiveChebyshevPlan:
    r"""Evaluate a Chebyshev series by recursive splits at powers of two.

    For a series $p$ of degree $d$ and $m=2^{\lfloor\log_2 d\rfloor}$, the
    identity $T_{m+j}=2T_mT_j-T_{m-j}$ gives $p=L+T_m H$ with $\deg L<m$ and
    $\deg H=d-m<m$; $L$ and $H$ split recursively until a factor of degree
    below ``baby_step`` (default $2^{\lceil r/2\rceil}$ for
    $r=\lceil\log_2(d+1)\rceil$) can be summed directly over the basis values
    $T_j$ within its depth. The result consumes $r$ depths.

    Basis values follow $T_{a+b}=2T_aT_b-T_{a-b}$ with $a$ the largest power of
    two below $a+b$ ($T_{2a}=2T_a^2-1$): the subtracted term multiplies
    $T_{a-b}$ by the constant $-1$ at the product's scale, so each basis value
    costs one product, one relinearization and one rescale.

    $H$ is rescaled before its product with $T_m$; $L$ is accumulated at the
    parent's product scale without relinearization or rescale, so a
    three-component $T_mH$ and the terms of $L$ share the parent's single
    relinearization and rescale. Constant terms of unrescaled factors are
    added after the parent's rescale.

    Scales follow the actual input scale: basis values follow
    $s_{a+b}=s_as_b/\rho$, and every factor accumulates at one product scale
    reached by plaintext constants of scale $s/s_j$. ``working_scale`` is the
    root's product scale before its rescale; the default is the binary64
    successor of the power of two nearest $\Delta_{out}\rho$, where
    $\Delta_{out}$ is the evaluation's target output scale. The output scale is
    ``working_scale`` divided by the last rescale ratio. Inputs at depth $d_0$
    require nested bases $B_{d_0}\supseteq\cdots\supseteq B_{d_0+r}$.
    """

    baby_step: int | None = None
    working_scale: float | None = None

    def __post_init__(self) -> None:
        if self.baby_step is not None and self.baby_step < 2:
            raise ValueError('baby_step must be at least two')
        if self.working_scale is not None and not (
            math.isfinite(self.working_scale) and self.working_scale > 0
        ):
            raise ValueError('working_scale must be finite and positive')

    def _schedule(self, polynomial: Polynomial) -> _Schedule:
        if polynomial.basis != 'chebyshev':
            raise ValueError('RecursiveChebyshevPlan requires Chebyshev basis')
        _validate_polynomial_coefficients(polynomial)
        return _schedule(_trim(polynomial.coefficients), self.baby_step)

    def required_depths(self, polynomial: Polynomial) -> int:
        """Return $\\lceil\\log_2(d+1)\\rceil$ for degree $d$ (at least one)."""
        return self._schedule(polynomial).required_depths

    def operation_inventory(self, polynomial: Polynomial) -> dict[str, int]:
        """Count products, relinearizations and rescales of one evaluation."""
        return dict(self._schedule(polynomial).inventory)

    def relinearization_depths(
        self, polynomial: Polynomial, *, input_depth: int = 0
    ) -> tuple[int, ...]:
        """Return the depths of the three-component values relinearized."""
        return tuple(
            input_depth + depth
            for depth in self._schedule(polynomial).relinearization_depths
        )

    def evaluate(
        self,
        arithmetic: Arithmetic,
        ciphertext: Ciphertext,
        polynomial: Polynomial,
        *,
        relinearization_key: RelinearizationKey
        | EvaluationKeySet
        | Mapping[int, RelinearizationKey]
        | None = None,
    ) -> Ciphertext:
        """Evaluate the schedule; the result keeps the input polynomial domain."""

        schedule = self._schedule(polynomial)
        engine = arithmetic.engine
        config = engine.config
        entry = ciphertext.depth
        for depth in range(entry, entry + schedule.required_depths):
            if not config.is_subbasis(depth, depth + 1):
                raise ValueError(
                    'RecursiveChebyshevPlan requires nested bases over the '
                    'consumed depths'
                )
        if schedule.relinearization_depths and relinearization_key is None:
            raise ValueError(
                'nonlinear Chebyshev evaluation requires relinearization key'
            )
        divisors = tuple(
            float(config.rescale_divisor(entry + depth))
            for depth in range(schedule.required_depths)
        )
        domain = ciphertext.polynomial_domain
        x = (
            ciphertext
            if domain == 'ntt'
            else engine.coefficient_domain_to_ntt_domain(ciphertext)
        )
        values: dict[int, Ciphertext] = {1: x}
        restricted: dict[tuple[int, int], Ciphertext] = {}

        def at_depth(degree: int, relative: int) -> Ciphertext:
            value = values[degree]
            if value.depth == entry + relative:
                return value
            token = (degree, relative)
            if token not in restricted:
                restricted[token] = engine.mod_switch_to_depth(
                    value, entry + relative
                )
            return restricted[token]

        def at_scale(value: Ciphertext, scale: float) -> Ciphertext:
            # ``value`` is a fresh product; relabel its scale by at most a
            # few binary64 units.
            if value.scale != scale:
                value.scale = scale
            return value

        def constant_product(value: Ciphertext, constant: complex, target):
            plaintext = _prepare_scalar(
                engine,
                value,
                constant,
                _factor(value.scale, target),
                'ntt',
                constant_cache=arithmetic.constant_cache,
                prepare_materials=arithmetic._prepare_materials,
            )
            return at_scale(engine.multiply_plaintext(value, plaintext), target)

        def finish(value: Ciphertext, components: int) -> Ciphertext:
            if components == 3:
                # Three-component products occur only at relinearization depths.
                assert relinearization_key is not None
                return arithmetic.relinearize(
                    value,
                    relinearization_key,
                    rescale=True,
                    output_domain='ntt',
                )
            return engine.rescale_to_next_depth(value)

        for step in schedule.basis:
            depth = step.depth - 1
            product = engine.multiply(
                at_depth(step.left, depth), at_depth(step.right, depth)
            )
            product = engine.add(product, product)
            if step.subtract:
                product = engine.add(
                    product,
                    constant_product(
                        at_depth(step.subtract, depth), -1.0, product.scale
                    ),
                )
            result = finish(product, 3)
            values[step.degree] = (
                result if step.subtract else arithmetic.add_scalar(result, -1.0)
            )

        def evaluate(node: _Node, scale: float) -> tuple[Ciphertext, complex]:
            if not node.split_degree:
                accumulator = None
                for degree, value in _leaf_terms(node):
                    term = constant_product(
                        at_depth(degree, node.working_depth), value, scale
                    )
                    accumulator = (
                        term
                        if accumulator is None
                        else engine.add(accumulator, term)
                    )
                pending = node.coefficients[0]
            else:
                split = at_depth(node.split_degree, node.working_depth)
                if node.high is None:
                    accumulator = constant_product(
                        split, node.high_constant, scale
                    )
                else:
                    high, _ = evaluate(
                        node.high,
                        scale / split.scale * divisors[node.high.working_depth],
                    )
                    accumulator = at_scale(engine.multiply(high, split), scale)
                if node.low is None:
                    pending = node.low_constant
                else:
                    low, pending = evaluate(node.low, scale)
                    accumulator = engine.add(accumulator, low)
            assert accumulator is not None
            if node.rescale:
                accumulator = finish(accumulator, node.product_components)
                if pending:
                    accumulator = arithmetic.add_scalar(accumulator, pending)
                pending = 0j
            return accumulator, pending

        working = self.working_scale
        if working is None:
            target = arithmetic.target_scales[entry + schedule.required_depths]
            working = math.nextafter(
                math.ldexp(1.0, round(math.log2(target * divisors[-1]))),
                math.inf,
            )
        result, _ = evaluate(schedule.root, working)
        if domain == 'coefficient' or not arithmetic.retain_ntt:
            result = engine.ntt_domain_to_coefficient_domain(result)
        return result
