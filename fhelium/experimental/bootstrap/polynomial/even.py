"""Even Chebyshev evaluation through the T2 composition identity."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass, replace
from typing import Any, cast

from fhelium.values import EvaluationKeySet, Ciphertext, RelinearizationKey
from fhelium.experimental.bootstrap.arithmetic import BootstrapArithmetic
from .approximation import PolynomialApproximation


@dataclass(frozen=True)
class EvenChebyshevEvaluator:
    r"""Evaluate sum a_(2j) T_(2j)(x) as sum a_(2j) T_j(T2(x)).

    ``evaluator`` selects the inner polynomial schedule. Forming T2(x)=2x²−1
    consumes one ciphertext multiplication, relinearization and complete Q
    group rescale. The inner evaluator consumes the resulting actual scale.
    Odd coefficients must be zero; constants use the inner evaluator directly.
    Output preserves input representation and batch axes.
    """

    evaluator: Any

    @staticmethod
    def _fold(polynomial: PolynomialApproximation) -> PolynomialApproximation:
        if polynomial.basis != "chebyshev" or any(
            polynomial.coefficients[1::2]
        ):
            raise ValueError(
                "Even Chebyshev evaluation requires zero odd coefficients"
            )
        coefficients = list(polynomial.coefficients[::2])
        while len(coefficients) > 1 and coefficients[-1] == 0:
            coefficients.pop()
        return replace(
            polynomial,
            coefficients=tuple(coefficients),
            name=polynomial.name + "_in_T2",
        )

    def required_depths(self, polynomial: PolynomialApproximation) -> int:
        folded = self._fold(polynomial)
        if folded.degree == 0:
            return self.evaluator.required_depths(folded)
        return 1 + self.evaluator.required_depths(folded)

    def operation_inventory(
        self, polynomial: PolynomialApproximation
    ) -> dict[str, int]:
        folded = self._fold(polynomial)
        counts = dict(self.evaluator.operation_inventory(folded))
        if folded.degree > 0:
            for name in (
                "ciphertext_multiplications",
                "relinearizations",
                "rescale_operations",
                "total_multiplications",
            ):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def relinearization_depths(
        self, polynomial: PolynomialApproximation, *, input_depth: int = 0
    ) -> tuple[int, ...] | None:
        """Compose the actual T2 product and the folded evaluator's consumers."""
        folded = self._fold(polynomial)
        declare = getattr(self.evaluator, "relinearization_depths", None)
        if not callable(declare):
            return None
        nonlinear = folded.degree > 0
        depths = cast(
            tuple[int, ...] | None,
            declare(folded, input_depth=input_depth + int(nonlinear)),
        )
        if depths is None:
            return None
        return tuple(
            sorted(set(depths) | ({input_depth} if nonlinear else set()))
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
        folded = self._fold(polynomial)
        value = ciphertext
        if folded.degree > 0:
            if relinearization_key is None:
                raise ValueError(
                    "Even Chebyshev evaluation requires a relinearization key"
                )
            value = arithmetic.multiply_relinearize_rescale(
                value, value, relinearization_key=relinearization_key
            )
            value = arithmetic.engine.add(value, value)
            value = arithmetic.add_scalar(value, -1.0)
        result = self.evaluator.evaluate(
            arithmetic, value, folded, relinearization_key=relinearization_key
        )
        if result.polynomial_domain != ciphertext.polynomial_domain:
            if ciphertext.polynomial_domain == "coefficient":
                result = arithmetic.engine.ntt_domain_to_coefficient_domain(
                    result
                )
            else:
                result = arithmetic.engine.coefficient_domain_to_ntt_domain(
                    result
                )
        return result
