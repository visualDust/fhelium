"""Validation and inventory helpers shared by polynomial evaluators."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fhelium.values import Ciphertext, RelinearizationKey

if TYPE_CHECKING:
    from fhelium.eager import Engine
    from fhelium.experimental.bootstrap.polynomial.approximation import (
        PolynomialApproximation,
    )


def _inventory(
    *,
    ciphertext_multiplications: int,
    coefficient_multiplications: int,
    alignment_multiplications: int,
    coefficient_rescales: int | None = None,
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
        'rescale_operations': (
            ciphertext_multiplications
            + alignment_multiplications
            + (
                coefficient_multiplications
                if coefficient_rescales is None
                else coefficient_rescales
            )
        ),
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
    engine: Engine,
    ciphertext: Ciphertext,
    polynomial: PolynomialApproximation,
    *,
    required_depths: int,
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
    if (
        ciphertext.polynomial_domain,
        ciphertext.residue_representation,
    ) not in {('coefficient', 'standard'), ('ntt', 'montgomery')}:
        raise ValueError(
            f'{evaluator_name} requires coefficient/standard or '
            'NTT/Montgomery ciphertext input'
        )
    ciphertext.assert_state(modulus_basis='Q', components=2)
    engine.validate_ciphertext(ciphertext)
    output_depth = ciphertext.depth + required_depths
    if output_depth > engine.max_depth:
        raise ValueError(
            f'{evaluator_name} needs {required_depths} depth transitions '
            f'from entry depth {ciphertext.depth}, but the final public '
            f'depth is {engine.max_depth}'
        )
    if requires_relinearization:
        if relinearization_key is None:
            raise ValueError(f'{evaluator_name} requires a relinearization key')
        if type(relinearization_key) is not RelinearizationKey:
            raise TypeError("relinearization_key must be a RelinearizationKey")
        engine.validate_key_switch_key(relinearization_key)
        if relinearization_key.modulus_basis != 'QP':
            raise ValueError("relinearization_key requires QP basis")
