"""Assess complete HE moduli against one exact published parameter table.

The built-in budgets reproduce the parameter limits published in
[*Security Guidelines for Implementing Homomorphic
Encryption*](https://doi.org/10.62056/anxra69p1), IACR Communications in
Cryptology 2025. The source data was generated with lattice-estimator and the
classical RC.MATZOV cost model measured in ring operations. Its attack set
includes ``primal_usvp``, ``primal_bdd``, ``hybrid_bdd`` (for
``N <= 2^14``), and ``hybrid_dual``. These values are estimator outputs rather
than security proofs. The functions perform no interpolation or extrapolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fhelium.config import CkksConfig


_TABLE_ERROR_STANDARD_DEVIATION = 3.19
_TABLE_TARGET_BITS = frozenset({128, 192, 256})
_TABLE_SECRET_DISTRIBUTIONS = frozenset({"gaussian", "ternary"})

# Keys are (secret distribution, classical target bits, ring dimension N);
# values are the maximum base-two bit widths of the complete modulus q.
_TABLE_MAXIMUM_MODULUS_BITS: dict[tuple[str, int, int], int] = {
    # Uniform ternary secret.
    ("ternary", 128, 1024): 26,
    ("ternary", 128, 2048): 53,
    ("ternary", 128, 4096): 106,
    ("ternary", 128, 8192): 214,
    ("ternary", 128, 16384): 430,
    ("ternary", 128, 32768): 868,
    ("ternary", 128, 65536): 1747,
    ("ternary", 128, 131072): 3523,
    ("ternary", 192, 2048): 36,
    ("ternary", 192, 4096): 73,
    ("ternary", 192, 8192): 147,
    ("ternary", 192, 16384): 297,
    ("ternary", 192, 32768): 597,
    ("ternary", 192, 65536): 1199,
    ("ternary", 192, 131072): 2411,
    ("ternary", 256, 2048): 27,
    ("ternary", 256, 4096): 56,
    ("ternary", 256, 8192): 114,
    ("ternary", 256, 16384): 230,
    ("ternary", 256, 32768): 462,
    ("ternary", 256, 65536): 929,
    ("ternary", 256, 131072): 1866,
    # Discrete-Gaussian secret with the table's parameters.
    ("gaussian", 128, 1024): 28,
    ("gaussian", 128, 2048): 55,
    ("gaussian", 128, 4096): 108,
    ("gaussian", 128, 8192): 216,
    ("gaussian", 128, 16384): 432,
    ("gaussian", 128, 32768): 870,
    ("gaussian", 128, 65536): 1749,
    ("gaussian", 128, 131072): 3525,
    ("gaussian", 192, 2048): 38,
    ("gaussian", 192, 4096): 75,
    ("gaussian", 192, 8192): 149,
    ("gaussian", 192, 16384): 299,
    ("gaussian", 192, 32768): 599,
    ("gaussian", 192, 65536): 1201,
    ("gaussian", 192, 131072): 2413,
    ("gaussian", 256, 2048): 30,
    ("gaussian", 256, 4096): 58,
    ("gaussian", 256, 8192): 116,
    ("gaussian", 256, 16384): 232,
    ("gaussian", 256, 32768): 464,
    ("gaussian", 256, 65536): 931,
    ("gaussian", 256, 131072): 1868,
}


def _lookup_maximum_modulus_bits(
    *,
    ring_dimension: int,
    target_bits: int,
    secret_distribution: str,
    error_stddev: float,
) -> int | None:
    """Return one exact table entry, or ``None`` when none matches."""

    if error_stddev != _TABLE_ERROR_STANDARD_DEVIATION:
        return None
    return _TABLE_MAXIMUM_MODULUS_BITS.get(
        (secret_distribution, target_bits, ring_dimension)
    )


@dataclass(frozen=True, slots=True)
class SecurityAssessment:
    r"""Immutable result of one exact built-in parameter assessment.

    ``status`` is ``"meets"`` when $\lceil\log_2 q\rceil$ is at most the exact
    table budget, ``"exceeds"`` when it is larger, and ``"unsupported"`` when
    no exact table row matches the assumptions. An unsupported result has
    ``None`` for ``maximum_modulus_bits`` and ``modulus_margin_bits``. A
    negative modulus margin reports how far a supported parameter tuple
    exceeds its modulus budget; it is not a bit-security margin.
    """

    status: Literal["meets", "exceeds", "unsupported"]
    ring_dimension: int
    target_bits: int
    secret_distribution: str
    error_stddev: float
    modulus_bits: int
    maximum_modulus_bits: int | None
    modulus_margin_bits: int | None
    reason: str | None


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _requested_modulus_bits(
    *,
    modulus: int | None,
    moduli: Sequence[int] | None,
) -> int:
    if (modulus is None) == (moduli is None):
        raise ValueError("Specify exactly one of modulus or moduli")

    if modulus is not None:
        product = _positive_integer("modulus", modulus)
        if product < 2:
            raise ValueError("modulus must be at least 2")
    else:
        if isinstance(moduli, (str, bytes)) or not isinstance(moduli, Sequence):
            raise TypeError("moduli must be a sequence of integers")
        if len(moduli) == 0:
            raise ValueError("moduli must contain at least one modulus")
        exact_moduli = tuple(
            _positive_integer(f"moduli[{index}]", value)
            for index, value in enumerate(moduli)
        )
        if any(value < 2 for value in exact_moduli):
            raise ValueError("every entry in moduli must be at least 2")
        product = math.prod(exact_moduli)

    # For positive integer q, ceil(log2(q)) is exact in integer arithmetic.
    return (product - 1).bit_length()


def assess_security(
    ring_dimension: int,
    *,
    modulus: int | None = None,
    moduli: Sequence[int] | None = None,
    target_bits: int = 128,
    secret_distribution: str = "ternary",
    error_stddev: float = _TABLE_ERROR_STANDARD_DEVIATION,
) -> SecurityAssessment:
    """Assess a complete modulus against one exact built-in budget row.

    Args:
        ring_dimension: Polynomial-ring dimension ``N``.
        modulus: Exact complete parameter modulus ``q``.  For CKKS hybrid key
            switching this is ``Q * P``.
        moduli: Exact factors of the complete parameter modulus.  Specify this
            or ``modulus``, but not both.
        target_bits: Classical security category.
        secret_distribution: Exact table secret distribution, ``"ternary"``
            or ``"gaussian"``.
        error_stddev: Gaussian error standard deviation. The built-in budgets
            support exactly ``3.19``.

    Returns:
        A structured assessment with status, exact integer modulus-bit width,
        budget, margin, and an unsupported reason when applicable. Parameters
        without an exact row return ``status="unsupported"``; this function
        never interpolates or extrapolates.

    Raises:
        TypeError: If an input has the wrong structural type.
        ValueError: If a numeric input is non-positive or non-finite, or the
            modulus inputs are missing or ambiguous.
    """

    ring_dimension = _positive_integer("ring_dimension", ring_dimension)
    target_bits = _positive_integer("target_bits", target_bits)
    if not isinstance(secret_distribution, str):
        raise TypeError("secret_distribution must be a string")
    if not isinstance(error_stddev, (int, float)) or isinstance(
        error_stddev, bool
    ):
        raise TypeError("error_stddev must be a real number")
    error_stddev = float(error_stddev)
    if not math.isfinite(error_stddev) or error_stddev <= 0.0:
        raise ValueError("error_stddev must be positive and finite")

    requested_bits = _requested_modulus_bits(modulus=modulus, moduli=moduli)
    maximum_bits = _lookup_maximum_modulus_bits(
        ring_dimension=ring_dimension,
        target_bits=target_bits,
        secret_distribution=secret_distribution,
        error_stddev=error_stddev,
    )

    reason: str | None = None
    if maximum_bits is None:
        reasons: list[str] = []
        if target_bits not in _TABLE_TARGET_BITS:
            reasons.append(f"Unsupported target_bits={target_bits} category.")
        if secret_distribution not in _TABLE_SECRET_DISTRIBUTIONS:
            reasons.append(
                f"Unsupported secret_distribution={secret_distribution!r}."
            )
        if error_stddev != _TABLE_ERROR_STANDARD_DEVIATION:
            reasons.append(
                "The built-in budgets require error_stddev exactly "
                f"{_TABLE_ERROR_STANDARD_DEVIATION}."
            )
        if (
            target_bits in _TABLE_TARGET_BITS
            and secret_distribution in _TABLE_SECRET_DISTRIBUTIONS
            and error_stddev == _TABLE_ERROR_STANDARD_DEVIATION
        ):
            reasons.append(
                "No exact built-in row matches "
                f"ring_dimension={ring_dimension}, target_bits={target_bits}."
            )
        reasons.append(
            "Parameters outside the exact table require an external security "
            "assessment; see the security guide."
        )
        reason = " ".join(reasons)
        status: Literal["meets", "exceeds", "unsupported"] = "unsupported"
        modulus_margin_bits = None
    else:
        modulus_margin_bits = maximum_bits - requested_bits
        status = "meets" if modulus_margin_bits >= 0 else "exceeds"

    return SecurityAssessment(
        status=status,
        ring_dimension=ring_dimension,
        target_bits=target_bits,
        secret_distribution=secret_distribution,
        error_stddev=error_stddev,
        modulus_bits=requested_bits,
        maximum_modulus_bits=maximum_bits,
        modulus_margin_bits=modulus_margin_bits,
        reason=reason,
    )


def assess_config_security(config: CkksConfig) -> SecurityAssessment:
    """Assess a :class:`~fhelium.config.CkksConfig` complete QP modulus."""

    # Import lazily to keep the utility module independent of config import
    # order while still providing an exact public type check.
    from fhelium.config import CkksConfig

    if not isinstance(config, CkksConfig):
        raise TypeError("config must be a CkksConfig")
    return assess_security(
        config.N,
        moduli=config.moduli,
        target_bits=config.security_bits,
        secret_distribution="ternary",
        error_stddev=config.sigma,
    )


__all__ = ["SecurityAssessment", "assess_config_security", "assess_security"]
