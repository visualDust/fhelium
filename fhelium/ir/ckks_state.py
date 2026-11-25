"""Metadata equations shared by CKKS execution and callable capture.

These functions derive scale, prime-row, and representation metadata from operation inputs.
"""

from __future__ import annotations

from fractions import Fraction

from typing import Literal, TypeVar

from fhelium.config import CkksConfig
from fhelium.values._scale import coerce_scale
from fhelium.values.state import (
    ModulusBasis,
    PolynomialDomain,
    ResidueRepresentation,
)


_V = TypeVar("_V")


def product_scale(lhs: float, rhs: float) -> float:
    r"""Return $\Delta_{out}=\Delta_{lhs}\Delta_{rhs}$ without rescaling."""

    return lhs * rhs


def quotient_scale(scale: float, divisor: int | Fraction) -> float:
    r"""Return $\Delta_{out}=\Delta R/D$ for the rescale ratio $D/R$."""

    return scale / float(divisor)


def output_residues(domain: PolynomialDomain) -> ResidueRepresentation:
    """Return the ciphertext residue representation for a selected domain."""

    if domain not in {"coefficient", "ntt"}:
        raise ValueError("output_domain must be 'coefficient' or 'ntt'")
    return "montgomery" if domain == "ntt" else "standard"


def transform_state(
    direction: Literal["forward", "inverse"], *, ciphertext: bool
) -> tuple[PolynomialDomain, ResidueRepresentation]:
    """Return the public state after a forward or inverse negacyclic NTT.

    Forward transforms produce NTT/Montgomery values. Inverse transforms produce
    coefficient/standard ciphertexts or coefficient/Montgomery plaintexts.
    """

    if direction == "forward":
        return "ntt", "montgomery"
    return "coefficient", "standard" if ciphertext else "montgomery"


def depth_prime_ids(
    config: CkksConfig, depth: int, basis: ModulusBasis
) -> tuple[int, ...]:
    """Select the Q basis at a depth and append the P rows for a QP value."""

    q = config.q_prime_ids(depth)
    if basis == "QP":
        return (*q, *range(config.num_q_primes, config.total_num_primes))
    return q


def reinterpreted_scale(
    current: float, target: float, max_relative_change: float | None = None
) -> float:
    """Validate the caller's metadata-only scale change and return its target."""

    from fhelium.errors import ScaleMismatchError

    scale = coerce_scale(target, value_name="target_scale")
    if max_relative_change is not None:
        limit = float(max_relative_change)
        if limit < 0:
            raise ValueError("max_relative_change must be non-negative")
        change = max(current / scale, scale / current) - 1.0
        if change > limit:
            raise ScaleMismatchError(
                operation="reinterpret_at_scale",
                lhs_name="current",
                lhs_scale=current,
                rhs_name="target",
                rhs_scale=scale,
            )
    return scale


__all__ = [
    "depth_prime_ids",
    "output_residues",
    "product_scale",
    "quotient_scale",
    "reinterpreted_scale",
    "transform_state",
]


def allocation_state(state: dict[str, _V], **updates: _V) -> dict[str, _V]:
    """Preserve mathematical metadata while dropping an allocation's old layout."""
    return {
        **{
            name: value
            for name, value in state.items()
            if name not in {"shape", "strides", "requires_grad"}
        },
        **updates,
    }
