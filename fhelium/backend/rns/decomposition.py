"""Hybrid-RNS key-switch decomposition metadata."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.config import CkksConfig

from fhelium.backend.rns.chain import RnsChain


def _partition_q_digits(
    q_prime_ids: tuple[int, ...],
    q_moduli: tuple[int, ...],
    p_moduli: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Partition aligned Q rows below product(P), retaining oversized singletons."""
    p_product = prod(p_moduli)
    digits: list[tuple[int, ...]] = []
    start = 0
    while start < len(q_moduli):
        stop = start
        product = 1
        while stop < len(q_moduli) and product * q_moduli[stop] < p_product:
            product *= q_moduli[stop]
            stop += 1
        if stop == start:
            stop += 1
        digits.append(tuple(q_prime_ids[start:stop]))
        start = stop
    return tuple(digits)


@dataclass(frozen=True)
class RnsDecompositionDigit:
    """One active hybrid-RNS digit and its stable key-storage digit index."""

    key_digit_index: int
    prime_ids: tuple[int, ...]


class HybridRnsDecomposition:
    r"""Partition $Q$ into composite RNS digits for hybrid key switching.

    Contiguous Q rows form digits whose product is below the special modulus
    $P$. A single Q prime at or above $P$ forms its own digit. Dropping a Q
    prefix at a later depth may shrink or remove an active digit, but every
    remaining digit keeps its depth-zero ``key_digit_index`` so it selects
    the correct axis of the depth-zero key-switching key. ``digit_index`` enumerates the active digits at the requested depth; ``key_digit_index`` identifies their stored key planes.
    """

    def __init__(
        self,
        chain: RnsChain,
        q_moduli: tuple[int, ...],
        p_moduli: tuple[int, ...],
    ) -> None:
        self.chain = chain
        q_prime_ids = chain.q_prime_ids
        if (
            len(q_moduli) != chain.num_q_primes
            or len(p_moduli) != chain.num_p_primes
        ):
            raise ValueError("decomposition moduli differ from the RNS chain")
        self.depth_zero_digits = _partition_q_digits(
            q_prime_ids, q_moduli, p_moduli
        )

    @property
    def digit_count(self) -> int:
        return len(self.depth_zero_digits)

    def digits_at_depth(self, depth: int) -> tuple[RnsDecompositionDigit, ...]:
        self.chain.check_depth(depth)
        start_row = self.chain.basis_row_starts[depth]
        active: list[RnsDecompositionDigit] = []
        for key_digit_index, fixed_digit in enumerate(self.depth_zero_digits):
            prime_ids = tuple(
                prime_id for prime_id in fixed_digit if prime_id >= start_row
            )
            if prime_ids:
                active.append(RnsDecompositionDigit(key_digit_index, prime_ids))
        return tuple(active)

    def digit_rows(
        self, depth: int, *, include_p: bool = False
    ) -> tuple[tuple[int, ...], ...]:
        """Return ordered parameter rows for active Q digits."""

        rows = tuple(digit.prime_ids for digit in self.digits_at_depth(depth))
        if include_p:
            rows += (self.chain.p_prime_ids,)
        return rows

    def component_digit_rows(self, depth: int) -> tuple[tuple[int, ...], ...]:
        """Return digit rows relative to a compact depth-specific component."""

        start_row = self.chain.basis_row_starts[depth]
        return tuple(
            tuple(prime_id - start_row for prime_id in digit.prime_ids)
            for digit in self.digits_at_depth(depth)
        )


def key_decomposition_facts(
    config: CkksConfig, depth: int, key_prime_ids, digit_prime_ids
) -> dict[str, object]:
    """Describe intersections with the Q supports encrypted in stored key planes."""
    q_ids = tuple(range(config.q_row_start(depth), config.num_q_primes))
    key_prime_ids = tuple(key_prime_ids)
    digit_prime_ids = tuple(tuple(rows) for rows in digit_prime_ids)
    stored_q = tuple(i for rows in digit_prime_ids for i in rows)
    p_ids = key_prime_ids[len(stored_q) :]
    if (
        key_prime_ids[: len(stored_q)] != stored_q
        or not p_ids
        or len(set(key_prime_ids)) != len(key_prime_ids)
        or any(not 0 <= i < config.total_num_primes for i in key_prime_ids)
    ):
        raise ValueError(
            "Key rows must store their Q supports before distinct auxiliary catalog rows"
        )
    active = tuple(
        (index, tuple(i for i in rows if i in q_ids))
        for index, rows in enumerate(digit_prime_ids)
    )
    active = tuple((index, rows) for index, rows in active if rows)
    if tuple(i for _, rows in active for i in rows) != q_ids:
        raise ValueError(
            "Key digit supports do not partition the active Q basis"
        )
    key_row_start = key_prime_ids.index(q_ids[0])
    qp_ids = (*q_ids, *p_ids)
    if key_prime_ids[key_row_start : key_row_start + len(qp_ids)] != qp_ids:
        raise ValueError(
            "Active QP rows must form a contiguous physical key slice"
        )
    return {
        "digits": tuple(
            (rows[0] - q_ids[0], rows[-1] + 1 - q_ids[0], index)
            for index, rows in active
        ),
        "q_count": len(q_ids),
        "p_count": len(p_ids),
        "q_prime_ids": q_ids,
        "p_prime_ids": p_ids,
        "qp_prime_ids": qp_ids,
        "key_row_start": key_row_start,
        "key_prime_ids": key_prime_ids,
        "digit_prime_ids": digit_prime_ids,
    }
