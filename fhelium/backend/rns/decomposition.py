"""Hybrid-RNS key-switch decomposition metadata."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from fhelium.backend.rns.chain import RnsChain


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
    the correct axis of the depth-zero key-switching key. The enumeration index
    returned at one depth is only local ``digit_index`` and is intentionally
    distinct from that stable key index.
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
        self.depth_zero_digits = tuple(digits)

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
