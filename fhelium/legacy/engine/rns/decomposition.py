"""Hybrid-RNS key-switch decomposition metadata."""

from __future__ import annotations

from dataclasses import dataclass

from fhelium.legacy.engine.rns.chain import RnsChain


@dataclass(frozen=True)
class RnsDecompositionDigit:
    """One active hybrid-RNS digit and its stable key-storage digit index."""

    key_digit_index: int
    prime_ids: tuple[int, ...]


class HybridRnsDecomposition:
    r"""Partition $Q$ into composite RNS digits for hybrid key switching.

    Scale-prime rows are split into fixed digits whose maximum width is
    $|P|$.  The base Q prime forms the final singleton digit.  Dropping a Q
    prefix at a later depth may shrink or remove an active digit, but every
    remaining digit retains its depth-zero ``key_digit_index`` so it selects
    the correct axis of the depth-zero key-switching key. The enumeration index
    returned at one depth is only local ``digit_index`` and is intentionally
    distinct from that stable key index.
    """

    def __init__(self, chain: RnsChain) -> None:
        self.chain = chain
        scale_prime_ids = chain.q_prime_ids[:-1]
        digit_width = chain.num_p_primes
        scale_digits = tuple(
            tuple(scale_prime_ids[start : start + digit_width])
            for start in range(0, len(scale_prime_ids), digit_width)
        )
        self.depth_zero_digits = scale_digits + ((chain.base_q_prime_id,),)

    @property
    def digit_count(self) -> int:
        return len(self.depth_zero_digits)

    def digits_at_depth(self, depth: int) -> tuple[RnsDecompositionDigit, ...]:
        self.chain.check_depth(depth)
        active: list[RnsDecompositionDigit] = []
        for key_digit_index, fixed_digit in enumerate(self.depth_zero_digits):
            prime_ids = tuple(
                prime_id for prime_id in fixed_digit if prime_id >= depth
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

        return tuple(
            tuple(prime_id - depth for prime_id in digit.prime_ids)
            for digit in self.digits_at_depth(depth)
        )
