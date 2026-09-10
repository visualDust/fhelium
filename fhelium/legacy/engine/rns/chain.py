"""Mathematical Q/P modulus-chain layout for rank-local CKKS values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RnsChain:
    r"""Prime identities and active bases for a CKKS modulus chain.

    Prime ids follow the engine's parameter order ``[Q | P]``.
    For Q ids ``(0, ..., num_q_primes - 1)`` and P ids in the fixed suffix,
    public depth $\ell$ has

    $$
    I_\ell=(\ell,\ldots,\mathtt{num\_q\_primes}-1),\qquad
    Q_\ell=\prod_{i\in I_\ell}q_i.
    $$

    ``include_p=True`` appends every P id and therefore describes $Q_\ell P$;
    it is an internal layout selector rather than semantic value metadata.
    These ids map limb row ``j`` to parameter prime ``prime_ids[j]`` exactly.
    """

    num_q_primes: int
    num_p_primes: int

    def __post_init__(self) -> None:
        if self.num_q_primes < 1:
            raise ValueError("a CKKS RNS chain requires at least one Q prime")
        if self.num_p_primes < 1:
            raise ValueError("key switching requires at least one P prime")

    @property
    def rns_basis_depth_count(self) -> int:
        return self.num_q_primes

    @property
    def total_modulus_count(self) -> int:
        return self.num_q_primes + self.num_p_primes

    @property
    def q_prime_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_q_primes))

    @property
    def p_prime_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_q_primes, self.total_modulus_count))

    @property
    def base_q_prime_id(self) -> int:
        return self.num_q_primes - 1

    def check_depth(self, depth: int) -> None:
        if not 0 <= depth < self.rns_basis_depth_count:
            raise ValueError(
                f"depth must be in [0, {self.rns_basis_depth_count}), got {depth}"
            )

    def q_prime_ids_at_depth(self, depth: int) -> tuple[int, ...]:
        r"""Return ordered ids for $Q_\ell$."""

        self.check_depth(depth)
        return tuple(range(depth, self.num_q_primes))

    def qp_prime_ids_at_depth(self, depth: int) -> tuple[int, ...]:
        r"""Return ordered ids for $Q_\ell P$."""

        return self.q_prime_ids_at_depth(depth) + self.p_prime_ids

    def prime_ids(
        self, depth: int, *, include_p: bool = False
    ) -> tuple[int, ...]:
        return (
            self.qp_prime_ids_at_depth(depth)
            if include_p
            else self.q_prime_ids_at_depth(depth)
        )

    def parameter_rows(self, depth: int, *, include_p: bool = False) -> slice:
        """Return the zero-copy interval for the selected basis."""

        self.check_depth(depth)
        stop = self.total_modulus_count if include_p else self.num_q_primes
        return slice(depth, stop)
