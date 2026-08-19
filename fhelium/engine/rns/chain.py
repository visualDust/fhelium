"""Mathematical Q/P modulus-chain layout for rank-local CKKS values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RnsChain:
    r"""Prime identities and active bases for a CKKS modulus chain.

    Prime ids follow the engine's canonical parameter order ``[Q | P]``.
    For Q ids ``(0, ..., num_q_primes - 1)`` and P ids in the fixed suffix,
    public level $\ell$ has

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
    def rns_basis_level_count(self) -> int:
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

    def check_level(self, level: int) -> None:
        if not 0 <= level < self.rns_basis_level_count:
            raise ValueError(
                f"level must be in [0, {self.rns_basis_level_count}), got {level}"
            )

    def q_prime_ids_at_level(self, level: int) -> tuple[int, ...]:
        r"""Return ordered ids for $Q_\ell$."""

        self.check_level(level)
        return tuple(range(level, self.num_q_primes))

    def qp_prime_ids_at_level(self, level: int) -> tuple[int, ...]:
        r"""Return ordered ids for $Q_\ell P$."""

        return self.q_prime_ids_at_level(level) + self.p_prime_ids

    def prime_ids(
        self, level: int, *, include_p: bool = False
    ) -> tuple[int, ...]:
        return (
            self.qp_prime_ids_at_level(level)
            if include_p
            else self.q_prime_ids_at_level(level)
        )

    def parameter_rows(self, level: int, *, include_p: bool = False) -> slice:
        """Return the zero-copy interval for the selected canonical basis."""

        self.check_level(level)
        stop = self.total_modulus_count if include_p else self.num_q_primes
        return slice(level, stop)
