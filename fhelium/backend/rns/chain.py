"""Mathematical Q/P modulus-chain layout for rank-local CKKS values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RnsChain:
    r"""Prime identities and active bases for a CKKS modulus chain.

    Prime ids follow the RNS parameter order ``[Q | P]``. ``basis_rows[d]``
    lists the increasing Q prime ids $I_d$ active at depth ``d``:

    $$
    Q_d=\prod_{i\in I_d}q_i.
    $$

    ``include_p=True`` appends every P id and therefore describes $Q_dP$;
    it is an internal layout selector rather than semantic value metadata.
    Limb row ``j`` uses parameter prime ``prime_ids[j]``.
    """

    num_q_primes: int
    num_p_primes: int
    basis_rows: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.num_q_primes < 1:
            raise ValueError("a CKKS RNS chain requires at least one Q prime")
        if self.num_p_primes < 1:
            raise ValueError("key switching requires at least one P prime")
        rows = self.basis_rows
        if rows is None:
            rows = tuple(
                tuple(range(start, self.num_q_primes))
                for start in range(self.num_q_primes)
            )
            object.__setattr__(self, "basis_rows", rows)
        if not rows or any(
            not basis or not 0 <= basis[0] or basis[-1] >= self.num_q_primes
            for basis in rows
        ):
            raise ValueError("Q bases must be nonempty Q prime id lists")

    @property
    def basis_count(self) -> int:
        assert self.basis_rows is not None
        return len(self.basis_rows)

    @property
    def total_modulus_count(self) -> int:
        return self.num_q_primes + self.num_p_primes

    @property
    def q_prime_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_q_primes))

    @property
    def p_prime_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_q_primes, self.total_modulus_count))

    def check_depth(self, depth: int) -> None:
        if not 0 <= depth < self.basis_count:
            raise ValueError(
                f"depth must be in [0, {self.basis_count}), got {depth}"
            )

    def q_prime_ids_at_depth(self, depth: int) -> tuple[int, ...]:
        r"""Return ordered ids for $Q_d$."""

        self.check_depth(depth)
        assert self.basis_rows is not None
        return self.basis_rows[depth]

    def qp_prime_ids_at_depth(self, depth: int) -> tuple[int, ...]:
        r"""Return ordered ids for $Q_dP$."""

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
        """Return the zero-copy parameter interval for the selected basis.

        A basis is one interval when its ids are consecutive; a QP basis
        additionally requires its Q ids to end at the last Q prime. Other
        bases select their rows with :meth:`prime_ids`.
        """

        ids = self.prime_ids(depth, include_p=include_p)
        if ids != tuple(range(ids[0], ids[0] + len(ids))):
            raise ValueError("this basis is not one parameter interval")
        return slice(ids[0], ids[0] + len(ids))
