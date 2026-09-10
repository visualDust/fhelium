"""Mathematical Q/P modulus-chain layout for rank-local CKKS values."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RnsChain:
    r"""Prime identities and active bases for a CKKS modulus chain.

    Prime ids follow the RNS parameter order ``[Q | P]``. Q rows are
    partitioned by ``q_depth_group_sizes``. If $s_d$ is the flattened row
    offset of group $d$, the active Q identifiers are

    $$
    I_d=(s_d,\ldots,\mathtt{num\_q\_primes}-1),\qquad
    Q_d=\prod_{i\in I_d}q_i.
    $$

    ``include_p=True`` appends every P id and therefore describes $Q_dP$;
    it is an internal layout selector rather than semantic value metadata.
    These ids map limb row ``j`` to parameter prime ``prime_ids[j]`` exactly.
    """

    num_q_primes: int
    num_p_primes: int
    q_depth_group_sizes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_q_primes < 1:
            raise ValueError("a CKKS RNS chain requires at least one Q prime")
        if self.num_p_primes < 1:
            raise ValueError("key switching requires at least one P prime")
        sizes = self.q_depth_group_sizes
        if sizes is None:
            sizes = (1,) * self.num_q_primes
            object.__setattr__(self, "q_depth_group_sizes", sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("Q depth groups must contain positive row counts")
        if sum(sizes) != self.num_q_primes:
            raise ValueError("Q depth groups must cover every Q prime")

    @property
    def basis_row_starts(self) -> tuple[int, ...]:
        """Return the flattened Q-row offset of every chain position."""

        assert self.q_depth_group_sizes is not None
        starts = [0]
        for size in self.q_depth_group_sizes[:-1]:
            starts.append(starts[-1] + size)
        return tuple(starts)

    @property
    def basis_count(self) -> int:
        assert self.q_depth_group_sizes is not None
        return len(self.q_depth_group_sizes)

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
        return tuple(range(self.basis_row_starts[depth], self.num_q_primes))

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
        """Return the zero-copy interval for the selected basis."""

        self.check_depth(depth)
        stop = self.total_modulus_count if include_p else self.num_q_primes
        return slice(self.basis_row_starts[depth], stop)
