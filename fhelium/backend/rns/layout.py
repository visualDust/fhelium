"""Placement-independent RNS chain and hybrid-decomposition metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fhelium.backend.rns.chain import RnsChain
from fhelium.backend.rns.decomposition import (
    HybridRnsDecomposition,
)


@dataclass(frozen=True)
class RnsDigitSpec:
    """One active hybrid-RNS digit in ``[Q | P]`` row order.

    ``digit_index`` is the digit's local position at this depth.
    ``key_digit_index`` is its stable depth-zero key-storage position; dropped
    Q rows may make these values differ. ``prime_ids`` identify parameter
    primes, while ``component_row_ids`` index the compact depth-specific Q
    component tensor.
    """

    depth: int
    digit_index: int
    key_digit_index: int
    prime_ids: tuple[int, ...]
    component_row_ids: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"RnsDigitSpec(depth={self.depth}, "
            f"digit_index={self.digit_index}, "
            f"key_digit_index={self.key_digit_index}, "
            f"prime_ids={self.prime_ids}, "
            f"component_row_ids={self.component_row_ids})"
        )


class RnsLayout:
    r"""Mathematical Q/P layout independent of execution placement.

    This object intentionally contains no device assignment or communication
    policy.  An SPMD workload may partition returned prime ids, but
    doing so does not change the local CKKS value or native-kernel ABI.

    For any tensor ``[..., limb, coefficient_or_ntt_index]``, limb ``j`` maps
    exactly to ``prime_ids(...)[j]``. ``include_p`` chooses between internal
    $Q_d$ and $Q_dP$ row sets; semantic values carry ``modulus_basis``
    separately and must not expose this implementation flag.
    """

    def __init__(
        self,
        chain: RnsChain,
        hybrid_decomposition: HybridRnsDecomposition,
    ) -> None:
        if hybrid_decomposition.chain is not chain:
            raise ValueError("hybrid decomposition belongs to another chain")
        self.chain = chain
        self.hybrid_decomposition = hybrid_decomposition
        self.basis_count = chain.basis_count
        self._digit_specs_by_depth = tuple(
            self._build_digit_specs(depth)
            for depth in range(self.basis_count)
        )
        # Cache the row projections used by each key-switch sub-operation.
        # RnsDigitSpec remains the per-depth source for these derived views.
        self._digit_rows_by_depth = tuple(
            tuple(spec.prime_ids for spec in specs)
            for specs in self._digit_specs_by_depth
        )
        self._qp_digit_rows_by_depth = tuple(
            rows + (self.chain.p_prime_ids,)
            for rows in self._digit_rows_by_depth
        )
        self._component_digit_rows_by_depth = tuple(
            tuple(spec.component_row_ids for spec in specs)
            for specs in self._digit_specs_by_depth
        )
        self._depth_by_row_count = {
            include_p: {
                self.row_count(depth, include_p=include_p): depth
                for depth in range(self.chain.basis_count)
            }
            for include_p in (False, True)
        }

    def _build_digit_specs(self, depth: int) -> tuple[RnsDigitSpec, ...]:
        start_row = self.chain.basis_row_starts[depth]
        return tuple(
            RnsDigitSpec(
                depth=depth,
                digit_index=digit_index,
                key_digit_index=digit.key_digit_index,
                prime_ids=digit.prime_ids,
                component_row_ids=tuple(
                    prime_id - start_row for prime_id in digit.prime_ids
                ),
            )
            for digit_index, digit in enumerate(
                self.hybrid_decomposition.digits_at_depth(depth)
            )
        )

    @property
    def key_digit_count(self) -> int:
        return self.hybrid_decomposition.digit_count

    def _check_depth(self, depth: int) -> None:
        self.chain.check_depth(depth)

    def prime_ids(
        self, depth: int, *, include_p: bool = False
    ) -> tuple[int, ...]:
        """Return modulus ids in tensor limb order."""

        return self.chain.prime_ids(depth, include_p=include_p)

    def row_count(self, depth: int, *, include_p: bool = False) -> int:
        return len(self.prime_ids(depth, include_p=include_p))

    def depth_for_active_row_count(
        self,
        row_count: int,
        *,
        include_p: bool = False,
    ) -> int:
        """Return the chain position represented by one complete active basis.

        Configured Q depth groups have distinct cumulative row starts, so a
        complete Q or QP tensor's row count identifies its chain position. A
        partial group, digit, or arbitrary prime interval cannot use this
        lookup.
        """

        if type(row_count) is not int or row_count <= 0:
            raise ValueError("active RNS row count must be a positive integer")
        depth = self._depth_by_row_count[include_p].get(row_count)
        if depth is None:
            raise ValueError(
                "active RNS row count does not identify a configured basis: "
                f"rows={row_count}, include_p={include_p}"
            )
        return depth

    def start_row(self, depth: int) -> int:
        self._check_depth(depth)
        return self.chain.basis_row_starts[depth]

    def parameter_rows(self, depth: int, *, include_p: bool = False) -> slice:
        return self.chain.parameter_rows(depth, include_p=include_p)

    def select_values(
        self,
        depth: int,
        values: Sequence[int],
        *,
        include_p: bool = False,
    ) -> tuple[int, ...]:
        """Select global-prime-indexed values in the active basis order."""

        return tuple(
            values[prime_id]
            for prime_id in self.prime_ids(depth, include_p=include_p)
        )

    def digit_rows(
        self, depth: int, *, include_p: bool = False
    ) -> tuple[tuple[int, ...], ...]:
        """Parameter rows for active hybrid-RNS digits."""

        self._check_depth(depth)
        return (
            self._qp_digit_rows_by_depth[depth]
            if include_p
            else self._digit_rows_by_depth[depth]
        )

    def component_digit_rows(self, depth: int) -> tuple[tuple[int, ...], ...]:
        """Return local limb indices for each active Q decomposition digit."""

        self._check_depth(depth)
        return self._component_digit_rows_by_depth[depth]

    def digit_spec(self, depth: int, digit_index: int) -> RnsDigitSpec:
        specs = self.digit_specs(depth)
        if not 0 <= digit_index < len(specs):
            raise ValueError(
                f"digit_index must be in [0, {len(specs)}) for "
                f"depth={depth}; got {digit_index}"
            )
        return specs[digit_index]

    def digit_specs(self, depth: int) -> tuple[RnsDigitSpec, ...]:
        self._check_depth(depth)
        return self._digit_specs_by_depth[depth]

    def __str__(self) -> str:
        return (
            f"RnsLayout(Q={self.chain.num_q_primes}, "
            f"P={self.chain.num_p_primes}, "
            f"max_depth={self.basis_count - 1}, "
            f"key_digits={self.key_digit_count}, "
            f"depth0_Q_rows={self.row_count(0)}, "
            f"depth0_QP_rows={self.row_count(0, include_p=True)})"
        )

    __repr__ = __str__
