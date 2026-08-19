"""Placement-independent RNS chain and hybrid-decomposition metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fhelium.engine.rns.chain import RnsChain
from fhelium.engine.rns.decomposition import HybridRnsDecomposition


@dataclass(frozen=True)
class RnsDigitSpec:
    """One active hybrid-RNS digit in canonical ``[Q | P]`` row order.

    ``digit_index`` is the digit's local position at this level.
    ``key_digit_index`` is its stable level-zero key-storage position; dropped
    Q rows may make these values differ. ``prime_ids`` identify parameter
    primes, while ``component_row_ids`` index the compact level-specific Q
    component tensor.
    """

    level: int
    digit_index: int
    key_digit_index: int
    prime_ids: tuple[int, ...]
    component_row_ids: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"RnsDigitSpec(level={self.level}, "
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
    $Q_\ell$ and $Q_\ell P$ row sets; semantic values carry ``modulus_basis``
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
        self.rns_basis_level_count = chain.rns_basis_level_count
        self._digit_specs_by_level = tuple(
            self._build_digit_specs(level)
            for level in range(self.rns_basis_level_count)
        )
        # Compatibility projections remain cached rather than rebuilding
        # tuples on every key-switch sub-operation.  RnsDigitSpec is the
        # authoritative per-level metadata and these views are derived from it
        # exactly once.
        self._digit_rows_by_level = tuple(
            tuple(spec.prime_ids for spec in specs)
            for specs in self._digit_specs_by_level
        )
        self._qp_digit_rows_by_level = tuple(
            rows + (self.chain.p_prime_ids,)
            for rows in self._digit_rows_by_level
        )
        self._component_digit_rows_by_level = tuple(
            tuple(spec.component_row_ids for spec in specs)
            for specs in self._digit_specs_by_level
        )

    def _build_digit_specs(self, level: int) -> tuple[RnsDigitSpec, ...]:
        return tuple(
            RnsDigitSpec(
                level=level,
                digit_index=digit_index,
                key_digit_index=digit.key_digit_index,
                prime_ids=digit.prime_ids,
                component_row_ids=tuple(
                    prime_id - level for prime_id in digit.prime_ids
                ),
            )
            for digit_index, digit in enumerate(
                self.hybrid_decomposition.digits_at_level(level)
            )
        )

    @property
    def key_digit_count(self) -> int:
        return self.hybrid_decomposition.digit_count

    def _check_level(self, level: int) -> None:
        self.chain.check_level(level)

    def prime_ids(
        self, level: int, *, include_p: bool = False
    ) -> tuple[int, ...]:
        """Return exact modulus ids in tensor limb order."""

        return self.chain.prime_ids(level, include_p=include_p)

    def row_count(self, level: int, *, include_p: bool = False) -> int:
        return len(self.prime_ids(level, include_p=include_p))

    def start_row(self, level: int) -> int:
        self._check_level(level)
        return level

    def parameter_rows(self, level: int, *, include_p: bool = False) -> slice:
        return self.chain.parameter_rows(level, include_p=include_p)

    def select_values(
        self,
        level: int,
        values: Sequence[int],
        *,
        include_p: bool = False,
    ) -> tuple[int, ...]:
        """Select global-prime-indexed values in the active basis order."""

        return tuple(
            values[prime_id]
            for prime_id in self.prime_ids(level, include_p=include_p)
        )

    def digit_rows(
        self, level: int, *, include_p: bool = False
    ) -> tuple[tuple[int, ...], ...]:
        """Canonical parameter rows for active hybrid-RNS digits."""

        self._check_level(level)
        return (
            self._qp_digit_rows_by_level[level]
            if include_p
            else self._digit_rows_by_level[level]
        )

    def component_digit_rows(self, level: int) -> tuple[tuple[int, ...], ...]:
        """Return local limb indices for each active Q decomposition digit."""

        self._check_level(level)
        return self._component_digit_rows_by_level[level]

    def digit_spec(self, level: int, digit_index: int) -> RnsDigitSpec:
        specs = self.digit_specs(level)
        if not 0 <= digit_index < len(specs):
            raise ValueError(
                f"digit_index must be in [0, {len(specs)}) for "
                f"level={level}; got {digit_index}"
            )
        return specs[digit_index]

    def digit_specs(self, level: int) -> tuple[RnsDigitSpec, ...]:
        self._check_level(level)
        return self._digit_specs_by_level[level]

    def __str__(self) -> str:
        return (
            f"RnsLayout(Q={self.chain.num_q_primes}, "
            f"P={self.chain.num_p_primes}, "
            f"levels={self.rns_basis_level_count}, "
            f"key_digits={self.key_digit_count}, "
            f"level0_Q_rows={self.row_count(0)}, "
            f"level0_QP_rows={self.row_count(0, include_p=True)})"
        )

    __repr__ = __str__
