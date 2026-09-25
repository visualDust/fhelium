"""RNS quotient tables and depth-indexed rescaling data."""

from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.backend.ntt.context import NttContext
from typing import Mapping
import torch
from .context import RnsContext


def rescale_inverse_matrix(rns_context: RnsContext) -> torch.Tensor:
    montgomery = rns_context.montgomery_parameters
    rows = []
    for dropped_id, dropped_prime in enumerate(montgomery.moduli):
        rows.append(
            [
                0
                if remaining_id == dropped_id
                else (pow(dropped_prime, -1, remaining_prime) * montgomery.R)
                % remaining_prime
                for remaining_id, remaining_prime in enumerate(
                    montgomery.moduli
                )
            ]
        )
    return torch.tensor(
        rows,
        dtype=rns_context.dtype,
        device=rns_context.device,
    )


def moddown_inverse_tables(
    rns_context: RnsContext,
) -> tuple[torch.Tensor, ...]:
    config = rns_context.config
    montgomery = rns_context.montgomery_parameters
    p_moduli = montgomery.moduli[-config.num_p_primes :][::-1]
    inverse_rows = [
        [
            (pow(dropped, -1, modulus) * montgomery.R) % modulus
            for modulus in montgomery.moduli[: -drop_step - 1]
        ]
        for drop_step, dropped in enumerate(p_moduli)
    ]
    depth0_prime_ids = rns_context.rns_layout.prime_ids(0, include_p=True)
    base_rows = [
        torch.tensor(
            [
                inverse_rows[drop_step][prime_id]
                for prime_id in depth0_prime_ids[: -drop_step - 1]
            ],
            dtype=rns_context.dtype,
            device=rns_context.device,
        )
        for drop_step in range(config.num_p_primes)
    ]
    tables = []
    for depth in range(config.max_depth + 1):
        start = rns_context.basis_parameters(depth).parameter_row_start
        rows = [row[start:] for row in base_rows]
        packed = torch.empty(
            (len(rows), max(row.numel() for row in rows)),
            dtype=rns_context.dtype,
            device=rns_context.device,
        )
        for row_index, row in enumerate(rows):
            packed[row_index, : row.numel()] = row
        tables.append(packed)
    return tuple(tables)


@dataclass(frozen=True)
class RescaleTables:
    """Own pairwise and prefix-product inverses used by RNS rescaling."""

    rns_context: RnsContext
    dropped_prime_inverses_montgomery: torch.Tensor
    _active_views: Mapping[
        tuple[int, bool], tuple[torch.Tensor, torch.Tensor, int]
    ] = field(init=False, repr=False, compare=False)
    _prefix_inverses: dict[tuple[int, int, bool], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        table = self.dropped_prime_inverses_montgomery
        prime_count = len(self.rns_context.montgomery_parameters.moduli)
        if tuple(table.shape) != (prime_count, prime_count):
            raise ValueError(
                "Rescale inverse matrix shape differs from the modulus set"
            )
        views: dict[
            tuple[int, bool], tuple[torch.Tensor, torch.Tensor, int]
        ] = {}
        for depth in range(self.rns_context.q_row_stop - 1):
            for include_p in (False, True):
                stop = (
                    self.rns_context.qp_row_stop
                    if include_p
                    else self.rns_context.q_row_stop
                )
                row_count = stop - depth
                remaining_ids = tuple(range(depth + 1, stop))
                views[(row_count, include_p)] = (
                    self.rns_context.rns_parameters_for_prime_ids(
                        remaining_ids
                    ),
                    table[depth, depth + 1 : stop],
                    int(self.rns_context.montgomery_parameters.moduli[depth]),
                )
        object.__setattr__(self, "_active_views", MappingProxyType(views))

    @property
    def device(self) -> torch.device:
        return self.rns_context.device

    def active_views(
        self,
        row_count: int,
        *,
        include_p: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return preselected parameters, inverses, and dropped modulus."""

        try:
            return self._active_views[(row_count, include_p)]
        except KeyError:
            raise ValueError(
                "Rescale requires a complete Q or QP basis with another "
                "active Q row"
            ) from None

    def prefix_inverse(
        self, row_count: int, drop_count: int, *, include_p: bool
    ) -> torch.Tensor:
        r"""Return $M^{-1}R\bmod q_i$ for each surviving prime.

        $M$ is the product of the leading ``drop_count`` active primes. The
        resource retains the vector for reuse with the same prime interval.
        """

        if drop_count == 1:
            return self.active_views(row_count, include_p=include_p)[1]
        key = (row_count, drop_count, include_p)
        if key not in self._prefix_inverses:
            context = self.rns_context
            stop = context.qp_row_stop if include_p else context.q_row_stop
            start = stop - row_count
            moduli = context.montgomery_parameters.moduli
            divisor = prod(moduli[start : start + drop_count])
            radix = context.montgomery_parameters.R
            self._prefix_inverses[key] = torch.tensor(
                [
                    pow(divisor, -1, q) * radix % q
                    for q in moduli[start + drop_count : stop]
                ],
                dtype=context.dtype,
                device=context.device,
            )
        return self._prefix_inverses[key]

    def tensor_operands(
        self,
        row_count: int,
        drop_count: int,
        *,
        include_p: bool,
        ntt_context: NttContext | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """Expose quotient tables and scalar rounding offsets for one group."""
        tensors: dict[str, torch.Tensor] = {}
        half_primes = []
        for step in range(drop_count):
            parameters, inverse, prime = self.active_views(
                row_count - step, include_p=include_p
            )
            tensors[f"step{step}_parameters"] = parameters
            tensors[f"step{step}_inverse"] = inverse
            half_primes.append(prime // 2)
        facts: dict[str, object] = {
            "drop_count": drop_count,
            "half_primes": tuple(half_primes),
        }
        if ntt_context is not None:
            stop = (
                self.rns_context.qp_row_stop
                if include_p
                else self.rns_context.q_row_stop
            )
            start = stop - row_count
            for basis, ids, inverse in (
                ("dropped", tuple(range(start, start + drop_count)), True),
                ("remaining", tuple(range(start + drop_count, stop)), False),
            ):
                direction = "inverse" if inverse else "forward"
                for index, tensor in enumerate(
                    ntt_context.tensor_operands(ids, inverse=inverse)
                ):
                    tensors[f"{basis}_{direction}_{index}"] = tensor
            tensors["prefix_inverse"] = self.prefix_inverse(
                row_count, drop_count, include_p=include_p
            )
            facts["ntt_backend"] = ntt_context.ntt_backend_name
        return tensors, facts
