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


def centered_basis_extension_tables(
    source: RnsContext,
    target: RnsContext,
    source_prime_ids: tuple[int, ...],
    target_prime_ids: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    r"""Prepare Garner and centered-extension operands for two residue bases.

    Source and destination use the same word format and device. For source
    product Q, subtraction of Q mod p recenters the upper CRT half in each
    target row p. All row parameters and half digits are ordinary tensors;
    execution does not retain either preparation context.
    """
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("Basis-extension contexts must share dtype and device")
    source_moduli = tuple(source.config.moduli[i] for i in source_prime_ids)
    target_moduli = tuple(target.config.moduli[i] for i in target_prime_ids)
    width = len(source_moduli)
    if not width or not target_moduli:
        raise ValueError("Basis extension requires nonempty bases")
    dtype, device = source.dtype, source.device
    radix = source.montgomery_parameters.R
    normalizers, propagation, extension = [], [], []
    prefix = 1
    for row, modulus in enumerate(source_moduli[:-1]):
        prefix *= modulus
        normalizers.append(
            pow(prefix, -1, source_moduli[row + 1])
            * radix
            % source_moduli[row + 1]
        )
        propagation.append(
            [
                prefix * radix % q if column >= row + 2 else 0
                for column, q in enumerate(source_moduli)
            ]
        )
        extension.append(
            [
                prefix * target.montgomery_parameters.montgomery_r2[i] % q
                for i, q in zip(target_prime_ids, target_moduli, strict=True)
            ]
        )
    product = prod(source_moduli)
    remainder = product // 2
    half_digits = []
    for modulus in source_moduli:
        remainder, digit = divmod(remainder, modulus)
        half_digits.append(digit)
    source_index = torch.tensor(
        source_prime_ids, dtype=torch.int64, device=device
    )
    target_index = torch.tensor(
        target_prime_ids, dtype=torch.int64, device=device
    )
    source_parameters = source.rns_parameter_tensor.index_select(
        1, source_index
    )
    lo, hi, neg_lo, neg_hi = source_parameters[1:5].unbind(0)
    return {
        "source_parameters": source_parameters,
        "target_parameters": target.rns_parameter_tensor.index_select(
            1, target_index
        ),
        "source_lo": lo,
        "source_hi": hi,
        "source_neg_lo": neg_lo,
        "source_neg_hi": neg_hi,
        "normalizers": torch.tensor(normalizers, dtype=dtype, device=device),
        "propagation": torch.tensor(
            propagation, dtype=dtype, device=device
        ).reshape(width - 1, width),
        "extension": torch.tensor(
            extension, dtype=dtype, device=device
        ).reshape(width - 1, len(target_moduli)),
        "half_digits": torch.tensor(half_digits, dtype=dtype, device=device),
        "source_product": torch.tensor(
            [product % q for q in target_moduli], dtype=dtype, device=device
        ),
        "target_moduli": torch.tensor(
            target_moduli, dtype=dtype, device=device
        ),
    }


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


def trailing_quotient_inverses(
    moduli: tuple[int, ...],
    drop_count: int,
    radix: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    r"""Build $p_j^{-1}R\bmod q_i$ for successive trailing-row removal.

    Row j removes the j-th prime from the end. Only columns before that
    prime participate; unused columns are zero.
    """
    rows = torch.zeros(
        (drop_count, len(moduli) - 1), dtype=dtype, device=device
    )
    for j, prime in enumerate(reversed(moduli[-drop_count:])):
        values = [pow(prime, -1, q) * radix % q for q in moduli[: -j - 1]]
        rows[j, : len(values)] = torch.tensor(
            values, dtype=dtype, device=device
        )
    return rows


def moddown_inverse_tables(
    rns_context: RnsContext,
) -> tuple[torch.Tensor, ...]:
    config = rns_context.config
    montgomery = rns_context.montgomery_parameters
    depth0_prime_ids = rns_context.rns_layout.prime_ids(0, include_p=True)
    base = trailing_quotient_inverses(
        tuple(montgomery.moduli[i] for i in depth0_prime_ids),
        config.num_p_primes,
        montgomery.R,
        dtype=rns_context.dtype,
        device=rns_context.device,
    )
    base_rows = [
        base[j, : len(depth0_prime_ids) - j - 1]
        for j in range(config.num_p_primes)
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


MODDOWN_RESCALE_TABLE_NAMES = (
    "joint_remaining_parameters",
    "joint_dropped_parameters",
    "joint_ordered_parameters",
    "joint_p_times_base",
    "joint_inverses",
    "joint_product_inverse",
    "joint_dropped_offset",
    "joint_remaining_offset",
    "joint_drop_positions",
    "joint_remaining_positions",
)


def moddown_rescale_tables(
    context: RnsContext,
    q_prime_ids: tuple[int, ...],
    p_prime_ids: tuple[int, ...],
    drop_prime_ids: tuple[int, ...],
    *,
    moddown_rounding: str = "nearest",
    rescale_rounding: str = "nearest",
) -> dict[str, torch.Tensor]:
    r"""Prepare the exact quotient of $v+Pa+H_P+PH_D$ by $PD$.

    Q and P identify actual ordered input rows; D identifies the complete group consumed by rescaling. The remaining Q rows retain their input order. All returned values are ordinary numerical Tensors on this RNS context's device.
    """
    if moddown_rounding not in {"nearest", "floor"} or rescale_rounding not in {
        "nearest",
        "floor",
    }:
        raise ValueError("quotient rounding must be nearest or floor")
    q, p, d = tuple(q_prime_ids), tuple(p_prime_ids), tuple(drop_prime_ids)
    if not d or len(set(d)) != len(d) or not set(d) <= set(q):
        raise ValueError(
            "The rescale group must be a nonempty subset of Q rows"
        )
    remaining = tuple(i for i in q if i not in d)
    if not remaining:
        raise ValueError("Rescaling must retain Q rows")
    dropped, ordered = (*d, *p), (*remaining, *d, *p)
    moduli, radix = context.config.moduli, context.montgomery_parameters.R
    P, D = prod(moduli[i] for i in p), prod(moduli[i] for i in d)
    H = ((P - 1) // 2 if moddown_rounding == "nearest" else 0) + P * (
        (D - 1) // 2 if rescale_rounding == "nearest" else 0
    )

    def tensor(values, dtype=None):
        return torch.tensor(
            values,
            dtype=context.dtype if dtype is None else dtype,
            device=context.device,
        )

    def broadcast(values):
        return tensor(values).view(-1, 1).expand(-1, context.config.N)

    return {
        "joint_remaining_parameters": context.rns_parameters_for_prime_ids(
            remaining
        ),
        "joint_dropped_parameters": context.rns_parameters_for_prime_ids(
            dropped
        ),
        "joint_ordered_parameters": context.rns_parameters_for_prime_ids(
            ordered
        ),
        "joint_p_times_base": tensor(tuple(P * radix % moduli[i] for i in q)),
        "joint_inverses": trailing_quotient_inverses(
            tuple(moduli[i] for i in ordered),
            len(dropped),
            radix,
            dtype=context.dtype,
            device=context.device,
        ),
        "joint_product_inverse": tensor(
            tuple(
                pow(P * D, -1, moduli[i]) * radix % moduli[i] for i in remaining
            )
        ),
        "joint_dropped_offset": broadcast(
            tuple(H % moduli[i] for i in dropped)
        ),
        "joint_remaining_offset": broadcast(
            tuple(H % moduli[i] for i in remaining)
        ),
        "joint_drop_positions": tensor(
            tuple(q.index(i) for i in d), torch.long
        ),
        "joint_remaining_positions": tensor(
            tuple(q.index(i) for i in remaining), torch.long
        ),
    }
