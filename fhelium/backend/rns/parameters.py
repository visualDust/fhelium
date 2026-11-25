"""Device-resident parameter tables for RNS operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.backend.rns.montgomery import MontgomeryParameters
from fhelium.backend.rns.layout import RnsLayout
from fhelium.native.wrapper import rns_ops


@dataclass(frozen=True)
class RnsRowParameters:
    r"""Device tables for one ordered ``prime_ids`` sequence.

    Every tensor is integral on one execution device. One-dimensional tables
    have shape ``[limb]`` in ``prime_ids`` order. ``montgomery_r2[j]`` is
    $R^2\bmod q_i$, ``scaled_montgomery_r2[j]`` is
    $\Delta_0R^2\bmod q_i$, and ``twice_modulus[j]`` is $2q_i$ for
    ``i=prime_ids[j]``. The four reduction vectors are respectively the low
    and high split words of $q_i$ and $-q_i^{-1}\bmod R$.

    Mixed-radix tables exist only for multi-row source digits. Normalizers have
    shape ``[digit - 1]``; propagation coefficients have shape
    ``[digit - 1, digit]``; basis-extension coefficients have shape
    ``[digit - 1, destination_limb]`` in depth-zero QP destination
    order. Their entries include the Montgomery factors required by their
    native consumers.

    ``parameter_row_start`` identifies the first row in the context's
    depth-zero QP order. ``native_parameters`` is the cached
    ``[parameter, limb]`` table consumed by native RNS kernels; it is a
    zero-copy view when ``prime_ids`` is one interval and a gathered copy
    otherwise.
    """

    prime_ids: tuple[int, ...]
    parameter_row_start: int
    native_parameters: torch.Tensor
    modulus_tensor: torch.Tensor
    montgomery_reduction_parameters: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
    montgomery_r2: torch.Tensor
    scaled_montgomery_r2: torch.Tensor
    twice_modulus: torch.Tensor
    moduli: tuple[int, ...]
    mixed_radix_normalizers: torch.Tensor | None = None
    basis_extension_coefficients: torch.Tensor | None = None
    mixed_radix_propagation_coefficients: torch.Tensor | None = None



class RnsParameterStore:
    r"""Build context-owned parameter views for RNS basis extension.

    Each store belongs to one device-local context with ``[Q | P]`` prime
    order. The configured depth bases select active Q intervals; a QP basis
    appends the fixed P rows. Interval selections are views of the source
    table storage; other selections gather rows in the requested order.
    """

    def __init__(
        self,
        *,
        rns_layout: RnsLayout,
        montgomery_parameters: MontgomeryParameters,
        device: torch.device,
        torch_dtype: torch.dtype,
        montgomery_reduction_parameter_tables: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        native_parameter_tensor: torch.Tensor,
        modulus_tensor: torch.Tensor,
        montgomery_r2: torch.Tensor,
        scaled_montgomery_r2: torch.Tensor,
        twice_modulus: torch.Tensor,
        moduli: list[int],
    ) -> None:
        self.rns_layout = rns_layout
        self.montgomery_parameters = montgomery_parameters
        self.device = device
        self.torch_dtype = torch_dtype
        self.basis_count = rns_layout.chain.basis_count
        self.qp_row_stop = rns_layout.chain.total_modulus_count
        self.q_row_stop = rns_layout.chain.num_q_primes
        self.montgomery_reduction_parameter_tables = (
            montgomery_reduction_parameter_tables
        )
        self.native_parameter_tensor = native_parameter_tensor
        self.modulus_tensor = modulus_tensor
        self.montgomery_r2 = montgomery_r2
        self.scaled_montgomery_r2 = scaled_montgomery_r2
        self.twice_modulus = twice_modulus
        self.moduli = tuple(moduli)

        self._row_parameters = {
            key: self._build_row_parameters(key)
            for key in self._required_row_keys()
        }
        self._attach_basis_extension_coefficients()

    def row_parameters(self, key) -> RnsRowParameters:
        """Prepare parameter and mixed-radix tables for ordered prime ids."""
        key = tuple(key)
        if (
            not key
            or len(set(key)) != len(key)
            or any(not 0 <= i < self.qp_row_stop for i in key)
        ):
            raise ValueError(
                "RNS row parameters require distinct context prime ids"
            )
        if key not in self._row_parameters:
            self._row_parameters[key] = self._build_row_parameters(key)
        result = self._row_parameters[key]
        self._attach_digit_coefficients(result)
        return result

    def basis_parameters(
        self, depth: int, *, include_p: bool = False
    ) -> RnsRowParameters:
        return self.row_parameters(
            self._active_basis_key(depth, include_p=include_p)
        )

    def twice_modulus_for_basis(
        self, depth: int, *, include_p: bool = False
    ) -> torch.Tensor:
        return self.basis_parameters(depth, include_p=include_p).twice_modulus

    def moduli_for_basis(
        self, depth: int, *, include_p: bool = False
    ) -> list[int]:
        return list(self.basis_parameters(depth, include_p=include_p).moduli)

    def _build_row_parameters(
        self, prime_ids: tuple[int, ...]
    ) -> RnsRowParameters:
        modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi = (
            self.montgomery_reduction_parameter_tables
        )
        start = prime_ids[0]
        if prime_ids == tuple(range(start, start + len(prime_ids))):
            rows: slice | torch.Tensor = slice(start, start + len(prime_ids))
        else:
            rows = torch.tensor(
                prime_ids, dtype=torch.long, device=self.native_parameter_tensor.device
            )
        return RnsRowParameters(
            prime_ids=prime_ids,
            parameter_row_start=start,
            modulus_tensor=self.modulus_tensor[rows],
            native_parameters=self.native_parameter_tensor[:, rows],
            montgomery_reduction_parameters=(
                modulus_lo[rows],
                modulus_hi[rows],
                neg_inv_modulus_lo[rows],
                neg_inv_modulus_hi[rows],
            ),
            montgomery_r2=self.montgomery_r2[rows],
            scaled_montgomery_r2=self.scaled_montgomery_r2[rows],
            twice_modulus=self.twice_modulus[rows],
            moduli=tuple(self.moduli[i] for i in prime_ids),
        )

    def _active_basis_key(
        self, depth: int, *, include_p: bool
    ) -> tuple[int, ...]:
        return self.rns_layout.prime_ids(depth, include_p=include_p)

    def _required_row_keys(self) -> list[tuple[int, ...]]:
        full_rows = len(self.rns_layout.prime_ids(0, include_p=True))
        keys: list[tuple[int, ...]] = [(row_id,) for row_id in range(full_rows)]
        keys.extend(
            self._active_basis_key(depth, include_p=include_p)
            for depth in range(self.basis_count)
            for include_p in (False, True)
        )
        keys.extend(
            digit_spec.prime_ids
            for depth in range(self.rns_layout.basis_count)
            for digit_spec in self.rns_layout.digit_specs(depth)
        )
        keys.append(self.rns_layout.chain.p_prime_ids)
        return list(dict.fromkeys(key for key in keys if key))

    def _attach_basis_extension_coefficients(self) -> None:
        for depth in range(self.rns_layout.basis_count):
            for spec in self.rns_layout.digit_specs(depth):
                self.row_parameters(spec.prime_ids)

    def _attach_digit_coefficients(
        self, row_parameters: RnsRowParameters
    ) -> None:
        if (
            len(row_parameters.prime_ids) == 1
            or row_parameters.mixed_radix_normalizers is not None
        ):
            return
        source_moduli = [
            self.montgomery_parameters.moduli[index]
            for index in row_parameters.prime_ids
        ]
        destination_moduli = [
            self.montgomery_parameters.moduli[index]
            for index in self.rns_layout.prime_ids(0, include_p=True)
        ]
        normalizers, propagation, extension = rns_ops.mixed_radix_tables(
            source_moduli, destination_moduli, self.torch_dtype
        )
        object.__setattr__(
            row_parameters,
            "mixed_radix_normalizers",
            normalizers.to(self.device),
        )
        object.__setattr__(
            row_parameters,
            "mixed_radix_propagation_coefficients",
            propagation.to(self.device),
        )
        object.__setattr__(
            row_parameters,
            "basis_extension_coefficients",
            extension.to(self.device),
        )
