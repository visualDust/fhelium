"""Device-resident parameter tables for RNS operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.engine.rns.montgomery import MontgomeryParameters
from fhelium.engine.rns.layout import RnsLayout


@dataclass(frozen=True)
class RnsRowParameters:
    r"""Device tables for one contiguous ``prime_ids`` interval.

    Every tensor is integral on one execution device. One-dimensional tables
    have shape ``[limb]`` in ``prime_ids`` order. ``montgomery_r2[j]`` is
    $R^2\bmod q_i$, ``scaled_montgomery_r2[j]`` is
    $\Delta_0R^2\bmod q_i$, and ``twice_modulus[j]`` is $2q_i$ for
    ``i=prime_ids[j]``. The four reduction vectors are respectively the low
    and high split words of $q_i$ and $-q_i^{-1}\bmod R$.

    Mixed-radix tables exist only for multi-row source digits. Normalizers have
    shape ``[digit - 1]``; propagation coefficients have shape
    ``[digit - 1, digit]``; basis-extension coefficients have shape
    ``[digit - 1, destination_limb]`` in canonical level-zero QP destination
    order. Their entries include the Montgomery factors required by their
    native consumers.

    ``parameter_row_start`` identifies the first row in the engine's canonical
    level-zero QP order. ``native_parameters`` is the cached zero-copy
    ``[parameter, limb]`` view consumed by native RNS kernels.
    """

    prime_ids: tuple[int, ...]
    parameter_row_start: int
    native_parameters: torch.Tensor
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
    r"""Build engine-owned parameter views for RNS basis extension.

    A process owns one device and one dense canonical ``[Q | P]`` prime order.
    Level $\ell$ selects the contiguous interval beginning at Q prime id
    ``level``; a Q basis ends before P and a QP basis includes the fixed P
    suffix. Views preserve this order and do not allocate or mutate the
    source tables. The store contains no device fanout or communication policy.
    """

    def __init__(
        self,
        *,
        rns_layout: RnsLayout,
        montgomery_parameters: MontgomeryParameters,
        device: torch.device,
        torch_dtype: torch.dtype,
        rns_basis_level_count: int,
        level_row_starts: list[int],
        basis_row_stops: tuple[int, int],
        montgomery_reduction_parameter_tables: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        native_parameter_tensor: torch.Tensor,
        montgomery_r2: torch.Tensor,
        scaled_montgomery_r2: torch.Tensor,
        twice_modulus: torch.Tensor,
        moduli: list[int],
    ) -> None:
        self.rns_layout = rns_layout
        self.montgomery_parameters = montgomery_parameters
        self.device = device
        self.torch_dtype = torch_dtype
        self.rns_basis_level_count = rns_basis_level_count
        self.level_row_starts = level_row_starts
        self.qp_row_stop, self.q_row_stop = basis_row_stops
        self.montgomery_reduction_parameter_tables = (
            montgomery_reduction_parameter_tables
        )
        self.native_parameter_tensor = native_parameter_tensor
        self.montgomery_r2 = montgomery_r2
        self.scaled_montgomery_r2 = scaled_montgomery_r2
        self.twice_modulus = twice_modulus
        self.moduli = tuple(moduli)

        self._row_parameters = {
            key: self._build_row_parameters(key[0], key[-1] + 1)
            for key in self._required_row_keys()
        }
        self._attach_basis_extension_coefficients()

    def _active_range(self, level: int, include_p: bool) -> tuple[int, int]:
        stop = self.qp_row_stop if include_p else self.q_row_stop
        return self.level_row_starts[level], stop

    def row_parameters(self, key) -> RnsRowParameters:
        return self._row_parameters[tuple(key)]

    def basis_parameters(
        self, level: int, *, include_p: bool = False
    ) -> RnsRowParameters:
        return self.row_parameters(
            self._active_basis_key(level, include_p=include_p)
        )

    def twice_modulus_for_basis(
        self, level: int, *, include_p: bool = False
    ) -> torch.Tensor:
        return self.basis_parameters(level, include_p=include_p).twice_modulus

    def moduli_for_basis(
        self, level: int, *, include_p: bool = False
    ) -> list[int]:
        return list(self.basis_parameters(level, include_p=include_p).moduli)

    def _build_row_parameters(
        self, row_start: int, row_stop: int
    ) -> RnsRowParameters:
        modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi = (
            self.montgomery_reduction_parameter_tables
        )
        return RnsRowParameters(
            prime_ids=tuple(range(row_start, row_stop)),
            parameter_row_start=row_start,
            native_parameters=self.native_parameter_tensor[
                :, row_start:row_stop
            ],
            montgomery_reduction_parameters=(
                modulus_lo[row_start:row_stop],
                modulus_hi[row_start:row_stop],
                neg_inv_modulus_lo[row_start:row_stop],
                neg_inv_modulus_hi[row_start:row_stop],
            ),
            montgomery_r2=self.montgomery_r2[row_start:row_stop],
            scaled_montgomery_r2=self.scaled_montgomery_r2[row_start:row_stop],
            twice_modulus=self.twice_modulus[row_start:row_stop],
            moduli=self.moduli[row_start:row_stop],
        )

    def _active_basis_key(
        self, level: int, *, include_p: bool
    ) -> tuple[int, ...]:
        start, stop = self._active_range(level, include_p)
        return tuple(range(start, stop))

    def _required_row_keys(self) -> list[tuple[int, ...]]:
        full_rows = len(self.rns_layout.prime_ids(0, include_p=True))
        keys: list[tuple[int, ...]] = [(row_id,) for row_id in range(full_rows)]
        keys.extend(
            self._active_basis_key(level, include_p=include_p)
            for level in range(self.rns_basis_level_count)
            for include_p in (False, True)
        )
        keys.extend(
            digit_spec.prime_ids
            for level in range(self.rns_basis_level_count)
            for digit_spec in self.rns_layout.digit_specs(level)
        )
        keys.append(self.rns_layout.chain.p_prime_ids)
        return list(dict.fromkeys(key for key in keys if key))

    def _attach_basis_extension_coefficients(self) -> None:
        destination_prime_ids = self.rns_layout.prime_ids(0, include_p=True)
        destination_moduli = [
            self.montgomery_parameters.moduli[index]
            for index in destination_prime_ids
        ]
        destination_r2 = [
            self.montgomery_parameters.montgomery_r2[index]
            for index in destination_prime_ids
        ]

        for level in range(self.rns_basis_level_count):
            for digit_spec in self.rns_layout.digit_specs(level):
                source_prime_ids = digit_spec.prime_ids
                row_parameters = self.row_parameters(source_prime_ids)
                if row_parameters.mixed_radix_normalizers is not None:
                    continue

                digit_width = len(source_prime_ids)
                source_moduli = [
                    self.montgomery_parameters.moduli[index]
                    for index in source_prime_ids
                ]
                basis_products = [source_moduli[0]]
                for index in range(1, digit_width - 1):
                    basis_products.append(
                        basis_products[-1] * source_moduli[index]
                    )

                normalizers = []
                propagation = torch.zeros(
                    (max(digit_width - 1, 0), digit_width),
                    dtype=self.torch_dtype,
                    device=self.device,
                )
                for component_index in range(digit_width - 1):
                    inverse = pow(
                        basis_products[component_index],
                        -1,
                        source_moduli[component_index + 1],
                    )
                    normalizers.append(
                        (inverse * self.montgomery_parameters.R)
                        % source_moduli[component_index + 1]
                    )
                    for target_index in range(component_index + 2, digit_width):
                        propagation[component_index, target_index] = (
                            basis_products[component_index]
                            * self.montgomery_parameters.R
                        ) % source_moduli[target_index]

                if not normalizers:
                    continue
                object.__setattr__(
                    row_parameters,
                    "mixed_radix_normalizers",
                    torch.tensor(
                        normalizers,
                        dtype=self.torch_dtype,
                        device=self.device,
                    ),
                )
                object.__setattr__(
                    row_parameters,
                    "mixed_radix_propagation_coefficients",
                    propagation,
                )
                object.__setattr__(
                    row_parameters,
                    "basis_extension_coefficients",
                    torch.tensor(
                        [
                            [
                                (basis_product * r2) % modulus
                                for r2, modulus in zip(
                                    destination_r2,
                                    destination_moduli,
                                    strict=True,
                                )
                            ]
                            for basis_product in basis_products[
                                : digit_width - 1
                            ]
                        ],
                        dtype=self.torch_dtype,
                        device=self.device,
                    ),
                )
