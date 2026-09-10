r"""Concrete rotate-many key-switch execution with shared digit preparation.

`_RotationHoistExecutor` prepares the hybrid-RNS digits of one coefficient-domain
ciphertext component once, then consumes those NTT/Montgomery QP digits for
multiple direct rotation keys. Coefficient outputs use a transformed $c_0$
for each key. NTT outputs share the original $c_0$ evaluations and retain Q
evaluations during ModDown. The product accumulator gathers each digit's NTT
indices while reading it instead of materializing a permuted digit tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fhelium.config import CkksConfig
from fhelium.values import RotationKey
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ckks._moddown import moddown_ntt_qp_to_q
from fhelium.backend.ckks.resources import KeySwitchExecutionResource
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.layout import RnsDigitSpec
from fhelium.native.wrapper import ckks_ops, rns_ops


@dataclass(frozen=True)
class _PreparedRotationDigits:
    r"""Reusable NTT/Montgomery QP digits for one ciphertext component.

    `ntt_digits_qp` has shape
    ``[digit, *batch, active_qp_limb, ntt_index]``.  `digit` uses active local
    order at `depth`; each consumer separately resolves the corresponding
    stable key-storage digit through `RnsDigitSpec.key_digit_index`. The
    executor creates and consumes this private value within one rotate-many
    call.
    """

    depth: int
    ntt_digits_qp: torch.Tensor


class _RotationHoistExecutor:
    """Prepare and consume shared hybrid-RNS digits for direct rotations."""

    def __init__(
        self,
        *,
        config: CkksConfig,
        rns_context: RnsContext,
        ntt_context: NttContext,
        moddown_p_drop_inverses_montgomery_by_depth: tuple[torch.Tensor, ...],
        galois_generator: int = 3,
    ) -> None:
        self.config = config
        self.rns_context = rns_context
        self.ntt_context = ntt_context
        self.moddown_p_drop_inverses_montgomery_by_depth = (
            moddown_p_drop_inverses_montgomery_by_depth
        )
        self.galois_generator = galois_generator
        self._mixed_radix_native_arg_cache: dict[
            tuple[int, int],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._bit_reverse_index_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._ntt_galois_source_index_cache: dict[
            tuple[int, int, str], torch.Tensor
        ] = {}
        self._coefficient_galois_cache: dict[
            tuple[int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def rotate_component(
        self,
        component: torch.Tensor,
        *,
        depth: int,
        key: RotationKey,
    ) -> torch.Tensor:
        """Apply the coefficient-domain automorphism for one slot rotation."""

        normalized = key.rotation_step % self.config.N
        if normalized == 0:
            return component.clone()
        cache_key = (normalized, str(component.device))
        tables = self._coefficient_galois_cache.get(cache_key)
        if tables is None:
            modulus = 2 * self.config.N
            exponent = -normalized if self.galois_generator == 5 else normalized
            galois_element = pow(
                self.galois_generator,
                exponent % self.config.N,
                modulus,
            )
            source = torch.arange(
                self.config.N,
                dtype=torch.int64,
                device=component.device,
            )
            mapped = (galois_element * source) % modulus
            destination = mapped % self.config.N
            sign = torch.where(
                ((mapped // self.config.N) & 1) != 0,
                torch.tensor(-1, dtype=torch.int8, device=component.device),
                torch.tensor(1, dtype=torch.int8, device=component.device),
            )
            source_indices = torch.empty(
                self.config.N,
                dtype=torch.int32,
                device=component.device,
            )
            source_indices[destination] = source.to(torch.int32)
            source_sign = torch.empty(
                self.config.N,
                dtype=torch.int8,
                device=component.device,
            )
            source_sign[destination] = sign
            tables = (source_indices, source_sign)
            self._coefficient_galois_cache[cache_key] = tables
        return ckks_ops.apply_coefficient_galois_automorphism(
            component,
            tables[0],
            tables[1],
            self.rns_context.twice_modulus_for_basis(depth, include_p=False),
        )

    def _mixed_radix_native_args(
        self,
        digit_spec: RnsDigitSpec,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cache_key = (digit_spec.depth, digit_spec.digit_index)
        cached = self._mixed_radix_native_arg_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = self.rns_context.row_parameters(digit_spec.prime_ids)
        normalizers = rows.mixed_radix_normalizers
        propagation = rows.mixed_radix_propagation_coefficients
        if normalizers is None or propagation is None:
            raise RuntimeError("Multi-row digit lacks mixed-radix tables")
        modulus_lo, modulus_hi, neg_inv_lo, neg_inv_hi = (
            rows.montgomery_reduction_parameters
        )
        result = (
            normalizers.contiguous(),
            propagation.contiguous(),
            modulus_lo.contiguous(),
            modulus_hi.contiguous(),
            neg_inv_lo.contiguous(),
            neg_inv_hi.contiguous(),
        )
        self._mixed_radix_native_arg_cache[cache_key] = result
        return result

    def _decompose_digit_mixed_radix(
        self,
        component: torch.Tensor,
        digit_spec: RnsDigitSpec,
    ) -> torch.Tensor:
        source_rows = digit_spec.component_row_ids
        component_count = len(source_rows)
        source = component[
            ...,
            source_rows[0] : source_rows[-1] + 1,
            :,
        ].clone()
        if component_count == 1:
            return source
        if component_count <= 16:
            return rns_ops.mixed_radix_decompose(
                source,
                *self._mixed_radix_native_args(digit_spec),
            )

        values = (
            source[..., 0, :]
            .unsqueeze(-2)
            .repeat(*([1] * (source.ndim - 2)), component_count, 1)
        )
        rows = self.rns_context.row_parameters(digit_spec.prime_ids)
        normalizers = rows.mixed_radix_normalizers
        propagation = rows.mixed_radix_propagation_coefficients
        if normalizers is None or propagation is None:
            raise RuntimeError("Multi-row digit lacks mixed-radix tables")
        for component_index in range(component_count - 1):
            current_row = component_index + 1
            update = (
                source[..., current_row, :] - values[..., current_row, :]
            ).unsqueeze(-2)
            rns_ops.montgomery_mul_row_scalars_(
                update,
                normalizers[component_index][None],
                self.rns_context.rns_parameters_for_prime_ids(
                    (digit_spec.prime_ids[current_row],)
                ),
            )
            values[..., current_row, :] = update.squeeze(-2)
            first_later = current_row + 1
            if first_later < component_count:
                propagated = update.repeat(
                    *([1] * (update.ndim - 2)),
                    component_count - first_later,
                    1,
                )
                rns_ops.montgomery_mul_row_scalars_(
                    propagated,
                    propagation[component_index, first_later:],
                    self.rns_context.rns_parameters_for_prime_ids(
                        digit_spec.prime_ids[first_later:]
                    ),
                )
                values[..., first_later:, :] += propagated
        return values

    def _extend_digit_to_qp(
        self,
        mixed_radix_components: torch.Tensor,
        digit_spec: RnsDigitSpec,
    ) -> torch.Tensor:
        active = self.rns_context.basis_parameters(
            digit_spec.depth,
            include_p=True,
        )
        source = self.rns_context.row_parameters(digit_spec.prime_ids)
        coefficients = source.basis_extension_coefficients
        if coefficients is None:
            coefficients = torch.empty(
                0,
                0,
                dtype=mixed_radix_components.dtype,
                device=mixed_radix_components.device,
            )
        start = active.parameter_row_start
        stop = start + len(active.prime_ids)
        return rns_ops.mixed_radix_basis_extend_to_montgomery(
            mixed_radix_components,
            coefficients[:, start:stop],
            active.native_parameters,
            len(active.prime_ids),
        )

    def prepare(
        self,
        component: torch.Tensor,
        depth: int,
    ) -> _PreparedRotationDigits:
        """Materialize reusable NTT/Montgomery QP digits for `component`."""

        digit_specs = self.rns_context.rns_layout.digit_specs(depth)
        active_start = self.rns_context.basis_parameters(
            depth,
            include_p=True,
        ).parameter_row_start
        digits: torch.Tensor | None = None
        for digit_spec in digit_specs:
            mixed = self._decompose_digit_mixed_radix(component, digit_spec)
            digit_qp = self._extend_digit_to_qp(mixed, digit_spec)
            self.ntt_context.forward_montgomery_(
                digit_qp,
                include_p=True,
                parameter_row_start=active_start,
            )
            if digits is None:
                digits = torch.empty(
                    (len(digit_specs), *digit_qp.shape),
                    dtype=digit_qp.dtype,
                    device=digit_qp.device,
                )
            digits[digit_spec.digit_index].copy_(digit_qp)
        if digits is None:
            raise RuntimeError("Rotation hoisting requires an active RNS digit")
        return _PreparedRotationDigits(
            depth=depth,
            ntt_digits_qp=digits,
        )

    def _bit_reverse_indices(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (self.config.N, str(device))
        cached = self._bit_reverse_index_cache.get(cache_key)
        if cached is not None:
            return cached
        source = torch.arange(self.config.N, dtype=torch.int64, device=device)
        remaining = source.clone()
        reversed_indices = torch.zeros_like(source)
        for _ in range(self.config.N.bit_length() - 1):
            reversed_indices = (reversed_indices << 1) | (remaining & 1)
            remaining >>= 1
        self._bit_reverse_index_cache[cache_key] = reversed_indices
        return reversed_indices

    def _ntt_galois_source_indices(
        self,
        rotation_step: int,
        device: torch.device,
    ) -> torch.Tensor:
        normalized = rotation_step % self.config.N
        cache_key = (self.config.N, normalized, str(device))
        cached = self._ntt_galois_source_index_cache.get(cache_key)
        if cached is not None:
            return cached
        bit_reversed = self._bit_reverse_indices(device)
        destination_exponents = 2 * bit_reversed + 1
        exponent = -normalized if self.galois_generator == 5 else normalized
        galois_element = pow(
            self.galois_generator,
            exponent % self.config.N,
            2 * self.config.N,
        )
        source_bit_reversed = (
            (destination_exponents * galois_element) % (2 * self.config.N) - 1
        ) // 2
        indices = bit_reversed.index_select(
            0,
            source_bit_reversed.to(torch.long),
        ).to(torch.int32)
        self._ntt_galois_source_index_cache[cache_key] = indices
        return indices

    def _moddown(
        self,
        accumulator0_qp: torch.Tensor,
        accumulator1_qp: torch.Tensor,
        depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.ntt_context.inverse_to_standard_(accumulator0_qp, include_p=True)
        self.ntt_context.inverse_to_standard_(accumulator1_qp, include_p=True)
        p_count = self.config.num_p_primes
        inverses = self.moddown_p_drop_inverses_montgomery_by_depth[depth]
        parameters = self.rns_context.basis_parameters(
            depth,
            include_p=True,
        ).native_parameters
        return (
            ckks_ops.keyswitch_moddown_qp_to_q(
                accumulator0_qp[..., :-p_count, :],
                accumulator0_qp[..., -p_count:, :],
                inverses,
                parameters,
            ),
            ckks_ops.keyswitch_moddown_qp_to_q(
                accumulator1_qp[..., :-p_count, :],
                accumulator1_qp[..., -p_count:, :],
                inverses,
                parameters,
            ),
        )

    def apply(
        self,
        rotated_c0: torch.Tensor,
        prepared: _PreparedRotationDigits,
        key: RotationKey,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume prepared digits for one direct rotation key."""

        accumulator = self._accumulate(prepared, key)
        correction0, correction1 = self._moddown(
            accumulator[0], accumulator[1], prepared.depth
        )
        return (
            self.rns_context.add_standard(rotated_c0, correction0),
            correction1,
        )

    def apply_ntt(
        self,
        c0_ntt: torch.Tensor,
        prepared: _PreparedRotationDigits,
        key: RotationKey,
        plan: KeySwitchExecutionResource,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate shared c0 evaluations and return Q NTT/Montgomery components."""

        accumulator = self._accumulate(prepared, key)
        corrections = moddown_ntt_qp_to_q(accumulator, plan, prepared.depth)
        indices = self._ntt_galois_source_indices(
            key.rotation_step, c0_ntt.device
        )
        rotated_c0 = ckks_ops.apply_ntt_galois_automorphism(c0_ntt, indices)
        return (
            self.rns_context.add_lazy(rotated_c0, corrections[0]),
            corrections[1],
        )

    def _accumulate(
        self,
        prepared: _PreparedRotationDigits,
        key: RotationKey,
    ) -> torch.Tensor:
        """Accumulate both QP key products with the NTT permutation in the load."""

        prototype = prepared.ntt_digits_qp[0]
        accumulator = torch.zeros(
            (2, *prototype.shape),
            dtype=prototype.dtype,
            device=prototype.device,
        )
        active = self.rns_context.basis_parameters(
            prepared.depth,
            include_p=True,
        )
        source_indices = self._ntt_galois_source_indices(
            key.rotation_step,
            prototype.device,
        )
        digit_specs = self.rns_context.rns_layout.digit_specs(prepared.depth)
        if (
            prototype.is_cuda
            and len(prepared.ntt_digits_qp) <= 5
            and tuple(spec.key_digit_index for spec in digit_specs)
            == tuple(range(len(digit_specs)))
        ):
            ckks_ops.keyswitch_accumulate_products_(
                accumulator[0],
                accumulator[1],
                list(prepared.ntt_digits_qp),
                key.data,
                active.native_parameters,
                active.parameter_row_start,
                source_indices,
            )
            return accumulator
        for digit_spec, digit_qp in zip(
            digit_specs,
            prepared.ntt_digits_qp,
            strict=True,
        ):
            ckks_ops.keyswitch_accumulate_digit_products_(
                accumulator[0],
                accumulator[1],
                digit_qp,
                key.digit(digit_spec.key_digit_index),
                active.native_parameters,
                active.parameter_row_start,
                source_indices,
            )
        return accumulator


__all__: list[str] = []
