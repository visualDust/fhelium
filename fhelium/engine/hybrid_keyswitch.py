r"""CKKS key-switching helpers.

The central operation in this file is applying a key-switch key to one
ciphertext component.  In standard CKKS/RNS terminology the hot path is:

1. Split the source component into hybrid-RNS decomposition digits.
2. Convert each composite digit's residues to mixed-radix components.
3. Apply ModUp/basis extension to each digit on the active $Q_\ell\cup P$ basis.
4. NTT the extended digit, multiply it by the matching key digit, and
   accumulate the two switched output components in the $Q_\ell\cup P$ basis.
5. INTT the accumulators and ModDown/divide-by-$P$ back to the active $Q_\ell$
   basis.
6. Add the first correction component to the original ciphertext component.

Hybrid decomposition digits and mixed-radix components are named separately
so the code maps directly to the standard CKKS key-switch flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fhelium.config import CkksConfig
from fhelium.core import KeySwitchKey
from fhelium.engine.direct_keyswitch_consumer import (
    DirectKeySwitchDigitConsumer,
)
from fhelium.engine.galois import rotation_galois_element
from fhelium.native.wrapper import ckks_ops, rns_ops

if TYPE_CHECKING:
    from fhelium.engine.rns.layout import RnsDigitSpec
    from fhelium.engine.rns.runtime import RnsRuntime


@dataclass(frozen=True)
class PreparedRotationKeySwitch:
    """Reusable NTT-domain QP digits for rotations of one component.

    Decomposition, ModUp, and forward NTT are independent of the rotation
    step and are therefore materialized once.  Coefficient-domain and
    mixed-radix intermediates are deliberately not retained: consumers need
    only the final NTT digits, while tests can construct debug references
    explicitly without expanding the production artifact representation.

    ``ntt_digits_qp`` is an integral tensor on the engine device with shape
    ``[digit, *batch, limb, ntt_index]``. Limb order is the exact active
    ``rns_layout.prime_ids(level, include_p=True)`` order; every residue is
    Montgomery/lazy. ``digit`` is local active order and callers resolve each
    stable ``key_digit_index`` separately before indexing key storage.
    """

    level: int
    batch_shape: tuple[int, ...]
    ntt_digits_qp: torch.Tensor


class HybridKeySwitcher:
    """Execute CKKS hybrid-RNS direct and prepared key switching."""

    def __init__(
        self,
        *,
        config: CkksConfig,
        rns_runtime: RnsRuntime,
        moddown_p_drop_inverses_montgomery_by_level: list[torch.Tensor],
        direct_digit_consumer: DirectKeySwitchDigitConsumer,
        galois_generator: int = 3,
    ) -> None:
        self.config = config
        self.rns_runtime = rns_runtime
        self.rns_layout = rns_runtime.rns_layout
        self.moddown_p_drop_inverses_montgomery_by_level = (
            moddown_p_drop_inverses_montgomery_by_level
        )
        self.direct_digit_consumer = direct_digit_consumer
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

    def __str__(self) -> str:
        return (
            "HybridKeySwitcher("
            f"backend={self.rns_runtime.ntt_backend_name!r}, "
            f"device={self.rns_runtime.device}, "
            f"mixed_radix_native_arg_cache="
            f"{len(self._mixed_radix_native_arg_cache)})"
        )

    __repr__ = __str__

    def _decompose_digit_mixed_radix(
        self,
        component: torch.Tensor,
        digit_spec: RnsDigitSpec,
    ) -> torch.Tensor:
        r"""Convert one hybrid digit's Q residues to mixed-radix components.

        ``component`` is integral ``[*batch, active_q_limb, coefficient]`` on the
        runtime device in coefficient/standard lazy form. ``digit_spec`` keeps
        local active order separate from stable key storage order. If its exact
        source prime ids are $b_0,\ldots,b_{D-1}$, the non-aliasing output has shape
        ``[*batch, D, coefficient]`` and standard digits satisfying
        $x=\sum_r d_r\prod_{t<r}b_t$ modulo the source product. Prime-row
        identity, dtype, device, coefficient domain, and lazy range are kept.
        """

        source_prime_ids = digit_spec.prime_ids
        source_rows = digit_spec.component_row_ids
        component_count = len(source_rows)
        source_residues = component[
            ..., source_rows[0] : source_rows[-1] + 1, :
        ].clone()

        if component_count == 1:
            return source_residues

        if component_count <= 8:
            (
                mixed_radix_normalizers,
                mixed_radix_propagation_coefficients,
                modulus_lo,
                modulus_hi,
                neg_inv_modulus_lo,
                neg_inv_modulus_hi,
            ) = self._mixed_radix_native_args(digit_spec)
            return rns_ops.mixed_radix_decompose(
                source_residues,
                mixed_radix_normalizers,
                mixed_radix_propagation_coefficients,
                modulus_lo,
                modulus_hi,
                neg_inv_modulus_lo,
                neg_inv_modulus_hi,
            )

        # General reference path for digit widths outside the native bound.
        mixed_radix_components = (
            source_residues[..., 0, :]
            .unsqueeze(-2)
            .repeat(*([1] * (source_residues.ndim - 2)), component_count, 1)
        )
        source_digit_parameters = self.rns_runtime.row_parameters(
            tuple(source_prime_ids)
        )
        mixed_radix_normalizers = (
            source_digit_parameters.mixed_radix_normalizers
        )
        mixed_radix_propagation_coefficients = (
            source_digit_parameters.mixed_radix_propagation_coefficients
        )
        if (
            mixed_radix_normalizers is None
            or mixed_radix_propagation_coefficients is None
        ):
            raise RuntimeError("missing mixed-radix tables for multi-row digit")
        for component_index in range(component_count - 1):
            current_row = component_index + 1
            digit_normalizer = mixed_radix_normalizers[component_index][None]
            new_component = (
                source_residues[..., current_row, :]
                - mixed_radix_components[..., current_row, :]
            ).unsqueeze(-2)
            rns_ops.montgomery_mul_row_scalars_(
                new_component,
                digit_normalizer,
                self.rns_runtime.rns_parameters_for_prime_ids(
                    (source_prime_ids[current_row],)
                ),
            )
            mixed_radix_components[..., current_row, :] = new_component.squeeze(
                -2
            )

            first_later_row = current_row + 1
            if first_later_row < component_count:
                propagation = mixed_radix_propagation_coefficients[
                    component_index, first_later_row:
                ]
                propagated = new_component.repeat(
                    *([1] * (new_component.ndim - 2)),
                    component_count - first_later_row,
                    1,
                )
                rns_ops.montgomery_mul_row_scalars_(
                    propagated,
                    propagation,
                    self.rns_runtime.rns_parameters_for_prime_ids(
                        tuple(source_prime_ids[first_later_row:])
                    ),
                )
                mixed_radix_components[..., first_later_row:, :] += propagated
        return mixed_radix_components

    def _mixed_radix_native_args(
        self, digit_spec: RnsDigitSpec
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return cached dense scalar tables for native ModUp construction."""

        cache_key = (digit_spec.level, digit_spec.digit_index)
        cached = self._mixed_radix_native_arg_cache.get(cache_key)
        if cached is not None:
            return cached

        source_prime_ids = digit_spec.prime_ids
        source_digit_parameters = self.rns_runtime.row_parameters(
            tuple(source_prime_ids)
        )
        mixed_radix_normalizers = (
            source_digit_parameters.mixed_radix_normalizers
        )
        propagation = (
            source_digit_parameters.mixed_radix_propagation_coefficients
        )
        if mixed_radix_normalizers is None or propagation is None:
            raise RuntimeError("missing dense mixed-radix native tables")
        modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi = (
            source_digit_parameters.montgomery_reduction_parameters
        )
        cached = (
            mixed_radix_normalizers.contiguous(),
            propagation.contiguous(),
            modulus_lo.contiguous(),
            modulus_hi.contiguous(),
            neg_inv_modulus_lo.contiguous(),
            neg_inv_modulus_hi.contiguous(),
        )
        self._mixed_radix_native_arg_cache[cache_key] = cached
        return cached

    def _extend_digit_to_qp(
        self,
        mixed_radix_components: torch.Tensor,
        digit_spec: RnsDigitSpec,
    ) -> torch.Tensor:
        r"""Basis-extend one mixed-radix digit into active $Q_\ell P$.

        Input is integral ``[*batch, digit, coefficient]`` in standard
        mixed-radix form. Output is non-aliasing
        ``[*batch, destination_limb, coefficient]`` in coefficient-domain
        Montgomery lazy form, with rows exactly
        ``rns_layout.prime_ids(level, include_p=True)``. It evaluates
        $\sum_r d_r\prod_{t<r}b_t$ modulo every destination prime; no scale,
        polynomial, or public batch semantics change.
        """

        active_basis = self.rns_runtime.basis_parameters(
            digit_spec.level, include_p=True
        )
        destination_row_count = len(active_basis.prime_ids)
        source_prime_ids = digit_spec.prime_ids
        basis_extension_coefficients = self.rns_runtime.row_parameters(
            tuple(source_prime_ids)
        ).basis_extension_coefficients
        if basis_extension_coefficients is None:
            basis_extension_coefficients = torch.empty(
                0,
                0,
                dtype=mixed_radix_components.dtype,
                device=mixed_radix_components.device,
            )

        active_row_start = active_basis.parameter_row_start
        active_row_stop = active_row_start + destination_row_count
        active_basis_extension_coefficients = basis_extension_coefficients[
            :, active_row_start:active_row_stop
        ]
        active_params = active_basis.native_parameters

        return rns_ops.mixed_radix_basis_extend_to_montgomery(
            mixed_radix_components,
            active_basis_extension_coefficients,
            active_params,
            destination_row_count,
        )

    def apply_key_switch(
        self,
        component: torch.Tensor,
        key_switch_key: KeySwitchKey,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A direct switch has exactly one key consumer, so materializing every
        # extended digit provides no reuse.  More importantly, transforming
        # the digit axis as one batch makes the QP NTT working set exceed L2
        # at large parameter sets (for example logN=16 at level 0), regressing even
        # a genuinely unbatched ciphertext.  Stream one disposable digit for
        # every backend; only the final NTT/consumer implementation differs.
        # ``prepare_rotation_digits`` remains the explicit materialization
        # point for rotate-many hoisting, where multiple keys reuse the
        # same digits.
        return self._apply_key_switch_streaming(
            component,
            key_switch_key,
            level,
        )

    def _apply_key_switch_streaming(
        self,
        component: torch.Tensor,
        key_switch_key: KeySwitchKey,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one disposable QP digit at a time for a direct switch.

        Direct relinearization and single-key rotation need no reusable
        hoisted digits. They therefore retain one QP digit scratch and two QP
        accumulators. The engine-bound digit consumer owns whether the
        representation transition and multiply-accumulate use ordinary or
        fused kernels; this executor remains independent of the NTT backend.
        """

        accumulator0_qp: torch.Tensor | None = None
        accumulator1_qp: torch.Tensor | None = None
        digit_specs = self.rns_layout.digit_specs(level)
        active_row_start = self.rns_runtime.basis_parameters(
            level, include_p=True
        ).parameter_row_start
        for digit_spec in digit_specs:
            mixed_radix_components = self._decompose_digit_mixed_radix(
                component,
                digit_spec,
            )
            disposable_digit_qp = self._extend_digit_to_qp(
                mixed_radix_components,
                digit_spec,
            )
            if accumulator0_qp is None:
                accumulator0_qp = torch.zeros_like(disposable_digit_qp)
                accumulator1_qp = torch.zeros_like(disposable_digit_qp)
            assert accumulator1_qp is not None

            self.direct_digit_consumer.consume_(
                disposable_digit_qp,
                key_switch_key.digit(digit_spec.key_digit_index),
                accumulator0_qp,
                accumulator1_qp,
                parameter_row_start=active_row_start,
                key_row_start=active_row_start,
            )

        if accumulator0_qp is None or accumulator1_qp is None:
            raise RuntimeError("key switching requires at least one RNS digit")
        return self._moddown_switch_accumulators(
            accumulator0_qp,
            accumulator1_qp,
            level,
        )

    def prepare_rotation_digits(
        self, component: torch.Tensor, level: int
    ) -> PreparedRotationKeySwitch:
        """Materialize only reusable NTT-domain QP digits for rotate-many."""

        digit_specs = self.rns_layout.digit_specs(level)
        digit_count = len(digit_specs)
        active_row_start = self.rns_runtime.basis_parameters(
            level, include_p=True
        ).parameter_row_start
        ntt_digits_qp: torch.Tensor | None = None
        for digit_spec in digit_specs:
            mixed_radix_components = self._decompose_digit_mixed_radix(
                component, digit_spec
            )
            digit_qp = self._extend_digit_to_qp(
                mixed_radix_components, digit_spec
            )

            # A hybrid digit is an internal decomposition stage, not another
            # public message batch member. Transforming one digit at a time
            # preserves cross-stage cache locality; only the final NTT values
            # are copied into the explicit rotation-hoisting artifact.
            self.rns_runtime.forward_montgomery_(
                digit_qp,
                include_p=True,
                parameter_row_start=active_row_start,
            )
            if ntt_digits_qp is None:
                ntt_digits_qp = torch.empty(
                    (digit_count, *digit_qp.shape),
                    dtype=digit_qp.dtype,
                    device=digit_qp.device,
                )
            ntt_digits_qp[digit_spec.digit_index].copy_(digit_qp)

        if ntt_digits_qp is None:
            raise RuntimeError(
                "rotation hoisting requires at least one RNS digit"
            )
        return PreparedRotationKeySwitch(
            level=level,
            batch_shape=tuple(component.shape[:-2]),
            ntt_digits_qp=ntt_digits_qp,
        )

    def _apply_prepared_rotation(
        self,
        prepared: PreparedRotationKeySwitch,
        key_switch_key: KeySwitchKey,
        *,
        rotation_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototype = prepared.ntt_digits_qp[0]
        accumulator0_qp = torch.zeros_like(prototype)
        accumulator1_qp = torch.zeros_like(prototype)
        digit_specs = self.rns_layout.digit_specs(prepared.level)
        for digit_spec, digit_qp_ntt in zip(
            digit_specs, prepared.ntt_digits_qp, strict=True
        ):
            self._accumulate_prepared_rotation_digit_(
                digit_qp_ntt,
                key_switch_key,
                digit_spec,
                accumulator0_qp,
                accumulator1_qp,
                rotation_step=rotation_step,
            )
        return self._moddown_switch_accumulators(
            accumulator0_qp, accumulator1_qp, prepared.level
        )

    def _accumulate_prepared_rotation_digit_(
        self,
        extended_digit_qp_ntt: torch.Tensor,
        key_switch_key: KeySwitchKey,
        digit_spec: RnsDigitSpec,
        accumulator0_qp: torch.Tensor,
        accumulator1_qp: torch.Tensor,
        *,
        rotation_step: int,
    ) -> None:
        extended_digit_qp_ntt = self._apply_ntt_galois_automorphism(
            extended_digit_qp_ntt, rotation_step
        )

        active_basis = self.rns_runtime.basis_parameters(
            digit_spec.level, include_p=True
        )
        ckks_ops.keyswitch_accumulate_digit_products_(
            accumulator0_qp,
            accumulator1_qp,
            extended_digit_qp_ntt,
            key_switch_key.digit(digit_spec.key_digit_index),
            active_basis.native_parameters,
            active_basis.parameter_row_start,
        )

    def _apply_ntt_galois_automorphism(
        self,
        extended_digit_qp_ntt: torch.Tensor,
        rotation_step: int,
    ) -> torch.Tensor:
        """Apply a Galois automorphism as a rank-local NTT-domain gather."""

        N = self.config.N
        rotation_step %= N
        if rotation_step == 0:
            return extended_digit_qp_ntt
        return ckks_ops.apply_ntt_galois_automorphism(
            extended_digit_qp_ntt,
            self._ntt_galois_source_indices(
                rotation_step, extended_digit_qp_ntt.device
            ),
        )

    def _bit_reverse_indices(
        self, N: int, device: torch.device
    ) -> torch.Tensor:
        cache_key = (N, str(device))
        cached = self._bit_reverse_index_cache.get(cache_key)
        if cached is not None:
            return cached

        source = torch.arange(N, dtype=torch.int64, device=device)
        remaining = source.clone()
        bit_reversed = torch.zeros_like(source)
        for _ in range(N.bit_length() - 1):
            bit_reversed = (bit_reversed << 1) | (remaining & 1)
            remaining >>= 1
        self._bit_reverse_index_cache[cache_key] = bit_reversed
        return bit_reversed

    def _ntt_galois_source_indices(
        self, rotation_step: int, device: torch.device
    ) -> torch.Tensor:
        N = self.config.N
        rotation_step %= N
        cache_key = (N, rotation_step, str(device))
        cached = self._ntt_galois_source_index_cache.get(cache_key)
        if cached is not None:
            return cached

        bit_reversed = self._bit_reverse_indices(N, device)
        dest_exp = 2 * bit_reversed + 1
        galois_element = rotation_galois_element(
            N,
            rotation_step,
            self.galois_generator,
        )
        src_bit_reversed = ((dest_exp * galois_element) % (2 * N) - 1) // 2
        source_indices = bit_reversed.index_select(
            0, src_bit_reversed.to(torch.long)
        ).to(torch.int32)
        self._ntt_galois_source_index_cache[cache_key] = source_indices
        return source_indices

    def _moddown_switch_accumulators(
        self,
        accumulator0_qp: torch.Tensor,
        accumulator1_qp: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Apply sequential P ModDown to two QP key-switch accumulators.

        Inputs are integral NTT/Montgomery
        ``[*batch, active_qp_limb, ntt_index]`` tensors. They are inverse-transformed
        in place, then each P prime is rounded away using
        ``moddown_p_drop_inverses_montgomery_by_level[level]``. Returned
        tensors are newly allocated coefficient/standard Q-only residues with
        exact $Q_\ell$ rows and the same public batch axes.
        """

        self.rns_runtime.inverse_to_standard_(accumulator0_qp, include_p=True)
        self.rns_runtime.inverse_to_standard_(accumulator1_qp, include_p=True)

        p_count = self.config.num_p_primes
        moddown_p_drop_inverses_montgomery = (
            self.moddown_p_drop_inverses_montgomery_by_level[level]
        )
        active_params = self.rns_runtime.basis_parameters(
            level, include_p=True
        ).native_parameters
        correction0 = ckks_ops.keyswitch_moddown_qp_to_q(
            accumulator0_qp[..., :-p_count, :],
            accumulator0_qp[..., -p_count:, :],
            moddown_p_drop_inverses_montgomery,
            active_params,
        )
        correction1 = ckks_ops.keyswitch_moddown_qp_to_q(
            accumulator1_qp[..., :-p_count, :],
            accumulator1_qp[..., -p_count:, :],
            moddown_p_drop_inverses_montgomery,
            active_params,
        )
        return correction0, correction1

    def _switch_prepared_rotation(
        self,
        rotated_c0: torch.Tensor,
        prepared: PreparedRotationKeySwitch,
        key_switch_key: KeySwitchKey,
        *,
        rotation_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_q_rows = len(self.rns_layout.prime_ids(prepared.level))
        expected_c0_shape = (
            *prepared.batch_shape,
            expected_q_rows,
            self.config.N,
        )
        if tuple(rotated_c0.shape) != expected_c0_shape:
            raise ValueError(
                "Rotated c0 shape does not match prepared rotation digits: "
                f"{tuple(rotated_c0.shape)} != {expected_c0_shape}"
            )
        correction0, correction1 = self._apply_prepared_rotation(
            prepared,
            key_switch_key,
            rotation_step=rotation_step,
        )
        switched_c0 = self.rns_runtime.add_canonical(rotated_c0, correction0)
        return switched_c0, correction1
