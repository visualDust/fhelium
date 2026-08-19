"""CKKS secret, public, rotation, conjugation, and key-switch key generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fhelium.config import CkksConfig
from fhelium.core import (
    CkksContextSpec,
    ConjugationKey,
    KeySwitchKey,
    ModulusBasis,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.engine.galois import (
    apply_coefficient_galois_automorphism,
    rotation_galois_element,
)
from fhelium.errors import (
    PolynomialDomainError,
    ResidueRepresentationError,
    SecretKeyModulusBasisError,
)
from fhelium.native.wrapper import rns_ops

if TYPE_CHECKING:
    from fhelium.engine.rns.runtime import RnsRuntime
    from fhelium.rng import Csprng


class CkksKeyGenerator:
    r"""Construct dense CKKS keys on one process-local execution device.

    Key payloads use integral dtype and exact canonical prime rows. Secret-key
    data is ``[limb, ntt_index]``; public keys are
    ``[key_component, limb, ntt_index]``; key-switch keys are
    ``[key_digit, key_component, limb, ntt_index]``. Returned keys are always
    NTT/Montgomery/lazy at level zero in the stated Q or QP basis and own their
    payload storage. Local ``digit_index`` is resolved to stable
    ``key_digit_index`` before key tensor indexing.
    """

    def __init__(
        self,
        *,
        config: CkksConfig,
        context: CkksContextSpec,
        device: torch.device,
        rng: Csprng,
        rns_runtime: RnsRuntime,
        p_product_montgomery_q: torch.Tensor,
        galois_generator: int = 3,
    ) -> None:
        self.config = config
        self.context = context
        self.device = device
        self._rng = rng
        self.rns_runtime = rns_runtime
        self.rns_layout = rns_runtime.rns_layout
        self.p_product_montgomery_q = p_product_montgomery_q
        self.galois_generator = galois_generator

    def __str__(self) -> str:
        return (
            "CkksKeyGenerator("
            f"context={self.context.context_id!r}, "
            f"backend={self.rns_runtime.ntt_backend_name!r}, "
            f"device={self.device})"
        )

    __repr__ = __str__

    @staticmethod
    def _prime_ids(tensor: torch.Tensor, *, limb_axis: int) -> tuple[int, ...]:
        return tuple(range(tensor.size(limb_axis)))

    def _require_ntt_montgomery_secret_key(
        self,
        secret_key: SecretKey,
        *,
        operation: str,
    ) -> None:
        if secret_key.context_id != self.context.context_id:
            raise ValueError(
                "SecretKey belongs to another CKKS context: "
                f"{secret_key.context_id} != {self.context.context_id}"
            )
        if secret_key.data.size(-1) != self.config.N:
            raise ValueError(
                "SecretKey polynomial degree does not match key generator: "
                f"{secret_key.data.size(-1)} != {self.config.N}"
            )
        if secret_key.data.device != self.device:
            raise ValueError(
                "SecretKey device does not match key generator: "
                f"{secret_key.data.device} != {self.device}"
            )
        expected_prime_ids = self.rns_layout.prime_ids(
            0, include_p=secret_key.modulus_basis == "QP"
        )
        if secret_key.prime_ids != expected_prime_ids:
            raise ValueError(
                "SecretKey local RNS structure differs from the key generator: "
                f"{secret_key.prime_ids} != {expected_prime_ids}"
            )
        if secret_key.polynomial_domain != "ntt":
            raise PolynomialDomainError(
                value_name=f"SecretKey for {operation}",
                expected="ntt",
                actual=secret_key.polynomial_domain,
            )
        if secret_key.residue_representation != "montgomery":
            raise ResidueRepresentationError(
                value_name=f"SecretKey for {operation}",
                expected="montgomery",
                actual=secret_key.residue_representation,
            )

    def _require_qp_secret_key(
        self,
        secret_key: SecretKey,
        *,
        operation: str,
    ) -> None:
        self._require_ntt_montgomery_secret_key(
            secret_key,
            operation=operation,
        )
        if secret_key.modulus_basis != "QP":
            raise SecretKeyModulusBasisError(
                operation=operation,
                expected="QP",
                actual=secret_key.modulus_basis,
            )

    def create_secret_key(
        self, *, modulus_basis: ModulusBasis = "QP"
    ) -> SecretKey:
        r"""Sample ternary $s(X)$ and return its level-zero NTT/Montgomery RNS.

        Output shape is ``[limb, ntt_index]`` with exact Q or QP ``prime_ids``
        selected by ``modulus_basis``. Sampling and all temporary transitions
        are functional from the caller's perspective.
        """

        if modulus_basis not in ("Q", "QP"):
            raise ValueError(f"Unsupported modulus_basis: {modulus_basis!r}")
        include_p = modulus_basis == "QP"
        uniform_ternary = self._rng.randint(amax=3, shift=-1, repeats=1)[0][0]
        unsigned_ternary = self.rns_runtime.lift_centered_coefficients(
            uniform_ternary, level=0, include_p=include_p
        )
        self.rns_runtime.forward_to_montgomery_(
            unsigned_ternary, include_p=include_p
        )
        data = unsigned_ternary
        return SecretKey(
            data=data,
            context_id=self.context.context_id,
            prime_ids=self._prime_ids(data, limb_axis=0),
            polynomial_domain="ntt",
            modulus_basis=modulus_basis,
            residue_representation="montgomery",
        )

    def create_public_key(
        self,
        secret_key: SecretKey,
        *,
        modulus_basis: ModulusBasis = "Q",
        uniform_component: torch.Tensor | None = None,
    ) -> PublicKey:
        r"""Generate ``(k_0,k_1)`` satisfying $k_0+k_1s=e$ modulo the basis.

        Output is integral ``[key_component=2, limb, ntt_index]`` in
        level-zero NTT/Montgomery form with exact Q or QP rows. ``secret_key``
        and optional ``uniform_component`` are read-only and never alias the
        returned stacked tensor.
        """

        if modulus_basis not in ("Q", "QP"):
            raise ValueError(f"Unsupported modulus_basis: {modulus_basis!r}")
        include_p = modulus_basis == "QP"
        if include_p:
            self._require_qp_secret_key(
                secret_key,
                operation="Public-key generation on the QP basis",
            )
        else:
            self._require_ntt_montgomery_secret_key(
                secret_key,
                operation="Public-key generation",
            )

        level = 0
        error = self._rng.discrete_gaussian(repeats=1)[0][0]
        error = self.rns_runtime.lift_centered_coefficients(
            error, level, include_p=include_p
        )
        self.rns_runtime.forward_to_montgomery_(error, include_p=include_p)

        if uniform_component is None:
            repeats = (
                self.config.num_p_primes
                if secret_key.modulus_basis == "QP"
                else 0
            )
            uniform_component_data = self._rng.randint(
                [self.rns_runtime.moduli_for_basis(level, include_p=include_p)],
                repeats=repeats,
            )[0]
        else:
            uniform_component_data = uniform_component

        secret_key_rows = secret_key.data
        if not include_p and secret_key.modulus_basis == "QP":
            secret_key_rows = secret_key_rows[: -self.config.num_p_primes]

        secret_times_uniform = self.rns_runtime.montgomery_mul(
            uniform_component_data, secret_key_rows, include_p=include_p
        )
        public_component0 = self.rns_runtime.sub_lazy(
            error, secret_times_uniform, include_p=include_p
        )

        public_key_data = torch.stack(
            (public_component0, uniform_component_data), dim=0
        )
        return PublicKey(
            data=public_key_data,
            context_id=self.context.context_id,
            prime_ids=self._prime_ids(public_key_data, limb_axis=1),
            polynomial_domain="ntt",
            modulus_basis=modulus_basis,
            residue_representation="montgomery",
        )

    def create_key_switch_key(
        self,
        source_secret_key: SecretKey,
        destination_secret_key: SecretKey,
        *,
        uniform_component_by_key_digit: torch.Tensor | None = None,
    ) -> KeySwitchKey:
        r"""Create a hybrid-RNS key from source to destination secret relation.

        Key construction belongs here rather than in the key-switch executor:
        it repeatedly creates public-key encryptions and does not participate
        in either the direct-streaming or prepared execution plans.

        Stable key digit $d$ satisfies, on that digit's embedded source rows,
        $k_{d,0}+k_{d,1}s_{\mathrm{dst}}=P s_{\mathrm{src}}+e_d$.
        Output is integral
        ``[key_digit, key_component=2, QP_limb, ntt_index]`` in
        NTT/Montgomery lazy form and exact level-zero QP order. Input keys and
        optional uniform components are not mutated or aliased.
        """

        self._require_qp_secret_key(
            source_secret_key,
            operation="Hybrid key-switch key generation (source key)",
        )
        self._require_qp_secret_key(
            destination_secret_key,
            operation="Hybrid key-switch key generation (destination key)",
        )
        level = 0

        source_secret_q = source_secret_key.data[
            : self.rns_runtime.q_row_stop
        ].clone()
        self.rns_runtime.montgomery_mul_row_scalars_(
            source_secret_q, self.p_product_montgomery_q
        )

        digit_specs = self.rns_layout.digit_specs(level)
        key_digits: list[torch.Tensor | None] = [None] * len(digit_specs)
        for digit_spec in digit_specs:
            source_prime_ids = digit_spec.prime_ids
            key_digit_index = digit_spec.key_digit_index
            uniform_component = (
                uniform_component_by_key_digit[key_digit_index]
                if uniform_component_by_key_digit is not None
                else None
            )
            public_key = self.create_public_key(
                destination_secret_key,
                modulus_basis="QP",
                uniform_component=uniform_component,
            )

            source_digit_key = tuple(source_prime_ids)
            row_start = source_prime_ids[0]
            row_stop = source_prime_ids[-1] + 1
            source_secret_digit = source_secret_q[row_start:row_stop]
            public_key_digit_c0 = public_key.k0[row_start:row_stop]
            twice_modulus = self.rns_runtime.row_parameters(
                source_digit_key
            ).twice_modulus
            updated_c0_digit = rns_ops.add_lazy_with_twice_modulus(
                public_key_digit_c0, source_secret_digit, twice_modulus
            )
            public_key_digit_c0.copy_(updated_c0_digit, non_blocking=True)
            key_digits[key_digit_index] = public_key.data

        if any(key_digit is None for key_digit in key_digits):
            raise RuntimeError(
                "Key-switch key generation left an empty hybrid-RNS digit"
            )
        key_data = torch.stack(
            [key_digit for key_digit in key_digits if key_digit is not None],
            dim=0,
        )
        return KeySwitchKey(
            data=key_data,
            context_id=self.context.context_id,
            prime_ids=tuple(range(key_data.size(2))),
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )

    def create_relinearization_key(
        self, secret_key: SecretKey
    ) -> RelinearizationKey:
        r"""Return QP key material that switches the $s^2$ phase term to $s$.

        The returned state and layout equal :meth:`create_key_switch_key`; the
        input secret key remains NTT/Montgomery QP and is not mutated.
        """

        self._require_qp_secret_key(
            secret_key,
            operation="Relinearization-key generation",
        )
        squared_secret_key_data = self.rns_runtime.montgomery_mul(
            secret_key.data, secret_key.data, include_p=True
        )
        squared_secret_key = SecretKey(
            data=squared_secret_key_data,
            context_id=self.context.context_id,
            prime_ids=secret_key.prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )
        key_switch_key = self.create_key_switch_key(
            squared_secret_key, secret_key
        )
        return RelinearizationKey(
            data=key_switch_key.data,
            context_id=key_switch_key.context_id,
            prime_ids=key_switch_key.prime_ids,
            polynomial_domain=key_switch_key.polynomial_domain,
            modulus_basis=key_switch_key.modulus_basis,
            residue_representation=key_switch_key.residue_representation,
        )

    def create_rotation_key(
        self,
        rotation_step: int,
        *,
        uniform_component_by_key_digit: torch.Tensor | None = None,
        secret_key: SecretKey,
    ) -> RotationKey:
        r"""Construct QP key material from $\sigma_g(s)$ back to $s$.

        ``rotation_step`` follows signed slot displacement and is stored as
        metadata. The distinct ``galois_element`` $g$ selects polynomial
        automorphism $\sigma_g$. Output uses key-switch-key axes and
        NTT/Montgomery QP state; input secret/uniform tensors are not mutated.
        """

        self._require_qp_secret_key(
            secret_key,
            operation="Rotation-key generation",
        )
        rotation_step = RotationKey.canonical_step(
            rotation_step,
            ring_dimension=self.config.N,
        )
        rotated = secret_key.data.clone()
        self.rns_runtime.inverse_montgomery_(rotated, include_p=True)
        rotated = apply_coefficient_galois_automorphism(
            rotated,
            rotation_galois_element(
                self.config.N,
                rotation_step,
                self.galois_generator,
            ),
            self.rns_runtime.moduli,
        )
        self.rns_runtime.forward_montgomery_(rotated, include_p=True)
        rotated_secret_key = SecretKey(
            data=rotated,
            context_id=self.context.context_id,
            prime_ids=secret_key.prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )
        key_switch_key = self.create_key_switch_key(
            rotated_secret_key,
            secret_key,
            uniform_component_by_key_digit=uniform_component_by_key_digit,
        )
        return RotationKey(
            data=key_switch_key.data,
            context_id=key_switch_key.context_id,
            prime_ids=key_switch_key.prime_ids,
            polynomial_domain=key_switch_key.polynomial_domain,
            modulus_basis=key_switch_key.modulus_basis,
            residue_representation=key_switch_key.residue_representation,
            rotation_step=rotation_step,
        )

    def create_conjugation_key(self, secret_key: SecretKey) -> ConjugationKey:
        r"""Construct QP key material from $\sigma_{2N-1}(s)$ back to $s$.

        Output uses key-switch-key axes and NTT/Montgomery QP state; the input
        secret key remains unchanged.
        """

        self._require_qp_secret_key(
            secret_key,
            operation="Conjugation-key generation",
        )

        conjugated = secret_key.data.clone()
        self.rns_runtime.inverse_montgomery_(conjugated, include_p=True)
        conjugated = apply_coefficient_galois_automorphism(
            conjugated,
            2 * self.config.N - 1,
            self.rns_runtime.moduli,
        )
        self.rns_runtime.forward_montgomery_(conjugated, include_p=True)
        conjugated_secret_key = SecretKey(
            data=conjugated,
            context_id=self.context.context_id,
            prime_ids=secret_key.prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )
        key_switch_key = self.create_key_switch_key(
            conjugated_secret_key, secret_key
        )
        return ConjugationKey(
            data=key_switch_key.data,
            context_id=key_switch_key.context_id,
            prime_ids=key_switch_key.prime_ids,
            polynomial_domain=key_switch_key.polynomial_domain,
            modulus_basis=key_switch_key.modulus_basis,
            residue_representation=key_switch_key.residue_representation,
        )
