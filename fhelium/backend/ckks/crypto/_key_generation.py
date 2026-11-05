"""CKKS secret, public, rotation, conjugation, and key-switch key generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fhelium.config import CkksConfig
from fhelium.values import (
    ConjugationKey,
    KeySwitchKey,
    ModulusBasis,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.backend.rns.automorphism import (
    apply_coefficient_galois_automorphism,
)
from ..rotation._galois import rotation_galois_element
from fhelium.errors import (
    PolynomialDomainError,
    ResidueRepresentationError,
    SecretKeyModulusBasisError,
)
from fhelium.native.wrapper import rns_ops
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.decomposition import _partition_q_digits
from fhelium.values._validation import validate_prime_ids
from fhelium.rng import Csprng


@dataclass(frozen=True)
class KeyGenerationResource:
    """Bind context, arithmetic, and random resources for one key device."""

    config: CkksConfig
    rng: Csprng
    rns_context: RnsContext
    ntt_context: NttContext
    p_product_montgomery_q: torch.Tensor

    def __post_init__(self) -> None:
        if self.rns_context.config != self.config:
            raise ValueError(
                "Key-generation RNS context differs from its CKKS configuration"
            )
        if self.ntt_context.rns_context is not self.rns_context:
            raise ValueError(
                "Key-generation NTT context composes another RNS context"
            )
        if self.rng.devices != [str(self.device)]:
            raise ValueError(
                "Key-generation random stream must own exactly its device"
            )
        table = self.p_product_montgomery_q
        if table.device != self.device:
            raise ValueError(
                "Key-generation P-product table is on another device"
            )
        if table.dtype != self.rns_context.dtype:
            raise TypeError("Key-generation P-product table has another dtype")

    @classmethod
    def create(
        cls,
        *,
        config: CkksConfig,
        rng: Csprng,
        rns_context: RnsContext,
        ntt_context: NttContext,
    ) -> KeyGenerationResource:
        """Build key-generation scalars on one concrete context device."""

        montgomery = rns_context.montgomery_parameters
        product_p = math.prod(montgomery.moduli[-config.num_p_primes :])
        pr = product_p * montgomery.R
        return cls(
            config,
            rng,
            rns_context,
            ntt_context,
            torch.tensor(
                [
                    pr % montgomery.moduli[prime_id]
                    for prime_id in rns_context.rns_layout.prime_ids(0)
                ],
                dtype=rns_context.dtype,
                device=rns_context.device,
            ),
        )

    def for_config(
        self, config: CkksConfig, *, dtype: torch.dtype | None = None
    ) -> KeyGenerationResource:
        """Prepare another modulus basis on this device and random stream."""
        if config.N != self.config.N:
            raise ValueError(
                "Key basis conversion requires the same ring degree"
            )
        rns = RnsContext(config, device=self.device, dtype=dtype)
        ntt = NttContext(rns, ntt_backend=self.ntt_context.ntt_backend_name)
        return KeyGenerationResource.create(
            config=config, rng=self.rng, rns_context=rns, ntt_context=ntt
        )

    @property
    def device(self) -> torch.device:
        """Return the concrete device shared by all key-generation resources."""

        return self.rns_context.device


class CkksKeyGenerator:
    r"""Construct CKKS keys from caller-supplied device resources and secret relations.

    Key payloads use configured integral dtype and prime-row order. Secret-key
    data is ``[limb, ntt_index]``; public keys are
    ``[key_component, limb, ntt_index]``; key-switch keys are
    ``[key_digit, key_component, limb, ntt_index]``. Returned keys are always
    NTT/Montgomery/lazy at depth zero in the stated Q or QP basis and own their
    payload storage. Local ``digit_index`` is resolved to stable
    ``key_digit_index`` before key tensor indexing.
    """

    @staticmethod
    def _prime_ids(tensor: torch.Tensor, *, limb_axis: int) -> tuple[int, ...]:
        return tuple(range(tensor.size(limb_axis)))

    @staticmethod
    def _require_ntt_montgomery_secret_key(
        resources: KeyGenerationResource,
        secret_key: SecretKey,
        *,
        operation: str,
    ) -> None:
        if secret_key.data.size(-1) != resources.config.N:
            raise ValueError(
                "SecretKey polynomial degree does not match key generator: "
                f"{secret_key.data.size(-1)} != {resources.config.N}"
            )
        if secret_key.data.device != resources.device:
            raise ValueError(
                "SecretKey device does not match key generator: "
                f"{secret_key.data.device} != {resources.device}"
            )
        expected_prime_ids = resources.rns_context.rns_layout.prime_ids(
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

    @staticmethod
    def _require_qp_secret_key(
        resources: KeyGenerationResource,
        secret_key: SecretKey,
        *,
        operation: str,
    ) -> None:
        CkksKeyGenerator._require_ntt_montgomery_secret_key(
            resources,
            secret_key,
            operation=operation,
        )
        if secret_key.modulus_basis != "QP":
            raise SecretKeyModulusBasisError(
                operation=operation,
                expected="QP",
                actual=secret_key.modulus_basis,
            )

    def convert_secret_key(
        self,
        resources: KeyGenerationResource,
        target: KeyGenerationResource,
        secret_key: SecretKey,
    ) -> SecretKey:
        r"""Re-express the centered secret polynomial in another RNS basis.

        Inverse NTT, centered CRT extension and target NTT preserve the
        polynomial modulo the target primes. The original source basis defines
        its centered representative. Destination storage and Montgomery radix
        belong to target; no sampling or input mutation occurs.
        """
        from fhelium.backend.rns.basis_extension import centered_basis_extend
        from fhelium.backend.rns.tables import centered_basis_extension_tables

        self._require_ntt_montgomery_secret_key(
            resources, secret_key, operation="Secret-key basis conversion"
        )
        if (
            resources.config.N != target.config.N
            or resources.device != target.device
        ):
            raise ValueError(
                "Secret-key basis conversion requires one ring and device"
            )
        source = secret_key.data.clone()
        resources.ntt_context.inverse_to_standard_(
            source, include_p=secret_key.modulus_basis == "QP"
        )
        work_dtype = (
            torch.int64
            if torch.int64 in (source.dtype, target.rns_context.dtype)
            else torch.int32
        )
        work_source = resources.rns_context
        if work_source.dtype != work_dtype:
            work_source = RnsContext(
                resources.config, device=resources.device, dtype=work_dtype
            )
        work_target = target.rns_context
        if work_target.dtype != work_dtype:
            work_target = RnsContext(
                target.config, device=target.device, dtype=work_dtype
            )
        source = source.to(work_dtype)
        target_ids = target.rns_context.rns_layout.prime_ids(0, include_p=True)
        tables = centered_basis_extension_tables(
            work_source, work_target, secret_key.prime_ids, target_ids
        )
        data = centered_basis_extend(source, tables).to(
            target.rns_context.dtype
        )
        target.ntt_context.forward_to_montgomery_(data, include_p=True)
        return SecretKey(data=data, prime_ids=target_ids)

    def prepare_secret_key(
        self,
        resources: KeyGenerationResource,
        coefficients: torch.Tensor,
        *,
        modulus_basis: ModulusBasis = "QP",
    ) -> SecretKey:
        r"""Lift caller-supplied integer coefficients and compute NTT(s)R.

        ``coefficients`` has shape ``[N]`` on the resource device and uses an
        integral dtype. The returned key owns Q or QP storage. This operation
        does not sample, infer a secret distribution, or alter the source.
        """
        if modulus_basis not in ("Q", "QP"):
            raise ValueError(f"Unsupported modulus_basis: {modulus_basis!r}")
        if coefficients.shape != (resources.config.N,):
            raise ValueError("Secret coefficients must have shape [N]")
        if coefficients.dtype not in (torch.int32, torch.int64):
            raise TypeError("Secret coefficients must use int32 or int64")
        if coefficients.device != resources.device:
            raise ValueError(
                "Secret coefficients and resources must share a device"
            )
        include_p = modulus_basis == "QP"
        data = resources.rns_context.lift_integer_coefficients_exact(
            coefficients, include_p=include_p, max_abs=2**63
        )
        resources.ntt_context.forward_to_montgomery_(data, include_p=include_p)
        return SecretKey(
            data=data,
            prime_ids=self._prime_ids(data, limb_axis=0),
            polynomial_domain="ntt",
            modulus_basis=modulus_basis,
            residue_representation="montgomery",
        )

    def create_secret_key(
        self,
        resources: KeyGenerationResource,
        *,
        modulus_basis: ModulusBasis = "QP",
        hamming_weight: int | None = None,
    ) -> SecretKey:
        r"""Sample ternary s and return its depth-zero NTT/Montgomery RNS.

        With no weight, coefficients are independent uniform {-1,0,1}.
        A supplied h selects a uniform h-element support with independent
        uniform signs. Sorting distinct 64-bit CSPRNG labels samples a uniform
        permutation; collisions cause the labels to be resampled. Both label
        and sign sampling consume the supplied cryptographic random stream.
        The output owns [limb, ntt_index] storage on its resource device.
        """
        if modulus_basis not in ("Q", "QP"):
            raise ValueError(f"Unsupported modulus_basis: {modulus_basis!r}")
        if hamming_weight is None:
            coefficients = resources.rng.randint(amax=3, shift=-1, repeats=1)[
                0
            ][0]
        else:
            n = resources.config.N
            if (
                not isinstance(hamming_weight, int)
                or not 0 <= hamming_weight <= n
            ):
                raise ValueError("hamming_weight must be an integer in [0, N]")
            coefficients = torch.zeros(
                n, dtype=resources.rns_context.dtype, device=resources.device
            )
            if hamming_weight:
                if hamming_weight == n:
                    indices = torch.arange(n, device=resources.device)
                else:
                    while True:
                        words = (
                            resources.rng.randbytes(shares=[0], repeats=2)[0][
                                :, :n
                            ].to(torch.int64)
                            & 0xFFFFFFFF
                        )
                        labels = (words[0] << 32) | words[1]
                        ordered, permutation = labels.sort()
                        if not bool((ordered[1:] == ordered[:-1]).any().item()):
                            indices = permutation[:hamming_weight]
                            break
                signs = (
                    resources.rng.randint(amax=2, repeats=1)[0][0][
                        :hamming_weight
                    ]
                    * 2
                    - 1
                )
                coefficients[indices] = signs
        include_p = modulus_basis == "QP"
        data = resources.rns_context.lift_centered_coefficients(
            coefficients, depth=0, include_p=include_p
        )
        resources.ntt_context.forward_to_montgomery_(data, include_p=include_p)
        return SecretKey(
            data=data,
            prime_ids=self._prime_ids(data, limb_axis=0),
            polynomial_domain="ntt",
            modulus_basis=modulus_basis,
            residue_representation="montgomery",
        )

    def create_public_key(
        self,
        resources: KeyGenerationResource,
        secret_key: SecretKey,
        *,
        modulus_basis: ModulusBasis = "Q",
        uniform_component: torch.Tensor | None = None,
        error_coefficients: torch.Tensor | None = None,
    ) -> PublicKey:
        r"""Generate ``(k_0,k_1)`` satisfying $k_0+k_1s=e$ modulo the basis.

        Output is integral ``[key_component=2, limb, ntt_index]`` in
        depth-zero NTT/Montgomery form with Q or QP rows. ``secret_key``
        and optional sampled uniform/error coefficients are read-only and
        never alias the returned stacked tensor. When error coefficients are
        omitted, the generator samples its configured discrete Gaussian.
        """

        if modulus_basis not in ("Q", "QP"):
            raise ValueError(f"Unsupported modulus_basis: {modulus_basis!r}")
        include_p = modulus_basis == "QP"
        if include_p:
            self._require_qp_secret_key(
                resources,
                secret_key,
                operation="Public-key generation on the QP basis",
            )
        else:
            self._require_ntt_montgomery_secret_key(
                resources,
                secret_key,
                operation="Public-key generation",
            )

        depth = 0
        error = (
            resources.rng.discrete_gaussian(repeats=1)[0][0]
            if error_coefficients is None
            else error_coefficients
        )
        if error.shape != (resources.config.N,):
            raise ValueError(
                "Public-key error coefficients must have shape "
                f"[{resources.config.N}]"
            )
        if error.dtype != resources.rns_context.dtype:
            raise TypeError("Public-key error coefficients have another dtype")
        if error.device != resources.device:
            raise ValueError(
                "Public-key error coefficients are on another device"
            )
        error = resources.rns_context.lift_centered_coefficients(
            error, depth, include_p=include_p
        )
        resources.ntt_context.forward_to_montgomery_(error, include_p=include_p)

        if uniform_component is None:
            repeats = resources.config.num_p_primes if include_p else 0
            uniform_component_data = resources.rng.randint(
                [
                    resources.rns_context.moduli_for_basis(
                        depth, include_p=include_p
                    )
                ],
                repeats=repeats,
            )[0]
        else:
            uniform_component_data = uniform_component

        secret_key_rows = secret_key.data
        if not include_p and secret_key.modulus_basis == "QP":
            secret_key_rows = secret_key_rows[: -resources.config.num_p_primes]

        secret_times_uniform = resources.rns_context.montgomery_mul(
            uniform_component_data, secret_key_rows, include_p=include_p
        )
        public_component0 = resources.rns_context.sub_lazy(
            error, secret_times_uniform, include_p=include_p
        )

        public_key_data = torch.stack(
            (public_component0, uniform_component_data), dim=0
        )
        return PublicKey(
            data=public_key_data,
            prime_ids=self._prime_ids(public_key_data, limb_axis=1),
            polynomial_domain="ntt",
            modulus_basis=modulus_basis,
            residue_representation="montgomery",
        )

    def create_key_switch_key(
        self,
        resources: KeyGenerationResource,
        source_secret_key: SecretKey,
        destination_secret_key: SecretKey,
        *,
        uniform_component_by_key_digit: torch.Tensor | None = None,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
        source_depth: int = 0,
        source_multiplier: int = 1,
        error_coefficients_by_key_digit: torch.Tensor | None = None,
    ) -> KeySwitchKey:
        r"""Create a hybrid-RNS key from source to destination secret relation.

        Each hybrid digit is constructed through a public-key encryption.

        Stable key digit $d$ satisfies, on that digit's embedded source rows,
        $k_{d,0}+k_{d,1}s_{\mathrm{dst}}=P m s_{\mathrm{src}}+e_d$.
        Here m is source_multiplier. Caller-supplied digit_prime_ids partition
        the active Q rows in order. At source_depth zero, omission uses the
        configured hybrid decomposition; at later depths it partitions active Q
        by the same product-below-P rule.
        Execution must use the same digit partition and source relation.
        Output is integral
        ``[key_digit, key_component=2, QP_limb, ntt_index]`` in
        NTT/Montgomery lazy form and source-depth Q followed by the selected P.
        Omitted p_prime_ids preserves the original auxiliary basis. Supplied
        auxiliary IDs select distinct catalog primes outside active Q; all
        planes are newly sampled for that P relation.
        Input keys and
        optional sampled uniform/error coefficients are not mutated or aliased.
        """

        self._require_qp_secret_key(
            resources,
            source_secret_key,
            operation="Hybrid key-switch key generation (source key)",
        )
        self._require_qp_secret_key(
            resources,
            destination_secret_key,
            operation="Hybrid key-switch key generation (destination key)",
        )
        depth = source_depth
        active_q = resources.rns_context.rns_layout.prime_ids(depth)
        supplied_p = p_prime_ids is not None
        selected_p = (
            resources.rns_context.rns_chain.p_prime_ids
            if p_prime_ids is None
            else validate_prime_ids(
                p_prime_ids,
                value_name="Key auxiliary basis",
                require_increasing=False,
            )
        )
        if any(
            i >= resources.config.total_num_primes for i in selected_p
        ) or set(selected_p).intersection(active_q):
            raise ValueError(
                "Key auxiliary rows must be catalog primes outside the source Q basis"
            )
        active_qp = (*active_q, *selected_p)
        p_moduli = tuple(resources.config.moduli[i] for i in selected_p)

        source_secret_q = source_secret_key.data[
            : resources.rns_context.q_row_stop
        ].clone()
        product = resources.p_product_montgomery_q
        if supplied_p or source_multiplier != 1:
            product = resources.rns_context.integer_scalar_parameters(
                math.prod(p_moduli) * source_multiplier
            )[: resources.config.num_q_primes]
        resources.rns_context.montgomery_mul_row_scalars_(
            source_secret_q, product
        )
        if digit_prime_ids is None:
            if depth == 0 and not supplied_p:
                digit_prime_ids = tuple(
                    spec.prime_ids
                    for spec in resources.rns_context.rns_layout.digit_specs(0)
                )
            else:
                digit_prime_ids = _partition_q_digits(
                    active_q,
                    resources.config.q_moduli[active_q[0] :],
                    p_moduli,
                )
        elif (
            any(not rows for rows in digit_prime_ids)
            or tuple(i for rows in digit_prime_ids for i in rows) != active_q
        ):
            raise ValueError("Key digits must partition the Q rows in order")
        if (
            error_coefficients_by_key_digit is not None
            and error_coefficients_by_key_digit.shape
            != (len(digit_prime_ids), resources.config.N)
        ):
            raise ValueError(
                "Key-switch error coefficients must have shape "
                f"[{len(digit_prime_ids)}, {resources.config.N}]"
            )
        key_digits: list[torch.Tensor | None] = [None] * len(digit_prime_ids)
        for key_digit_index, source_prime_ids in enumerate(digit_prime_ids):
            uniform_component = (
                uniform_component_by_key_digit[key_digit_index]
                if uniform_component_by_key_digit is not None
                else None
            )
            public_key = self.create_public_key(
                resources,
                destination_secret_key,
                modulus_basis="QP",
                uniform_component=uniform_component,
                error_coefficients=(
                    error_coefficients_by_key_digit[key_digit_index]
                    if error_coefficients_by_key_digit is not None
                    else None
                ),
            )

            source_digit_key = tuple(source_prime_ids)
            row_start = source_prime_ids[0]
            row_stop = source_prime_ids[-1] + 1
            source_secret_digit = source_secret_q[row_start:row_stop]
            public_key_digit_c0 = public_key.k0[row_start:row_stop]
            twice_modulus = resources.rns_context.rns_parameters_for_prime_ids(
                source_digit_key
            )[0]
            updated_c0_digit = rns_ops.add_lazy_with_twice_modulus(
                public_key_digit_c0, source_secret_digit, twice_modulus
            )
            public_key_digit_c0.copy_(updated_c0_digit, non_blocking=True)
            if active_qp == tuple(
                range(active_qp[0], resources.config.total_num_primes)
            ):
                key_digits[key_digit_index] = public_key.data[
                    :, active_qp[0] :
                ].contiguous()
            else:
                rows = torch.tensor(
                    active_qp, dtype=torch.long, device=resources.device
                )
                key_digits[key_digit_index] = public_key.data.index_select(
                    1, rows
                )

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
            prime_ids=active_qp,
            digit_prime_ids=digit_prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )

    def create_encapsulation_keys(
        self,
        resources: KeyGenerationResource,
        source_secret_key: SecretKey,
        intermediate_secret_key: SecretKey,
        *,
        source_depth: int,
        low_p_moduli: tuple[int, ...],
        return_prime: int,
        return_p_moduli: tuple[int, ...],
    ) -> tuple[KeySwitchKey, KeySwitchKey]:
        r"""Generate low s→t digits and one guard-scaled t→s return hint.

        Low digits encrypt P_low*E_j*s under t. The return digit encrypts
        guard*P_return*t under s across [guard,Q,P_return]. Both secrets are
        supplied by the caller; this factory samples only public-key randomness
        and error on the shared CSPRNG. It returns ordinary KeySwitchKey values.
        """
        from fhelium.backend.ckks.tables import encapsulation_configs

        low_config, return_config = encapsulation_configs(
            resources.config,
            source_depth,
            low_p_moduli,
            return_prime,
            return_p_moduli,
        )
        low = resources.for_config(
            low_config, dtype=resources.rns_context.dtype
        )
        high = resources.for_config(
            return_config, dtype=resources.rns_context.dtype
        )
        low_source = self.convert_secret_key(resources, low, source_secret_key)
        low_intermediate = self.convert_secret_key(
            resources, low, intermediate_secret_key
        )
        low_key = self.create_key_switch_key(low, low_source, low_intermediate)
        high_source = self.convert_secret_key(
            resources, high, source_secret_key
        )
        high_intermediate = self.convert_secret_key(
            resources, high, intermediate_secret_key
        )
        return_key = self.create_key_switch_key(
            high,
            high_intermediate,
            high_source,
            digit_prime_ids=(tuple(range(return_config.num_q_primes)),),
            source_multiplier=return_prime,
        )
        return low_key, return_key

    def create_relinearization_key(
        self,
        resources: KeyGenerationResource,
        secret_key: SecretKey,
        *,
        source_depth: int = 0,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
    ) -> RelinearizationKey:
        r"""Return QP key material that switches the $s^2$ phase term to $s$.

        The returned state and layout equal :meth:`create_key_switch_key`; the
        input secret key remains NTT/Montgomery QP and is not mutated.
        """

        self._require_qp_secret_key(
            resources,
            secret_key,
            operation="Relinearization-key generation",
        )
        squared_secret_key_data = resources.rns_context.montgomery_mul(
            secret_key.data, secret_key.data, include_p=True
        )
        squared_secret_key = SecretKey(
            data=squared_secret_key_data,
            prime_ids=secret_key.prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )
        key_switch_key = self.create_key_switch_key(
            resources,
            squared_secret_key,
            secret_key,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
        )
        return RelinearizationKey(
            data=key_switch_key.data,
            prime_ids=key_switch_key.prime_ids,
            digit_prime_ids=key_switch_key.digit_prime_ids,
            polynomial_domain=key_switch_key.polynomial_domain,
            modulus_basis=key_switch_key.modulus_basis,
            residue_representation=key_switch_key.residue_representation,
        )

    def create_rotation_key(
        self,
        resources: KeyGenerationResource,
        rotation_step: int,
        *,
        uniform_component_by_key_digit: torch.Tensor | None = None,
        secret_key: SecretKey,
        source_depth: int = 0,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
    ) -> RotationKey:
        r"""Construct QP key material from $\sigma_g(s)$ back to $s$.

        ``rotation_step`` follows signed slot displacement and is stored as
        metadata. The distinct ``galois_element`` $g$ selects polynomial
        automorphism $\sigma_g$. Output uses key-switch-key axes and
        NTT/Montgomery QP state; input secret/uniform tensors are not mutated.
        """

        self._require_qp_secret_key(
            resources,
            secret_key,
            operation="Rotation-key generation",
        )
        rotation_step = RotationKey.normalize_step(
            rotation_step,
            ring_dimension=resources.config.N,
        )
        rotated = secret_key.data.clone()
        resources.ntt_context.inverse_montgomery_(rotated, include_p=True)
        rotated = apply_coefficient_galois_automorphism(
            rotated,
            rotation_galois_element(
                resources.config.N,
                rotation_step,
                resources.config.galois_generator,
            ),
            resources.rns_context.moduli,
        )
        resources.ntt_context.forward_montgomery_(rotated, include_p=True)
        rotated_secret_key = SecretKey(
            data=rotated,
            prime_ids=secret_key.prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )
        key_switch_key = self.create_key_switch_key(
            resources,
            rotated_secret_key,
            secret_key,
            uniform_component_by_key_digit=uniform_component_by_key_digit,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
        )
        return RotationKey(
            data=key_switch_key.data,
            prime_ids=key_switch_key.prime_ids,
            digit_prime_ids=key_switch_key.digit_prime_ids,
            polynomial_domain=key_switch_key.polynomial_domain,
            modulus_basis=key_switch_key.modulus_basis,
            residue_representation=key_switch_key.residue_representation,
            rotation_step=rotation_step,
        )

    def create_conjugation_key(
        self,
        resources: KeyGenerationResource,
        secret_key: SecretKey,
        *,
        source_depth: int = 0,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None = None,
        p_prime_ids: tuple[int, ...] | None = None,
    ) -> ConjugationKey:
        r"""Construct QP key material from $\sigma_{2N-1}(s)$ back to $s$.

        Output uses key-switch-key axes and NTT/Montgomery QP state; the input
        secret key remains unchanged.
        """

        self._require_qp_secret_key(
            resources,
            secret_key,
            operation="Conjugation-key generation",
        )

        conjugated = secret_key.data.clone()
        resources.ntt_context.inverse_montgomery_(conjugated, include_p=True)
        conjugated = apply_coefficient_galois_automorphism(
            conjugated,
            2 * resources.config.N - 1,
            resources.rns_context.moduli,
        )
        resources.ntt_context.forward_montgomery_(conjugated, include_p=True)
        conjugated_secret_key = SecretKey(
            data=conjugated,
            prime_ids=secret_key.prime_ids,
            polynomial_domain="ntt",
            modulus_basis="QP",
            residue_representation="montgomery",
        )
        key_switch_key = self.create_key_switch_key(
            resources,
            conjugated_secret_key,
            secret_key,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
        )
        return ConjugationKey(
            data=key_switch_key.data,
            prime_ids=key_switch_key.prime_ids,
            digit_prime_ids=key_switch_key.digit_prime_ids,
            polynomial_domain=key_switch_key.polynomial_domain,
            modulus_basis=key_switch_key.modulus_basis,
            residue_representation=key_switch_key.residue_representation,
        )
