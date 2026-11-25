"""CKKS secret, public, rotation, conjugation, and key-switch key generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import cast


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
    SeedSpec,
)
from ..rotation._galois import rotation_galois_element
from fhelium.errors import (
    PolynomialDomainError,
    ResidueRepresentationError,
    SecretKeyModulusBasisError,
)
from fhelium.native.wrapper import ckks_ops, rns_ops
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.decomposition import _partition_q_digits
from fhelium.values._validation import validate_prime_ids
from fhelium.backend.random.stream import RandomStream, advance_


@dataclass(frozen=True)
class KeyGenerationResource:
    """Bind context, arithmetic, and random resources for one key device."""

    config: CkksConfig
    rng: RandomStream
    rns_context: RnsContext
    ntt_context: NttContext
    _tables: dict[object, object] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.rns_context.config != self.config:
            raise ValueError(
                "Key-generation RNS context differs from its CKKS configuration"
            )
        if self.ntt_context.rns_context is not self.rns_context:
            raise ValueError(
                "Key-generation NTT context composes another RNS context"
            )
        if self.rng.device != self.device:
            raise ValueError(
                "Key-generation random stream must own exactly its device"
            )

    @classmethod
    def create(
        cls,
        *,
        config: CkksConfig,
        rng: RandomStream,
        rns_context: RnsContext,
        ntt_context: NttContext,
    ) -> KeyGenerationResource:
        """Bind key-generation resources on one concrete context device."""

        return cls(config, rng, rns_context, ntt_context)

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

    @property
    def ntt_backend(self) -> str:
        """Return the NTT policy name of the native key-generation calls."""

        return self.ntt_context.ntt_backend_name

    def tables(
        self, prime_ids: tuple[int, ...], basis: str = ""
    ) -> dict[str, torch.Tensor]:
        """Return the parameters and NTT schedules of ``prime_ids``.

        An empty ``basis`` names them ``parameters``, ``forward_{i}`` and
        ``inverse_{i}``; another basis prefixes the schedule names with it.
        """
        key = ("tables", prime_ids, basis)
        tables = self._tables.get(key)
        if tables is None:
            prefix = f"{basis}_" if basis else ""
            tables = {}
            if not basis:
                tables["parameters"] = (
                    self.rns_context.rns_parameters_for_prime_ids(prime_ids)
                )
            for direction in ("forward", "inverse"):
                operands = self.ntt_context.tensor_operands(
                    prime_ids, inverse=direction == "inverse"
                )
                tables.update(
                    (f"{prefix}{direction}_{i}", value)
                    for i, value in enumerate(operands)
                )
            self._tables[key] = tables
        return cast(dict[str, torch.Tensor], tables)

    def source_product(
        self, p_moduli: tuple[int, ...], multiplier: int
    ) -> torch.Tensor:
        r"""Return $(m \prod P) R \bmod q_i$ for every Q row."""
        key = ("source_product", p_moduli, multiplier)
        product = self._tables.get(key)
        if product is None:
            factors = [*p_moduli, *([multiplier] if multiplier != 1 else [])]
            product = rns_ops.product_residues(
                factors,
                list(self.config.q_moduli),
                1,
                True,
                self.rns_context.dtype,
            ).to(self.device)
            self._tables[key] = product
        return cast(torch.Tensor, product)


class CkksKeyGenerator:
    r"""Construct CKKS keys from caller-supplied device resources and secret relations.

    Key payloads use configured integral dtype and prime-row order. Secret-key
    data is ``[limb, ntt_index]``; public keys are
    ``[key_component, limb, ntt_index]``; key-switch keys are
    ``[key_digit, key_component, limb, ntt_index]``. Returned keys are always
    NTT/Montgomery/lazy at depth zero in the stated Q or QP basis and own their
    payload storage. Local ``digit_index`` is resolved to stable
    ``key_digit_index`` before key tensor indexing. Generated key-switching
    keys carry a seed; ``expand_keys`` selects whether they also hold the
    expanded uniform component.
    """

    def __init__(self, *, expand_keys: bool = True) -> None:
        self.expand_keys = expand_keys

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
        work_dtype = (
            torch.int64
            if torch.int64
            in (resources.rns_context.dtype, target.rns_context.dtype)
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
        target_ids = target.rns_context.rns_layout.prime_ids(0, include_p=True)
        tables = {
            **resources.tables(secret_key.prime_ids, "source"),
            **target.tables(target_ids, "target"),
            **centered_basis_extension_tables(
                work_source, work_target, secret_key.prime_ids, target_ids
            ),
        }
        data = ckks_ops.convert_secret_key(
            secret_key.data,
            list(tables),
            list(tables.values()),
            resources.ntt_backend,
        )
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
        tables = resources.tables(
            resources.rns_context.rns_layout.prime_ids(
                0, include_p=modulus_basis == "QP"
            )
        )
        data = ckks_ops.prepare_secret_key(
            coefficients,
            list(tables),
            list(tables.values()),
            resources.ntt_backend,
        )
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

        With no weight, coefficients are independent uniform {-1,0,1}. A
        supplied h selects an h-element support by the smallest 127-bit
        random labels, with independent uniform signs. Both samplers consume
        one nonce of the device's random stream.
        The output owns [limb, ntt_index] storage on its resource device.
        """
        if modulus_basis not in ("Q", "QP"):
            raise ValueError(f"Unsupported modulus_basis: {modulus_basis!r}")
        n = resources.config.N
        if hamming_weight is not None and (
            not isinstance(hamming_weight, int) or not 0 <= hamming_weight <= n
        ):
            raise ValueError("hamming_weight must be an integer in [0, N]")
        tables = resources.tables(
            resources.rns_context.rns_layout.prime_ids(
                0, include_p=modulus_basis == "QP"
            )
        )
        data = ckks_ops.create_secret_key(
            n,
            list(tables),
            list(tables.values()),
            resources.rng.seed,
            resources.ntt_backend,
            hamming_weight,
        )
        advance_(resources.rng.seed)
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
        Each sampled component consumes one nonce, error first.
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
        if error_coefficients is not None:
            if error_coefficients.shape != (resources.config.N,):
                raise ValueError(
                    "Public-key error coefficients must have shape "
                    f"[{resources.config.N}]"
                )
            if error_coefficients.dtype != resources.rns_context.dtype:
                raise TypeError(
                    "Public-key error coefficients have another dtype"
                )
            if error_coefficients.device != resources.device:
                raise ValueError(
                    "Public-key error coefficients are on another device"
                )
        tables = resources.tables(
            resources.rns_context.rns_layout.prime_ids(0, include_p=include_p)
        )
        public_key_data = ckks_ops.create_public_key(
            secret_key.data,
            list(tables),
            list(tables.values()),
            resources.rng.seed,
            resources.rng.cdt_table,
            resources.ntt_backend,
            uniform_component,
            error_coefficients,
        )
        advance_(
            resources.rng.seed,
            (uniform_component is None) + (error_coefficients is None),
        )
        return PublicKey(
            data=public_key_data,
            prime_ids=self._prime_ids(public_key_data, limb_axis=1),
            polynomial_domain="ntt",
            modulus_basis=modulus_basis,
            residue_representation="montgomery",
        )

    def _key_switch_layout(
        self,
        resources: KeyGenerationResource,
        *,
        source_depth: int,
        digit_prime_ids: tuple[tuple[int, ...], ...] | None,
        p_prime_ids: tuple[int, ...] | None,
        source_multiplier: int,
        uniform_component_by_key_digit: torch.Tensor | None,
        error_coefficients_by_key_digit: torch.Tensor | None,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Select the key rows and digits; return native arguments and metadata."""
        active_q = resources.rns_context.rns_layout.prime_ids(source_depth)
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
        p_moduli = tuple(resources.config.moduli[i] for i in selected_p)
        if digit_prime_ids is None:
            digit_prime_ids = _partition_q_digits(
                active_q,
                tuple(resources.config.q_moduli[i] for i in active_q),
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
        all_rows = resources.rns_context.rns_layout.prime_ids(0, include_p=True)
        tables = {
            **resources.tables(all_rows),
            "source_product": resources.source_product(
                p_moduli, source_multiplier
            ),
        }
        arguments = (
            list(tables),
            list(tables.values()),
            resources.rng.seed,
            resources.rng.cdt_table,
            resources.ntt_backend,
            list(active_q),
            [len(rows) for rows in digit_prime_ids],
            list(selected_p),
            uniform_component_by_key_digit,
            error_coefficients_by_key_digit,
        )
        metadata: dict[str, object] = {
            "prime_ids": (*active_q, *selected_p),
            "digit_prime_ids": digit_prime_ids,
            "polynomial_domain": "ntt",
            "modulus_basis": "QP",
            "residue_representation": "montgomery",
        }
        return arguments, metadata

    def _key(
        self,
        cls: type[KeySwitchKey],
        resources: KeyGenerationResource,
        result: tuple[torch.Tensor, torch.Tensor | None, list[int], int],
        arguments: tuple[object, ...],
        metadata: dict[str, object],
        **extra: object,
    ) -> KeySwitchKey:
        """Wrap a native key, advance the stream, and compact it unless expanded."""
        data, seed, moduli, rounds = result
        digits = len(cast(tuple, metadata["digit_prime_ids"]))
        advance_(
            resources.rng.seed,
            (arguments[-2] is None) + digits * (arguments[-1] is None),
        )
        if seed is not None:
            metadata = {
                **metadata,
                "seed": seed,
                "seed_spec": SeedSpec(moduli=tuple(moduli), rounds=rounds),
            }
        key = cls(data=data, **metadata, **extra)
        return key if self.expand_keys or seed is None else key.compact()

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
        the active Q rows in order; omission partitions active Q into
        consecutive digits whose products are below P.
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
        arguments, metadata = self._key_switch_layout(
            resources,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
            source_multiplier=source_multiplier,
            uniform_component_by_key_digit=uniform_component_by_key_digit,
            error_coefficients_by_key_digit=error_coefficients_by_key_digit,
        )
        result = ckks_ops.create_key_switch_key(
            source_secret_key.data, destination_secret_key.data, *arguments
        )
        return self._key(KeySwitchKey, resources, result, arguments, metadata)

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
        arguments, metadata = self._key_switch_layout(
            resources,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
            source_multiplier=1,
            uniform_component_by_key_digit=None,
            error_coefficients_by_key_digit=None,
        )
        result = ckks_ops.create_relinearization_key(secret_key.data, *arguments)
        return cast(
            RelinearizationKey,
            self._key(RelinearizationKey, resources, result, arguments, metadata),
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
        arguments, metadata = self._key_switch_layout(
            resources,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
            source_multiplier=1,
            uniform_component_by_key_digit=uniform_component_by_key_digit,
            error_coefficients_by_key_digit=None,
        )
        result = ckks_ops.create_galois_key(
            secret_key.data,
            rotation_galois_element(
                resources.config.N,
                rotation_step,
                resources.config.galois_generator,
            ),
            *arguments,
        )
        return cast(
            RotationKey,
            self._key(
                RotationKey,
                resources,
                result,
                arguments,
                metadata,
                rotation_step=rotation_step,
            ),
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
        arguments, metadata = self._key_switch_layout(
            resources,
            source_depth=source_depth,
            digit_prime_ids=digit_prime_ids,
            p_prime_ids=p_prime_ids,
            source_multiplier=1,
            uniform_component_by_key_digit=None,
            error_coefficients_by_key_digit=None,
        )
        result = ckks_ops.create_galois_key(
            secret_key.data, 2 * resources.config.N - 1, *arguments
        )
        return cast(
            ConjugationKey,
            self._key(ConjugationKey, resources, result, arguments, metadata),
        )
