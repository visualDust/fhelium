"""Execute eager CKKS operations through registered Backend implementations.

`Engine` owns one CKKS configuration and lazily constructs arithmetic,
random-stream, key-material, and executable resources for each device used by
the caller. Ordinary evaluator methods dispatch from their Tensor operands;
factory-like calls use CPU unless another device is selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index as integer_index
from threading import RLock
from typing import Literal, cast, overload

import torch
from xdsl.ir import Operation

from fhelium.config import CkksConfig, Preset
from fhelium.values import (
    Ciphertext,
    CompressedPlaintext,
    ConjugationKey,
    KeySwitchKey,
    Plaintext,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from fhelium.values._scale import coerce_scale
from fhelium.utils.rotation import decompose_rotation_step
from fhelium.values.state import (
    ModulusBasis,
    PolynomialDomain,
    ResidueRepresentation,
)
from fhelium.errors import MaximumLevelError, ScaleMismatchError
from fhelium.backend.ckks import (
    CkksKeyGenerator,
    PUBLIC_KEY_RESOURCE_KIND,
    SECRET_KEY_RESOURCE_KIND,
)
from fhelium.backend.ckks.crypto import reconstruct_tail_q_coefficients_tensor
from fhelium.backend.resources import (
    BoundResource,
    ResourceBindings,
)
from fhelium.backend.execution import (
    OperationBackend,
)
from fhelium.backend.rns.chain import RnsChain
from fhelium.backend.rns.decomposition import HybridRnsDecomposition
from fhelium.backend.rns.layout import RnsLayout
from fhelium.backend.ntt.context import NttContext
from fhelium.backend.ntt.resources import (
    NTT_RESOURCE_KIND,
)
from fhelium.backend.rns.context import RnsContext
from fhelium.backend.rns.resources import (
    RNS_RESOURCE_KIND,
)
from fhelium.backend.ckks.materialization import (
    CkksDeviceResources,
)
from fhelium.backend.assembly import create_builtin_operation_registry
from fhelium.backend.ckks.resources import (
    KEY_SWITCH_PLAN_RESOURCE_KIND,
    RESCALE_RESOURCE_KIND,
    ckks_key_resource_kind,
)
from fhelium.eager._operation_dispatch import _EagerOperationDispatcher
from fhelium.eager._validation import CkksValidator
from fhelium.eager._key_inventory import KeyInventory
from fhelium.ir.dialects import ckks, ntt, rns
from fhelium.rng import Csprng


def _ciphertext_result(
    source: Ciphertext,
    data: torch.Tensor,
    *,
    level: int | None = None,
    scale: float | None = None,
    prime_ids: tuple[int, ...] | None = None,
    polynomial_domain: PolynomialDomain | None = None,
    modulus_basis: ModulusBasis | None = None,
    residue_representation: ResidueRepresentation | None = None,
) -> Ciphertext:
    """Build an Eager result from one source value and operation transition."""

    return Ciphertext._from_fields(
        data=data,
        level=source.level if level is None else level,
        scale=source.scale if scale is None else scale,
        prime_ids=source.prime_ids if prime_ids is None else prime_ids,
        polynomial_domain=(
            source.polynomial_domain
            if polynomial_domain is None
            else polynomial_domain
        ),
        modulus_basis=(
            source.modulus_basis if modulus_basis is None else modulus_basis
        ),
        residue_representation=(
            source.residue_representation
            if residue_representation is None
            else residue_representation
        ),
    )


def _plaintext_result(
    source: Plaintext,
    data: torch.Tensor,
    *,
    polynomial_domain: PolynomialDomain | None = None,
    residue_representation: ResidueRepresentation | None = None,
) -> Plaintext:
    """Build an Eager RNS plaintext result from one operation transition."""

    return Plaintext._from_fields(
        message=None,
        level=source.level,
        scale=source.scale,
        data=data,
        representation=source.representation,
        polynomial_domain=(
            source.polynomial_domain
            if polynomial_domain is None
            else polynomial_domain
        ),
        modulus_basis=source.modulus_basis,
        residue_representation=(
            source.residue_representation
            if residue_representation is None
            else residue_representation
        ),
        prime_ids=source.prime_ids,
    )


class Engine:
    """Own eager CKKS configuration and per-device execution resources.

    Source factories use PyTorch's default device when the caller omits
    ``device``.
    Tensor-consuming operations dispatch from operand placement and do not move
    operands implicitly. A boundary ``device=`` authorizes movement of that
    boundary's public value. Key material is not copied between devices unless
    the caller places it there or enables automatic key replication. Each used
    device owns its arithmetic tables, random stream, direct-dispatch entries, and key
    bindings.
    """

    def __init__(
        self,
        ckks_config: CkksConfig | Preset | dict[str, object] | None = None,
        *,
        ntt_backend: str | None = None,
        rng_seed: int | None = None,
        rng_nonce: int | None = None,
        allow_automatic_key_generation: bool = True,
        allow_automatic_key_replication: bool = False,
    ) -> None:
        """Construct one CKKS lifecycle with lazily created device resources."""

        if ckks_config is None:
            ckks_config = Preset.slots16384_scale40_levels16_int64
        if not isinstance(ckks_config, CkksConfig):
            ckks_config = CkksConfig.parse(ckks_config)
        if type(allow_automatic_key_replication) is not bool:
            raise TypeError("allow_automatic_key_replication must be a bool")
        if ckks_config.enforce_security_budget:
            ckks_config.validate_security_budget()

        self._config = ckks_config
        self._ntt_backend = ntt_backend
        self._rng_seed = rng_seed
        self._rng_nonce = rng_nonce
        self.allow_automatic_key_replication = allow_automatic_key_replication
        self._dispatchers: dict[torch.device, _EagerOperationDispatcher] = {}
        self._dispatcher_lock = RLock()
        self._key_generator = CkksKeyGenerator()
        self.allow_automatic_key_generation = allow_automatic_key_generation
        rns_chain = RnsChain(
            num_q_primes=ckks_config.num_q_primes,
            num_p_primes=ckks_config.num_p_primes,
        )
        self._rns_layout = RnsLayout(
            rns_chain,
            HybridRnsDecomposition(rns_chain),
        )
        self._validator = CkksValidator(
            ckks_config,
            self._rns_layout,
        )
        self._keys = KeyInventory(self._validator)
        self._installed_evaluation_keys: dict[str, KeySwitchKey] = {}
        self._key_replicas: dict[
            tuple[int, int, int, torch.device],
            PublicKey | SecretKey | KeySwitchKey,
        ] = {}

    def _dispatcher_for(
        self,
        device: torch.device | str,
    ) -> _EagerOperationDispatcher:
        selected_device = torch.device(device)
        cached = self._dispatchers.get(selected_device)
        if cached is not None:
            return cached
        with self._dispatcher_lock:
            cached = self._dispatchers.get(selected_device)
            if cached is not None:
                return cached
            return self._create_dispatcher(selected_device)

    def _operation_backend(
        self,
        device: torch.device | str,
    ) -> OperationBackend:
        """Return the shared operation Backend for one device."""

        dispatcher = self._dispatcher_for(device)
        dispatcher.materialize_all()
        return dispatcher.backend

    def _rns_context_for(
        self,
        device: torch.device | str,
    ) -> RnsContext:
        """Return the device-local RNS context for specialized algorithms."""

        return self._dispatcher_for(device).rns_context

    def _ntt_context_for(
        self,
        device: torch.device | str,
    ) -> NttContext:
        """Return the device-local NTT context for specialized algorithms."""

        return self._dispatcher_for(device).ntt_context

    def _rng_for(
        self,
        device: torch.device | str,
    ) -> Csprng:
        """Return the random stream for one concrete device."""

        return self._dispatcher_for(device).rng

    def _create_dispatcher(
        self,
        selected_device: torch.device,
    ) -> _EagerOperationDispatcher:
        """Construct one Eager dispatcher while holding its map lock."""

        device_resources = CkksDeviceResources(
            config=self.config,
            rns_layout=self._rns_layout,
            device=selected_device,
            ntt_backend=self._ntt_backend,
            rng_seed=self._rng_seed,
            rng_nonce=self._rng_nonce,
        )
        ntt_context = device_resources.ntt_context
        operation_registry = create_builtin_operation_registry(
            ntt_backend_name=ntt_context.ntt_backend_name,
        )
        backend = OperationBackend(operation_registry)
        dispatcher = _EagerOperationDispatcher(
            device_resources=device_resources,
            backend=backend,
        )
        self._dispatchers[selected_device] = dispatcher
        return dispatcher

    @property
    def config(self) -> CkksConfig:
        """Return the Engine's fixed CKKS configuration."""

        return self._config

    @property
    def galois_generator(self) -> int:
        """Return the Engine's Galois generator."""

        return self.config.galois_generator

    def ntt_backend_name(self, device: torch.device | str) -> str:
        """Return the NTT backend selected for one concrete device."""

        return self._ntt_context_for(device).ntt_backend_name

    @property
    def ring_dimension(self) -> int:
        """Return the Engine's polynomial ring dimension."""

        return self.config.N

    @property
    def level0_qp_prime_ids(self) -> tuple[int, ...]:
        """Return the level-zero QP prime identifiers."""

        return self._rns_layout.prime_ids(0, include_p=True)

    @property
    def key_digit_count(self) -> int:
        """Return the hybrid key-switch digit count."""

        return self._rns_layout.key_digit_count

    @property
    def dtype(self) -> torch.dtype:
        """Return the Engine's RNS storage dtype."""

        return self.config.torch_dtype

    @property
    def public_level_count(self) -> int:
        """Return the number of Q-chain levels available to public values."""

        return self.config.num_scale_primes

    @property
    def final_public_level(self) -> int:
        """Return the last valid public level index."""

        return self.public_level_count - 1

    @property
    def num_slots(self) -> int:
        """Return the number of complex slots in one CKKS polynomial."""

        return self.config.N // 2

    def _execute(
        self,
        operation_type: type[Operation],
        *inputs: torch.Tensor,
        resource_kinds: Sequence[str] = (),
        resources: Sequence[BoundResource] = (),
        bindings: ResourceBindings | None = None,
        attributes: Mapping[str, object] | None = None,
        bases: Sequence[str | None] = (),
        result_count: int = 1,
        implementation: str | None = None,
        in_place: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Dispatch one operation through the input device's resource bundle."""

        dispatcher = self._dispatcher_for(inputs[0].device)
        fixed: list[BoundResource] = []
        for kind in resource_kinds:
            if kind == "rns":
                fixed.extend(
                    dispatcher.resources(
                        (("active-rns-parameters", RNS_RESOURCE_KIND),)
                    )
                )
            elif kind == "ntt":
                fixed.extend(
                    dispatcher.resources(
                        (("active-ntt-plan", NTT_RESOURCE_KIND),)
                    )
                )
            elif kind == "rescale":
                fixed.extend(
                    dispatcher.resources(
                        (("active-rescale-plan", RESCALE_RESOURCE_KIND),)
                    )
                )
            elif kind == "key_switch":
                fixed.extend(
                    dispatcher.resources(
                        (
                            (
                                "active-key-switch-plan",
                                KEY_SWITCH_PLAN_RESOURCE_KIND,
                            ),
                        )
                    )
                )
            else:
                raise ValueError(f"Unknown Eager resource group {kind!r}")

        selected_resources = (*fixed, *resources)
        if operation_type in {
            ckks.AddScalarOp,
            ckks.EncodeOp,
            ckks.EncryptOp,
            ckks.MultiplyScalarOp,
        }:
            with dispatcher.random_lock:
                return dispatcher.execute(
                    operation_type,
                    *inputs,
                    resources=selected_resources,
                    bindings=bindings,
                    attributes=attributes,
                    bases=bases,
                    result_count=result_count,
                    implementation=implementation,
                    in_place=in_place,
                )
        return dispatcher.execute(
            operation_type,
            *inputs,
            resources=selected_resources,
            bindings=bindings,
            attributes=attributes,
            bases=bases,
            result_count=result_count,
            implementation=implementation,
            in_place=in_place,
        )

    def _key_switch_bindings(
        self,
        device: torch.device,
        *keys: BoundResource,
    ) -> ResourceBindings:
        dispatcher = self._dispatcher_for(device)
        key_switch = dispatcher.resources(
            (
                (
                    "active-key-switch-plan",
                    KEY_SWITCH_PLAN_RESOURCE_KIND,
                ),
            )
        )
        return ResourceBindings((*key_switch, *keys))

    def create_secret_key(
        self,
        *,
        modulus_basis: Literal["Q", "QP"] = "QP",
        device: torch.device | str | None = None,
    ) -> SecretKey:
        """Generate a secret key in NTT/Montgomery representation."""

        target = torch.get_default_device() if device is None else device
        dispatcher = self._dispatcher_for(target)
        with dispatcher.random_lock:
            return self._key_generator.create_secret_key(
                dispatcher.key_generation,
                modulus_basis=modulus_basis,
            )

    def create_public_key(
        self,
        secret_key: SecretKey,
        *,
        modulus_basis: Literal["Q", "QP"] = "Q",
        device: torch.device | str | None = None,
    ) -> PublicKey:
        """Generate public encryption material for `secret_key`."""

        self._validator.validate_secret_key(secret_key)
        selected_secret = (
            secret_key.to(device) if device is not None else secret_key
        )
        dispatcher = self._dispatcher_for(selected_secret.device)
        with dispatcher.random_lock:
            return self._key_generator.create_public_key(
                dispatcher.key_generation,
                selected_secret,
                modulus_basis=modulus_basis,
            )

    def create_key_switch_key(
        self,
        source_secret_key: SecretKey,
        destination_secret_key: SecretKey,
        *,
        uniform_component_by_key_digit: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> KeySwitchKey:
        """Generate hybrid material between two secret-key relations."""

        self._validator.validate_secret_key(source_secret_key)
        self._validator.validate_secret_key(destination_secret_key)
        if device is None:
            source = source_secret_key
            destination = destination_secret_key
        else:
            source = source_secret_key.to(device)
            destination = destination_secret_key.to(device)
            if uniform_component_by_key_digit is not None:
                uniform_component_by_key_digit = (
                    uniform_component_by_key_digit.to(device)
                )
        dispatcher = self._dispatcher_for(source.device)
        with dispatcher.random_lock:
            return self._key_generator.create_key_switch_key(
                dispatcher.key_generation,
                source,
                destination,
                uniform_component_by_key_digit=uniform_component_by_key_digit,
            )

    def create_relinearization_key(
        self,
        secret_key: SecretKey,
        *,
        device: torch.device | str | None = None,
    ) -> RelinearizationKey:
        """Generate CT3-to-CT2 relinearization material."""

        self._validator.validate_secret_key(secret_key)
        selected = secret_key.to(device) if device is not None else secret_key
        dispatcher = self._dispatcher_for(selected.device)
        with dispatcher.random_lock:
            return self._key_generator.create_relinearization_key(
                dispatcher.key_generation,
                selected,
            )

    def create_rotation_key(
        self,
        rotation_step: int,
        secret_key: SecretKey,
        *,
        device: torch.device | str | None = None,
    ) -> RotationKey:
        """Generate material for one normalized CKKS slot rotation."""

        self._validator.validate_secret_key(secret_key)
        selected = secret_key.to(device) if device is not None else secret_key
        dispatcher = self._dispatcher_for(selected.device)
        with dispatcher.random_lock:
            return self._key_generator.create_rotation_key(
                dispatcher.key_generation,
                rotation_step,
                secret_key=selected,
            )

    def create_conjugation_key(
        self,
        secret_key: SecretKey,
        *,
        device: torch.device | str | None = None,
    ) -> ConjugationKey:
        """Generate CKKS slot-conjugation material."""

        self._validator.validate_secret_key(secret_key)
        selected = secret_key.to(device) if device is not None else secret_key
        dispatcher = self._dispatcher_for(selected.device)
        with dispatcher.random_lock:
            return self._key_generator.create_conjugation_key(
                dispatcher.key_generation,
                selected,
            )

    @staticmethod
    def _key_replica_token(
        key: PublicKey | SecretKey | KeySwitchKey,
    ) -> tuple[int, int, int]:
        data = key.data
        return id(key), id(data), data._version

    def _key_on_device(
        self,
        key: PublicKey | SecretKey | KeySwitchKey,
        device: torch.device,
        *,
        operation_name: str,
    ) -> PublicKey | SecretKey | KeySwitchKey:
        if key.device == device:
            return key
        if not self.allow_automatic_key_replication:
            raise ValueError(
                f"{operation_name} requires {type(key).__name__} material on "
                f"{device}, but the selected key is on {key.device}. Create "
                f"a device copy with key.to({str(device)!r}) and pass or "
                "install that copy, or enable "
                "Engine(allow_automatic_key_replication=True)."
            )
        if not self._owns_key(key):
            return key.to(device)
        token = (*self._key_replica_token(key), device)
        self._key_replicas = {
            cached_token: replica
            for cached_token, replica in self._key_replicas.items()
            if cached_token[0] != id(key) or cached_token == token
        }
        replica = self._key_replicas.get(token)
        if replica is None:
            replica = key.to(device)
            self._key_replicas[token] = replica
        return replica

    def _owns_key(
        self,
        key: PublicKey | SecretKey | KeySwitchKey,
    ) -> bool:
        """Return whether the Engine inventory owns a key's lifecycle."""

        return (
            key is self._keys.secret_key
            or key is self._keys.public_key
            or (
                key is self._keys.relinearization_key
                or key is self._keys.conjugation_key
                or any(
                    key is rotation_key
                    for rotation_key in self._keys.rotation_keys.values()
                )
                or any(
                    key is installed
                    for installed in self._installed_evaluation_keys.values()
                )
            )
        )

    def _discard_key_replicas(
        self,
        key: PublicKey | SecretKey | KeySwitchKey,
    ) -> None:
        key_id = id(key)
        self._key_replicas = {
            token: replica
            for token, replica in self._key_replicas.items()
            if token[0] != key_id
        }

    def _key_resource(
        self,
        symbol: str,
        key: PublicKey | SecretKey,
    ) -> ResourceBindings:
        kind = (
            PUBLIC_KEY_RESOURCE_KIND
            if isinstance(key, PublicKey)
            else SECRET_KEY_RESOURCE_KIND
        )
        dispatcher = self._dispatcher_for(key.device)
        token = id(key)
        cached = dispatcher.operation_key_resources.get(symbol)
        if cached is not None and cached[0] == token:
            return ResourceBindings((cached[1],))
        if cached is not None:
            dispatcher.discard_resources((cached[1],))
        if isinstance(key, PublicKey):
            self._validator.validate_public_key(key)
        else:
            self._validator.validate_secret_key(key)
        resource = BoundResource(
            symbol,
            kind,
            key,
        )
        dispatcher.operation_key_resources[symbol] = (token, resource)
        return ResourceBindings((resource,))

    def bind_public_key(
        self,
        symbol: str,
        key: PublicKey,
    ) -> ResourceBindings:
        """Bind validated public encryption material to a Program symbol."""

        return self._key_resource(symbol, key)

    def bind_secret_key(
        self,
        symbol: str,
        key: SecretKey,
    ) -> ResourceBindings:
        """Bind validated decryption material to a Program symbol."""

        return self._key_resource(symbol, key)

    def plaintext(
        self,
        message: Sequence[object] | torch.Tensor | complex | float | int,
        *,
        level: int = 0,
        scale: float | None = None,
        device: torch.device | str | None = None,
    ) -> Plaintext:
        """Construct a slots plaintext without coefficient encoding."""

        if type(level) is not int:
            raise TypeError("level must be an integer")
        if not 0 <= level < self.public_level_count:
            raise ValueError(f"level must be in [0, {self.final_public_level}]")
        actual_scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="Plaintext",
        )
        target = torch.get_default_device() if device is None else device
        return Plaintext(
            message=torch.as_tensor(message, device=target).detach().clone(),
            level=level,
            scale=actual_scale,
        )

    def encode(
        self,
        message: Sequence[object] | torch.Tensor | complex | float | int,
        *,
        level: int = 0,
        scale: float | None = None,
        device: torch.device | str | None = None,
    ) -> Plaintext:
        """Encode ordered CKKS slots into integer polynomial coefficients."""

        actual_scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="encode scale",
        )
        if type(level) is not int or not 0 <= level <= self.final_public_level:
            raise ValueError(f"level must be in [0, {self.final_public_level}]")
        target = torch.get_default_device() if device is None else device
        message_tensor = torch.as_tensor(message, device=target)
        coefficients = cast(
            torch.Tensor,
            self._execute(
                ckks.EncodeOp,
                message_tensor,
                attributes={"level": level, "scale": actual_scale},
            ),
        )
        return Plaintext._from_fields(
            message=None,
            level=level,
            scale=actual_scale,
            data=coefficients,
            representation="integer_coefficients",
            polynomial_domain="coefficient",
            modulus_basis=None,
            residue_representation=None,
            prime_ids=(),
        )

    def integer_coefficients_to_rns(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: Literal["Q", "QP"] = "Q",
    ) -> Plaintext:
        """Reduce integer coefficients into coefficient-domain standard RNS."""

        if plaintext.is_slots:
            if plaintext.message is None:
                raise ValueError("slots plaintext has no message Tensor")
            plaintext = self.encode(
                plaintext.message,
                level=plaintext.level,
                scale=plaintext.scale,
                device=plaintext.message.device,
            )
        if not plaintext.is_integer_coefficients or plaintext.data is None:
            raise ValueError(
                "operation requires integer_coefficients plaintext"
            )
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.IntegerCoefficientsToRnsOp,
                plaintext.data,
                attributes={
                    "level": plaintext.level,
                    "modulus_basis": modulus_basis,
                },
            ),
        )
        return Plaintext._from_fields(
            message=None,
            level=plaintext.level,
            scale=plaintext.scale,
            data=data,
            representation="rns",
            polynomial_domain="coefficient",
            modulus_basis=modulus_basis,
            residue_representation="standard",
            prime_ids=self._rns_layout.prime_ids(
                plaintext.level,
                include_p=modulus_basis == "QP",
            ),
        )

    def prepare_plaintext_for_addition(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: Literal["Q", "QP"] = "Q",
    ) -> Plaintext:
        """Return coefficient-domain Montgomery RNS for plaintext addition."""

        standard = self.integer_coefficients_to_rns(
            plaintext,
            modulus_basis=modulus_basis,
        )
        return self.standard_residues_to_montgomery_residues(standard)

    def prepare_plaintext_for_multiplication(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: Literal["Q", "QP"] = "Q",
    ) -> Plaintext:
        """Return NTT/Montgomery RNS for ciphertext multiplication."""

        addition_ready = self.prepare_plaintext_for_addition(
            plaintext,
            modulus_basis=modulus_basis,
        )
        return cast(
            Plaintext,
            self.coefficient_domain_to_ntt_domain(addition_ready),
        )

    def prepare_public_operand(
        self,
        public: object,
        ciphertext: Ciphertext,
        *,
        operation: Literal["add", "multiply"],
        source_role: Literal["message", "plaintext", "static"],
        scale_mode: Literal[
            "runtime_plaintext_scale",
            "default_scale",
            "ciphertext_scale",
        ],
    ) -> Plaintext:
        """Prepare one typed public value for an eager ciphertext operation."""

        if type(ciphertext) is not Ciphertext:
            raise TypeError("prepare_public_operand requires Ciphertext")
        if operation not in {"add", "multiply"}:
            raise ValueError("Public operand operation is unsupported")
        if source_role not in {"message", "plaintext", "static"}:
            raise ValueError("Public operand source_role is unsupported")
        if scale_mode not in {
            "runtime_plaintext_scale",
            "default_scale",
            "ciphertext_scale",
        }:
            raise ValueError("Public operand scale_mode is unsupported")
        if source_role == "plaintext" and not isinstance(public, Plaintext):
            raise TypeError("plaintext source_role requires a Plaintext value")
        if source_role != "plaintext" and isinstance(public, Plaintext):
            raise TypeError(
                f"{source_role} source_role does not accept a Plaintext value"
            )
        if scale_mode == "runtime_plaintext_scale":
            if not isinstance(public, Plaintext):
                raise TypeError(
                    "runtime_plaintext_scale requires a Plaintext value"
                )
            scale = public.scale
        elif scale_mode == "default_scale":
            scale = self.config.default_scale
        else:
            scale = ciphertext.scale
        return self._prepare_public_value(
            public,
            ciphertext,
            operation,
            float(scale),
        )

    def _prepare_public_value(
        self,
        public: object,
        ciphertext: Ciphertext,
        operation: str,
        scale: float,
    ) -> Plaintext:
        """Execute the public-value preparation selected by a value-operation plan."""

        if operation not in {"add", "multiply"}:
            raise ValueError(
                "Public-value preparation operation is unsupported"
            )
        if isinstance(public, Plaintext):
            if public.is_approximate_coefficients:
                raise ValueError(
                    "approximate_coefficients Plaintext is decode-only"
                )
            if public.level != ciphertext.level:
                raise ValueError("Plaintext and ciphertext levels differ")
            if operation == "add" and public.scale != ciphertext.scale:
                raise ValueError("Plaintext addition requires equal scales")
            if public.is_slots:
                assert public.message is not None
                prepared_source = self.encode(
                    public.message,
                    level=public.level,
                    scale=public.scale,
                    device=ciphertext.device,
                )
            elif public.is_integer_coefficients:
                prepared_source = public
            else:
                assert public.is_rns
                if public.modulus_basis != ciphertext.modulus_basis:
                    raise ValueError("Plaintext and ciphertext bases differ")
                if public.prime_ids != ciphertext.prime_ids:
                    raise ValueError(
                        "Plaintext and ciphertext prime IDs differ"
                    )
                prepared = public
                if operation == "add" and prepared.polynomial_domain == "ntt":
                    converted = self.ntt_domain_to_coefficient_domain(prepared)
                    prepared = cast(Plaintext, converted)
                if prepared.residue_representation == "standard":
                    prepared = self.standard_residues_to_montgomery_residues(
                        prepared
                    )
                if (
                    operation == "multiply"
                    and prepared.polynomial_domain == "coefficient"
                ):
                    converted = self.coefficient_domain_to_ntt_domain(prepared)
                    prepared = cast(Plaintext, converted)
                return prepared
        else:
            if not isinstance(
                public,
                (Sequence, torch.Tensor, complex, float, int),
            ):
                raise TypeError(
                    "Public CKKS operand is not an encodable message"
                )
            prepared_source = self.encode(
                public,
                level=ciphertext.level,
                scale=scale,
                device=ciphertext.device,
            )
        prepare = (
            self.prepare_plaintext_for_addition
            if operation == "add"
            else self.prepare_plaintext_for_multiplication
        )
        return prepare(
            prepared_source,
            modulus_basis=cast(Literal["Q", "QP"], ciphertext.modulus_basis),
        )

    def decode(
        self,
        plaintext: Plaintext,
        *,
        is_real: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decode coefficient plaintext data on one caller-selected device."""

        if device is not None:
            plaintext = plaintext.to(device)
        target = plaintext.device
        if plaintext.is_slots:
            if plaintext.message is None:
                raise ValueError("slots plaintext has no message Tensor")
            plaintext = self.encode(
                plaintext.message,
                level=plaintext.level,
                scale=plaintext.scale,
                device=target,
            )
        if (
            plaintext.representation
            not in {
                "integer_coefficients",
                "approximate_coefficients",
            }
            or plaintext.data is None
        ):
            raise ValueError("decode requires coefficient plaintext data")
        return cast(
            torch.Tensor,
            self._execute(
                ckks.DecodeOp,
                plaintext.data,
                attributes={
                    "scale": plaintext.scale,
                    "is_real": int(is_real),
                },
            ),
        )

    def encrypt(
        self,
        plaintext: Plaintext,
        public_key: PublicKey | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Ciphertext:
        """Encrypt slots or integer coefficients under a public key."""

        if device is not None:
            plaintext = plaintext.to(device)
        target = plaintext.device
        key = self.public_key if public_key is None else public_key
        key = cast(
            PublicKey,
            self._key_on_device(
                key,
                target,
                operation_name="encrypt",
            ),
        )
        symbol = "public-key"
        if plaintext.is_slots:
            if plaintext.message is None:
                raise ValueError("slots plaintext has no message Tensor")
            plaintext = self.encode(
                plaintext.message,
                level=plaintext.level,
                scale=plaintext.scale,
                device=target,
            )
        if not plaintext.is_integer_coefficients or plaintext.data is None:
            raise ValueError("encrypt requires integer_coefficients plaintext")
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.EncryptOp,
                plaintext.data,
                attributes={
                    "key_symbol": symbol,
                    "level": plaintext.level,
                },
                bindings=self._key_resource(symbol, key),
            ),
        )
        basis = key.modulus_basis
        return Ciphertext._from_fields(
            data=data,
            level=plaintext.level,
            scale=plaintext.scale,
            prime_ids=self._dispatcher_for(
                target
            ).rns_context.rns_layout.prime_ids(
                plaintext.level,
                include_p=basis == "QP",
            ),
            polynomial_domain="coefficient",
            modulus_basis=basis,
            residue_representation="standard",
        )

    def decrypt(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Plaintext:
        """Decrypt a ciphertext to bounded coefficient plaintext data."""

        if device is not None:
            ciphertext = ciphertext.to(device)
        target = ciphertext.device
        key = self.secret_key if secret_key is None else secret_key
        key = cast(
            SecretKey,
            self._key_on_device(
                key,
                target,
                operation_name="decrypt",
            ),
        )
        self._validator.validate_ciphertext(ciphertext)
        symbol = "secret-key"
        coefficients = cast(
            torch.Tensor,
            self._execute(
                ckks.DecryptOp,
                ciphertext.data,
                attributes={
                    "key_symbol": symbol,
                    "level": ciphertext.level,
                    "modulus_basis": ciphertext.modulus_basis,
                },
                bindings=self._key_resource(symbol, key),
            ),
        )
        return Plaintext._from_fields(
            message=None,
            level=ciphertext.level,
            scale=ciphertext.scale,
            data=coefficients,
            representation="approximate_coefficients",
            polynomial_domain="coefficient",
            modulus_basis=None,
            residue_representation=None,
            prime_ids=(),
        )

    def encrypt_message(
        self,
        message: Sequence[object] | torch.Tensor | complex | float | int,
        public_key: PublicKey | None = None,
        *,
        level: int = 0,
        scale: float | None = None,
        device: torch.device | str | None = None,
    ) -> Ciphertext:
        """Encode and encrypt one ordered CKKS slot message."""

        return self.encrypt(
            self.encode(message, level=level, scale=scale, device=device),
            public_key,
            device=device,
        )

    def decrypt_message(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey | None = None,
        *,
        is_real: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decrypt and decode on the ciphertext or selected boundary device."""

        return self.decode(
            self.decrypt(ciphertext, secret_key, device=device),
            is_real=is_real,
            device=device,
        )

    def zero_plaintext_like(self, plaintext: Plaintext) -> Plaintext:
        """Construct semantic zero with the same plaintext state and shape."""

        result = plaintext.clone()
        if result.message is not None:
            result.message.zero_()
        elif result.data is not None:
            result.data.zero_()
        else:
            raise ValueError("Plaintext has no materialized payload")
        return result

    def encrypt_zero_like(
        self,
        ciphertext: Ciphertext,
        public_key: PublicKey | None = None,
    ) -> Ciphertext:
        """Return a randomized encryption of zero in `ciphertext` state."""

        self._validator.validate_ciphertext(ciphertext)
        zero = torch.zeros(
            (*ciphertext.batch_shape, self.num_slots),
            dtype=torch.complex128,
            device=ciphertext.device,
        )
        result = self.encrypt_message(
            zero,
            public_key,
            level=ciphertext.level,
            scale=ciphertext.scale,
            device=ciphertext.device,
        )
        if result.modulus_basis != ciphertext.modulus_basis:
            raise ValueError(
                "Public-key basis differs from the reference ciphertext basis"
            )
        if ciphertext.polynomial_domain == "ntt":
            result = cast(
                Ciphertext,
                self.coefficient_domain_to_ntt_domain(result),
            )
        return result

    def install_evaluation_key(self, symbol: str, key: KeySwitchKey) -> None:
        """Install one validated key under an evaluation-resource symbol."""

        previous = self._installed_evaluation_keys.get(symbol)
        if previous is not None and previous is not key:
            self._discard_key_replicas(previous)
        self._bind_key(symbol, key)
        if isinstance(key, RelinearizationKey):
            self._keys.set_relinearization_key(key)
        elif isinstance(key, ConjugationKey):
            self._keys.set_conjugation_key(key)
        elif isinstance(key, RotationKey):
            self._keys.set_rotation_key(key, step=key.rotation_step)
        self._installed_evaluation_keys[symbol] = key

    def remove_evaluation_key(self, symbol: str) -> None:
        """Remove one installed evaluation-key binding if present."""

        removed = self._installed_evaluation_keys.pop(symbol, None)
        if removed is not None:
            self._discard_key_replicas(removed)
        for dispatcher in self._dispatchers.values():
            cached_resource = dispatcher.evaluation_key_resources.pop(
                symbol, None
            )
            if cached_resource is not None:
                dispatcher.discard_resources((cached_resource[1],))
        if symbol == "relinearization-key" and isinstance(
            removed,
            RelinearizationKey,
        ):
            if self._keys.relinearization_key is removed:
                self._keys.set_relinearization_key(None)
        elif symbol == "conjugation-key" and isinstance(
            removed,
            ConjugationKey,
        ):
            if self._keys.conjugation_key is removed:
                self._keys.set_conjugation_key(None)
        elif isinstance(removed, RotationKey) and symbol == (
            f"rotation-key:{removed.rotation_step}"
        ):
            if self._keys.rotation_key(removed.rotation_step) is removed:
                self._keys.set_rotation_key(
                    None,
                    step=removed.rotation_step,
                )

    @property
    def secret_key(self) -> SecretKey:
        """Return the installed secret key, generating one when permitted."""

        key = self._keys.secret_key
        if key is None:
            if not self.allow_automatic_key_generation:
                raise RuntimeError("Secret-key generation is disabled")
            key = self.create_secret_key()
            self.set_secret_key(key)
        return key

    def set_secret_key(self, key: SecretKey | None) -> None:
        """Install or remove the secret key and invalidate dependent keys."""

        if key is self._keys.secret_key:
            return
        self._installed_evaluation_keys.clear()
        for dispatcher in self._dispatchers.values():
            replaced_resources = tuple(
                resource
                for _, resource in dispatcher.evaluation_key_resources.values()
            ) + tuple(
                resource
                for _, resource in dispatcher.operation_key_resources.values()
            )
            dispatcher.evaluation_key_resources.clear()
            dispatcher.operation_key_resources.clear()
            dispatcher.discard_resources(replaced_resources)
        self._key_replicas.clear()
        self._keys.set_secret_key(key)

    @property
    def public_key(self) -> PublicKey:
        """Return the installed public key, generating one when absent."""

        key = self._keys.public_key
        if key is None:
            key = self.create_public_key(self.secret_key)
            self._keys.set_public_key(key)
        return key

    def set_public_key(self, key: PublicKey | None) -> None:
        """Install or remove the public encryption key."""

        for dispatcher in self._dispatchers.values():
            replaced_resources = tuple(
                resource
                for _, resource in dispatcher.operation_key_resources.values()
            )
            dispatcher.operation_key_resources.clear()
            dispatcher.discard_resources(replaced_resources)
        self._key_replicas.clear()
        self._keys.set_public_key(key)

    @property
    def relinearization_key(self) -> RelinearizationKey:
        """Return installed relinearization material, generating it when absent."""

        key = self._keys.relinearization_key
        if key is None:
            if not self.allow_automatic_key_generation:
                raise KeyError("No relinearization key is installed")
            key = self.create_relinearization_key(self.secret_key)
            self.set_relinearization_key(key)
        return key

    def set_relinearization_key(
        self,
        key: RelinearizationKey | None,
    ) -> None:
        """Install or remove relinearization material."""

        symbol = "relinearization-key"
        if key is None:
            self.remove_evaluation_key(symbol)
            self._keys.set_relinearization_key(None)
            return
        self.install_evaluation_key(symbol, key)

    @property
    def rotation_keys(self) -> dict[int, RotationKey]:
        """Return the normalized rotation-key map."""

        return self._keys.rotation_keys

    def rotation_key(self, step: int) -> RotationKey:
        """Return rotation material for `step`, generating it when permitted."""

        key = self._keys.rotation_key(step)
        if key is None:
            if not self.allow_automatic_key_generation:
                raise KeyError(f"No rotation key is installed for step {step}")
            key = self.create_rotation_key(step, self.secret_key)
            self.set_rotation_key(key)
        return key

    def set_rotation_key(self, key: RotationKey) -> None:
        """Install a key under its normalized self-described rotation step."""

        self.install_evaluation_key(
            f"rotation-key:{key.rotation_step}",
            key,
        )

    @property
    def conjugation_key(self) -> ConjugationKey | None:
        """Return installed conjugation material, if any."""

        return self._keys.conjugation_key

    def set_conjugation_key(self, key: ConjugationKey | None) -> None:
        """Install or remove conjugation material."""

        symbol = "conjugation-key"
        if key is None:
            self.remove_evaluation_key(symbol)
            self._keys.set_conjugation_key(None)
            return
        self.install_evaluation_key(symbol, key)

    def validate_ciphertext(self, value: Ciphertext) -> None:
        """Validate a ciphertext against this runtime's complete RNS layout."""

        self._validator.validate_ciphertext(value)

    def validate_public_key(self, key: PublicKey) -> None:
        """Validate public encryption material against this runtime."""

        self._validator.validate_public_key(key)

    def validate_secret_key(self, key: SecretKey) -> None:
        """Validate secret-key storage and context against this runtime."""

        self._validator.validate_secret_key(key)

    def validate_key_switch_key(self, key: KeySwitchKey) -> None:
        """Validate key-switch material against this runtime."""

        self._validator.validate_key_switch_key(key)

    def reconstruct_tail_q_coefficients(
        self,
        residues: torch.Tensor,
        ciphertext: Ciphertext,
    ) -> torch.Tensor:
        """Reconstruct bounded coefficients from a ciphertext phase tensor.

        `residues` must have the ciphertext's batch, Q-row, coefficient,
        dtype, and device layout in coefficient-domain standard form. The
        returned binary64 tensor is suitable for CKKS decoding but is not a
        full-Q Chinese-remainder reconstruction.
        """

        self.validate_ciphertext(ciphertext)
        ciphertext.assert_state(
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="standard",
            components=2,
        )
        if not isinstance(residues, torch.Tensor):
            raise TypeError("residues must be a torch.Tensor")
        if tuple(residues.shape) != tuple(ciphertext.c0.shape):
            raise ValueError(
                "residue shape differs from the ciphertext component: "
                f"{tuple(residues.shape)} != {tuple(ciphertext.c0.shape)}"
            )
        if residues.dtype != ciphertext.data.dtype:
            raise TypeError("residue dtype differs from the ciphertext")
        if residues.device != ciphertext.data.device:
            raise ValueError("residue device differs from the ciphertext")
        dispatcher = self._dispatcher_for(residues.device)
        return reconstruct_tail_q_coefficients_tensor(
            residues,
            level=ciphertext.level,
            includes_p=ciphertext.includes_p,
            rns_context=dispatcher.rns_context,
            reconstruction=dispatcher.decrypt_reconstruction(),
        )

    def _bind_key(self, symbol: str, key: KeySwitchKey) -> BoundResource:
        dispatcher = self._dispatcher_for(key.device)
        token = id(key)
        cached = dispatcher.evaluation_key_resources.get(symbol)
        if cached is not None and cached[0] == token:
            return cached[1]
        if cached is not None:
            dispatcher.discard_resources((cached[1],))
        self._validator.validate_key_switch_key(key)
        kind = ckks_key_resource_kind(key)
        resource = BoundResource(
            symbol,
            kind,
            key,
        )
        dispatcher.evaluation_key_resources[symbol] = (token, resource)
        return resource

    def _key_bindings(
        self,
        keys: Mapping[str, KeySwitchKey],
        *,
        device: torch.device | None = None,
        operation_name: str = "bind_evaluation_keys",
    ) -> ResourceBindings:
        selected = {
            symbol: cast(
                KeySwitchKey,
                self._key_on_device(
                    key,
                    key.device if device is None else device,
                    operation_name=operation_name,
                ),
            )
            for symbol, key in keys.items()
        }
        return ResourceBindings(
            tuple(
                self._bind_key(symbol, key) for symbol, key in selected.items()
            )
        )

    def bind_evaluation_keys(
        self,
        keys: Mapping[str, KeySwitchKey],
    ) -> ResourceBindings:
        """Return adapter-validated bindings for concrete semantic key slots."""

        return self._key_bindings(keys)

    @staticmethod
    def _replace_plaintext_(target: Plaintext, source: Plaintext) -> Plaintext:
        target.message = source.message
        target.level = source.level
        target.scale = source.scale
        target.data = source.data
        target.representation = source.representation
        target.polynomial_domain = source.polynomial_domain
        target.modulus_basis = source.modulus_basis
        target.residue_representation = source.residue_representation
        target.prime_ids = source.prime_ids
        return target

    def _replace_value_(
        self,
        target: Ciphertext | Plaintext,
        source: Ciphertext | Plaintext,
    ) -> Ciphertext | Plaintext:
        if isinstance(target, Ciphertext) and isinstance(source, Ciphertext):
            return target.replace_(source)
        if isinstance(target, Plaintext) and isinstance(source, Plaintext):
            return self._replace_plaintext_(target, source)
        raise TypeError("CKKS in-place result kind differs from its input")

    @staticmethod
    def _operand_bases(
        *values: Ciphertext | Plaintext,
    ) -> tuple[str | None, ...]:
        return tuple(value.modulus_basis for value in values)

    def add(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Add ciphertext residues and preserve their eager metadata."""

        bases = self._operand_bases(lhs, rhs)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.AddStandardOp,
                lhs.data,
                rhs.data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            return lhs
        return _ciphertext_result(lhs, data)

    def add_(self, lhs: Ciphertext, rhs: Ciphertext) -> Ciphertext:
        """Add `rhs` into `lhs` and return `lhs`."""

        return self.add(lhs, rhs, inplace=True)

    def add_scalar(
        self,
        ciphertext: Ciphertext,
        scalar: int | float,
        *,
        scalar_scale: float | None = None,
    ) -> Ciphertext:
        r"""Add a real scalar quantized at `scalar_scale` to component zero.

        `scalar_scale` selects the integer coefficient used for the scalar;
        omitting it uses the ciphertext's actual scale. The ciphertext scale,
        level, and representation metadata are preserved. The ciphertext must
        use coefficient-domain standard residues.
        """

        if (
            ciphertext.polynomial_domain != "coefficient"
            or ciphertext.residue_representation != "standard"
        ):
            raise ValueError(
                "add_scalar requires coefficient-domain standard residues"
            )
        actual_scalar_scale = coerce_scale(
            ciphertext.scale if scalar_scale is None else scalar_scale,
            value_name="add_scalar scalar scale",
        )
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.AddScalarOp,
                ciphertext.data,
                attributes={
                    "scalar": float(scalar),
                    "scalar_scale": actual_scalar_scale,
                },
                bases=self._operand_bases(ciphertext),
            ),
        )
        return _ciphertext_result(ciphertext, data)

    def sum_ciphertexts(
        self,
        ciphertexts: Sequence[Ciphertext],
    ) -> Ciphertext:
        """Return the sum of a non-empty compatible ciphertext sequence."""

        if not ciphertexts:
            raise ValueError("sum_ciphertexts requires at least one value")
        if len(ciphertexts) == 1:
            value = ciphertexts[0]
            return _ciphertext_result(value, value.data.clone())
        first = ciphertexts[0]
        result = _ciphertext_result(first, first.data.clone())
        for ciphertext in ciphertexts[1:]:
            self.add_(result, ciphertext)
        return result

    def sum_ciphertext_batch(
        self,
        batch: Ciphertext,
        *,
        dim: int = 0,
    ) -> Ciphertext:
        """Reduce one logical ciphertext batch axis by modular addition."""

        if not batch.is_batched:
            raise ValueError("sum_ciphertext_batch requires a batched value")
        logical_dim = dim if dim >= 0 else dim + len(batch.batch_shape)
        if not 0 <= logical_dim < len(batch.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape {batch.batch_shape}"
            )
        current = batch
        odd_tails: list[Ciphertext] = []
        while current.batch_shape[logical_dim] > 1:
            count = current.batch_shape[logical_dim]
            pair_count = count // 2
            data_dim = logical_dim + 1
            if count % 2:
                odd_tails.append(
                    _ciphertext_result(
                        current,
                        current.data.select(data_dim, count - 1),
                    )
                )
            lhs = _ciphertext_result(
                current,
                current.data.narrow(data_dim, 0, pair_count),
            )
            rhs = _ciphertext_result(
                current,
                current.data.narrow(data_dim, pair_count, pair_count),
            )
            current = self.add(lhs, rhs)
        result = _ciphertext_result(
            current,
            current.data.select(logical_dim + 1, 0),
        )
        if current is batch:
            result = _ciphertext_result(result, result.data.clone())
        for tail in reversed(odd_tails):
            result = self.add(result, tail)
        return result

    def subtract(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Subtract ciphertext residues and preserve their eager metadata."""

        bases = self._operand_bases(lhs, rhs)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.SubtractStandardOp,
                lhs.data,
                rhs.data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            return lhs
        return _ciphertext_result(lhs, data)

    def subtract_(self, lhs: Ciphertext, rhs: Ciphertext) -> Ciphertext:
        """Subtract `rhs` from `lhs` and return `lhs`."""

        return self.subtract(lhs, rhs, inplace=True)

    def negate(
        self,
        value: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Negate ciphertext residues and preserve their eager metadata."""

        bases = self._operand_bases(value)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.NegateStandardOp,
                value.data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            return value
        return _ciphertext_result(value, data)

    def negate_(self, value: Ciphertext) -> Ciphertext:
        """Negate `value` and return the same ciphertext object."""

        return self.negate(value, inplace=True)

    @overload
    def coefficient_domain_to_ntt_domain(
        self,
        value: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext: ...

    @overload
    def coefficient_domain_to_ntt_domain(
        self,
        value: Plaintext,
        *,
        inplace: bool = False,
    ) -> Plaintext: ...

    def coefficient_domain_to_ntt_domain(
        self,
        value: Ciphertext | Plaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext | Plaintext:
        """Return a coefficient-domain RNS value in NTT/Montgomery form."""

        if value.data is None:
            raise ValueError("NTT execution requires RNS Tensor data")
        is_ciphertext = type(value) is Ciphertext
        operation_type: type[Operation] = (
            ntt.CoefficientStandardToNttMontgomeryOp
            if is_ciphertext
            else ntt.CoefficientMontgomeryToNttMontgomeryOp
        )
        value_data = cast(torch.Tensor, value.data)
        bases = self._operand_bases(value)
        if is_ciphertext:
            data = value_data if inplace else value_data.clone()
            for component in data.unbind(0):
                self._execute(
                    operation_type,
                    component,
                    resource_kinds=("ntt",),
                    bases=bases,
                    in_place=True,
                )
        else:
            data = cast(
                torch.Tensor,
                self._execute(
                    operation_type,
                    value_data,
                    resource_kinds=("ntt",),
                    bases=bases,
                    in_place=inplace,
                ),
            )
        if inplace:
            value.data = data
            value.polynomial_domain = "ntt"
            value.residue_representation = "montgomery"
            return value
        result: Ciphertext | Plaintext = (
            _ciphertext_result(
                cast(Ciphertext, value),
                data,
                polynomial_domain="ntt",
                residue_representation="montgomery",
            )
            if is_ciphertext
            else _plaintext_result(
                cast(Plaintext, value),
                data,
                polynomial_domain="ntt",
                residue_representation="montgomery",
            )
        )
        return result

    @overload
    def coefficient_domain_to_ntt_domain_(
        self,
        value: Ciphertext,
    ) -> Ciphertext: ...

    @overload
    def coefficient_domain_to_ntt_domain_(
        self,
        value: Plaintext,
    ) -> Plaintext: ...

    def coefficient_domain_to_ntt_domain_(
        self,
        value: Ciphertext | Plaintext,
    ) -> Ciphertext | Plaintext:
        """Transform `value` to NTT/Montgomery form and return `value`."""

        return self.coefficient_domain_to_ntt_domain(value, inplace=True)

    @overload
    def ntt_domain_to_coefficient_domain(
        self,
        value: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext: ...

    @overload
    def ntt_domain_to_coefficient_domain(
        self,
        value: Plaintext,
        *,
        inplace: bool = False,
    ) -> Plaintext: ...

    def ntt_domain_to_coefficient_domain(
        self,
        value: Ciphertext | Plaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext | Plaintext:
        """Return an NTT-domain RNS value in coefficient form."""

        if value.data is None:
            raise ValueError("NTT execution requires RNS Tensor data")
        is_ciphertext = type(value) is Ciphertext
        result_representation = "standard" if is_ciphertext else "montgomery"
        operation_type = (
            ntt.NttMontgomeryToCoefficientStandardOp
            if is_ciphertext
            else ntt.InverseMontgomeryOp
        )
        value_data = cast(torch.Tensor, value.data)
        bases = self._operand_bases(value)
        if is_ciphertext:
            data = value_data if inplace else value_data.clone()
            for component in data.unbind(0):
                self._execute(
                    operation_type,
                    component,
                    resource_kinds=("ntt",),
                    bases=bases,
                    in_place=True,
                )
        else:
            data = cast(
                torch.Tensor,
                self._execute(
                    operation_type,
                    value_data,
                    resource_kinds=("ntt",),
                    bases=bases,
                    in_place=inplace,
                ),
            )
        if inplace:
            value.data = data
            value.polynomial_domain = "coefficient"
            value.residue_representation = result_representation
            return value
        result = (
            _ciphertext_result(
                cast(Ciphertext, value),
                data,
                polynomial_domain="coefficient",
                residue_representation="standard",
            )
            if is_ciphertext
            else _plaintext_result(
                cast(Plaintext, value),
                data,
                polynomial_domain="coefficient",
                residue_representation=result_representation,
            )
        )
        return result

    @overload
    def ntt_domain_to_coefficient_domain_(
        self,
        value: Ciphertext,
    ) -> Ciphertext: ...

    @overload
    def ntt_domain_to_coefficient_domain_(
        self,
        value: Plaintext,
    ) -> Plaintext: ...

    def ntt_domain_to_coefficient_domain_(
        self,
        value: Ciphertext | Plaintext,
    ) -> Ciphertext | Plaintext:
        """Transform `value` to coefficient form and return `value`."""

        return self.ntt_domain_to_coefficient_domain(value, inplace=True)

    def standard_residues_to_montgomery_residues(
        self,
        plaintext: Plaintext,
        *,
        inplace: bool = False,
    ) -> Plaintext:
        """Return coefficient-domain plaintext residues in Montgomery form."""

        if plaintext.data is None:
            raise ValueError("Residue conversion requires RNS Tensor data")
        plaintext_data = cast(torch.Tensor, plaintext.data)
        bases = self._operand_bases(plaintext)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.StandardToMontgomeryOp,
                plaintext_data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            plaintext.data = data
            plaintext.residue_representation = "montgomery"
            return plaintext
        return _plaintext_result(
            plaintext,
            data,
            residue_representation="montgomery",
        )

    def standard_residues_to_montgomery_residues_(
        self,
        plaintext: Plaintext,
    ) -> Plaintext:
        """Convert `plaintext` to Montgomery residues and return it."""

        return self.standard_residues_to_montgomery_residues(
            plaintext,
            inplace=True,
        )

    def montgomery_residues_to_standard_residues(
        self,
        plaintext: Plaintext,
        *,
        inplace: bool = False,
    ) -> Plaintext:
        """Return coefficient-domain plaintext residues in standard form."""

        if plaintext.data is None:
            raise ValueError("Residue conversion requires RNS Tensor data")
        plaintext_data = cast(torch.Tensor, plaintext.data)
        bases = self._operand_bases(plaintext)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.MontgomeryToStandardOp,
                plaintext_data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            plaintext.data = data
            plaintext.residue_representation = "standard"
            return plaintext
        return _plaintext_result(
            plaintext,
            data,
            residue_representation="standard",
        )

    def montgomery_residues_to_standard_residues_(
        self,
        plaintext: Plaintext,
    ) -> Plaintext:
        """Convert `plaintext` to standard residues and return it."""

        return self.montgomery_residues_to_standard_residues(
            plaintext,
            inplace=True,
        )

    def rescale_to_next_drop_prime(self, *, level: int) -> int:
        """Return the Q prime removed by rescale at `level`."""

        if type(level) is not int:
            raise TypeError("level must be an integer")
        if level < 0:
            raise ValueError("level must be non-negative")
        final_drop_level = self.config.num_q_primes - 2
        if level > final_drop_level:
            raise MaximumLevelError(
                level=level,
                maximum_level=final_drop_level,
            )
        prime_id = self._rns_layout.prime_ids(level)[0]
        return int(self.config.moduli[prime_id])

    def rescale_to_next_output_scale(
        self,
        input_scale: float,
        *,
        level: int,
    ) -> float:
        """Return the scale produced by one rescale transition."""

        scale = coerce_scale(input_scale, value_name="input_scale")
        return coerce_scale(
            scale / self.rescale_to_next_drop_prime(level=level),
            value_name="rescale output",
        )

    def rescale_to_next_level(
        self,
        value: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
        inplace: bool = False,
    ) -> Ciphertext:
        """Drop the leading Q prime and return the next ciphertext level."""

        if rounding not in {"nearest", "floor"}:
            raise ValueError("rounding must be 'nearest' or 'floor'")
        if value.data.size(-2) <= 1:
            raise MaximumLevelError(
                level=value.level,
                maximum_level=self.final_public_level,
            )
        dropped_prime = self.config.moduli[value.prime_ids[0]]
        bases = self._operand_bases(value)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.RescaleDropLeadingPrimeOp,
                value.data,
                resource_kinds=("rescale",),
                attributes={
                    "rounding": (
                        "truncate" if rounding == "floor" else rounding
                    ),
                },
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            value.data = data
            value.level += 1
            value.prime_ids = value.prime_ids[1:]
            value.scale /= float(dropped_prime)
            return value
        return _ciphertext_result(
            value,
            data,
            level=value.level + 1,
            prime_ids=value.prime_ids[1:],
            scale=value.scale / float(dropped_prime),
        )

    def rescale_to_next_level_(
        self,
        value: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        """Rescale `value` to its next level and return `value`."""

        return self.rescale_to_next_level(
            value,
            rounding=rounding,
            inplace=True,
        )

    def rescale_to_structural_base(
        self,
        value: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        """Drop the last public Q prime for bootstrap modulus raising.

        The input must be a final-public-level coefficient-domain standard-Q
        ciphertext. The result uses private level `public_level_count` and the
        one-prime structural Q basis provided by the configured RNS layout.
        This transition uses the registered rescale lowering and native
        implementation; ordinary :meth:`rescale_to_next_level` continues to
        reject a transition beyond the public level interval.
        """

        if rounding not in {"nearest", "floor"}:
            raise ValueError("rounding must be 'nearest' or 'floor'")
        if value.data.size(-2) <= 1:
            raise ValueError("structural-base rescale requires another RNS row")
        dropped_prime = self.config.moduli[value.prime_ids[0]]
        bases = self._operand_bases(value)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.RescaleDropLeadingPrimeOp,
                value.data,
                resource_kinds=("rescale",),
                attributes={
                    "rounding": (
                        "truncate" if rounding == "floor" else rounding
                    )
                },
                bases=bases,
            ),
        )
        return _ciphertext_result(
            value,
            data,
            level=self.public_level_count,
            prime_ids=value.prime_ids[1:],
            scale=value.scale / float(dropped_prime),
        )

    def mod_switch_to_level(
        self,
        value: Ciphertext,
        target_level: int,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Restrict a ciphertext to the RNS rows of `target_level`."""

        prime_ids = self._rns_layout.prime_ids(
            target_level,
            include_p=value.modulus_basis == "QP",
        )
        result_rows = len(prime_ids)
        source_rows = value.data.size(-2)
        if not 0 < result_rows <= source_rows:
            raise ValueError("target_level requires unavailable RNS rows")
        selected = value.data.narrow(
            -2,
            source_rows - result_rows,
            result_rows,
        )
        data = selected if inplace else selected.clone()
        if inplace:
            value.data = data
            value.level = target_level
            value.prime_ids = prime_ids
            return value
        return _ciphertext_result(
            value,
            data,
            level=target_level,
            prime_ids=prime_ids,
        )

    def mod_switch_to_next_level(
        self,
        value: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Restrict a ciphertext to the next public RNS level."""

        return self.mod_switch_to_level(
            value,
            value.level + 1,
            inplace=inplace,
        )

    def mod_switch_to_next_level_(self, value: Ciphertext) -> Ciphertext:
        """Restrict `value` to the next level and return `value`."""

        return self.mod_switch_to_next_level(value, inplace=True)

    def mod_switch_to_level_(
        self,
        value: Ciphertext,
        target_level: int,
    ) -> Ciphertext:
        """Restrict `value` to `target_level` and return `value`."""

        return self.mod_switch_to_level(value, target_level, inplace=True)

    def reinterpret_at_scale(
        self,
        value: Ciphertext,
        target_scale: float,
        *,
        max_relative_change: float | None = None,
        inplace: bool = False,
    ) -> Ciphertext:
        """Return cloned residues with replacement scale metadata."""

        scale = coerce_scale(target_scale, value_name="target_scale")
        if max_relative_change is not None:
            relative_limit = float(max_relative_change)
            if relative_limit < 0:
                raise ValueError("max_relative_change must be non-negative")
            change = max(value.scale / scale, scale / value.scale) - 1.0
            if change > relative_limit:
                raise ScaleMismatchError(
                    operation="reinterpret_at_scale",
                    lhs_name="current",
                    lhs_scale=value.scale,
                    rhs_name="target",
                    rhs_scale=scale,
                )
        if inplace:
            value.scale = scale
            return value
        return _ciphertext_result(value, value.data.clone(), scale=scale)

    def reinterpret_at_scale_(
        self,
        value: Ciphertext,
        target_scale: float,
        *,
        max_relative_change: float | None = None,
    ) -> Ciphertext:
        """Change `value` scale metadata and return `value`."""

        return self.reinterpret_at_scale(
            value,
            target_scale,
            max_relative_change=max_relative_change,
            inplace=True,
        )

    def multiply(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
    ) -> Ciphertext:
        """Multiply CT2 inputs with the eager convolution implementation."""

        result_scale = lhs.scale * rhs.scale
        bases = self._operand_bases(lhs, rhs)
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.MultiplyOp,
                lhs.data,
                rhs.data,
                bases=bases,
                implementation="native-ct2-convolution",
            ),
        )
        return _ciphertext_result(
            lhs,
            data,
            scale=result_scale,
        )

    def multiply_scalar(
        self,
        ciphertext: Ciphertext,
        scalar: int | float,
        *,
        scalar_scale: float | None = None,
    ) -> Ciphertext:
        r"""Multiply by a real scalar without rescaling the result.

        `scalar_scale` controls scalar quantization and defaults to the input
        ciphertext's actual scale. The output actual scale is the product of
        the ciphertext scale and `scalar_scale`; level and representation are
        preserved.
        """

        actual_scalar_scale = coerce_scale(
            ciphertext.scale if scalar_scale is None else scalar_scale,
            value_name="multiply_scalar scalar scale",
        )
        result_scale = coerce_scale(
            ciphertext.scale * actual_scalar_scale,
            value_name="multiply_scalar result",
        )
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.MultiplyScalarOp,
                ciphertext.data,
                attributes={
                    "scalar": float(scalar),
                    "scalar_scale": actual_scalar_scale,
                },
                bases=self._operand_bases(ciphertext),
            ),
        )
        return _ciphertext_result(ciphertext, data, scale=result_scale)

    def multiply_integer_scalar(
        self,
        ciphertext: Ciphertext,
        scalar: int,
    ) -> Ciphertext:
        """Multiply by an integer without changing scale or level."""

        data = cast(
            torch.Tensor,
            self._execute(
                ckks.MultiplyIntegerScalarOp,
                ciphertext.data,
                attributes={"scalar": integer_index(scalar)},
                bases=self._operand_bases(ciphertext),
            ),
        )
        return _ciphertext_result(ciphertext, data)

    def add_plaintext(
        self,
        ciphertext: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Add a prepared plaintext to ciphertext component zero."""

        if isinstance(plaintext, CompressedPlaintext):
            inputs = [ciphertext.data, plaintext.data]
            if plaintext.implicit_data is not None:
                inputs.append(plaintext.implicit_data)
            data = cast(
                torch.Tensor,
                self._execute(
                    ckks.AddCompressedPlaintextOp,
                    *inputs,
                    attributes={
                        "compression_layout": plaintext.compression_layout
                    },
                    bases=(
                        ciphertext.modulus_basis,
                        plaintext.modulus_basis,
                        *((None,) if len(inputs) == 3 else ()),
                    ),
                ),
            )
            if inplace:
                ciphertext.data = data
                return ciphertext
            return _ciphertext_result(ciphertext, data)
        prepared = plaintext
        if prepared.data is None:
            raise ValueError("Plaintext addition requires RNS Tensor data")
        prepared_data = cast(torch.Tensor, prepared.data)
        bases = self._operand_bases(ciphertext, prepared)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.AddPlaintextOp,
                ciphertext.data,
                prepared_data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            return ciphertext
        return _ciphertext_result(ciphertext, data)

    def add_plaintext_(
        self,
        ciphertext: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
    ) -> Ciphertext:
        """Add `plaintext` into `ciphertext` and return `ciphertext`."""

        return self.add_plaintext(ciphertext, plaintext, inplace=True)

    def multiply_plaintext(
        self,
        ciphertext: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        """Multiply a ciphertext by an NTT/Montgomery plaintext."""

        if isinstance(plaintext, CompressedPlaintext):
            result_scale = ciphertext.scale * plaintext.scale
            data = cast(
                torch.Tensor,
                self._execute(
                    ckks.MultiplyCompressedPlaintextOp,
                    ciphertext.data,
                    plaintext.data,
                    attributes={
                        "compression_layout": plaintext.compression_layout
                    },
                    bases=(
                        ciphertext.modulus_basis,
                        plaintext.modulus_basis,
                    ),
                ),
            )
            if inplace:
                ciphertext.data = data
                ciphertext.scale = result_scale
                return ciphertext
            return _ciphertext_result(
                ciphertext,
                data,
                scale=result_scale,
            )
        prepared = plaintext
        if prepared.data is None:
            raise ValueError(
                "Plaintext multiplication requires RNS Tensor data"
            )
        result_scale = ciphertext.scale * plaintext.scale
        prepared_data = cast(torch.Tensor, prepared.data)
        bases = self._operand_bases(ciphertext, prepared)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.MultiplyPlaintextOp,
                ciphertext.data,
                prepared_data,
                resource_kinds=("rns",),
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            ciphertext.data = data
            ciphertext.scale = result_scale
            return ciphertext
        return _ciphertext_result(
            ciphertext,
            data,
            scale=result_scale,
        )

    def multiply_plaintext_(
        self,
        ciphertext: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
    ) -> Ciphertext:
        """Replace `ciphertext` by its plaintext product and return it."""

        return self.multiply_plaintext(ciphertext, plaintext, inplace=True)

    def _extract_component_tensor(
        self,
        data: torch.Tensor,
        component: int,
        *,
        basis: str,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._execute(
                rns.ExtractComponentOp,
                data,
                attributes={"component": component},
                bases=(basis,),
            ),
        )

    def _key_switch_corrections(
        self,
        source: torch.Tensor,
        *,
        level: int,
        key: KeySwitchKey,
        key_symbol: str,
    ) -> torch.Tensor:
        key = cast(
            KeySwitchKey,
            self._key_on_device(
                key,
                source.device,
                operation_name="key switching",
            ),
        )
        key_resource = self._bind_key(key_symbol, key)
        dispatcher = self._dispatcher_for(source.device)
        rns_resource = dispatcher.resources(
            (("active-rns-parameters", RNS_RESOURCE_KIND),)
        )
        key_switch_resource = dispatcher.resources(
            (
                (
                    "active-key-switch-plan",
                    KEY_SWITCH_PLAN_RESOURCE_KIND,
                ),
            )
        )
        accumulator: torch.Tensor | None = None
        for digit_index, digit_spec in enumerate(
            self._rns_layout.digit_specs(level)
        ):
            lifted = cast(
                torch.Tensor,
                self._execute(
                    rns.HybridModUpDigitOp,
                    source,
                    resource_kinds=("rns", "key_switch"),
                    attributes={"digit_index": digit_index},
                    bases=("Q",),
                ),
            )
            transformed = cast(
                torch.Tensor,
                self._execute(
                    ntt.CoefficientMontgomeryToNttMontgomeryOp,
                    lifted,
                    resource_kinds=("ntt",),
                    bases=("QP",),
                ),
            )
            product = cast(
                torch.Tensor,
                self._execute(
                    rns.KeySwitchDigitProductOp,
                    transformed,
                    resources=(
                        key_resource,
                        *rns_resource,
                        *key_switch_resource,
                    ),
                    attributes={"key_digit_index": digit_spec.key_digit_index},
                    bases=("QP",),
                ),
            )
            if accumulator is None:
                accumulator = product
            else:
                accumulator = cast(
                    torch.Tensor,
                    self._execute(
                        rns.AddMontgomeryLazyOp,
                        accumulator,
                        product,
                        resource_kinds=("rns",),
                        bases=("QP", "QP"),
                    ),
                )
        if accumulator is None:
            raise ValueError("Key switching requires at least one RNS digit")
        coefficient = cast(
            torch.Tensor,
            self._execute(
                ntt.NttMontgomeryToCoefficientStandardOp,
                accumulator,
                resource_kinds=("ntt",),
                bases=("QP",),
            ),
        )
        return cast(
            torch.Tensor,
            self._execute(
                rns.ModDownQpToQOp,
                coefficient,
                resource_kinds=("rns", "key_switch"),
                bases=("QP",),
            ),
        )

    def _assemble_key_switch(
        self,
        component0: torch.Tensor,
        switched_component: torch.Tensor,
        *,
        value: Ciphertext,
        key: KeySwitchKey,
        key_symbol: str,
        component1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        corrections = self._key_switch_corrections(
            switched_component,
            level=value.level,
            key=key,
            key_symbol=key_symbol,
        )
        correction0 = self._extract_component_tensor(
            corrections,
            0,
            basis="Q",
        )
        correction1 = self._extract_component_tensor(
            corrections,
            1,
            basis="Q",
        )
        result0 = cast(
            torch.Tensor,
            self._execute(
                rns.AddStandardOp,
                component0,
                correction0,
                resource_kinds=("rns",),
                bases=("Q", "Q"),
            ),
        )
        if component1 is not None:
            result1 = cast(
                torch.Tensor,
                self._execute(
                    rns.AddStandardOp,
                    component1,
                    correction1,
                    resource_kinds=("rns",),
                    bases=("Q", "Q"),
                ),
            )
        else:
            result1 = correction1
        return cast(
            torch.Tensor,
            self._execute(
                rns.PackTwoComponentsOp,
                result0,
                result1,
                bases=("Q", "Q"),
            ),
        )

    def relinearize(
        self,
        value: Ciphertext,
        key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        """Reduce CT3 to CT2 through registered eager execution."""

        symbol = "relinearization-key"
        selected = key if key is not None else self.relinearization_key
        selected = cast(
            RelinearizationKey,
            self._key_on_device(
                selected,
                value.device,
                operation_name="relinearize",
            ),
        )
        key_resource = self._bind_key(symbol, selected)
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.RelinearizeOp,
                value.data,
                bindings=self._key_switch_bindings(value.device, key_resource),
                bases=("Q",),
                implementation="native-relinearize-streaming",
            ),
        )
        return _ciphertext_result(
            value,
            data,
            polynomial_domain="coefficient",
            residue_representation="standard",
        )

    def switch_key(self, value: Ciphertext, key: KeySwitchKey) -> Ciphertext:
        """Switch one key relation without storing caller-owned key material."""

        symbol = "key-switch-key:caller"
        component0 = self._extract_component_tensor(
            value.data,
            0,
            basis="Q",
        )
        component1 = self._extract_component_tensor(
            value.data,
            1,
            basis="Q",
        )
        data = self._assemble_key_switch(
            component0,
            component1,
            value=value,
            key=key,
            key_symbol=symbol,
        )
        return _ciphertext_result(value, data)

    def rotate_with_key(
        self,
        value: Ciphertext,
        key: RotationKey,
    ) -> Ciphertext:
        """Rotate CKKS slots with one caller-owned direct rotation key."""

        return self._rotate_with_key(value, key)

    def _rotate_with_key(
        self,
        value: Ciphertext,
        key: RotationKey,
    ) -> Ciphertext:
        """Rotate with one key through eager automorphism and key switching."""

        key = cast(
            RotationKey,
            self._key_on_device(
                key,
                value.device,
                operation_name="rotate",
            ),
        )
        symbol = f"rotation-key:{key.rotation_step}"
        key_resource = self._bind_key(symbol, key)
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.RotateOp,
                value.data,
                resources=(key_resource,),
                bases=("Q",),
                implementation="native-rotate-streaming",
            ),
        )
        return _ciphertext_result(value, data)

    def rotate_by_step(
        self,
        value: Ciphertext,
        rotation_step: int,
    ) -> Ciphertext:
        """Rotate slots with the installed key for `rotation_step`."""

        normalized = RotationKey.normalize_step(
            rotation_step,
            ring_dimension=self.ring_dimension,
        )
        if normalized == 0:
            return _ciphertext_result(value, value.data.clone())
        result = value
        for key in self._rotation_key_path(normalized):
            result = self._rotate_with_key(result, key)
        return result

    def _rotation_key_path(
        self, normalized_step: int
    ) -> tuple[RotationKey, ...]:
        installed = self._keys.rotation_keys
        direct = installed.get(normalized_step)
        if direct is not None:
            return (direct,)
        if self.allow_automatic_key_generation:
            return (self.rotation_key(normalized_step),)
        decomposition = decompose_rotation_step(
            normalized_step,
            self.num_slots,
            installed,
        )
        return tuple(
            self._keys.require_rotation_key(step) for step in decomposition
        )

    def rotate_many_by_steps(
        self,
        value: Ciphertext,
        rotation_steps: Sequence[int],
        *,
        use_hoisting: bool = True,
    ) -> list[Ciphertext]:
        """Return caller-scheduled independent or hoisted slot rotations.

        ``rotation_steps`` determines the requested outputs. Setting
        ``use_hoisting`` selects one scheduled hoisted group for every direct
        key in that sequence; setting it to ``False`` executes each direct
        rotation independently. This method does not search for rotation
        patterns or choose a group from surrounding operations.
        """

        paths: list[tuple[RotationKey, ...]] = []
        for step in rotation_steps:
            normalized = RotationKey.normalize_step(
                step,
                ring_dimension=self.ring_dimension,
            )
            paths.append(
                () if normalized == 0 else self._rotation_key_path(normalized)
            )
        direct_indices = [
            index for index, path in enumerate(paths) if len(path) == 1
        ]
        direct_keys = [paths[index][0] for index in direct_indices]
        direct = (
            self._hoisted_rotate_many(value, direct_keys)
            if use_hoisting
            else [self.rotate_with_key(value, key) for key in direct_keys]
        )
        results: dict[int, Ciphertext] = dict(
            zip(direct_indices, direct, strict=True)
        )
        for index, path in enumerate(paths):
            if index in results:
                continue
            result = _ciphertext_result(value, value.data.clone())
            for key in path:
                result = self.rotate_with_key(result, key)
            results[index] = result
        return [results[index] for index in range(len(paths))]

    def rotate_many_with_keys(
        self,
        value: Ciphertext,
        keys: Sequence[RotationKey],
        *,
        use_hoisting: bool = True,
    ) -> list[Ciphertext]:
        """Return caller-scheduled independent or hoisted key rotations.

        Setting ``use_hoisting`` selects one scheduled hoisted group containing
        the supplied keys. The engine preserves their order and does not add,
        remove, or regroup offsets.
        """

        return (
            self._hoisted_rotate_many(value, list(keys))
            if use_hoisting
            else [self.rotate_with_key(value, key) for key in keys]
        )

    def _hoisted_rotate_many(
        self,
        value: Ciphertext,
        entries: Sequence[RotationKey | None],
    ) -> list[Ciphertext]:
        """Execute one caller-selected hoisted rotation group."""

        nonzero = [
            cast(
                RotationKey,
                self._key_on_device(
                    key,
                    value.device,
                    operation_name="rotate_many_with_keys",
                ),
            )
            for key in entries
            if key is not None
        ]
        if not nonzero:
            return [
                _ciphertext_result(value, value.data.clone()) for _ in entries
            ]
        key_resources: list[BoundResource] = []
        for index, key in enumerate(nonzero):
            symbol = f"rotation-key:{key.rotation_step}:{index}"
            key_resources.append(self._bind_key(symbol, key))
        executed = self._execute(
            ckks.RotateManyOp,
            value.data,
            resources=key_resources,
            bases=("Q",),
            result_count=len(nonzero),
            implementation="native-rotate-many-hoisted",
        )
        tensors = executed if isinstance(executed, tuple) else (executed,)
        results = [_ciphertext_result(value, tensor) for tensor in tensors]
        rotated = iter(results)
        output = [
            (
                _ciphertext_result(value, value.data.clone())
                if key is None
                else next(rotated)
            )
            for key in entries
        ]
        return output

    def rotate_by_step_(
        self,
        value: Ciphertext,
        rotation_step: int,
    ) -> Ciphertext:
        """Replace `value` with its slot rotation and return `value`."""

        return value.replace_(self.rotate_by_step(value, rotation_step))

    def conjugate(
        self,
        value: Ciphertext,
        key: ConjugationKey | None = None,
    ) -> Ciphertext:
        """Conjugate CKKS slots with caller-owned key material."""

        selected = (
            key if key is not None else self._keys.require_conjugation_key()
        )
        transformed = cast(
            torch.Tensor,
            self._execute(
                rns.CoefficientAutomorphismOp,
                value.data,
                resource_kinds=("rns",),
                attributes={"galois_element": 2 * self.ring_dimension - 1},
                bases=("Q",),
            ),
        )
        component0 = self._extract_component_tensor(
            transformed,
            0,
            basis="Q",
        )
        component1 = self._extract_component_tensor(
            transformed,
            1,
            basis="Q",
        )
        data = self._assemble_key_switch(
            component0,
            component1,
            value=value,
            key=selected,
            key_symbol="active-conjugation-key",
        )
        return _ciphertext_result(value, data)

    def __str__(self) -> str:
        """Return the CKKS configuration and initialized resource state."""

        return (
            "Engine("
            f"config={self.config}, "
            f"ntt_backend={self._ntt_backend!r}, "
            "allow_automatic_key_replication="
            f"{self.allow_automatic_key_replication}, "
            f"initialized_devices={tuple(map(str, self._dispatchers))})"
        )

    __repr__ = __str__


__all__ = ["Engine"]
