"""Execute eager CKKS operations through registered Backend implementations.

`Engine` owns one CKKS configuration and lazily constructs arithmetic,
random-stream, key-material, and executable resources for each device used by
the caller. Ordinary evaluator methods dispatch from their Tensor operands;
factory-like calls use CPU unless another device is selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from operator import index as integer_index
from threading import RLock
from typing import Literal, cast, overload

import torch
from xdsl.ir import Operation

from fhelium.config import CkksConfig, Preset
from fhelium.backend.rns.format import RnsExecutionFormat
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
from fhelium.errors import MaximumDepthError, ScaleMismatchError
from fhelium.backend.ckks import (
    CkksKeyGenerator,
    PUBLIC_KEY_RESOURCE_KIND,
    SECRET_KEY_RESOURCE_KIND,
)
from fhelium.backend.ckks.crypto import reconstruct_q_coefficients_tensor
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
    depth: int | None = None,
    scale: float | None = None,
    prime_ids: tuple[int, ...] | None = None,
    polynomial_domain: PolynomialDomain | None = None,
    modulus_basis: ModulusBasis | None = None,
    residue_representation: ResidueRepresentation | None = None,
) -> Ciphertext:
    """Build an Eager result from one source value and operation transition."""

    return Ciphertext._from_fields(
        data=data,
        depth=source.depth if depth is None else depth,
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
        depth=source.depth,
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
    r"""Own eager CKKS configuration and per-device execution resources.

    Cheon-Kim-Kim-Song (CKKS) ciphertexts encode approximate complex slot
    values as polynomials.  A two-component ciphertext (CT2) has phase
    $c_0+c_1s$; a three-component ciphertext (CT3) has phase
    $c_0+c_1s+c_2s^2$ for secret polynomial $s$.  The residue
    number system (RNS) stores a polynomial row modulo each active ciphertext
    prime in Q.  QP adds the special P primes used by hybrid key switching.
    The number-theoretic transform (NTT) turns negacyclic polynomial products
    into pointwise products.  Montgomery representation stores residue
    $x_i$ as $x_iR_i\bmod q_i$ so modular products avoid division.
    Hybrid key switching groups Q rows into digits, extends each digit to QP,
    multiplies it by evaluation-key components, and removes P.  Each plaintext
    and ciphertext records its own actual scale $\Delta$, which maps
    encoded integers to approximate slot values.

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
        rns_dtype: torch.dtype | None = None,
        rng_seed: int | None = None,
        rng_nonce: int | None = None,
        allow_automatic_key_generation: bool = True,
        allow_automatic_key_replication: bool = False,
    ) -> None:
        """Construct one CKKS lifecycle with lazily created device resources."""

        if ckks_config is None:
            ckks_config = Preset.slots16384_scale40_depth16_int64
        if not isinstance(ckks_config, CkksConfig):
            ckks_config = CkksConfig.parse(ckks_config)
        if type(allow_automatic_key_replication) is not bool:
            raise TypeError("allow_automatic_key_replication must be a bool")
        if ckks_config.enforce_security_budget:
            ckks_config.validate_security_budget()

        self._config = ckks_config
        self._rns_execution_format = RnsExecutionFormat.select(
            ckks_config.moduli, rns_dtype
        )
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
            q_depth_group_sizes=tuple(
                len(group) for group in ckks_config.q_depth_groups
            ),
        )
        self._rns_layout = RnsLayout(
            rns_chain,
            HybridRnsDecomposition(
                rns_chain,
                self.config.q_moduli,
                self.config.p_moduli,
            ),
        )
        self._validator = CkksValidator(
            ckks_config,
            self._rns_layout,
            self.dtype,
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
            rns_dtype=self.dtype,
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
    def depth0_qp_prime_ids(self) -> tuple[int, ...]:
        """Return the depth-zero QP prime identifiers."""

        return self._rns_layout.prime_ids(0, include_p=True)

    @property
    def key_digit_count(self) -> int:
        """Return the hybrid key-switch digit count."""

        return self._rns_layout.key_digit_count

    @property
    def dtype(self) -> torch.dtype:
        """Return the Engine's RNS storage dtype."""

        return self._rns_execution_format.dtype

    @property
    def max_depth(self) -> int:
        """Return the greatest public CKKS depth in this Engine."""

        return self.config.max_depth

    def depth_remaining(self, value: Ciphertext | Plaintext | int) -> int:
        """Return how many public rescale transitions remain."""

        depth = value if isinstance(value, int) else value.depth
        self._validator._validate_depth(depth)
        return self.max_depth - depth

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
        r"""Sample the secret polynomial and materialize its RNS transforms.

        The key generator samples ternary coefficients $s_j\in\{-1,0,1\}$, reduces
        them modulo every depth-zero Q row and optional P row, and stores
        $\operatorname{NTT}(s)R$ in Montgomery form.  This factory delegates to
        ``CkksKeyGenerator.create_secret_key``; it does not issue an IR operation."""

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
        r"""Generate a public-key pair for a secret polynomial.

        For a sampled uniform polynomial $a$ and error $e$, the returned
        $(k_0,k_1)=(-as+e,a)$ satisfies $k_0+k_1s=e$ modulo each selected
        depth-zero Q or QP prime.  Components are stored in NTT/Montgomery form by
        ``CkksKeyGenerator.create_public_key``."""

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
        r"""Generate hybrid-RNS material from one secret relation to another.

        For source and destination secrets $s_{src}$ and $s_{dst}$, key digit
        $d$ satisfies
        $k_{d,0}+k_{d,1}s_{dst}=P s_{src}+e_d$ on that digit's embedded source rows,
        where $P$ is the product of special primes.  The generator returns
        ``[digit, component=2, QP row, NTT index]`` data in NTT/Montgomery form."""

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
        r"""Generate key-switch material from $s^2$ to $s$.

        This is the evaluation key used to replace the $c_2s^2$ phase term of a CT3
        product by two CT2 correction components.  The QP key digits are generated by
        ``CkksKeyGenerator.create_relinearization_key`` in NTT/Montgomery form."""

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
        r"""Generate key-switch material from $\sigma_g(s)$ to $s$.

        The signed slot displacement is normalized modulo the slot count and mapped to
        an odd Galois element $g$.  The ring automorphism
        $\sigma_g:X\mapsto X^g$ rotates the slots while changing the secret
        relation; the returned QP key digits restore the original relation."""

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
        r"""Generate key-switch material from $\sigma_{2N-1}(s)$ to $s$.

        The automorphism $X\mapsto X^{-1}$ gives complex conjugation under the CKKS
        embedding.  ``CkksKeyGenerator.create_conjugation_key`` returns the QP
        NTT/Montgomery key digits needed to restore the original secret relation."""

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
        depth: int = 0,
        scale: float | None = None,
        device: torch.device | str | None = None,
    ) -> Plaintext:
        r"""Wrap ordered public slots with planned CKKS depth and actual scale.

        This method stores the message tensor without applying the CKKS embedding.
        ``encode`` later maps the slots to a polynomial.  ``scale`` defaults to the
        configuration's planning value but is stored on this plaintext as its own
        actual scale."""

        if type(depth) is not int:
            raise TypeError("depth must be an integer")
        if not 0 <= depth <= self.max_depth:
            raise ValueError(f"depth must be in [0, {self.max_depth}]")
        actual_scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="Plaintext",
        )
        target = torch.get_default_device() if device is None else device
        return Plaintext(
            message=torch.as_tensor(message, device=target).detach().clone(),
            depth=depth,
            scale=actual_scale,
        )

    def encode(
        self,
        message: Sequence[object] | torch.Tensor | complex | float | int,
        *,
        depth: int = 0,
        scale: float | None = None,
        device: torch.device | str | None = None,
    ) -> Plaintext:
        r"""Encode ordered CKKS slots as scaled integer coefficients.

        For the configured embedding $\sigma$, message $m$, and this value's actual
        scale $\Delta$, the returned polynomial is
        $a=\operatorname{RandRound}(\Delta\sigma^{-1}(m))$.  The method dispatches
        ``ckks.EncodeOp`` to ``native-ckks-encode`` and returns coefficient-domain
        ``integer_coefficients`` state at ``depth``; no RNS prime rows exist yet."""

        actual_scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="encode scale",
        )
        if type(depth) is not int or not 0 <= depth <= self.max_depth:
            raise ValueError(f"depth must be in [0, {self.max_depth}]")
        target = torch.get_default_device() if device is None else device
        message_tensor = torch.as_tensor(message, device=target)
        coefficients = cast(
            torch.Tensor,
            self._execute(
                ckks.EncodeOp,
                message_tensor,
                attributes={"depth": depth, "scale": actual_scale},
            ),
        )
        return Plaintext._from_fields(
            message=None,
            depth=depth,
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
        r"""Reduce an integer plaintext polynomial into active prime rows.

        At plaintext depth $\ell$, every coefficient $a_j$ becomes
        $a_j\bmod q_i$ in each active Q row and, for ``QP``,
        $a_j\bmod p_i$ in each special P row.  ``ckks.IntegerCoefficientsToRnsOp``
        dispatches to ``native-ckks-integer-coefficients-to-rns``.  The result is
        coefficient-domain standard RNS with unchanged depth and actual scale."""

        if plaintext.is_slots:
            if plaintext.message is None:
                raise ValueError("slots plaintext has no message Tensor")
            plaintext = self.encode(
                plaintext.message,
                depth=plaintext.depth,
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
                    "depth": plaintext.depth,
                    "modulus_basis": modulus_basis,
                },
            ),
        )
        return Plaintext._from_fields(
            message=None,
            depth=plaintext.depth,
            scale=plaintext.scale,
            data=data,
            representation="rns",
            polynomial_domain="coefficient",
            modulus_basis=modulus_basis,
            residue_representation="standard",
            prime_ids=self._rns_layout.prime_ids(
                plaintext.depth,
                include_p=modulus_basis == "QP",
            ),
        )

    def prepare_plaintext_for_addition(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: Literal["Q", "QP"] = "Q",
        polynomial_domain: PolynomialDomain = "coefficient",
    ) -> Plaintext:
        r"""Prepare a plaintext for modular addition to component zero.

        The method first issues ``ckks.IntegerCoefficientsToRnsOp`` as needed,
        then ``rns.StandardToMontgomeryOp``. ``polynomial_domain="ntt"`` also
        transforms the prepared polynomial for direct addition to an
        NTT/Montgomery ciphertext. Message, depth, actual scale, and requested
        Q or QP rows are unchanged."""

        if polynomial_domain not in ("coefficient", "ntt"):
            raise ValueError("polynomial_domain must be 'coefficient' or 'ntt'")
        standard = self.integer_coefficients_to_rns(
            plaintext,
            modulus_basis=modulus_basis,
        )
        prepared = self.standard_residues_to_montgomery_residues(standard)
        if polynomial_domain == "ntt":
            return cast(
                Plaintext,
                self.coefficient_domain_to_ntt_domain(prepared),
            )
        return prepared

    def prepare_plaintext_for_multiplication(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: Literal["Q", "QP"] = "Q",
    ) -> Plaintext:
        r"""Prepare a plaintext for pointwise NTT ciphertext multiplication.

        After addition preparation, the method dispatches
        ``ntt.CoefficientMontgomeryToNttMontgomeryOp``.  Every active-prime polynomial
        is transformed while retaining its Montgomery factor.  Depth and plaintext
        actual scale remain unchanged for the eventual ``rns.MultiplyPlaintextOp``."""

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
        r"""Prepare a typed public value for encrypted addition or multiplication.

        Addition selects the ciphertext actual scale so the prepared plaintext can be
        added to component zero.  Multiplication retains a supplied plaintext scale or
        uses the configured default for a message/static value; its scale later
        multiplies the ciphertext scale.  This Eager method composes ``ckks.EncodeOp``
        and registered RNS/NTT transitions rather than emitting the Compile-only
        ``ckks.Prepare*Op`` classes."""

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
        r"""Execute encoding and representation changes selected for a public operand.

        The result has the ciphertext's depth, Q or QP rows, and device.  Addition
        returns coefficient/Montgomery RNS at matching actual scale; multiplication
        returns NTT/Montgomery RNS at its selected plaintext scale.  Existing prepared
        RNS payloads are reused when their represented state already fits."""

        if operation not in {"add", "multiply"}:
            raise ValueError(
                "Public-value preparation operation is unsupported"
            )
        if isinstance(public, Plaintext):
            if public.is_approximate_coefficients:
                raise ValueError(
                    "approximate_coefficients Plaintext is decode-only"
                )
            if public.depth != ciphertext.depth:
                raise ValueError("Plaintext and ciphertext depths differ")
            if operation == "add" and public.scale != ciphertext.scale:
                raise ValueError("Plaintext addition requires equal scales")
            if public.is_slots:
                assert public.message is not None
                prepared_source = self.encode(
                    public.message,
                    depth=public.depth,
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
                depth=ciphertext.depth,
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
        r"""Decode coefficient data into the configured CKKS slot order.

        For coefficient polynomial $a$ at actual scale $\Delta$, the result is
        $\sigma(a/\Delta)$, restricted to the configured number of slots.
        ``is_real`` takes the real part after embedding.  The method dispatches
        ``ckks.DecodeOp`` to ``native-ckks-decode``; RNS plaintexts must first be
        reconstructed to bounded coefficient data."""

        if device is not None:
            plaintext = plaintext.to(device)
        target = plaintext.device
        if plaintext.is_slots:
            if plaintext.message is None:
                raise ValueError("slots plaintext has no message Tensor")
            plaintext = self.encode(
                plaintext.message,
                depth=plaintext.depth,
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
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Encrypt an integer coefficient plaintext under a public key.

        For public key $(k_0,k_1)$, sampled binary $v$, and errors $e_0,e_1$,
        ``ckks.EncryptOp`` computes
        $(c_0,c_1)=(k_0v+a+e_0,\ k_1v+e_1)$ modulo each active prime.  The
        ``native-ckks-encrypt`` implementation returns a CT2 Q or QP
        ciphertext with the plaintext's depth and actual scale.
        ``output_domain`` selects coefficient/standard or NTT/Montgomery
        output. NTT output transforms the error-and-message terms and adds them
        directly to the public-key products."""

        if output_domain not in ("coefficient", "ntt"):
            raise ValueError("output_domain must be 'coefficient' or 'ntt'")
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
                depth=plaintext.depth,
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
                    "depth": plaintext.depth,
                    "output_domain": output_domain,
                },
                bindings=self._key_resource(symbol, key),
            ),
        )
        basis = key.modulus_basis
        return Ciphertext._from_fields(
            data=data,
            depth=plaintext.depth,
            scale=plaintext.scale,
            prime_ids=self._dispatcher_for(
                target
            ).rns_context.rns_layout.prime_ids(
                plaintext.depth,
                include_p=basis == "QP",
            ),
            polynomial_domain=output_domain,
            modulus_basis=basis,
            residue_representation=(
                "standard" if output_domain == "coefficient" else "montgomery"
            ),
        )

    def decrypt(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Plaintext:
        r"""Evaluate the ciphertext phase and reconstruct bounded coefficients.

        For CT2 the phase is $c_0+c_1s$; for CT3 it is
        $c_0+c_1s+c_2s^2$, evaluated in every active prime. Both component
        counts may use coefficient/standard or NTT/Montgomery input. NTT input
        is summed before one inverse transform; coefficient input transforms
        each nonconstant term for multiplication by the matching secret power.
        The returned approximate-coefficient plaintext retains depth and actual
        scale."""

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
                    "depth": ciphertext.depth,
                    "modulus_basis": ciphertext.modulus_basis,
                    "input_domain": ciphertext.polynomial_domain,
                },
                bindings=self._key_resource(symbol, key),
            ),
        )
        return Plaintext._from_fields(
            message=None,
            depth=ciphertext.depth,
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
        depth: int = 0,
        scale: float | None = None,
        device: torch.device | str | None = None,
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Encode slots and encrypt the resulting polynomial.

        This composes ``encode`` and ``encrypt``: it forms
        $a=\operatorname{RandRound}(\Delta\sigma^{-1}(m))$, then encrypts $a$ as
        a randomized CT2 value through ``ckks.EncodeOp`` and ``ckks.EncryptOp``.  The
        output records the selected depth, actual scale, public-key basis, and
        coefficient/standard representation."""

        return self.encrypt(
            self.encode(message, depth=depth, scale=scale, device=device),
            public_key,
            device=device,
            output_domain=output_domain,
        )

    def decrypt_message(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey | None = None,
        *,
        is_real: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        r"""Decrypt a ciphertext phase and decode its CKKS slots.

        This composes ``ckks.DecryptOp`` with ``ckks.DecodeOp``.  It reconstructs an
        approximate coefficient polynomial $a$, then returns
        $\sigma(a/\Delta)$ using the ciphertext's actual scale $\Delta$.
        ``is_real`` projects the decoded slots to their real parts."""

        return self.decode(
            self.decrypt(ciphertext, secret_key, device=device),
            is_real=is_real,
            device=device,
        )

    def zero_plaintext_like(self, plaintext: Plaintext) -> Plaintext:
        r"""Return the additive identity in the same plaintext representation.

        Every stored slot or polynomial coefficient is set to zero while depth, actual
        scale, prime rows, polynomial domain, and Montgomery state are copied.  No
        registered operation is dispatched."""

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
        r"""Return a randomized encryption of zero in a ciphertext's represented state.

        The method encodes and encrypts a zero slot vector at the reference depth and
        actual scale, then applies the same registered NTT transition if the reference
        is in NTT/Montgomery form.  The result decrypts to encryption noise around
        zero and has matching Q or QP rows and representation."""

        self._validator.validate_ciphertext(ciphertext)
        zero = torch.zeros(
            (*ciphertext.batch_shape, self.num_slots),
            dtype=torch.complex128,
            device=ciphertext.device,
        )
        result = self.encrypt_message(
            zero,
            public_key,
            depth=ciphertext.depth,
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

    def reconstruct_q_coefficients(
        self,
        residues: torch.Tensor,
        ciphertext: Ciphertext,
    ) -> torch.Tensor:
        r"""Reconstruct centered coefficients from all active Q rows.

        The active Q basis represents each coefficient modulo its product $M$.
        Mixed-radix reconstruction returns its centered representative in
        $(-M/2,M/2]$.  This utility delegates to the decryption reconstruction
        routine and does not decode slots or divide by the CKKS scale."""

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
        return reconstruct_q_coefficients_tensor(
            residues,
            depth=ciphertext.depth,
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
        target.depth = source.depth
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
        r"""Add ciphertext components modulo every active prime.

        Inputs have matching component, row, polynomial-domain, residue, and scale
        state, with residues in each prime's standard storage range.  For each
        component $j$, row $q_i$, and stored position,
        $c'_j=c^{(lhs)}_j+c^{(rhs)}_j\pmod {q_i}$.  The method dispatches
        ``rns.AddStandardOp`` to ``native-rns-linear``.  Component count, depth, prime
        rows, polynomial domain, residue representation, and actual scale are
        preserved; ``inplace=True`` writes the same result into ``lhs``."""

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
        r"""Add $rhs$ componentwise into $lhs$ and return $lhs$.

        This is the in-place form of ``add`` and dispatches ``rns.AddStandardOp``.
        All represented state is preserved."""

        return self.add(lhs, rhs, inplace=True)

    def add_scalar(
        self,
        ciphertext: Ciphertext,
        scalar: int | float,
        *,
        scalar_scale: float | None = None,
    ) -> Ciphertext:
        r"""Add a quantized real scalar to the constant coefficient of $c_0$.

        For scalar $u$ and scalar scale $\delta$, ``ckks.AddScalarOp`` samples
        $k=\operatorname{RandRound}(u\delta)$ and adds $k\bmod q_i$ to component
        zero's constant coefficient in every active row.  The ciphertext actual scale
        $\Delta$ stays unchanged, so its decoded increment is $k/\Delta$;
        $\delta=\Delta$ approximates addition by $u$."""

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
        r"""Sum a non-empty sequence by repeated modular ciphertext addition.

        The method clones the first value and repeatedly dispatches
        ``rns.AddStandardOp``.  Mathematically each output component is
        $\sum_t c^{(t)}_j\pmod {q_i}$.  Compatible depth, rows, representation,
        component count, and actual scale are preserved."""

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
        r"""Reduce one batch axis by pairwise modular ciphertext addition.

        A tree of ``rns.AddStandardOp`` calls computes the sum over ``dim`` for every
        component, prime row, and coefficient.  Pairing changes evaluation order but
        not modular addition.  The selected batch axis is removed; CKKS depth, actual
        scale, rows, and representation remain unchanged."""

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
        r"""Subtract ciphertext components modulo every active prime.

        Inputs have matching component, row, polynomial-domain, residue, and scale
        state, with residues in each prime's standard storage range.  For each
        component $j$ and row $q_i$,
        $c'_j=c^{(lhs)}_j-c^{(rhs)}_j\pmod {q_i}$.  The method dispatches
        ``rns.SubtractStandardOp`` to ``native-rns-linear`` and preserves depth, scale,
        prime rows, component count, polynomial domain, and residue representation.
        ``inplace=True`` writes into ``lhs``."""

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
        r"""Subtract $rhs$ componentwise from $lhs$ and return $lhs$.

        This in-place form dispatches ``rns.SubtractStandardOp`` and retains the
        ciphertext's represented state."""

        return self.subtract(lhs, rhs, inplace=True)

    def negate(
        self,
        value: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Negate every ciphertext component modulo its active primes.

        Residues may describe coefficient polynomials or NTT values but must be in
        each prime's standard storage range.  For each component and prime row,
        ``rns.NegateStandardOp`` computes
        $c'_j=-c_j\pmod {q_i}$ through ``native-rns-linear``.  The result decrypts to
        the additive inverse and preserves depth, actual scale, rows, component count,
        and representation."""

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
        r"""Negate every component of $value$ in place and return it.

        This is the in-place ``rns.NegateStandardOp`` path; all represented state is
        unchanged."""

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
        r"""Apply the forward negacyclic NTT to every RNS polynomial.

        For ciphertext standard residues, each component dispatches
        ``ntt.CoefficientStandardToNttMontgomeryOp`` and stores
        $\operatorname{NTT}(c_j)R$.  For plaintext Montgomery residues, the method
        dispatches ``ntt.CoefficientMontgomeryToNttMontgomeryOp``.  Depth, prime rows,
        basis, component shape, and actual scale are preserved; the result state is
        NTT/Montgomery."""

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
            self._execute(
                operation_type,
                data,
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
        r"""Transform $value$ to NTT/Montgomery form in place.

        This dispatches the same registered NTT operation as
        ``coefficient_domain_to_ntt_domain`` and returns the input object with
        unchanged depth, prime rows, and actual scale."""

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
        r"""Apply the inverse negacyclic NTT to every RNS polynomial.

        Ciphertext components dispatch ``ntt.NttMontgomeryToCoefficientStandardOp``
        and end in coefficient/standard form.  A plaintext dispatches
        ``ntt.InverseMontgomeryOp`` and ends in coefficient/Montgomery form.  The ring
        element, depth, Q or QP rows, component shape, and actual scale are preserved."""

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
            self._execute(
                operation_type,
                data,
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
        r"""Transform $value$ from NTT form in place and return it.

        The method dispatches the ciphertext or plaintext inverse NTT operation used
        by ``ntt_domain_to_coefficient_domain``; it does not alter depth, rows, or
        actual scale."""

        return self.ntt_domain_to_coefficient_domain(value, inplace=True)

    def standard_residues_to_montgomery_residues(
        self,
        plaintext: Plaintext,
        *,
        inplace: bool = False,
    ) -> Plaintext:
        r"""Multiply each coefficient residue by its Montgomery radix.

        ``rns.StandardToMontgomeryOp`` maps $x_i$ to $x_iR_i\bmod q_i$ through
        ``native-rns-transition``.  The plaintext remains in coefficient domain with
        unchanged polynomial, Q or QP rows, depth, and actual scale."""

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
        r"""Convert plaintext rows to Montgomery representation in place.

        This dispatches ``rns.StandardToMontgomeryOp`` and returns the input plaintext;
        its polynomial, depth, prime rows, and actual scale do not change."""

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
        r"""Remove the Montgomery factor from each coefficient residue.

        ``rns.MontgomeryToStandardOp`` maps $x_iR_i$ to $x_i\bmod q_i$ through
        ``native-rns-transition``.  The plaintext polynomial, coefficient domain,
        Q or QP rows, depth, and actual scale are preserved."""

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
        r"""Convert plaintext rows to standard residues in place.

        This dispatches ``rns.MontgomeryToStandardOp`` and returns the input plaintext
        without changing its polynomial, depth, rows, or actual scale."""

        return self.montgomery_residues_to_standard_residues(
            plaintext,
            inplace=True,
        )

    def rescale_divisor(self, *, depth: int) -> int:
        r"""Return the active Q divisor removed at ``depth``.

        ``rescale_to_next_depth`` divides each ciphertext coefficient and its actual
        scale by the product of the Q rows assigned to this transition."""

        if type(depth) is not int:
            raise TypeError("depth must be an integer")
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if depth >= self.max_depth:
            raise MaximumDepthError(
                depth=depth,
                maximum_depth=self.max_depth,
            )
        return self.config.rescale_divisor(depth)

    def rescale_output_scale(
        self,
        input_scale: float,
        *,
        depth: int,
    ) -> float:
        r"""Compute the per-value actual scale after one rescale.

        For input scale $\Delta$ and depth-group product $M_d$, the result is
        $\Delta'=\Delta/M_d$. This helper performs no residue arithmetic."""

        scale = coerce_scale(input_scale, value_name="input_scale")
        return coerce_scale(
            scale / self.rescale_divisor(depth=depth),
            value_name="rescale output",
        )

    def _rescale_to_next_basis(
        self,
        value: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Divide-round by the leading Q group and advance one depth.

        The input is Q residue data in coefficient/standard or NTT/Montgomery
        representation. ``rns.RescaleDropLeadingPrimesOp`` computes the rounded
        quotient by the complete group product $M_d$ and removes that group's
        rows. With NTT input, the dropped rows determine one coefficient
        correction, whose transform is added to the scaled surviving evaluations.
        Depth advances once and scale becomes $\Delta/M_d$."""

        if rounding not in {"nearest", "floor"}:
            raise ValueError("rounding must be 'nearest' or 'floor'")
        if value.data.size(-2) <= 1:
            raise MaximumDepthError(
                depth=value.depth,
                maximum_depth=self.max_depth,
            )
        next_prime_ids = self._rns_layout.prime_ids(
            value.depth + 1,
            include_p=value.modulus_basis == "QP",
        )
        drop_count = len(value.prime_ids) - len(next_prime_ids)
        dropped_modulus = math.prod(
            int(self.config.moduli[prime_id])
            for prime_id in value.prime_ids[:drop_count]
        )
        bases = self._operand_bases(value)
        input_domain = value.polynomial_domain
        data = cast(
            torch.Tensor,
            self._execute(
                rns.RescaleDropLeadingPrimesOp,
                value.data,
                resource_kinds=(
                    ("rescale", "rns", "ntt")
                    if input_domain == "ntt"
                    else ("rescale",)
                ),
                attributes={
                    "drop_count": drop_count,
                    "rounding": rounding,
                    "input_domain": input_domain,
                    "output_domain": input_domain,
                },
                bases=bases,
                in_place=inplace,
            ),
        )
        if inplace:
            value.data = data
            value.depth += 1
            value.prime_ids = next_prime_ids
            value.scale /= float(dropped_modulus)
            return value
        return _ciphertext_result(
            value,
            data,
            depth=value.depth + 1,
            prime_ids=next_prime_ids,
            scale=value.scale / float(dropped_modulus),
        )

    def rescale_to_next_depth(
        self,
        value: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Divide by the current Q depth-group modulus and consume one depth.

        A depth group may contain multiple RNS primes. The operation removes
        every row in that group, advances ``depth`` once, and divides the
        value's actual scale by the complete group product.
        """

        if value.depth >= self.max_depth:
            raise MaximumDepthError(
                depth=value.depth,
                maximum_depth=self.max_depth,
            )
        return self._rescale_to_next_basis(
            value,
            rounding=rounding,
            inplace=inplace,
        )

    def rescale_to_next_depth_(
        self,
        value: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        r"""Rescale $value$ by its leading Q group in place and return it.

        The underlying row operations remove every prime in the group; the
        public value advances one depth and divides scale by the group product."""

        return self.rescale_to_next_depth(
            value,
            rounding=rounding,
            inplace=True,
        )

    def mod_switch_to_depth(
        self,
        value: Ciphertext,
        target_depth: int,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Discard Q rows until ``target_depth`` without quotient scaling.

        If the source rows are $(q_\ell,\ldots,q_L)$, the result retains the suffix
        $(q_t,\ldots,q_L)$; P rows are retained for QP values.  Surviving residues
        and actual scale $\Delta$ are unchanged.  Eager performs this tensor-row
        selection directly, with the same mathematical transition represented in IR
        by ``rns.RestrictDepthOp``/``ckks.ModSwitchOp``."""

        prime_ids = self._rns_layout.prime_ids(
            target_depth,
            include_p=value.modulus_basis == "QP",
        )
        result_rows = len(prime_ids)
        source_rows = value.data.size(-2)
        if not 0 < result_rows <= source_rows:
            raise ValueError("target_depth requires unavailable RNS rows")
        selected = value.data.narrow(
            -2,
            source_rows - result_rows,
            result_rows,
        )
        data = selected if inplace else selected.clone()
        if inplace:
            value.data = data
            value.depth = target_depth
            value.prime_ids = prime_ids
            return value
        return _ciphertext_result(
            value,
            data,
            depth=target_depth,
            prime_ids=prime_ids,
        )

    def mod_switch_to_next_depth(
        self,
        value: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Discard the next Q depth group and advance one depth.

        This calls ``mod_switch_to_depth`` for $\ell+1$.  Unlike rescaling, it does
        not divide coefficients or actual scale and does not dispatch a Backend
        operation in Eager."""

        return self.mod_switch_to_depth(
            value,
            value.depth + 1,
            inplace=inplace,
        )

    def mod_switch_to_next_depth_(self, value: Ciphertext) -> Ciphertext:
        r"""Discard the next Q depth group in place and return $value$.

        The actual scale and surviving residues are unchanged; only depth and row
        metadata advance."""

        return self.mod_switch_to_next_depth(value, inplace=True)

    def mod_switch_to_depth_(
        self,
        value: Ciphertext,
        target_depth: int,
    ) -> Ciphertext:
        r"""Restrict $value$ in place to ``target_depth`` and return it.

        The method performs direct row selection, preserving actual scale and every
        surviving residue."""

        return self.mod_switch_to_depth(value, target_depth, inplace=True)

    def reinterpret_at_scale(
        self,
        value: Ciphertext,
        target_scale: float,
        *,
        max_relative_change: float | None = None,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Replace actual-scale metadata without arithmetic.

        The payload, depth, prime rows, polynomial domain, and Montgomery state are
        copied unchanged while scale $\Delta$ becomes ``target_scale``
        $\Delta'$.  Decoding therefore interprets the same coefficients as
        $x/\Delta'$.  Eager performs the metadata change directly; IR represents it
        with ``ckks.ReinterpretScaleOp``/``rns.ReinterpretScaleOp``."""

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
        r"""Replace $value$'s actual-scale metadata in place and return it.

        No residues or rows change.  ``max_relative_change`` limits the permitted
        ratio between the old and new scale."""

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
        r"""Multiply two CT2 ciphertexts into one CT3 ciphertext.

        Both inputs are two-component NTT/Montgomery ciphertexts with matching depth
        and active prime rows.  For $a=(a_0,a_1)$ and $b=(b_0,b_1)$,
        ``ckks.MultiplyOp`` computes
        $(a_0b_0,\ a_0b_1+a_1b_0,\ a_1b_1)$ in each active-prime negacyclic ring.
        The ``native-ct2-convolution`` implementation consumes NTT/Montgomery inputs.
        Depth and rows are retained, and actual scale becomes
        $\Delta_a\Delta_b$; relinearization and rescaling remain separate calls."""

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
        r"""Multiply a ciphertext by a stochastically quantized real scalar.

        For scalar $u$ and scalar scale $\delta$, ``ckks.MultiplyScalarOp`` samples
        $k=\operatorname{RandRound}(u\delta)$ and multiplies every component by
        $k$ modulo all active primes.  The result actual scale is
        $\Delta\delta$, so it represents multiplication by approximately $u$.
        Depth, rows, component count, and representation are preserved."""

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
        r"""Multiply every ciphertext component by an integer $k$.

        ``ckks.MultiplyIntegerScalarOp`` computes $c'_j=kc_j$ modulo each active
        prime through ``native-ckks-scalar-arithmetic``.  Because $k$ is unscaled,
        actual scale, depth, rows, component count, and representation do not change."""

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
        r"""Add a prepared plaintext polynomial to ciphertext component zero.

        For ordinary RNS plaintext $p$, ``rns.AddPlaintextOp`` computes
        $(c_0+p,c_1,\ldots)$ through ``native-plaintext-arithmetic``.  Compressed
        input dispatches ``ckks.AddCompressedPlaintextOp`` and expands the same values
        logically.  Matching depth, prime rows, and actual scale are required; output
        state and later components are preserved."""

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
        if prepared.polynomial_domain != ciphertext.polynomial_domain:
            raise ValueError(
                "Plaintext addition requires matching polynomial domains"
            )
        if (
            prepared.residue_representation != "montgomery"
            or (
                ciphertext.polynomial_domain == "coefficient"
                and ciphertext.residue_representation != "standard"
            )
            or (
                ciphertext.polynomial_domain == "ntt"
                and ciphertext.residue_representation != "montgomery"
            )
        ):
            raise ValueError(
                "Plaintext addition requires a Montgomery plaintext and a "
                "coefficient/standard or NTT/Montgomery ciphertext"
            )
        prepared_data = cast(torch.Tensor, prepared.data)
        bases = self._operand_bases(ciphertext, prepared)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.AddPlaintextOp,
                ciphertext.data,
                prepared_data,
                resource_kinds=("rns",),
                attributes={"polynomial_domain": ciphertext.polynomial_domain},
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
        r"""Add prepared plaintext $p$ into ciphertext component zero in place.

        This is the in-place ordinary/compressed plaintext addition path and returns
        the input ciphertext with unchanged represented state."""

        return self.add_plaintext(ciphertext, plaintext, inplace=True)

    def multiply_plaintext(
        self,
        ciphertext: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Multiply each ciphertext component by a prepared plaintext.

        For NTT/Montgomery plaintext $p$, ``rns.MultiplyPlaintextOp`` computes
        $c'_j=c_jp$ pointwise modulo every active prime.  Compressed input dispatches
        ``ckks.MultiplyCompressedPlaintextOp`` for the same expanded polynomial.
        Depth, rows, and component count remain; actual scale becomes
        $\Delta_c\Delta_p$, with no implicit rescale."""

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
        r"""Replace a ciphertext by its prepared-plaintext product.

        The method uses the ordinary or compressed registered multiplication path,
        updates actual scale to $\Delta_c\Delta_p$, and returns the input object."""

        return self.multiply_plaintext(ciphertext, plaintext, inplace=True)

    def sum_plaintext_products(
        self,
        ciphertexts: Sequence[Ciphertext],
        plaintexts: Sequence[Plaintext],
    ) -> Ciphertext:
        r"""Compute a sum of prepared-plaintext products in one operation.

        For equally sized sequences, the result is

        $$
        y=\sum_{t=0}^{T-1}c_t p_t.
        $$

        Every ciphertext must have the same NTT/Montgomery Q or QP state,
        depth, active rows, component shape, and actual scale.  Every prepared
        plaintext must match those rows and have a common actual scale.  The
        result preserves the ciphertext state and has scale
        $\Delta_c\Delta_p$.  The registered RNS implementation may fuse the
        products and modular accumulation without materializing each product.
        """

        if not ciphertexts or len(ciphertexts) != len(plaintexts):
            raise ValueError(
                "sum_plaintext_products requires equal non-empty sequences"
            )
        first_ciphertext = ciphertexts[0]
        first_plaintext = plaintexts[0]
        plaintext_data: list[torch.Tensor] = []
        for plaintext in plaintexts:
            if plaintext.data is None:
                raise ValueError(
                    "sum_plaintext_products requires RNS plaintext data"
                )
            plaintext_data.append(plaintext.data)
        ciphertext_data = tuple(value.data for value in ciphertexts)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.MontgomeryWeightedSumOp,
                *ciphertext_data,
                *plaintext_data,
                resource_kinds=("rns",),
                attributes={"term_count": len(ciphertexts)},
                bases=self._operand_bases(*ciphertexts, *plaintexts),
            ),
        )
        return _ciphertext_result(
            first_ciphertext,
            data,
            scale=first_ciphertext.scale * first_plaintext.scale,
        )

    def sum_plaintext_product_groups(
        self,
        ciphertexts: Sequence[Ciphertext],
        plaintext_groups: Sequence[Sequence[Plaintext]],
    ) -> Ciphertext:
        r"""Apply several prepared-plaintext rows to shared ciphertext terms.

        For $T$ ciphertexts and a caller-supplied $G\times T$ matrix of
        prepared plaintexts, batch result $g$ is

        $$
        y_g=\sum_{t=0}^{T-1}c_t p_{g,t}.
        $$

        Every operand follows :meth:`sum_plaintext_products`' matching
        NTT/Montgomery state, depth, row, shape, and scale requirements. Each
        result has scale $\Delta_c\Delta_p$ and preserves the ciphertext
        state. The output inserts the group axis as its first batch axis, so
        :meth:`Ciphertext.unbind_batch` recovers separate results. The
        operation executes the supplied rectangular matrix; it
        does not choose how a caller partitions a linear transform. A Backend
        implementation may reuse a ciphertext load across plaintext rows
        without stacking the inputs or materializing individual products.
        """

        if not ciphertexts or not plaintext_groups:
            raise ValueError(
                "sum_plaintext_product_groups requires non-empty terms and groups"
            )
        term_count = len(ciphertexts)
        if any(len(group) != term_count for group in plaintext_groups):
            raise ValueError(
                "each plaintext group must contain one term per ciphertext"
            )
        first_ciphertext = ciphertexts[0]
        first_plaintext = plaintext_groups[0][0]
        plaintexts = tuple(
            plaintext for group in plaintext_groups for plaintext in group
        )
        plaintext_data: list[torch.Tensor] = []
        for plaintext in plaintexts:
            if plaintext.data is None:
                raise ValueError(
                    "sum_plaintext_product_groups requires RNS plaintext data"
                )
            plaintext_data.append(plaintext.data)
        data = cast(
            torch.Tensor,
            self._execute(
                rns.MontgomeryWeightedSumsOp,
                *(value.data for value in ciphertexts),
                *plaintext_data,
                resource_kinds=("rns",),
                attributes={
                    "term_count": term_count,
                    "group_count": len(plaintext_groups),
                },
                bases=self._operand_bases(*ciphertexts, *plaintexts),
            ),
        )
        return _ciphertext_result(
            first_ciphertext,
            data,
            scale=first_ciphertext.scale * first_plaintext.scale,
        )

    def sum_rotated_plaintext_product_groups(
        self,
        ciphertext: Ciphertext,
        rotation_keys: Sequence[RotationKey | None],
        plaintext_groups: Sequence[Sequence[Plaintext]],
    ) -> Ciphertext:
        r"""Apply prepared-plaintext rows to one direct rotation group.

        A ``None`` key denotes the unrotated input; each other entry supplies
        its own signed rotation step. For $T$ entries and a caller-supplied
        $G\times T$ plaintext matrix, output batch $g$ is

        $$
        y_g=\sum_{t=0}^{T-1}p_{g,t}\operatorname{Rot}_{b_t}(c).
        $$

        The input is coefficient/standard Q CT2. Every direct rotation
        completes hybrid key switching and ModDown before its NTT/Montgomery
        plaintext product. The output inserts a first batch axis of length
        $G$, remains at the input depth, and has scale $\Delta_c\Delta_p$.
        The caller supplies the group membership and key inventory; Backend
        execution may reuse digit preparation and ciphertext loads within
        these supplied groups.
        """

        if not rotation_keys or not plaintext_groups:
            raise ValueError(
                "sum_rotated_plaintext_product_groups requires non-empty "
                "rotations and groups"
            )
        if sum(key is None for key in rotation_keys) > 1:
            raise ValueError("the unrotated entry may occur at most once")
        term_count = len(rotation_keys)
        if any(len(group) != term_count for group in plaintext_groups):
            raise ValueError(
                "each plaintext group must contain one term per rotation"
            )
        selected_keys = [
            cast(
                RotationKey,
                self._key_on_device(
                    key,
                    ciphertext.device,
                    operation_name="sum_rotated_plaintext_product_groups",
                ),
            )
            for key in rotation_keys
            if key is not None
        ]
        key_resources = [
            self._bind_key(f"rotation-key:{key.rotation_step}:{index}", key)
            for index, key in enumerate(selected_keys)
        ]
        plaintexts = tuple(
            plaintext for group in plaintext_groups for plaintext in group
        )
        plaintext_data: list[torch.Tensor] = []
        for plaintext in plaintexts:
            if plaintext.data is None:
                raise ValueError(
                    "sum_rotated_plaintext_product_groups requires RNS plaintext data"
                )
            plaintext_data.append(plaintext.data)
        first_plaintext = plaintext_groups[0][0]
        baby_steps = tuple(
            0 if key is None else key.rotation_step for key in rotation_keys
        )
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.GroupedRotationWeightedSumOp,
                ciphertext.data,
                *plaintext_data,
                resources=key_resources,
                attributes={
                    "baby_steps": baby_steps,
                    "term_count": term_count,
                    "group_count": len(plaintext_groups),
                },
                bases=self._operand_bases(ciphertext, *plaintexts),
                implementation="native-grouped-rotation-weighted-sum",
            ),
        )
        return _ciphertext_result(
            ciphertext,
            data,
            scale=ciphertext.scale * first_plaintext.scale,
            polynomial_domain="ntt",
            residue_representation="montgomery",
        )

    def relinearize(
        self,
        value: Ciphertext,
        key: RelinearizationKey | None = None,
        *,
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Map a CT3 product phase back to two components.

        The input is a three-component NTT/Montgomery product.  For phase
        $c_0+c_1s+c_2s^2$, the relinearization key switches the $c_2s^2$ term into
        corrections $(d_0,d_1)$.  ``ckks.RelinearizeOp`` dispatches to
        ``native-relinearize-streaming`` and returns
        $(c_0+d_0,c_1+d_1)$. ``output_domain="ntt"`` retains $c_0,c_1$
        and the corrections in NTT/Montgomery form; the default returns
        coefficient/standard Q rows. Depth and actual scale are unchanged."""

        if output_domain not in ("coefficient", "ntt"):
            raise ValueError("output_domain must be 'coefficient' or 'ntt'")
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
                attributes={"output_domain": output_domain},
                implementation="native-relinearize-streaming",
            ),
        )
        return _ciphertext_result(
            value,
            data,
            polynomial_domain=output_domain,
            residue_representation=(
                "standard" if output_domain == "coefficient" else "montgomery"
            ),
        )

    def switch_key(
        self,
        value: Ciphertext,
        key: KeySwitchKey,
        *,
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Switch a CT2 ciphertext from a source secret to a destination secret.

        The input is a two-component coefficient/standard Q ciphertext.  For
        $c_0+c_1s_{src}$, hybrid key switching maps the second term to
        corrections $(d_0,d_1)$ under $s_{dst}$, producing
        $(c_0+d_0,d_1)$. ``ckks.SwitchKeyOp`` dispatches the whole operation
        through one streaming hybrid-RNS implementation, which shares its QP
        accumulator across decomposition digits. ``output_domain="ntt"``
        combines $c_0$ with its correction before the forward transform.
        Depth, Q rows, actual scale, and CT2 shape are preserved."""

        if output_domain not in ("coefficient", "ntt"):
            raise ValueError("output_domain must be 'coefficient' or 'ntt'")
        symbol = "key-switch-key:caller"
        selected = cast(
            KeySwitchKey,
            self._key_on_device(
                key, value.device, operation_name="key switching"
            ),
        )
        key_resource = self._bind_key(symbol, selected)
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.SwitchKeyOp,
                value.data,
                bindings=self._key_switch_bindings(value.device, key_resource),
                attributes={
                    "key_symbol": symbol,
                    "output_domain": output_domain,
                },
                bases=("Q",),
                implementation="native-key-switch-streaming",
            ),
        )
        return _ciphertext_result(
            value,
            data,
            polynomial_domain=output_domain,
            residue_representation=(
                "standard" if output_domain == "coefficient" else "montgomery"
            ),
        )

    def rotate_with_key(
        self,
        value: Ciphertext,
        key: RotationKey,
        *,
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Apply the slot displacement carried by a direct rotation key.

        The input is a two-component coefficient/standard or NTT/Montgomery Q
        ciphertext. The call dispatches ``ckks.RotateOp`` to
        ``native-rotate-streaming``. Its
        Galois automorphism $\sigma_g:X\mapsto X^g$ rotates encoded slots and hybrid
        key switching restores secret $s$ from $\sigma_g(s)$.  CT2 shape, depth,
        active Q rows and actual scale are preserved. ``output_domain="ntt"``
        requests NTT/Montgomery output. For NTT input and output, the
        automorphism and unchanged component-zero contribution remain in NTT
        form while only component one is inverted for key switching."""

        if output_domain not in ("coefficient", "ntt"):
            raise ValueError("output_domain must be 'coefficient' or 'ntt'")
        return self._rotate_with_key(value, key, output_domain=output_domain)

    def _rotate_with_key(
        self,
        value: Ciphertext,
        key: RotationKey,
        *,
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Execute one ``ckks.RotateOp`` automorphism and key switch.

        The bound rotation key supplies its signed displacement and the corresponding
        Galois relation.  ``native-rotate-streaming`` returns a CT2 ciphertext in the
        same depth, Q rows and actual scale, with the selected output representation."""

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
                attributes={
                    "input_domain": value.polynomial_domain,
                    "output_domain": output_domain,
                },
                bases=("Q",),
                implementation="native-rotate-streaming",
            ),
        )
        return _ciphertext_result(
            value,
            data,
            polynomial_domain=output_domain,
            residue_representation="montgomery"
            if output_domain == "ntt"
            else "standard",
        )

    def rotate_by_step(
        self,
        value: Ciphertext,
        rotation_step: int,
    ) -> Ciphertext:
        r"""Cyclically rotate CKKS slots by a signed displacement.

        The step is normalized modulo the slot count.  A zero step clones the input;
        an installed direct key issues one ``ckks.RotateOp`` and a decomposed key path
        issues successive rotations whose displacements sum modulo the slot count.
        Each stage preserves CT2 shape, depth, Q rows, and actual scale."""

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
        r"""Return the requested cyclic slot rotations in caller order.

        Each normalized step denotes the same mathematical result as
        ``rotate_by_step``.  Direct single-key rotations may be grouped into one
        ``ckks.RotateManyOp`` dispatched to ``native-rotate-many-hoisted``; hoisting
        shares digit decomposition and Q-to-QP basis extension but does not regroup
        or alter outputs.  Multi-key paths remain successive ``ckks.RotateOp``
        calls."""

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
        output_domain: PolynomialDomain = "coefficient",
    ) -> list[Ciphertext]:
        r"""Apply each supplied rotation key and preserve its output order.

        With hoisting, one ``ckks.RotateManyOp`` shares key-switch preparation while
        each result equals the corresponding independent ``ckks.RotateOp``.  Without
        hoisting, keys dispatch separately.  Every output keeps CT2 shape, depth,
        active Q rows and actual scale. ``output_domain="ntt"`` returns
        NTT/Montgomery results, equivalent to transforming each coefficient
        result. The hoisted implementation retains Q evaluations during ModDown;
        independent rotations combine c0 with the coefficient correction before
        its forward NTT."""

        if output_domain not in ("coefficient", "ntt"):
            raise ValueError("output_domain must be 'coefficient' or 'ntt'")
        if use_hoisting:
            return self._hoisted_rotate_many(
                value, list(keys), output_domain=output_domain
            )
        return [
            self.rotate_with_key(value, key, output_domain=output_domain)
            for key in keys
        ]

    def _hoisted_rotate_many(
        self,
        value: Ciphertext,
        entries: Sequence[RotationKey | None],
        *,
        output_domain: PolynomialDomain = "coefficient",
    ) -> list[Ciphertext]:
        r"""Execute one caller-selected group through ``ckks.RotateManyOp``.

        ``native-rotate-many-hoisted`` shares hybrid decomposition of component one,
        then applies each key's Galois automorphism and key product independently.
        Each output is mathematically the requested slot rotation and retains the
        input's depth, scale and Q rows. The selected output domain determines
        coefficient/standard or NTT/Montgomery representation."""

        identity = (
            self.coefficient_domain_to_ntt_domain(value)
            if output_domain == "ntt" and any(key is None for key in entries)
            else value
        )

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
                _ciphertext_result(identity, identity.data.clone())
                for _ in entries
            ]
        key_resources: list[BoundResource] = []
        for index, key in enumerate(nonzero):
            symbol = f"rotation-key:{key.rotation_step}:{index}"
            key_resources.append(self._bind_key(symbol, key))
        executed = self._execute(
            ckks.RotateManyOp,
            value.data,
            resources=key_resources,
            attributes={"output_domain": output_domain},
            bases=("Q",),
            result_count=len(nonzero),
            implementation="native-rotate-many-hoisted",
        )
        tensors = executed if isinstance(executed, tuple) else (executed,)
        results = [
            _ciphertext_result(
                value,
                tensor,
                polynomial_domain=output_domain,
                residue_representation="montgomery"
                if output_domain == "ntt"
                else "standard",
            )
            for tensor in tensors
        ]
        rotated = iter(results)
        output = [
            (
                _ciphertext_result(identity, identity.data.clone())
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
        r"""Replace $value$ with its cyclic slot rotation and return it.

        The method uses ``rotate_by_step`` and therefore dispatches the required
        ``ckks.RotateOp`` sequence while preserving depth, rows, and actual scale."""

        return value.replace_(self.rotate_by_step(value, rotation_step))

    def conjugate(
        self,
        value: Ciphertext,
        key: ConjugationKey | None = None,
        *,
        output_domain: PolynomialDomain = "coefficient",
    ) -> Ciphertext:
        r"""Apply complex conjugation to every CKKS slot.

        The input is a two-component coefficient/standard Q ciphertext.  The method
        dispatches ``ckks.ConjugateOp`` to a whole-operation implementation. It
        applies $g=2N-1$, which substitutes $X\mapsto X^{-1}$, then uses the
        streaming hybrid-RNS key-switch path to return from $\sigma_g(s)$ to
        $s$. ``output_domain`` selects coefficient/standard or
        NTT/Montgomery output. CT2 shape, depth, Q rows, and actual scale remain
        unchanged."""

        if output_domain not in ("coefficient", "ntt"):
            raise ValueError("output_domain must be 'coefficient' or 'ntt'")
        selected = (
            key if key is not None else self._keys.require_conjugation_key()
        )
        selected = cast(
            ConjugationKey,
            self._key_on_device(
                selected, value.device, operation_name="conjugate"
            ),
        )
        key_resource = self._bind_key("conjugation-key", selected)
        data = cast(
            torch.Tensor,
            self._execute(
                ckks.ConjugateOp,
                value.data,
                bindings=self._key_switch_bindings(value.device, key_resource),
                bases=("Q",),
                attributes={"output_domain": output_domain},
                implementation="native-key-switch-streaming",
            ),
        )
        return _ciphertext_result(
            value,
            data,
            polynomial_domain=output_domain,
            residue_representation=(
                "standard" if output_domain == "coefficient" else "montgomery"
            ),
        )

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
