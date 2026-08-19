"""CKKS operations for one configuration and local device."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import Literal, cast, overload
from uuid import uuid4

import torch

from fhelium.config import CkksConfig, Preset
from fhelium.core import (
    Ciphertext,
    CkksContextSpec,
    CompressedPlaintext,
    ConjugationKey,
    KeySwitchKey,
    ModulusBasis,
    Plaintext,
    PolynomialDomain,
    PublicKey,
    RelinearizationKey,
    ResidueRepresentation,
    RotationKey,
    RotationKeySet,
    SecretKey,
)
from fhelium.core.rotation import decompose_rotation_step
from fhelium.core.scale import coerce_scale
from fhelium.engine.ckks_decryptor import CkksDecryptor
from fhelium.engine.ckks_encryptor import CkksEncryptor
from fhelium.engine.ckks_plaintext_codec import CkksPlaintextCodec
from fhelium.engine.ckks_rescale import CkksRescaler
from fhelium.engine.direct_keyswitch_consumer import (
    create_direct_keyswitch_digit_consumer,
)
from fhelium.engine.galois import (
    apply_coefficient_galois_automorphism,
    coefficient_galois_gather_indices,
    rotation_galois_element,
)
from fhelium.engine.hybrid_keyswitch import HybridKeySwitcher
from fhelium.engine.key_generator import CkksKeyGenerator
from fhelium.engine.rns.runtime import RnsRuntime
from fhelium.errors import (
    MaximumLevelError,
    ScaleMismatchError,
)
from fhelium.native.wrapper import ckks_ops
from fhelium.rng import Csprng


class CkksEngine:
    r"""CKKS execution facade for dense process-local values.

    The engine fixes $N=2^{\mathtt{logN}}$, the ordered Q and P primes, one
    integral tensor dtype, one local device, and an NTT backend. Public values
    carry exact level, actual scale, polynomial domain, modulus basis,
    residue form, and exact ``prime_ids``; operations never infer hidden level
    or scale alignment.

    ``ntt_backend`` is an exact engine execution-policy name. When omitted,
    CUDA engines select ``fhelium.DEFAULT_NTT_BACKEND`` and CPU engines select
    ``fhelium.DEFAULT_CPU_NTT_BACKEND``. Engine
    construction does not benchmark hardware. A selected backend name must support
    both the device and ring dimension. The backend affects table layout and
    kernel execution, not CKKS context identity.

    Construction creates one :class:`~fhelium.rng.Csprng` from the engine
    configuration and device. ``rng_seed`` and ``rng_nonce`` provide fixed
    stream material for controlled tests and benchmarks; applications should
    leave them unset so the generator obtains production entropy.

    """

    def __init__(
        self,
        ckks_config: CkksConfig | Preset | dict[str, object] | None = None,
        *,
        device: torch.device | str | None = None,
        allow_sk_gen: bool = True,
        galois_generator: int = 3,
        ntt_backend: str | None = None,
        rng_seed: int | None = None,
        rng_nonce: int | None = None,
    ) -> None:
        if ckks_config is None:
            ckks_config = Preset.slots16384_scale40_levels16_int64
        if not isinstance(ckks_config, CkksConfig):
            ckks_config = CkksConfig.parse(ckks_config)
        self.config = ckks_config
        if rng_seed is not None and type(rng_seed) is not int:
            raise TypeError("rng_seed must be an integer or None")
        if rng_nonce is not None and type(rng_nonce) is not int:
            raise TypeError("rng_nonce must be an integer or None")
        if galois_generator not in {3, 5}:
            raise ValueError("galois_generator must be 3 or 5")
        if self.config.enforce_security_budget:
            # Fail before selecting a device or constructing native/RNS
            # runtime state.  The assessment covers the complete QP modulus.
            self.config.validate_security_budget()

        from fhelium.native import (
            native_backend_available,
            require_native_backend,
        )

        if device is None:
            if torch.cuda.is_available() and native_backend_available("cuda"):
                device = torch.device("cuda", torch.cuda.current_device())
            elif native_backend_available("cpu"):
                device = torch.device("cpu")
            else:
                # Preserve the backend-specific diagnostic when the installed
                # product is CUDA-only but no CUDA device is usable.
                require_native_backend("cpu")
                raise AssertionError("require_native_backend must raise")
        selected_device = torch.device(device)
        if selected_device.type not in {"cpu", "cuda"}:
            raise ValueError("CkksEngine requires a CPU or CUDA device")
        if selected_device.type == "cpu":
            selected_device = torch.device("cpu")
        elif selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        require_native_backend(selected_device.type)
        self.device = selected_device

        self.rns_runtime = RnsRuntime(
            self.config,
            device=self.device,
            ntt_backend=ntt_backend,
        )
        self.ntt_backend_name = self.rns_runtime.ntt_backend_name
        self.rns_layout = self.rns_runtime.rns_layout
        self.montgomery_parameters = self.rns_runtime.montgomery_parameters
        self.context = CkksContextSpec(
            logN=self.config.logN,
            default_scale=float(self.config.default_scale),
            q_moduli=tuple(self.config.q_moduli),
            p_moduli=tuple(self.config.p_moduli),
            galois_generator=galois_generator,
        )

        self._rng = Csprng(
            num_coefs=self.config.N,
            num_channels=[len(self.rns_layout.prime_ids(0))],
            num_repeating_channels=max(self.config.num_p_primes, 2),
            sigma=self.config.sigma,
            devices=[str(self.device)],
            torch_dtype=self.config.torch_dtype,
            seed=rng_seed,
            nonce=rng_nonce,
        )
        self.galois_generator = galois_generator
        self.id = str(uuid4())
        self.allow_sk_gen = allow_sk_gen

        self._create_p_product_montgomery_q()
        self._create_keyswitch_moddown_parameters()
        self._create_rescale_dropped_q_inverses_montgomery()

        self._plaintext_codec = CkksPlaintextCodec(
            config=self.config,
            context=self.context,
            device=self.device,
            rng=self._rng,
            rns_layout=self.rns_layout,
            rns_runtime=self.rns_runtime,
            galois_generator=self.galois_generator,
            engine_id=self.id,
            validate_public_level=self._validate_public_level,
        )
        self._encryptor = CkksEncryptor(
            config=self.config,
            device=self.device,
            rng=self._rng,
            rns_layout=self.rns_layout,
            rns_runtime=self.rns_runtime,
            montgomery_parameters=self.montgomery_parameters,
            plaintext_codec=self._plaintext_codec,
            engine_id=self.id,
            validate_public_level=self._validate_public_level,
            ciphertext_from_components=self._ciphertext_from_components,
        )
        self._decryptor = CkksDecryptor(
            config=self.config,
            device=self.device,
            rns_layout=self.rns_layout,
            rns_runtime=self.rns_runtime,
            montgomery_parameters=self.montgomery_parameters,
            plaintext_codec=self._plaintext_codec,
            engine_id=self.id,
        )
        self._key_generator = CkksKeyGenerator(
            config=self.config,
            context=self.context,
            device=self.device,
            rng=self._rng,
            rns_runtime=self.rns_runtime,
            p_product_montgomery_q=self.p_product_montgomery_q,
            galois_generator=self.galois_generator,
        )
        self._hybrid_key_switcher = HybridKeySwitcher(
            config=self.config,
            rns_runtime=self.rns_runtime,
            moddown_p_drop_inverses_montgomery_by_level=(
                self.moddown_p_drop_inverses_montgomery_by_level
            ),
            direct_digit_consumer=create_direct_keyswitch_digit_consumer(
                self.rns_runtime
            ),
            galois_generator=self.galois_generator,
        )
        self._rescaler = CkksRescaler(
            engine_id=self.id,
            device=self.device,
            public_level_count=self.public_level_count,
            rns_runtime=self.rns_runtime,
            montgomery_parameters=self.montgomery_parameters,
            dropped_q_inverses_montgomery_by_level=(
                self.rescale_dropped_q_inverses_montgomery_by_level
            ),
            assert_engine_ciphertext=self._assert_engine_ciphertext,
            ciphertext_from_components=self._ciphertext_from_components,
            rescale_to_next_output_scale=self.rescale_to_next_output_scale,
        )
        self._secret_key: SecretKey | None = None
        self._public_key: PublicKey | None = None
        self._relinearization_key: RelinearizationKey | None = None
        self._rotation_keys = RotationKeySet()

    @property
    def public_level_count(self) -> int:
        r"""Number of ordinary public CKKS levels.

        The count equals ``config.num_scale_primes``. Public levels satisfy
        $0\leq\mathtt{level}<\mathtt{public\_level\_count}$. The subsequent
        one-prime structural-basis state is internal to bootstrap entry.
        """

        return self.config.num_scale_primes

    @property
    def final_public_level(self) -> int:
        r"""Greatest ordinary public CKKS level.

        The value is ``public_level_count - 1``. Its active Q basis contains
        the final scale prime and the structural base prime. Public next-level
        transitions require a source below this level. The bootstrap structural
        transition consumes a ciphertext at this level.
        """

        return self.public_level_count - 1

    @property
    def rng(self) -> Csprng:
        """Stable random-number generator shared by all engine components."""

        return self._rng

    @property
    def num_slots(self) -> int:
        r"""Semantic CKKS slot count $S=N/2$."""

        return self.config.N // 2

    # ------------------------------------------------------------------
    # Key lifecycle.
    # ------------------------------------------------------------------

    def create_secret_key(
        self, *, modulus_basis: ModulusBasis = "QP"
    ) -> SecretKey:
        r"""Sample a fresh secret polynomial without installing it.

        Coefficients of $s(X)\in R$ are sampled from ``{-1, 0, 1}``, reduced
        over the complete level-zero Q or QP basis, and transformed to
        NTT-domain Montgomery residues. The result has layout
        ``[limb, ntt_index]``, engine integral dtype/device, final extent $N$,
        and exact level-zero ``prime_ids``. Generation allocates independent
        key storage and does not change the engine's installed key lifecycle.
        """

        modulus_basis = self._validate_modulus_basis(modulus_basis)
        return self._key_generator.create_secret_key(
            modulus_basis=modulus_basis
        )

    def create_public_key(
        self,
        secret_key: SecretKey,
        *,
        modulus_basis: ModulusBasis = "Q",
    ) -> PublicKey:
        r"""Create a public encryption key for ``secret_key``.

        The two generated components satisfy

        $$
        k_0(X)+k_1(X)s(X)=e(X)\pmod{B_0},
        $$

        where $B_0$ is Q or QP as requested. Output layout is
        ``[key_component=2, limb, ntt_index]`` in level-zero NTT-domain
        Montgomery form with engine integral dtype/device and exact
        ``prime_ids``. The input secret key is not mutated and the public key
        is not installed on the engine.
        """

        modulus_basis = self._validate_modulus_basis(modulus_basis)
        self._assert_engine_key(
            secret_key,
            expected_type=SecretKey,
            modulus_basis="QP" if modulus_basis == "QP" else None,
        )
        return self._key_generator.create_public_key(
            secret_key, modulus_basis=modulus_basis
        )

    def create_relinearization_key(
        self, secret_key: SecretKey
    ) -> RelinearizationKey:
        r"""Create QP key-switch material from $s(X)^2$ to $s(X)$.

        The result replaces $d_2(X)s(X)^2$ by two corrections under $s(X)$,
        reducing a three-component product to two while preserving level
        and actual scale up to key-switch error. Storage is
        ``[key_digit, key_component=2, limb, ntt_index]`` in complete
        level-zero QP, NTT-domain Montgomery form on the engine device. It is
        returned without being installed; ``secret_key`` is not mutated.
        """

        self._assert_engine_key(
            secret_key, expected_type=SecretKey, modulus_basis="QP"
        )
        return self._key_generator.create_relinearization_key(secret_key)

    def create_rotation_key(
        self, rotation_step: int, secret_key: SecretKey
    ) -> RotationKey:
        r"""Create QP key-switch material for one signed slot rotation.

        For canonical step $r$, generation derives the Galois element $g$,
        applies $\sigma_g$ to $s(X)$, and creates a key switch from
        $\sigma_g(s(X))$ back to $s(X)$. The matching operation produces
        $m'_j=m_{(j-r)\bmod S}$, equal to ``torch.roll(m, shifts=r)``.
        The result uses the full level-zero QP basis, NTT-domain Montgomery
        form, engine integral dtype/device, and independent storage; it is not
        installed automatically.
        """

        self._assert_engine_key(
            secret_key, expected_type=SecretKey, modulus_basis="QP"
        )
        return self._key_generator.create_rotation_key(
            self._canonical_rotation_step(rotation_step),
            secret_key=secret_key,
        )

    def create_conjugation_key(self, secret_key: SecretKey) -> ConjugationKey:
        r"""Create QP material from $\sigma_{-1}(s(X))$ back to $s(X)$.

        The result has layout
        ``[key_digit, key_component=2, limb, ntt_index]`` in complete
        level-zero QP, NTT-domain Montgomery form with engine integral
        dtype/device and exact ``prime_ids``. It enables semantic slot
        conjugation and is returned without installation; ``secret_key`` is
        not mutated.
        """

        self._assert_engine_key(
            secret_key, expected_type=SecretKey, modulus_basis="QP"
        )
        return self._key_generator.create_conjugation_key(secret_key)

    def create_key_switch_key(
        self,
        source_secret_key: SecretKey,
        destination_secret_key: SecretKey,
        *,
        uniform_component_by_key_digit: torch.Tensor | None = None,
    ) -> KeySwitchKey:
        r"""Create QP material that switches source phases to destination phases.

        The direction is

        $$
        c_0(X)+c_1(X)s_{\mathrm{source}}(X)
        \longmapsto
        c'_0(X)+c'_1(X)s_{\mathrm{destination}}(X).
        $$

        Both secret keys must be complete level-zero QP NTT/Montgomery values
        for this engine. The output layout is
        ``[key_digit, key_component=2, limb, ntt_index]`` with engine integral
        dtype/device and exact QP ``prime_ids``. Optional uniform components
        are indexed by stable ``key_digit_index``. Inputs are not mutated and
        the result is not installed.
        """

        self._assert_engine_key(
            source_secret_key,
            expected_type=SecretKey,
            modulus_basis="QP",
        )
        self._assert_engine_key(
            destination_secret_key,
            expected_type=SecretKey,
            modulus_basis="QP",
        )
        return self._key_generator.create_key_switch_key(
            source_secret_key,
            destination_secret_key,
            uniform_component_by_key_digit=uniform_component_by_key_digit,
        )

    # ------------------------------------------------------------------
    # Message, Plaintext, and Ciphertext conversion facade.
    # ------------------------------------------------------------------

    @property
    def secret_key(self) -> SecretKey:
        """Return the installed secret key, generating one lazily if allowed."""

        if self._secret_key is None:
            if not self.allow_sk_gen:
                raise RuntimeError("Secret-key generation is disabled")
            self._secret_key = self.create_secret_key()
        return self._secret_key

    def set_secret_key(self, key: SecretKey) -> None:
        """Install ``key`` and invalidate all dependent installed keys.

        The key tensor is retained rather than cloned. Installed public,
        relinearization, and rotation keys are cleared because this value type
        does not carry a symbolic key-lineage identifier.
        """

        self._assert_engine_key(key, expected_type=SecretKey)
        self._secret_key = key
        self._public_key = None
        self._relinearization_key = None
        self._rotation_keys = RotationKeySet()

    @property
    def public_key(self) -> PublicKey:
        """Return or lazily generate the installed public encryption key."""

        if self._public_key is None:
            self._public_key = self.create_public_key(self.secret_key)
        return self._public_key

    def set_public_key(self, key: PublicKey) -> None:
        """Validate and install ``key`` by reference without changing others."""

        self._assert_engine_key(key, expected_type=PublicKey)
        self._public_key = key

    @property
    def relinearization_key(self) -> RelinearizationKey:
        """Return or lazily generate installed $s(X)^2$-to-$s(X)$ material."""

        if self._relinearization_key is None:
            self._relinearization_key = self.create_relinearization_key(
                self.secret_key
            )
        return self._relinearization_key

    def set_relinearization_key(self, key: RelinearizationKey) -> None:
        """Validate and install QP relinearization material by reference."""

        self._assert_engine_key(
            key, expected_type=RelinearizationKey, modulus_basis="QP"
        )
        self._relinearization_key = key

    @property
    def rotation_keys(self) -> RotationKeySet:
        """Mutable installed mapping from canonical signed steps to keys."""

        return self._rotation_keys

    def rotation_key(self, rotation_step: int) -> RotationKey:
        """Return the matching installed key, generating it lazily if allowed."""

        rotation_step = self._canonical_rotation_step(rotation_step)
        if rotation_step not in self._rotation_keys:
            if not self.allow_sk_gen:
                raise KeyError(
                    "No rotation key materialized for rotation step "
                    f"{rotation_step}"
                )
            self._rotation_keys.add(
                self.create_rotation_key(rotation_step, self.secret_key)
            )
        return self._rotation_keys[rotation_step]

    def set_rotation_key(self, key: RotationKey) -> None:
        """Install a key under its self-described canonical rotation step."""

        self._assert_engine_key(
            key, expected_type=RotationKey, modulus_basis="QP"
        )
        self._rotation_keys.add(key)

    def plaintext(self, message, *, level: int = 0, scale=None) -> Plaintext:
        r"""Create a lazy slots-only plaintext with an actual scale argument.

        No encoding arithmetic is performed. The value stores semantic slots
        $m$ and $\Delta(p)$ for the later mapping

        $$
        p_i=\operatorname{SRound}\!\left(
          \Delta(p)\,\mathcal{E}^{-1}_g(m)_i
        \right).
        $$

        Args:
            message: Tensor-like real or complex slot values.
            level: Public CKKS level at which a later encode will materialize
                the message.
            scale: Positive finite per-value scale. The context's
                ``config.default_scale`` is used only when this is ``None``.

        Returns:
            A detached, cloned slots ``Plaintext`` on the engine device. It has
            no polynomial domain, modulus basis, residue form, or
            ``prime_ids`` and does not alias the caller's tensor.

        Raises:
            InvalidScaleError: If ``scale`` is not positive and finite.
            ValueError: If the requested level or message layout is invalid.
        """

        return self._plaintext_codec.plaintext(
            message, level=level, scale=scale
        )

    def encode(
        self,
        message,
        *,
        level: int = 0,
        scale=None,
    ) -> Plaintext:
        r"""Encode slots at one programmer-selected level and actual scale.

        For $m\in\mathbb{C}^S$, compute

        $$
        p_i=\operatorname{SRound}\!\left(
          \Delta\,\mathcal{E}^{-1}_g(m)_i
        \right),\qquad
        \mathbb{E}[\operatorname{SRound}(x)]=x.
        $$

        Args:
            message: Tensor-like real or complex slot values.
            level: Public CKKS level for the encoded value.
            scale: Positive finite per-value scale, or ``None`` to use
                ``config.default_scale``.
        Returns:
            A new exact ``integer_coefficients`` plaintext with layout
            ``[*batch, coefficient]``, final extent $N$, engine integral dtype
            and device, coefficient domain, and the selected binary64 actual
            scale. It has no RNS basis or ``prime_ids``.

        Raises:
            InvalidScaleError: If ``scale`` is not positive and finite.
            ValueError: If the requested level or arithmetic state is invalid.
        """

        return self._plaintext_codec.encode(
            message,
            level=level,
            scale=scale,
        )

    def integer_coefficients_to_rns(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: ModulusBasis = "Q",
    ) -> Plaintext:
        r"""Reduce an integer polynomial to coefficient-domain standard RNS.

        For every exact active ``prime_ids[i]`` with modulus $q_i$, compute

        $$
        r_{i,j}=p_j\bmod q_i.
        $$

        Input ``[*batch, coefficient]`` becomes
        ``[*batch, limb, coefficient]`` with engine integral dtype/device and
        final extent $N$. It performs neither an NTT nor Montgomery conversion.

        The result has ``representation="rns"``,
        ``polynomial_domain="coefficient"``, and
        ``residue_representation="standard"``. Its level, scale, semantic
        polynomial, and exact active ``prime_ids`` are preserved. The input
        must be the ``integer_coefficients`` result of :meth:`encode`; this
        transition never performs encoding or CRT reconstruction implicitly.
        """

        modulus_basis = self._validate_modulus_basis(modulus_basis)
        return self._plaintext_codec.integer_coefficients_to_rns(
            plaintext, modulus_basis=modulus_basis
        )

    def prepare_plaintext_for_addition(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: ModulusBasis = "Q",
    ) -> Plaintext:
        r"""Prepare coefficient-domain Montgomery RNS data for addition.

        For each exact active ``prime_ids[i]`` with modulus $q_i$, compute

        $$
        \widetilde{r}_{i,j}=(p_j\bmod q_i)R\bmod q_i.
        $$

        The result has layout ``[*batch, limb, coefficient]``, engine integral
        dtype/device, and final extent $N$; no NTT is applied.

        The result's state is ``(representation="rns",
        polynomial_domain="coefficient",
        residue_representation="montgomery")``. Level, scale, semantic
        polynomial, modulus basis, and exact ``prime_ids`` are preserved.

        Semantically, this convenience operation is equivalent to

        ``standard_residues_to_montgomery_residues(
        integer_coefficients_to_rns(plaintext))``.

        The implementation reuses its newly allocated intermediate storage.
        """

        rns = self.integer_coefficients_to_rns(
            plaintext, modulus_basis=modulus_basis
        )
        return self._to_montgomery_plaintext_(rns)

    def prepare_plaintext_for_multiplication(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: ModulusBasis = "Q",
    ) -> Plaintext:
        r"""Prepare NTT-domain Montgomery RNS data for multiplication.

        For each exact active ``prime_ids[i]`` with modulus $q_i$, reduce
        $p(X)$ and compute its negacyclic NTT in Montgomery form. Input
        ``[*batch, coefficient]`` becomes ``[*batch, limb, ntt_index]`` with
        engine integral dtype/device and final extent $N$. This changes only
        arithmetic representation; it does not change level or actual scale.

        The result's state is ``(representation="rns",
        polynomial_domain="ntt",
        residue_representation="montgomery")`` with the selected Q or QP
        ``modulus_basis`` and its exact ``prime_ids``.

        Semantically, this convenience operation is equivalent to

        ``coefficient_domain_to_ntt_domain(
        standard_residues_to_montgomery_residues(
        integer_coefficients_to_rns(plaintext)))``.

        The implementation reuses its newly allocated intermediate storage.
        """

        rns = self.integer_coefficients_to_rns(
            plaintext, modulus_basis=modulus_basis
        )
        self._to_montgomery_plaintext_(rns)
        return self._to_ntt_plaintext_(rns)

    def decode(self, plaintext: Plaintext, *, is_real: bool = False):
        r"""Decode a plaintext with its own actual per-value scale.

        $$
        m_{\mathrm{approx}}=
        \mathcal{E}_g(p)/\Delta(p).
        $$

        Slots input is encoded first at its stored scale. Otherwise the input
        is exact integral ``integer_coefficients`` or finite binary64
        ``approximate_coefficients`` with layout ``[*batch, coefficient]``,
        final extent $N$, and engine device. RNS input is rejected because CRT
        reconstruction is distinct. The functional output is CPU
        ``[*batch, slot]`` with extent $S=N/2$ and is complex unless
        ``is_real=True`` selects its real part.
        """

        return self._plaintext_codec.decode(plaintext, is_real=is_real)

    def encrypt(
        self,
        plaintext: Plaintext,
        public_key: PublicKey | None = None,
    ) -> Ciphertext:
        r"""Encrypt slots or exact integer coefficients under a public key.

        The output phase satisfies

        $$
        c_0(X)+c_1(X)s(X)=p(X)+e(X)\pmod{B_\ell},
        $$

        Slots input is first encoded at its stored actual scale. Decode-only
        ``approximate_coefficients`` and RNS input are rejected. With
        configured encryption noise $e(X)$, the result is a new dense
        ``[component=2, *batch, limb, coefficient]`` ciphertext in coefficient
        domain and standard residues, over Q or QP according to the public
        key. It uses engine integral dtype/device, exact active ``prime_ids``,
        unchanged level, and $\Delta(c)=\Delta(p)$. Input and key are not
        mutated, and output storage does not alias them.
        """

        if not isinstance(plaintext, Plaintext):
            raise TypeError(
                f"encrypt expects Plaintext, got {type(plaintext).__name__}"
            )
        prepared_plaintext = self._plaintext_codec._ensure_integer_plaintext(
            plaintext
        )
        self._validate_public_level(prepared_plaintext.level)
        if prepared_plaintext.data is None:
            raise RuntimeError(
                "Coefficient Plaintext data was not materialized"
            )
        return self._encryptor.encrypt(
            prepared_plaintext,
            self._resolve_public_key(public_key),
        )

    def decrypt(
        self,
        ct: Ciphertext,
        secret_key: SecretKey | None = None,
    ) -> Plaintext:
        r"""Decrypt to bounded binary64 coefficients for decoding.

        The RNS phase is

        $$
        u(X)=\sum_{j=0}^{d-1}c_j(X)s(X)^j\pmod{B_\ell},
        \qquad d\in\{2,3\}.
        $$

        The current decrypt path reconstructs the centered class from the trailing
        Q-prime pair into a finite ``torch.float64``
        ``[*batch, coefficient]`` tensor on the engine device. It is bounded
        ``approximate_coefficients`` for decoding, not an exact full-$Q_\ell$
        CRT inverse and not encryptable RNS input. The returned plaintext keeps
        ``ct.level`` and $\Delta(p)=\Delta(ct)$; scale division occurs only in
        :meth:`decode`. Inputs are unchanged and output storage is independent.
        """

        if not isinstance(ct, Ciphertext):
            raise TypeError(
                f"decrypt expects Ciphertext, got {type(ct).__name__}"
            )
        self._assert_engine_ciphertext(ct)
        return self._decryptor.decrypt(
            ct,
            self._resolve_secret_key(secret_key),
        )

    def encrypt_message(
        self,
        message,
        public_key: PublicKey | None = None,
        *,
        level: int = 0,
        scale=None,
    ) -> Ciphertext:
        r"""Encode and encrypt slots at the requested level and actual scale.

        First compute

        $$
        p_i=\operatorname{SRound}\!\left(
          \Delta\,\mathcal{E}^{-1}_g(m)_i
        \right),
        $$

        then encrypt so that

        $$
        c_0(X)+c_1(X)s(X)=p(X)+e(X)\pmod{B_\ell}.
        $$

        Args:
            message: Tensor-like real or complex slot values.
            public_key: Compatible key, or the engine-installed/default key.
            level: Public CKKS level for the new ciphertext.
            scale: Positive finite per-value scale, or ``None`` to use
                ``config.default_scale``.

        Returns:
            A new two-component ``[component, *batch, limb, coefficient]``
            ciphertext in coefficient domain and standard residues, with
            engine integral dtype/device, exact Q or QP ``prime_ids`` selected
            by the public key, unchanged level, and actual scale $\Delta$.

        Raises:
            InvalidScaleError: If ``scale`` is not positive and finite.
            ValueError: If the message, level, key, or direct-encode range is
                invalid.
        """

        level = self._validate_public_level(level)
        scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="Ciphertext",
        )
        return self._encryptor.encrypt_message(
            message,
            self._resolve_public_key(public_key),
            level=level,
            scale=scale,
        )

    def decrypt_message(
        self,
        ct: Ciphertext,
        secret_key: SecretKey | None = None,
        *,
        is_real: bool = False,
    ):
        r"""Decrypt and decode using the ciphertext's actual scale.

        $$
        m_{\mathrm{approx}}=
        \mathcal{E}_g\!\left(
          \sum_{j=0}^{d-1}c_j(X)s(X)^j\bmod B_\ell
        \right)/\Delta(ct),
        \qquad d\in\{2,3\}.
        $$

        Decryption uses the bounded approximate-coefficient reconstruction described
        by :meth:`decrypt`; it is not exact CRT. The result is CPU
        ``[*batch, slot]`` and is complex unless ``is_real=True``. The
        ciphertext and secret key are unchanged.
        """

        if not isinstance(ct, Ciphertext):
            raise TypeError(
                f"decrypt expects Ciphertext, got {type(ct).__name__}"
            )
        self._assert_engine_ciphertext(ct)
        return self._decryptor.decrypt_message(
            ct,
            self._resolve_secret_key(secret_key),
            is_real=is_real,
        )

    # ------------------------------------------------------------------
    # State transitions and arithmetic.
    # ------------------------------------------------------------------

    @overload
    def coefficient_domain_to_ntt_domain(
        self, value: Ciphertext
    ) -> Ciphertext: ...

    @overload
    def coefficient_domain_to_ntt_domain(
        self, value: Plaintext
    ) -> Plaintext: ...

    def coefficient_domain_to_ntt_domain(
        self, value: Ciphertext | Plaintext
    ) -> Ciphertext | Plaintext:
        r"""Return an RNS value after its negacyclic forward NTT.

        For every limb prime $q_i$, this preserves the polynomial ring element
        while changing the final tensor axis from coefficients to NTT
        evaluations. Plaintext axes are composable: input must already be
        coefficient-domain Montgomery RNS, and the transition is
        ``(coefficient, montgomery)`` to ``(ntt, montgomery)``. Ciphertext
        states are deliberately coupled: the transition is
        ``(coefficient, standard)`` to ``(ntt, montgomery)`` and includes
        standard-to-Montgomery conversion.

        Layout ``[*batch, limb, N]`` for plaintext or
        ``[component, *batch, limb, N]`` for ciphertext, engine integral
        dtype/device, level, actual scale, component count, Q/QP basis, and
        exact ``prime_ids`` are preserved. The functional result has
        independent storage. Input already in NTT domain is rejected because
        this is a strict source-to-target transition. No CRT reconstruction
        occurs.
        """

        self._assert_domain_transition_source(
            value, source_domain="coefficient"
        )
        return self._apply_forward_ntt_(value.clone())

    @overload
    def coefficient_domain_to_ntt_domain_(
        self, value: Ciphertext
    ) -> Ciphertext: ...

    @overload
    def coefficient_domain_to_ntt_domain_(
        self, value: Plaintext
    ) -> Plaintext: ...

    def coefficient_domain_to_ntt_domain_(
        self, value: Ciphertext | Plaintext
    ) -> Ciphertext | Plaintext:
        """Apply :meth:`coefficient_domain_to_ntt_domain` in place.

        The payload tensor is transformed in its existing storage and
        ``polynomial_domain`` is updated; ciphertext also changes
        ``residue_representation`` to ``"montgomery"``. Aliases observe both
        payload and metadata mutation. Input must be in coefficient domain.
        """

        self._assert_domain_transition_source(
            value, source_domain="coefficient"
        )
        return self._apply_forward_ntt_(value)

    @overload
    def ntt_domain_to_coefficient_domain(
        self, value: Ciphertext
    ) -> Ciphertext: ...

    @overload
    def ntt_domain_to_coefficient_domain(
        self, value: Plaintext
    ) -> Plaintext: ...

    def ntt_domain_to_coefficient_domain(
        self, value: Ciphertext | Plaintext
    ) -> Ciphertext | Plaintext:
        r"""Return an RNS value after its normalized inverse NTT.

        For every limb prime $q_i$, this preserves the RNS polynomial while
        changing NTT evaluations back to coefficients. Plaintext axes are
        composable: ``(ntt, montgomery)`` becomes
        ``(coefficient, montgomery)``. Ciphertext states are coupled:
        ``(ntt, montgomery)`` becomes ``(coefficient, standard)`` through
        inverse NTT and Montgomery reduction.

        Tensor shape, engine integral dtype/device, level, actual scale,
        component count, Q/QP basis, and exact ``prime_ids`` are preserved.
        The functional result owns independent storage. Input already in
        coefficient domain is rejected because this is a strict
        source-to-target transition. It remains RNS; no CRT reconstruction
        occurs.
        """

        self._assert_domain_transition_source(value, source_domain="ntt")
        return self._apply_inverse_ntt_(value.clone())

    @overload
    def ntt_domain_to_coefficient_domain_(
        self, value: Ciphertext
    ) -> Ciphertext: ...

    @overload
    def ntt_domain_to_coefficient_domain_(
        self, value: Plaintext
    ) -> Plaintext: ...

    def ntt_domain_to_coefficient_domain_(
        self, value: Ciphertext | Plaintext
    ) -> Ciphertext | Plaintext:
        """Apply :meth:`ntt_domain_to_coefficient_domain` in place; result remains RNS.

        The payload tensor is transformed in its existing storage and
        ``polynomial_domain`` is updated; ciphertext also changes
        ``residue_representation`` to ``"standard"``. Aliases observe payload
        and metadata mutation. Input must be in NTT domain.
        """

        self._assert_domain_transition_source(value, source_domain="ntt")
        return self._apply_inverse_ntt_(value)

    def standard_residues_to_montgomery_residues(
        self, plaintext: Plaintext
    ) -> Plaintext:
        r"""Convert coefficient-domain standard RNS to Montgomery residues.

        For limb prime $q_i$, map $r_{i,j}$ to $r_{i,j}R\bmod q_i$. Input and
        output layout is ``[*batch, limb, coefficient]`` with engine integral
        dtype/device. Representation, level, actual scale, Q/QP basis, and
        exact ``prime_ids`` are preserved. The functional output has
        independent storage. Input already using Montgomery residues is
        rejected because this is a strict source-to-target transition.
        """

        self._assert_plaintext_residue_transition_source(
            plaintext,
            source_residues="standard",
        )
        return self._to_montgomery_plaintext_(plaintext.clone())

    def standard_residues_to_montgomery_residues_(
        self, plaintext: Plaintext
    ) -> Plaintext:
        """Apply :meth:`standard_residues_to_montgomery_residues` in existing storage.

        ``residue_representation`` becomes ``"montgomery"`` and aliases
        observe payload/metadata mutation. Input must use standard residues.
        """

        self._assert_plaintext_residue_transition_source(
            plaintext,
            source_residues="standard",
        )
        return self._to_montgomery_plaintext_(plaintext)

    def montgomery_residues_to_standard_residues(
        self, plaintext: Plaintext
    ) -> Plaintext:
        r"""Convert coefficient-domain Montgomery RNS to standard residues.

        For limb prime $q_i$, Montgomery-reduce each residue to its ordinary
        representative. Input and output layout is
        ``[*batch, limb, coefficient]`` with engine integral dtype/device.
        Representation, level, actual scale, Q/QP basis, and exact
        ``prime_ids`` are preserved. The functional output has independent
        storage. Input already using standard residues is rejected because
        this is a strict source-to-target transition.
        """

        self._assert_plaintext_residue_transition_source(
            plaintext,
            source_residues="montgomery",
        )
        return self._to_standard_plaintext_(plaintext.clone())

    def montgomery_residues_to_standard_residues_(
        self, plaintext: Plaintext
    ) -> Plaintext:
        """Apply :meth:`montgomery_residues_to_standard_residues` in existing storage.

        ``residue_representation`` becomes ``"standard"`` and aliases observe
        payload/metadata mutation. Input must use Montgomery residues.
        """

        self._assert_plaintext_residue_transition_source(
            plaintext,
            source_residues="montgomery",
        )
        return self._to_standard_plaintext_(plaintext)

    def rescale_to_next_drop_prime(self, *, level: int) -> int:
        r"""Return the Q prime used by one rescale-to-next transition.

        For source level $\ell$ with ordered active Q-prime identifiers
        $I_\ell$, the result is the modulus $q_i$ for the leading identifier
        $i=I_\ell[0]$. :meth:`rescale_to_next_level` divides by this integer and
        removes its residue row.

        Args:
            level: Current public CKKS level. It must have a following public
                level, so the final legal public level is not accepted.

        Returns:
            The canonical leading active Q modulus at ``level``.

        Raises:
            TypeError: If ``level`` is not an integer.
            ValueError: If ``level`` is negative.
            MaximumLevelError: If no further public rescale level exists.
        """

        if type(level) is not int:
            raise TypeError("level must be an integer")
        if level < 0:
            raise ValueError(f"level must be non-negative; got {level}.")
        if level >= self.final_public_level:
            raise MaximumLevelError(
                level=level,
                maximum_level=self.final_public_level,
            )
        dropped_prime_id = self.rns_layout.prime_ids(level)[0]
        return int(self.montgomery_parameters.moduli[dropped_prime_id])

    def rescale_to_next_output_scale(
        self,
        input_scale: float,
        *,
        level: int,
    ) -> float:
        r"""Calculate the binary64 output scale of one rescale transition.

        The result is the same binary64 calculation used by
        :meth:`rescale_to_next_level`:

        $$
        \Delta_{\mathrm{out}}=
        \frac{\Delta_{\mathrm{in}}}{q_{\mathrm{drop}}}.
        $$

        Args:
            input_scale: Positive finite scale immediately before rescale.
            level: Current public CKKS level.

        Returns:
            The positive finite binary64 scale after dropping the level's
            leading active Q modulus.

        Raises:
            InvalidScaleError: If ``input_scale`` or the resulting quotient is
                not a positive finite binary64 value.
            TypeError: If ``level`` is not an integer.
            ValueError: If ``level`` is negative.
            MaximumLevelError: If no further public rescale level exists.
        """

        scale = coerce_scale(
            input_scale,
            value_name="rescale input",
        )
        modulus = self.rescale_to_next_drop_prime(level=level)
        return coerce_scale(
            scale / float(modulus),
            value_name="rescale result",
        )

    def rescale_to_next_level(
        self,
        ct: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        r"""Rescale a ciphertext to the next public CKKS level.

        For each component polynomial,

        $$
        c'=\operatorname{Round}\!\left(
          \frac{c}{q_{\mathrm{drop}}}
        \right)\pmod{B_{\ell+1}},
        \qquad
        \Delta(c')=\frac{\Delta(c)}{q_{\mathrm{drop}}}.
        $$

        For Q input, $B_{\ell+1}=Q_{\ell+1}$; for QP input,
        $B_{\ell+1}=Q_{\ell+1}P$ and every P row is retained.
        ``rounding="nearest"`` selects nearest-integer quotient and
        ``rounding="floor"`` subtracts the least nonnegative dropped residue
        before exact division.

        Args:
            ct: Full-layout coefficient-domain, standard-residue ciphertext at
                a non-final public level. Two- and three-component Q or QP
                values are accepted.
            rounding: Quotient rule, either ``"nearest"`` or ``"floor"``.

        Returns:
            A new coefficient-domain standard ciphertext at ``ct.level + 1``
            with the same two/three component count and Q/QP basis, engine
            integral dtype/device, exact ``prime_ids=ct.prime_ids[1:]``, and
            canonical residues in $[0,q_i)$. ``ct`` is unchanged and output
            storage does not alias it.

        Raises:
            MaximumLevelError: If ``ct`` is already at the final public level.
            InvalidScaleError: If the output scale is not positive and finite.
            ValueError: If the ciphertext state or engine layout is invalid.
        """

        return self._rescaler.rescale_to_next_level(ct, rounding=rounding)

    def rescale_to_next_level_(
        self,
        ct: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        r"""Advance one public level and update ``ct`` in place.

        Native kernels update the remaining RNS rows through views into the
        original allocation. The method then narrows ``ct.data`` and updates
        ``level``, ``prime_ids``, and the actual per-value scale.

        The mathematical quotient, scale transition, Q/QP row selection, and
        canonical standard-residue output are identical to
        :meth:`rescale_to_next_level`. Only storage ownership differs.

        Args:
            ct: Full-layout coefficient-domain, standard-residue ciphertext at
                a non-final public level. Aliases observe the mutation.
            rounding: Quotient rule, either ``"nearest"`` or ``"floor"``.

        Returns:
            ``ct`` itself after the rescale transition.

        Raises:
            MaximumLevelError: If ``ct`` is already at the final public level.
            InvalidScaleError: If the output scale is not positive and finite.
            ValueError: If the ciphertext state or engine layout is invalid.
        """

        return self._rescaler.rescale_to_next_level_(ct, rounding=rounding)

    def mod_switch_to_next_level(self, ct: Ciphertext) -> Ciphertext:
        r"""Restrict a ciphertext to the next public RNS basis.

        For every component,

        $$
        c'=c\pmod{B_{\ell+1}},\qquad
        \Delta(c')=\Delta(c).
        $$

        This transition drops the leading Q row, preserves coefficient
        representatives and scale, and retains all P rows for QP input.
        Message preservation requires the represented centered value to remain
        within the smaller modulus.

        Args:
            ct: Full-layout ciphertext at a non-final public CKKS level. Its
                component count, polynomial domain, representation, and Q/QP
                basis are preserved.

        Returns:
            A new ciphertext at ``ct.level + 1`` with unchanged component
            count, domain, basis, residue form, dtype/device, actual scale, and
            ``prime_ids=ct.prime_ids[1:]``. ``ct`` is unchanged and storage is
            independent.

        Raises:
            MaximumLevelError: If ``ct`` is already at the final public level.
            ValueError: If ``ct`` is incompatible with this engine or does not
                contain the complete active RNS layout.
        """

        self._assert_engine_ciphertext(ct)
        if ct.level >= self.final_public_level:
            raise MaximumLevelError(
                level=ct.level,
                maximum_level=self.final_public_level,
            )
        return self._copy_at_level(ct, ct.level + 1)

    def mod_switch_to_next_level_(self, ct: Ciphertext) -> Ciphertext:
        """Restrict ``ct`` to the next public RNS basis in place.

        The mathematical RNS restriction and no-wrap condition are identical
        to :meth:`mod_switch_to_next_level`. ``ct.data`` is narrowed to a view of its
        existing allocation; level and exact ``prime_ids`` change, while
        actual scale and all other state axes are preserved. Aliases observe
        metadata mutation and retained storage rows.

        Args:
            ct: Full-layout ciphertext at a non-final public level. Aliases
                observe its narrowed tensor and updated level metadata.

        Returns:
            ``ct`` itself with unchanged scale and one fewer active Q row.

        Raises:
            MaximumLevelError: If ``ct`` is already at the final public level.
            ValueError: If ``ct`` is incompatible with this engine or does not
                contain the complete active RNS layout.
        """

        self._assert_engine_ciphertext(ct)
        if ct.level >= self.final_public_level:
            raise MaximumLevelError(
                level=ct.level,
                maximum_level=self.final_public_level,
            )
        return self._restrict_to_level_(ct, ct.level + 1)

    def mod_switch_to_level(
        self,
        ct: Ciphertext,
        target_level: int,
    ) -> Ciphertext:
        r"""Restrict a ciphertext to the RNS basis at ``target_level``.

        For target level $t\geq\ell$,

        $$
        c'=c\pmod{B_t},\qquad \Delta(c')=\Delta(c).
        $$

        The transition drops the first $t-\ell$ active Q rows, preserves
        coefficient representatives and scale, and retains P rows for QP
        input. Message preservation requires the represented centered value to
        remain within $B_t$.

        Args:
            ct: Full-layout ciphertext whose current level is no later than
                ``target_level``.
            target_level: Destination public level in the inclusive range
                ``[ct.level, final_public_level]``.

        Returns:
            A new ciphertext at ``target_level`` with unchanged component
            count, domain, Q/QP basis, residue form, dtype/device, and actual
            scale. Exact ``prime_ids`` are restricted accordingly. Passing the
            current level returns a full clone; output never aliases ``ct``.

        Raises:
            TypeError: If ``target_level`` is not an integer.
            ValueError: If the target is earlier than ``ct.level``, beyond the
                final public level, or the ciphertext layout is incompatible.
        """

        self._validate_mod_switch_target(ct, target_level)
        return self._copy_at_level(ct, target_level)

    def mod_switch_to_level_(
        self,
        ct: Ciphertext,
        target_level: int,
    ) -> Ciphertext:
        r"""In-place form of :meth:`mod_switch_to_level`.

        The same RNS restriction and no-wrap condition apply. The narrowed
        tensor remains a view into the original allocation; level and exact
        ``prime_ids`` change while actual scale and other state axes are
        preserved. Aliases observe mutation.

        Args:
            ct: Full-layout ciphertext to mutate.
            target_level: Destination public level in the inclusive range
                ``[ct.level, final_public_level]``.

        Returns:
            ``ct`` itself with unchanged scale. Passing its current level is a
            no-op.

        Raises:
            TypeError: If ``target_level`` is not an integer.
            ValueError: If the target or ciphertext layout is invalid.
        """

        self._validate_mod_switch_target(ct, target_level)
        return self._restrict_to_level_(ct, target_level)

    def reinterpret_at_scale(
        self,
        ct: Ciphertext,
        target_scale: float,
        *,
        max_relative_change: float | None = None,
    ) -> Ciphertext:
        r"""Return a metadata-only reinterpretation at ``target_scale``.

        Ciphertext residues are not modified. Consequently the decoded message
        obeys

        $$
        c'=c,\qquad
        \Delta(c')=\Delta_{\mathrm{target}},\qquad
        m'=m\frac{\Delta(c)}{\Delta_{\mathrm{target}}}.
        $$

        When supplied, ``max_relative_change`` bounds the symmetric ratio
        between the current and target scales.

        Args:
            ct: Ciphertext whose payload and original metadata remain
                unchanged.
            target_scale: Positive finite scale used to reinterpret the same
                residues.
            max_relative_change: Optional upper bound on the symmetric ratio
                $\max(\Delta_{\mathrm{old}}/\Delta_{\mathrm{target}},
                \Delta_{\mathrm{target}}/\Delta_{\mathrm{old}})-1$.

        Returns:
            A new ciphertext with cloned payload, independent storage, and
            ``target_scale``. Level, component count, domain, basis, residue
            form, dtype/device, and exact ``prime_ids`` are unchanged.

        Raises:
            InvalidScaleError: If ``target_scale`` is invalid.
            ScaleMismatchError: If the provided relative-change bound is
                exceeded.
            ValueError: If the bound itself or ciphertext state is invalid.
        """

        self._assert_local_ciphertext(ct)
        result = ct.clone()
        return self._set_reinterpreted_scale_(
            result,
            target_scale,
            max_relative_change=max_relative_change,
        )

    def reinterpret_at_scale_(
        self,
        ct: Ciphertext,
        target_scale: float,
        *,
        max_relative_change: float | None = None,
    ) -> Ciphertext:
        r"""Reinterpret ``ct`` at ``target_scale`` in place.

        The equations and guard are identical to :meth:`reinterpret_at_scale`.
        Only ``ct.scale`` is mutated; payload storage and every other state axis
        remain unchanged. Aliases to the object observe the metadata change.

        Args:
            ct: Ciphertext whose scale metadata is mutated; residues are not.
            target_scale: Positive finite replacement scale.
            max_relative_change: Optional symmetric relative-change bound.

        Returns:
            ``ct`` itself after changing only its scale metadata.

        Raises:
            InvalidScaleError: If ``target_scale`` is invalid.
            ScaleMismatchError: If the provided relative-change bound is
                exceeded.
            ValueError: If the bound itself or ciphertext state is invalid.
        """

        self._assert_local_ciphertext(ct)
        return self._set_reinterpreted_scale_(
            ct,
            target_scale,
            max_relative_change=max_relative_change,
        )

    def zero_plaintext_like(self, plaintext: Plaintext) -> Plaintext:
        r"""Construct semantic zero in the same exact plaintext state.

        Slots, integer coefficients, approximate coefficients, or RNS payloads
        are materialized as zero with matching batch shape, level, actual
        scale, representation, domain, Q/QP basis, residue form, and exact
        ``prime_ids``. The result uses the engine-compatible dtype/device for
        newly encoded storage and owns independent storage. The input is not
        mutated.
        """

        self._validate_public_level(plaintext.level)
        if plaintext.context_id not in (None, self.context.context_id):
            raise ValueError("Plaintext belongs to another CKKS context")
        if plaintext.message is None:
            message = torch.zeros(
                (*plaintext.batch_shape, self.num_slots),
                dtype=torch.complex128,
                device=self.device,
            )
        else:
            message = torch.zeros_like(plaintext.message)
        if plaintext.is_slots:
            return Plaintext(
                message=message,
                level=plaintext.level,
                scale=plaintext.scale,
            )
        if plaintext.is_approximate_coefficients:
            assert plaintext.data is not None
            result = plaintext.clone()
            assert result.data is not None
            result.data.zero_()
            return result
        encoded = self.encode(
            message,
            level=plaintext.level,
            scale=plaintext.scale,
        )
        if plaintext.is_integer_coefficients:
            return encoded
        if not plaintext.is_rns or plaintext.modulus_basis is None:
            raise ValueError(
                "Plaintext has no supported materialized representation"
            )
        if plaintext.residue_representation == "standard":
            return self.integer_coefficients_to_rns(
                encoded,
                modulus_basis=plaintext.modulus_basis,
            )
        if plaintext.polynomial_domain == "coefficient":
            return self.prepare_plaintext_for_addition(
                encoded,
                modulus_basis=plaintext.modulus_basis,
            )
        return self.prepare_plaintext_for_multiplication(
            encoded,
            modulus_basis=plaintext.modulus_basis,
        )

    def encrypt_zero_like(
        self,
        ct: Ciphertext,
        public_key: PublicKey | None = None,
    ) -> Ciphertext:
        r"""Create a randomized secure encryption of semantic zero.

        The output phase satisfies
        $c_0(X)+c_1(X)s(X)=e(X)\pmod{B_\ell}$ and decodes to
        zero up to encryption error. It is a new two-component ciphertext with
        ``ct``'s batch shape, level, actual scale, coefficient/NTT domain,
        Q/QP basis, standard/Montgomery form, engine dtype/device, and exact
        ``prime_ids``. The supplied public key must select the same basis.
        Neither input nor key is mutated and no storage aliases them.
        """

        self._assert_engine_ciphertext(ct)
        zero = torch.zeros(
            (*ct.batch_shape, self.num_slots),
            dtype=torch.complex128,
            device=self.device,
        )
        result = self.encrypt_message(
            zero,
            public_key,
            level=ct.level,
            scale=ct.scale,
        )
        if result.modulus_basis != ct.modulus_basis:
            raise ValueError(
                "encrypt_zero_like public key basis does not match the "
                f"reference Ciphertext basis: {result.modulus_basis!r} != {ct.modulus_basis!r}"
            )
        return (
            self.coefficient_domain_to_ntt_domain_(result)
            if ct.is_ntt_domain
            else result
        )

    def multiply(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
    ) -> Ciphertext:
        r"""Multiply two NTT ciphertexts without relinearization or rescale.

        Both inputs must be two-component NTT values, and the output is a
        three-component NTT value.  Programs call
        :meth:`relinearize` and :meth:`rescale_to_next_level` as separate calls. This also
        makes communication required by a limb-parallel implementation
        visible.

        For $a(X)=a_0+a_1s$ and $b(X)=b_0+b_1s$, component convolution gives

        $$
        d_0=a_0b_0,\qquad
        d_1=a_0b_1+a_1b_0,\qquad
        d_2=a_1b_1\pmod{B_\ell},
        $$

        with $\Delta(d)=\Delta(a)\Delta(b)$.

        Args:
            lhs: Two-component NTT/Montgomery ciphertext.
            rhs: Layout-compatible two-component NTT/Montgomery ciphertext.
                Its scale may differ from ``lhs.scale``.

        Returns:
            A new three-component NTT/Montgomery ciphertext at unchanged level
            and Q/QP basis, with engine integral dtype/device, the exact shared
            ``prime_ids``, the operands' batch shape, and product actual scale.
            Inputs are unchanged and output storage is independent.

        Raises:
            InvalidScaleError: If the binary64 product scale is not positive
                and finite.
            ValueError: If either input state or the shared layout is invalid.
        """

        lhs.assert_state(
            polynomial_domain="ntt",
            residue_representation="montgomery",
            components=2,
        )
        rhs.assert_state(
            polynomial_domain="ntt",
            residue_representation="montgomery",
            components=2,
        )
        self._assert_same_cipher_layout(lhs, rhs)
        product_scale = coerce_scale(
            lhs.scale * rhs.scale,
            value_name="multiply result",
        )
        d0 = self.rns_runtime.montgomery_mul(
            lhs.c0, rhs.c0, prime_ids=lhs.prime_ids
        )
        x0y1 = self.rns_runtime.montgomery_mul(
            lhs.c0, rhs.c1, prime_ids=lhs.prime_ids
        )
        x1y0 = self.rns_runtime.montgomery_mul(
            lhs.c1, rhs.c0, prime_ids=lhs.prime_ids
        )
        d1 = self.rns_runtime.add_lazy(x0y1, x1y0, prime_ids=lhs.prime_ids)
        d2 = self.rns_runtime.montgomery_mul(
            lhs.c1, rhs.c1, prime_ids=lhs.prime_ids
        )
        return self._ciphertext_from_components(
            (d0, d1, d2),
            level=lhs.level,
            scale=product_scale,
            polynomial_domain="ntt",
            modulus_basis=lhs.modulus_basis,
            residue_representation="montgomery",
            prime_ids=lhs.prime_ids,
        )

    def relinearize(
        self,
        ct: Ciphertext,
        relinearization_key: RelinearizationKey | None = None,
    ) -> Ciphertext:
        r"""Key-switch a three-component product back to two components.

        Relinearization transforms the phase relation

        $$
        d_0(X)+d_1(X)s(X)+d_2(X)s(X)^2
        \longmapsto
        c'_0(X)+c'_1(X)s(X)
        $$

        using QP material directed from $s(X)^2$ to $s(X)$. It preserves the
        represented message, level, and actual scale up to configured
        key-switch error.

        Args:
            ct: Three-component Q-basis NTT/Montgomery ciphertext.
            relinearization_key: Compatible QP key. The engine-installed key is
                used when this argument is omitted.

        Returns:
            A new two-component coefficient-domain standard ciphertext with
            Q basis, the same batch shape, engine integral dtype/device, exact
            active Q ``prime_ids``, level, and actual scale as ``ct``. Inputs
            are unchanged and output storage is independent.

        Raises:
            ValueError: If the ciphertext or key state is incompatible.
        """

        ct.assert_state(
            polynomial_domain="ntt",
            modulus_basis="Q",
            residue_representation="montgomery",
            components=3,
        )
        self._assert_engine_ciphertext(ct)
        relinearization_key = relinearization_key or self.relinearization_key
        self._assert_engine_key(
            relinearization_key,
            expected_type=RelinearizationKey,
            modulus_basis="QP",
        )
        d0 = ct.c0.clone()
        d1 = ct.c1.clone()
        d2 = ct.c2.clone()
        self.rns_runtime.inverse_to_standard_(d0)
        self.rns_runtime.inverse_to_standard_(d1)
        self.rns_runtime.inverse_to_standard_(d2)
        correction0, correction1 = self._hybrid_key_switcher.apply_key_switch(
            d2, relinearization_key, ct.level
        )
        d0 = d0 + correction0
        d1 = d1 + correction1
        self.rns_runtime.canonicalize_residues_(d0)
        self.rns_runtime.canonicalize_residues_(d1)
        return self._ciphertext_from_components(
            [d0, d1],
            level=ct.level,
            scale=ct.scale,
            polynomial_domain="coefficient",
            modulus_basis=ct.modulus_basis,
            residue_representation="standard",
        )

    def switch_key(self, ct: Ciphertext, key: KeySwitchKey) -> Ciphertext:
        r"""Switch a two-component ciphertext from source to destination key.

        ``key`` must have been generated for the declared direction

        $$
        c_0(X)+c_1(X)s_{\mathrm{source}}(X)
        \longmapsto
        c'_0(X)+c'_1(X)s_{\mathrm{destination}}(X).
        $$

        The key object does not self-identify these lineages; the caller owns
        that invariant. The represented message, level, and actual scale are
        preserved up to key-switch error. Input must be coefficient-domain
        standard, two-component, full-layout Q RNS. Output is a new Q value
        with unchanged batch shape, engine dtype/device, and exact active Q
        ``prime_ids``; inputs are not mutated and storage is independent.
        """

        ct.assert_state(
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="standard",
            components=2,
        )
        self._assert_engine_ciphertext(ct)
        self._assert_engine_key(
            key, expected_type=KeySwitchKey, modulus_basis="QP"
        )
        return self._apply_key_switch(ct, key)

    def add(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Add ciphertexts with exactly equal scale and arithmetic state.

        The method performs no level alignment, scale reinterpretation,
        relinearization, or domain conversion.

        $$
        c'_j=c_{\mathrm{lhs},j}+c_{\mathrm{rhs},j}\pmod{B_\ell},
        \qquad
        \Delta(c')=\Delta(c_{\mathrm{lhs}})=\Delta(c_{\mathrm{rhs}}).
        $$

        Args:
            lhs: Left ciphertext and, when ``inplace`` is true, destination.
            rhs: Ciphertext with identical layout, shape, and binary64 scale.
            inplace: Mutate and return ``lhs`` when true; otherwise return a
                new ciphertext and leave both inputs unchanged.

        Returns:
            The sum with unchanged component count, level, domain, Q/QP basis,
            residue form, engine dtype/device, and exact ``prime_ids``. The
            functional result owns independent storage; ``inplace=True``
            mutates and returns ``lhs`` so aliases observe canonicalized
            payload changes.

        Raises:
            ScaleMismatchError: If ``lhs.scale != rhs.scale``.
            ValueError: If context, level, state, shape, or RNS layout differs.
        """

        self._assert_same_cipher_layout(
            lhs,
            rhs,
            require_same_scale=True,
            operation="add",
        )
        return self._add_ciphertext_payloads(lhs, rhs, inplace=inplace)

    def add_(self, lhs: Ciphertext, rhs: Ciphertext) -> Ciphertext:
        """Add ``rhs`` to ``lhs`` in place with :meth:`add`'s state requirements.

        Only ``lhs`` payload storage is mutated; all metadata is preserved and
        aliases of ``lhs`` observe the component-wise modular sum.
        """

        return self.add(lhs, rhs, inplace=True)

    def sum_ciphertexts(self, ciphertexts: Sequence[Ciphertext]) -> Ciphertext:
        r"""Return the rank-local sum of compatible ciphertexts.

        For each component $j$,

        $$
        c'_j=\sum_i c_{i,j}\pmod{B_\ell}
        $$

        with one exactly shared actual scale. Every state axis, batch shape,
        dtype/device, and exact ``prime_ids`` must match.

        Args:
            ciphertexts: Non-empty sequence satisfying :meth:`add`'s exact
                layout and exact-scale requirements.

        Returns:
            A new ciphertext. Every input remains unchanged.

        Raises:
            ScaleMismatchError: If any scale differs from the first value.
            ValueError: If the sequence is empty or another value is layout
                incompatible.
        """

        if not ciphertexts:
            raise ValueError(
                "sum_ciphertexts requires at least one value; construct an "
                "transparent or encrypted zero when an empty sum is intended"
            )
        first = ciphertexts[0]
        self._assert_local_ciphertext(first)
        result = first.clone()
        for ciphertext in ciphertexts[1:]:
            self._assert_local_ciphertext(ciphertext)
            self._assert_matching_cipher_layout(
                first,
                ciphertext,
                require_same_scale=True,
                operation="add",
            )
            self._add_ciphertext_payloads(result, ciphertext, inplace=True)
        return result

    # Batch reduction complements sum_ciphertexts: callers that have already
    # exposed homogeneous logical work as one tensor keep that work batched
    # through each native addition round instead of unbinding it into Python.
    def sum_ciphertext_batch(
        self,
        batch: Ciphertext,
        *,
        dim: int = 0,
    ) -> Ciphertext:
        r"""Reduce one logical batch axis by modular ciphertext addition.

        A binary tree adds contiguous halves through the native batched RNS
        kernels. This exposes the complete remaining batch, component, limb,
        and polynomial-index work to each round instead of launching one
        addition per selected value. Odd tails are folded after the batched
        rounds.

        Args:
            batch: Ciphertext with at least one logical batch axis.
            dim: Logical batch axis to remove. Ciphertext component and RNS
                axes are not addressable through this argument.

        Returns:
            A new ciphertext with the selected batch axis removed. Level,
            scale, polynomial domain, modulus basis, residue representation,
            component count, device, dtype, and exact ``prime_ids`` are
            preserved. The input is unchanged.

        Raises:
            IndexError: If ``dim`` does not identify a logical batch axis.
            ValueError: If ``batch`` is unbatched or incompatible with this
                engine.
        """

        self._assert_local_ciphertext(batch)
        if not batch.is_batched:
            raise ValueError("sum_ciphertext_batch requires a batched value")
        logical_dim = dim if dim >= 0 else dim + len(batch.batch_shape)
        if not 0 <= logical_dim < len(batch.batch_shape):
            raise IndexError(
                f"Batch dimension {dim} is outside shape "
                f"{tuple(batch.batch_shape)}"
            )
        current = batch
        odd_tails: list[Ciphertext] = []
        while current.batch_shape[logical_dim] > 1:
            count = current.batch_shape[logical_dim]
            pair_count = count // 2
            if count % 2:
                odd_tails.append(
                    current.select_batch(count - 1, dim=logical_dim)
                )
            data_dim = logical_dim + 1
            lhs_data = current.data.narrow(data_dim, 0, pair_count)
            rhs_data = current.data.narrow(data_dim, pair_count, pair_count)
            current = self.add(
                current.with_data(lhs_data),
                current.with_data(rhs_data),
            )
        result = current.select_batch(0, dim=logical_dim)
        if current is batch:
            result = result.clone()
        for tail in reversed(odd_tails):
            self.add_(result, tail)
        return result

    def subtract(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Subtract ciphertexts with exactly equal scale and state.

        $$
        c'_j=c_{\mathrm{lhs},j}-c_{\mathrm{rhs},j}\pmod{B_\ell},
        \qquad
        \Delta(c')=\Delta(c_{\mathrm{lhs}})=\Delta(c_{\mathrm{rhs}}).
        $$

        Args:
            lhs: Minuend and, when ``inplace`` is true, destination.
            rhs: Subtrahend with identical layout, shape, and binary64 scale.
            inplace: Mutate and return ``lhs`` when true.

        Returns:
            The difference with unchanged component count, level, domain,
            Q/QP basis, residue form, engine dtype/device, and exact
            ``prime_ids``. The functional result owns independent storage;
            ``inplace=True`` mutates and returns ``lhs``.

        Raises:
            ScaleMismatchError: If ``lhs.scale != rhs.scale``.
            ValueError: If context, level, state, shape, or RNS layout differs.
        """

        self._assert_same_cipher_layout(
            lhs,
            rhs,
            require_same_scale=True,
            operation="subtract",
        )
        return self._subtract_ciphertext_payloads(lhs, rhs, inplace=inplace)

    def subtract_(self, lhs: Ciphertext, rhs: Ciphertext) -> Ciphertext:
        """Subtract ``rhs`` from ``lhs`` in place under :meth:`subtract`.

        Only ``lhs`` payload storage is mutated; all metadata is preserved and
        aliases observe the component-wise modular difference.
        """

        return self.subtract(lhs, rhs, inplace=True)

    def negate(self, ct: Ciphertext, *, inplace: bool = False) -> Ciphertext:
        r"""Negate a ciphertext without changing its state metadata.

        $$
        c'_j=-c_j\pmod{B_\ell},\qquad \Delta(c')=\Delta(c).
        $$

        Component count, level, domain, Q/QP basis, residue form,
        dtype/device, and exact ``prime_ids`` are preserved. The functional
        result owns independent storage; ``inplace=True`` mutates canonicalized
        residues in ``ct`` and aliases observe the change.
        """

        self._assert_engine_ciphertext(ct)
        if inplace:
            for component_id in range(ct.component_count):
                component = ct.component(component_id)
                try:
                    component.view(-1, component.size(-2), component.size(-1))
                except RuntimeError as error:
                    raise ValueError(
                        "In-place negate requires a zero-copy collapsible "
                        "dense batch prefix"
                    ) from error
        result = ct if inplace else ct.clone()
        result.data.neg_()
        active_moduli = self.rns_runtime.moduli[
            result.prime_ids[0] : result.prime_ids[-1] + 1
        ]
        for component_id in range(result.component_count):
            component = result.component(component_id)
            component.remainder_(
                active_moduli.view(
                    *([1] * (component.ndim - 2)),
                    active_moduli.numel(),
                    1,
                )
            )
        return result

    def negate_(self, ct: Ciphertext) -> Ciphertext:
        """Negate ``ct`` in place with :meth:`negate`'s state requirements."""

        return self.negate(ct, inplace=True)

    def add_plaintext(
        self,
        ct: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Add an operation-ready plaintext at exactly the ciphertext scale.

        $$
        c'_0=c_0+p\pmod{B_\ell},\qquad
        c'_j=c_j\quad(j>0),\qquad
        \Delta(c')=\Delta(c)=\Delta(p).
        $$

        No scale or level alignment is implicit.

        Args:
            ct: Two-component coefficient-domain standard ciphertext.
            plaintext: Coefficient-domain RNS ``Plaintext`` or compatible
                compressed form at the same level, basis, rows, and exact
                binary64 scale. A batched plaintext must match
                ``ct.batch_shape`` exactly; an unbatched plaintext broadcasts
                over every ciphertext batch entry.
            inplace: Update only ``ct.c0`` and return ``ct`` when true.

        Returns:
            A two-component coefficient-domain standard ciphertext with
            unchanged level, Q/QP basis, batch shape, engine dtype/device,
            exact ``prime_ids``, and shared actual scale. Functional mode
            allocates independent output; ``inplace=True`` mutates only
            ``ct.c0`` and aliases observe the change.

        Raises:
            ScaleMismatchError: If the two scale values are not exactly equal.
            ValueError: If the plaintext or ciphertext arithmetic state,
                context, layout, device, or batch shape is incompatible.
        """

        ct.assert_state(
            polynomial_domain="coefficient",
            residue_representation="standard",
            components=2,
        )
        self._assert_engine_ciphertext(ct)
        if plaintext.scale != ct.scale:
            raise ScaleMismatchError(
                operation="add_plaintext",
                lhs_name="ciphertext",
                lhs_scale=ct.scale,
                rhs_name="plaintext",
                rhs_scale=plaintext.scale,
            )
        if isinstance(plaintext, CompressedPlaintext):
            pt_data = self._prepare_compressed_plaintext_operand(
                plaintext, ct, polynomial_domain="coefficient"
            )
        else:
            pt_data = self._prepare_plaintext_operand(
                plaintext, ct, polynomial_domain="coefficient"
            )
        params = self.rns_runtime.rns_parameters_for(
            ct.c0, include_p=ct.includes_p
        )
        if inplace:
            if isinstance(plaintext, CompressedPlaintext):
                self._add_compressed_plaintext_component(
                    ct.c0,
                    plaintext,
                    pt_data,
                    params,
                    inplace=True,
                )
            else:
                ckks_ops.add_prepared_plaintext_component_(
                    ct.c0,
                    pt_data,
                    params,
                )
            return ct
        if isinstance(plaintext, CompressedPlaintext):
            c0 = self._add_compressed_plaintext_component(
                ct.c0,
                plaintext,
                pt_data,
                params,
                inplace=False,
            )
        else:
            c0 = ckks_ops.add_prepared_plaintext_component(
                ct.c0,
                pt_data,
                params,
            )
        result = self._ciphertext_from_components(
            [c0, ct.c1.clone()],
            level=ct.level,
            scale=ct.scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
        )
        return result

    def add_plaintext_(
        self,
        ct: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
    ) -> Ciphertext:
        """Mutate ``ct.c0`` with :meth:`add_plaintext`'s exact input requirements."""

        return self.add_plaintext(ct, plaintext, inplace=True)

    def multiply_plaintext(
        self,
        ct: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
        *,
        inplace: bool = False,
    ) -> Ciphertext:
        r"""Multiply an NTT ciphertext by an operation-ready plaintext.

        The operands need not have equal scales. This operation preserves the
        ciphertext level and records their binary64 scale product.

        $$
        c'_j=c_jp\pmod{B_\ell},\qquad
        \Delta(c')=\Delta(c)\Delta(p).
        $$

        Both operands remain in NTT/Montgomery representation. Callers place
        :meth:`coefficient_domain_to_ntt_domain` before a multiplication region
        and :meth:`ntt_domain_to_coefficient_domain` before an operation that
        requires coefficient-domain standard residues, such as rescale.

        Args:
            ct: Two-component NTT-domain Montgomery ciphertext.
            plaintext: NTT/Montgomery RNS ``Plaintext`` or compatible
                compressed form at the same context, level, basis, rows, and
                batch layout. A batched plaintext must match
                ``ct.batch_shape`` exactly; an unbatched plaintext broadcasts
                over every ciphertext batch entry.
            inplace: Replace ``ct`` with the result and return it when true.

        Returns:
            A new two-component NTT-domain Montgomery ciphertext at
            unchanged level and Q/QP basis, with the ciphertext batch shape,
            engine integral dtype/device, exact ``prime_ids``, and product
            actual scale. Functional mode leaves inputs unchanged and owns
            independent storage; ``inplace=True`` replaces all ``ct`` state
            with the new storage, so old tensor aliases keep the old allocation.

        Raises:
            InvalidScaleError: If the product scale is not positive and finite.
            ValueError: If the plaintext or ciphertext arithmetic state,
                context, layout, device, or batch shape is incompatible.
        """

        ct.assert_state(
            polynomial_domain="ntt",
            residue_representation="montgomery",
            components=2,
        )
        self._assert_engine_ciphertext(ct)
        if isinstance(plaintext, CompressedPlaintext):
            pt_ntt = self._prepare_compressed_plaintext_operand(
                plaintext, ct, polynomial_domain="ntt"
            )
        else:
            pt_ntt = self._prepare_plaintext_operand(
                plaintext, ct, polynomial_domain="ntt"
            )
        product_scale = coerce_scale(
            ct.scale * plaintext.scale,
            value_name="multiply_plaintext result",
        )
        if isinstance(plaintext, CompressedPlaintext):
            c0 = self._multiply_compressed_plaintext_component(
                ct.c0,
                plaintext,
                pt_ntt,
                prime_ids=ct.prime_ids,
            )
            c1 = self._multiply_compressed_plaintext_component(
                ct.c1,
                plaintext,
                pt_ntt,
                prime_ids=ct.prime_ids,
            )
        else:
            c0 = self.rns_runtime.montgomery_mul(
                ct.c0, pt_ntt, prime_ids=ct.prime_ids
            )
            c1 = self.rns_runtime.montgomery_mul(
                ct.c1, pt_ntt, prime_ids=ct.prime_ids
            )
        result = self._ciphertext_from_components(
            [c0, c1],
            level=ct.level,
            scale=product_scale,
            polynomial_domain="ntt",
            modulus_basis=ct.modulus_basis,
            residue_representation="montgomery",
        )
        return ct.replace_(result) if inplace else result

    def multiply_plaintext_(
        self,
        ct: Ciphertext,
        plaintext: Plaintext | CompressedPlaintext,
    ) -> Ciphertext:
        """Replace ``ct`` by its product under :meth:`multiply_plaintext`."""

        return self.multiply_plaintext(ct, plaintext, inplace=True)

    # ------------------------------------------------------------------
    # Rotation and true NTT-domain hoisting.
    # ------------------------------------------------------------------

    def rotate_by_step(
        self,
        ct: Ciphertext,
        rotation_step: int,
    ) -> Ciphertext:
        r"""Rotate by one signed step using the engine-owned key inventory.

        For $r=\mathtt{rotation\_step}$ and $S=N/2$, the result satisfies

        $$
        m'_j=m_{(j-r)\bmod S}.
        $$

        The step is canonicalized modulo $S$. A matching direct key is used
        when installed, generated lazily when allowed, or composed from the
        installed key inventory. Use :meth:`rotate_with_key` when the caller
        owns the exact direct key.
        """

        if not isinstance(rotation_step, Integral):
            raise TypeError("rotation_step must be an integer")
        self._assert_rotation_ciphertext(ct)
        canonical_step = self._canonical_rotation_step(int(rotation_step))
        path = self._rotation_path_for_step(canonical_step, {})
        return self._apply_rotation_path(ct, path)

    def rotate_with_key(
        self,
        ct: Ciphertext,
        key: RotationKey,
    ) -> Ciphertext:
        r"""Rotate slots using the canonical signed step carried by ``key``.

        For $r=\mathtt{key.rotation_step}$ and $S=N/2$,

        $$
        m'_j=m_{(j-r)\bmod S},
        $$

        matching ``torch.roll(m, shifts=r)``. The backend applies the Galois
        automorphism $\sigma_g$ to both components, then uses ``key`` in the
        direction $\sigma_g(s(X))\longmapsto s(X)$. ``rotation_step`` is a slot
        displacement and is not the Galois element.

        Input must be a two-component coefficient-domain standard full-layout
        Q ciphertext. Output is a new Q ciphertext with unchanged batch shape,
        level, actual scale, engine integral dtype/device, and exact active Q
        ``prime_ids``; key switching adds its configured approximation error.
        Inputs are unchanged and output storage is independent. Step zero
        returns a clone without automorphism or key switch.
        """

        self._assert_rotation_ciphertext(ct)
        self._assert_engine_key(
            key, expected_type=RotationKey, modulus_basis="QP"
        )
        rotation_step = self._canonical_rotation_step(key.rotation_step)
        if rotation_step != key.rotation_step:
            raise ValueError(
                "Rotation key must carry a canonical signed rotation step: "
                f"key={key.rotation_step}, canonical={rotation_step}"
            )
        return self._apply_rotation_key(ct, key, rotation_step)

    def rotate_many_by_steps(
        self,
        ct: Ciphertext,
        rotation_steps: Sequence[int],
        *,
        use_hoisting: bool = True,
    ) -> list[Ciphertext]:
        r"""Rotate by ordered signed steps using engine-owned keys.

        Output order matches ``rotation_steps``. Every result owns independent
        storage, including duplicate and zero-step entries. Direct-key paths
        share hybrid decomposition when ``use_hoisting`` is true; a step that
        requires a composed key path is evaluated independently.
        """

        self._assert_rotation_ciphertext(ct)
        canonical_steps: list[int] = []
        for rotation_step in rotation_steps:
            if not isinstance(rotation_step, Integral):
                raise TypeError("All rotation steps must be integers")
            canonical_steps.append(
                self._canonical_rotation_step(int(rotation_step))
            )
        resolved: dict[int, RotationKey] = {}
        paths = [
            self._rotation_path_for_step(step, resolved)
            for step in canonical_steps
        ]
        if not use_hoisting:
            return [self._apply_rotation_path(ct, path) for path in paths]

        direct_indices: list[int] = []
        direct_entries: list[tuple[int, RotationKey | None]] = []
        outputs: dict[int, Ciphertext] = {}
        for index, (step, path) in enumerate(
            zip(canonical_steps, paths, strict=True)
        ):
            if len(path) <= 1:
                direct_indices.append(index)
                direct_entries.append((step, None if not path else path[0][1]))
            else:
                outputs[index] = self._apply_rotation_path(ct, path)

        direct_outputs = self._apply_rotation_key_sequence(
            ct,
            direct_entries,
            use_hoisting=True,
        )
        outputs.update(zip(direct_indices, direct_outputs, strict=True))
        return [outputs[index] for index in range(len(paths))]

    def rotate_many_with_keys(
        self,
        ct: Ciphertext,
        rotation_keys: Sequence[RotationKey],
        *,
        use_hoisting: bool = True,
    ) -> list[Ciphertext]:
        r"""Rotate once for each ordered caller-owned direct key.

        Every key self-describes its canonical signed step. Output order and
        duplicates match ``rotation_keys``; results own independent storage.
        The keys are not installed into the engine.
        """

        self._assert_rotation_ciphertext(ct)
        entries: list[tuple[int, RotationKey | None]] = []
        for key in rotation_keys:
            self._assert_engine_key(
                key,
                expected_type=RotationKey,
                modulus_basis="QP",
            )
            rotation_step = self._canonical_rotation_step(key.rotation_step)
            if rotation_step != key.rotation_step:
                raise ValueError(
                    "Rotation key must carry a canonical signed rotation "
                    f"step: key={key.rotation_step}, canonical={rotation_step}"
                )
            entries.append((rotation_step, key))
        return self._apply_rotation_key_sequence(
            ct,
            entries,
            use_hoisting=use_hoisting,
        )

    def rotate_by_step_(
        self,
        ct: Ciphertext,
        rotation_step: int,
    ) -> Ciphertext:
        r"""Replace ``ct`` by :meth:`rotate_by_step`'s result.

        The result satisfies $m'_j=m_{(j-r)\bmod S}$. The Python object
        identity is preserved, but ``ct.data`` is replaced by newly allocated
        rotated/key-switched storage; aliases to the old tensor retain the old
        allocation, while aliases to the object observe all replaced state.
        """

        return ct.replace_(self.rotate_by_step(ct, rotation_step))

    def conjugate(self, ct: Ciphertext, key: ConjugationKey) -> Ciphertext:
        r"""Apply complex conjugation to every semantic CKKS slot.

        $$
        m'_j=\overline{m_j}.
        $$

        The backend applies $\sigma_{-1}:X\mapsto X^{-1}$ to both coefficient
        polynomials and uses ``key`` in the direction
        $\sigma_{-1}(s(X))\longmapsto s(X)$. Input must be two-component,
        coefficient-domain, standard-residue, full-layout Q RNS. Output is a
        new Q ciphertext with unchanged batch shape, level, actual scale,
        engine integral dtype/device, and exact active Q ``prime_ids``, up to
        key-switch error. Inputs are unchanged and output storage is
        independent.
        """

        ct.assert_state(
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="standard",
            components=2,
        )
        self._assert_engine_ciphertext(ct)
        self._assert_engine_key(
            key, expected_type=ConjugationKey, modulus_basis="QP"
        )
        active_moduli = self.rns_runtime.moduli[
            ct.prime_ids[0] : ct.prime_ids[-1] + 1
        ]
        conjugated = self._ciphertext_from_components(
            [
                apply_coefficient_galois_automorphism(
                    ct.c0,
                    2 * self.config.N - 1,
                    active_moduli,
                ),
                apply_coefficient_galois_automorphism(
                    ct.c1,
                    2 * self.config.N - 1,
                    active_moduli,
                ),
            ],
            level=ct.level,
            scale=ct.scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
        )
        return self._apply_key_switch(conjugated, key)

    def __str__(self) -> str:
        return (
            f"CkksEngine(id={self.id}, context={self.context}, "
            f"device={self.device}, "
            f"backend={self.ntt_backend_name!r}, "
            f"rng={type(self.rng).__name__})"
        )

    __repr__ = __str__

    # ------------------------------------------------------------------
    # Internal validation and value construction.
    # ------------------------------------------------------------------

    def _validate_public_level(self, level: object) -> int:
        """Require one exact level addressable by this public CKKS context."""

        if type(level) is not int:
            raise TypeError(
                f"level must be an integer, got {type(level).__name__}"
            )
        if not 0 <= level < self.public_level_count:
            raise ValueError(
                "level must satisfy 0 <= level < "
                f"{self.public_level_count}; got {level}"
            )
        return level

    @staticmethod
    def _validate_modulus_basis(value: object) -> ModulusBasis:
        """Require one exact public modulus-basis selector."""

        if not isinstance(value, str):
            raise TypeError(
                f"modulus_basis must be 'Q' or 'QP', got {type(value).__name__}"
            )
        if value not in ("Q", "QP"):
            raise ValueError(
                f"modulus_basis must be 'Q' or 'QP', got {value!r}"
            )
        return cast(ModulusBasis, value)

    def _ciphertext_from_components(
        self,
        components: Sequence[torch.Tensor],
        *,
        level: int,
        scale: float | None,
        polynomial_domain: PolynomialDomain,
        modulus_basis: ModulusBasis,
        residue_representation: ResidueRepresentation,
        prime_ids: Sequence[int] | None = None,
    ) -> Ciphertext:
        """Pack equal-layout components into one independent ciphertext.

        Each input has layout ``[*batch, limb, coefficient_or_ntt_index]``,
        identical shape/dtype/device, and limb rows ordered exactly as
        ``prime_ids`` (or the engine's active basis rows). The output has
        layout ``[component, *batch, limb, coefficient_or_ntt_index]`` and
        copies every component into new storage. Domain, Q/QP basis, standard
        or Montgomery form, level, and actual scale are recorded as supplied;
        no arithmetic conversion occurs.
        """

        if len(components) not in (2, 3):
            raise ValueError(
                f"Ciphertext requires two or three components, got {len(components)}."
            )
        shapes = {tuple(component.shape) for component in components}
        if len(shapes) != 1:
            raise ValueError(
                f"Ciphertext component shapes must match: {shapes}"
            )
        # Allocate the final component-major payload once. Avoid a hidden
        # torch.stack packing step whose temporary/output behavior obscures
        # copy costs during native operation review.
        local = torch.empty(
            (len(components), *components[0].shape),
            dtype=components[0].dtype,
            device=components[0].device,
        )
        for component_id, component in enumerate(components):
            local[component_id].copy_(component)
        return Ciphertext(
            data=local,
            level=level,
            scale=(
                self.config.default_scale
                if scale is None
                else coerce_scale(scale, value_name="Ciphertext")
            ),
            context_id=self.context.context_id,
            prime_ids=(
                tuple(prime_ids)
                if prime_ids is not None
                else self.rns_layout.prime_ids(
                    level, include_p=modulus_basis == "QP"
                )
            ),
            polynomial_domain=polynomial_domain,
            modulus_basis=modulus_basis,
            residue_representation=residue_representation,
        )

    def _assert_same_cipher_layout(
        self,
        a: Ciphertext,
        b: Ciphertext,
        *,
        require_same_scale: bool = False,
        operation: str = "ciphertext operation",
    ) -> None:
        self._assert_local_ciphertext(a)
        self._assert_local_ciphertext(b)
        self._assert_matching_cipher_layout(
            a,
            b,
            require_same_scale=require_same_scale,
            operation=operation,
        )

    @staticmethod
    def _assert_matching_cipher_layout(
        a: Ciphertext,
        b: Ciphertext,
        *,
        require_same_scale: bool = False,
        operation: str = "ciphertext operation",
    ) -> None:
        """Compare two engine-local ciphertext layouts."""

        fields = (
            "context_id",
            "level",
            "polynomial_domain",
            "modulus_basis",
            "residue_representation",
            "component_count",
            "prime_ids",
        )
        mismatches = [
            name for name in fields if getattr(a, name) != getattr(b, name)
        ]
        if mismatches:
            raise ValueError(
                f"Ciphertext layouts differ in {mismatches}: a={a}, b={b}."
            )
        if a.data.shape != b.data.shape:
            raise ValueError(
                "Ciphertext tensor shapes differ: "
                f"{tuple(a.data.shape)} vs {tuple(b.data.shape)}."
            )
        if require_same_scale and a.scale != b.scale:
            raise ScaleMismatchError(
                operation=operation,
                lhs_name="lhs",
                lhs_scale=a.scale,
                rhs_name="rhs",
                rhs_scale=b.scale,
            )

    def validate_ciphertext(self, ciphertext: Ciphertext) -> None:
        """Validate a complete dense ciphertext against this engine.

        The check re-runs the value's mutable-storage invariants and requires the
        exact context, ring dimension, dtype, device, public level, modulus
        basis, and complete active ``prime_ids`` expected by ordinary engine
        operations. It performs no evaluator work and does not mutate the
        ciphertext.
        """

        self._assert_engine_ciphertext(ciphertext)

    def validate_public_key(self, key: PublicKey) -> None:
        """Validate public encryption material against this engine.

        The check re-runs the key's mutable-storage invariants and requires the
        exact context, ring dimension, dtype, device, polynomial domain,
        residue representation, and prime layout consumed by online
        encryption. The key is not installed or mutated.
        """

        if type(key) is not PublicKey:
            raise TypeError(f"Expected PublicKey, got {type(key).__name__}")
        PublicKey(
            key.data,
            key.context_id,
            key.prime_ids,
            key.polynomial_domain,
            key.modulus_basis,
            key.residue_representation,
        )
        self._assert_engine_key(key, expected_type=PublicKey)

    def validate_key_switch_key(self, key: KeySwitchKey) -> None:
        """Validate complete QP key-switch material against this engine.

        The concrete key type is preserved while context, ring dimension,
        dtype, device, NTT/Montgomery state, QP prime layout, and hybrid digit
        count are checked. The key is not installed or mutated.
        """

        if not isinstance(key, KeySwitchKey):
            raise TypeError(f"Expected KeySwitchKey, got {type(key).__name__}")
        key._with_resident_tensors((key.data,))
        self._assert_engine_key(
            key,
            expected_type=type(key),
            modulus_basis="QP",
        )
        if isinstance(key, RotationKey):
            if type(key.rotation_step) is not int:
                raise TypeError(
                    "RotationKey rotation_step must be an integer, got "
                    f"{type(key.rotation_step).__name__}"
                )
            canonical_step = RotationKey.canonical_step(
                key.rotation_step,
                ring_dimension=self.config.N,
            )
            if key.rotation_step != canonical_step:
                raise ValueError(
                    "RotationKey rotation_step is not canonical for this "
                    f"engine: {key.rotation_step} != {canonical_step}"
                )

    def _assert_local_ciphertext(
        self,
        ct: Ciphertext,
        *,
        _allow_structural_base: bool = False,
    ) -> None:
        """Validate a ciphertext that may contain a local RNS-limb interval.

        The common compatibility requirements cover the CKKS context, ring
        dimension, rank-local device, and RNS structure active at the
        ciphertext's level and declared Q or QP basis. ``prime_ids`` must be a
        nonempty, ordered, contiguous interval of that structure. Unlike
        :meth:`_assert_engine_ciphertext`, this check accepts a proper interval
        for RNS-limb-partitioned workflows.

        Operation-specific requirements such as a full RNS layout, component
        count, polynomial domain, Montgomery representation, and scale remain
        the caller's responsibility.

        Args:
            ct: Dense local ciphertext to validate.

        Raises:
            ValueError: If the ciphertext is incompatible with this engine or
                its RNS interval is invalid.
        """

        if not isinstance(ct, Ciphertext):
            raise TypeError(f"Expected Ciphertext, got {type(ct).__name__}")
        if _allow_structural_base and ct.level == self.public_level_count:
            pass
        else:
            self._validate_public_level(ct.level)
        if ct.context_id != self.context.context_id:
            raise ValueError(
                "Ciphertext belongs to another CKKS context: "
                f"{ct.context_id} != {self.context.context_id}"
            )
        if ct.ring_dimension != self.config.N:
            raise ValueError(
                "Ciphertext ring dimension does not match engine: "
                f"{ct.ring_dimension} != {self.config.N}"
            )
        if ct.data.device != self.device:
            raise ValueError(
                "Ciphertext device does not match this rank-local engine: "
                f"{ct.data.device} != {self.device}"
            )
        if ct.data.dtype != self.config.torch_dtype:
            raise TypeError(
                "Ciphertext dtype does not match engine: "
                f"{ct.data.dtype} != {self.config.torch_dtype}"
            )
        expected_prime_ids = self.rns_layout.prime_ids(
            ct.level, include_p=ct.includes_p
        )
        if not ct.prime_ids:
            raise ValueError("Ciphertext local RNS structure cannot be empty")
        try:
            start = expected_prime_ids.index(ct.prime_ids[0])
        except ValueError as exc:
            raise ValueError(
                "Ciphertext local RNS structure is outside the active basis: "
                f"prime_ids={ct.prime_ids}, active={expected_prime_ids}"
            ) from exc
        expected_local_ids = expected_prime_ids[
            start : start + len(ct.prime_ids)
        ]
        if ct.prime_ids != expected_local_ids:
            raise ValueError(
                "Ciphertext local RNS structure must be a contiguous ordered "
                f"interval of the active basis: prime_ids={ct.prime_ids}, "
                f"active={expected_prime_ids}"
            )

    def _assert_engine_ciphertext(self, ct: Ciphertext) -> None:
        """Validate one complete dense ciphertext against this engine."""

        if not isinstance(ct, Ciphertext):
            raise TypeError(f"Expected Ciphertext, got {type(ct).__name__}")
        # Reconstruct around the same storage to re-run the complete mutable
        # value invariants without cloning the tensor payload.
        ct.with_data(ct.data)
        self._assert_local_ciphertext(ct)
        expected_prime_ids = self.rns_layout.prime_ids(
            ct.level, include_p=ct.includes_p
        )
        if ct.prime_ids != expected_prime_ids:
            raise ValueError(
                "This full-layout primitive received a different local RNS "
                f"structure: ciphertext prime_ids={ct.prime_ids}, "
                f"engine prime_ids={expected_prime_ids}. Use an local "
                "primitive and communication plan for limb-partitioned data."
            )

    def _assert_structural_base_ciphertext(self, ct: Ciphertext) -> None:
        """Validate the private one-prime bootstrap structural value."""

        if ct.level != self.public_level_count:
            raise ValueError(
                "Structural-base Ciphertext must use private level "
                f"{self.public_level_count}, got {ct.level}"
            )
        self._assert_local_ciphertext(ct, _allow_structural_base=True)
        expected_prime_ids = self.rns_layout.prime_ids(ct.level)
        if ct.prime_ids != expected_prime_ids:
            raise ValueError(
                "Structural-base Ciphertext RNS layout differs from engine: "
                f"{ct.prime_ids} != {expected_prime_ids}"
            )

    def _assert_engine_rns_plaintext(self, plaintext: Plaintext) -> None:
        """Validate one complete local RNS plaintext against this engine."""

        if not isinstance(plaintext, Plaintext):
            raise TypeError(
                f"Expected Plaintext, got {type(plaintext).__name__}"
            )
        if (
            not plaintext.is_rns
            or plaintext.message is not None
            or plaintext.data is None
        ):
            raise ValueError(
                "Plaintext transition requires representation='rns'"
            )
        # Reconstruct around the same storage to re-run the complete mutable
        # value invariants without cloning the tensor payload.
        Plaintext(
            message=plaintext.message,
            level=plaintext.level,
            scale=plaintext.scale,
            data=plaintext.data,
            context_id=plaintext.context_id,
            representation=plaintext.representation,
            polynomial_domain=plaintext.polynomial_domain,
            modulus_basis=plaintext.modulus_basis,
            residue_representation=plaintext.residue_representation,
            prime_ids=plaintext.prime_ids,
        )
        self._validate_public_level(plaintext.level)
        if plaintext.context_id != self.context.context_id:
            raise ValueError("Plaintext belongs to another CKKS context")
        if plaintext.data.device != self.device:
            raise ValueError("Plaintext data is on the wrong local device")
        if plaintext.data.dtype != self.config.torch_dtype:
            raise TypeError(
                "Plaintext dtype does not match engine: "
                f"{plaintext.data.dtype} != {self.config.torch_dtype}"
            )
        if plaintext.data.size(-1) != self.config.N:
            raise ValueError(
                "Plaintext polynomial degree does not match engine: "
                f"{plaintext.data.size(-1)} != {self.config.N}"
            )
        expected_prime_ids = self.rns_layout.prime_ids(
            plaintext.level,
            include_p=plaintext.modulus_basis == "QP",
        )
        if plaintext.prime_ids != expected_prime_ids:
            raise ValueError(
                "Plaintext RNS structure differs from engine: "
                f"{plaintext.prime_ids} != {expected_prime_ids}"
            )

    def _assert_domain_transition_source(
        self,
        value: Ciphertext | Plaintext,
        *,
        source_domain: Literal["coefficient", "ntt"],
    ) -> None:
        """Validate one complete source state before a domain transition."""

        operation = (
            "coefficient_domain_to_ntt_domain"
            if source_domain == "coefficient"
            else "ntt_domain_to_coefficient_domain"
        )
        domain_source = (
            "a coefficient-domain"
            if source_domain == "coefficient"
            else "an NTT-domain"
        )
        if isinstance(value, Plaintext):
            self._assert_engine_rns_plaintext(value)
            if value.polynomial_domain != source_domain:
                raise ValueError(
                    f"{operation} requires {domain_source} plaintext"
                )
            if value.residue_representation != "montgomery":
                raise ValueError(
                    f"{operation} requires Montgomery plaintext residues"
                )
            return
        if not isinstance(value, Ciphertext):
            raise TypeError(
                "Domain transition expects Ciphertext or Plaintext, got "
                f"{type(value).__name__}"
            )
        self._assert_engine_ciphertext(value)
        if value.polynomial_domain != source_domain:
            raise ValueError(f"{operation} requires {domain_source} ciphertext")
        expected_residues = (
            "standard" if source_domain == "coefficient" else "montgomery"
        )
        if value.residue_representation != expected_residues:
            raise ValueError(
                f"{operation} requires {expected_residues} ciphertext residues"
            )

    def _assert_plaintext_residue_transition_source(
        self,
        plaintext: Plaintext,
        *,
        source_residues: Literal["standard", "montgomery"],
    ) -> None:
        """Validate one coefficient-domain plaintext residue source state."""

        operation = (
            "standard_residues_to_montgomery_residues"
            if source_residues == "standard"
            else "montgomery_residues_to_standard_residues"
        )
        residue_name = (
            "standard" if source_residues == "standard" else "Montgomery"
        )
        self._assert_engine_rns_plaintext(plaintext)
        if plaintext.polynomial_domain != "coefficient":
            raise ValueError(
                f"{operation} requires a coefficient-domain plaintext"
            )
        if plaintext.residue_representation != source_residues:
            raise ValueError(f"{operation} requires {residue_name} residues")

    def _assert_engine_key(
        self,
        key: SecretKey | PublicKey | KeySwitchKey,
        *,
        expected_type: type[object],
        modulus_basis: ModulusBasis | None = None,
    ) -> None:
        """Validate that a dense key belongs to this rank-local engine.

        The compatibility requirements cover the exact concrete key type, CKKS
        context, polynomial degree, configured dtype and local device, NTT
        polynomial domain, Montgomery residues, optional exact basis, complete
        level-zero prime layout, and key-switch digit shape where applicable.

        Args:
            key: Dense secret, public, or key-switch key to validate.
            expected_type: Exact concrete key class required by the operation.
            modulus_basis: Exact RNS basis required by the calling operation, or
                ``None`` to accept either Q or QP and validate its matching
                prime layout.

        Raises:
            ValueError: If the key is incompatible with this engine or the
                requested basis.
        """

        if type(key) is not expected_type:
            raise TypeError(
                f"Expected {expected_type.__name__}, got {type(key).__name__}"
            )
        if key.context_id != self.context.context_id:
            raise ValueError(
                f"{type(key).__name__} belongs to another CKKS context: "
                f"{key.context_id} != {self.context.context_id}"
            )
        if key.data.size(-1) != self.config.N:
            raise ValueError(
                f"{type(key).__name__} polynomial degree does not match engine: "
                f"{key.data.size(-1)} != {self.config.N}"
            )
        if key.data.device != self.device:
            raise ValueError(
                f"{type(key).__name__} device does not match engine: "
                f"{key.data.device} != {self.device}"
            )
        if key.data.dtype != self.config.torch_dtype:
            raise TypeError(
                f"{type(key).__name__} dtype does not match engine: "
                f"{key.data.dtype} != {self.config.torch_dtype}"
            )
        if key.polynomial_domain != "ntt":
            raise ValueError(
                f"{type(key).__name__} must be in NTT polynomial domain"
            )
        if key.residue_representation != "montgomery":
            raise ValueError(
                f"{type(key).__name__} must use Montgomery residues"
            )
        if modulus_basis is not None and key.modulus_basis != modulus_basis:
            raise ValueError(
                f"{type(key).__name__} requires basis {modulus_basis}, got {key.modulus_basis}"
            )
        expected_prime_ids = self.rns_layout.prime_ids(
            0, include_p=key.modulus_basis == "QP"
        )
        if key.prime_ids != expected_prime_ids:
            raise ValueError(
                f"{type(key).__name__} local RNS structure differs from the "
                f"engine: key prime_ids={key.prime_ids}, "
                f"engine prime_ids={expected_prime_ids}"
            )
        if isinstance(key, KeySwitchKey):
            if key.digit_count != self.rns_layout.key_digit_count:
                raise ValueError(
                    f"{type(key).__name__} key digit count differs from engine: "
                    f"{key.digit_count} != {self.rns_layout.key_digit_count}"
                )
            expected_shape = (
                self.rns_layout.key_digit_count,
                2,
                len(expected_prime_ids),
                self.config.N,
            )
            if tuple(key.data.shape) != expected_shape:
                raise ValueError(
                    f"{type(key).__name__} shape differs from engine: "
                    f"{tuple(key.data.shape)} != {expected_shape}"
                )

    # ------------------------------------------------------------------
    # Internal RNS and Montgomery parameter construction.
    # ------------------------------------------------------------------

    def _create_rescale_dropped_q_inverses_montgomery(self) -> None:
        r"""Build per-level dropped-Q inverse vectors for rescale.

        Entry ``level`` is an engine-integral CPU or CUDA tensor with layout
        ``[remaining_qp_limb]``. Element $i$ is
        $q_{\mathrm{drop}}^{-1}R\bmod r_i$ for the corresponding row of exact
        ``rns_layout.prime_ids(level, include_p=True)[1:]``. The table is
        Montgomery-form scalar metadata, not a polynomial tensor. It is newly
        allocated and retained by the engine.
        """

        self.rescale_dropped_q_inverses_montgomery_by_level: list[
            torch.Tensor
        ] = []
        for level in range(self.public_level_count):
            # Store the complete remaining QP interval. A Q ciphertext uses
            # its leading Q-only prefix; a QP ciphertext also consumes the P
            # entries. This keeps one canonical table per public level.
            active_prime_ids = self.rns_layout.prime_ids(level, include_p=True)
            remaining_moduli = [
                self.montgomery_parameters.moduli[index]
                for index in active_prime_ids[1:]
            ]
            dropped_prime = self.montgomery_parameters.moduli[
                active_prime_ids[0]
            ]
            dropped_q_inverses_montgomery = [
                (pow(dropped_prime, -1, modulus) * self.montgomery_parameters.R)
                % modulus
                for modulus in remaining_moduli
            ]
            self.rescale_dropped_q_inverses_montgomery_by_level.append(
                torch.tensor(
                    dropped_q_inverses_montgomery,
                    dtype=self.config.torch_dtype,
                    device=self.device,
                )
            )

    def _create_keyswitch_moddown_parameters(self) -> None:
        r"""Build sequential P-ModDown dropped-prime inverse tables.

        ``moddown_p_drop_inverses_montgomery_by_level[level]`` is a dense
        engine-integral tensor on the engine device with layout
        ``[p_drop_step, surviving_limb]``. Valid entries store
        $p_{\mathrm{drop}}^{-1}R\bmod r_i$ for each surviving QP row after
        that sequential P-prime drop. Row order follows the exact active
        ``prime_ids`` at ``level``; unused packed tail entries are never
        consumed. Construction allocates engine-owned table storage.
        """

        montgomery_radix = self.montgomery_parameters.R
        p_moduli = self.montgomery_parameters.moduli[
            -self.config.num_p_primes :
        ][::-1]
        moduli = self.montgomery_parameters.moduli
        drop_prime_inverse_rows = [
            [
                (pow(dropped_p_modulus, -1, modulus) * montgomery_radix)
                % modulus
                for modulus in moduli[: -drop_step - 1]
            ]
            for drop_step, dropped_p_modulus in enumerate(p_moduli)
        ]
        base_rows: list[torch.Tensor] = []
        for drop_step in range(self.config.num_p_primes):
            active_prime_ids = self.rns_layout.prime_ids(0, include_p=True)
            row = [
                drop_prime_inverse_rows[drop_step][prime_id]
                for prime_id in active_prime_ids[: -drop_step - 1]
            ]
            base_rows.append(
                torch.tensor(
                    row,
                    device=self.device,
                    dtype=self.config.torch_dtype,
                )
            )

        self.moddown_p_drop_inverses_montgomery_by_level: list[
            torch.Tensor
        ] = []
        for level in range(self.public_level_count):
            start = self.rns_layout.start_row(level)
            rows = [
                base_rows[drop_step][start:]
                for drop_step in range(self.config.num_p_primes)
            ]
            max_len = max(row.numel() for row in rows)
            packed = torch.empty(
                (len(rows), max_len),
                dtype=self.config.torch_dtype,
                device=self.device,
            )
            for row_id, row in enumerate(rows):
                packed[row_id, : row.numel()] = row
            self.moddown_p_drop_inverses_montgomery_by_level.append(packed)

    def _create_p_product_montgomery_q(self) -> None:
        r"""Build $PR\bmod q_i$ for every level-zero Q row.

        The result has layout ``[q_limb]``, engine integral dtype/device,
        Montgomery scalar form, and row order
        ``rns_layout.prime_ids(0, include_p=False)``. It is newly allocated and
        used in hybrid key-switch key generation.
        """

        product_p = math.prod(
            self.montgomery_parameters.moduli[-self.config.num_p_primes :]
        )
        pr = product_p * self.montgomery_parameters.R
        dest = self.rns_layout.prime_ids(0)
        self.p_product_montgomery_q = torch.tensor(
            [pr % self.montgomery_parameters.moduli[index] for index in dest],
            device=self.device,
            dtype=self.config.torch_dtype,
        )

    # ------------------------------------------------------------------
    # Internal key resolution.
    # ------------------------------------------------------------------

    def _canonical_rotation_step(self, rotation_step: int) -> int:
        return RotationKey.canonical_step(
            rotation_step,
            ring_dimension=self.config.N,
        )

    def _resolve_public_key(self, key: PublicKey | None) -> PublicKey:
        resolved = self.public_key if key is None else key
        self._assert_engine_key(resolved, expected_type=PublicKey)
        return resolved

    def _resolve_secret_key(self, key: SecretKey | None) -> SecretKey:
        resolved = self.secret_key if key is None else key
        self._assert_engine_key(resolved, expected_type=SecretKey)
        return resolved

    # ------------------------------------------------------------------
    # Internal plaintext representation transitions.
    # ------------------------------------------------------------------

    def _apply_forward_ntt_(
        self,
        value: Ciphertext | Plaintext,
    ) -> Ciphertext | Plaintext:
        """Apply a forward NTT to a checked coefficient-domain value."""

        if isinstance(value, Plaintext):
            return self._to_ntt_plaintext_(value)
        for component_id in range(value.component_count):
            self.rns_runtime.forward_to_montgomery_(
                value.component(component_id),
                include_p=value.includes_p,
            )
        value.polynomial_domain = "ntt"
        value.residue_representation = "montgomery"
        return value

    def _apply_inverse_ntt_(
        self,
        value: Ciphertext | Plaintext,
    ) -> Ciphertext | Plaintext:
        """Apply an inverse NTT to a checked NTT-domain value."""

        if isinstance(value, Plaintext):
            assert value.data is not None
            self.rns_runtime.inverse_montgomery_(
                value.data,
                include_p=value.modulus_basis == "QP",
            )
            value.polynomial_domain = "coefficient"
            return value
        for component_id in range(value.component_count):
            self.rns_runtime.inverse_to_standard_(
                value.component(component_id),
                include_p=value.includes_p,
            )
        value.polynomial_domain = "coefficient"
        value.residue_representation = "standard"
        return value

    def _to_montgomery_plaintext_(self, plaintext: Plaintext) -> Plaintext:
        """Convert an engine-local coefficient plaintext to Montgomery form."""

        if plaintext.polynomial_domain != "coefficient":
            raise ValueError("Residue conversion requires coefficient domain")
        if plaintext.residue_representation != "standard":
            raise ValueError(
                "standard_residues_to_montgomery_residues requires "
                "standard residues"
            )
        assert plaintext.data is not None
        self.rns_runtime.to_montgomery_(
            plaintext.data,
            include_p=plaintext.modulus_basis == "QP",
        )
        plaintext.residue_representation = "montgomery"
        return plaintext

    def _to_ntt_plaintext_(self, plaintext: Plaintext) -> Plaintext:
        """Transform an engine-local Montgomery plaintext to NTT form."""

        if plaintext.residue_representation != "montgomery":
            raise ValueError("Plaintext NTT requires Montgomery residues")
        if plaintext.polynomial_domain != "coefficient":
            raise ValueError(
                "coefficient_domain_to_ntt_domain requires a "
                "coefficient-domain plaintext"
            )
        assert plaintext.data is not None
        self.rns_runtime.forward_montgomery_(
            plaintext.data,
            include_p=plaintext.modulus_basis == "QP",
        )
        plaintext.polynomial_domain = "ntt"
        return plaintext

    def _to_standard_plaintext_(self, plaintext: Plaintext) -> Plaintext:
        """Convert an engine-local coefficient plaintext to standard form."""

        if plaintext.polynomial_domain != "coefficient":
            raise ValueError("Residue conversion requires coefficient domain")
        if plaintext.residue_representation != "montgomery":
            raise ValueError(
                "montgomery_residues_to_standard_residues requires "
                "Montgomery residues"
            )
        assert plaintext.data is not None
        self.rns_runtime.from_montgomery_(
            plaintext.data,
            include_p=plaintext.modulus_basis == "QP",
        )
        plaintext.residue_representation = "standard"
        return plaintext

    # ------------------------------------------------------------------
    # Internal scale and level transitions.
    # ------------------------------------------------------------------

    def _rescale_final_public_level_to_structural_base(
        self,
        ct: Ciphertext,
        *,
        rounding: Literal["nearest", "floor"] = "nearest",
    ) -> Ciphertext:
        """Rescale into the structural Q basis consumed by ModRaise.

        This engine-owned transition connects the final public level to the
        private bootstrap representation.
        """

        return self._rescaler._rescale_final_public_level_to_structural_base(
            ct,
            rounding=rounding,
        )

    def _validate_mod_switch_target(
        self,
        ct: Ciphertext,
        target_level: int,
    ) -> None:
        """Validate a full-layout ciphertext and one public target level."""

        self._assert_engine_ciphertext(ct)
        if type(target_level) is not int:
            raise TypeError("target_level must be an integer")
        if not ct.level <= target_level <= self.final_public_level:
            raise ValueError(
                "mod_switch_to_level requires "
                f"{ct.level} <= target_level <= {self.final_public_level}; "
                f"got {target_level}."
            )

    def _copy_at_level(
        self,
        ct: Ciphertext,
        target_level: int,
    ) -> Ciphertext:
        """Copy a ciphertext onto an already resolved suffix level."""

        if target_level == ct.level:
            return ct.clone()
        rows_to_drop = target_level - ct.level
        return self._ciphertext_from_components(
            [
                ct.component(index)[..., rows_to_drop:, :]
                for index in range(ct.component_count)
            ],
            level=target_level,
            scale=ct.scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
            prime_ids=ct.prime_ids[rows_to_drop:],
        )

    @staticmethod
    def _restrict_to_level_(
        ct: Ciphertext,
        target_level: int,
    ) -> Ciphertext:
        """Narrow a ciphertext onto an already resolved suffix level."""

        rows_to_drop = target_level - ct.level
        if rows_to_drop:
            ct.data = ct.data[..., rows_to_drop:, :]
            ct.prime_ids = ct.prime_ids[rows_to_drop:]
            ct.level = target_level
        return ct

    @staticmethod
    def _set_reinterpreted_scale_(
        ct: Ciphertext,
        target_scale: float,
        *,
        max_relative_change: float | None,
    ) -> Ciphertext:
        """Apply a scale-metadata reinterpretation to ``ct``."""

        target = coerce_scale(
            target_scale,
            value_name="reinterpret_at_scale target",
        )
        if max_relative_change is not None:
            if isinstance(max_relative_change, (bool, str, bytes)):
                raise ValueError(
                    "max_relative_change must be a non-negative finite float"
                )
            try:
                relative_limit = float(max_relative_change)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "max_relative_change must be a non-negative finite float"
                ) from error
            if not math.isfinite(relative_limit) or relative_limit < 0.0:
                raise ValueError(
                    "max_relative_change must be a non-negative finite float"
                )
            relative_change = max(ct.scale / target, target / ct.scale) - 1.0
            if relative_change > relative_limit:
                raise ScaleMismatchError(
                    operation="reinterpret_at_scale",
                    lhs_name="current",
                    lhs_scale=ct.scale,
                    rhs_name="target",
                    rhs_scale=target,
                )
        ct.scale = target
        return ct

    # ------------------------------------------------------------------
    # Internal zero construction.
    # ------------------------------------------------------------------

    def _transparent_zero_like(self, ct: Ciphertext) -> Ciphertext:
        r"""Construct internal transparent components $c_j(X)=0$.

        This is not an encryption of zero and must not be treated as a
        semantically secure output ciphertext. The new allocation matches
        ``ct`` in component count, ``[*batch, limb, index]`` shape,
        dtype/device, level, actual scale, domain, Q/QP basis, residue form, and
        exact ``prime_ids``; it does not alias ``ct``.
        """

        self._assert_engine_ciphertext(ct)
        return self._ciphertext_from_components(
            [
                torch.zeros_like(ct.component(i))
                for i in range(ct.component_count)
            ],
            level=ct.level,
            scale=ct.scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
        )

    # ------------------------------------------------------------------
    # Internal arithmetic.
    # ------------------------------------------------------------------

    def _apply_key_switch(
        self, ct: Ciphertext, key: KeySwitchKey
    ) -> Ciphertext:
        """Apply concrete key-switch material to an engine-local ciphertext."""

        correction0, correction1 = self._hybrid_key_switcher.apply_key_switch(
            ct.c1, key, ct.level
        )
        switched_c0 = self.rns_runtime.add_canonical(ct.c0, correction0)
        return self._ciphertext_from_components(
            [switched_c0, correction1],
            level=ct.level,
            scale=ct.scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
        )

    def _add_ciphertext_payloads(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        inplace: bool,
    ) -> Ciphertext:
        """Add ciphertexts whose local layouts and scales already match."""

        result = lhs if inplace else lhs.clone()
        result_batch = self._flatten_component_batch(result.data)
        rhs_batch = self._flatten_component_batch(rhs.data)
        if result_batch is not None and rhs_batch is not None:
            self.rns_runtime.add_canonical_(
                result_batch,
                rhs_batch,
                prime_ids=result.prime_ids,
            )
        else:
            for component_index in range(result.component_count):
                self.rns_runtime.add_canonical_(
                    result.component(component_index),
                    rhs.component(component_index),
                    prime_ids=result.prime_ids,
                )
        return result

    @staticmethod
    def _flatten_component_batch(data: torch.Tensor) -> torch.Tensor | None:
        """Return a zero-copy native batch view, or ``None`` if unavailable."""

        try:
            return data.view(-1, data.size(-2), data.size(-1))
        except RuntimeError:
            return None

    def _subtract_ciphertext_payloads(
        self,
        lhs: Ciphertext,
        rhs: Ciphertext,
        *,
        inplace: bool,
    ) -> Ciphertext:
        """Subtract ciphertexts whose local layouts and scales already match."""

        result = lhs if inplace else lhs.clone()
        result_batch = self._flatten_component_batch(result.data)
        rhs_batch = self._flatten_component_batch(rhs.data)
        if result_batch is not None and rhs_batch is not None:
            self.rns_runtime.sub_canonical_(
                result_batch,
                rhs_batch,
                prime_ids=result.prime_ids,
            )
        else:
            for component_index in range(result.component_count):
                self.rns_runtime.sub_canonical_(
                    result.component(component_index),
                    rhs.component(component_index),
                    prime_ids=result.prime_ids,
                )
        return result

    def _prepare_plaintext_operand(
        self,
        plaintext: Plaintext,
        ct: Ciphertext,
        *,
        polynomial_domain: PolynomialDomain,
    ) -> torch.Tensor:
        """Validate plaintext state against an engine-local ciphertext.

        The returned tensor has layout
        ``[*batch, limb, coefficient_or_ntt_index]``, engine integral
        dtype/device, exact ``prime_ids`` equal to the ciphertext limb order,
        matching Q/QP basis, Montgomery residues, and the requested polynomial
        domain. A nonempty plaintext batch shape must equal the ciphertext
        batch shape; a truly unbatched plaintext may broadcast over
        ``ct.batch_shape``. The return aliases ``plaintext.data`` and neither
        value is mutated.
        """

        if plaintext.level != ct.level:
            raise ValueError(
                f"Plaintext level {plaintext.level} does not match "
                f"ciphertext level {ct.level}."
            )
        if plaintext.data is None:
            raise ValueError(
                "Plaintext is unencoded; encode it with representation='rns' and the "
                "domain required by the public operation."
            )
        if plaintext.context_id != ct.context_id:
            raise ValueError(
                "Plaintext and Ciphertext contexts differ: "
                f"{plaintext.context_id} != {ct.context_id}"
            )
        if (
            plaintext.representation != "rns"
            or plaintext.polynomial_domain != polynomial_domain
            or plaintext.residue_representation != "montgomery"
        ):
            raise ValueError(
                "Plaintext has the wrong arithmetic state for this operation: "
                "expected representation='rns', "
                f"polynomial_domain={polynomial_domain!r}, "
                "residue_representation='montgomery'; got "
                f"representation={plaintext.representation!r}, "
                f"polynomial_domain={plaintext.polynomial_domain!r}, "
                "residue_representation="
                f"{plaintext.residue_representation!r}."
            )
        self._assert_engine_rns_plaintext(plaintext)
        if plaintext.modulus_basis != ct.modulus_basis:
            raise ValueError(
                "Plaintext and Ciphertext bases differ: "
                f"{plaintext.modulus_basis!r} != {ct.modulus_basis!r}"
            )
        if plaintext.prime_ids != ct.prime_ids:
            raise ValueError(
                "Plaintext and Ciphertext RNS layouts differ: "
                f"{plaintext.prime_ids} != {ct.prime_ids}"
            )
        if plaintext.batch_shape and plaintext.batch_shape != ct.batch_shape:
            raise ValueError(
                "A batched RNS Plaintext batch shape must exactly match the "
                "Ciphertext batch shape; only a genuinely unbatched RNS "
                "Plaintext may broadcast: "
                f"{tuple(plaintext.batch_shape)} != {tuple(ct.batch_shape)}"
            )
        return plaintext.data

    def _prepare_compressed_plaintext_operand(
        self,
        plaintext: CompressedPlaintext,
        ct: Ciphertext,
        *,
        polynomial_domain: PolynomialDomain,
    ) -> torch.Tensor:
        """Validate compact state against an engine-local ciphertext.

        The result aliases dense integral
        ``[*batch, limb, unique_index]`` storage on the engine device. Exact
        ``prime_ids``, Q/QP basis, Montgomery form, and requested domain match
        ``ct``. A nonempty batch shape must match exactly; only an unbatched
        value may broadcast. ``implicit_data``, when present, has
        ``[*batch, limb]`` shape and matching dtype/device. No mutation occurs.
        """

        if plaintext.level != ct.level:
            raise ValueError(
                f"CompressedPlaintext level {plaintext.level} does not match "
                f"ciphertext level {ct.level}."
            )
        if plaintext.context_id != ct.context_id:
            raise ValueError(
                "CompressedPlaintext and Ciphertext contexts differ: "
                f"{plaintext.context_id} != {ct.context_id}"
            )
        if (
            plaintext.polynomial_domain != polynomial_domain
            or plaintext.residue_representation != "montgomery"
        ):
            raise ValueError(
                "CompressedPlaintext has the wrong arithmetic state for this "
                "operation: expected "
                f"polynomial_domain={polynomial_domain!r}, "
                "residue_representation='montgomery'; got "
                f"polynomial_domain={plaintext.polynomial_domain!r}, "
                "residue_representation="
                f"{plaintext.residue_representation!r}."
            )
        if plaintext.data.device != self.device:
            raise ValueError(
                "CompressedPlaintext data is on the wrong local device: "
                f"{plaintext.data.device} != {self.device}"
            )
        if plaintext.data.dtype != self.config.torch_dtype:
            raise TypeError(
                "CompressedPlaintext dtype does not match engine: "
                f"{plaintext.data.dtype} != {self.config.torch_dtype}"
            )
        if (
            plaintext.implicit_data is not None
            and plaintext.implicit_data.dtype != self.config.torch_dtype
        ):
            raise TypeError(
                "CompressedPlaintext implicit dtype does not match engine"
            )
        if plaintext.ring_dimension != self.config.N:
            raise ValueError(
                "CompressedPlaintext ring dimension does not match the "
                f"engine: {plaintext.ring_dimension} != {self.config.N}"
            )
        if plaintext.modulus_basis != ct.modulus_basis:
            raise ValueError(
                "CompressedPlaintext and Ciphertext bases differ: "
                f"{plaintext.modulus_basis!r} != {ct.modulus_basis!r}"
            )
        if plaintext.prime_ids != ct.prime_ids:
            raise ValueError(
                "CompressedPlaintext and Ciphertext RNS layouts differ: "
                f"{plaintext.prime_ids} != {ct.prime_ids}"
            )
        if plaintext.batch_shape and plaintext.batch_shape != ct.batch_shape:
            raise ValueError(
                "A batched CompressedPlaintext batch shape must exactly match "
                "the Ciphertext batch shape; only a genuinely unbatched value "
                "may broadcast: "
                f"{tuple(plaintext.batch_shape)} != {tuple(ct.batch_shape)}"
            )
        return plaintext.data

    def _add_compressed_plaintext_component(
        self,
        ciphertext_component: torch.Tensor,
        plaintext: CompressedPlaintext,
        prepared_data: torch.Tensor,
        rns_params: torch.Tensor,
        *,
        inplace: bool,
    ) -> torch.Tensor:
        r"""Add compact coefficient-domain Montgomery RNS to one component.

        ``ciphertext_component`` has ``[*batch, limb, coefficient]`` and
        standard residues; ``prepared_data`` has
        ``[*plaintext_batch, limb, unique_index]`` and Montgomery residues.
        Both use engine integral dtype/device and limb order
        ``plaintext.prime_ids``. ``rns_params`` is
        ``[parameter, limb]`` for those exact primes. The native kernel expands
        the declared compression layout and computes
        $c'_0=c_0+p\pmod{B_\ell}$ with unbatched-plaintext broadcasting only.
        Functional mode allocates non-aliasing canonical standard residues;
        in-place mode mutates and returns the ciphertext component storage.
        """

        if plaintext.compression_layout == "cyclic":
            if inplace:
                ckks_ops.add_cyclic_compressed_plaintext_component_(
                    ciphertext_component,
                    prepared_data,
                    rns_params,
                )
                return ciphertext_component
            return ckks_ops.add_cyclic_compressed_plaintext_component(
                ciphertext_component,
                prepared_data,
                rns_params,
            )
        if plaintext.compression_layout == "contiguous":
            if inplace:
                ckks_ops.add_contiguous_compressed_plaintext_component_(
                    ciphertext_component,
                    prepared_data,
                    rns_params,
                )
                return ciphertext_component
            return ckks_ops.add_contiguous_compressed_plaintext_component(
                ciphertext_component,
                prepared_data,
                rns_params,
            )
        if plaintext.compression_layout != "strided_sparse":
            raise ValueError(
                "Unsupported CompressedPlaintext compression layout for "
                f"addition: {plaintext.compression_layout!r}"
            )
        if plaintext.implicit_data is None:
            raise RuntimeError(
                "strided_sparse CompressedPlaintext has no implicit_data"
            )
        if inplace:
            ckks_ops.add_strided_plaintext_component_(
                ciphertext_component,
                prepared_data,
                plaintext.implicit_data,
                rns_params,
            )
            return ciphertext_component
        return ckks_ops.add_strided_plaintext_component(
            ciphertext_component,
            prepared_data,
            plaintext.implicit_data,
            rns_params,
        )

    def _multiply_compressed_plaintext_component(
        self,
        ciphertext_component_ntt: torch.Tensor,
        plaintext: CompressedPlaintext,
        prepared_data: torch.Tensor,
        *,
        prime_ids: tuple[int, ...],
    ) -> torch.Tensor:
        r"""Multiply one NTT component by compact NTT plaintext data.

        Both tensors are engine-integral on one execution device in Montgomery form.
        ``ciphertext_component_ntt`` is
        ``[*batch, limb, ntt_index]`` and ``prepared_data`` is
        ``[*plaintext_batch, limb, unique_ntt_index]``; limb row $i$ is modulo
        exact ``prime_ids[i]``. Cyclic/contiguous expansion defines $p$ over
        the full NTT axis and the result is $c'_j=c_jp\pmod{B_\ell}$.
        An unbatched plaintext may broadcast; output has the ciphertext shape,
        is newly allocated, and does not alias either input. The allowed lazy
        Montgomery residue interval is inherited from ``montgomery_mul``.
        """

        if plaintext.compression_layout == "cyclic":
            return self.rns_runtime.montgomery_mul_cyclic_compressed(
                ciphertext_component_ntt,
                prepared_data,
                prime_ids=prime_ids,
            )
        if plaintext.compression_layout == "contiguous":
            return self.rns_runtime.montgomery_mul_contiguous_compressed(
                ciphertext_component_ntt,
                prepared_data,
                prime_ids=prime_ids,
            )
        raise ValueError(
            "multiply_plaintext requires a cyclic or contiguous NTT-domain "
            "CompressedPlaintext; got "
            f"{plaintext.compression_layout!r}"
        )

    # ------------------------------------------------------------------
    # Internal rotation implementation.
    # ------------------------------------------------------------------

    def _assert_rotation_ciphertext(self, ct: Ciphertext) -> None:
        """Validate the ciphertext requirements shared by rotation entry points."""

        ct.assert_state(
            polynomial_domain="coefficient",
            modulus_basis="Q",
            residue_representation="standard",
            components=2,
        )
        self._assert_engine_ciphertext(ct)

    def _assert_rotation_key_for_step(
        self,
        key: RotationKey,
        rotation_step: int,
    ) -> None:
        """Validate one external key for an already canonical requested step."""

        self._assert_engine_key(
            key, expected_type=RotationKey, modulus_basis="QP"
        )
        if key.rotation_step != rotation_step:
            raise ValueError(
                "Rotation key step does not match requested rotation: "
                f"key={key.rotation_step}, requested={rotation_step}"
            )

    def _engine_rotation_key(
        self,
        rotation_step: int,
        resolved: dict[int, RotationKey],
    ) -> RotationKey:
        """Return one checked engine-owned key for a canonical direct step."""

        cached = resolved.get(rotation_step)
        if cached is not None:
            return cached
        try:
            key = self.rotation_keys[rotation_step]
        except KeyError:
            key = self.rotation_key(rotation_step)
        self._assert_rotation_key_for_step(key, rotation_step)
        resolved[rotation_step] = key
        return key

    def _rotation_path_for_step(
        self,
        rotation_step: int,
        resolved: dict[int, RotationKey],
    ) -> tuple[tuple[int, RotationKey], ...]:
        """Resolve an engine-owned key path for one canonical step."""

        if rotation_step == 0:
            return ()
        try:
            return (
                (
                    rotation_step,
                    self._engine_rotation_key(rotation_step, resolved),
                ),
            )
        except KeyError:
            decomposition = decompose_rotation_step(
                rotation_step,
                self.num_slots,
                self.rotation_keys,
            )
            return tuple(
                (step, self._engine_rotation_key(step, resolved))
                for step in decomposition
            )

    def _apply_rotation_path(
        self,
        ct: Ciphertext,
        path: Sequence[tuple[int, RotationKey]],
    ) -> Ciphertext:
        """Apply an ordered engine-local rotation-key path."""

        if not path:
            return ct.clone()
        result = ct
        for rotation_step, key in path:
            result = self._apply_rotation_key(result, key, rotation_step)
        return result

    def _apply_rotation_key_sequence(
        self,
        ct: Ciphertext,
        entries: Sequence[tuple[int, RotationKey | None]],
        *,
        use_hoisting: bool,
    ) -> list[Ciphertext]:
        """Apply an ordered sequence of checked direct rotation keys."""

        nonzero_count = sum(step != 0 for step, _ in entries)
        if not use_hoisting or nonzero_count < 2:
            results: list[Ciphertext] = []
            for step, key in entries:
                if step == 0:
                    results.append(ct.clone())
                    continue
                assert key is not None
                results.append(self._apply_rotation_key(ct, key, step))
            return results

        prepared = self._hybrid_key_switcher.prepare_rotation_digits(
            ct.c1, ct.level
        )
        results = []
        for step, key in entries:
            if step == 0:
                results.append(ct.clone())
                continue
            assert key is not None
            rotated_c0 = self._apply_rotation_automorphism(
                ct.c0,
                rotation_step=step,
                level=ct.level,
                include_p=ct.includes_p,
            )
            switched0, switched1 = (
                self._hybrid_key_switcher._switch_prepared_rotation(
                    rotated_c0,
                    prepared,
                    key,
                    rotation_step=step,
                )
            )
            results.append(
                self._ciphertext_from_components(
                    [switched0, switched1],
                    level=ct.level,
                    scale=ct.scale,
                    polynomial_domain=ct.polynomial_domain,
                    modulus_basis=ct.modulus_basis,
                    residue_representation=ct.residue_representation,
                )
            )
        return results

    def _apply_rotation_key(
        self,
        ct: Ciphertext,
        key: RotationKey,
        rotation_step: int,
    ) -> Ciphertext:
        """Apply one canonical rotation key to an engine-local ciphertext."""

        if rotation_step == 0:
            return ct.clone()
        rotated = self._ciphertext_from_components(
            [
                self._apply_rotation_automorphism(
                    ct.c0,
                    rotation_step=rotation_step,
                    level=ct.level,
                    include_p=ct.includes_p,
                ),
                self._apply_rotation_automorphism(
                    ct.c1,
                    rotation_step=rotation_step,
                    level=ct.level,
                    include_p=ct.includes_p,
                ),
            ],
            level=ct.level,
            scale=ct.scale,
            polynomial_domain=ct.polynomial_domain,
            modulus_basis=ct.modulus_basis,
            residue_representation=ct.residue_representation,
        )
        return self._apply_key_switch(rotated, key)

    def _apply_rotation_automorphism(
        self,
        component: torch.Tensor,
        *,
        rotation_step: int,
        level: int,
        include_p: bool,
    ) -> torch.Tensor:
        r"""Apply the polynomial automorphism for one slot rotation.

        ``component`` is an engine-integral tensor on the engine device with
        layout ``[*batch, limb, coefficient]`` and standard residues; limb
        order is the engine's exact Q or QP ``prime_ids`` at ``level``.
        For the Galois element $g$ derived from ``rotation_step``, the kernel
        computes $\sigma_g(c)(X)=c(X^g)$ in
        $R=\mathbb{Z}[X]/(X^N+1)$, including the coefficient sign induced by
        negacyclic reduction. Output has the same shape/dtype/device and
        canonical standard residues, is newly allocated, and does not alias
        ``component``. The zero step returns a clone.
        """

        if component.size(-1) != self.config.N:
            raise ValueError(
                "Ciphertext polynomial degree does not match engine"
            )
        rotation_step %= self.config.N
        if rotation_step == 0:
            return component.clone()
        source_indices, source_sign = coefficient_galois_gather_indices(
            self.config.N,
            rotation_galois_element(
                self.config.N, rotation_step, self.galois_generator
            ),
            component.device,
        )
        return ckks_ops.apply_coefficient_galois_automorphism(
            component,
            source_indices,
            source_sign,
            self.rns_runtime.twice_modulus_for_basis(
                level, include_p=include_p
            ),
        )
