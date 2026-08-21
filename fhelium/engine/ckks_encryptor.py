"""CKKS encryption for Plaintext values."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from fhelium.config import CkksConfig
from fhelium.core import Ciphertext, Plaintext, PublicKey
from fhelium.core.scale import coerce_scale
from fhelium.engine.ckks_plaintext_codec import CkksPlaintextCodec
from fhelium.engine.rns.montgomery import MontgomeryParameters
from fhelium.engine.rns.layout import RnsLayout
from fhelium.engine.rns.runtime import RnsRuntime
from fhelium.rng import Csprng


class CkksEncryptor:
    r"""Encrypt integer coefficients under one engine-owned public key policy.

    The component constructs two-component phases
    $c_0(X)+c_1(X)s(X)=p(X)+e(X)\pmod{B_\ell}$. It owns no key inventory,
    context, random-number generator, or native runtime; those dependencies are
    bound by :class:`~fhelium.engine.CkksEngine`.
    """

    def __init__(
        self,
        *,
        config: CkksConfig,
        device: torch.device,
        rng: Csprng,
        rns_layout: RnsLayout,
        rns_runtime: RnsRuntime,
        montgomery_parameters: MontgomeryParameters,
        plaintext_codec: CkksPlaintextCodec,
        engine_id: str,
        validate_public_level: Callable[[object], int],
        ciphertext_from_components: Callable[..., Ciphertext],
    ) -> None:
        self.config = config
        self.device = device
        self._rng = rng
        self.rns_layout = rns_layout
        self.rns_runtime = rns_runtime
        self.montgomery_parameters = montgomery_parameters
        self.plaintext_codec = plaintext_codec
        self.engine_id = engine_id
        self._validate_public_level = validate_public_level
        self._ciphertext_from_components = ciphertext_from_components

    def __str__(self) -> str:
        return (
            f"CkksEncryptor(engine_id={self.engine_id}, "
            f"logN={self.config.logN}, device={self.device})"
        )

    __repr__ = __str__

    def encrypt(
        self,
        plaintext: Plaintext,
        public_key: PublicKey,
    ) -> Ciphertext:
        r"""Encrypt integer coefficients under ``public_key``.

        For $p(X)\in R$, generate a phase

        $$
        c_0(X)+c_1(X)s(X)=p(X)+e(X)\pmod{B_\ell}.
        $$

        Slots input is encoded first using its stored actual scale. Decode-only
        ``approximate_coefficients`` and RNS input are rejected. Input layout
        ``[*batch, coefficient]`` has engine integral dtype/device and final
        extent $N$. The new output has
        ``[component=2, *batch, limb, coefficient]`` layout, coefficient
        domain, canonical standard residues, Q or QP basis selected by the
        public key, active ``prime_ids``, unchanged level, and
        $\Delta(c)=\Delta(p)$. Inputs are not mutated and output storage is
        independent.
        """

        if not isinstance(plaintext, Plaintext):
            raise TypeError(
                f"encrypt expects Plaintext, got {type(plaintext).__name__}"
            )
        plaintext = self.plaintext_codec._ensure_integer_plaintext(plaintext)
        self._validate_public_level(plaintext.level)
        if plaintext.data is None:
            raise RuntimeError(
                "Coefficient Plaintext data was not materialized"
            )
        include_p = public_key.modulus_basis == "QP"
        max_abs = int(torch.max(torch.abs(plaintext.data)).item())
        self._check_direct_decode_range(max_abs, plaintext.level)
        plaintext_rns = self.rns_runtime.lift_integer_coefficients_exact(
            plaintext.data,
            plaintext.level,
            include_p=include_p,
            max_abs=max_abs,
        )
        return self._encrypt_rns_plaintext(
            plaintext_rns,
            public_key=public_key,
            level=plaintext.level,
            scale=plaintext.scale,
        )

    def _check_direct_decode_range(self, max_abs: int, level: int) -> None:
        r"""Require coefficients inside the bounded direct-decode interval.

        If $D_\ell$ is the product of the trailing one or two active Q primes,
        encryption requires
        $\mathtt{max\_abs}<\lfloor D_\ell/4\rfloor$. The extra factor of
        two inside the centered interval reserves room for encryption and
        key-switch noise. No tensor or engine state is mutated.
        """

        q_prime_ids = self.rns_layout.prime_ids(level)
        decode_prime_ids = q_prime_ids[-2:]
        decode_product = math.prod(
            int(self.montgomery_parameters.moduli[index])
            for index in decode_prime_ids
        )
        # Reserve a factor of two inside the centered interval for encryption
        # and key-switch noise rather than accepting its mathematical edge.
        supported_max = decode_product // 4
        if max_abs >= supported_max:
            raise OverflowError(
                "Encoded coefficient exceeds the direct decoder range: "
                f"max_abs={max_abs}, supported_max={supported_max} at level "
                f"{level}"
            )

    def _encrypt_rns_plaintext(
        self,
        plaintext_rns: torch.Tensor,
        *,
        public_key: PublicKey,
        level: int,
        scale: float,
    ) -> Ciphertext:
        r"""Encrypt one coefficient-domain standard RNS plaintext tensor.

        ``plaintext_rns`` has layout ``[*batch, limb, coefficient]``, engine
        integral dtype/device, final extent $N$, and limb row $i$ modulo the
        active ``prime_ids[i]`` selected by ``level`` and the public-key
        Q/QP basis. ``public_key`` is
        ``[key_component=2, level_zero_limb, ntt_index]`` in NTT-domain
        Montgomery form. Sampling $v,e_0,e_1$ yields

        $$
        c_0=vk_0+p+e_0,\qquad c_1=vk_1+e_1,
        $$

        and therefore
        $c_0(X)+c_1(X)s(X)=p(X)+e(X)\pmod{B_\ell}$. Output is newly allocated
        ``[component=2, *batch, limb, coefficient]`` with canonical standard
        residues, engine dtype/device, active ``prime_ids``, and supplied
        level/actual scale. Inputs do not alias the output.
        """

        include_p = public_key.modulus_basis == "QP"
        expected_prime_ids = self.rns_layout.prime_ids(
            level,
            include_p=include_p,
        )
        if plaintext_rns.ndim < 2 or plaintext_rns.size(-2) != len(
            expected_prime_ids
        ):
            raise ValueError(
                "Encryption plaintext must have active "
                "[*batch, limb, coeff] layout: "
                f"shape={tuple(plaintext_rns.shape)}, "
                f"prime_ids={expected_prime_ids}"
            )

        batch_shape = plaintext_rns.shape[:-2]
        batch_size = batch_shape.numel()
        e0e1 = self._rng.discrete_gaussian(repeats=2 * batch_size)[0].view(
            2, *batch_shape, self.config.N
        )
        e0_tiled = self.rns_runtime.lift_centered_coefficients(
            e0e1[0], level, include_p=include_p
        )
        e1_tiled = self.rns_runtime.lift_centered_coefficients(
            e0e1[1], level, include_p=include_p
        )

        pte0 = self.rns_runtime.add_lazy(
            plaintext_rns,
            e0_tiled,
            include_p=include_p,
        )

        start = self.rns_runtime.level_row_starts[level]
        pk0 = public_key.k0[start:]
        pk1 = public_key.k1[start:]
        v = self._rng.randint(amax=2, shift=0, repeats=batch_size)[0].view(
            *batch_shape, self.config.N
        )
        v = self.rns_runtime.lift_centered_coefficients(
            v,
            level,
            include_p=include_p,
        )
        self.rns_runtime.forward_to_montgomery_(v, include_p=include_p)
        vpk0 = self.rns_runtime.montgomery_mul(
            v,
            pk0,
            include_p=include_p,
        )
        vpk1 = self.rns_runtime.montgomery_mul(
            v,
            pk1,
            include_p=include_p,
        )
        self.rns_runtime.inverse_to_standard_lazy_(vpk0, include_p=include_p)
        self.rns_runtime.inverse_to_standard_lazy_(vpk1, include_p=include_p)

        ct0 = self.rns_runtime.add_canonical(
            vpk0,
            pte0,
            include_p=include_p,
        )
        ct1 = self.rns_runtime.add_canonical(
            vpk1,
            e1_tiled,
            include_p=include_p,
        )
        return self._ciphertext_from_components(
            [ct0, ct1],
            level=level,
            scale=scale,
            polynomial_domain="coefficient",
            modulus_basis="QP" if include_p else "Q",
            residue_representation="standard",
        )

    def encrypt_message(
        self,
        message,
        public_key: PublicKey,
        *,
        level: int = 0,
        scale=None,
    ) -> Ciphertext:
        r"""Encode and encrypt one CKKS message at the provided per-value scale.

        For canonical slots $m$, compute

        $$
        p_i=\operatorname{SRound}\!\left(
          \Delta\,\mathcal{E}^{-1}_g(m)_i
        \right)
        $$

        and encrypt so that
        $c_0(X)+c_1(X)s(X)=p(X)+e(X)\pmod{B_\ell}$.

        Coefficients are quantized with the codec's stochastic-rounding law.
        Ordinary bounded coefficients use the native CUDA lift. If any scaled
        coefficient exceeds int64, safe coefficients are still stochastically
        rounded while already-integral wide binary64 coefficients are converted
        to Python integers and reduced into each RNS row. Both paths produce
        the same semantic ciphertext representation. Output has layout
        ``[component=2, *batch, limb, coefficient]``, engine integral
        dtype/device, coefficient domain, canonical standard residues, Q
        or QP ``prime_ids`` selected by the key, requested level, and actual
        scale $\Delta$. It owns independent storage; message/key inputs are not
        mutated.
        """

        self._validate_public_level(level)
        scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="Ciphertext",
        )
        include_p = public_key.modulus_basis == "QP"
        unscaled = self.plaintext_codec._inverse_embed_slots(message)
        prime_ids = self.rns_layout.prime_ids(
            level,
            include_p=include_p,
        )
        scaled = unscaled * scale
        max_abs_scaled = float(torch.max(torch.abs(scaled)).item())
        self._check_direct_decode_range(math.ceil(max_abs_scaled), level)
        dtype_info = torch.iinfo(self.config.torch_dtype)
        fits_integer_dtype = bool(
            torch.all(
                (scaled >= dtype_info.min) & (scaled < dtype_info.max)
            ).item()
        )
        if fits_integer_dtype:
            coefficients = self._rng.randround(scaled)
            plaintext_rns = self.rns_runtime.lift_integer_coefficients_exact(
                coefficients,
                level,
                include_p=include_p,
                max_abs=math.ceil(max_abs_scaled),
            )
        else:
            # The strict upper comparison is intentional: int64.max rounds to
            # 2**63 in binary64, so ``<= int64.max`` could admit an
            # unrepresentable positive coefficient at that threshold.
            safe_mask = (scaled >= dtype_info.min) & (scaled < dtype_info.max)
            safe_scaled = torch.where(
                safe_mask, scaled, torch.zeros_like(scaled)
            )
            safe_rounded = self._rng.randround(safe_scaled)
            flat_scaled = scaled.detach().cpu().reshape(-1, scaled.size(-1))
            flat_safe_mask = (
                safe_mask.detach().cpu().reshape(-1, safe_mask.size(-1))
            )
            flat_safe_rounded = (
                safe_rounded.detach().cpu().reshape(-1, safe_rounded.size(-1))
            )

            def rounded_wide_coefficient(
                row_index: int,
                coefficient_index: int,
            ) -> int:
                if flat_safe_mask[row_index, coefficient_index]:
                    return int(flat_safe_rounded[row_index, coefficient_index])
                value = float(flat_scaled[row_index, coefficient_index])
                if not value.is_integer():
                    raise RuntimeError(
                        "A coefficient outside the configured integer dtype "
                        "range retained a fractional binary64 part"
                    )
                return int(value)

            moduli = [
                self.montgomery_parameters.moduli[prime_id]
                for prime_id in prime_ids
            ]
            plaintext_rns = torch.tensor(
                [
                    [
                        [
                            rounded_wide_coefficient(
                                row_index,
                                coefficient_index,
                            )
                            % modulus
                            for coefficient_index, _ in enumerate(row)
                        ]
                        for modulus in moduli
                    ]
                    for row_index, row in enumerate(flat_scaled)
                ],
                dtype=self.config.torch_dtype,
                device=self.device,
            ).reshape(*unscaled.shape[:-1], len(prime_ids), unscaled.size(-1))
        return self._encrypt_rns_plaintext(
            plaintext_rns,
            public_key=public_key,
            level=level,
            scale=scale,
        )
