"""CKKS decryption and bounded coefficient reconstruction."""

from __future__ import annotations

import torch

from fhelium.config import CkksConfig
from fhelium.core import Ciphertext, Plaintext, SecretKey
from fhelium.engine.ckks_plaintext_codec import CkksPlaintextCodec
from fhelium.engine.rns.montgomery import MontgomeryParameters
from fhelium.engine.rns.layout import RnsLayout
from fhelium.engine.rns.runtime import RnsRuntime
from fhelium.native.wrapper import rns_ops


class CkksDecryptor:
    r"""Decrypt ciphertext phases to bounded approximate coefficients.

    The component evaluates two- or three-component secret-key phases in RNS
    and reconstructs only a bounded trailing-Q class into binary64
    coefficients for decoding. That reconstruction is not full-$Q_\ell$ CRT
    reconstruction. Key inventory and engine validation remain engine-owned.
    """

    def __init__(
        self,
        *,
        config: CkksConfig,
        device: torch.device,
        rns_layout: RnsLayout,
        rns_runtime: RnsRuntime,
        montgomery_parameters: MontgomeryParameters,
        plaintext_codec: CkksPlaintextCodec,
        engine_id: str,
    ) -> None:
        self.config = config
        self.device = device
        self.rns_layout = rns_layout
        self.rns_runtime = rns_runtime
        self.montgomery_parameters = montgomery_parameters
        self.plaintext_codec = plaintext_codec
        self.engine_id = engine_id
        self._decrypt_mixed_radix_cache: dict[
            tuple[int, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def __str__(self) -> str:
        return (
            f"CkksDecryptor(engine_id={self.engine_id}, "
            f"logN={self.config.logN}, device={self.device})"
        )

    __repr__ = __str__

    def decrypt(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey,
    ) -> Plaintext:
        r"""Decrypt into bounded binary64 coefficient reconstruction.

        Secret-key evaluation first forms

        $$
        u(X)=\sum_{j=0}^{d-1}c_j(X)s(X)^j\pmod{B_\ell},
        \qquad d\in\{2,3\},
        $$

        as coefficient-domain canonical standard RNS. The direct decrypt path then
        reconstructs the centered class from the trailing one/two Q primes
        into finite ``torch.float64`` ``[*batch, coefficient]`` on the engine
        device. This ``approximate_coefficients`` result is valid only for
        decoding; it is not full-$Q_\ell$ CRT and cannot return to RNS.
        Level and actual scale are preserved. Inputs are unchanged and output
        storage is independent.
        """

        if ciphertext.includes_p and secret_key.modulus_basis != "QP":
            raise ValueError(
                "Decrypting a QP Ciphertext requires a QP SecretKey"
            )
        plaintext_rns = self._decrypt_to_coefficient_standard_rns(
            ciphertext,
            secret_key,
        )
        coeff = self._reconstruct_tail_q_coefficients_float64(
            plaintext_rns,
            ciphertext,
        )
        return self.plaintext_codec._wrap_approximate_plaintext(
            coeff,
            level=ciphertext.level,
            scale=ciphertext.scale,
        )

    def _decrypt_to_coefficient_standard_rns(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey,
    ) -> torch.Tensor:
        r"""Evaluate the secret-key phase into coefficient-domain RNS.

        ``ciphertext`` has layout
        ``[component, *batch, limb, coefficient_or_ntt_index]`` with
        ``prime_ids``. Two-component input must be coefficient/standard and
        computes $c_0(X)+c_1(X)s(X)$; three-component input must be
        NTT/Montgomery and computes
        $c_0(X)+c_1(X)s(X)+c_2(X)s(X)^2$. ``secret_key`` is
        ``[level_zero_limb, ntt_index]`` in Montgomery form. Output is newly
        allocated ``[*batch, limb, coefficient]`` with engine integral
        dtype/device, the ciphertext Q/QP basis and rows, and canonical
        standard residues. Inputs are not mutated.
        """

        level = ciphertext.level
        secret_data = secret_key.data[
            self.rns_runtime.level_row_starts[level] :
        ]
        if not ciphertext.includes_p and secret_key.modulus_basis == "QP":
            secret_data = secret_data[: -self.config.num_p_primes]
        if ciphertext.component_count == 3:
            ciphertext.assert_state(
                polynomial_domain="ntt",
                residue_representation="montgomery",
                components=3,
            )
            d0 = ciphertext.c0.clone()
            self.rns_runtime.inverse_to_standard_(
                d0, include_p=ciphertext.includes_p
            )
            d1_s = self.rns_runtime.montgomery_mul(
                ciphertext.c1,
                secret_data,
                include_p=ciphertext.includes_p,
            )
            s2 = self.rns_runtime.montgomery_mul(
                secret_data,
                secret_data,
                include_p=ciphertext.includes_p,
            )
            d2_s2 = self.rns_runtime.montgomery_mul(
                ciphertext.c2,
                s2,
                include_p=ciphertext.includes_p,
            )
            self.rns_runtime.inverse_to_standard_lazy_(
                d1_s, include_p=ciphertext.includes_p
            )
            self.rns_runtime.inverse_to_standard_lazy_(
                d2_s2, include_p=ciphertext.includes_p
            )
            plaintext_rns = self.rns_runtime.add_lazy(
                d0, d1_s, include_p=ciphertext.includes_p
            )
            return self.rns_runtime.add_canonical(
                plaintext_rns,
                d2_s2,
                include_p=ciphertext.includes_p,
            )

        ciphertext.assert_state(
            polynomial_domain="coefficient",
            residue_representation="standard",
            components=2,
        )
        a = ciphertext.c1.clone()
        self.rns_runtime.forward_to_montgomery_(
            a, include_p=ciphertext.includes_p
        )
        sa = self.rns_runtime.montgomery_mul(
            a, secret_data, include_p=ciphertext.includes_p
        )
        self.rns_runtime.inverse_to_standard_lazy_(
            sa, include_p=ciphertext.includes_p
        )
        return self.rns_runtime.add_canonical(
            ciphertext.c0, sa, include_p=ciphertext.includes_p
        )

    def _reconstruct_tail_q_coefficients_float64(
        self,
        plaintext_rns: torch.Tensor,
        ciphertext: Ciphertext,
    ) -> torch.Tensor:
        r"""Reconstruct the bounded centered trailing-Q class into binary64.

        ``plaintext_rns`` is engine-integral
        ``[*batch, limb, coefficient]`` on the engine device in coefficient
        domain and standard form; rows map exactly to ``ciphertext.prime_ids``.
        The method selects the trailing one/two active Q rows, canonicalizes
        them, performs mixed-radix centering modulo their product $D_\ell$,
        and converts the centered representative in
        $[-\lfloor D_\ell/2\rfloor,\lceil D_\ell/2\rceil)$ to
        ``torch.float64``. The functional output is
        ``[*batch, coefficient]`` on the same device and does not alias input.
        Conversion to binary64 can round wide integers; this is not
        full-$Q_\ell$ CRT reconstruction.
        """

        q_prime_ids = self.rns_layout.prime_ids(ciphertext.level)
        # Use the trailing scale/base pair for client-side decoding. Its product
        # is the centered-coefficient dynamic range of the direct
        # decoder. The convenience encoder checks its coefficient bound so
        # supported message inputs cannot silently alias this basis.
        source_prime_ids = tuple(q_prime_ids[-2:])
        source_at = [
            ciphertext.prime_ids.index(prime_id)
            for prime_id in source_prime_ids
        ]
        source = plaintext_rns.index_select(
            -2,
            torch.tensor(source_at, device=plaintext_rns.device),
        ).clone()
        start = source_prime_ids[0]
        stop = source_prime_ids[-1] + 1
        source_params = self.rns_runtime.rns_parameter_tensor[:, start:stop]
        rns_ops.canonicalize_residues_(source, source_params)
        source_moduli = tuple(
            int(self.montgomery_parameters.moduli[index])
            for index in source_prime_ids
        )
        if len(source_prime_ids) == 1:
            modulus = source_moduli[0]
            centered = source.squeeze(-2)
            return torch.where(
                centered > modulus // 2,
                centered - modulus,
                centered,
            ).to(torch.float64)

        normalizers, propagation = self._decrypt_mixed_radix_tables(
            source_prime_ids
        )
        lo, hi, neg_lo, neg_hi = source_params[1:5]

        def decompose(residues: torch.Tensor) -> torch.Tensor:
            digits = rns_ops.mixed_radix_decompose(
                residues,
                normalizers,
                propagation,
                lo,
                hi,
                neg_lo,
                neg_hi,
            )
            rns_ops.canonicalize_residues_(digits, source_params)
            return digits

        positive_digits = decompose(source)
        negative = self._mixed_radix_is_above_half(
            positive_digits,
            source_moduli,
        )
        moduli_tensor = torch.tensor(
            source_moduli,
            dtype=source.dtype,
            device=source.device,
        ).view(*([1] * (source.ndim - 2)), -1, 1)
        negated_source = torch.where(
            source == 0, source, moduli_tensor - source
        )
        negative_digits = decompose(negated_source)

        def reconstruct(digits: torch.Tensor) -> torch.Tensor:
            value = digits[..., -1, :].to(torch.float64)
            for row in range(len(source_moduli) - 2, -1, -1):
                value = value * source_moduli[row] + digits[..., row, :]
            return value

        positive_value = reconstruct(positive_digits)
        negative_value = reconstruct(negative_digits)
        return torch.where(negative, -negative_value, positive_value)

    def _decrypt_mixed_radix_tables(
        self,
        source_prime_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return cached tables for ``source_prime_ids`` order.

        For $L$ source primes, ``normalizers`` has shape ``[L-1]`` and
        ``propagation`` has shape ``[L-1, L]``. Both use engine integral
        dtype/device and contain Montgomery-form modular scalars aligned to
        stable parameter prime identifiers. Cached tensors are engine-owned;
        callers receive aliases and must not mutate them.
        """

        cached = self._decrypt_mixed_radix_cache.get(source_prime_ids)
        if cached is not None:
            return cached
        moduli = self.montgomery_parameters.moduli
        source_moduli = [int(moduli[index]) for index in source_prime_ids]
        prefix_products = []
        product = source_moduli[0]
        for index in range(len(source_moduli) - 1):
            if index:
                product *= source_moduli[index]
            prefix_products.append(product)
        normalizers = []
        propagation = torch.zeros(
            (len(source_moduli) - 1, len(source_moduli)),
            dtype=self.config.torch_dtype,
            device=self.device,
        )
        for component_index, prefix in enumerate(prefix_products):
            next_modulus = source_moduli[component_index + 1]
            normalizers.append(
                (pow(prefix, -1, next_modulus) * self.montgomery_parameters.R)
                % next_modulus
            )
            for target_index in range(
                component_index + 2,
                len(source_moduli),
            ):
                propagation[component_index, target_index] = (
                    prefix * self.montgomery_parameters.R
                ) % source_moduli[target_index]
        cached = (
            torch.tensor(
                normalizers,
                dtype=self.config.torch_dtype,
                device=self.device,
            ),
            propagation,
        )
        self._decrypt_mixed_radix_cache[source_prime_ids] = cached
        return cached

    @staticmethod
    def _mixed_radix_is_above_half(
        digits: torch.Tensor,
        moduli: tuple[int, ...],
    ) -> torch.Tensor:
        r"""Compare mixed-radix digits with half the represented modulus.

        ``digits`` has layout ``[*batch, digit, coefficient]`` and integral
        dtype/device; digit row $i$ uses radix ``moduli[i]``. The result is a
        newly allocated boolean ``[*batch, coefficient]`` mask that is true
        exactly when the represented nonnegative integer exceeds
        $\lfloor\prod_i q_i/2\rfloor$. Inputs are not mutated.
        """

        half = 1
        for modulus in moduli:
            half *= modulus
        half //= 2
        half_digits = []
        for modulus in moduli:
            half, digit = divmod(half, modulus)
            half_digits.append(digit)
        equal = torch.ones_like(digits[..., 0, :], dtype=torch.bool)
        above = torch.zeros_like(equal)
        for row in range(len(moduli) - 1, -1, -1):
            above |= equal & (digits[..., row, :] > half_digits[row])
            equal &= digits[..., row, :] == half_digits[row]
        return above

    def decrypt_message(
        self,
        ciphertext: Ciphertext,
        secret_key: SecretKey,
        *,
        is_real: bool = False,
    ):
        r"""Decrypt and decode with the ciphertext's actual scale.

        This composes :meth:`decrypt`'s bounded approximate-coefficient
        reconstruction with
        $m_{\mathrm{approx}}=\mathcal{E}_g(p)/\Delta(ciphertext)$. Output is
        CPU ``[*batch, slot]`` with final extent $S=N/2$, complex unless
        ``is_real=True``. It is approximate and does not provide a
        decrypt-to-encrypt representation round trip. Inputs are unchanged.
        """

        plaintext = self.decrypt(ciphertext, secret_key)
        return self.plaintext_codec.decode(plaintext, is_real=is_real)
