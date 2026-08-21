"""CKKS message and Plaintext conversion for one local engine."""

from __future__ import annotations

from collections.abc import Callable

import torch

from fhelium.config import CkksConfig
from fhelium.core import (
    CkksContextSpec,
    ModulusBasis,
    Plaintext,
)
from fhelium.core.scale import coerce_scale
from fhelium.engine import slot_embedding
from fhelium.engine.rns.layout import RnsLayout
from fhelium.engine.rns.runtime import RnsRuntime
from fhelium.rng import Csprng


class CkksPlaintextCodec:
    r"""Convert slots through typed CKKS plaintext representations.

    The codec owns the semantic path

    $$
    m\in\mathbb{C}^S
    \longrightarrow p\in R
    \longrightarrow (p\bmod q_i)_{i\in I_\ell}
    $$

    and the inverse slot embedding for decoding. Encoding returns integer
    coefficients; RNS reduction is a separate transition. Decoding accepts
    integer coefficients or bounded binary64 approximate decrypt
    coefficients, never unreconstructed RNS rows.
    """

    def __init__(
        self,
        *,
        config: CkksConfig,
        context: CkksContextSpec,
        device: torch.device,
        rng: Csprng,
        rns_layout: RnsLayout,
        rns_runtime: RnsRuntime,
        galois_generator: int,
        engine_id: str,
        validate_public_level: Callable[[object], int],
    ) -> None:
        self.config = config
        self.context = context
        self.device = device
        self._rng = rng
        self.rns_layout = rns_layout
        self.rns_runtime = rns_runtime
        self.galois_generator = galois_generator
        self.engine_id = engine_id
        self._validate_public_level = validate_public_level

    @property
    def num_slots(self) -> int:
        """Semantic CKKS slot count fixed by the bound configuration."""

        return self.config.N // 2

    def __str__(self) -> str:
        return (
            f"CkksPlaintextCodec(engine_id={self.engine_id}, "
            f"logN={self.config.logN}, slots={self.num_slots}, "
            f"scale_bits={self.config.scale_bits}, "
            f"device={self.device})"
        )

    __repr__ = __str__

    def plaintext(self, message, *, level: int = 0, scale=None) -> Plaintext:
        r"""Create an unencoded slots-only :class:`Plaintext`.

        No embedding or quantization occurs. The input is detached, cloned,
        and moved to the engine device while preserving its inferred dtype and
        shape; a later encode interprets the final axis as slots and preserves
        leading batch axes. The value records level and actual scale
        $\Delta(p)$ but has no RNS domain, basis, residue form, or
        ``prime_ids``.
        """

        self._validate_public_level(level)
        scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="Plaintext",
        )
        return Plaintext(
            message=torch.as_tensor(message).detach().clone().to(self.device),
            level=level,
            scale=float(scale),
        )

    def _wrap_integer_plaintext(
        self,
        coeff: torch.Tensor,
        *,
        level: int,
        scale: float,
    ) -> Plaintext:
        """Wrap ``[*batch, coefficient]`` storage without copying.

        ``coeff`` is engine-dtype integral data on the engine device with
        final extent $N$; the result is coefficient-domain
        ``integer_coefficients`` with no RNS basis or ``prime_ids``.
        """

        return Plaintext(
            message=None,
            level=level,
            scale=scale,
            data=coeff,
            context_id=self.context.context_id,
            representation="integer_coefficients",
            polynomial_domain="coefficient",
        )

    def _wrap_approximate_plaintext(
        self,
        coefficients: torch.Tensor,
        *,
        level: int,
        scale: float,
    ) -> Plaintext:
        r"""Wrap bounded binary64 decrypt coefficients without copying.

        ``coefficients`` has layout ``[*batch, coefficient]``, dtype
        ``torch.float64``, engine device, and final extent $N$. It represents
        the centered trailing-Q class used only for decoding, not a
        full-$Q_\ell$ CRT inverse and not an encryptable/RNS-convertible value.
        """

        return Plaintext(
            message=None,
            level=level,
            scale=scale,
            data=coefficients,
            context_id=self.context.context_id,
            representation="approximate_coefficients",
            polynomial_domain="coefficient",
        )

    def _slot_tensor(self, message) -> torch.Tensor:
        r"""Materialize ``[*batch, slot]`` with final extent $S=N/2$.

        Scalars repeat to all slots; shorter final axes are zero padded.
        Leading batch axes and dtype are preserved, and the result is consumed
        on the engine device. The result can alias an already materialized
        full-slot input; callers do not mutate it.
        """

        return slot_embedding.make_slot_tensor(
            message,
            num_slots=self.num_slots,
            device=self.device,
        )

    def _inverse_embed_slots(
        self,
        message,
    ) -> torch.Tensor:
        r"""Return $\mathcal{E}^{-1}_g(m)$ before scaling or rounding.

        The input semantic shape is ``[*batch, slot]`` and the output is a
        functional binary64 real ``[*batch, coefficient]`` tensor with final
        extent $N$ on the engine device.
        """

        return slot_embedding.inverse_embed_slots(
            self._slot_tensor(message),
            device=self.device,
            galois_generator=self.galois_generator,
        )

    def _encode_slots_to_integer_coefficients(
        self,
        message,
        *,
        scale: float,
    ) -> torch.Tensor:
        r"""Return integer coefficients for CKKS encoding.

        For canonical slots $m$, compute

        $$
        p_i=\operatorname{SRound}\!\left(
          \Delta\,\mathcal{E}^{-1}_g(m)_i
        \right).
        $$

        The result is a new engine-integral ``[*batch, coefficient]`` tensor
        with final extent $N$ on the engine device. It is coefficient-domain
        storage, not RNS data.
        """

        return slot_embedding.encode_slots(
            self._slot_tensor(message),
            rng=self._rng,
            scale=scale,
            device=self.device,
            galois_generator=self.galois_generator,
        )

    def encode(
        self,
        message,
        *,
        level: int = 0,
        scale=None,
    ) -> Plaintext:
        r"""Encode slots into one integer-coefficient plaintext.

        This produces

        $$
        p_i=\operatorname{SRound}\!\left(
          \Delta\,\mathcal{E}^{-1}_g(m)_i
        \right),\qquad
        \mathbb{E}[\operatorname{SRound}(x)]=x,
        $$

        independently of the selected RNS level. The functional result is
        ``integer_coefficients`` with layout
        ``[*batch, coefficient]``, final extent $N$, engine integral dtype and
        device, coefficient domain, actual scale $\Delta$, and no RNS basis or
        ``prime_ids``. Use :meth:`integer_coefficients_to_rns` for modular reduction.
        """

        self._validate_public_level(level)
        scale = coerce_scale(
            self.config.default_scale if scale is None else scale,
            value_name="Plaintext",
        )
        return self._wrap_integer_plaintext(
            self._encode_slots_to_integer_coefficients(message, scale=scale),
            level=level,
            scale=scale,
        )

    def integer_coefficients_to_rns(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: ModulusBasis = "Q",
    ) -> Plaintext:
        r"""Reduce integer coefficients to standard RNS.

        For each active row ``prime_ids[i]`` with modulus $q_i$, compute
        $r_{i,j}=p_j\bmod q_i$. Input layout
        ``[*batch, coefficient]`` becomes
        ``[*batch, limb, coefficient]`` with engine integral dtype/device and
        final extent $N$. The output basis is $Q_\ell$ or $Q_\ell P$ exactly
        as requested; its ordered ``prime_ids`` map every limb row. Level and
        actual scale are preserved. No NTT, Montgomery conversion, rounding,
        or CRT reconstruction occurs, and the functional output does not alias
        input coefficient storage.
        """

        return self._integer_coefficients_to_rns(
            plaintext,
            modulus_basis=modulus_basis,
        )

    def _integer_coefficients_to_rns(
        self,
        plaintext: Plaintext,
        *,
        modulus_basis: ModulusBasis,
    ) -> Plaintext:
        """Implement :meth:`integer_coefficients_to_rns` after validating the representation."""

        if not isinstance(plaintext, Plaintext):
            raise TypeError(
                "RNS conversion expects Plaintext, got "
                f"{type(plaintext).__name__}"
            )
        if not plaintext.is_integer_coefficients:
            raise ValueError(
                "RNS conversion requires representation="
                "'integer_coefficients'; approximate decrypt coefficients "
                "cannot be converted back to RNS"
            )
        plaintext = self._ensure_integer_plaintext(plaintext)
        data = plaintext.data
        if data is None:
            raise RuntimeError(
                "Coefficient Plaintext data was not materialized"
            )
        include_p = modulus_basis == "QP"
        data = self.rns_runtime.lift_integer_coefficients_exact(
            data,
            plaintext.level,
            include_p=include_p,
        )
        return Plaintext(
            message=None,
            level=plaintext.level,
            scale=plaintext.scale,
            data=data,
            context_id=self.context.context_id,
            representation="rns",
            polynomial_domain="coefficient",
            modulus_basis=modulus_basis,
            residue_representation="standard",
            prime_ids=self.rns_layout.prime_ids(
                plaintext.level,
                include_p=include_p,
            ),
        )

    def _ensure_integer_plaintext(self, plaintext: Plaintext) -> Plaintext:
        """Return an integer-coefficient value without mutating the input."""

        context_id = self.context.context_id
        if plaintext.context_id not in (None, context_id):
            raise ValueError(
                "Plaintext belongs to another CKKS context: "
                f"{plaintext.context_id} != {context_id}"
            )
        if plaintext.data is not None:
            return self._validate_coefficient_plaintext(
                plaintext, allow_approximate=False
            )
        if plaintext.message is None:
            raise ValueError(
                "Cannot encode a Plaintext without a source message"
            )
        return self.encode(
            plaintext.message,
            level=plaintext.level,
            scale=plaintext.scale,
        )

    def _validate_coefficient_plaintext(
        self,
        plaintext: Plaintext,
        *,
        allow_approximate: bool,
    ) -> Plaintext:
        """Validate one engine-local integer or approximate coefficient value."""

        if not isinstance(plaintext, Plaintext):
            raise TypeError(
                f"Expected Plaintext, got {type(plaintext).__name__}"
            )
        allowed = (
            ("integer_coefficients", "approximate_coefficients")
            if allow_approximate
            else ("integer_coefficients",)
        )
        if plaintext.representation not in allowed:
            expected = " or ".join(repr(item) for item in allowed)
            raise ValueError(
                f"This operation requires representation={expected}, got "
                f"{plaintext.representation!r}. Encode a separate "
                "integer-coefficient Plaintext when required."
            )
        self._validate_public_level(plaintext.level)
        if plaintext.context_id != self.context.context_id:
            raise ValueError("Plaintext belongs to another CKKS context")
        if (
            plaintext.polynomial_domain != "coefficient"
            or plaintext.modulus_basis is not None
            or plaintext.residue_representation is not None
            or plaintext.prime_ids
        ):
            raise ValueError(
                "Coefficient Plaintext has incompatible arithmetic metadata"
            )
        data = plaintext.data
        if not isinstance(data, torch.Tensor):
            raise TypeError("Coefficient Plaintext data must be a torch.Tensor")
        if data.layout != torch.strided:
            raise TypeError(
                "Coefficient Plaintext data must use dense strided storage"
            )
        if data.size(-1) != self.config.N:
            raise ValueError(
                "Plaintext encoded polynomial ring dimension does not match "
                f"engine: {data.size(-1)} != {self.config.N}"
            )
        if data.device != self.device:
            raise ValueError(
                "Plaintext encoded data is on the wrong local device: "
                f"{data.device} != {self.device}"
            )
        if plaintext.is_integer_coefficients:
            if data.dtype != self.config.torch_dtype:
                raise TypeError(
                    "integer_coefficients Plaintext dtype does not match "
                    f"engine: {data.dtype} != {self.config.torch_dtype}"
                )
        else:
            if data.dtype != torch.float64:
                raise TypeError(
                    "approximate_coefficients Plaintext data must use float64"
                )
            if not bool(torch.all(torch.isfinite(data)).item()):
                raise ValueError(
                    "approximate_coefficients Plaintext data must be finite"
                )
        return plaintext

    def decode(self, plaintext: Plaintext, *, is_real: bool = False):
        r"""Decode integer or approximate coefficients into CPU slots.

        For coefficient data $p$ and the value's actual scale $\Delta(p)$,

        $$
        m_{\mathrm{approx}}=
        \mathcal{E}_g(p)/\Delta(p).
        $$

        The accepted payload is integral ``integer_coefficients`` or
        finite ``torch.float64`` ``approximate_coefficients``, each with layout
        ``[*batch, coefficient]``, final extent $N$, and engine device. Slots
        input is encoded first under its stored scale. RNS input is rejected;
        decode never performs implicit CRT reconstruction. The functional
        result has shape ``[*batch, slot]``, final extent $S=N/2$, resides on
        CPU, and is complex unless ``is_real=True`` selects its real part.
        """

        if not isinstance(plaintext, Plaintext):
            raise TypeError(
                f"decode expects Plaintext, got {type(plaintext).__name__}"
            )
        if plaintext.is_slots:
            coefficient_plaintext = self._ensure_integer_plaintext(plaintext)
        elif plaintext.representation in (
            "integer_coefficients",
            "approximate_coefficients",
        ):
            coefficient_plaintext = self._validate_coefficient_plaintext(
                plaintext, allow_approximate=True
            )
        else:
            raise ValueError(
                "decode requires integer_coefficients or "
                "approximate_coefficients; RNS reconstruction is not implicit"
            )
        if coefficient_plaintext.data is None:
            raise RuntimeError(
                "Coefficient Plaintext data was not materialized"
            )
        decoded = slot_embedding.decode_slots(
            coefficient_plaintext.data,
            scale=coefficient_plaintext.scale,
            galois_generator=self.galois_generator,
        )
        result = decoded[..., : self.num_slots].cpu()
        return result.real if is_real else result
