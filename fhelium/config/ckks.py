"""Define CKKS parameter presets and the validated configuration model."""

import math
from collections.abc import Mapping
from enum import Enum
from functools import cached_property
from typing import Any

import torch

from fhelium.config._prime_catalog import get_prime_catalog
from fhelium.config.security import (
    SecurityAssessment,
    _lookup_maximum_modulus_bits,
    assess_config_security,
)
from fhelium.errors import (
    InsufficientPrimeCatalogError,
    MessagePrimeCatalogEntryNotFoundError,
    ScalePrimeCatalogEntryNotFoundError,
    SecurityBudgetExceededError,
    SecurityParametersUnsupportedError,
)


class Preset(Enum):
    """Maintained CKKS parameter baselines.

    Each member name records the complex slot capacity, default scale-prime
    bit width, number of public levels, and integral tensor dtype in its
    baseline configuration. ``int32`` members use the 30-bit residue buffer;
    ``int64`` members use the 62-bit residue buffer. All baselines select the
    128-bit classical security category, Gaussian error standard deviation
    3.19, uniform-ternary secret sampling, and a ring-specific P-prime count.
    :meth:`CkksConfig.parse` accepts keyword overrides when an application
    needs a derived configuration.
    """

    slots8192_scale30_levels9_int64 = "slots8192-scale30-levels9-int64"
    slots8192_scale40_levels7_int64 = "slots8192-scale40-levels7-int64"
    slots8192_scale50_levels5_int64 = "slots8192-scale50-levels5-int64"
    slots16384_scale30_levels21_int64 = "slots16384-scale30-levels21-int64"
    slots16384_scale40_levels16_int64 = "slots16384-scale40-levels16-int64"
    slots16384_scale50_levels12_int64 = "slots16384-scale50-levels12-int64"
    slots32768_scale30_levels45_int64 = "slots32768-scale30-levels45-int64"
    slots32768_scale40_levels34_int64 = "slots32768-scale40-levels34-int64"
    slots32768_scale50_levels27_int64 = "slots32768-scale50-levels27-int64"
    slots65536_scale30_levels95_int64 = "slots65536-scale30-levels95-int64"
    slots65536_scale40_levels72_int64 = "slots65536-scale40-levels72-int64"
    slots65536_scale50_levels58_int64 = "slots65536-scale50-levels58-int64"
    slots8192_scale25_levels14_int32 = "slots8192-scale25-levels14-int32"
    slots16384_scale25_levels29_int32 = "slots16384-scale25-levels29-int32"
    slots32768_scale25_levels24_int32 = "slots32768-scale25-levels24-int32"
    slots65536_scale25_levels14_int32 = "slots65536-scale25-levels14-int32"


_PRESET_CONFIGS: dict[Preset, dict[str, int]] = {
    Preset.slots8192_scale30_levels9_int64: {
        "logN": 14,
        "scale_bits": 30,
        "num_scale_primes": 9,
        "num_p_primes": 1,
    },
    Preset.slots8192_scale40_levels7_int64: {
        "logN": 14,
        "scale_bits": 40,
        "num_scale_primes": 7,
        "num_p_primes": 1,
    },
    Preset.slots8192_scale50_levels5_int64: {
        "logN": 14,
        "scale_bits": 50,
        "num_scale_primes": 5,
        "num_p_primes": 1,
    },
    Preset.slots16384_scale30_levels21_int64: {
        "logN": 15,
        "scale_bits": 30,
        "num_scale_primes": 21,
        "num_p_primes": 2,
    },
    Preset.slots16384_scale40_levels16_int64: {
        "logN": 15,
        "scale_bits": 40,
        "num_scale_primes": 16,
        "num_p_primes": 2,
    },
    Preset.slots16384_scale50_levels12_int64: {
        "logN": 15,
        "scale_bits": 50,
        "num_scale_primes": 12,
        "num_p_primes": 2,
    },
    Preset.slots32768_scale30_levels45_int64: {
        "logN": 16,
        "scale_bits": 30,
        "num_scale_primes": 45,
        "num_p_primes": 4,
    },
    Preset.slots32768_scale40_levels34_int64: {
        "logN": 16,
        "scale_bits": 40,
        "num_scale_primes": 34,
        "num_p_primes": 4,
    },
    Preset.slots32768_scale50_levels27_int64: {
        "logN": 16,
        "scale_bits": 50,
        "num_scale_primes": 27,
        "num_p_primes": 4,
    },
    Preset.slots65536_scale30_levels95_int64: {
        "logN": 17,
        "scale_bits": 30,
        "num_scale_primes": 95,
        "num_p_primes": 6,
    },
    Preset.slots65536_scale40_levels72_int64: {
        "logN": 17,
        "scale_bits": 40,
        "num_scale_primes": 72,
        "num_p_primes": 6,
    },
    Preset.slots65536_scale50_levels58_int64: {
        "logN": 17,
        "scale_bits": 50,
        "num_scale_primes": 58,
        "num_p_primes": 6,
    },
    Preset.slots8192_scale25_levels14_int32: {
        "buffer_bit_length": 30,
        "logN": 14,
        "scale_bits": 25,
        "num_scale_primes": 14,
        "num_p_primes": 1,
    },
    Preset.slots16384_scale25_levels29_int32: {
        "buffer_bit_length": 30,
        "logN": 15,
        "scale_bits": 25,
        "num_scale_primes": 29,
        "num_p_primes": 2,
    },
    Preset.slots32768_scale25_levels24_int32: {
        "buffer_bit_length": 30,
        "logN": 16,
        "scale_bits": 25,
        "num_scale_primes": 24,
        "num_p_primes": 4,
    },
    Preset.slots65536_scale25_levels14_int32: {
        "buffer_bit_length": 30,
        "logN": 17,
        "scale_bits": 25,
        "num_scale_primes": 14,
        "num_p_primes": 6,
    },
}


class CkksConfig:
    r"""Immutable CKKS mathematical and security parameters.

    The configuration defines CKKS over
    $R = \mathbb{Z}[X]/(X^N+1)$, where $N=2^{\mathtt{logN}}$ and the complex
    slot count is $S=N/2$. Compatible values and keys share this cryptographic
    context.

    At public level $\ell$, the active ordinary ciphertext modulus is
    $Q_\ell=\prod_{i\in I_\ell}q_i$. ``num_scale_primes`` is the positive
    number of scale-prime rows selected into the Q chain and the number of
    public levels.
    The final public level contains the last scale prime and the structural
    base Q prime, so level zero has ``num_scale_primes - 1`` public one-level
    transitions. ``num_q_primes`` includes the additional structural base.
    The bootstrap-entry transition produces the one-prime structural basis.
    The key-switch modulus is $P=\prod_j p_j$.

    ``scale_bits`` selects ordinary scale primes and the default
    encoding/planning scale $\Delta_0=2^{\mathtt{scale\_bits}}$. Value
    creation selects $\Delta_0$ when the scale argument is omitted. Each live
    plaintext and ciphertext carries its actual scale $\Delta(v)$.
    ``base_prime_bits`` independently selects the structural base Q prime. An
    omitted value selects the message-prime catalog width. The packaged catalog
    accepts a provided value equal to ``scale_bits``.

    ``total_modulus_bits`` covers the complete QP parameter modulus, both
    $Q_0$ and $P$, and ``maximum_modulus_bits`` is the corresponding security
    budget. The built-in table supports Gaussian error standard deviation
    ``sigma=3.19`` and classical categories 128, 192, and 256. Engine
    construction checks the complete QP product before native initialization
    when ``enforce_security_budget`` is true. Disabling that check transfers
    parameter and sampler assessment to the caller.
    """

    def __init__(
        self,
        *,
        buffer_bit_length: int = 62,
        scale_bits: int = 40,
        base_prime_bits: int | None = None,
        logN: int = 15,
        num_scale_primes: int | None = 16,
        num_p_primes: int = 2,
        sigma: float = 3.19,
        security_bits: int = 128,
        enforce_security_budget: bool = True,
    ):
        if buffer_bit_length not in (30, 62):
            raise ValueError(
                "buffer_bit_length must be 32-2=30 or 64-2=62."
                "CPU and CUDA execution support int32 and int64."
            )
        if num_scale_primes is not None:
            if type(num_scale_primes) is not int:
                raise TypeError("num_scale_primes must be an integer")
            if num_scale_primes < 1:
                raise ValueError("num_scale_primes must be at least 1")
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
            raise TypeError("sigma must be a real number")
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be positive and finite")
        if type(security_bits) is not int:
            raise TypeError("security_bits must be an integer")
        if security_bits <= 0:
            raise ValueError("security_bits must be positive")
        if type(enforce_security_budget) is not bool:
            raise TypeError("enforce_security_budget must be a boolean")

        self.buffer_bit_length = buffer_bit_length
        self.scale_bits = scale_bits
        self.base_prime_bits = base_prime_bits
        self.logN = logN
        self._num_scale_primes_requested = num_scale_primes
        self.num_p_primes = num_p_primes
        self.sigma = sigma
        self.security_bits = security_bits
        self.enforce_security_budget = enforce_security_budget
        self._initialized = True

    def __setattr__(self, name: str, value: object) -> None:
        if self.__dict__.get("_initialized", False):
            raise AttributeError(
                "CkksConfig is immutable; construct a new config"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_initialized", False):
            raise AttributeError(
                "CkksConfig is immutable; construct a new config"
            )
        object.__delattr__(self, name)

    def dumps(self) -> dict[str, object]:
        """
        Serialize to a dictionary for easy saving or logging.
        """
        return {
            "buffer_bit_length": self.buffer_bit_length,
            "scale_bits": self.scale_bits,
            "base_prime_bits": self.base_prime_bits,
            "logN": self.logN,
            "num_scale_primes": self.num_scale_primes,
            "num_p_primes": self.num_p_primes,
            "sigma": self.sigma,
            "security_bits": self.security_bits,
            "enforce_security_budget": self.enforce_security_budget,
        }

    # ---- construction helpers ------------------------------------------------
    @classmethod
    def parse(
        cls,
        src: Mapping[str, Any] | Preset,
        **overrides: Any,
    ) -> "CkksConfig":
        """Resolve a parameter baseline into a CKKS configuration.

        ``src`` is either a maintained :class:`Preset` or a mapping accepted
        by :class:`CkksConfig`. Keyword overrides replace fields from that
        baseline before configuration validation and derived-value evaluation.
        """
        base = _PRESET_CONFIGS[src] if isinstance(src, Preset) else src
        merged = {**base, **overrides}
        return cls(**merged)

    # ---- simple derived values ----------------------------------------------
    @cached_property
    def N(self) -> int:
        r"""Ring dimension $N=2^{\mathtt{logN}}$ for $R$.

        The corresponding complex CKKS slot count is $S=N/2$.
        """

        return 1 << self.logN

    @cached_property
    def inverse_ntt_scale(self) -> tuple[int, ...]:
        r"""Return $N^{-1}\bmod m$ for every Q/P modulus row $m$.

        The result follows :attr:`moduli` order: ordinary $q_i$ rows followed
        by special $p_j$ rows. Each value is the normalization factor used by
        the inverse NTT for that row.
        """

        return tuple(pow(self.N, -1, qi) for qi in self.moduli)

    @cached_property
    def int_scale(self) -> int:
        r"""Integer default scale $\Delta_0=2^{\mathtt{scale\_bits}}$."""

        return 1 << self.scale_bits

    @cached_property
    def default_scale(self) -> float:
        r"""Binary64 default encoding and planning scale $\Delta_0$.

        Value creation selects this scale when its scale argument is omitted.
        Arithmetic reads and updates the actual scale stored on each value.
        """

        return float(self.int_scale)

    @cached_property
    def torch_dtype(self):
        return {30: torch.int32, 62: torch.int64}[self.buffer_bit_length]

    @cached_property
    def message_bits(self) -> int:
        """Legacy message-prime catalog width used for structural Q and P.

        This name does not denote CKKS message precision. Renaming the catalog
        selector and its packaged resources requires a separate versioned
        catalog migration.
        """

        # W - 2 bits, where W is the signed machine-word width.
        return self.buffer_bit_length - 2

    # ---- primes + security budget -------------------------------------------
    @cached_property
    def maximum_modulus_bits(self) -> int:
        """Built-in budget for the complete QP modulus bit width.

        Raises:
            SecurityParametersUnsupportedError: If this configuration does
                not match a table row.
        """

        maximum = _lookup_maximum_modulus_bits(
            ring_dimension=self.N,
            target_bits=self.security_bits,
            secret_distribution="ternary",
            error_stddev=self.sigma,
        )
        if maximum is None:
            raise SecurityParametersUnsupportedError(
                ring_dimension=self.N,
                target_bits=self.security_bits,
                secret_distribution="ternary",
                error_stddev=self.sigma,
                reason=(
                    "No built-in budget matches this configuration; "
                    "see the security guide for external "
                    "assessment requirements."
                ),
            )
        return maximum

    @cached_property
    def security_assessment(self) -> SecurityAssessment:
        """Structured table assessment of the complete QP modulus."""

        return assess_config_security(self)

    @cached_property
    def _message_and_p_primes(self) -> tuple[int, ...]:
        try:
            return tuple(
                get_prime_catalog().message_primes(
                    self.message_bits,
                    self.N,
                )
            )
        except KeyError as error:
            raise MessagePrimeCatalogEntryNotFoundError(
                coefficient_bits=self.message_bits,
                ring_dimension=self.N,
            ) from error

    @cached_property
    def _scale_primes(self) -> tuple[int, ...]:
        try:
            return tuple(
                get_prime_catalog().scale_primes(
                    self.scale_bits,
                    self.N,
                )
            )
        except KeyError as error:
            raise ScalePrimeCatalogEntryNotFoundError(
                scale_bits=self.scale_bits,
                ring_dimension=self.N,
            ) from error

    @cached_property
    def _base_and_p_primes(self) -> tuple[int, ...]:
        r"""Select the structural Q base followed by key-switch P primes.

        The default path preserves the existing catalog layout: the first
        message-width prime is Q's structural base and subsequent primes form
        $P$. When ``base_prime_bits`` is set, Q's base instead comes from a
        reserved scale-width catalog entry, while P still uses message-width
        primes.  Keeping P unchanged avoids coupling the bootstrap precision
        choice to hybrid key-switch decomposition.

        Returns:
            ``[q_structural_base, *p_primes]`` in canonical catalog order,
            where the first row belongs to $Q_\ell$ and the remaining rows
            multiply to $P$.

        Raises:
            ValueError: If the requested base width is unsupported.
            InsufficientPrimeCatalogError: If the catalog cannot provide all P
                rows after selecting the base.
        """

        if self.base_prime_bits is not None:
            if self.base_prime_bits != self.scale_bits:
                raise ValueError(
                    "The current prime catalog supports an overridden base "
                    "prime only when base_prime_bits == scale_bits"
                )
            # Use a scale-width prime not otherwise selected into the public
            # scale chain. P remains in the message-prime catalog.
            base_prime = self._scale_primes[-1]
            p_primes = self._message_and_p_primes[: self.num_p_primes]
            if len(p_primes) != self.num_p_primes:
                raise InsufficientPrimeCatalogError(
                    prime_kind="message",
                    ring_dimension=self.N,
                    required_count=self.num_p_primes,
                    available_count=len(p_primes),
                )
            return (base_prime, *p_primes)
        needed = 1 + self.num_p_primes
        primes = self._message_and_p_primes[:needed]
        if len(primes) != needed:
            raise InsufficientPrimeCatalogError(
                prime_kind="message",
                ring_dimension=self.N,
                required_count=needed,
                available_count=len(primes),
            )
        return tuple(primes)

    @cached_property
    def num_scale_primes(self) -> int:
        """Number of selected scale-prime rows and ordinary public levels.

        Public levels are ``[0, num_scale_primes)``. The final public level
        retains one scale prime plus the structural base, giving
        ``num_scale_primes - 1`` public transitions from level zero. The count
        is at least one. A configured count is validated against
        catalog capacity when :attr:`moduli` is constructed. An omitted count
        is filled greedily within the security-table modulus-bit budget.

        Raises:
            ValueError: If automatic derivation cannot fit one scale prime in
                the security budget.
        """
        if self._num_scale_primes_requested is not None:
            return self._num_scale_primes_requested

        # Greedily fill using exact integer products and bit widths.  This path
        # is opt-in through num_scale_primes=None; maintained presets carry
        # fixed counts so a future table update cannot alter their depth.
        modulus = math.prod(self._base_and_p_primes)
        radix = 1 << self.buffer_bit_length
        if any(4 * prime >= radix for prime in self._base_and_p_primes):
            raise ValueError(
                "The structural Q/P primes violate the native Montgomery "
                "requirement 4 * modulus < 2**buffer_bit_length"
            )
        count = 0
        reserved_primes = set(self._base_and_p_primes)
        while count < len(self._scale_primes):
            scale_prime = self._scale_primes[count]
            if scale_prime in reserved_primes or 4 * scale_prime >= radix:
                break
            candidate = modulus * scale_prime
            if (candidate - 1).bit_length() > self.maximum_modulus_bits:
                break
            modulus = candidate
            count += 1
        if count < 1:
            raise ValueError(
                "The security budget must permit at least one scale prime"
            )
        return count

    @cached_property
    def moduli(self) -> tuple[int, ...]:
        r"""Complete ordered QP parameter-modulus list.

        The order is ``[scale_q_primes, structural_q_prime, p_primes]``.
        The ordinary rows form $Q_0=\prod_i q_i$, and the special rows form
        $P=\prod_j p_j$.
        """
        scale_moduli = self._scale_primes[: self.num_scale_primes]
        if len(scale_moduli) != self.num_scale_primes:
            raise InsufficientPrimeCatalogError(
                prime_kind="scale",
                ring_dimension=self.N,
                required_count=self.num_scale_primes,
                available_count=len(scale_moduli),
            )
        moduli = scale_moduli + self._base_and_p_primes
        if len(set(moduli)) != len(moduli):
            raise ValueError(
                "The selected Q/P modulus chain contains duplicate primes; "
                "reduce num_scale_primes or choose another base-prime policy"
            )
        radix = 1 << self.buffer_bit_length
        for prime_id, modulus in enumerate(moduli):
            if 4 * modulus >= radix:
                raise ValueError(
                    "The selected Q/P modulus at prime_id "
                    f"{prime_id} violates the native Montgomery requirement "
                    "4 * modulus < 2**buffer_bit_length; reduce "
                    "num_scale_primes or select a smaller scale-prime family"
                )
        return tuple(moduli)

    @cached_property
    def q_moduli(self) -> tuple[int, ...]:
        r"""Ordered ordinary-prime rows whose level subsets form $Q_\ell$."""

        return self.moduli[: self.num_q_primes]

    @cached_property
    def p_moduli(self) -> tuple[int, ...]:
        r"""Ordered special-prime rows whose product is $P$."""

        return self.moduli[self.num_q_primes :]

    # ---- counts + security checks -------------------------------------------
    @cached_property
    def num_q_primes(self) -> int:
        r"""Number of ordinary Q primes, including one structural base prime.

        Therefore
        $\mathtt{num\_q\_primes}=\mathtt{num\_scale\_primes}+1$.
        """

        return self.num_scale_primes + 1

    @cached_property
    def total_num_primes(self) -> int:
        """Number of rows in the complete QP parameter basis."""

        return self.num_q_primes + self.num_p_primes

    @cached_property
    def total_modulus_bits(self) -> int:
        r"""Bit width $\lceil\log_2(Q_0P)\rceil$ of the complete QP modulus.

        This value covers both ordinary Q primes and special P primes; it is
        not the width of Q alone.
        """

        return (math.prod(self.moduli) - 1).bit_length()

    def validate_security_budget(self) -> SecurityAssessment:
        """Require a supported assessment that meets its QP budget.

        Returns:
            The immutable structured assessment when the budget is met.

        Raises:
            SecurityParametersUnsupportedError: If no built-in row
                matches this configuration.
            SecurityBudgetExceededError: If the complete QP modulus exceeds
                the matching table budget.
        """

        assessment = self.security_assessment
        if assessment.status == "unsupported":
            raise SecurityParametersUnsupportedError(
                ring_dimension=self.N,
                target_bits=self.security_bits,
                secret_distribution="ternary",
                error_stddev=self.sigma,
                reason=assessment.reason
                or "No built-in budget matches this configuration.",
            )
        if assessment.status == "exceeds":
            maximum = assessment.maximum_modulus_bits
            if maximum is None:
                raise RuntimeError(
                    "A below-budget assessment must report its table budget"
                )
            raise SecurityBudgetExceededError(
                scale_bits=self.scale_bits,
                ring_dimension=self.N,
                num_scale_primes=self.num_scale_primes,
                maximum_modulus_bits=maximum,
                requested_modulus_bits=assessment.modulus_bits,
            )
        return assessment

    # ---- pretty printing -----------------------------------------------------
    def __repr__(self) -> str:
        return (
            "CkksConfig("
            f"buffer_bit_length={self.buffer_bit_length}, "
            f"scale_bits={self.scale_bits}, "
            f"base_prime_bits={self.base_prime_bits!r}, "
            f"logN={self.logN}, "
            f"num_scale_primes={self.num_scale_primes}, "
            f"num_p_primes={self.num_p_primes}, "
            f"sigma={self.sigma!r}, "
            f"security_bits={self.security_bits}, "
            f"enforce_security_budget={self.enforce_security_budget})"
        )

    def __str__(self) -> str:
        assessment = self.security_assessment
        maximum_modulus_bits: int | str = (
            assessment.maximum_modulus_bits
            if assessment.maximum_modulus_bits is not None
            else "unsupported"
        )
        return (
            f"CkksConfig(buffer_bit_length={self.buffer_bit_length}, "
            f"scale_bits={self.scale_bits}, "
            f"base_prime_bits={self.base_prime_bits}, logN={self.logN}, "
            f"N={self.N}, num_slots={self.N // 2}, "
            f"num_scale_primes={self.num_scale_primes}, "
            f"num_q_primes={self.num_q_primes}, "
            f"num_p_primes={self.num_p_primes}, "
            f"total_num_primes={self.total_num_primes}, "
            f"total_modulus_bits={self.total_modulus_bits}, "
            f"maximum_modulus_bits={maximum_modulus_bits}, "
            f"sigma={self.sigma}, security_bits={self.security_bits}, "
            f"enforce_security_budget={self.enforce_security_budget})"
        )
