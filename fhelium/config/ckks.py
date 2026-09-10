"""Define CKKS parameter presets and the validated configuration model."""

import math
from collections.abc import Mapping
from enum import Enum
from functools import cached_property
from typing import Any, cast

from fhelium._version import __version__
from fhelium.config._prime_catalog import get_prime_catalog
from fhelium.config.security import (
    SecurityAssessment,
    _lookup_maximum_modulus_bits,
    assess_config_security,
)
from fhelium.errors import (
    InsufficientPrimeCatalogError,
    ScalingPrimeCatalogEntryNotFoundError,
    SpecialPrimeCatalogEntryNotFoundError,
    SecurityBudgetExceededError,
    SecurityParametersUnsupportedError,
)


class Preset(Enum):
    """Built-in CKKS parameter presets.

    Each member name records the complex slot capacity, default scale bits,
    maximum public depth, and the Engine's selected residue dtype for its
    prime set. All baselines select the 128-bit classical
    security category, Gaussian error standard deviation
    3.19, uniform-ternary secret sampling, and a ring-specific P-prime count.
    :meth:`CkksConfig.parse` accepts keyword overrides when an application
    needs a derived configuration.
    """

    slots8192_scale30_depth9_int64 = "slots8192-scale30-depth9-int64"
    slots8192_scale40_depth7_int64 = "slots8192-scale40-depth7-int64"
    slots8192_scale50_depth5_int64 = "slots8192-scale50-depth5-int64"
    slots16384_scale30_depth21_int64 = "slots16384-scale30-depth21-int64"
    slots16384_scale40_depth16_int64 = "slots16384-scale40-depth16-int64"
    slots16384_scale50_depth12_int64 = "slots16384-scale50-depth12-int64"
    slots32768_scale30_depth45_int64 = "slots32768-scale30-depth45-int64"
    slots32768_scale40_depth34_int64 = "slots32768-scale40-depth34-int64"
    slots32768_scale50_depth27_int64 = "slots32768-scale50-depth27-int64"
    slots32768_scale50_depth29_int64 = "slots32768-scale50-depth29-int64"
    slots65536_scale30_depth95_int64 = "slots65536-scale30-depth95-int64"
    slots65536_scale40_depth72_int64 = "slots65536-scale40-depth72-int64"
    slots65536_scale50_depth58_int64 = "slots65536-scale50-depth58-int64"
    slots8192_scale25_depth14_int32 = "slots8192-scale25-depth14-int32"
    slots16384_scale25_depth29_int32 = "slots16384-scale25-depth29-int32"
    slots32768_scale25_depth24_int32 = "slots32768-scale25-depth24-int32"
    slots65536_scale25_depth14_int32 = "slots65536-scale25-depth14-int32"


_PRESET_CONFIGS: dict[Preset, dict[str, int]] = {
    Preset.slots8192_scale30_depth9_int64: {
        "logN": 14,
        "scaling_prime_bits": 30,
        "max_depth": 9,
        "num_p_primes": 1,
    },
    Preset.slots8192_scale40_depth7_int64: {
        "logN": 14,
        "scaling_prime_bits": 40,
        "max_depth": 7,
        "num_p_primes": 1,
    },
    Preset.slots8192_scale50_depth5_int64: {
        "logN": 14,
        "scaling_prime_bits": 50,
        "max_depth": 5,
        "num_p_primes": 1,
    },
    Preset.slots16384_scale30_depth21_int64: {
        "logN": 15,
        "scaling_prime_bits": 30,
        "max_depth": 21,
        "num_p_primes": 2,
    },
    Preset.slots16384_scale40_depth16_int64: {
        "logN": 15,
        "scaling_prime_bits": 40,
        "max_depth": 16,
        "num_p_primes": 2,
    },
    Preset.slots16384_scale50_depth12_int64: {
        "logN": 15,
        "scaling_prime_bits": 50,
        "max_depth": 12,
        "num_p_primes": 2,
    },
    Preset.slots32768_scale30_depth45_int64: {
        "logN": 16,
        "scaling_prime_bits": 30,
        "max_depth": 45,
        "num_p_primes": 4,
    },
    Preset.slots32768_scale40_depth34_int64: {
        "logN": 16,
        "scaling_prime_bits": 40,
        "max_depth": 34,
        "num_p_primes": 4,
    },
    Preset.slots32768_scale50_depth27_int64: {
        "logN": 16,
        "scaling_prime_bits": 50,
        "max_depth": 27,
        "num_p_primes": 4,
    },
    Preset.slots32768_scale50_depth29_int64: {
        "logN": 16,
        "scaling_prime_bits": 50,
        "max_depth": 29,
        "num_p_primes": 4,
    },
    Preset.slots65536_scale30_depth95_int64: {
        "logN": 17,
        "scaling_prime_bits": 30,
        "max_depth": 95,
        "num_p_primes": 6,
    },
    Preset.slots65536_scale40_depth72_int64: {
        "logN": 17,
        "scaling_prime_bits": 40,
        "max_depth": 72,
        "num_p_primes": 6,
    },
    Preset.slots65536_scale50_depth58_int64: {
        "logN": 17,
        "scaling_prime_bits": 50,
        "max_depth": 58,
        "num_p_primes": 6,
    },
    Preset.slots8192_scale25_depth14_int32: {
        "special_prime_bits": 28,
        "logN": 14,
        "scaling_prime_bits": 25,
        "max_depth": 14,
        "num_p_primes": 1,
    },
    Preset.slots16384_scale25_depth29_int32: {
        "special_prime_bits": 28,
        "logN": 15,
        "scaling_prime_bits": 25,
        "max_depth": 29,
        "num_p_primes": 2,
    },
    Preset.slots32768_scale25_depth24_int32: {
        "special_prime_bits": 28,
        "logN": 16,
        "scaling_prime_bits": 25,
        "max_depth": 24,
        "num_p_primes": 4,
    },
    Preset.slots65536_scale25_depth14_int32: {
        "special_prime_bits": 28,
        "logN": 17,
        "scaling_prime_bits": 25,
        "max_depth": 14,
        "num_p_primes": 6,
    },
}


def _dumped_moduli(
    value: object,
    *,
    name: str,
    require_nonempty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        type(modulus) is not int for modulus in value
    ):
        raise TypeError(
            f"Serialized CKKS {name} must be a sequence of integers"
        )
    if require_nonempty and not value:
        raise ValueError(f"Serialized CKKS {name} must be nonempty")
    return tuple(value)


def _preset_parameters(preset: Preset) -> dict[str, object]:
    """Resolve one built-in recipe into exact Q and P moduli."""

    recipe = _PRESET_CONFIGS[preset]
    log_n = recipe["logN"]
    degree = 1 << log_n
    scaling_bits = recipe["scaling_prime_bits"]
    special_bits = recipe.get("special_prime_bits", 60)
    max_depth = recipe["max_depth"]
    p_count = recipe["num_p_primes"]
    catalog = get_prime_catalog()
    try:
        scaling_candidates = catalog.scaling_primes(scaling_bits, degree)
    except KeyError as error:
        raise ScalingPrimeCatalogEntryNotFoundError(
            prime_bits=scaling_bits, ring_dimension=degree
        ) from error
    scaling_count = max_depth
    if len(scaling_candidates) < scaling_count:
        raise InsufficientPrimeCatalogError(
            prime_kind="scaling",
            ring_dimension=degree,
            required_count=scaling_count,
            available_count=len(scaling_candidates),
        )
    scaling = tuple(scaling_candidates[:scaling_count])
    try:
        special_candidates = catalog.special_primes(special_bits, degree)
    except KeyError as error:
        raise SpecialPrimeCatalogEntryNotFoundError(
            prime_bits=special_bits, ring_dimension=degree
        ) from error
    if len(scaling_candidates) < scaling_count + 1:
        raise InsufficientPrimeCatalogError(
            prime_kind="scaling",
            ring_dimension=degree,
            required_count=(scaling_count + 1),
            available_count=len(scaling_candidates),
        )
    terminal = scaling_candidates[scaling_count]
    selected_q = (*scaling, terminal)
    special_available = [
        prime for prime in special_candidates if prime not in selected_q
    ]
    if len(special_available) < p_count:
        raise InsufficientPrimeCatalogError(
            prime_kind="special",
            ring_dimension=degree,
            required_count=p_count,
            available_count=len(special_available),
        )
    special = tuple(special_available[:p_count])
    return {
        "default_scale": float(1 << scaling_bits),
        "q_depth_groups": tuple((prime,) for prime in scaling)
        + ((terminal,),),
        "p_moduli": special,
        "logN": log_n,
    }


class CkksConfig:
    r"""Immutable CKKS mathematical and security parameters.

    The configuration defines CKKS over
    $R=\mathbb{Z}[X]/(X^N+1)$, where $N=2^{\mathtt{logN}}$, together with
    the exact ciphertext-modulus chain and hybrid key-switch modulus.

    ``q_depth_groups`` contains ordered Q groups. At depth ``d``, the active
    Q basis is the concatenation of groups ``d:``. A rescale removes group
    ``d`` and advances to ``d + 1``. The last group is the terminal basis:
    ``max_depth == len(q_depth_groups) - 1`` and no further rescale exists there.
    Depth identifies the active basis; it does not count multiplications or
    record how the value reached that basis.

    ``p_moduli`` contains the special primes whose product is the key-switch
    modulus P. ``default_scale`` is used only when value creation omits a
    scale; every live plaintext and ciphertext carries its own actual scale.
    The Engine selects the residue dtype and Montgomery radix from these exact
    primes, and each device-local RNS context materializes the corresponding
    arithmetic tables.
    """

    def __init__(
        self,
        *,
        default_scale: float,
        q_depth_groups: tuple[tuple[int, ...], ...],
        p_moduli: tuple[int, ...],
        logN: int,
        sigma: float = 3.19,
        security_bits: int = 128,
        enforce_security_budget: bool = True,
        galois_generator: int = 3,
    ) -> None:
        if type(logN) is not int or logN < 1:
            raise ValueError("logN must be a positive integer")
        if not isinstance(default_scale, (int, float)) or isinstance(
            default_scale, bool
        ):
            raise TypeError("default_scale must be a real number")
        default_scale = float(default_scale)
        if not math.isfinite(default_scale) or default_scale <= 0.0:
            raise ValueError("default_scale must be positive and finite")
        if not isinstance(q_depth_groups, (list, tuple)) or any(
            not isinstance(group, (list, tuple)) or not group
            for group in q_depth_groups
        ):
            raise TypeError(
                "q_depth_groups must be a sequence of nonempty prime groups"
            )
        groups = tuple(
            _dumped_moduli(
                group,
                name=f"q_depth_groups[{index}]",
                require_nonempty=True,
            )
            for index, group in enumerate(q_depth_groups)
        )
        if not groups:
            raise ValueError(
                "q_depth_groups must contain at least one group"
            )
        special = _dumped_moduli(
            p_moduli, name="p_moduli", require_nonempty=True
        )
        all_moduli = tuple(prime for group in groups for prime in group) + special
        if len(set(all_moduli)) != len(all_moduli):
            raise ValueError("Q and P must contain distinct primes")
        if any(prime <= 2 or prime % 2 == 0 for prime in all_moduli):
            raise ValueError("Every Q/P modulus must be an odd integer above two")
        N = 1 << logN
        if any((prime - 1) % (2 * N) for prime in all_moduli):
            raise ValueError(
                "Every Q/P modulus must support the configured negacyclic NTT"
            )
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
            raise TypeError("sigma must be a real number")
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be positive and finite")
        if type(security_bits) is not int or security_bits <= 0:
            raise ValueError("security_bits must be a positive integer")
        if type(enforce_security_budget) is not bool:
            raise TypeError("enforce_security_budget must be a boolean")
        if galois_generator not in {3, 5}:
            raise ValueError("galois_generator must be 3 or 5")

        self.default_scale = default_scale
        self.q_depth_groups = groups
        self.p_moduli = special
        self.logN = logN
        self.sigma = sigma
        self.security_bits = security_bits
        self.enforce_security_budget = enforce_security_budget
        self.galois_generator = galois_generator
        self._initialized = True

    def __setattr__(self, name: str, value: object) -> None:
        if self.__dict__.get("_initialized", False):
            raise AttributeError("CkksConfig is immutable; construct a new config")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_initialized", False):
            raise AttributeError("CkksConfig is immutable; construct a new config")
        object.__delattr__(self, name)

    @classmethod
    def parse(
        cls,
        src: Mapping[str, Any] | Preset,
        **overrides: Any,
    ) -> "CkksConfig":
        """Resolve a preset or a serialized exact configuration."""

        if isinstance(src, Preset):
            parameters = _preset_parameters(src)
        else:
            parameters = dict(src)
            version = parameters.pop("fhelium_version", None)
            if version is not None and version != __version__:
                raise ValueError(
                    "Serialized CKKS configuration requires FHElium "
                    f"{version}; installed version is {__version__}"
                )
        parameters.update(overrides)
        expected = {
            "default_scale",
            "q_depth_groups",
            "p_moduli",
            "logN",
            "sigma",
            "security_bits",
            "enforce_security_budget",
            "galois_generator",
        }
        unexpected = parameters.keys() - expected
        if unexpected:
            name = min(unexpected)
            raise TypeError(f"unexpected keyword argument {name!r}")
        return cls(
            default_scale=cast(float, parameters["default_scale"]),
            q_depth_groups=cast(
                tuple[tuple[int, ...], ...], parameters["q_depth_groups"]
            ),
            p_moduli=cast(tuple[int, ...], parameters["p_moduli"]),
            logN=cast(int, parameters["logN"]),
            sigma=cast(float, parameters.get("sigma", 3.19)),
            security_bits=cast(int, parameters.get("security_bits", 128)),
            enforce_security_budget=cast(
                bool, parameters.get("enforce_security_budget", True)
            ),
            galois_generator=cast(
                int, parameters.get("galois_generator", 3)
            ),
        )

    def dumps(self) -> dict[str, object]:
        """Return a versioned dictionary containing the exact parameter set."""

        return {
            "fhelium_version": __version__,
            "default_scale": self.default_scale,
            "q_depth_groups": [list(group) for group in self.q_depth_groups],
            "p_moduli": list(self.p_moduli),
            "logN": self.logN,
            "sigma": self.sigma,
            "security_bits": self.security_bits,
            "enforce_security_budget": self.enforce_security_budget,
            "galois_generator": self.galois_generator,
        }

    @cached_property
    def N(self) -> int:
        """Polynomial ring dimension."""

        return 1 << self.logN

    @cached_property
    def num_slots(self) -> int:
        """Number of complex CKKS slots, $N/2$."""

        return self.N // 2

    @cached_property
    def max_depth(self) -> int:
        """Greatest public CKKS depth represented by this Q chain."""

        return len(self.q_depth_groups) - 1

    def depth_remaining(self, depth: int) -> int:
        """Return the public rescale transitions remaining at ``depth``."""

        if type(depth) is not int or not 0 <= depth <= self.max_depth:
            raise ValueError(f"depth must be in [0, {self.max_depth}]")
        return self.max_depth - depth

    @cached_property
    def q_moduli(self) -> tuple[int, ...]:
        """Q primes in depth-group and within-group order."""

        return tuple(prime for group in self.q_depth_groups for prime in group)

    @cached_property
    def moduli(self) -> tuple[int, ...]:
        """Complete QP prime sequence used by RNS resources."""

        return self.q_moduli + self.p_moduli

    @cached_property
    def num_q_primes(self) -> int:
        """Number of prime rows in the complete Q chain."""

        return len(self.q_moduli)

    @cached_property
    def num_p_primes(self) -> int:
        """Number of special-prime rows in P."""

        return len(self.p_moduli)

    @cached_property
    def total_num_primes(self) -> int:
        """Number of prime rows in the complete QP basis."""

        return len(self.moduli)

    def rescale_divisor(self, depth: int) -> int:
        """Return the Q-group product removed at one public depth."""

        if type(depth) is not int or not 0 <= depth < self.max_depth:
            raise ValueError(f"rescale depth must be in [0, {self.max_depth})")
        return math.prod(self.q_depth_groups[depth])

    def q_row_start(self, depth: int) -> int:
        """Return the first Q row active at ``depth``."""

        if type(depth) is not int or not 0 <= depth <= self.max_depth:
            raise ValueError(f"depth must be in [0, {self.max_depth}]")
        return sum(len(group) for group in self.q_depth_groups[:depth])

    def active_q_moduli(self, depth: int) -> tuple[int, ...]:
        """Return the ordered Q primes active at ``depth``."""

        return self.q_moduli[self.q_row_start(depth) :]

    @cached_property
    def inverse_ntt_scale(self) -> tuple[int, ...]:
        r"""Return $N^{-1}\bmod m$ in complete QP prime order."""

        return tuple(pow(self.N, -1, modulus) for modulus in self.moduli)

    @cached_property
    def total_modulus_bits(self) -> int:
        """Bit width of the complete QP product."""

        return (math.prod(self.moduli) - 1).bit_length()

    @cached_property
    def maximum_modulus_bits(self) -> int:
        """Built-in complete-QP budget for the selected security category."""

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
                    "No built-in budget matches this configuration; see the "
                    "security guide for external assessment requirements."
                ),
            )
        return maximum

    @cached_property
    def security_assessment(self) -> SecurityAssessment:
        """Return the built-in assessment of the complete QP product."""

        return assess_config_security(self)

    def validate_security_budget(self) -> SecurityAssessment:
        """Require the complete QP product to meet the configured budget."""

        assessment = self.security_assessment
        if assessment.status == "unsupported":
            raise SecurityParametersUnsupportedError(
                ring_dimension=self.N,
                target_bits=self.security_bits,
                secret_distribution="ternary",
                error_stddev=self.sigma,
                reason=assessment.reason or "No built-in budget matches this configuration.",
            )
        if assessment.status == "exceeds":
            maximum = assessment.maximum_modulus_bits
            if maximum is None:
                raise RuntimeError(
                    "A below-budget assessment must report its table budget"
                )
            raise SecurityBudgetExceededError(
                ring_dimension=self.N,
                max_depth=self.max_depth,
                maximum_modulus_bits=maximum,
                requested_modulus_bits=assessment.modulus_bits,
            )
        return assessment

    def __repr__(self) -> str:
        return (
            "CkksConfig("
            f"default_scale={self.default_scale!r}, "
            f"q_depth_groups={self.q_depth_groups!r}, "
            f"p_moduli={self.p_moduli!r}, "
            f"logN={self.logN}, sigma={self.sigma!r}, "
            f"security_bits={self.security_bits}, "
            f"enforce_security_budget={self.enforce_security_budget}, "
            f"galois_generator={self.galois_generator})"
        )

    def __str__(self) -> str:
        assessment = self.security_assessment
        maximum: int | str = (
            assessment.maximum_modulus_bits
            if assessment.maximum_modulus_bits is not None
            else "unsupported"
        )
        return (
            f"CkksConfig(logN={self.logN}, N={self.N}, "
            f"num_slots={self.num_slots}, max_depth={self.max_depth}, "
            f"Q_primes={self.num_q_primes}, P_primes={self.num_p_primes}, "
            f"default_scale={self.default_scale}, "
            f"total_modulus_bits={self.total_modulus_bits}, "
            f"maximum_modulus_bits={maximum}, sigma={self.sigma}, "
            f"security_bits={self.security_bits}, "
            f"enforce_security_budget={self.enforce_security_budget}, "
            f"galois_generator={self.galois_generator})"
        )
