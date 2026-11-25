"""Define CKKS parameter presets and the validated configuration model."""

import math
from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Mapping
from enum import Enum
from functools import cached_property
from typing import Any, cast

from fhelium._version import __version__
from fhelium.config.security import (
    SecurityAssessment,
    _lookup_maximum_modulus_bits,
    assess_config_security,
)
from fhelium.errors import (
    SecurityBudgetExceededError,
    SecurityParametersUnsupportedError,
)
from fhelium.native.wrapper import ckks_ops


class Preset(Enum):
    """Built-in CKKS parameter presets.

    Each member name records the complex slot capacity, the maximum-depth
    default scale bits, and the maximum public depth. Every preset uses NTT
    primes below $2^{30}$, so the Engine selects int32 residues for it; the
    Q order and depth bases follow the rescale-ratio construction described
    by :class:`CkksConfig`. A ``-bootN`` suffix marks a chain whose first 16
    depths form the segment consumed by the ``"3/38"`` bootstrap profile of
    :func:`fhelium.bootstrap.plan`, leaving ``N`` application depths after a
    refresh. Its Q primes outside the application chain have about 25, 28 and
    30 bits, and the P directory holds every remaining prime below $2^{30}$
    that fits the security budget; key layouts take their auxiliary primes
    from P and from Q primes inactive at the source depth. These presets use
    Galois generator 5, which full-slot bootstrapping requires. All presets select
    the 128-bit classical security
    category, Gaussian error standard deviation 3.19, and uniform-ternary
    secret sampling. :meth:`CkksConfig.parse` accepts keyword overrides when an
    application needs a derived configuration.
    """

    slots8192_scale30_depth9 = "slots8192-scale30-depth9"
    slots8192_scale40_depth7 = "slots8192-scale40-depth7"
    slots8192_scale50_depth5 = "slots8192-scale50-depth5"
    slots16384_scale30_depth22 = "slots16384-scale30-depth22"
    slots16384_scale40_depth16 = "slots16384-scale40-depth16"
    slots16384_scale50_depth11 = "slots16384-scale50-depth11"
    slots32768_scale30_depth46 = "slots32768-scale30-depth46"
    slots32768_scale40_depth34 = "slots32768-scale40-depth34"
    slots32768_scale50_depth27 = "slots32768-scale50-depth27"
    slots65536_scale30_depth100 = "slots65536-scale30-depth100"
    slots65536_scale40_depth75 = "slots65536-scale40-depth75"
    slots65536_scale50_depth55 = "slots65536-scale50-depth55"
    slots32768_scale40_depth20_boot4 = "slots32768-scale40-depth20-boot4"
    slots32768_scale40_depth24_boot8 = "slots32768-scale40-depth24-boot8"
    slots32768_scale40_depth29_boot13 = "slots32768-scale40-depth29-boot13"
    slots32768_scale50_depth24_boot8 = "slots32768-scale50-depth24-boot8"
    slots32768_scale50_depth28_boot12 = "slots32768-scale50-depth28-boot12"


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


def _transition(
    bases: tuple[tuple[int, ...], ...], source: int, target: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_ids = set(bases[source])
    target_ids = set(bases[target])
    return (
        tuple(i for i in bases[source] if i not in target_ids),
        tuple(i for i in bases[target] if i not in source_ids),
    )


def balanced_scales(
    q_moduli: tuple[int, ...],
    q_bases: tuple[tuple[int, ...], ...],
    final_scale: float,
) -> tuple[float, ...]:
    r"""Return the default scale of every depth for one Q chain.

    With $\rho_d=D_d/R_d$ the rescale ratio from depth $d$ to $d+1$, the
    maximum depth uses ``final_scale`` and every earlier depth uses
    $\Delta_d=\sqrt{\Delta_{d+1}\rho_d}$, so that the rescaled product of
    two depth-$d$ default-scale values has scale $\Delta_{d+1}$.
    """

    scales = [float(final_scale)]
    for depth in range(len(q_bases) - 2, -1, -1):
        dropped, restored = _transition(q_bases, depth, depth + 1)
        ratio = Fraction(
            math.prod(q_moduli[i] for i in dropped),
            math.prod(q_moduli[i] for i in restored),
        )
        scales.append(math.sqrt(scales[-1] * float(ratio)))
    return tuple(reversed(scales))


def _preset_parameters(preset: Preset) -> dict[str, object]:
    """Resolve a preset to its Q and P prime values with libfhelium."""

    log_n, q_moduli, basis_rows, basis_sizes, scales, p_moduli = (
        ckks_ops.preset_parameters(preset.value)
    )
    bases = []
    start = 0
    for size in basis_sizes:
        bases.append(tuple(basis_rows[start : start + size]))
        start += size
    return {
        "q_moduli": tuple(q_moduli),
        "q_bases": tuple(bases),
        "scales": tuple(scales),
        "p_moduli": tuple(p_moduli),
        "logN": log_n,
        **({"galois_generator": 5} if "-boot" in preset.value else {}),
    }


@dataclass(frozen=True)
class KeyLayout:
    r"""Rows of a hybrid key-switching key whose source basis is $B_d$.

    ``q_prime_ids`` lists $B_d$, ``p_prime_ids`` the auxiliary primes of
    $P_d$, and ``digit_prime_ids`` consecutive runs of $B_d$ whose products
    are below $P_d$.
    """

    depth: int
    q_prime_ids: tuple[int, ...]
    p_prime_ids: tuple[int, ...]
    digit_prime_ids: tuple[tuple[int, ...], ...]


def _greedy_digits(ids, moduli, limit):
    digits, digit, product = [], [], 1
    for i in ids:
        if moduli[i] >= limit:
            return None
        if digit and product * moduli[i] >= limit:
            digits.append(tuple(digit))
            digit, product = [], 1
        digit.append(i)
        product *= moduli[i]
    return (*digits, tuple(digit))


def _minimax_digits(ids, moduli, count):
    """Split ``ids`` into ``count`` runs minimizing the largest run product."""
    prefix = [1]
    for i in ids:
        prefix.append(prefix[-1] * moduli[i])
    n = len(ids)
    best = {(0, 0): (1, None)}
    for k in range(1, count + 1):
        for stop in range(k, n - count + k + 1):
            best[k, stop] = min(
                (
                    max(best[k - 1, start][0], prefix[stop] // prefix[start]),
                    start,
                )
                for start in range(k - 1, stop)
                if (k - 1, start) in best
            )
    digits, stop = [], n
    for k in range(count, 0, -1):
        start = best[k, stop][1]
        digits.append(tuple(ids[start:stop]))
        stop = start
    return tuple(reversed(digits))


class CkksConfig:
    r"""Immutable CKKS mathematical and security parameters.

    The configuration defines CKKS over $R=\mathbb{Z}[X]/(X^N+1)$, where $N=2^{\mathtt{logN}}$, together with an ordered Q prime list, the active basis and default scale of every public depth, and the hybrid key-switch modulus P.

    ``q_bases[d]`` lists, in increasing order, the ids of the Q primes
    $B_d$ active at depth ``d``; ``max_depth == len(q_bases) - 1``. Tensor
    row ``j`` of a depth-``d`` value is modulo ``q_moduli[q_bases[d][j]]``.
    $B_0$ holds every Q prime. A
    rescale from depth $d$ to $d+1$ drops $D_d=B_d\setminus B_{d+1}$,
    restores $R_d=B_{d+1}\setminus B_d$, and divides every coefficient and
    the actual scale by $\rho_d=D_d/R_d$. A chain whose bases are Q suffixes
    has $R_d=\varnothing$ and divides by the dropped group product; see
    :meth:`suffix_chain`. A prime may leave a basis and return at a later
    depth. Depth identifies the active basis; it does not count
    multiplications or record how a value reached that basis.

    ``scales[d]`` is the scale used at depth ``d`` when value creation omits a
    scale; every live plaintext and ciphertext carries its own actual scale.
    ``p_moduli`` is the catalog of primes reserved for key-switch auxiliary
    moduli. A key generated without an explicit layout uses all of them as P;
    a key whose source basis is $B_d$ may instead use any configured prime
    outside $B_d$. The security budget applies to the product of every
    configured Q and P prime, which bounds the modulus of every key. The Engine
    selects the residue dtype and Montgomery radix from the prime values, and
    each device-local RNS context materializes the corresponding arithmetic
    tables.
    """

    def __init__(
        self,
        *,
        q_moduli: tuple[int, ...],
        q_bases: tuple[tuple[int, ...], ...],
        scales: tuple[float, ...],
        p_moduli: tuple[int, ...],
        logN: int,
        sigma: float = 3.19,
        security_bits: int = 128,
        enforce_security_budget: bool = True,
        galois_generator: int = 3,
    ) -> None:
        if type(logN) is not int or logN < 1:
            raise ValueError("logN must be a positive integer")
        q = _dumped_moduli(q_moduli, name="q_moduli", require_nonempty=True)
        if not isinstance(q_bases, (list, tuple)) or not q_bases:
            raise TypeError("q_bases must be a nonempty sequence")
        bases = []
        for basis in q_bases:
            if (
                not isinstance(basis, (list, tuple))
                or not basis
                or any(type(i) is not int for i in basis)
                or not 0 <= basis[0]
                or basis[-1] >= len(q)
                or any(a >= b for a, b in zip(basis, basis[1:]))
            ):
                raise ValueError(
                    "q_bases entries must be nonempty increasing Q prime ids"
                )
            bases.append(tuple(basis))
        bases_tuple = tuple(bases)
        if bases_tuple[0] != tuple(range(len(q))):
            raise ValueError("the depth-0 basis must hold every Q prime")
        for depth in range(len(bases_tuple) - 1):
            if not _transition(bases_tuple, depth, depth + 1)[0]:
                raise ValueError(
                    "every rescale must drop at least one Q prime"
                )
        if not isinstance(scales, (list, tuple)) or len(scales) != len(
            bases_tuple
        ):
            raise ValueError("scales must contain one scale per depth")
        scale_values = []
        for scale in scales:
            if not isinstance(scale, (int, float)) or isinstance(scale, bool):
                raise TypeError("scales must contain real numbers")
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError("scales must be positive and finite")
            scale_values.append(scale)
        special = _dumped_moduli(
            p_moduli, name="p_moduli", require_nonempty=True
        )
        ckks_ops.validate_moduli(logN, list(q + special))
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

        self.q_moduli = q
        self.q_bases = bases_tuple
        self.scales = tuple(scale_values)
        self.p_moduli = special
        self.logN = logN
        self.sigma = sigma
        self.security_bits = security_bits
        self.enforce_security_budget = enforce_security_budget
        self.galois_generator = galois_generator
        self._initialized = True

    @classmethod
    def suffix_chain(
        cls,
        *,
        q_depth_groups: tuple[tuple[int, ...], ...],
        scale: float,
        p_moduli: tuple[int, ...],
        logN: int,
        **options: Any,
    ) -> "CkksConfig":
        r"""Build a chain whose depth-$d$ basis is the concatenation of groups ``d:``.

        Each rescale drops one group and restores no prime, so it divides by
        the group product. Every depth uses ``scale`` as its default scale.
        """

        if not isinstance(q_depth_groups, (list, tuple)) or any(
            not isinstance(group, (list, tuple)) or not group
            for group in q_depth_groups
        ) or not q_depth_groups:
            raise TypeError(
                "q_depth_groups must be a nonempty sequence of prime groups"
            )
        q_moduli = tuple(prime for group in q_depth_groups for prime in group)
        starts = [0]
        for group in q_depth_groups[:-1]:
            starts.append(starts[-1] + len(group))
        return cls(
            q_moduli=q_moduli,
            q_bases=tuple(tuple(range(start, len(q_moduli))) for start in starts),
            scales=(float(scale),) * len(q_depth_groups),
            p_moduli=tuple(p_moduli),
            logN=logN,
            **options,
        )

    @classmethod
    def catalog_suffix_chain(
        cls,
        *,
        logN: int,
        scale_bits: int,
        depth: int,
        p_count: int,
        p_bits: int = 60,
        catalog_logN: int | None = None,
        **options: Any,
    ) -> "CkksConfig":
        r"""Build a single-prime suffix chain from the packaged prime catalog.

        Q holds the first ``depth + 1`` catalog primes of ``scale_bits`` bits
        for ring dimension $2^{\mathtt{catalog\_logN}}$ (default ``logN``), one
        per depth group, and P holds the first ``p_count`` special primes of
        ``p_bits`` bits outside Q. Catalog primes of a larger ring dimension
        satisfy $q\equiv1\pmod{2N}$ for every smaller $N$. Every depth uses
        the default scale $2^{\mathtt{scale\_bits}}$. Primes of 30 or more bits
        select int64 residues.
        """

        ring_dimension = 1 << (logN if catalog_logN is None else catalog_logN)
        q = ckks_ops.scaling_primes(scale_bits, ring_dimension)[: depth + 1]
        if len(q) != depth + 1:
            raise ValueError(
                f"the prime catalog holds fewer than {depth + 1} "
                f"{scale_bits}-bit primes for N={ring_dimension}"
            )
        p = [
            prime
            for prime in ckks_ops.special_primes(p_bits, ring_dimension)
            if prime not in q
        ][:p_count]
        return cls.suffix_chain(
            q_depth_groups=tuple((prime,) for prime in q),
            scale=float(2**scale_bits),
            p_moduli=tuple(p),
            logN=logN,
            **options,
        )

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
        """Resolve a preset or deserialize a CKKS parameter set."""

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
            "q_moduli",
            "q_bases",
            "scales",
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
            q_moduli=cast(tuple[int, ...], tuple(parameters["q_moduli"])),
            q_bases=cast(
                tuple[tuple[int, ...], ...],
                tuple(tuple(basis) for basis in parameters["q_bases"]),
            ),
            scales=cast(tuple[float, ...], tuple(parameters["scales"])),
            p_moduli=cast(tuple[int, ...], tuple(parameters["p_moduli"])),
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
        """Serialize the parameter values and package version into a dictionary."""

        return {
            "fhelium_version": __version__,
            "q_moduli": list(self.q_moduli),
            "q_bases": [list(basis) for basis in self.q_bases],
            "scales": list(self.scales),
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

        return len(self.q_bases) - 1

    def _check_depth(self, depth: int) -> None:
        if type(depth) is not int or not 0 <= depth <= self.max_depth:
            raise ValueError(f"depth must be in [0, {self.max_depth}]")

    def key_layout(self, depth: int) -> KeyLayout:
        r"""Return the default key layout for consumers with basis $B_d$.

        The auxiliary primes are the largest configured primes outside
        $B_d$, which may include Q primes inactive at depth ``d``. With
        $q=|B_d|$ auxiliary prime count $p$ and digit count $k$, one key
        switch lifts and multiplies $k(q+p)$ rows and divides $q+p$ rows by
        $P_d$; the layout minimizes the row work $(k+1)(q+p)$, then the key
        rows $k(q+p)$, then $p$. The digits are consecutive runs of $B_d$ that
        minimize the largest digit product below $P_d$. The product of every
        configured prime bounds $Q_dP_d$ under the security budget.
        """

        self._check_depth(depth)
        cache = self.__dict__.setdefault("_key_layouts", {})
        if depth in cache:
            return cache[depth]
        moduli = self.moduli
        q_ids = self.q_bases[depth]
        active = set(q_ids)
        eligible = sorted(
            (i for i in range(self.total_num_primes) if i not in active),
            key=lambda i: (-moduli[i], i),
        )
        best = None
        limit = 1
        q = len(q_ids)
        for width, prime_id in enumerate(eligible, start=1):
            limit *= moduli[prime_id]
            digits = _greedy_digits(q_ids, moduli, limit)
            if digits is None:
                continue
            k = len(digits)
            cost = ((k + 1) * (q + width), k * (q + width), width)
            if best is None or cost < best[0]:
                best = (cost, k, tuple(sorted(eligible[:width])))
        if best is None:
            raise ValueError(
                f"the configured primes outside B_{depth} cannot bound a key digit"
            )
        _, count, p_ids = best
        layout = KeyLayout(
            depth, q_ids, p_ids, _minimax_digits(q_ids, moduli, count)
        )
        cache[depth] = layout
        return layout

    def depth_remaining(self, depth: int) -> int:
        """Return the public rescale transitions remaining at ``depth``."""

        self._check_depth(depth)
        return self.max_depth - depth

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

    @cached_property
    def is_suffix_chain(self) -> bool:
        """Whether every depth basis is a Q suffix, so no rescale restores primes."""

        return all(
            basis == tuple(range(basis[0], self.num_q_primes))
            for basis in self.q_bases
        )

    @property
    def suffix_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return the Q prime groups dropped by successive rescales of a suffix chain."""

        if not self.is_suffix_chain:
            raise ValueError("Depth groups exist only for suffix chains")
        starts = [basis[0] for basis in self.q_bases] + [self.num_q_primes]
        return tuple(
            self.q_moduli[start:stop] for start, stop in zip(starts, starts[1:])
        )

    def scale_at(self, depth: int) -> float:
        """Return the default scale of values created at ``depth``."""

        self._check_depth(depth)
        return self.scales[depth]

    def q_prime_ids(self, depth: int) -> tuple[int, ...]:
        """Return the ordered Q prime ids active at ``depth``."""

        self._check_depth(depth)
        return self.q_bases[depth]

    def active_q_moduli(self, depth: int) -> tuple[int, ...]:
        """Return the ordered Q primes active at ``depth``."""

        self._check_depth(depth)
        return tuple(self.q_moduli[i] for i in self.q_bases[depth])

    def is_subbasis(self, source: int, target: int) -> bool:
        """Whether the basis of depth ``target`` is contained in that of ``source``."""

        self._check_depth(source)
        self._check_depth(target)
        return set(self.q_bases[target]) <= set(self.q_bases[source])

    def basis_transition(
        self, source: int, target: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the dropped and restored Q prime ids from ``source`` to ``target``."""

        self._check_depth(source)
        self._check_depth(target)
        return _transition(self.q_bases, source, target)

    def transition_ratio(self, source: int, target: int) -> Fraction:
        r"""Return $D/R$ for the transition from ``source`` to ``target``."""

        dropped, restored = self.basis_transition(source, target)
        return Fraction(
            math.prod(self.q_moduli[i] for i in dropped),
            math.prod(self.q_moduli[i] for i in restored),
        )

    def rescale_divisor(self, depth: int) -> Fraction:
        r"""Return the rescale ratio $\rho_d=D_d/R_d$ from ``depth`` to ``depth + 1``."""

        if type(depth) is not int or not 0 <= depth < self.max_depth:
            raise ValueError(f"rescale depth must be in [0, {self.max_depth})")
        return self.transition_ratio(depth, depth + 1)

    @cached_property
    def inverse_ntt_scale(self) -> tuple[int, ...]:
        r"""Return $N^{-1}\bmod m$ in complete QP prime order."""

        return tuple(pow(self.N, -1, modulus) for modulus in self.moduli)

    @cached_property
    def total_modulus_bits(self) -> int:
        """Bit width of the complete QP product."""

        return ckks_ops.modulus_bits(list(self.moduli))

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
            f"q_moduli={self.q_moduli!r}, "
            f"q_bases={self.q_bases!r}, "
            f"scales={self.scales!r}, "
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
            f"scale_bits=({math.log2(self.scales[0]):.2f}, "
            f"{math.log2(self.scales[-1]):.2f}), "
            f"total_modulus_bits={self.total_modulus_bits}, "
            f"maximum_modulus_bits={maximum}, sigma={self.sigma}, "
            f"security_bits={self.security_bits}, "
            f"enforce_security_budget={self.enforce_security_budget}, "
            f"galois_generator={self.galois_generator})"
        )
