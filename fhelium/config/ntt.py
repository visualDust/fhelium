"""Configuration and compatibility checks for CPU and CUDA NTT backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias


@dataclass(frozen=True)
class IndexedRadix2Policy:
    """Table-driven indexed radix-2 execution on CPU and CUDA.

    CPU parallelizes the flattened batch/limb/butterfly space separately for
    each stage, with a stage barrier preserving transform dependencies. CUDA
    launches one kernel per stage. The same indexed schedule also provides the
    cross-device baseline for validating compact CUDA policies.
    """

    name: str
    """Name used by the backend registry and engine API."""


@dataclass(frozen=True)
class CompactRadix2Policy:
    r"""Radix-2 execution over compact twiddle rows.

    For $N=2^L$ the transform makes two passes over each prime row. One pass
    applies $\lfloor L/2 \rfloor$ stages to coefficients spaced
    $2^{L-\lfloor L/2 \rfloor}$ apart, and the other applies the remaining
    stages to contiguous tiles; forward transforms run the spaced pass first
    and inverse transforms run it last. Shoup32 tables pair each standard root
    with its 32-bit quotient word and require CUDA int32 residues with primes
    below 2^30.
    """

    name: str
    """Name used by the backend registry and engine API."""

    twiddle_encoding: Literal["montgomery", "shoup32"] = "montgomery"
    """Numerical layout of each compact root table."""

    def __post_init__(self) -> None:
        if self.twiddle_encoding not in ("montgomery", "shoup32"):
            raise ValueError("twiddle_encoding must be montgomery or shoup32")


@dataclass(frozen=True)
class CompactFixedRadixPolicy:
    r"""A fixed-radix NTT schedule using the same radix for every digit.

    Let $N=2^L$ and $\mathtt{radix}=2^b$. A compatible transform has $D=L/b$
    fixed-radix digits and is rejected when $L$ is not divisible by $b$.
    Shared-memory tile size and fusion depth are native CUDA implementation
    choices: they do not affect the mathematical policy, table layout, or
    backend identity represented here.
    """

    name: str
    """Name used by the backend registry and engine API."""

    radix: Literal[4, 8, 16]
    """Butterfly radix used by every transform digit."""

    def __post_init__(self) -> None:
        if self.radix not in (4, 8, 16):
            raise ValueError("Fixed radix must be 4, 8, or 16")

    @property
    def radix_bits(self) -> Literal[2, 3, 4]:
        r"""Return $\log_2(\mathtt{radix})$, the bits covered by one digit."""

        if self.radix == 4:
            return 2
        if self.radix == 8:
            return 3
        return 4


NttBackendPolicy: TypeAlias = (
    IndexedRadix2Policy | CompactRadix2Policy | CompactFixedRadixPolicy
)

_POLICY_SEQUENCE: Final[tuple[NttBackendPolicy, ...]] = (
    IndexedRadix2Policy(name="radix2_indexed"),
    CompactRadix2Policy(name="radix2_compact"),
    CompactRadix2Policy(
        name="radix2_compact_shoup32",
        twiddle_encoding="shoup32",
    ),
    CompactFixedRadixPolicy(
        name="radix4_compact",
        radix=4,
    ),
    CompactFixedRadixPolicy(
        name="radix8_compact",
        radix=8,
    ),
    CompactFixedRadixPolicy(
        name="radix16_compact",
        radix=16,
    ),
)

NTT_BACKEND_POLICIES: Final = MappingProxyType(
    {policy.name: policy for policy in _POLICY_SEQUENCE}
)
SUPPORTED_NTT_BACKENDS: Final = tuple(NTT_BACKEND_POLICIES)
# This fallback applies across supported ring dimensions and CUDA devices.
# Execution owners can select another named policy when constructing NTT resources.
DEFAULT_NTT_BACKEND: Final[str] = "radix2_compact"
# The indexed radix-2 policy is the CPU production backend as well as the
# cross-device validation baseline. Compact policies remain GPU-only.
DEFAULT_CPU_NTT_BACKEND: Final[str] = "radix2_indexed"


def resolve_ntt_backend_policy(name: str) -> NttBackendPolicy:
    """Return the policy named by configuration.

    Names must match a registered backend name, including case.
    """

    try:
        return NTT_BACKEND_POLICIES[name]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_NTT_BACKENDS)
        raise ValueError(
            f"Unsupported NTT backend {name!r}; supported backends: {supported}"
        ) from error


def validate_ntt_backend_for_log_n(
    policy: NttBackendPolicy,
    log_ring_dimension: int,
) -> None:
    """Reject a named policy that cannot factor the requested ring size."""

    if log_ring_dimension <= 0:
        raise ValueError("log_ring_dimension must be positive")
    if isinstance(policy, CompactFixedRadixPolicy):
        if log_ring_dimension % policy.radix_bits != 0:
            raise ValueError(
                f"NTT backend {policy.name!r} requires logN divisible by "
                f"{policy.radix_bits}; got logN={log_ring_dimension}"
            )


def compatible_ntt_backends(log_ring_dimension: int) -> tuple[str, ...]:
    """Return registered policy names executable for one ``logN``.

    Names retain registry order. Strict fixed-radix policies whose digit width
    does not divide ``log_ring_dimension`` are omitted; compact radix-2
    policies remain available for every supported ring dimension. A
    non-positive dimension raises :class:`ValueError`.
    """

    if log_ring_dimension <= 0:
        raise ValueError("log_ring_dimension must be positive")
    compatible: list[str] = []
    for policy in _POLICY_SEQUENCE:
        try:
            validate_ntt_backend_for_log_n(policy, log_ring_dimension)
        except ValueError:
            continue
        compatible.append(policy.name)
    return tuple(compatible)
