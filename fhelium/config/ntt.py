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
    """Canonical name used by the backend registry and engine API."""


@dataclass(frozen=True)
class CompactRadix2Policy:
    """Grouped radix-2 execution over canonical compact twiddle rows.

    ``grouped_radix2_stage_count`` counts radix-2 stages fused into a global
    kernel. The ``smem8`` backend-name suffix records the current native
    production implementation, whose shared-memory tile and fusion depth are
    compiled CUDA resources rather than Python policy parameters.
    """

    name: str
    """Canonical name used by the backend registry and engine API."""

    grouped_radix2_stage_count: Literal[2, 3, 4]
    """Number of radix-2 stages fused by each global-memory kernel."""

    def __post_init__(self) -> None:
        if self.grouped_radix2_stage_count not in (2, 3, 4):
            raise ValueError("grouped_radix2_stage_count must be 2, 3, or 4")

    @property
    def group_width(self) -> int:
        """Number of coefficients held by one grouped radix-2 tuple."""

        return 1 << self.grouped_radix2_stage_count


@dataclass(frozen=True)
class CompactFixedRadixPolicy:
    r"""Strict execution in which every digit has exactly one radix.

    Let $N=2^L$ and $\mathtt{radix}=2^b$. A compatible transform has $D=L/b$
    fixed-radix digits and is rejected when $L$ is not divisible by $b$.
    Shared-memory tile size and fusion depth are native CUDA implementation
    choices: they do not affect the mathematical policy, table layout, or
    backend identity represented here.
    """

    name: str
    """Canonical name used by the backend registry and engine API."""

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
    CompactRadix2Policy(
        name="radix2_compact_group4_smem8",
        grouped_radix2_stage_count=2,
    ),
    CompactRadix2Policy(
        name="radix2_compact_group8_smem8",
        grouped_radix2_stage_count=3,
    ),
    CompactRadix2Policy(
        name="radix2_compact_group16_smem8",
        grouped_radix2_stage_count=4,
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
# One versioned, process-independent fallback is used for every supported
# logN and CUDA device. Selection never consults hardware, runs a benchmark,
# or dispatches through a per-logN table; applications opt into any other
# named policy on CkksEngine.
DEFAULT_NTT_BACKEND: Final[str] = "radix2_compact_group8_smem8"
# The indexed radix-2 policy is the CPU production backend as well as the
# cross-device validation baseline. Compact/grouped policies remain CUDA.
DEFAULT_CPU_NTT_BACKEND: Final[str] = "radix2_indexed"


def resolve_ntt_backend_policy(name: str) -> NttBackendPolicy:
    """Return the policy named by configuration.

    Names are deliberately not case-normalized and no compatibility aliases
    are accepted.
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
    """Return canonical policy names executable for one ``logN``.

    Names retain registry order. Strict fixed-radix policies whose digit width
    does not divide ``log_ring_dimension`` are omitted; grouped radix-2
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
