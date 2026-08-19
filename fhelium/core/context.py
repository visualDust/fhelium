"""Immutable CKKS context metadata."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from typing import ClassVar

from fhelium.core.scale import coerce_scale


@dataclass(frozen=True)
class CkksContextSpec:
    r"""Device-placement-independent CKKS context description.

    The context fixes $N=2^{\mathtt{logN}}$, $S=N/2$, the ordered ordinary
    primes used to form $Q_\ell$, the special-prime product $P$, the default
    encoding/planning scale $\Delta_0$, and the Galois generator. It is
    metadata-only and independent of process rank, device ownership, or
    communication decisions. ``default_scale`` is not an invariant imposed
    on values: every plaintext and ciphertext carries its own actual scale
    $\Delta(v)$.
    """

    representation: ClassVar[str] = "direct_per_value_scale_v1"

    logN: int
    default_scale: float
    q_moduli: tuple[int, ...]
    p_moduli: tuple[int, ...] = ()
    galois_generator: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_scale",
            coerce_scale(
                self.default_scale,
                value_name="CkksContextSpec default",
            ),
        )
        if self.logN <= 0:
            raise ValueError(f"logN must be positive, got {self.logN}.")
        if not self.q_moduli:
            raise ValueError("CkksContextSpec requires at least one Q modulus.")
        if self.galois_generator not in {3, 5}:
            raise ValueError("galois_generator must be 3 or 5")
        object.__setattr__(
            self, "q_moduli", tuple(int(modulus) for modulus in self.q_moduli)
        )
        object.__setattr__(
            self, "p_moduli", tuple(int(modulus) for modulus in self.p_moduli)
        )

    @property
    def N(self) -> int:
        return 1 << self.logN

    @property
    def num_slots(self) -> int:
        return self.N // 2

    @property
    def num_q_primes(self) -> int:
        return len(self.q_moduli)

    @property
    def num_p_primes(self) -> int:
        return len(self.p_moduli)

    @cached_property
    def context_id(self) -> str:
        """Stable SHA-256 identity of all mathematical context parameters."""

        q_moduli = ",".join(str(value) for value in self.q_moduli)
        p_moduli = ",".join(str(value) for value in self.p_moduli)
        payload = (
            f"representation={self.representation};logN={self.logN};"
            f"default_scale={self.default_scale!r};Q={q_moduli};"
            f"P={p_moduli};galois_generator={self.galois_generator}"
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        return (
            "CkksContextSpec("
            f"representation={self.representation!r}, logN={self.logN}, "
            f"N={self.N}, slots={self.num_slots}, "
            f"default_scale={self.default_scale!r}, "
            f"num_q={self.num_q_primes}, num_p={self.num_p_primes}, "
            f"galois_generator={self.galois_generator}, "
            f"context_id={self.context_id[:12]}...)"
        )

    __repr__ = __str__
