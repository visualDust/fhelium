r"""Modulus-raise methods of a bootstrap plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fhelium.values import Ciphertext


@dataclass(frozen=True)
class Centered:
    r"""Extend each component's centered representative from $q_b$ to the target Q basis.

    For a source ciphertext $(c_0,c_1)$ modulo $q_b$ with $|c_i|\le q_b/2$, the
    raised ciphertext decrypts to $m+q_bI$ with $I=\lfloor (c_0+c_1s)/q_b\rceil$,
    so $|I|\le(\lVert s\rVert_1+1)/2$. Depth changes from the source depth to
    the target depth; the actual scale is unchanged.
    """

    def _select_keys(self, engine, plan, keys, device):
        return None

    def _raise(self, engine, value: Ciphertext, *, plan, target_depth: int, keys):
        del plan, keys
        return engine.modulus_raise(value, target_depth)


@dataclass(frozen=True)
class Encapsulated:
    r"""Raise through an intermediate secret $t$ of Hamming weight $h$.

    The source ciphertext is key-switched from $s$ to $t$ over a small
    auxiliary basis, centered-raised, and switched back to $s$ with a
    guard-scaled return key. The overflow polynomial then satisfies
    $|I|\le(h+1)/2$ independently of the weight of $s$. ``intermediate_weight``
    is required; :func:`fhelium.bootstrap.create_evaluation_keys` samples $t$
    and adds its two :class:`~fhelium.values.EncapsulationKey` values to the
    inventory. The auxiliary primes are the largest P primes: one low prime,
    one guard prime and further return primes until their product reaches
    $q_b$. The
    security of the joint publication of these keys depends on $h$, the
    auxiliary moduli and the main parameters, and is analyzed by the caller.
    """

    intermediate_weight: int
    moddown_rounding: Literal["nearest", "floor"] = "nearest"

    def __post_init__(self) -> None:
        if type(self.intermediate_weight) is not int or self.intermediate_weight <= 0:
            raise ValueError("intermediate_weight must be a positive integer")

    def _select_keys(self, engine, plan, keys, device):
        from fhelium.bootstrap._encapsulation import _select_keys

        return _select_keys(engine, plan, keys, device)

    def _raise(self, engine, value: Ciphertext, *, plan, target_depth: int, keys):
        from fhelium.bootstrap._encapsulation import _encapsulated_raise

        return _encapsulated_raise(
            engine, self, value, plan=plan, target_depth=target_depth, keys=keys
        )


__all__ = ["Centered", "Encapsulated"]
