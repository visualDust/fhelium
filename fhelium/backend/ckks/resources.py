"""CKKS key kinds and resources used by Tensor implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

from fhelium.backend.ntt.context import NttContext
from fhelium.backend.rns.context import RnsContext
from fhelium.values import (
    ConjugationKey,
    KeySwitchKey,
    PublicKey,
    RelinearizationKey,
    RotationKey,
    SecretKey,
)
from .crypto._resources import (
    PUBLIC_KEY_RESOURCE_KIND,
    SECRET_KEY_RESOURCE_KIND,
)

RESCALE_RESOURCE_KIND = "rescale-plan"
KEY_SWITCH_PLAN_RESOURCE_KIND = "key-switch-plan"
KEY_SWITCH_KEY_RESOURCE_KIND = "key-switch-key"
RELINEARIZATION_KEY_RESOURCE_KIND = "relinearization-key"
ROTATION_KEY_RESOURCE_KIND = "rotation-key"
CONJUGATION_KEY_RESOURCE_KIND = "conjugation-key"

_CKKS_KEY_RESOURCE_KIND_BY_TYPE: dict[type[object], str] = {
    PublicKey: PUBLIC_KEY_RESOURCE_KIND,
    SecretKey: SECRET_KEY_RESOURCE_KIND,
    KeySwitchKey: KEY_SWITCH_KEY_RESOURCE_KIND,
    RelinearizationKey: RELINEARIZATION_KEY_RESOURCE_KIND,
    RotationKey: ROTATION_KEY_RESOURCE_KIND,
    ConjugationKey: CONJUGATION_KEY_RESOURCE_KIND,
}
CKKS_KEY_RESOURCE_KINDS = frozenset(_CKKS_KEY_RESOURCE_KIND_BY_TYPE.values())


def ckks_key_resource_kind(key: object) -> str:
    """Return the Backend resource kind for one concrete CKKS key value."""

    try:
        return _CKKS_KEY_RESOURCE_KIND_BY_TYPE[type(key)]
    except KeyError:
        raise TypeError(
            "CKKS key resources require PublicKey, SecretKey, KeySwitchKey, "
            "RelinearizationKey, RotationKey, or ConjugationKey; got "
            f"{type(key).__name__}"
        ) from None


@dataclass(frozen=True)
class RescaleExecutionResource:
    """Own pairwise prime-drop inverses used by CKKS rescaling."""

    rns_context: RnsContext
    dropped_prime_inverses_montgomery: torch.Tensor
    _active_views: Mapping[
        tuple[int, bool], tuple[torch.Tensor, torch.Tensor, int]
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        table = self.dropped_prime_inverses_montgomery
        prime_count = len(self.rns_context.montgomery_parameters.moduli)
        if tuple(table.shape) != (prime_count, prime_count):
            raise ValueError(
                "Rescale inverse matrix shape differs from the modulus set"
            )
        views: dict[
            tuple[int, bool], tuple[torch.Tensor, torch.Tensor, int]
        ] = {}
        for level in range(self.rns_context.q_row_stop - 1):
            for include_p in (False, True):
                stop = (
                    self.rns_context.qp_row_stop
                    if include_p
                    else self.rns_context.q_row_stop
                )
                row_count = stop - level
                remaining_ids = tuple(range(level + 1, stop))
                views[(row_count, include_p)] = (
                    self.rns_context.rns_parameters_for_prime_ids(
                        remaining_ids
                    ),
                    table[level, level + 1 : stop],
                    int(self.rns_context.montgomery_parameters.moduli[level]),
                )
        object.__setattr__(self, "_active_views", MappingProxyType(views))

    @property
    def device(self) -> torch.device:
        return self.rns_context.device

    def active_views(
        self,
        row_count: int,
        *,
        include_p: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return preselected parameters, inverses, and dropped modulus."""

        try:
            return self._active_views[(row_count, include_p)]
        except KeyError:
            raise ValueError(
                "Rescale requires a complete Q or QP basis with another "
                "active Q row"
            ) from None


@dataclass(frozen=True)
class KeySwitchExecutionResource:
    """Own the tables used by hybrid CKKS key switching."""

    rns_context: RnsContext
    ntt_context: NttContext
    moddown_p_drop_inverses_montgomery_by_level: tuple[torch.Tensor, ...]
    galois_generator: int = 3

    def __post_init__(self) -> None:
        if self.ntt_context.rns_context is not self.rns_context:
            raise ValueError(
                "Key-switch NTT context composes another RNS context"
            )
        if not self.moddown_p_drop_inverses_montgomery_by_level:
            raise ValueError("Key-switch resource requires ModDown tables")
        if self.galois_generator not in {3, 5}:
            raise ValueError(
                "Key-switch resource galois generator must be 3 or 5"
            )

    @property
    def device(self) -> torch.device:
        return self.rns_context.device

    @property
    def moddown_tables(self) -> tuple[torch.Tensor, ...]:
        """Return the level-indexed P-drop inverse tables read by ModDown."""

        return self.moddown_p_drop_inverses_montgomery_by_level


__all__ = [
    "CKKS_KEY_RESOURCE_KINDS",
    "CONJUGATION_KEY_RESOURCE_KIND",
    "KEY_SWITCH_KEY_RESOURCE_KIND",
    "KEY_SWITCH_PLAN_RESOURCE_KIND",
    "RELINEARIZATION_KEY_RESOURCE_KIND",
    "RESCALE_RESOURCE_KIND",
    "ROTATION_KEY_RESOURCE_KIND",
    "KeySwitchExecutionResource",
    "RescaleExecutionResource",
    "ckks_key_resource_kind",
]
