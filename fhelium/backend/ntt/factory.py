"""Construct configured NTT schedule executors from policies and tables."""

from __future__ import annotations

import torch

from fhelium.config.ntt import (
    CompactFixedRadixPolicy,
    CompactRadix2Policy,
    IndexedRadix2Policy,
    NttBackendPolicy,
)
from fhelium.backend.ntt.executors.compact_radix2 import (
    CompactRadix2NttBackend,
)
from fhelium.backend.ntt.executors.indexed_radix2 import (
    IndexedRadix2NttBackend,
)
from fhelium.backend.ntt.executors.power_of_two_radix import (
    CompactPowerOfTwoRadixNttBackend,
)
from fhelium.backend.ntt.interface import NttBackend
from fhelium.backend.ntt.tables import (
    CompactPowerOfTwoRadixTables,
    CompactRadix2Tables,
    IndexedRadix2Tables,
    NttTables,
)


def create_ntt_backend(
    policy: NttBackendPolicy,
    *,
    ntt_tables: NttTables,
    rns_params: torch.Tensor,
) -> NttBackend:
    """Construct a schedule executor only from a matching policy/table pair."""

    if isinstance(policy, IndexedRadix2Policy):
        if not isinstance(ntt_tables, IndexedRadix2Tables):
            raise TypeError(
                f"Policy {policy.name!r} requires IndexedRadix2Tables, got "
                f"{type(ntt_tables).__name__}"
            )
        return IndexedRadix2NttBackend(
            policy=policy,
            ntt_tables=ntt_tables,
            rns_params=rns_params,
        )

    if isinstance(policy, CompactRadix2Policy):
        if not isinstance(ntt_tables, CompactRadix2Tables):
            raise TypeError(
                f"Policy {policy.name!r} requires CompactRadix2Tables, got "
                f"{type(ntt_tables).__name__}"
            )
        return CompactRadix2NttBackend(
            policy=policy,
            ntt_tables=ntt_tables,
            rns_params=rns_params,
        )

    if isinstance(policy, CompactFixedRadixPolicy):
        if not isinstance(ntt_tables, CompactPowerOfTwoRadixTables):
            raise TypeError(
                f"Policy {policy.name!r} requires CompactPowerOfTwoRadixTables, got "
                f"{type(ntt_tables).__name__}"
            )
        return CompactPowerOfTwoRadixNttBackend(
            policy=policy,
            ntt_tables=ntt_tables,
            rns_params=rns_params,
        )

    raise AssertionError(f"Unhandled NTT policy {policy}")
