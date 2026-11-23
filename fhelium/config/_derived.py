"""Configuration facts derived once per CKKS parameter set.

Program operations carry their CKKS parameters as JSON text. :func:`config_from_text` returns one :class:`CkksConfig` object per distinct text, so the facts that capture, lowering, selection and Backend preparation derive from the same parameters are stored once in :func:`derived_fact`'s table on that object.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable
from functools import lru_cache
from typing import TypeVar

from .ckks import CkksConfig

_T = TypeVar("_T")


@lru_cache(maxsize=64)
def config_from_text(text: str) -> CkksConfig:
    """Parse serialized CKKS parameters, returning one object per distinct text."""

    return CkksConfig.parse(json.loads(text))


def derived_fact(
    config: CkksConfig, kind: str, key: Hashable, compute: Callable[[], _T]
) -> _T:
    """Return the fact ``kind`` of ``config`` for ``key``, computing it on first use.

    Facts are functions of the configuration and the key. Callers that mutate a returned container copy it first.
    """

    # The configuration is immutable; cached_property stores its values the same way.
    table = config.__dict__.get("_derived_facts")
    if table is None:
        table = config.__dict__["_derived_facts"] = {}
    entry = (kind, key)
    try:
        return table[entry]
    except KeyError:
        value = table[entry] = compute()
        return value


__all__ = ["config_from_text", "derived_fact"]
