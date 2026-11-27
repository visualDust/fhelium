"""Positional table operands of the whole native operations.

A whole native operation receives each table family as one ``Tensor?[]``
argument. The slot order is the field order of the family's table struct in
``fhelium/operation_tables.h`` and is reported by the
``table_slot_names`` operator; an absent table is ``None``. Families with
digit groups (``key_switch`` and ``encapsulation``) append one group of
``digit{i}_`` slots per key-switching digit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import cache

import torch

from fhelium.native.wrapper import ckks_ops

# Prepared ring tables store the forward NTT parameter column as
# ``parameters``; the ring struct names it ``forward_0``.
_ALIASES = {"forward_0": "parameters"}


@cache
def slot_names(family: str, digit_count: int = 0) -> tuple[str, ...]:
    """Return the positional slot names of one table family."""
    return tuple(ckks_ops.table_slot_names(family, digit_count))


def digit_count(names: Iterable[str], prefix: str = "") -> int:
    """Return the number of ``{prefix}digit{i}_`` groups among ``names``."""
    pattern = re.compile(re.escape(prefix) + r"digit(\d+)_")
    return 1 + max(
        (
            int(match.group(1))
            for name in names
            if (match := pattern.match(name))
        ),
        default=-1,
    )


def table_slots(
    family: str,
    tables: Mapping[str, torch.Tensor],
    *,
    prefix: str = "",
    digits: int | None = None,
) -> list[torch.Tensor | None]:
    """Order named ``tables`` into the positional slots of ``family``.

    ``prefix`` selects the names of one NTT basis or direction, such as
    ``q_forward_`` for the ``ntt`` family. The digit-group count is read from
    the names unless ``digits`` is given.
    """
    if digits is None:
        digits = digit_count(
            tables, "low_" if family == "encapsulation" else ""
        )
    slots: list[torch.Tensor | None] = []
    for name in slot_names(family, digits):
        value = tables.get(prefix + name)
        if value is None and name in _ALIASES:
            value = tables.get(prefix + _ALIASES[name])
        slots.append(value)
    return slots


def named_slots(
    family: str,
    names: Iterable[str],
    values: Iterable[torch.Tensor],
    **options: object,
) -> list[torch.Tensor | None]:
    """Order parallel ``names`` and ``values`` into the slots of ``family``."""
    return table_slots(family, dict(zip(names, values, strict=True)), **options)  # type: ignore[arg-type]


@cache
def _galois_tables(
    log_n: int, galois_element: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    # Built outside any active tracing or fake mode: the tables are prepared
    # data that a traced operation reads, not operations of its trace.
    from torch.utils._python_dispatch import _disable_current_modes

    with _disable_current_modes():
        return tuple(ckks_ops.galois_table_set(log_n, galois_element, device))


def galois_slots(
    ring_dimension: int,
    galois_elements: Iterable[int],
    device: torch.device | str,
) -> list[torch.Tensor | None]:
    """Return the Galois tables of ``galois_elements`` as positional slots.

    Each element contributes its NTT-domain permutation and its coefficient
    permutation indices and signs. The tables are built once per ring
    dimension, element and device and are then reused, so a traced or captured
    operation reads prepared Tensors.
    """
    log_n = ring_dimension.bit_length() - 1
    device = torch.device(device)
    return [
        table
        for element in galois_elements
        for table in _galois_tables(log_n, int(element), device)
    ]


def conjugation_slots(
    ring_dimension: int, device: torch.device | str
) -> list[torch.Tensor | None]:
    """Return the Galois tables of complex conjugation, $X\\mapsto X^{2N-1}$."""
    return galois_slots(ring_dimension, (2 * ring_dimension - 1,), device)


__all__ = [
    "conjugation_slots",
    "digit_count",
    "galois_slots",
    "named_slots",
    "slot_names",
    "table_slots",
]
