"""Argument conversion from Backend invocations to whole libfhelium operations.

Whole operations receive their numerical operands as the parallel
``table_names`` and ``tables`` lists of a named table, and key-switching
operations receive the flattened fields of the C++ ``KeySwitchOptions``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


def named_tables(
    names: Sequence[str], values: Sequence[torch.Tensor]
) -> tuple[list[str], list[torch.Tensor]]:
    """Return the trailing ``len(names)`` values paired with their names."""
    names = list(names)
    return names, list(values[len(values) - len(names) :])


def mapping_tables(
    tensors: Mapping[str, torch.Tensor], prefix: str = ""
) -> tuple[list[str], list[torch.Tensor]]:
    """Return the named-table lists of a mapping, with names under ``prefix``."""
    return [prefix + name for name in tensors], list(tensors.values())


def flat_digits(digits: Iterable[Iterable[int]]) -> list[int]:
    """Flatten ``(start, stop, key_digit)`` hybrid digit triples."""
    return [int(value) for span in digits for value in span]


def key_switch_options(
    attributes: Mapping[str, object], prefix: str = ""
) -> dict[str, object]:
    """Return the key-switch fields of invocation attributes.

    ``prefix`` selects attributes of a nested key switch, such as the
    ``low_`` switch of encapsulated modulus raising.
    """

    def get(name, default=None):
        value = attributes.get(prefix + name, default)
        if value is None:
            raise KeyError(prefix + name)
        return value

    return {
        "ntt_backend": str(get("ntt_backend")),
        "digits": flat_digits(get("digits")),
        "key_row_start": int(get("key_row_start", 0)),
        "key_p_row_start": int(attributes.get(prefix + "key_p_row_start", -1)),
        "key_q_rows": [int(i) for i in attributes.get(prefix + "key_q_rows", ())],
        "basis_conversion": str(get("basis_conversion", "exact")),
        "p_count": int(get("p_count")),
        "moddown_rounding": str(
            attributes.get(
                "rounding", attributes.get("moddown_rounding", "nearest")
            )
        ),
        "input_domain": str(get("input_domain", "coefficient")),
        "output_domain": str(get("output_domain", "coefficient")),
        "galois_generator": int(get("galois_generator", 5)),
        "key_seeds": [],
        "key_rounds": int(attributes.get(prefix + "key_rounds", 0)),
    }


def seeded_key_operands(
    attributes: Mapping[str, object],
    inputs: Sequence[torch.Tensor],
    start: int,
    count: int,
    prefix: str = "",
) -> tuple[dict[str, object], int]:
    """Return the key-switch options with the seed operands at ``start``.

    Seeded keys (``key_rounds`` > 0) supply ``count`` seed operands after the
    operation's other operands; dense keys supply none. The second result is
    the index after the seed operands.
    """
    options = key_switch_options(attributes, prefix)
    if options["key_rounds"]:
        options["key_seeds"] = list(inputs[start : start + count])
        start += count
    return options, start


__all__ = [
    "flat_digits",
    "key_switch_options",
    "seeded_key_operands",
    "mapping_tables",
    "named_tables",
]
