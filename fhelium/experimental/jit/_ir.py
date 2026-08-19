"""Serializable capture metadata for the canonical xDSL Program.

The xDSL module owned by ``Program`` is the canonical graph and SSA
representation. This module supplies its capture-time role vocabulary, target
symbols, and literal encoding.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

ValueRole = Literal["encrypted", "message", "plaintext", "static"]
CallKind = Literal["function", "method", "module"]


def target_symbol(target: object) -> str:
    """Return a stable textual symbol for a captured call target.

    The symbol is descriptive metadata. Program import preserves it as text;
    runtime target selection uses an audited public Torch target or an explicit
    ``workspace['torch_handlers']`` binding.
    """

    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", type(target).__module__)
    name = getattr(target, "__name__", None)
    if (
        isinstance(module, str)
        and isinstance(name, str)
        and name
        and (
            module == "torch"
            or module.startswith("torch.")
            or module in {"operator", "_operator"}
        )
    ):
        return f"{module}.{name}"
    qualname = getattr(target, "__qualname__", None)
    if not isinstance(qualname, str) or not qualname:
        qualname = getattr(target, "__name__", type(target).__qualname__)
    return f"{module}.{qualname}"


def encode_literal(value: object) -> object:
    """Encode one immutable Python value as canonical JSON-compatible data.

    Capture represents Tensor values as symbolic material operations and
    retains their live values in the Workspace; this encoder accepts the
    immutable literal forms stored directly in Program attributes.
    """

    if isinstance(value, torch.Tensor):
        raise TypeError("Tensor values must be emitted as symbolic materials")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JIT real literals must be finite")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("JIT complex literals must be finite")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, torch.dtype):
        return {"kind": "torch.dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"kind": "torch.device", "value": str(value)}
    if isinstance(value, torch.layout):
        return {"kind": "torch.layout", "value": str(value)}
    value_type = type(value)
    raise TypeError(
        "JIT cannot serialize executable literal type "
        f"{value_type.__module__}.{value_type.__qualname__}; bind it through "
        "a symbolic material or extension operation"
    )


__all__ = [
    "CallKind",
    "ValueRole",
    "encode_literal",
    "target_symbol",
]
