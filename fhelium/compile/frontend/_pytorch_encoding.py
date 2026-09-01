"""Encode PyTorch capture targets and literals as neutral IR attributes."""

from __future__ import annotations

import math

import torch


def target_symbol(target: object) -> str:
    """Return a stable textual description of a captured Torch call target.

    The result names frontend semantics; it does not authorize an importer to
    load or execute a Python object with that name.
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
    """Encode one immutable Python frontend value as JSON-compatible data."""

    if isinstance(value, torch.Tensor):
        raise TypeError("Tensor values must use symbolic material references")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Compile-time real literals must be finite")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Compile-time complex literals must be finite")
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
        "The PyTorch frontend cannot serialize literal type "
        f"{value_type.__module__}.{value_type.__qualname__}; represent it as "
        "a symbolic material or frontend extension operation"
    )


__all__ = ["encode_literal", "target_symbol"]
