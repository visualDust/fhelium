"""Assemble dialect-owned operation specifications into one checked catalog."""

from __future__ import annotations

from xdsl.dialects.builtin import UnrealizedConversionCastOp

from ._operation_catalog import (
    OperationEffect,
    OperationSpec,
    OperationSpecRegistry,
    OperationValidator,
)
from .dialects import (
    REGISTERED_DIALECTS,
    ckks,
    core,
    distributed,
    logical,
    memory,
    ntt,
    rns,
    semantic,
)
from .dialects import torch as torch_dialect

_DIALECT_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    *core.OPERATION_SPECS,
    *semantic.OPERATION_SPECS,
    *logical.OPERATION_SPECS,
    *ckks.OPERATION_SPECS,
    *rns.OPERATION_SPECS,
    *ntt.OPERATION_SPECS,
    *memory.OPERATION_SPECS,
    *distributed.OPERATION_SPECS,
    *torch_dialect.OPERATION_SPECS,
)

_BRIDGE_OPERATION_SPECS = (
    OperationSpec(
        UnrealizedConversionCastOp.name,
        "conversion",
        1,
        1,
        (None,),
        (None,),
        operation_type=UnrealizedConversionCastOp,
    ),
)


def _verify_dialect_catalog(specifications: tuple[OperationSpec, ...]) -> None:
    """Require one dialect-owned specification per registered operation."""

    registered = {
        operation_type.name: operation_type
        for dialect in REGISTERED_DIALECTS
        for operation_type in dialect.operations
    }
    described = {
        specification.name: specification.operation_type
        for specification in specifications
    }
    missing = tuple(sorted(registered.keys() - described.keys()))
    extra = tuple(sorted(described.keys() - registered.keys()))
    mismatched = tuple(
        sorted(
            name
            for name in registered.keys() & described.keys()
            if registered[name] is not described[name]
        )
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            "Dialect operation specification inventory differs from registered "
            f"operations: missing={missing}, extra={extra}, "
            f"class_mismatch={mismatched}"
        )


_verify_dialect_catalog(_DIALECT_OPERATION_SPECS)

DEFAULT_OPERATION_SPECS = OperationSpecRegistry(
    (*_DIALECT_OPERATION_SPECS, *_BRIDGE_OPERATION_SPECS)
)
"""Specifications for every first-party operation and boundary cast."""


__all__ = [
    "DEFAULT_OPERATION_SPECS",
    "OperationEffect",
    "OperationSpec",
    "OperationSpecRegistry",
    "OperationValidator",
]
