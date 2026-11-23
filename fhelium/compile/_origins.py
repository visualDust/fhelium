"""Locate candidate local graphs using capture and lowering source hints.

Hints describe where operations were emitted. Consumers establish reuse eligibility from the current SSA graph, its operand ordering and aliases, and current decision dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.ir import Operation


@dataclass
class OperationOrigins:
    """Own request-local operation locations for one concrete Program module."""

    module: Operation
    entries: dict[Operation, dict[str, Any]] = field(default_factory=dict)
    serial: int = 0

    def occurrence(self, source: str) -> str:
        self.serial += 1
        return f"{source}:{self.serial}"


def origin_store(compilation) -> OperationOrigins | None:
    """Supply lowering hints when this request has a local-plan consumer."""
    if compilation.workspace.get("local_plan_reuse") is False:
        return None
    from ._selection_plan import SelectionPlans

    if not isinstance(
        compilation.workspace.get(SelectionPlans), SelectionPlans
    ):
        return None
    store = compilation.workspace.get(OperationOrigins)
    if (
        not isinstance(store, OperationOrigins)
        or store.module is not compilation.program.module
    ):
        store = OperationOrigins(compilation.program.module)
        compilation.workspace[OperationOrigins] = store
    return store


def get_origin(operation, workspace) -> dict[str, Any] | None:
    """Return a hint only for a live operation represented by this workspace."""
    if workspace.get("local_plan_reuse") is False:
        return None
    store = workspace.get(OperationOrigins)
    if (
        not isinstance(store, OperationOrigins)
        or operation.parent_block() is None
    ):
        return None
    return store.entries.get(operation)


def restore_origins(program, workspace, operations, records) -> None:
    """Restore JSON hints against the corresponding current capture operations."""
    if workspace.get("local_plan_reuse") is False:
        workspace.pop(OperationOrigins, None)
        return
    entries = {
        operation: record["origin"]
        for operation, record in zip(operations, records, strict=True)
        if record.get("origin") is not None
        and operation.parent_block() is not None
    }
    workspace[OperationOrigins] = OperationOrigins(program.module, entries)


def clone_origins(
    source_program, target_program, source_workspace, target_workspace
) -> None:
    """Transfer source locations across an ordinary structure-preserving clone.

    A changed operation sequence loses these optional hints; no positional remapping is attempted across rewrites.
    """
    source = source_workspace.get(OperationOrigins)
    target_workspace.pop(OperationOrigins, None)
    if target_workspace.get("local_plan_reuse") is False:
        return
    if (
        not isinstance(source, OperationOrigins)
        or source.module is not source_program.module
    ):
        return
    old_ops, new_ops = (
        tuple(source_program.walk()),
        tuple(target_program.walk()),
    )
    if len(old_ops) != len(new_ops) or any(
        old.name != new.name for old, new in zip(old_ops, new_ops, strict=True)
    ):
        return
    entries = {
        new: source.entries[old]
        for old, new in zip(old_ops, new_ops, strict=True)
        if old in source.entries
    }
    target_workspace[OperationOrigins] = OperationOrigins(
        target_program.module, entries, source.serial
    )


def replace_origin(
    compilation, operation, replacements, *, lowering: str
) -> None:
    """Associate replacement roles with a lowering occurrence and its source site."""
    store = origin_store(compilation)
    if store is None:
        return
    parent = store.entries.pop(operation, None)
    occurrence = store.occurrence("lowering")
    for sequence, replacement in enumerate(replacements):
        if replacement in store.entries:
            # A producer's more specific repeated-body schema remains available.
            continue
        store.entries[replacement] = {
            "source": "lowering",
            "schema": f"{lowering}/v1",
            "occurrence": occurrence,
            "sequence": sequence,
            "role": replacement.name,
            **({"parent": parent} if parent is not None else {}),
        }


__all__ = [
    "OperationOrigins",
    "clone_origins",
    "get_origin",
    "origin_store",
    "replace_origin",
    "restore_origins",
]
