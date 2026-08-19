"""Remove unreachable known-pure values while preserving effect roots."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.func import ReturnOp
from xdsl.rewriter import Rewriter

from .._dialect import operation_name
from .._program import Program
from ._base import PassResult, PassStats
from ._utils import display_name, program_operations

_KNOWN_PURE_NAMES = frozenset(
    {
        "fhelium.material.ref",
        "fhelium.constant",
        "fhelium.semantic.add",
        "fhelium.semantic.subtract",
        "fhelium.semantic.multiply",
        "fhelium.semantic.negate",
        "fhelium.semantic.roll",
        "fhelium.logical.add.encrypted_encrypted",
        "fhelium.logical.add.encrypted_public",
        "fhelium.logical.add.public_encrypted",
        "fhelium.logical.subtract.encrypted_encrypted",
        "fhelium.logical.subtract.encrypted_public",
        "fhelium.logical.subtract.public_encrypted",
        "fhelium.logical.multiply.encrypted_encrypted",
        "fhelium.logical.multiply.encrypted_public",
        "fhelium.logical.multiply.public_encrypted",
        "fhelium.logical.negate.encrypted",
        "fhelium.logical.roll.encrypted",
        "fhelium.ckks.negate",
        "fhelium.ckks.rotate",
        "fhelium.ckks.to_ntt",
        "fhelium.ckks.from_ntt",
        "fhelium.ckks.add",
        "fhelium.ckks.subtract",
        "fhelium.ckks.multiply",
        "fhelium.ckks.add_plaintext",
        "fhelium.ckks.multiply_plaintext",
        "fhelium.ckks.relinearize",
        "fhelium.ckks.rescale",
    }
)


def _known_pure(name: str) -> bool:
    return name in _KNOWN_PURE_NAMES or name in {
        f"fhelium.ckks.prepare.{operation}.{role}"
        for operation in ("add", "multiply")
        for role in ("message", "plaintext", "static")
    }


@dataclass(frozen=True)
class EliminateDeadValuesPass:
    """Delete dead operations from a closed set of pure names.

    Every unknown or extension operation is an effectful liveness root. Known
    operations carrying properties, regions, or successors are also retained,
    so this module-wide pass preserves unclassified and structural effects.
    """

    name: str = "eliminate-dead-values"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Compute module-wide SSA liveness from returns and effect roots."""

        del workspace
        operations = program_operations(program)
        live_values = {
            operand
            for operation in program.walk()
            if isinstance(operation, ReturnOp)
            for operand in operation.operands
        }
        root_operations = {
            operation
            for operation in operations
            if not _known_pure(operation_name(operation))
        }
        for operation in root_operations:
            live_values.update(operation.results)

        changed = True
        while changed:
            changed = False
            for operation in reversed(operations):
                if operation in root_operations or any(
                    result in live_values for result in operation.results
                ):
                    for operand in operation.operands:
                        if operand not in live_values:
                            live_values.add(operand)
                            changed = True

        removable = tuple(
            operation
            for operation in operations
            if _known_pure(operation_name(operation))
            and not operation.properties
            and not operation.regions
            and not operation.successors
            and not any(result in live_values for result in operation.results)
        )
        if not removable:
            return PassResult.unchanged(program)
        names = tuple(display_name(operation) for operation in removable)
        for operation in reversed(removable):
            Rewriter.erase_op(operation)
        preview = ", ".join(names[:8])
        if len(names) > 8:
            preview += f", ... (+{len(names) - 8})"
        return PassResult(
            program,
            PassStats(
                matched=len(removable),
                transformed=len(removable),
                removed=len(removable),
            ),
            (f"removed unreachable pure values: {preview}",),
        )
