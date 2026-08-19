"""Lower captured semantic operations to role-explicit logical operations."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from .._dialect import operation_name
from .._program import Program
from ._base import PassResult, PassStats
from ._utils import (
    display_name,
    operand_role,
    program_operations,
    replace_operation,
    replacement_operation,
    result_role,
)


@dataclass(frozen=True)
class LowerSemanticToLogicalPass:
    """Classify recognized encrypted semantic operations by operand roles.

    The pass scans every top-level function block and rewrites only supported
    single-result local FHElium semantic operations. Other names, unknown roles,
    regions, successors, and unsupported arities remain structurally intact and
    are reported as skipped where they match the semantic surface.
    """

    name: str = "lower-semantic-to-logical"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Lower matching operations while preserving extension dialect IR."""

        del workspace
        matched = transformed = skipped = 0
        diagnostics: list[str] = []
        for operation in program_operations(program):
            name = operation_name(operation)
            prefix = "fhelium.semantic."
            if result_role(operation) != "encrypted" or not name.startswith(
                prefix
            ):
                continue
            semantic = name.removeprefix(prefix)
            if semantic in {"negate", "roll"}:
                matched += 1
                if (
                    operation.regions
                    or operation.successors
                    or len(operation.operands) != 1
                    or operand_role(operation.operands[0]) != "encrypted"
                ):
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: semantic {semantic} "
                        "requires one encrypted operand"
                    )
                    continue
                replace_operation(
                    operation,
                    replacement_operation(
                        operation,
                        f"fhelium.logical.{semantic}.encrypted",
                    ),
                )
                transformed += 1
                continue
            if semantic not in {"add", "subtract", "multiply"}:
                continue
            matched += 1
            roles = tuple(
                operand_role(operand) for operand in operation.operands
            )
            if (
                operation.regions
                or operation.successors
                or len(roles) != 2
                or any(
                    role not in {"encrypted", "message", "plaintext", "static"}
                    for role in roles
                )
            ):
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: semantic {semantic} has "
                    "unknown roles, regions, successors, or arity"
                )
                continue
            operand_classes = tuple(
                "encrypted" if role == "encrypted" else "public"
                for role in roles
            )
            if operand_classes not in {
                ("encrypted", "encrypted"),
                ("encrypted", "public"),
                ("public", "encrypted"),
            }:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: semantic {semantic} requires "
                    "at least one encrypted operand"
                )
                continue
            suffix = "_".join(operand_classes)
            replace_operation(
                operation,
                replacement_operation(
                    operation,
                    f"fhelium.logical.{semantic}.{suffix}",
                ),
            )
            transformed += 1
        if transformed == 0:
            return PassResult.unchanged(
                program,
                matched=matched,
                skipped=skipped,
                diagnostics=tuple(diagnostics),
            )
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                skipped=skipped,
            ),
            tuple(diagnostics),
        )
