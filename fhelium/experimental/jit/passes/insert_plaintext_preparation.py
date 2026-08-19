"""Insert operation-specific plaintext preparation operations."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import StringAttr

from .._dialect import (
    create_ir_context,
    create_operation,
    operation_name,
    value_type,
)
from .._program import Program
from ._base import PassResult, PassStats
from ._utils import (
    bool_attribute,
    display_name,
    operand_role,
    program_operations,
    with_bool_attribute,
)


@dataclass(frozen=True)
class InsertPlaintextPreparationPass:
    """Insert a plaintext-preparation operation for each logical mixed op.

    The operation records whether addition uses the consumer ciphertext scale,
    multiplication encodes a message/static value at the engine default scale,
    or multiplication retains a caller-owned Plaintext's runtime scale. An
    operation marker makes repeated application idempotent.
    """

    name: str = "insert-plaintext-preparation"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Prepare recognized public operands across all function blocks."""

        del workspace
        matched = transformed = skipped = 0
        diagnostics: list[str] = []
        ir_context = create_ir_context()
        for operation in program_operations(program):
            name = operation_name(operation)
            prefix = "fhelium.logical."
            if not name.startswith(prefix):
                continue
            parts = name.removeprefix(prefix).split(".")
            if len(parts) != 2 or parts[0] not in {
                "add",
                "subtract",
                "multiply",
            }:
                continue
            arithmetic, operand_classes = parts
            if operand_classes not in {
                "encrypted_public",
                "public_encrypted",
            } or bool_attribute(operation.attributes, "plaintext_prepared"):
                continue
            matched += 1
            if len(operation.operands) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: mixed logical operation "
                    "requires exactly two operands"
                )
                continue
            encrypted_index = 0 if operand_classes == "encrypted_public" else 1
            public_index = 1 - encrypted_index
            public = operation.operands[public_index]
            ciphertext = operation.operands[encrypted_index]
            public_role = operand_role(public)
            if public_role not in {"message", "plaintext", "static"}:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: public operand has an "
                    "unknown extension role"
                )
                continue
            prepare_kind = (
                "add" if arithmetic in {"add", "subtract"} else "multiply"
            )
            if prepare_kind == "add":
                scale_mode = "ciphertext_scale"
            elif public_role == "plaintext":
                scale_mode = "runtime_plaintext_scale"
            else:
                scale_mode = "default_scale"
            prepare = create_operation(
                ir_context,
                f"fhelium.ckks.prepare.{prepare_kind}.{public_role}",
                operands=(public, ciphertext),
                result_types=(value_type("plaintext"),),
                attributes={
                    "operation": StringAttr(prepare_kind),
                    "source_role": StringAttr(public_role),
                    "scale_mode": StringAttr(scale_mode),
                },
            )
            prepare.results[
                0
            ].name_hint = f"{display_name(operation)}_plaintext"
            block = operation.parent_block()
            if block is None:
                raise ValueError("logical operation is not attached to a block")
            block.insert_op_before(prepare, operation)
            operation.operands[public_index] = prepare.results[0]
            operation.attributes = with_bool_attribute(
                operation.attributes, "plaintext_prepared", True
            )
            operation.attributes["scale_mode"] = StringAttr(scale_mode)
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
                inserted=transformed,
                skipped=skipped,
            ),
            tuple(diagnostics),
        )
