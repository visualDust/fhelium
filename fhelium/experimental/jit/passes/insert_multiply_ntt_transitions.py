"""Insert NTT transitions required by encrypted multiplication."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from .._dialect import create_ir_context, create_operation, operation_name
from .._program import Program
from ._base import PassResult, PassStats
from ._utils import (
    bool_attribute,
    display_name,
    program_operations,
    with_bool_attribute,
)


@dataclass(frozen=True)
class InsertMultiplyNttTransitionsPass:
    """Insert ciphertext NTT transitions for logical multiplication.

    Each encrypted operand of an unmarked binary multiply receives one
    ``fhelium.ckks.to_ntt`` operation. This covers both ciphertext-ciphertext
    and ciphertext-plaintext multiplication and records an idempotence marker.
    Malformed matches remain unchanged as successful skipped operations.
    """

    name: str = "insert-multiply-ntt-transitions"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Insert transitions module-wide or return a legal no-op report."""

        del workspace
        matched = transformed = skipped = 0
        diagnostics: list[str] = []
        ir_context = create_ir_context()
        inserted = 0
        multiply_roles = {
            "fhelium.logical.multiply.encrypted_encrypted": (0, 1),
            "fhelium.logical.multiply.encrypted_public": (0,),
            "fhelium.logical.multiply.public_encrypted": (1,),
        }
        for operation in program_operations(program):
            encrypted_indices = multiply_roles.get(operation_name(operation))
            if encrypted_indices is None or bool_attribute(
                operation.attributes, "multiply_ntt_transitions_inserted"
            ):
                continue
            matched += 1
            if len(operation.operands) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: CT×CT multiply requires "
                    "exactly two operands"
                )
                continue
            block = operation.parent_block()
            if block is None:
                raise ValueError(
                    "multiply operation is not attached to a block"
                )
            converted = []
            for index in encrypted_indices:
                operand = operation.operands[index]
                transition = create_operation(
                    ir_context,
                    "fhelium.ckks.to_ntt",
                    operands=(operand,),
                    result_types=(operand.type,),
                )
                side = "lhs" if index == 0 else "rhs"
                transition.results[
                    0
                ].name_hint = f"{display_name(operation)}_{side}_ntt"
                block.insert_op_before(transition, operation)
                converted.append((index, transition.results[0]))
                inserted += 1
            for index, value in converted:
                operation.operands[index] = value
            operation.attributes = with_bool_attribute(
                operation.attributes,
                "multiply_ntt_transitions_inserted",
                True,
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
                inserted=inserted,
                skipped=skipped,
            ),
            tuple(diagnostics),
        )
