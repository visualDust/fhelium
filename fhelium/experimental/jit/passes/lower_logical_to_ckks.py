"""Lower role-explicit logical operations to explicit CKKS primitives."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import StringAttr

from .._dialect import create_ir_context, create_operation, operation_name
from .._program import Program
from ._base import PassResult, PassStats
from ._utils import (
    bool_attribute,
    display_name,
    program_operations,
    replace_operation,
    replacement_operation,
    string_attribute,
    with_obligations,
)


@dataclass(frozen=True)
class LowerLogicalToCkksPass:
    """Lower logical operations with local CKKS prerequisites.

    Recognized operations are rewritten across all top-level function blocks.
    Locally unresolved patterns are retained and counted as skipped, allowing
    later specialized passes to supply policy or extension handling.
    """

    name: str = "lower-logical-to-ckks"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Lower locally ready operations and report retained patterns."""

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        ir_context = create_ir_context()
        for operation in program_operations(program):
            name = operation_name(operation)
            prefix = "fhelium.logical."
            if not name.startswith(prefix):
                continue
            parts = name.removeprefix(prefix).split(".")
            if len(parts) != 2:
                continue
            arithmetic, operand_classes = parts
            if arithmetic not in {
                "add",
                "subtract",
                "multiply",
                "negate",
                "roll",
            }:
                continue
            matched += 1
            if arithmetic in {"negate", "roll"}:
                if (
                    operand_classes != "encrypted"
                    or len(operation.operands) != 1
                ):
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: logical {arithmetic} "
                        "requires one encrypted operand"
                    )
                    continue
                target = "negate" if arithmetic == "negate" else "rotate"
                replace_operation(
                    operation,
                    replacement_operation(operation, f"fhelium.ckks.{target}"),
                )
                transformed += 1
                continue
            if len(operation.operands) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: logical {arithmetic} "
                    "requires exactly two operands"
                )
                continue
            if operand_classes == "encrypted_encrypted":
                if arithmetic == "multiply" and not bool_attribute(
                    operation.attributes,
                    "multiply_ntt_transitions_inserted",
                ):
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: CT×CT multiply lacks "
                        "explicit NTT transitions"
                    )
                    continue
                attributes = {}
                if arithmetic == "multiply":
                    attributes = with_obligations(
                        {"rescale_condition": StringAttr("always")},
                        {"relinearize", "rescale"},
                    )
                replace_operation(
                    operation,
                    replacement_operation(
                        operation,
                        f"fhelium.ckks.{arithmetic}",
                        attributes=attributes,
                    ),
                )
                transformed += 1
                continue
            if operand_classes not in {
                "encrypted_public",
                "public_encrypted",
            }:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: unsupported logical roles "
                    f"{operand_classes!r}"
                )
                continue
            if not bool_attribute(operation.attributes, "plaintext_prepared"):
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: mixed operation lacks "
                    "plaintext preparation"
                )
                continue
            encrypted_index = 0 if operand_classes == "encrypted_public" else 1
            public_index = 1 - encrypted_index
            ciphertext = operation.operands[encrypted_index]
            plaintext = operation.operands[public_index]
            if arithmetic == "add":
                replace_operation(
                    operation,
                    replacement_operation(
                        operation,
                        "fhelium.ckks.add_plaintext",
                        operands=(ciphertext, plaintext),
                        attributes={},
                    ),
                )
                transformed += 1
                continue
            if arithmetic == "multiply":
                scale_mode = string_attribute(
                    operation.attributes, "scale_mode", "default_scale"
                )
                assert scale_mode is not None
                condition = (
                    "plaintext_scale_not_one"
                    if scale_mode == "runtime_plaintext_scale"
                    else "always"
                )
                attributes = with_obligations(
                    {"rescale_condition": StringAttr(condition)}, {"rescale"}
                )
                replace_operation(
                    operation,
                    replacement_operation(
                        operation,
                        "fhelium.ckks.multiply_plaintext",
                        operands=(ciphertext, plaintext),
                        attributes=attributes,
                    ),
                )
                transformed += 1
                continue

            block = operation.parent_block()
            if block is None:
                raise ValueError("logical operation is not attached to a block")
            negated = create_operation(
                ir_context,
                "fhelium.ckks.negate",
                operands=(ciphertext,),
                result_types=(operation.results[0].type,),
            )
            negated.results[0].name_hint = f"{display_name(operation)}_negated"
            block.insert_op_before(negated, operation)
            inserted += 1
            if operand_classes == "public_encrypted":
                replace_operation(
                    operation,
                    replacement_operation(
                        operation,
                        "fhelium.ckks.add_plaintext",
                        operands=(negated.results[0], plaintext),
                        attributes={},
                    ),
                )
                transformed += 1
                continue
            summed = create_operation(
                ir_context,
                "fhelium.ckks.add_plaintext",
                operands=(negated.results[0], plaintext),
                result_types=(operation.results[0].type,),
            )
            summed.results[0].name_hint = f"{display_name(operation)}_sum"
            block.insert_op_before(summed, operation)
            inserted += 1
            replace_operation(
                operation,
                replacement_operation(
                    operation,
                    "fhelium.ckks.negate",
                    operands=(summed.results[0],),
                    attributes={},
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
                inserted=inserted,
                skipped=skipped,
            ),
            tuple(diagnostics),
        )
