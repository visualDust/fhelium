"""Insert eager rescale for scheduling obligations."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import SSAValue

from .._dialect import create_ir_context, create_operation
from .._dialect import operation_name
from .._program import Program
from ._base import PassResult, PassStats
from ._utils import (
    display_name,
    insert_after_and_replace_uses,
    obligations,
    program_operations,
    string_attribute,
    with_obligations,
)


@dataclass(frozen=True)
class InsertRescalePass:
    """Materialize locally ready rescale obligations at their source.

    A single-result operation with only a rescale obligation receives an
    immediately following rescale operation. Other obligations and malformed
    conditional-rescale inputs retain the source operation and produce skipped
    diagnostics.
    """

    name: str = "insert-rescale"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Insert ready rescales module-wide and report local blockers."""

        del workspace
        matched = transformed = skipped = 0
        diagnostics: list[str] = []
        ir_context = create_ir_context()
        for operation in program_operations(program):
            pending = obligations(operation)
            if "rescale" not in pending or len(operation.results) != 1:
                continue
            matched += 1
            blockers = pending - {"rescale"}
            if blockers:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: rescale waits for obligations "
                    f"{sorted(blockers)}"
                )
                continue
            condition = (
                string_attribute(
                    operation.attributes, "rescale_condition", "always"
                )
                or "always"
            )
            operands: list[SSAValue] = [operation.results[0]]
            if condition == "plaintext_scale_not_one":
                if len(operation.operands) != 2:
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: conditional rescale lacks "
                        "prepared operand"
                    )
                    continue
                operands.append(operation.operands[1])
            operation.attributes = with_obligations(operation.attributes, ())
            plaintext_multiply = (
                operation_name(operation) == "fhelium.ckks.multiply_plaintext"
            )
            if plaintext_multiply:
                coefficient = create_operation(
                    ir_context,
                    "fhelium.ckks.from_ntt",
                    operands=(operation.results[0],),
                    result_types=(operation.results[0].type,),
                )
                coefficient.results[
                    0
                ].name_hint = f"{display_name(operation)}_coefficient"
                insert_after_and_replace_uses(operation, coefficient)
                operands[0] = coefficient.results[0]
            rescaled = create_operation(
                ir_context,
                "fhelium.ckks.rescale",
                operands=operands,
                result_types=(operation.results[0].type,),
                attributes={"condition": StringAttr(condition)},
            )
            rescaled.results[
                0
            ].name_hint = f"{display_name(operation)}_rescaled"
            if plaintext_multiply:
                insert_after_and_replace_uses(coefficient, rescaled)
            else:
                insert_after_and_replace_uses(operation, rescaled)
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
