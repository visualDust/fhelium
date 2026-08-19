"""Insert eager relinearization for scheduling obligations."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import StringAttr

from .._dialect import create_ir_context, create_operation
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
class InsertRelinearizationPass:
    """Materialize local relinearization obligations.

    Each eligible single-result operation receives an immediately following
    relinearize operation; existing uses are redirected to that result and any
    remaining scheduling obligations are transferred to it.
    """

    name: str = "insert-relinearization"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        """Insert marked key switches module-wide or return a legal no-op."""

        del workspace
        transformed = 0
        ir_context = create_ir_context()
        for operation in program_operations(program):
            pending = obligations(operation)
            if "relinearize" not in pending or len(operation.results) != 1:
                continue
            remaining = pending - {"relinearize"}
            condition = string_attribute(
                operation.attributes, "rescale_condition", "always"
            )
            operation.attributes = with_obligations(operation.attributes, ())
            relinearized = create_operation(
                ir_context,
                "fhelium.ckks.relinearize",
                operands=(operation.results[0],),
                result_types=(operation.results[0].type,),
                attributes=with_obligations(
                    {"rescale_condition": StringAttr(condition or "always")},
                    remaining,
                ),
            )
            relinearized.results[
                0
            ].name_hint = f"{display_name(operation)}_relinearized"
            insert_after_and_replace_uses(operation, relinearized)
            transformed += 1
        if transformed == 0:
            return PassResult.unchanged(program)
        return PassResult(
            program,
            PassStats(
                matched=transformed,
                transformed=transformed,
                inserted=transformed,
            ),
        )
