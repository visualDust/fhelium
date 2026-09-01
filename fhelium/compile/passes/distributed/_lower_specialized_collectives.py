"""Lower specialized collective intent to generic visible combine regions."""

from __future__ import annotations

from ..._pipeline import (
    DecisionRecord,
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Block
from xdsl.rewriter import Rewriter

from fhelium.ir import (
    EXECUTION_IMPLEMENTATION_ATTRIBUTE,
    Program,
)
from fhelium.ir.dialects import ckks, distributed

from .._operation_transforms import program_operations


@dataclass(frozen=True)
class LowerSpecializedCollectivesPass:
    """Replace selected specialized collectives with generic region forms.

    The pass exposes local combine arithmetic without proving associativity,
    rank uniformity, cross-rank ordering, or deadlock freedom. Callers remain
    free to preserve specialized operations for whole-operation providers.
    """

    lower_ciphertext_add: bool = True
    name: str = "lower-specialized-collectives"

    def __post_init__(self) -> None:
        if type(self.lower_ciphertext_add) is not bool:
            raise TypeError("lower_ciphertext_add must be bool")

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Expose CKKS add as the combine region of ciphertext all-reduce."""

        del workspace
        matches = tuple(
            operation
            for operation in program_operations(program)
            if isinstance(operation, distributed.AllReduceAddCiphertextOp)
        )
        if not matches:
            return PassResult.unchanged(program)
        if not self.lower_ciphertext_add:
            preserved_decisions = tuple(
                DecisionRecord(
                    operation.name,
                    "preserve",
                    ("preserve", "generic-combine-region"),
                    ("specialized collective preserved by caller",),
                )
                for operation in matches
            )
            return PassResult.unchanged(
                program,
                matched=len(matches),
                skipped=len(matches),
                decisions=preserved_decisions,
            )

        decisions: list[DecisionRecord] = []
        for operation in matches:
            assigned = operation.attributes.get(
                EXECUTION_IMPLEMENTATION_ATTRIBUTE
            )
            if isinstance(assigned, StringAttr):
                raise ValueError(
                    "Specialized collective "
                    f"{operation.name!r} requests Backend implementation "
                    f"{assigned.data!r}; preserve the operation or remove "
                    "the assignment before lowering"
                )
            value_type = operation.value.type
            combine = Block(arg_types=(value_type, value_type))
            addition = ckks.AddOp(
                combine.args[0],
                combine.args[1],
                value_type,
            )
            combine.add_ops((addition, distributed.YieldOp(addition)))
            replacement = distributed.AllReduceOp(
                operation.value,
                operation.group,
                combine,
                result_type=operation.result.type,
                attributes=operation.attributes,
            )
            replacement.result.name_hint = operation.result.name_hint
            Rewriter.replace_op(
                operation,
                (replacement,),
                new_results=(replacement.result,),
            )
            decisions.append(
                DecisionRecord(
                    operation.name,
                    "generic-combine-region",
                    ("preserve", "generic-combine-region"),
                    ("combine=fhelium_ckks.add",),
                )
            )
        return PassResult(
            program,
            PassStats(
                matched=len(matches),
                transformed=len(matches),
                inserted=3 * len(matches),
                removed=len(matches),
            ),
            decisions=tuple(decisions),
        )


__all__ = ["LowerSpecializedCollectivesPass"]
