"""Lower specialized collective intent to generic visible combine regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    DecisionRecord,
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Block
from xdsl.rewriter import Rewriter

from fhelium.ir import (
    EXECUTION_IMPLEMENTATION_ATTRIBUTE,
)
from fhelium.ir.dialects import ckks, distributed
from fhelium.ir.dialects._common import OpenStateType

from .._operation_transforms import program_operations


@dataclass(frozen=True)
class LowerSpecializedCollectivesPass:
    """Replace selected specialized collectives with generic region forms.

    The pass exposes local combine arithmetic without proving associativity,
    rank uniformity, cross-rank ordering, or deadlock freedom. Callers remain
    free to preserve specialized operations for whole-operation providers.

    With ``max_group_size``, only reductions over groups of at most that many
    ranks are lowered. The default pipeline lowers two-rank groups, where the
    gathered fold receives the same bytes as a reduce-scatter and all-gather
    and needs one collective and one modular addition, and preserves larger
    groups for the reduce-scatter implementation.
    """

    lower_ciphertext_add: bool = True
    max_group_size: int | None = None
    name: str = "lower-specialized-collectives"

    def __post_init__(self) -> None:
        if type(self.lower_ciphertext_add) is not bool:
            raise TypeError("lower_ciphertext_add must be bool")

    def _lowered(self, operation: distributed.AllReduceAddCiphertextOp) -> bool:
        if self.max_group_size is None:
            return True
        size = cast(OpenStateType, operation.group.type).state.data.get("size")
        return (
            isinstance(size, IntegerAttr)
            and size.value.data <= self.max_group_size
        )

    def run(self, compilation: "Compilation") -> PassResult:
        """Expose CKKS add as the combine region of ciphertext all-reduce."""
        program = compilation.program
        workspace = compilation.workspace

        del workspace
        matches = tuple(
            operation
            for operation in program_operations(program)
            if isinstance(operation, distributed.AllReduceAddCiphertextOp)
            and self._lowered(operation)
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
                parameters=tuple(operation.parameters),
            )
            attributes = dict(operation.attributes)
            configuration = attributes.pop("ckks_config", None)
            if configuration is not None:
                addition.attributes["ckks_config"] = configuration
            combine.add_ops((addition, distributed.YieldOp(addition)))
            replacement = distributed.AllReduceOp(
                operation.value,
                operation.group,
                combine,
                result_type=operation.result.type,
                attributes=attributes,
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
