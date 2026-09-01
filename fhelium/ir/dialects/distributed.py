"""Rank-local structured operations for SPMD collective execution.

Programs refer to launch-bound process groups and observe the current group rank
without owning process-group lifecycle.  Collective operations describe only
one rank's local operand and result.  Their local verifiers do not prove
cross-rank ordering, uniform control flow, associativity, or deadlock freedom.
"""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    region_def,
    result_def,
    traits_def,
)
from xdsl.traits import IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException

from .._operation_catalog import (
    OperationSpec,
    flat_with_attributes,
    flat_without_attributes,
    registered_operation_spec,
    unsupported_attributes,
)
from ._common import OpenStateType
from .ckks import CiphertextType


@irdl_attr_definition
class GroupType(OpenStateType):
    """Launch-bound collective group with optional provider metadata."""

    name = "fhelium_dist.group"


class _GroupQueryOp(IRDLOperation):
    group = operand_def(GroupType)
    result = result_def(IndexType)
    traits = traits_def(Pure())

    def __init__(
        self,
        group: SSAValue | Operation,
        *,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[group],
            result_types=[IndexType()],
            attributes=attributes,
        )


@irdl_op_definition
class RankOp(_GroupQueryOp):
    """Read this SPMD instance's rank in one launch-bound group."""

    name = "fhelium_dist.rank"


@irdl_op_definition
class GroupSizeOp(_GroupQueryOp):
    """Read the number of ranks in one launch-bound group."""

    name = "fhelium_dist.group_size"


@irdl_op_definition
class BroadcastOp(IRDLOperation):
    """Broadcast one rank-local value from a statically selected group rank."""

    name = "fhelium_dist.broadcast"

    value = operand_def()
    group = operand_def(GroupType)
    result = result_def()
    root = attr_def(IntegerAttr)

    def __init__(
        self,
        value: SSAValue | Operation,
        group: SSAValue | Operation,
        *,
        root: int | IntegerAttr,
        result_type: Attribute | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        attrs["root"] = (
            root
            if isinstance(root, IntegerAttr)
            else IntegerAttr.from_int_and_width(root, 64)
        )
        value_type = SSAValue.get(value).type
        super().__init__(
            operands=[value, group],
            result_types=[result_type or value_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.value.type != self.result.type:
            raise VerifyException(
                "fhelium_dist.broadcast operand and result types must match"
            )


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Return one local combine result from a generic collective region."""

    name = "fhelium_dist.yield"

    value = operand_def()
    traits = traits_def(IsTerminator())

    def __init__(self, value: SSAValue | Operation) -> None:
        super().__init__(operands=[value])


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    """Reduce one local value across a group with a visible combine region."""

    name = "fhelium_dist.all_reduce"

    value = operand_def()
    group = operand_def(GroupType)
    result = result_def()
    combine = region_def("single_block")

    def __init__(
        self,
        value: SSAValue | Operation,
        group: SSAValue | Operation,
        combine: Region | Block,
        *,
        result_type: Attribute | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        value_type = SSAValue.get(value).type
        region = combine if isinstance(combine, Region) else Region(combine)
        super().__init__(
            operands=[value, group],
            result_types=[result_type or value_type],
            regions=[region],
            attributes=attributes,
        )

    def verify_(self) -> None:
        if self.value.type != self.result.type:
            raise VerifyException(
                "fhelium_dist.all_reduce operand and result types must match"
            )
        blocks = tuple(self.combine.blocks)
        if len(blocks) != 1:
            raise VerifyException(
                "fhelium_dist.all_reduce combine region must have one block"
            )
        block = blocks[0]
        if len(block.args) != 2:
            raise VerifyException(
                "fhelium_dist.all_reduce combine block requires two arguments"
            )
        if any(argument.type != self.value.type for argument in block.args):
            raise VerifyException(
                "fhelium_dist.all_reduce combine arguments must match the "
                "reduced value type"
            )
        terminator = block.last_op
        if not isinstance(terminator, YieldOp):
            raise VerifyException(
                "fhelium_dist.all_reduce combine block must end with "
                "fhelium_dist.yield"
            )
        if terminator.value.type != self.value.type:
            raise VerifyException(
                "fhelium_dist.all_reduce yielded value must match the "
                "reduced value type"
            )


@irdl_op_definition
class AllReduceAddCiphertextOp(IRDLOperation):
    """Reduce ciphertexts across a group using rank-local CKKS addition."""

    name = "fhelium_dist.all_reduce_add_ciphertext"

    value = operand_def(CiphertextType)
    group = operand_def(GroupType)
    result = result_def(CiphertextType)

    def __init__(
        self,
        value: SSAValue | Operation,
        group: SSAValue | Operation,
        *,
        result_type: Attribute | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        value_type = SSAValue.get(value).type
        super().__init__(
            operands=[value, group],
            result_types=[result_type or value_type],
            attributes=attributes,
        )

    def verify_(self) -> None:
        if self.value.type != self.result.type:
            raise VerifyException(
                "fhelium_dist.all_reduce_add_ciphertext operand and result "
                "types must match"
            )


def _same_value_type_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if len(operation.operands) >= 1 and len(operation.results) == 1:
        if operation.operands[0].type != operation.results[0].type:
            diagnostics.append("operand and result types must match")
    return tuple(diagnostics)


def _broadcast_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_with_attributes("root")(operation))
    diagnostics.extend(_same_value_type_specification(operation))
    return tuple(diagnostics)


def _all_reduce_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(_same_value_type_specification(operation))
    if operation.properties:
        diagnostics.append("properties are unsupported")
    if operation.successors:
        diagnostics.append("successors are unsupported")
    unsupported = unsupported_attributes(operation, ())
    if unsupported:
        diagnostics.append(f"unsupported attributes {list(unsupported)}")
    if len(operation.regions) != 1:
        diagnostics.append("requires one combine region")
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        RankOp,
        "distributed",
        validator=flat_without_attributes,
    ),
    registered_operation_spec(
        GroupSizeOp,
        "distributed",
        validator=flat_without_attributes,
    ),
    registered_operation_spec(
        BroadcastOp,
        "distributed",
        effect="opaque",
        validator=_broadcast_specification,
    ),
    registered_operation_spec(
        AllReduceOp,
        "distributed",
        effect="opaque",
        validator=_all_reduce_specification,
    ),
    registered_operation_spec(
        AllReduceAddCiphertextOp,
        "distributed",
        effect="opaque",
        validator=lambda operation: (
            *flat_without_attributes(operation),
            *_same_value_type_specification(operation),
        ),
    ),
    registered_operation_spec(
        YieldOp,
        "distributed",
        validator=flat_without_attributes,
    ),
)
"""Local specifications owned by the distributed dialect."""


FHEliumDistributed = Dialect(
    "fhelium_dist",
    [
        RankOp,
        GroupSizeOp,
        BroadcastOp,
        AllReduceOp,
        AllReduceAddCiphertextOp,
        YieldOp,
    ],
    [GroupType],
)
"""Rank-local SPMD collective operations and group resources."""


__all__ = [
    "AllReduceAddCiphertextOp",
    "AllReduceOp",
    "BroadcastOp",
    "FHEliumDistributed",
    "GroupSizeOp",
    "GroupType",
    "OPERATION_SPECS",
    "RankOp",
    "YieldOp",
]
