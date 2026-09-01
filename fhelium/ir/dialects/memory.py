"""Rank-local device and memory-transfer operations.

Placement changes are represented by real operations rather than attributes on
unrelated arithmetic.  A transfer consumes one value and one launch-bound
device resource and returns the same IR value type.  Concrete movement,
allocation, and same-device alias choices remain Backend responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException

from .._operation_catalog import (
    OperationSpec,
    flat_with_attributes,
    registered_operation_spec,
)
from ._common import OpenStateType


@irdl_attr_definition
class DeviceType(OpenStateType):
    """Launch-bound execution device with optional provider metadata."""

    name = "fhelium_memory.device"


@irdl_op_definition
class TransferOp(IRDLOperation):
    """Move or copy one rank-local value to a selected device and memory space."""

    name = "fhelium_memory.transfer"

    value = operand_def()
    target = operand_def(DeviceType)
    result = result_def()
    memory_space = opt_attr_def(StringAttr)
    traits = traits_def(Pure())

    def __init__(
        self,
        value: SSAValue | Operation,
        target: SSAValue | Operation,
        *,
        result_type: Attribute | None = None,
        memory_space: str | StringAttr | None = None,
        attributes: Mapping[str, Attribute] | None = None,
    ) -> None:
        attrs: dict[str, Attribute | None] = dict(attributes or {})
        attrs["memory_space"] = (
            StringAttr(memory_space)
            if isinstance(memory_space, str)
            else memory_space
        )
        value_type = SSAValue.get(value).type
        super().__init__(
            operands=[value, target],
            result_types=[result_type or value_type],
            attributes=attrs,
        )

    def verify_(self) -> None:
        if self.value.type != self.result.type:
            raise VerifyException(
                "fhelium_memory.transfer source and result types must match"
            )
        if self.memory_space is not None and self.memory_space.data not in {
            "default",
            "pageable_host",
            "pinned_host",
        }:
            raise VerifyException(
                "fhelium_memory.transfer memory_space must be "
                "'default', 'pageable_host', or 'pinned_host'"
            )


def _transfer_specification(operation: Operation) -> tuple[str, ...]:
    diagnostics = list(flat_with_attributes("memory_space")(operation))
    if len(operation.operands) == 2 and len(operation.results) == 1:
        if operation.operands[0].type != operation.results[0].type:
            diagnostics.append("source and result types must match")
    return tuple(diagnostics)


OPERATION_SPECS: tuple[OperationSpec, ...] = (
    registered_operation_spec(
        TransferOp,
        "memory",
        validator=_transfer_specification,
    ),
)
"""Local specifications owned by the memory dialect."""


FHEliumMemory = Dialect(
    "fhelium_memory",
    [TransferOp],
    [DeviceType],
)
"""Rank-local device and memory-transfer dialect."""


__all__ = [
    "DeviceType",
    "FHEliumMemory",
    "OPERATION_SPECS",
    "TransferOp",
]
