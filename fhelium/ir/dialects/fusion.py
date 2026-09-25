"""Backend-independent regions of operations selected for joint execution."""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import Attribute, Block, Dialect, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    region_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException

from .._operation_catalog import OperationSpec
from .._dependencies import OperationDependencies, region_dependencies


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Return the selected results of a fusion region."""

    name = "fhelium_fusion.yield"
    values = var_operand_def()
    traits = traits_def(IsTerminator())

    def __init__(self, values: Sequence[SSAValue]) -> None:
        super().__init__(operands=[values])


@irdl_op_definition
class FusedOp(IRDLOperation):
    """Execute a pure region with one selected Backend implementation.

    The body preserves the original operation semantics and returns every value
    used outside the region. Block arguments correspond to the outer operands,
    including resource bindings. Fusion itself does not change value state or
    imply a particular device, code generator, or CUDA Graph capture.
    """

    name = "fhelium_fusion.execute"
    inputs = var_operand_def()
    outputs = var_result_def()
    body = region_def("single_block")
    traits = traits_def(Pure(), IsolatedFromAbove())

    def dependencies(self) -> OperationDependencies:
        """Describe result reads in this operation's value coordinates."""
        return region_dependencies(self)

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        body: Block,
        *,
        attributes: dict[str, Attribute] | None = None,
    ) -> None:
        super().__init__(
            operands=[inputs],
            result_types=[result_types],
            regions=[Region(body)],
            attributes=attributes,
        )

    def verify_(self) -> None:
        block = self.body.block
        if tuple(arg.type for arg in block.args) != tuple(
            arg.type for arg in self.inputs
        ):
            raise VerifyException(
                "Fusion block arguments must match outer operands"
            )
        terminator = block.last_op
        if not isinstance(terminator, YieldOp):
            raise VerifyException(
                "Fusion region requires a fusion.yield terminator"
            )
        if tuple(value.type for value in terminator.values) != tuple(
            value.type for value in self.outputs
        ):
            raise VerifyException(
                "Fusion yielded types must match result types"
            )
        from .. import DEFAULT_OPERATION_SPECS

        for op in block.ops:
            if op is terminator:
                continue
            if (
                type(op) is UnrealizedConversionCastOp
                and len(op.inputs) == len(op.outputs) == 1
            ):
                continue
            spec = DEFAULT_OPERATION_SPECS.get(op.name)
            if (
                spec is None
                or spec.operation_type is not type(op)
                or spec.effect != "pure"
            ):
                raise VerifyException(
                    "Fusion bodies require registered pure operation semantics"
                )


FHEliumFusion = Dialect("fhelium_fusion", [FusedOp, YieldOp], [])
OPERATION_SPECS = (
    OperationSpec(
        FusedOp.name,
        "fusion",
        None,
        None,
        operation_type=FusedOp,
        dependencies=region_dependencies,
    ),
    OperationSpec(YieldOp.name, "fusion", None, 0, operation_type=YieldOp),
)

__all__ = ["FHEliumFusion", "FusedOp", "YieldOp", "OPERATION_SPECS"]
