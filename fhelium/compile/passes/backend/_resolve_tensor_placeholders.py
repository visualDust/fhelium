"""Match Program Tensor references against an ordinary symbol table."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass

import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.dialects.func import CallOp
from xdsl.rewriter import Rewriter

from fhelium.ir.dialects import core, fusion
from fhelium.ir.dialects._common import OpenStateType, state_attribute

from ..._pipeline import PassResult, PassStats


@dataclass(frozen=True)
class ResolveTensorPlaceholdersPass:
    """Bind Program Tensor references from a symbol-to-Tensor table.

    A partial table resolves the symbols it contains and leaves other references unchanged. Linking checks the references still required for execution. Refined facts propagate through fusion regions and into the arguments and results of called functions.
    """

    name: str = "resolve-tensor-placeholders"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        materials = compilation.material_bindings
        matched = resolved = 0
        for operation in program.walk():
            if not isinstance(operation, core.MaterialRefOp):
                continue
            matched += 1
            if operation.symbol is None:
                continue
            tensor = materials.get(operation.symbol.data)
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Tensor table entry {operation.symbol.data!r} is not a Tensor"
                )
            value_type = operation.value.type
            if isinstance(value_type, OpenStateType):
                facts = {
                    "shape": state_attribute(tuple(tensor.shape)),
                    "strides": state_attribute(tuple(tensor.stride())),
                    "dtype": state_attribute(str(tensor.dtype)),
                    "device": state_attribute(str(tensor.device)),
                }
                additions = {
                    name: value
                    for name, value in facts.items()
                    if name not in value_type.state.data
                    or value_type.state.data[name] == StringAttr("unknown")
                }
                for name in ("shape", "strides"):
                    represented = value_type.state.data.get(name)
                    actual = facts[name]
                    if (
                        isinstance(represented, ArrayAttr)
                        and isinstance(actual, ArrayAttr)
                        and len(represented) == len(actual)
                    ):
                        dimensions = tuple(
                            known
                            if isinstance(known, IntegerAttr)
                            and known.value.data >= 0
                            else supplied
                            for known, supplied in zip(
                                represented, actual, strict=True
                            )
                        )
                        if dimensions != tuple(represented):
                            additions[name] = ArrayAttr(dimensions)
                if additions:
                    Rewriter.replace_value_with_new_type(
                        operation.value, value_type.with_state(additions)
                    )
            resolved += 1

        def refine_interfaces(block):
            for operation in tuple(block.ops):
                if isinstance(operation, fusion.FusedOp):
                    for argument, operand in zip(
                        operation.body.block.args, operation.inputs, strict=True
                    ):
                        if argument.type != operand.type:
                            Rewriter.replace_value_with_new_type(
                                argument, operand.type
                            )
                for region in operation.regions:
                    for nested in region.blocks:
                        refine_interfaces(nested)
                if isinstance(operation, fusion.FusedOp):
                    for result, value in zip(
                        operation.outputs,
                        operation.body.block.last_op.values,
                        strict=True,
                    ):
                        if result.type != value.type:
                            Rewriter.replace_value_with_new_type(
                                result, value.type
                            )

        functions = {
            function.sym_name.data: function for function in program.functions
        }
        main = program.single_block("main")
        called = []
        for operation in main.ops:
            if not isinstance(operation, CallOp):
                continue
            callee = functions[operation.callee.root_reference.data]
            if callee not in called:
                called.append(callee)
                for argument, operand in zip(
                    callee.body.block.args, operation.arguments, strict=True
                ):
                    if argument.type != operand.type:
                        Rewriter.replace_value_with_new_type(
                            argument, operand.type
                        )
        for callee in called:
            refine_interfaces(callee.body.block)
            callee.update_function_type()
        for operation in main.ops:
            if not isinstance(operation, CallOp):
                continue
            outputs = functions[
                operation.callee.root_reference.data
            ].function_type.outputs
            for result, result_type in zip(operation.res, outputs, strict=True):
                if result.type != result_type:
                    Rewriter.replace_value_with_new_type(result, result_type)
        refine_interfaces(main)
        program.function().update_function_type()
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=resolved,
                skipped=matched - resolved,
            ),
        )


__all__ = ["ResolveTensorPlaceholdersPass"]
