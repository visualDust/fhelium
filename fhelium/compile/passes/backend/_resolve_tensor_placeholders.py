"""Match Program Tensor references against an ordinary symbol table."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass

import torch
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr
from xdsl.rewriter import Rewriter

from fhelium.ir.dialects import core
from fhelium.ir.dialects._common import OpenStateType

from ..._pipeline import PassResult, PassStats


@dataclass(frozen=True)
class ResolveTensorPlaceholdersPass:
    """Supply available Tensors while leaving missing references unresolved.

    The pass performs no allocation, generation or callback. Callers can run it
    with a partial table; linking checks references that are still required.
    """

    name: str = "resolve-tensor-placeholders"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        shared_data = compilation.workspace
        materials = compilation.material_bindings
        matched = resolved = 0
        for operation in program.single_block("main").walk():
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
                    "shape": ArrayAttr(
                        IntegerAttr(int(n), 64) for n in tensor.shape
                    ),
                    "strides": ArrayAttr(
                        IntegerAttr(int(n), 64) for n in tensor.stride()
                    ),
                    "dtype": StringAttr(str(tensor.dtype)),
                    "device": StringAttr(str(tensor.device)),
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
