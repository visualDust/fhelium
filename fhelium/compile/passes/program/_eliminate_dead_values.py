"""Remove unused pure computations across represented execution regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass

from xdsl.ir import Block
from xdsl.rewriter import Rewriter


from ..._pipeline import PassResult, PassStats
from .._operation_transforms import display_name
from ._purity import known_pure, known_region_blocks


@dataclass(frozen=True)
class EliminateDeadValuesPass:
    """Erase unused operations according to registered effects.

    The pass handles high-level CKKS, lowered RNS/NTT, and pure fusion regions.
    It enters known structured regions, preserving their terminators and result
    interfaces. Unknown operations/regions, random draws, mutation, and opaque
    effects remain roots. Registration must name the exact operation class;
    inheriting a Pure trait does not grant an extension permission to disappear.
    """

    name: str = "eliminate-dead-values"

    def run(self, compilation: "Compilation") -> PassResult:
        """Walk blocks in reverse dependency order and remove dead producers."""
        program = compilation.program
        workspace = compilation.workspace

        del workspace
        removed: list[str] = []
        removed_count = 0
        matched = 0

        def visit(block: Block) -> None:
            nonlocal matched, removed_count
            for operation in reversed(tuple(block.ops)):
                for nested in known_region_blocks(operation):
                    visit(nested)
                if not known_pure(operation):
                    continue
                matched += 1
                if any(any(result.uses) for result in operation.results):
                    continue
                removed.append(display_name(operation))
                removed_count += sum(1 for _ in operation.walk())
                Rewriter.erase_op(operation)

        for function in program.functions:
            for block in function.body.blocks:
                visit(block)
        if not removed:
            return PassResult.unchanged(program, matched=matched)
        preview = ", ".join(removed[:8])
        if len(removed) > 8:
            preview += f", ... (+{len(removed) - 8})"
        return PassResult(
            program,
            PassStats(
                matched=matched, transformed=len(removed), removed=removed_count
            ),
            (f"removed unused pure computations: {preview}",),
        )
