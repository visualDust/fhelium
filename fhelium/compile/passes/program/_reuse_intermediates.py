"""Reuse internal deterministic transforms and simplify reference plumbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from dataclasses import dataclass

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks, core, ntt, rns

from ..._pipeline import DecisionRecord, PassResult, PassStats
from ._purity import known_pure, known_region_blocks

# These operations are deterministic mathematical transforms. Reuse is restricted
# to internal results, so independently returned arrays retain distinct storage.
_TRANSFORMS = (
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.ToMontgomeryResiduesOp,
    ckks.ToStandardResiduesOp,
    ntt.CoefficientStandardToNttMontgomeryOp,
    ntt.CoefficientMontgomeryToNttMontgomeryOp,
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.NttMontgomeryToCoefficientMontgomeryOp,
    rns.StandardToMontgomeryOp,
    rns.MontgomeryToStandardOp,
)
_VIEWS = (
    UnrealizedConversionCastOp,
    rns.ExtractComponentOp,
    rns.RestrictDepthOp,
    rns.ReinterpretScaleOp,
    ckks.ModSwitchOp,
    ckks.ReinterpretScaleOp,
)
# These built-in mathematical consumers produce fresh arrays rather than
# returning a view of an operand. Unknown consumers end the alias proof.
_FRESH_CONSUMERS = (
    *_TRANSFORMS,
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.NegateOp,
    ckks.MultiplyOp,
    ckks.AddPlaintextOp,
    ckks.MultiplyPlaintextOp,
    ckks.RelinearizeOp,
    ckks.RotateOp,
    ckks.RotateManyOp,
    ckks.ConjugateOp,
    ckks.RescaleOp,
    rns.AddStandardOp,
    rns.SubtractStandardOp,
    rns.NegateStandardOp,
    rns.AddPlaintextOp,
    rns.MultiplyPlaintextOp,
    rns.MontgomeryMultiplyOp,
    rns.PackTwoComponentsOp,
    rns.PackThreeComponentsOp,
    rns.RescaleDropLeadingPrimesOp,
)


def _internal(
    value: SSAValue, block: Block, seen: set[SSAValue] | None = None
) -> bool:
    visited = set() if seen is None else seen
    if value in visited:
        return True
    visited.add(value)
    for use in tuple(value.uses):
        consumer = use.operation
        if consumer.parent_block() is not block:
            return False
        if type(consumer) in _VIEWS and known_pure(consumer):
            if not all(
                _internal(result, block, visited) for result in consumer.results
            ):
                return False
        elif (
            type(consumer) not in _FRESH_CONSUMERS
            or not known_pure(consumer)
            or EXECUTION_IMPLEMENTATION_ATTRIBUTE in consumer.attributes
        ):
            return False
    return True


def _expression_key(operation: Operation) -> tuple[object, ...]:
    return (
        type(operation),
        tuple(operation.operands),
        tuple(result.type for result in operation.results),
        tuple(sorted(operation.attributes.items())),
    )


@dataclass(frozen=True)
class ReuseIntermediatesPass:
    """Remove duplicate references/casts and share internal transform results.

    Transform reuse requires the same SSA operands, result types, and attributes.
    A mutation, random draw, unknown operation, or region ends a local reuse run.
    Directly returned results, views that escape, and unknown consumers prevent
    reuse so this pass does not merge independently observable output storage.
    It does not cancel approximate arithmetic or move rounding across sums.
    """

    name: str = "reuse-intermediates"

    def run(self, compilation: "Compilation") -> PassResult:
        program = compilation.program
        workspace = compilation.workspace
        del workspace
        references = casts = transforms = 0
        decisions: list[DecisionRecord] = []

        def replace(operation: Operation, values: tuple[SSAValue, ...]) -> None:
            for source, target in zip(operation.results, values, strict=True):
                source.replace_all_uses_with(target)
            Rewriter.erase_op(operation)

        def visit(block: Block) -> None:
            nonlocal references, casts, transforms
            refs: dict[tuple[object, ...], Operation] = {}
            expressions: dict[tuple[object, ...], Operation] = {}
            cast_values: dict[tuple[object, ...], SSAValue] = {}
            for operation in tuple(block.ops):
                for child in known_region_blocks(operation):
                    visit(child)
                if type(operation) in (core.ResourceRefOp, core.MaterialRefOp):
                    key = _expression_key(operation)
                    previous = refs.get(key)
                    if previous is None:
                        refs[key] = operation
                    else:
                        replace(operation, tuple(previous.results))
                        references += 1
                    continue
                if (
                    type(operation) is UnrealizedConversionCastOp
                    and len(operation.inputs) == len(operation.outputs) == 1
                    and known_pure(operation)
                ):
                    source = operation.inputs[0]
                    while (
                        type(source.owner) is UnrealizedConversionCastOp
                        and len(source.owner.inputs)
                        == len(source.owner.outputs)
                        == 1
                        and known_pure(source.owner)
                    ):
                        source = source.owner.inputs[0]
                    operation.operands[0] = source
                    result = operation.outputs[0]
                    if source.type == result.type:
                        replace(operation, (source,))
                        casts += 1
                    else:
                        key = (
                            source,
                            result.type,
                            tuple(sorted(operation.attributes.items())),
                        )
                        previous = cast_values.get(key)
                        if previous is None:
                            cast_values[key] = result
                        else:
                            replace(operation, (previous,))
                            casts += 1
                    continue
                if not known_pure(operation) or operation.regions:
                    expressions.clear()
                    continue
                if (
                    type(operation) not in _TRANSFORMS
                    or EXECUTION_IMPLEMENTATION_ATTRIBUTE
                    in operation.attributes
                ):
                    continue
                if not all(
                    _internal(result, block) for result in operation.results
                ):
                    continue
                key = _expression_key(operation)
                previous = expressions.get(key)
                if previous is None:
                    expressions[key] = operation
                else:
                    replace(operation, tuple(previous.results))
                    transforms += 1
                    decisions.append(
                        DecisionRecord(
                            operation.name,
                            selected="reuse-prior-transform",
                            details=(
                                "same SSA input and represented transform state; results remain internal",
                            ),
                        )
                    )

        for function in program.functions:
            for block in function.body.blocks:
                visit(block)
        total = references + casts + transforms
        if not total:
            return PassResult.unchanged(program)
        return PassResult(
            program,
            PassStats(matched=total, transformed=total, removed=total),
            (
                f"deduplicated references={references}, removed casts={casts}, "
                f"reused transform operations={transforms}",
            ),
            tuple(decisions),
        )
