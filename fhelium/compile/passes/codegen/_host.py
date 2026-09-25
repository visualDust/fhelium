"""Emit prepared host control flow shared by execution and Python export."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, Operation, SSAValue

from fhelium.compile.codegen import PythonCodegenError
from fhelium.ir.dialects import core, distributed, fusion

from ._common import SourceNames, constant_literal


def tuple_source(values: Sequence[str]) -> str:
    return "(" + ", ".join(values) + ("," if len(values) == 1 else "") + ")"


def emit_host_body(
    block: Block,
    names: SourceNames,
    call: Callable[[Operation, tuple[str, ...]], str],
    reference: Callable[[Operation], str],
    *,
    prepared_regions: Callable[[Operation], bool],
    tuple_return: bool = True,
) -> list[str]:
    """Translate supported structured IR once, retaining dataflow in locals.

    Call and reference emission bind execution objects or produce their exported
    expressions. Generated code does not look up operation classes or SSA values.
    """
    lines: list[str] = []
    ordinal = 0
    binary = {
        arith.AddiOp: "+",
        arith.SubiOp: "-",
        arith.MuliOp: "*",
        arith.DivUIOp: "//",
        arith.DivSIOp: "//",
        arith.RemUIOp: "%",
        arith.RemSIOp: "%",
    }

    def write(indent: int, text: str) -> None:
        lines.extend("    " * indent + line for line in text.splitlines())

    def assign(
        indent: int, targets: Sequence[str], expressions: Sequence[str]
    ) -> None:
        if targets:
            write(indent, f"{', '.join(targets)} = {', '.join(expressions)}")
        else:
            write(indent, "pass")

    def emit(
        current: Block,
        indent: int,
        yielded: Sequence[str] | None = None,
        region: bool = False,
    ) -> None:
        nonlocal ordinal
        operations = tuple(current.ops)
        # Lexically captured values in structured regions live with their scope.
        flat = not any(
            op.regions and not prepared_regions(op) for op in operations
        )
        last_use: dict[SSAValue, Operation] = {
            result: op for op in operations for result in op.results
        }
        if flat:
            for op in operations:
                for value in op.operands:
                    if value in last_use:
                        last_use[value] = op
        releases: dict[Operation, list[SSAValue]] = {}
        if flat:
            for value, consumer in last_use.items():
                releases.setdefault(consumer, []).append(value)
        for op in operations:
            operands = tuple(names.operand(value) for value in op.operands)
            results = names.results(op)
            if isinstance(
                op, (ReturnOp, scf.YieldOp, distributed.YieldOp, fusion.YieldOp)
            ):
                if yielded is not None:
                    assign(indent, yielded, operands)
                else:
                    expression = (
                        tuple_source(operands)
                        if tuple_return or region
                        else (
                            operands[0]
                            if len(operands) == 1
                            else tuple_source(operands)
                            if operands
                            else "None"
                        )
                    )
                    if region:
                        expression = (
                            f"__fhelium_codegen_tensor_results({expression})"
                        )
                    write(indent, f"return {expression}")
                return
            if isinstance(op, UnrealizedConversionCastOp):
                if len(operands) != 1 or len(results) != 1:
                    raise PythonCodegenError(
                        "Host execution requires one-to-one boundary casts"
                    )
                assign(indent, results, operands)
            elif isinstance(op, (core.MaterialRefOp, core.ResourceRefOp)):
                assign(indent, results, (reference(op),))
            elif isinstance(op, core.ConstantOp):
                assign(indent, results, (repr(constant_literal(op)),))
            elif isinstance(op, arith.ConstantOp):
                if not isinstance(op.value, (IntegerAttr, FloatAttr)):
                    raise PythonCodegenError(
                        "Host arithmetic requires a scalar constant"
                    )
                assign(indent, results, (repr(op.value.value.data),))
            elif type(op) in binary:
                assign(
                    indent,
                    results,
                    (
                        f"__fhelium_codegen_index({operands[0]}) {binary[type(op)]} __fhelium_codegen_index({operands[1]})",
                    ),
                )
            elif isinstance(op, (arith.MinUIOp, arith.MaxUIOp)):
                function = "min" if isinstance(op, arith.MinUIOp) else "max"
                assign(
                    indent,
                    results,
                    (
                        f"{function}(__fhelium_codegen_index({operands[0]}), __fhelium_codegen_index({operands[1]}))",
                    ),
                )
            elif isinstance(op, arith.IndexCastOp):
                assign(
                    indent,
                    results,
                    (f"__fhelium_codegen_index({operands[0]})",),
                )
            elif isinstance(op, arith.SelectOp):
                assign(
                    indent,
                    results,
                    (
                        f"{operands[1]} if bool({operands[0]}) else {operands[2]}",
                    ),
                )
            elif isinstance(op, arith.CmpiOp):
                predicate = (
                    "==",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                )[op.predicate.value.data]
                assign(
                    indent,
                    results,
                    (
                        f"__fhelium_codegen_index({operands[0]}) {predicate} __fhelium_codegen_index({operands[1]})",
                    ),
                )
            elif isinstance(op, scf.IfOp):
                write(indent, f"if bool({operands[0]}):")
                emit(op.true_region.block, indent + 1, results)
                write(indent, "else:")
                emit(op.false_region.block, indent + 1, results)
            elif isinstance(op, scf.ForOp):
                body = op.body.block
                body_args = tuple(
                    names.input(value, i) for i, value in enumerate(body.args)
                )
                assign(indent, results, operands[3:])
                write(
                    indent,
                    f"for {body_args[0]} in range(__fhelium_codegen_index({operands[0]}), __fhelium_codegen_index({operands[1]}), __fhelium_codegen_index({operands[2]})):",
                )
                if body_args[1:]:
                    assign(indent + 1, body_args[1:], results)
                emit(body, indent + 1, results)
            else:
                regions = []
                if op.regions and not prepared_regions(op):
                    for body in op.regions:
                        region_name = f"__fhelium_codegen_region_{ordinal}"
                        ordinal += 1
                        arguments = tuple(
                            names.input(value, i)
                            for i, value in enumerate(body.block.args)
                        )
                        write(
                            indent,
                            f"def {region_name}(__fhelium_codegen_arguments):",
                        )
                        if arguments:
                            write(
                                indent + 1,
                                f"{', '.join(arguments)}{',' if len(arguments) == 1 else ''} = __fhelium_codegen_arguments",
                            )
                        emit(body.block, indent + 1, region=True)
                        regions.append(region_name)
                expression = call(op, tuple(regions))
                if results:
                    write(
                        indent,
                        f"{', '.join(results)}{',' if len(results) == 1 else ''} = {expression}",
                    )
                else:
                    write(
                        indent,
                        f"if {expression} != ():\n    raise ValueError('A result-free operation returned values')",
                    )
            if flat:
                expired = [
                    names.operand(value) for value in releases.get(op, ())
                ]
                if expired:
                    write(indent, "del " + ", ".join(expired))
        raise PythonCodegenError(
            "Host block requires a return or yield operation"
        )

    emit(block, 1)
    return lines
