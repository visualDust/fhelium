"""Component expression graphs and indexed loads for Triton code generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from math import prod
from typing import cast

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation, SSAValue

from fhelium.ir.dialects import ckks, fusion, ntt, rns

from ._layout import result_layout

ARITHMETIC_OPS = (
    ckks.AddCompressedPlaintextOp,
    ckks.MultiplyCompressedPlaintextOp,
    rns.AddStandardOp,
    rns.SubtractStandardOp,
    rns.NegateStandardOp,
    rns.MontgomeryMultiplyOp,
    rns.AddMontgomeryLazyOp,
    rns.MultiplyPlaintextOp,
    rns.AddPlaintextOp,
    rns.StandardToMontgomeryOp,
    rns.MontgomeryToStandardOp,
)
STRUCTURAL_OPS = (
    rns.ExtractComponentOp,
    rns.PackTwoComponentsOp,
    rns.PackThreeComponentsOp,
)
NTT_OPS = (
    ntt.CoefficientStandardToNttMontgomeryOp,
    ntt.CoefficientMontgomeryToNttMontgomeryOp,
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.NttMontgomeryToCoefficientMontgomeryOp,
)
SUPPORTED_OPERATIONS = (
    ARITHMETIC_OPS + STRUCTURAL_OPS + (rns.KeySwitchDigitProductOp,)
)


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Expr, ...] = ()
    data: tuple[int, ...] = ()


@dataclass(frozen=True)
class InputLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    component_axis: bool = False

    @property
    def components(self) -> int:
        return self.shape[0] if self.component_axis else 1

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.shape[1:-2] if self.component_axis else self.shape[:-2]


@dataclass(frozen=True)
class ValueExpr:
    components: tuple[Expr, ...]
    shape: tuple[int, ...]
    component_axis: bool
    root: int
    view: tuple[int, ...] = ()

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.shape[1:-2] if self.component_axis else self.shape[:-2]


@dataclass(frozen=True)
class Anchor:
    operation: Operation
    source: ValueExpr
    result: ValueExpr
    table_indices: tuple[int, ...]


@dataclass(frozen=True)
class BoundGraph:
    layouts: tuple[InputLayout, ...]
    outputs: tuple[ValueExpr, ...]
    roots: Mapping[int, ValueExpr]
    anchors: tuple[Anchor, ...] = ()

    @property
    def batch_count(self) -> int:
        return max(prod(v.batch_shape) for v in self.roots.values())


def component_axis(value_type: object) -> bool:
    if isinstance(value_type, ckks.CiphertextType):
        return True
    state = getattr(getattr(value_type, "state", None), "data", {})
    count = state.get("component_count", state.get("components"))
    return isinstance(count, IntegerAttr)


@dataclass
class ExpressionGraph:
    """Bind a region's SSA DAG to input shapes without specializing data values."""

    operation: fusion.FusedOp

    @cached_property
    def table_arguments(self) -> tuple[SSAValue, ...]:
        arguments = set(self.operation.body.block.args)
        used: set[SSAValue] = set()
        for operation in self.operation.body.block.ops:
            if isinstance(operation, NTT_OPS):
                used.update(operation.operands[1:])
            elif isinstance(operation, ARITHMETIC_OPS):
                used.add(operation.operands[-1])
            elif isinstance(operation, rns.KeySwitchDigitProductOp):
                used.update(operation.operands[1:])
        return tuple(
            arg
            for arg in self.operation.body.block.args
            if arg in arguments & used
        )

    @cached_property
    def tensor_types(self) -> tuple[object, ...]:
        return tuple(
            arg.type
            for arg in self.operation.body.block.args
            if arg not in self.table_arguments
        )

    @cached_property
    def input_indices(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        tensors = tuple(self.operation.body.block.args)
        return (
            tuple(
                i
                for i, arg in enumerate(tensors)
                if arg not in self.table_arguments
            ),
            tuple(
                i
                for i, arg in enumerate(tensors)
                if arg in self.table_arguments
            ),
        )

    @cached_property
    def parameter_index(self) -> int:
        for op in self.operation.body.block.ops:
            if isinstance(op, ARITHMETIC_OPS):
                return self.table_arguments.index(op.operands[-1])
            if isinstance(op, rns.KeySwitchDigitProductOp):
                return self.table_arguments.index(op.parameters)
        raise ValueError("Pointwise fusion has no RNS parameter operand")

    @cached_property
    def key_inputs(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (
                self.table_arguments.index(op.key),
                int(op.key_digit_index.value.data),
                int(
                    cast(IntegerAttr, op.attributes["key_row_start"]).value.data
                ),
            )
            for op in self.operation.body.block.ops
            if isinstance(op, rns.KeySwitchDigitProductOp)
        )

    def bind(self, layouts: tuple[InputLayout, ...]) -> BoundGraph:
        values: dict[SSAValue, ValueExpr] = {}
        roots: dict[int, ValueExpr] = {}
        anchors: list[Anchor] = []
        input_index = 0
        for arg in self.operation.body.block.args:
            if arg in self.table_arguments:
                continue
            layout = layouts[input_index]
            value = ValueExpr(
                tuple(
                    Expr("input", data=(input_index, c))
                    for c in range(layout.components)
                ),
                layout.shape,
                layout.component_axis,
                -input_index - 1,
            )
            roots[value.root] = value
            values[arg] = value
            input_index += 1
        if input_index + len(self.key_inputs) != len(layouts):
            raise ValueError(
                "Fusion Tensor inputs do not match the region arguments"
            )
        next_key_input = input_index
        outputs: tuple[ValueExpr, ...] = ()
        for ordinal, operation in enumerate(self.operation.body.block.ops):
            if isinstance(operation, fusion.YieldOp):
                outputs = tuple(values[v] for v in operation.values)
                continue
            if isinstance(operation, UnrealizedConversionCastOp):
                values[operation.outputs[0]] = values[operation.inputs[0]]
                continue
            operands = [
                values[v]
                for v in operation.operands
                if v not in self.table_arguments
            ]
            if isinstance(operation, rns.ExtractComponentOp):
                source = operands[0]
                index = int(operation.component.value.data)
                result_layout(
                    type(operation),
                    [source.shape],
                    [source.component_axis],
                    component=index,
                )
                values[operation.result] = ValueExpr(
                    (source.components[index],),
                    source.shape[1:],
                    False,
                    source.root,
                    (*source.view, index),
                )
                continue
            if isinstance(
                operation, (rns.PackTwoComponentsOp, rns.PackThreeComponentsOp)
            ):
                first = operands[0]
                result_layout(
                    type(operation),
                    [v.shape for v in operands],
                    [v.component_axis for v in operands],
                )
                result = ValueExpr(
                    tuple(v.components[0] for v in operands),
                    (len(operands), *first.shape),
                    True,
                    ordinal,
                )
            elif isinstance(operation, rns.KeySwitchDigitProductOp):
                digit = operands[0]
                result_layout(
                    type(operation), [digit.shape], [digit.component_axis]
                )
                key_layout = layouts[next_key_input]
                if key_layout.shape != (2, *digit.shape[-2:]):
                    raise ValueError(
                        "Key-switch key rows or polynomial extent differ from the active digit"
                    )
                key_components = tuple(
                    Expr("input", data=(next_key_input, c)) for c in range(2)
                )
                result = ValueExpr(
                    tuple(
                        Expr("multiply", (digit.components[0], key))
                        for key in key_components
                    ),
                    (2, *digit.shape),
                    True,
                    ordinal,
                )
                next_key_input += 1
            elif isinstance(operation, NTT_OPS):
                source = operands[0]
                result = ValueExpr(
                    tuple(
                        Expr("anchor", (expr,), (len(anchors), c))
                        for c, expr in enumerate(source.components)
                    ),
                    source.shape,
                    source.component_axis,
                    ordinal,
                )
                anchors.append(
                    Anchor(
                        operation,
                        source,
                        result,
                        tuple(
                            self.table_arguments.index(v)
                            for v in operation.operands[1:]
                        ),
                    )
                )
            elif isinstance(
                operation,
                (
                    ckks.AddCompressedPlaintextOp,
                    ckks.MultiplyCompressedPlaintextOp,
                ),
            ):
                result = compact_value(
                    type(operation),
                    operands,
                    ordinal,
                    layout=cast(
                        StringAttr, operation.attributes["compression_layout"]
                    ).data,
                    ntt_plaintext=getattr(
                        operation.attributes.get("polynomial_domain"),
                        "data",
                        "coefficient",
                    )
                    == "ntt",
                )
            elif isinstance(operation, ARITHMETIC_OPS):
                result = arithmetic_value(
                    type(operation),
                    operands,
                    ordinal,
                    ntt_plaintext=(
                        isinstance(operation, rns.AddPlaintextOp)
                        and operation.polynomial_domain.data == "ntt"
                    ),
                )
            else:
                raise ValueError(
                    f"Selected Triton expression code generation cannot implement {operation.name}"
                )
            values[operation.results[0]] = result
            roots[result.root] = result
        return BoundGraph(layouts, outputs, roots, tuple(anchors))


def compact_value(
    operation_type, operands, root, *, layout, ntt_plaintext=False
):
    """Bind compact plaintext indexing and arithmetic to the ciphertext extent."""
    first, compact, *implicit = operands
    result_layout(
        operation_type,
        [v.shape for v in operands],
        [v.component_axis for v in operands],
    )
    if (layout == "strided_sparse") != bool(implicit) or len(implicit) > 1:
        raise ValueError(
            "Compact arithmetic implicit operands must match the layout"
        )
    u, n = compact.shape[-1], first.shape[-1]
    expanded = Expr(
        "compact",
        (compact.components[0], *(v.components[0] for v in implicit)),
        (("cyclic", "contiguous", "strided_sparse").index(layout), u, n // u),
    )
    if operation_type is ckks.AddCompressedPlaintextOp:
        if not ntt_plaintext:
            expanded = Expr("from_montgomery", (expanded,))
        components = (
            Expr("add", (first.components[0], expanded)),
            *first.components[1:],
        )
    else:
        components = tuple(
            Expr("multiply", (component, expanded))
            for component in first.components
        )
    return ValueExpr(components, first.shape, first.component_axis, root)


def arithmetic_value(
    operation_type: type[Operation],
    operands: list[ValueExpr],
    root: int,
    *,
    ntt_plaintext: bool = False,
) -> ValueExpr:
    first = operands[0]
    result_layout(
        operation_type,
        [v.shape for v in operands],
        [v.component_axis for v in operands],
    )
    code = {
        rns.AddStandardOp: "add",
        rns.SubtractStandardOp: "subtract",
        rns.NegateStandardOp: "negate",
        rns.AddMontgomeryLazyOp: "add_lazy",
        rns.MontgomeryMultiplyOp: "multiply",
        rns.MultiplyPlaintextOp: "multiply",
        rns.StandardToMontgomeryOp: "to_montgomery",
        rns.MontgomeryToStandardOp: "from_montgomery",
        rns.AddPlaintextOp: "add_lazy",
    }[operation_type]
    plaintext_op = operation_type in (
        rns.MultiplyPlaintextOp,
        rns.AddPlaintextOp,
    )
    if operation_type is rns.AddPlaintextOp and not ntt_plaintext:
        raise ValueError(
            "Triton plaintext addition requires NTT-domain operands"
        )
    components: list[Expr] = []
    for i, lhs in enumerate(first.components):
        if operation_type is rns.AddPlaintextOp and i:
            components.append(lhs)
            continue
        args = [lhs]
        for value in operands[1:]:
            if plaintext_op:
                args.append(value.components[0])
            else:
                args.append(
                    value.components[i if len(value.components) > 1 else 0]
                )
        components.append(Expr(code, tuple(args)))
    return ValueExpr(tuple(components), first.shape, first.component_axis, root)


@dataclass
class Emitter:
    """Emit an expression at arbitrary coefficient, prime-row and batch indices."""

    layouts: tuple[InputLayout, ...]
    lines: list[str] = field(default_factory=list)
    cache: dict[tuple[Expr, str, str, str, str], str] = field(
        default_factory=dict
    )

    def emit(
        self,
        expr: Expr,
        index: str,
        row: str,
        batch: str,
        overrides: Mapping[Expr, str] | None = None,
        mask: str = "mask",
    ) -> str:
        if overrides is not None and expr in overrides:
            return overrides[expr]
        # Override sets are scoped to an emitter. NTT drivers use a fresh emitter
        # for each prolog/epilog so cached values cannot cross transform stages.
        key = (expr, index, row, batch, mask)
        if key in self.cache:
            return self.cache[key]
        if expr.op == "anchor":
            raise ValueError(
                "NTT anchor expression needs its stage-result override"
            )
        if expr.op == "compact":
            layout, unique, repeat = expr.data
            selected = (
                f"(({index}) % {unique})"
                if layout == 0
                else f"(({index}) // {repeat})"
            )
            explicit = self.emit(
                expr.args[0], selected, row, batch, overrides, mask
            )
            if layout == 2:
                implicit = self.emit(
                    expr.args[1], "0", row, batch, overrides, mask
                )
                result = f"e{len(self.cache)}"
                self.lines.append(
                    f"{result} = tl.where((({index}) % {repeat}) == 0, {explicit}, {implicit})"
                )
            else:
                result = explicit
            self.cache[key] = result
            return result
        args = [
            self.emit(a, index, row, batch, overrides, mask) for a in expr.args
        ]
        variable = f"e{len(self.cache)}"
        if expr.op == "input":
            number, component = expr.data
            layout = self.layouts[number]
            shape = layout.batch_shape
            strides = (
                layout.strides[1:-2]
                if layout.component_axis
                else layout.strides[:-2]
            )
            offset = [
                f"({index}) * {layout.strides[-1]}",
                f"({row}) * {layout.strides[-2]}",
            ]
            if layout.component_axis:
                offset.append(str(component * layout.strides[0]))
            for axis, (extent, stride) in enumerate(
                zip(shape, strides, strict=True)
            ):
                if extent > 1:
                    offset.append(
                        f"((({batch}) // {prod(shape[axis + 1 :])}) % {extent}) * {stride}"
                    )
            expression = f"tl.load(x{number} + {' + '.join(offset)}, mask={mask}, other=0).to(tl.int64)"
        elif expr.op == "multiply":
            expression = f"montgomery_mul({args[0]}, {args[1]}, q, k, RADIX)"
        elif expr.op == "to_montgomery":
            expression = f"montgomery_mul({args[0]}, r2, q, k, RADIX)"
        elif expr.op == "from_montgomery":
            expression = f"montgomery_reduce({args[0]}, q, k, RADIX)"
        elif expr.op == "add_lazy":
            expression = f"add_lazy({args[0]}, {args[1]}, q)"
        elif expr.op == "add":
            expression = f"canonicalize(add_lazy({args[0]}, {args[1]}, q), q)"
        elif expr.op == "subtract":
            expression = (
                f"canonicalize(subtract_lazy({args[0]}, {args[1]}, q), q)"
            )
        elif expr.op == "negate":
            expression = f"tl.where(canonicalize({args[0]}, q) == 0, 0, q - canonicalize({args[0]}, q))"
        else:
            raise ValueError(f"Unknown Triton expression: {expr.op}")
        self.lines.append(f"{variable} = {expression}")
        self.cache[key] = variable
        return variable
