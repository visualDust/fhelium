"""Component expression graphs and indexed loads for Triton code generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from math import prod
import json
from typing import cast

from xdsl.dialects.builtin import (
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation, SSAValue

from fhelium.ir.dialects import ckks, core, fusion, ntt, rns

from ._layout import result_layout
from ._repeat import graph_nodes, repeat_groups

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
    """Describe logical indexing and a preparation-proven pointer owner.

    ``owner`` indexes a region input sharing the allocation; ``offset`` is the
    element displacement from that input's pointer to this logical view.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    component_axis: bool = False
    owner: int | None = None
    offset: int = 0

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
    repeats: tuple[tuple[int, ...], ...] = ()

    @property
    def batch_count(self) -> int:
        return max(prod(v.batch_shape) for v in self.roots.values())


def component_axis(value_type: object) -> bool:
    if isinstance(value_type, ckks.CiphertextType):
        return True
    state = getattr(getattr(value_type, "state", None), "data", {})
    count = state.get("component_count", state.get("components"))
    return isinstance(count, IntegerAttr)


def material_view_owners(
    operation: fusion.FusedOp, data_indices: tuple[int, ...]
) -> tuple[tuple[int, int, int], ...]:
    """Group fusion data inputs that view one prepared material owner."""
    from ._mac import uncast

    owners: dict[str, tuple[int, int]] = {}
    views = {}
    for number, index in enumerate(data_indices):
        source = uncast(operation.inputs[index]).owner
        attribute = getattr(source, "attributes", {}).get(
            "fhelium.material_view"
        )
        if isinstance(attribute, StringAttr):
            views[number] = json.loads(attribute.data)
        elif (
            isinstance(source, core.MaterialRefOp) and source.symbol is not None
        ):
            views[number] = source.symbol.data
    # An owner material that is itself an input joins its views' allocation.
    named = {view["owner"] for view in views.values() if type(view) is dict}
    result = []
    for number, view in views.items():
        if type(view) is str:
            if view not in named:
                continue
            view = {"owner": view, "offset": 0}
        owner, origin = owners.setdefault(
            view["owner"], (number, view["offset"])
        )
        result.append((number, owner, view["offset"] - origin))
    return tuple(result)


@dataclass
class ExpressionGraph:
    """Represent a region's SSA dependencies and input layouts for code generation."""

    operation: fusion.FusedOp

    @cached_property
    def nodes(self):
        return tuple(
            graph_nodes(self.operation.body.block.ops, preserve_carried=True)
        )

    @cached_property
    def table_arguments(self) -> tuple[SSAValue, ...]:
        arguments = set(self.operation.body.block.args)
        used: set[SSAValue] = set()
        for node in self.nodes:
            operation = node.operation
            if isinstance(operation, ckks.RepeatOp):
                used.add(self._loop_parameter(node))
            elif isinstance(operation, NTT_OPS):
                used.update(node.operands[1:])
            elif isinstance(operation, ARITHMETIC_OPS):
                used.add(node.operands[-1])
            elif isinstance(operation, rns.KeySwitchDigitProductOp):
                used.update(node.operands[1:])
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
        for node in self.nodes:
            if isinstance(node.operation, ckks.RepeatOp):
                return self.table_arguments.index(self._loop_parameter(node))
            if isinstance(
                node.operation, ARITHMETIC_OPS + (rns.KeySwitchDigitProductOp,)
            ):
                return self.table_arguments.index(node.operands[-1])
        raise ValueError("Pointwise fusion has no RNS parameter operand")

    @cached_property
    def key_inputs(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (
                self.table_arguments.index(node.operands[1]),
                int(node.operation.key_digit_index.value.data),
                int(
                    cast(
                        IntegerAttr, node.operation.attributes["key_row_start"]
                    ).value.data
                ),
            )
            for node in self.nodes
            if isinstance(node.operation, rns.KeySwitchDigitProductOp)
        )

    @cached_property
    def round_inputs(self) -> dict[int, int]:
        """Map data inputs read as round-indexed MAC rows to their first row."""
        from ._mac import round_rows

        data = tuple(
            arg
            for arg in self.operation.body.block.args
            if arg not in self.table_arguments
        )
        numbers = {arg: number for number, arg in enumerate(data)}
        starts: dict[int, int] = {}
        for node in self.nodes:
            if not isinstance(node.operation, ckks.RepeatOp):
                continue
            for row in round_rows(node.operation) or ():
                source = node.operands[row.value.index - 1]
                if source not in numbers:
                    raise ValueError(
                        "Round-indexed rows require a batched region input"
                    )
                start = row.start.value.data
                if starts.setdefault(numbers[source], start) != start:
                    raise ValueError(
                        "Round-indexed rows of one input require one first row"
                    )
        selected = {data[number] for number in starts}
        if any(
            not isinstance(node.operation, ckks.RepeatOp)
            and selected.intersection(node.operands)
            for node in self.nodes
        ):
            raise ValueError(
                "A batched input read by round is also read as a whole"
            )
        return starts

    @staticmethod
    def _loop_parameter(node):
        from ._mac import mac_body, uncast

        multiply, _ = mac_body(node.operation)
        parameter = uncast(multiply.operands[-1])
        return node.operands[parameter.index - 1]

    @cached_property
    def view_owners(self) -> tuple[tuple[int, int, int], ...]:
        """Return ``(input, owner input, relative offset)`` for preparation-proven views.

        The first data input that views an owner identifies that owner; offsets are relative to that input's offset.
        """
        return material_view_owners(self.operation, self.input_indices[0])

    def _owner_layouts(self, layouts):
        """Use preparation-proven view owners; runtime data addresses are not evidence."""
        result = list(layouts)
        for number, owner, offset in self.view_owners:
            result[number] = replace(
                layouts[number], owner=owner, offset=offset
            )
        return tuple(result)

    def _carried_value(self, node, values, ordinal, steps):
        loop = node.operation
        count, shared, width = (
            loop.count.value.data,
            loop.shared_count.value.data,
            loop.table_width.value.data,
        )
        local = {
            arg: values[v]
            for arg, v in zip(
                tuple(loop.body.block.args)[1 : 1 + shared],
                node.operands[:shared],
                strict=True,
            )
            if v in values
        }
        for column in range(width):
            rows = tuple(
                values[node.operands[shared + i * width + column]]
                for i in range(count)
            )
            first = rows[0]
            local[loop.body.block.args[1 + shared + column]] = replace(
                first,
                components=tuple(
                    Expr("select", tuple(row.components[c] for row in rows))
                    for c in range(len(first.components))
                ),
            )
        initial = values[node.operands[-1]]
        local[loop.body.block.args[-1]] = replace(
            initial,
            components=tuple(
                Expr("carry", data=(c,)) for c in range(len(initial.components))
            ),
        )
        for operation in tuple(loop.body.block.ops)[:-1]:
            if isinstance(operation, UnrealizedConversionCastOp):
                local[operation.results[0]] = local[operation.operands[0]]
                continue
            if isinstance(operation, ckks.SelectBatchOp):
                source = local[operation.value]
                local[operation.result] = replace(
                    source,
                    components=tuple(
                        Expr(
                            "indexed_input",
                            data=(*expr.data, steps[expr.data[0]]),
                        )
                        for expr in source.components
                    ),
                )
                continue
            operands = [local[v] for v in operation.operands[:-1]]
            if isinstance(operation, ckks.MultiplyCompressedPlaintextOp):
                result = compact_value(
                    type(operation),
                    operands,
                    ordinal,
                    layout=operation.attributes["compression_layout"].data,
                    ntt_plaintext=True,
                )
            else:
                result = arithmetic_value(type(operation), operands, ordinal)
            local[operation.results[0]] = result
        final = local[loop.body.block.last_op.operands[-1]]
        return replace(
            final,
            root=ordinal,
            components=tuple(
                Expr("loop", (start, body), (count, c))
                for c, (start, body) in enumerate(
                    zip(initial.components, final.components, strict=True)
                )
            ),
        )

    def bind(self, layouts: tuple[InputLayout, ...]) -> BoundGraph:
        layouts = self._owner_layouts(layouts)
        # A round-indexed input binds as its first selected row; rounds
        # advance by the stride of the removed leading batch axis.
        steps = {}
        if self.round_inputs:
            rows = list(layouts)
            for number, start in self.round_inputs.items():
                layout = rows[number]
                axis = int(layout.component_axis)
                steps[number] = layout.strides[axis]
                rows[number] = replace(
                    layout,
                    shape=layout.shape[:axis] + layout.shape[axis + 1 :],
                    strides=layout.strides[:axis] + layout.strides[axis + 1 :],
                    offset=layout.offset + start * steps[number],
                )
            layouts = tuple(rows)
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
        for ordinal, node in enumerate(self.nodes):
            operation = node.operation
            if isinstance(operation, fusion.YieldOp):
                outputs = tuple(values[v] for v in node.operands)
                continue
            if isinstance(operation, ckks.RepeatOp):
                result = self._carried_value(node, values, ordinal, steps)
                values[node.results[0]] = result
                roots[result.root] = result
                continue
            if isinstance(operation, UnrealizedConversionCastOp):
                values[node.results[0]] = values[node.operands[0]]
                continue
            operands = [
                values[v]
                for v in node.operands
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
                values[node.results[0]] = ValueExpr(
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
                            for v in node.operands[1:]
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
            values[node.results[0]] = result
            roots[result.root] = result
        return BoundGraph(
            layouts,
            outputs,
            roots,
            tuple(anchors),
            repeat_groups(self.operation),
        )


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


class KernelArguments:
    """Name the data pointers and layout scalars of one generated kernel.

    Inputs are named ``x{k}`` in first-use order, so a kernel receives only
    the Tensors it reads and its source does not depend on the position of
    those Tensors within the fusion region. Layout scalars (strides, row
    counts, lengths) are declared as int64
    arguments and recorded in ``scalars`` in first-use order; the caller
    passes ``scalar_values`` in that order after the pointer arguments.
    Inputs sharing one stride tuple share one stride argument group.
    """

    def __init__(self, layouts: tuple[InputLayout, ...]):
        self.layouts = layouts
        self.inputs: list[int] = []
        self._groups: dict[tuple[int, ...], int] = {}
        self.scalars: list[tuple[str, int, str]] = []
        self._named: dict[str, int] = {}
        self._offsets: dict[int, int] = {}

    def pointer(self, number: int) -> str:
        owner = self.layouts[number].owner
        if owner is not None:
            number = owner
        if number not in self.inputs:
            self.inputs.append(number)
        return f"x{self.inputs.index(number)}"

    def scalar(self, name: str, value: int, dtype: str = "tl.int64") -> str:
        recorded = self._named.get(name)
        if recorded is None:
            self._named[name] = value
            self.scalars.append((name, value, dtype))
        elif recorded != value:
            raise ValueError(f"Kernel scalar {name} was bound to two values")
        return name

    def offset(self, number: int) -> str:
        # First-use numbering keeps names independent of region input order.
        index = self._offsets.setdefault(number, len(self._offsets))
        return self.scalar(f"view_offset{index}", self.layouts[number].offset)

    def stride(self, number: int, axis: int) -> str:
        layout = self.layouts[number]
        group = self._groups.setdefault(layout.strides, len(self._groups))
        return self.scalar(f"ld{group}_{axis}", layout.strides[axis])

    @property
    def pointer_parameters(self) -> list[str]:
        return [f"x{k}" for k in range(len(self.inputs))]

    @property
    def scalar_parameters(self) -> list[str]:
        return [f"{name}: {dtype}" for name, _, dtype in self.scalars]

    @property
    def scalar_values(self) -> tuple[int, ...]:
        return tuple(value for _, value, _ in self.scalars)


@dataclass
class Emitter:
    """Emit an expression at arbitrary coefficient, prime-row and batch indices."""

    layouts: tuple[InputLayout, ...]
    arguments: KernelArguments
    lines: list[str] = field(default_factory=list)
    cache: dict[tuple[Expr, str, str, str, str], str] = field(
        default_factory=dict
    )

    prefix: str = ""
    iteration: str | None = None
    pointers: dict[int, str] = field(default_factory=dict)

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
        if expr.op == "loop":
            count, component = expr.data
            number = len(self.cache)
            accumulator = f"{self.prefix}acc{number}"
            iteration = f"{self.prefix}j{number}"
            initial = self.emit(
                expr.args[0], index, row, batch, overrides, mask
            )
            limit = self.arguments.scalar(
                f"mac_count{len(self.arguments.scalars)}", count
            )
            child = Emitter(
                self.layouts,
                self.arguments,
                prefix=f"{self.prefix}mac{number}_",
                iteration=iteration,
            )
            substitutions = dict(overrides or {})
            substitutions[Expr("carry", data=(component,))] = accumulator
            result = child.emit(
                expr.args[1], index, row, batch, substitutions, mask
            )
            self.lines.extend(
                (
                    f"{accumulator} = {initial}",
                    f"for {iteration} in tl.range(0, {limit}):",
                )
            )
            self.lines.extend(
                "    " + line
                for line in (*child.lines, f"{accumulator} = {result}")
            )
            self.cache[key] = accumulator
            return accumulator
        if expr.op == "select":
            result = self.select(expr.args, index, row, batch, overrides, mask)
            self.cache[key] = result
            return result
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
                result = f"{self.prefix}e{len(self.cache)}"
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
        variable = f"{self.prefix}e{len(self.cache)}"
        if expr.op in {"input", "indexed_input"}:
            number, component = expr.data[:2]
            layout = self.layouts[number]
            names = self.arguments
            shape = layout.batch_shape
            rank = len(layout.strides)
            first_batch = 1 if layout.component_axis else 0
            offset = [
                f"({index}) * {names.stride(number, rank - 1)}",
                f"({row}) * {names.stride(number, rank - 2)}",
            ]
            if layout.component_axis:
                offset.append(f"{component} * {names.stride(number, 0)}")
            for axis, extent in enumerate(shape):
                if extent > 1:
                    offset.append(
                        f"((({batch}) // {prod(shape[axis + 1 :])}) % {extent}) * {names.stride(number, first_batch + axis)}"
                    )
            if layout.offset:
                offset.append(names.offset(number))
            if expr.op == "indexed_input":
                step = names.scalar(
                    f"mac_stride{len(names.scalars)}", expr.data[2]
                )
                offset.append(f"{self.iteration} * {step}")
            base_pointer = self.pointers.get(number) or names.pointer(number)
            pointer = f"{base_pointer} + {' + '.join(offset)}"
            # Uniform implicit values have scalar addresses in NTT endpoints.
            # Broadcast the pointer and mask together before the masked load.
            expression = f"tl.load(tl.broadcast({pointer}, {mask})[0], mask={mask}, other=0).to(tl.int64)"
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

    def select(self, expressions, index, row, batch, overrides, mask):
        first = expressions[0]
        if all(expr == first for expr in expressions):
            return self.emit(first, index, row, batch, overrides, mask)
        if all(expr.op == "input" for expr in expressions):
            layouts = tuple(self.layouts[expr.data[0]] for expr in expressions)
            base = layouts[0]
            step = layouts[1].offset - base.offset
            if base.owner is not None and all(
                layout.owner == base.owner
                and layout.shape == base.shape
                and layout.strides == base.strides
                and expr.data[1] == first.data[1]
                and layout.offset == base.offset + i * step
                for i, (expr, layout) in enumerate(
                    zip(expressions, layouts, strict=True)
                )
            ):
                return self.emit(
                    Expr("indexed_input", data=(*first.data, step)),
                    index,
                    row,
                    batch,
                    overrides,
                    mask,
                )
            if all(
                layout.shape == base.shape
                and layout.strides == base.strides
                and layout.offset == base.offset
                and expr.data[1] == first.data[1]
                for expr, layout in zip(expressions, layouts, strict=True)
            ):
                pointer = f"{self.prefix}pointer{len(self.cache)}"

                def select_pointer(start, end, indent=""):
                    if end - start == 1:
                        self.lines.append(
                            f"{indent}{pointer} = {self.arguments.pointer(expressions[start].data[0])}"
                        )
                        return
                    middle = (start + end) // 2
                    self.lines.append(
                        f"{indent}if {self.iteration} < {middle}:"
                    )
                    select_pointer(start, middle, indent + "    ")
                    self.lines.append(f"{indent}else:")
                    select_pointer(middle, end, indent + "    ")

                select_pointer(0, len(expressions))
                child = Emitter(
                    self.layouts,
                    self.arguments,
                    prefix=pointer + "_",
                    iteration=self.iteration,
                    pointers={first.data[0]: pointer},
                )
                result = child.emit(first, index, row, batch, overrides, mask)
                self.lines.extend(child.lines)
                return result
        result = f"{self.prefix}selected{len(self.cache)}"
        # Each branch reads the existing bank directly. No packed ciphertext copy
        # or device-side pointer table is introduced.
        for i, expression in enumerate(expressions):
            self.lines.append(
                (
                    f"if {self.iteration} == 0:"
                    if i == 0
                    else "else:"
                    if i == len(expressions) - 1
                    else f"elif {self.iteration} == {i}:"
                )
            )
            child = Emitter(
                self.layouts,
                self.arguments,
                prefix=f"{result}_{i}_",
                iteration=self.iteration,
            )
            value = child.emit(expression, index, row, batch, overrides, mask)
            self.lines.extend(
                "    " + line for line in (*child.lines, f"{result} = {value}")
            )
        return result
