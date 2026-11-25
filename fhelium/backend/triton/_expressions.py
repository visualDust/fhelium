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

from fhelium.ir._operation_structure import (
    ComponentBody,
    OperationStructure,
)
from fhelium.ir.dialects import ckks, core, fusion

from ._layout import generated_structure, result_layout


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Expr, ...] = ()
    data: tuple[int, ...] = ()


@dataclass(frozen=True)
class InputLayout:
    r"""Describe logical indexing and a preparation-proven pointer owner.

    ``owner`` indexes a region input sharing the allocation; ``offset`` is the
    element displacement from that input's pointer to this logical view.
    ``row_split`` $(q, g)$ addresses logical row $r\ge q$ at physical row
    $r+g$; a key digit whose active Q and P rows are stored as two slices uses
    it with $q$ active Q rows and $g$ unused rows between the slices.
    """

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    component_axis: bool = False
    owner: int | None = None
    offset: int = 0
    row_split: tuple[int, int] | None = None

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

    @cached_property
    def seed_inputs(self) -> frozenset[int]:
        """Input numbers read as int64 seed rows by seeded key residues."""
        found: set[int] = set()
        pending = [
            e
            for v in (*self.roots.values(), *(a.source for a in self.anchors))
            for e in v.components
        ]
        seen: set[int] = set()
        while pending:
            expr = pending.pop()
            if id(expr) in seen:
                continue
            seen.add(id(expr))
            if expr.op == "seeded":
                found.add(expr.data[0])
            pending.extend(expr.args)
        return frozenset(found)


_COMPONENT_AXES: dict[int, tuple[object, bool]] = {}


def component_axis(value_type: object) -> bool:
    """Report whether storage of ``value_type`` has a leading component axis; cached by type identity."""
    entry = _COMPONENT_AXES.get(id(value_type))
    if entry is not None and entry[0] is value_type:
        return entry[1]
    if isinstance(value_type, ckks.CiphertextType):
        result = True
    else:
        state = getattr(getattr(value_type, "state", None), "data", {})
        count = state.get("component_count", state.get("components"))
        result = isinstance(count, IntegerAttr)
    _COMPONENT_AXES[id(value_type)] = (value_type, result)
    return result


def _uncast(value: SSAValue) -> SSAValue:
    while type(value.owner) is UnrealizedConversionCastOp:
        op = value.owner
        if len(op.inputs) != 1 or len(op.outputs) != 1:
            break
        value = op.inputs[0]
    return value


def material_view_owners(
    operation: fusion.FusedOp, data_indices: tuple[int, ...]
) -> tuple[tuple[int, int, int], ...]:
    """Group fusion data inputs that view one prepared material owner."""

    owners: dict[str, tuple[int, int]] = {}
    views = {}
    for number, index in enumerate(data_indices):
        source = _uncast(operation.inputs[index]).owner
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
    def nodes(self) -> tuple[Operation, ...]:
        return tuple(self.operation.body.block.ops)

    @cached_property
    def table_arguments(self) -> tuple[SSAValue, ...]:
        arguments = set(self.operation.body.block.args)
        used: set[SSAValue] = set()
        for node in self.nodes:
            structure = generated_structure(node)
            if structure is not None:
                used.update(
                    node.operands[i]
                    for i in structure.operands(
                        "parameters", "table", "key", "seed"
                    )
                )
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
            structure = generated_structure(node)
            if structure is not None and structure.layout != "transform":
                for index in structure.operands("parameters"):
                    return self.table_arguments.index(node.operands[index])
        raise ValueError("Pointwise fusion has no RNS parameter operand")

    @cached_property
    def key_inputs(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return ``(table argument, stored key digit, key_row_start, key_p_row_start, seed table argument, rounds)`` of each key-digit product.

        Dense keys have seed table argument $-1$ and zero rounds.
        """
        inputs = []
        for node in self.nodes:
            structure = generated_structure(node)
            if structure is None:
                continue
            seeds = structure.operands("seed")
            for index in structure.operands("key"):
                inputs.append(
                    (
                        self.table_arguments.index(node.operands[index]),
                        _integer_attribute(node, "key_digit_index"),
                        _integer_attribute(node, "key_row_start"),
                        _integer_attribute(node, "key_p_row_start"),
                        self.table_arguments.index(node.operands[seeds[0]])
                        if seeds
                        else -1,
                        _integer_attribute(node, "key_rounds") if seeds else 0,
                    )
                )
        return tuple(inputs)

    @property
    def seeded_key_count(self) -> int:
        return sum(entry[4] >= 0 for entry in self.key_inputs)

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

    @cached_property
    def stacked_inputs(self) -> tuple[int, ...]:
        """Return the region inputs whose leading axis enumerates reduction terms."""
        arguments: list[SSAValue] = [
            arg
            for arg in self.operation.body.block.args
            if arg not in self.table_arguments
        ]
        stacked = []
        for node in self.nodes:
            structure = generated_structure(node)
            reduction = structure.reduction if structure is not None else None
            if reduction is None or reduction.stacked is None:
                continue
            value = _uncast(node.operands[reduction.stacked])
            if value not in arguments:
                raise ValueError(
                    "A stacked reduction operand must be a fusion region input"
                )
            stacked.append(arguments.index(value))
        return tuple(stacked)

    def bind(self, layouts: tuple[InputLayout, ...]) -> BoundGraph:
        layouts = self._owner_layouts(layouts)
        # A stacked input binds as its first term row; iterations advance by
        # the stride of the removed leading axis.
        steps = {}
        if self.stacked_inputs:
            rows = list(layouts)
            for number in self.stacked_inputs:
                layout = rows[number]
                axis = int(layout.component_axis)
                steps[number] = layout.strides[axis]
                rows[number] = replace(
                    layout,
                    shape=layout.shape[:axis] + layout.shape[axis + 1 :],
                    strides=layout.strides[:axis] + layout.strides[axis + 1 :],
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
        if input_index + len(self.key_inputs) + self.seeded_key_count != len(
            layouts
        ):
            raise ValueError(
                "Fusion Tensor inputs do not match the region arguments"
            )
        next_key_input = input_index
        # Seed rows follow all key views, in key order.
        next_seed_input = input_index + len(self.key_inputs)
        key_entries = iter(self.key_inputs)
        outputs: tuple[ValueExpr, ...] = ()
        for ordinal, operation in enumerate(self.nodes):
            node = operation
            if isinstance(operation, fusion.YieldOp):
                outputs = tuple(values[v] for v in node.operands)
                continue
            if isinstance(operation, UnrealizedConversionCastOp):
                values[node.results[0]] = values[node.operands[0]]
                continue
            structure = generated_structure(operation)
            if structure is None or not structure.accepted:
                raise ValueError(
                    f"Selected Triton expression code generation cannot implement {operation.name}"
                )
            data = {
                index: values[node.operands[index]]
                for index in structure.operands("data", "compact")
            }
            operands = list(data.values())
            first = operands[0]
            if structure.layout == "select":
                index = cast(int, structure.component)
                result_layout(structure, [first.shape], [first.component_axis])
                values[node.results[0]] = ValueExpr(
                    (first.components[index],),
                    first.shape[1:],
                    False,
                    first.root,
                    (*first.view, index),
                )
                continue
            if structure.layout == "transform":
                result = ValueExpr(
                    tuple(
                        Expr("anchor", (expr,), (len(anchors), c))
                        for c, expr in enumerate(first.components)
                    ),
                    first.shape,
                    first.component_axis,
                    ordinal,
                )
                anchors.append(
                    Anchor(
                        operation,
                        first,
                        result,
                        tuple(
                            self.table_arguments.index(node.operands[i])
                            for i in structure.operands("parameters", "table")
                        ),
                    )
                )
                values[node.results[0]] = result
                roots[result.root] = result
                continue
            keys: dict[int, tuple[Expr, ...]] = {}
            for index in structure.operands("key"):
                key_layout = layouts[next_key_input]
                entry = next(key_entries)
                seeded = entry[4] >= 0
                if key_layout.shape != (1 if seeded else 2, *first.shape[-2:]):
                    raise ValueError(
                        "Key-switch key rows or polynomial extent differ from the active digit"
                    )
                if seeded:
                    # k1 at view row r is generated at stored row
                    # key_row_start + r, after any row split.
                    keys[index] = (
                        Expr("input", data=(next_key_input, 0)),
                        Expr(
                            "seeded",
                            data=(
                                next_seed_input,
                                next_key_input,
                                entry[2],
                                entry[5],
                            ),
                        ),
                    )
                    next_seed_input += 1
                else:
                    keys[index] = tuple(
                        Expr("input", data=(next_key_input, c))
                        for c in range(2)
                    )
                next_key_input += 1
            result = structured_value(
                structure, data, ordinal, type(operation), keys, steps
            )
            values[node.results[0]] = result
            roots[result.root] = result
        return BoundGraph(
            layouts,
            outputs,
            roots,
            tuple(anchors),
        )


def structured_value(
    structure: OperationStructure,
    data: Mapping[int, ValueExpr],
    root: int,
    operation_type: type[Operation],
    keys: Mapping[int, tuple[Expr, ...]] | None = None,
    steps: Mapping[int, int] | None = None,
) -> ValueExpr:
    """Bind a registered component body to the operands at its ``data``/``compact`` positions.

    A declared reduction whose terms share one layout becomes a runtime loop: the first term initializes the accumulator and each later iteration reads its term operands through ``select`` expressions indexed by the loop variable. A stacked operand, bound at its first row, is read in iteration $j$ at row $j+1$ through the term stride in ``steps``.
    """
    if not structure.accepted:
        raise ValueError(
            f"Selected Triton expression code generation cannot implement {operation_type.name}"
        )
    reduction = structure.reduction
    if reduction is not None and reduction.stacked is not None:
        stacked = data[reduction.stacked]

        def advance(expr: Expr) -> Expr:
            if expr.op == "input":
                number = expr.data[0]
                return Expr(
                    "indexed_input",
                    data=(*expr.data, cast(Mapping, steps)[number], 1),
                )
            return replace(expr, args=tuple(advance(a) for a in expr.args))

        initial = structured_value(
            reduction.term,
            {0: data[reduction.terms[0][0]], 1: stacked},
            root,
            operation_type,
        )
        if len(reduction.terms) == 1:
            return initial
        ciphertexts = [data[t[0]] for t in reduction.terms[1:]]
        selected = {
            0: replace(
                ciphertexts[0],
                components=tuple(
                    Expr("select", tuple(v.components[c] for v in ciphertexts))
                    for c in range(len(ciphertexts[0].components))
                ),
            ),
            1: replace(
                stacked,
                components=tuple(advance(e) for e in stacked.components),
            ),
        }
        term = structured_value(reduction.term, selected, root, operation_type)
        return replace(
            initial,
            components=tuple(
                Expr(
                    "loop",
                    (start, Expr("add", (Expr("carry", data=(c,)), body))),
                    (len(reduction.terms) - 1, c),
                )
                for c, (start, body) in enumerate(
                    zip(initial.components, term.components, strict=True)
                )
            ),
        )
    if reduction is not None and len(reduction.terms) > 1:
        columns = tuple(zip(*reduction.terms, strict=True))
        if all(
            len({(data[i].shape, data[i].component_axis) for i in column}) == 1
            for column in columns
        ):
            initial = structured_value(
                reduction.term,
                {k: data[i] for k, i in enumerate(reduction.terms[0])},
                root,
                operation_type,
            )
            selected = {
                k: replace(
                    data[column[1]],
                    components=tuple(
                        Expr(
                            "select",
                            tuple(data[i].components[c] for i in column[1:]),
                        )
                        for c in range(len(data[column[1]].components))
                    ),
                )
                for k, column in enumerate(columns)
            }
            term = structured_value(
                reduction.term, selected, root, operation_type
            )
            return replace(
                initial,
                components=tuple(
                    Expr(
                        "loop",
                        (start, Expr("add", (Expr("carry", data=(c,)), body))),
                        (len(reduction.terms) - 1, c),
                    )
                    for c, (start, body) in enumerate(
                        zip(initial.components, term.components, strict=True)
                    )
                ),
            )
    operands = list(data.values())
    first = operands[0]
    result_layout(
        structure,
        [v.shape for v in operands],
        [v.component_axis for v in operands],
    )
    components: dict[int, tuple[Expr, ...]] = {
        index: value.components for index, value in data.items()
    }
    components.update(keys or {})
    counts = tuple(
        len(components.get(index, ())) for index in range(len(structure.roles))
    )

    def lower(item) -> Expr:
        if item.op == "load":
            operand, component = item.data
            return components[operand][component]
        args = tuple(lower(arg) for arg in item.args)
        if item.op == "compact":
            unique = data[item.args[0].data[0]].shape[-1]
            return Expr(
                "compact",
                args,
                (item.data[0], unique, first.shape[-1] // unique),
            )
        return Expr(item.op, args)

    lowered = tuple(
        lower(item) for item in cast(ComponentBody, structure.body)(counts)
    )
    if structure.layout == "pack":
        shape, axis = (len(operands), *first.shape), True
    elif structure.layout == "key_product":
        shape, axis = (2, *first.shape), True
    elif structure.layout == "component_product":
        shape, axis = (3, *first.shape[1:]), True
    else:
        shape, axis = first.shape, first.component_axis
    return ValueExpr(lowered, shape, axis, root)


class KernelArguments:
    """Name the data pointers and layout scalars of one generated kernel.

    Inputs are named ``x{k}`` in first-use order, so a kernel receives only
    the Tensors it reads and its source does not depend on the position of
    those Tensors within the fusion region. Layout scalars (strides, row
    counts, lengths) are declared as int64
    arguments and recorded in ``scalars`` in first-use order; the caller
    passes ``scalar_values`` in that order after the pointer arguments.
    Each input has its own stride argument group, so the source does not
    depend on whether two inputs happen to have equal strides.
    """

    def __init__(self, layouts: tuple[InputLayout, ...]):
        self.layouts = layouts
        self.inputs: list[int] = []
        self._groups: dict[int, int] = {}
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

    def row_split(self, number: int) -> tuple[str, str]:
        group = self._groups.setdefault(number, len(self._groups))
        split, gap = cast(tuple[int, int], self.layouts[number].row_split)
        return (
            self.scalar(f"row_split{group}", split),
            self.scalar(f"row_gap{group}", gap),
        )

    def stride(self, number: int, axis: int) -> str:
        layout = self.layouts[number]
        group = self._groups.setdefault(number, len(self._groups))
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
    # Elements 8k..8k+7 of the driver's ``index`` vector hold coefficients
    # 8m..8m+7 of one row, so seeded keys sample each segment once.
    segment_index: bool = False

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
                segment_index=self.segment_index,
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
        if expr.op == "seeded":
            seed, key_input, start, rounds = expr.data
            names = self.arguments
            physical = f"({row})"
            if self.layouts[key_input].row_split is not None:
                split, gap = names.row_split(key_input)
                physical = (
                    f"(({row}) + (({row}) >= {split}).to(tl.int64) * {gap})"
                )
            group = names._groups.setdefault(key_input, len(names._groups))
            first_row = names.scalar(f"key_row_start{group}", start)
            expression = (
                f"seeded_key_residue{8 if self.segment_index and index == 'index' else ''}("
                f"{names.pointer(seed)}, {first_row} + {physical}, "
                f"{index}, q, {rounds}, RADIX)"
            )
        elif expr.op in {"input", "indexed_input"}:
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
            if layout.row_split is not None:
                split, gap = names.row_split(number)
                offset[1] = (
                    f"(({row}) + (({row}) >= {split}).to(tl.int64) * {gap})"
                    f" * {names.stride(number, rank - 2)}"
                )
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
                start = expr.data[3] if len(expr.data) > 3 else 0
                iteration = (
                    f"({self.iteration} + {start})" if start else self.iteration
                )
                offset.append(f"{iteration} * {step}")
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
                and layout.row_split == base.row_split
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
                and layout.row_split == base.row_split
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
                    segment_index=self.segment_index,
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
                segment_index=self.segment_index,
            )
            value = child.emit(expression, index, row, batch, overrides, mask)
            self.lines.extend(
                "    " + line for line in (*child.lines, f"{result} = {value}")
            )
        return result


def _integer_attribute(operation: Operation, name: str) -> int:
    return int(cast(IntegerAttr, operation.attributes[name]).value.data)
