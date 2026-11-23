"""Fold captured arithmetic repetitions and restore their original SSA placement.

Repeat regions describe one ordered mathematical body. Their operand tables retain current SSA materials. Supported independent bodies remain represented through fusion; fallback expansion restores their recorded placement without reassociation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.ir import Block, BlockArgument, Operation
from xdsl.rewriter import Rewriter

from fhelium.ir.dialects import ckks
from ._origins import OperationOrigins, get_origin
from ._pipeline import PassResult, PassStats

_ARITHMETIC = (
    ckks.AddOp,
    ckks.SubtractOp,
    ckks.NegateOp,
    ckks.MultiplyPlaintextOp,
    ckks.MultiplyCompressedPlaintextOp,
)
_MULTIPLY = (ckks.MultiplyPlaintextOp, ckks.MultiplyCompressedPlaintextOp)


def _site(operation, workspace):
    origin = get_origin(operation, workspace)
    while origin and origin.get("source") != "capture":
        origin = origin.get("parent")
    site = None if origin is None else origin.get("site")
    if site is None:
        return None
    return (
        site.get("code"),
        site.get("module"),
        site.get("qualname"),
        site.get("firstline"),
        site.get("offset"),
    )


def _shape(operations):
    local = {
        value: (index, result)
        for index, operation in enumerate(operations)
        for result, value in enumerate(operation.results)
    }
    ports = {}
    shape = []
    for operation in operations:
        inputs = []
        for value in operation.operands:
            if value in local:
                inputs.append(("local", *local[value]))
            else:
                inputs.append(("port", ports.setdefault(value, len(ports))))
        shape.append(
            (
                type(operation),
                tuple(sorted(operation.attributes.items())),
                tuple(sorted(operation.properties.items())),
                tuple(value.type for value in operation.results),
                tuple(inputs),
            )
        )
    return tuple(shape), tuple(ports)


def _dominates(value, block, anchor, positions):
    if isinstance(value, BlockArgument):
        return value.owner is block
    owner = value.owner
    return (
        isinstance(owner, Operation)
        and owner.parent_block() is block
        and positions[owner] < positions[anchor]
    )


def _fold_group(groups, compilation, observed, replacements, positions):
    if len(groups) < 2 or not any(
        isinstance(op, _MULTIPLY) for op in groups[0]
    ):
        return False
    first = groups[0]
    block = first[0].parent_block()
    if block is None:
        return False
    signature, ports = _shape(first)
    rows = [ports]
    for group in groups[1:]:
        other, operands = _shape(group)
        if other != signature or tuple(
            value.type for value in operands
        ) != tuple(value.type for value in ports):
            return False
        rows.append(operands)
    all_operations = {operation for group in groups for operation in group}
    selected = set()
    for group in groups:
        for i, operation in enumerate(group):
            for j, value in enumerate(operation.results):
                if value in observed or any(
                    use.operation not in all_operations for use in value.uses
                ):
                    selected.add((i, j))
    shared, varying, carried = [], [], []
    port_arguments = {}
    previous_values = {
        value: (i, j)
        for i, operation in enumerate(first)
        for j, value in enumerate(operation.results)
    }
    for column, initial in enumerate(ports):
        column_values = tuple(row[column] for row in rows)
        if all(value is initial for value in column_values):
            if not _dominates(initial, block, first[0], positions):
                return False
            port_arguments[column] = ("shared", len(shared))
            shared.append(initial)
            continue
        carry_position = None
        if column_values[1] in previous_values:
            position = previous_values[column_values[1]]
            if all(
                column_values[round_]
                is groups[round_ - 1][position[0]].results[position[1]]
                for round_ in range(1, len(groups))
            ):
                carry_position = position
        if carry_position is not None:
            if (
                not _dominates(initial, block, first[0], positions)
                or first[carry_position[0]].results[carry_position[1]].type
                != initial.type
            ):
                return False
            port_arguments[column] = ("carry", len(carried))
            carried.append((initial, carry_position))
            selected.add(carry_position)
        else:
            if any(
                not _dominates(value, block, first[0], positions)
                for value in column_values
            ):
                return False
            port_arguments[column] = ("table", len(varying))
            varying.append(column_values)
    if not selected:
        return False
    # Cross-round CSE must not depend on identities hidden by the table arguments.
    # Repeated table entries and invariant arithmetic keep the ordinary expanded path.
    if any(len(set(column)) != len(column) for column in varying):
        return False
    dependent = {
        ports[column]
        for column, (kind, _) in port_arguments.items()
        if kind != "shared"
    }
    for operation in first:
        if not any(value in dependent for value in operation.operands):
            return False
        dependent.update(operation.results)

    # Equal alias partitions are part of the input fingerprint, not inferred from table lengths.
    def partition(values):
        first_indices = {}
        return tuple(
            first_indices.setdefault(value, index)
            for index, value in enumerate(values)
        )

    aliases = partition(ports)
    for row in rows:
        if partition(row) != aliases:
            return False
    outputs = tuple(sorted(selected))
    body = Block(
        arg_types=(
            IndexType(),
            *(value.type for value in shared),
            *(column[0].type for column in varying),
            *(value.type for value, _ in carried),
        )
    )
    count = len(groups)
    tables = tuple(
        column[round_] for round_ in range(count) for column in varying
    )
    result_types = tuple(
        group[i].results[j].type for group in groups for i, j in outputs
    ) + tuple(value.type for value, _ in carried)
    loop = ckks.RepeatOp(
        shared,
        tables,
        tuple(value for value, _ in carried),
        result_types,
        body,
        count=count,
        yield_count=len(outputs),
    )
    block.insert_op_before(loop, first[0])
    positions[loop] = positions[first[0]] - 2
    body_ports = {}
    for column, (kind, index) in port_arguments.items():
        offset = (
            1
            if kind == "shared"
            else 1 + len(shared)
            if kind == "table"
            else 1 + len(shared) + len(varying)
        )
        body_ports[ports[column]] = body.args[offset + index]
    for round_, group in enumerate(groups):
        inputs = tuple(
            loop.outputs[round_ * len(outputs) : (round_ + 1) * len(outputs)]
        )
        marker = ckks.RepeatIterationOp(inputs, iteration=round_)
        block.insert_op_before(marker, group[0])
        positions[marker] = positions[group[0]] - 1
        for position, output in zip(outputs, marker.outputs, strict=True):
            value = group[position[0]].results[position[1]]
            replacements.setdefault(value, output)
            for use in tuple(value.uses):
                if use.operation not in all_operations:
                    use.operation.operands[use.index] = output
    # The original first body retains operation provenance and option identity.
    for operation in first:
        block.detach_op(operation)
        operation.operands = tuple(
            body_ports.get(value, value) for value in operation.operands
        )
        body.add_op(operation)
    body.add_op(
        ckks.RepeatYieldOp(
            tuple(first[i].results[j] for i, j in outputs)
            + tuple(first[i].results[j] for _, (i, j) in carried)
        )
    )
    for group in reversed(groups[1:]):
        for operation in reversed(group):
            Rewriter.erase_op(operation, safe_erase=False)
    return True


def fold_repeats(compilation, *, observed=()):
    """Locate source-site candidates, then prove their current SSA correspondence."""
    if compilation.workspace.get("ckks_repeat") is False:
        return {}
    observed = set(observed)
    replacements = {}
    count = 0

    def visit(block):
        nonlocal count
        # Reserve two positions before each original operation for its loop
        # and iteration marker; later groups can depend on those new values.
        positions = {op: 3 * index for index, op in enumerate(block.ops)}
        runs, current = [], []
        for operation in tuple(block.ops):
            eligible = (
                type(operation) in _ARITHMETIC
                and _site(operation, compilation.workspace) is not None
                and not (
                    isinstance(operation.attributes.get("inplace"), IntegerAttr)
                    and operation.attributes["inplace"].value.data
                )
            )
            if eligible:
                current.append(operation)
            else:
                if current:
                    runs.append(tuple(current))
                    current = []
        if current:
            runs.append(tuple(current))
        buckets = defaultdict(list)
        for run in runs:
            token = tuple(
                (type(operation), _site(operation, compilation.workspace))
                for operation in run
            )
            buckets[token].append(run)
        for groups in buckets.values():
            if len(groups) > 1 and _fold_group(
                groups, compilation, observed, replacements, positions
            ):
                count += 1
        # A contiguous periodic suffix can carry an accumulator in its original order.
        for run in runs:
            remaining = [
                operation
                for operation in run
                if operation.parent_block() is block
            ]
            if not remaining:
                continue
            tokens = [
                (type(operation), _site(operation, compilation.workspace))
                for operation in remaining
            ]
            best = None
            for start, token in enumerate(tokens):
                if best is not None and len(tokens) - start <= best[0]:
                    break
                for end in range(start + 1, len(tokens)):
                    if tokens[end] != token:
                        continue
                    width = end - start
                    rounds = 1
                    while (
                        start + (rounds + 1) * width <= len(tokens)
                        and tokens[
                            start + rounds * width : start
                            + (rounds + 1) * width
                        ]
                        == tokens[start:end]
                    ):
                        rounds += 1
                    if rounds >= 2 and (
                        best is None or rounds * width > best[0]
                    ):
                        best = (rounds * width, start, width, rounds)
                    break
            if best is not None:
                _, start, width, rounds = best
                groups = [
                    tuple(
                        remaining[start + i * width : start + (i + 1) * width]
                    )
                    for i in range(rounds)
                ]
                if _fold_group(
                    groups, compilation, observed, replacements, positions
                ):
                    count += 1
        for operation in tuple(block.ops):
            if isinstance(operation, ckks.RepeatOp):
                visit(operation.body.block)

    for function in compilation.program.functions:
        visit(function.body.block)
    compilation.workspace["ckks_repeat_folded"] = count
    return replacements


def repeat_bindings(loop):
    return (*loop.shared, *loop.row(0), *loop.carried)


def sync_repeat_inputs(loop):
    """Bind one body only when every round has the same represented state."""
    width, count = loop.table_width.value.data, loop.count.value.data
    start = loop.shared_count.value.data
    table = tuple(
        value.type for value in loop.inputs[start : start + width * count]
    )
    first = table[:width]
    for offset in range(width, len(table), width):
        for column, item in enumerate(table[offset : offset + width]):
            if item is not first[column] and item != first[column]:
                return False
    for argument, value in zip(
        tuple(loop.body.block.args)[1:], repeat_bindings(loop), strict=True
    ):
        if argument.type != value.type:
            Rewriter.replace_value_with_new_type(argument, value.type)
    return True


def sync_repeat_results(loop):
    terminator = loop.body.block.last_op
    assert isinstance(terminator, ckks.RepeatYieldOp)
    width = loop.yield_count.value.data
    carried = tuple(terminator.values[width:])
    if tuple(value.type for value in carried) != tuple(
        value.type for value in loop.carried
    ):
        return False
    types = tuple(
        value.type for value in terminator.values[:width]
    ) * loop.count.value.data + tuple(value.type for value in carried)
    for result, typ in zip(tuple(loop.outputs), types, strict=True):
        if result.type != typ:
            Rewriter.replace_value_with_new_type(result, typ)
    return True


def sync_iteration(operation):
    for result, value in zip(
        tuple(operation.outputs), operation.inputs, strict=True
    ):
        if result.type != value.type:
            Rewriter.replace_value_with_new_type(result, value.type)


def expand_repeats(
    compilation, *, operation_sources=None, replacements=None, roots=None
):
    """Clone each round at its recorded placement; retain carry order and SSA ports."""
    from xdsl.dialects.arith import ConstantOp

    operation_sources = {} if operation_sources is None else operation_sources
    replacements = {} if replacements is None else replacements
    origin_store = compilation.workspace.get(OperationOrigins)
    expanded = 0

    def expand(loop):
        nonlocal expanded
        # Inner bodies expand once before the enclosing body's copies are made.
        for nested in tuple(loop.body.block.ops):
            if isinstance(nested, ckks.RepeatOp):
                expand(nested)
        block = loop.parent_block()
        assert block is not None
        width = loop.yield_count.value.data
        markers = {}
        for result in loop.outputs:
            for use in tuple(result.uses):
                if isinstance(use.operation, ckks.RepeatIterationOp):
                    markers[use.operation.iteration.value.data] = use.operation
        carried = tuple(loop.carried)
        round_outputs = []
        for round_ in range(loop.count.value.data):
            marker = markers.get(round_)
            anchor = loop if marker is None else marker
            mapper = dict(
                zip(
                    tuple(loop.body.block.args)[1:],
                    (*loop.shared, *loop.row(round_), *carried),
                    strict=True,
                )
            )
            index_argument = loop.body.block.args[0]
            if any(
                not isinstance(use.operation, ckks.SelectBatchOp)
                for use in index_argument.uses
            ):
                constant = ConstantOp.from_int_and_width(round_, IndexType())
                block.insert_op_before(constant, anchor)
                mapper[index_argument] = constant.result
            terminator = loop.body.block.last_op
            assert isinstance(terminator, ckks.RepeatYieldOp)
            for operation in tuple(loop.body.block.ops)[:-1]:
                if (
                    isinstance(operation, ckks.SelectBatchOp)
                    and operation.index is index_argument
                ):
                    # The round index becomes the row of an ordinary selection.
                    cloned = ckks.SelectBatchOp(
                        mapper.get(operation.value, operation.value),
                        start=operation.start.value.data + round_,
                    )
                    for name, attribute in operation.attributes.items():
                        cloned.attributes.setdefault(name, attribute)
                    mapper[operation.result] = cloned.result
                else:
                    cloned = operation.clone(value_mapper=mapper)
                from ._repeat_rotation import specialize_round

                block.insert_op_before(cloned, anchor)
                specialize_round(compilation, loop, cloned, round_)
                source = operation_sources.get(operation, operation)
                operation_sources[cloned] = source
                if (
                    isinstance(origin_store, OperationOrigins)
                    and operation in origin_store.entries
                ):
                    origin_store.entries[cloned] = origin_store.entries[
                        operation
                    ]
            yielded = tuple(
                mapper.get(value, value) for value in terminator.values
            )
            outputs, carried = yielded[:width], yielded[width:]
            round_outputs.extend(outputs)
            if marker is not None:
                for old, new in zip(
                    tuple(marker.outputs), outputs, strict=True
                ):
                    replacements[old] = new
                Rewriter.replace_op(marker, (), new_results=outputs)
        all_results = (*round_outputs, *carried)
        for old, new in zip(tuple(loop.outputs), all_results, strict=True):
            replacements[old] = new
        Rewriter.replace_op(loop, (), new_results=all_results)
        expanded += 1

    for operation in (
        tuple(compilation.program.walk()) if roots is None else roots
    ):
        if (
            isinstance(operation, ckks.RepeatOp)
            and operation.parent_block() is not None
        ):
            expand(operation)
    return expanded


def _supports_repeat_body(program_pass):
    # Pass identity matters: a caller replacement may keep the same recipe name.
    from . import _lower_and_fuse as recipe
    from .passes.lowering._driver import DEFAULT_CKKS_LOWERINGS

    supported = (
        recipe.ResolveTensorPlaceholdersPass,
        recipe.EliminateDeadValuesPass,
        recipe.ReuseIntermediatesPass,
        recipe.LowerSemanticToLogicalPass,
        recipe.ResolveRotationKeyOperandsPass,
        recipe.InsertMultiplyNttTransitionsPass,
        recipe.InsertPlaintextPreparationPass,
        recipe.LowerLogicalToCkksPass,
        recipe.AssignCkksDepthsPass,
        recipe.AssignCkksScalesPass,
        recipe.LowerMessagePlaintextPreparationPass,
        recipe.SelectNttImplementationsPass,
        recipe.RotationHoistingPass,
        recipe.SelectExecutionLoweringsPass,
        ExpandCkksRepeatsPass,
        recipe.FuseOperationsPass,
    )
    from fhelium.backend.triton._repeat import IdentifyTritonRepeatsPass

    if type(program_pass) not in (*supported, IdentifyTritonRepeatsPass):
        return False
    if type(program_pass) is recipe.SelectExecutionLoweringsPass:
        return (
            program_pass.lowerings is DEFAULT_CKKS_LOWERINGS
            and program_pass.execution_cost is None
        )
    return True


def repeat_loops(program):
    """Return the repeat operations of ``program`` in walk order."""
    return [
        operation
        for operation in program.walk()
        if isinstance(operation, ckks.RepeatOp)
    ]


def prepare_repeat_pass(compilation, program_pass, loops):
    """Bind repeat bodies to the current row state before ``program_pass``.

    ``loops`` are the repeats present in the Program; repeats nested in fusion regions are left to their Backend.
    """
    from fhelium.ir.dialects import fusion

    loops = [
        operation
        for operation in loops
        if not isinstance(operation.parent_op(), fusion.FusedOp)
    ]
    if not loops:
        return
    from ._repeat_rotation import is_rotation_repeat, supports_rotation_policy

    if compilation.workspace.get(
        "ckks_repeat_rotation", True
    ) is False or not supports_rotation_policy(program_pass):
        expand_repeats(
            compilation,
            roots=tuple(loop for loop in loops if is_rotation_repeat(loop)),
        )
        loops = [loop for loop in loops if loop.parent_block() is not None]
    if compilation.workspace.get(
        "ckks_repeat"
    ) is False or not _supports_repeat_body(program_pass):
        expand_repeats(compilation, roots=loops)
        return
    if any(not sync_repeat_inputs(loop) for loop in loops):
        expand_repeats(compilation, roots=loops)


def finish_repeat_pass(compilation):
    """Propagate body result state to repeat results and return the remaining repeats."""
    loops, iterations = [], []
    for operation in compilation.program.walk():
        if isinstance(operation, ckks.RepeatOp):
            loops.append(operation)
        elif isinstance(operation, ckks.RepeatIterationOp):
            iterations.append(operation)
    for operation in reversed(loops):
        if not sync_repeat_results(operation):
            expand_repeats(compilation)
            return []
    for operation in iterations:
        sync_iteration(operation)
    return loops


@dataclass(frozen=True)
class ExpandCkksRepeatsPass:
    """Prepare represented independent bodies or restore the expanded fusion graph."""

    name: str = "expand-ckks-repeats"

    def run(self, compilation):
        if compilation.workspace.get("ckks_repeat_fusion", True):
            from ._repeat_fusion import prepare_fusion_repeats

            count = prepare_fusion_repeats(compilation)
        else:
            count = expand_repeats(compilation)
        return PassResult(
            compilation.program, PassStats(matched=count, transformed=count)
        )
