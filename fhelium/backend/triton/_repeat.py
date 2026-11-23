"""Plan independent fusion subgraphs and emit shared NTT endpoint loops.

Repeat plans retain ordered operands and output ownership. Endpoint projection
keeps the existing NTT partitions and uses CTA-uniform scalar argument selection;
its runtime count is an int64 launch argument, including in AOT signatures.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from math import log2
from pathlib import Path

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Operation

from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE
from fhelium.ir.dialects import ckks, fusion, ntt, rns

# FusedOp attribute holding the planned repeat groups: one array of member
# operation indices per group. The Triton NTT code generator reads it to emit a
# runtime loop over the groups instead of one unrolled copy per group.
REPEAT_ATTRIBUTE = "triton_repeat"
_NTT = (
    ntt.CoefficientStandardToNttMontgomeryOp,
    ntt.CoefficientMontgomeryToNttMontgomeryOp,
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.NttMontgomeryToCoefficientMontgomeryOp,
)
_INVERSE = (
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.NttMontgomeryToCoefficientMontgomeryOp,
)
_MULTIPLY = (
    ckks.MultiplyCompressedPlaintextOp,
    rns.MontgomeryMultiplyOp,
    rns.MultiplyPlaintextOp,
    rns.StandardToMontgomeryOp,
)
_MIN_COUNT = 4
_MIN_SAVING = 256


def _sites(operation: Operation) -> int:
    """Estimate Montgomery sites in the current tiled endpoint partition."""
    if not operation.results:
        return 0
    state = getattr(
        getattr(operation.results[0].type, "state", None), "data", {}
    )
    count = state.get("component_count", state.get("components"))
    components = int(count.value.data) if isinstance(count, IntegerAttr) else 1
    if isinstance(operation, _MULTIPLY):
        return components
    if isinstance(operation, _NTT):
        dimension = state.get("ring_dimension")
        shape = state.get("shape")
        n = (
            int(dimension.value.data)
            if isinstance(dimension, IntegerAttr)
            else int(shape.data[-1].value.data)
            if isinstance(shape, ArrayAttr) and shape.data
            else 0
        )
        return components * min(8, int(log2(n))) if n else 0
    return 0


@dataclass(frozen=True)
class GraphNode:
    """One logical occurrence with SSA bindings and a shared operation definition."""

    operation: Operation
    operands: tuple
    results: tuple


def graph_nodes(operations, *, preserve_carried=False):
    """Visit represented occurrences without cloning operation definitions.

    Keys for body-local results include their iteration path. External materials
    retain their actual SSA identities; loop results alias the yielded keys.
    """

    def visit(items, values, scope):
        for op in items:
            if isinstance(op, ckks.RepeatOp) and not (
                preserve_carried and op.carried
            ):
                carried = tuple(values.get(v, v) for v in op.carried)
                yielded = []
                for i in range(op.count.value.data):
                    bindings = (*op.shared, *op.row(i))
                    local = dict(
                        zip(
                            tuple(op.body.block.args)[1:],
                            (*[values.get(v, v) for v in bindings], *carried),
                            strict=True,
                        )
                    )
                    yield from visit(
                        tuple(op.body.block.ops)[:-1], local, (*scope, op, i)
                    )
                    result = tuple(
                        local.get(v, v) for v in op.body.block.last_op.operands
                    )
                    width = op.yield_count.value.data
                    yielded.extend(result[:width])
                    carried = result[width:]
                values.update(
                    zip(op.results, (*yielded, *carried), strict=True)
                )
                continue
            operands = tuple(values.get(v, v) for v in op.operands)
            results = tuple((scope, v) if scope else v for v in op.results)
            values.update(zip(op.results, results, strict=True))
            yield GraphNode(op, operands, results)

    yield from visit(operations, {}, ())


def structured_groups(operation):
    """Project independent represented bodies to the common NTT anchor plan."""
    nodes = tuple(operation.body.block.ops)
    if not any(isinstance(op, ckks.RepeatOp) for op in nodes):
        return None
    total = sum(
        _sites(op)
        if not isinstance(op, ckks.RepeatOp)
        else op.count.value.data
        * sum(_sites(body) for body in op.body.block.ops)
        for op in nodes
    )
    anchor = 0
    groups = []
    for op in nodes:
        if isinstance(op, _NTT):
            anchor += 1
        elif isinstance(op, ckks.RepeatOp):
            body = tuple(op.body.block.ops)
            transforms = [item for item in body if isinstance(item, _NTT)]
            count = op.count.value.data
            saved = (count - 1) * sum(_sites(item) for item in body)
            if (
                not op.carried
                and len(transforms) == 1
                and isinstance(transforms[0], _INVERSE)
                and count >= _MIN_COUNT
                and saved >= _MIN_SAVING
                and saved * 4 >= total
            ):
                groups.append(tuple(range(anchor, anchor + count)))
            anchor += count * len(transforms)
    return tuple(groups)


@dataclass
class RepeatPlans:
    """Workspace-local structural tokens and immutable anchor-index plans."""

    tokens: dict[object, int] = field(default_factory=dict)
    regions: dict[object, tuple[tuple[int, ...], ...]] = field(
        default_factory=dict
    )

    def token(self, key: object) -> int:
        result = self.tokens.get(key)
        if result is None:
            result = len(self.tokens)
            self.tokens[key] = result
        return result

    def identify(
        self, operation: fusion.FusedOp
    ) -> tuple[tuple[int, ...], ...]:
        structured = structured_groups(operation)
        if structured is not None:
            return structured
        nodes = tuple(
            op
            for op in operation.body.block.ops
            if not isinstance(op, fusion.YieldOp)
        )
        # No repeated operation class means no repeated typed subgraph.
        if not any(
            n >= _MIN_COUNT
            for kind, n in Counter(type(op) for op in nodes).items()
            if kind is not UnrealizedConversionCastOp
        ):
            return ()
        positions = {op: i for i, op in enumerate(nodes)}
        attribute_tokens = {}

        def attribute(value):
            key = id(value)
            if key not in attribute_tokens:
                attribute_tokens[key] = self.token(value)
            return attribute_tokens[key]

        arguments = tuple(attribute(v.type) for v in operation.body.block.args)
        values = {v: i for i, v in enumerate(operation.body.block.args)}
        descriptions = []
        bases = {}
        for op in nodes:
            bases[op] = self.token(
                (
                    type(op),
                    tuple(
                        sorted(
                            (key, attribute(value))
                            for key, value in op.attributes.items()
                        )
                    ),
                    tuple(
                        sorted(
                            (key, attribute(value))
                            for key, value in op.properties.items()
                        )
                    ),
                    tuple(attribute(v.type) for v in op.operands),
                    tuple(attribute(v.type) for v in op.results),
                )
            )
            descriptions.append(
                (bases[op], tuple(values[v] for v in op.operands))
            )
            for value in op.results:
                values[value] = len(values)
        yielded = tuple(
            values[v] for v in operation.body.block.last_op.operands
        )
        signature = arguments, tuple(descriptions), yielded
        cached = self.regions.get(signature)
        if cached is not None:
            return cached

        value_keys = {
            v: self.token(("input", attribute(v.type)))
            for v in operation.body.block.args
        }
        fingerprints = {}
        depths = {}
        buckets = defaultdict(list)
        for op in nodes:
            fingerprints[op] = self.token(
                (
                    "operation",
                    bases[op],
                    tuple(value_keys[v] for v in op.operands),
                )
            )
            depths[op] = 1 + max(
                (depths.get(v.owner, 0) for v in op.operands), default=0
            )
            buckets[fingerprints[op]].append(op)
            for i, value in enumerate(op.results):
                value_keys[value] = self.token(("result", fingerprints[op], i))
        seeds = [
            roots for roots in buckets.values() if len(roots) >= _MIN_COUNT
        ]
        seeds.sort(key=lambda roots: depths[roots[0]], reverse=True)
        anchors = {
            op: i
            for i, op in enumerate(op for op in nodes if isinstance(op, _NTT))
        }
        total = sum(_sites(op) for op in nodes)
        claimed = set()
        plans = []
        for roots in seeds:
            if any(op in claimed for op in roots):
                continue
            groups = [set() for _ in roots]
            owners = {}
            todo = [tuple(roots)]
            while todo:
                column = todo.pop()
                if all(op in groups[i] for i, op in enumerate(column)):
                    continue
                for i, op in enumerate(column):
                    groups[i].add(op)
                    owners[op] = i
                for j in range(len(column[0].operands)):
                    operands = tuple(op.operands[j] for op in column)
                    if len(set(operands)) == 1:
                        continue
                    predecessors = tuple(v.owner for v in operands)
                    if (
                        len(set(predecessors)) == len(roots)
                        and all(op in positions for op in predecessors)
                        and len({fingerprints[op] for op in predecessors}) == 1
                        and all(
                            op not in owners or owners[op] == i
                            for i, op in enumerate(predecessors)
                        )
                    ):
                        todo.append(predecessors)
            union = set().union(*groups)
            if union & claimed or any(op.regions for op in union):
                continue
            if any(
                v.owner in union and v.owner not in group
                for group in groups
                for op in group
                for v in op.operands
            ):
                continue
            # Endpoint loop projection supports one inverse transform per iteration. Any
            # intermediate result used outside its iteration is a live-value
            # bank and remains unrolled.
            group_anchors = [
                tuple(op for op in group if isinstance(op, _NTT))
                for group in groups
            ]
            if any(
                len(items) != 1 or not isinstance(items[0], _INVERSE)
                for items in group_anchors
            ):
                continue
            if any(
                op is not root
                and any(
                    use.operation not in group
                    for value in op.results
                    for use in value.uses
                )
                for root, group in zip(roots, groups, strict=True)
                for op in group
            ):
                continue
            saved = (len(roots) - 1) * sum(_sites(op) for op in groups[0])
            if saved < _MIN_SAVING or saved * 4 < total:
                continue
            plans.append(tuple(anchors[items[0]] for items in group_anchors))
            claimed.update(union)
        result = tuple(plans)
        self.regions[signature] = result
        return result


def repeat_groups(operation: fusion.FusedOp) -> tuple[tuple[int, ...], ...]:
    value = operation.attributes.get(REPEAT_ATTRIBUTE)
    if not isinstance(value, ArrayAttr):
        return ()
    return tuple(
        tuple(int(index.value.data) for index in group) for group in value
    )


@dataclass(frozen=True)
class IdentifyTritonRepeatsPass:
    """Annotate independent large repeats after registered Triton fusion.

    ``workspace['triton_repeat']`` defaults to True. False retains the original
    unrolled endpoint representation. Analysis is cached by fusion structure;
    execution binding and Graph replay only consume the resulting annotation.
    """

    implementations: tuple[str, ...] = ()
    name: str = "identify-triton-repeats"

    def run(self, compilation):
        from fhelium.compile._pipeline import PassResult, PassStats

        if not self.implementations:
            return PassResult(compilation.program)
        # Workspace switch for runtime-loop generation of repeated Triton groups;
        # False keeps one unrolled copy of the code per group.
        enabled = compilation.workspace.get("triton_repeat", True)
        plans = (
            compilation.workspace.setdefault(RepeatPlans, RepeatPlans())
            if enabled
            else None
        )
        matched = transformed = visited = 0
        for op in compilation.program.module.walk():
            selected = op.attributes.get(EXECUTION_IMPLEMENTATION_ATTRIBUTE)
            if (
                not isinstance(op, fusion.FusedOp)
                or not isinstance(selected, StringAttr)
                or selected.data not in self.implementations
            ):
                continue
            visited += 1
            groups = plans.identify(op) if plans is not None else ()
            matched += len(groups)
            if groups:
                annotation = ArrayAttr(
                    ArrayAttr(IntegerAttr(index, 64) for index in group)
                    for group in groups
                )
                if op.attributes.get(REPEAT_ATTRIBUTE) != annotation:
                    op.attributes[REPEAT_ATTRIBUTE] = annotation
                    transformed += 1
            elif op.attributes.pop(REPEAT_ATTRIBUTE, None) is not None:
                transformed += 1
        return PassResult(
            compilation.program,
            PassStats(
                matched=visited,
                transformed=transformed,
                skipped=visited - transformed,
            ),
            diagnostics=(f"Identified {matched} independent repeat groups",)
            if matched
            else (),
        )


@dataclass(frozen=True)
class EndpointBody:
    anchor: int
    start: int
    stop: int
    partition: tuple[object, ...]


def _variables(nodes, context):
    return {
        n.id
        for node in nodes
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, context)
    }


def _canonical(nodes, parameters):
    local = _variables(nodes, ast.Store)
    locals_map = {}
    inputs = {}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in local:
                name = locals_map.setdefault(
                    node.id, f"local_{len(locals_map)}"
                )
            elif node.id in parameters:
                role = parameters[node.id]
                name = inputs.setdefault(
                    node.id, (f"argument_{len(inputs)}", role)
                )[0]
            else:
                # Shared live block values and arithmetic functions retain
                # identity. Selecting a different live SSA bank is not allowed.
                name = node.id
            return ast.copy_location(ast.Name(id=name, ctx=node.ctx), node)

    tree = Rename().visit(ast.Module(body=nodes, type_ignores=[]))
    return ast.dump(tree, include_attributes=False), locals_map, inputs


def project_repeats(lines, arguments, endpoints, groups):
    """Fold matching endpoint bodies without moving stores or phase barriers.

    Pointer/scalar choices are uniform across each CTA. Count arguments are
    appended after the existing runtime scalars; no device descriptor uploads
    or additional dispatches are introduced.
    """
    if not groups:
        return lines, arguments, ()
    by_anchor = {body.anchor: body for body in endpoints}
    candidates = []
    for group in groups:
        if not all(index in by_anchor for index in group):
            continue
        bodies = sorted(
            (by_anchor[index] for index in group), key=lambda body: body.start
        )
        if len({body.partition for body in bodies}) != 1 or any(
            a.stop != b.start for a, b in zip(bodies, bodies[1:])
        ):
            continue
        candidates.append(tuple((body.start, body.stop) for body in bodies))
    if not candidates:
        return lines, arguments, ()
    return _project_source(tuple(lines), tuple(arguments), tuple(candidates))


@lru_cache(maxsize=1)
def _projection_version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _projection_path(source_lines, parameter_declarations, candidates):
    from fhelium.compile._cache import cache_directory

    directory = cache_directory()
    if directory is None:
        return None
    try:
        signature = (
            1,
            _projection_version(),
            sys.version_info[:3],
            source_lines,
            parameter_declarations,
            candidates,
        )
        key = hashlib.sha256(_projection_json(signature).encode()).hexdigest()
        return directory / "triton-repeat" / key[:2] / (key + ".json")
    except OSError:
        return None


def _projection_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_projection(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["schema"] != 1 or record["key"] != path.stem:
            return None
        result = record["result"]
        if (
            record["sha256"]
            != hashlib.sha256(_projection_json(result).encode()).hexdigest()
        ):
            return None
        lines, declarations, counts = result
        if not all(isinstance(row, list) for row in result):
            return None
        if not all(
            isinstance(value, str)
            for row in (lines, declarations)
            for value in row
        ):
            return None
        if not all(type(value) is int and value > 0 for value in counts):
            return None
        return tuple(lines), tuple(declarations), tuple(counts)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return None


def _store_projection(path, result):
    record = {
        "schema": 1,
        "key": path.stem,
        "result": result,
        "sha256": hashlib.sha256(_projection_json(result).encode()).hexdigest(),
    }
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.stem + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_projection_json(record))
        os.replace(temporary, path)
    except OSError:
        # This optional artifact must not prevent source generation.
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


@lru_cache(maxsize=128)
def _project_source(source_lines, parameter_declarations, candidates):
    """Reuse endpoint text independently of live material and layout values.

    The key contains source, argument declarations and validated source spans.
    It excludes Tensor bindings, runtime scalar values and global anchor IDs.
    Cached results are immutable so later preparation cannot alter a prior plan.
    Persistent entries follow FHELIUM_CACHE_DIR, including its disable settings.
    """
    path = _projection_path(source_lines, parameter_declarations, candidates)
    if path is not None:
        cached = _load_projection(path)
        if cached is not None:
            return cached
    lines = list(source_lines)
    arguments = list(parameter_declarations)
    parameters = {
        arg.split(":")[0]: (
            arg.split(":", 1)[1].strip() if ":" in arg else "pointer"
        )
        for arg in arguments
    }
    total = sum(line.count("montgomery_mul(") for line in lines)
    replacements = []
    counts = []
    for bodies in candidates:
        parsed = [
            ast.parse("\n".join(lines[start:stop])).body
            for start, stop in bodies
        ]
        # Only the first tree is needed for emission. Normalize the temporary
        # analysis trees in place and retain original names in their maps.
        original = copy.deepcopy(ast.Module(body=parsed[0], type_ignores=[]))
        canonical = [_canonical(body, parameters) for body in parsed]
        shape, local, inputs = canonical[0]
        roles = tuple(role for name, role in inputs.values())
        if any(
            key != shape
            or tuple(role for name, role in mapping.values()) != roles
            for key, _, mapping in canonical[1:]
        ):
            continue
        definitions = set().union(*(mapping for _, mapping, _ in canonical))
        remaining = definitions.copy()
        escaped = False
        for statement in ast.parse("\n".join(lines[bodies[-1][1] :])).body:
            if remaining & _variables((statement,), ast.Load):
                escaped = True
                break
            remaining -= _variables((statement,), ast.Store)
        if escaped:
            continue
        sites = sum(
            line.count("montgomery_mul(")
            for line in lines[bodies[0][0] : bodies[0][1]]
        )
        saving = (len(bodies) - 1) * sites
        if saving < _MIN_SAVING or saving * 4 < total:
            continue
        number = len(counts)
        count = f"repeat{number}_count"
        iteration = f"repeat{number}_index"
        mappings = [tuple(mapping) for _, _, mapping in canonical]
        columns = tuple(zip(*mappings, strict=True))
        varying = [
            (i, column)
            for i, column in enumerate(columns)
            if len(set(column)) != 1
        ]
        substitutions = {
            name: f"repeat{number}_{value}" for name, value in local.items()
        }
        for i, column in varying:
            substitutions[column[0]] = f"repeat{number}_argument{i}"
        select = []
        if varying:
            for index in range(len(bodies)):
                if index == len(bodies) - 1:
                    select.append("else:")
                else:
                    keyword = "if" if index == 0 else "elif"
                    select.append(f"{keyword} {iteration} == {index}:")
                for i, column in varying:
                    value = column[index]
                    if roles[i] == "tl.constexpr":
                        value = f"tl.full((), {value}, tl.int64)"
                    select.append(f"    repeat{number}_argument{i} = {value}")

        class Substitute(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.copy_location(
                    ast.Name(
                        id=substitutions.get(node.id, node.id), ctx=node.ctx
                    ),
                    node,
                )

        template = Substitute().visit(original)
        body_lines = select + ast.unparse(template).splitlines()
        folded = [
            f"for {iteration} in tl.range(0, {count}, loop_unroll_factor=1):"
        ]
        folded.extend("    " + line for line in body_lines)
        replacements.append((bodies[0][0], bodies[-1][1], folded))
        counts.append(len(bodies))
        arguments.append(f"{count}: tl.int64")
    for start, stop, replacement in sorted(replacements, reverse=True):
        lines[start:stop] = replacement
    result = tuple(lines), tuple(arguments), tuple(counts)
    if path is not None:
        _store_projection(path, result)
    return result
