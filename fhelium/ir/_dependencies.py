"""Describe output-to-input element dependencies in an operation's coordinates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue

DependencyKind: TypeAlias = Literal["element", "reindexed", "mixing", "unknown"]


@dataclass(frozen=True)
class ValueDependency:
    """Describe one result's access to one operand along named logical axes.

    ``element`` reads the corresponding position; ``reindexed`` reads a
    mapped position, including selection and broadcast; ``mixing`` may read
    multiple positions. Missing axes have unknown relationships. Axis names
    refer to coordinates at the operation's IR level, such as ``slot``,
    ``coefficient``, ``limb``, or ``component``. The coefficient axis is
    the last RNS payload dimension in both coefficient and NTT form.
    """

    result_index: int
    operand_index: int
    axes: Mapping[str, DependencyKind] = field(default_factory=dict)


@dataclass(frozen=True)
class OperationDependencies:
    """Collect partial element relationships between results and operands.

    An omitted pair or axis is unknown. Descriptions bound possible reads;
    they need not enumerate the elements read for each numerical input.
    """

    relations: tuple[ValueDependency, ...] = ()

    def axes(
        self, result_index: int, operand_index: int
    ) -> Mapping[str, DependencyKind]:
        """Return known axis relationships for one result/operand pair."""
        for relation in self.relations:
            if (relation.result_index, relation.operand_index) == (
                result_index,
                operand_index,
            ):
                return relation.axes
        return {}

    def kind(
        self, result_index: int, operand_index: int, axis: str
    ) -> DependencyKind:
        """Read one relationship, retaining unknown pairs and axes."""
        return self.axes(result_index, operand_index).get(axis, "unknown")


UNKNOWN: DependencyKind = "unknown"

DependencyDescription: TypeAlias = (
    OperationDependencies | Callable[[Operation], OperationDependencies] | None
)


def operation_dependencies(operation: Operation) -> OperationDependencies:
    """Resolve the registered element relationships of a current IR instance."""
    from ._operation_specs import DEFAULT_OPERATION_SPECS

    spec = DEFAULT_OPERATION_SPECS.get(operation.name)
    if spec is None or (
        spec.operation_type is not None
        and spec.operation_type is not type(operation)
    ):
        return OperationDependencies()
    return spec.resolve_dependencies(operation)


def value_state(value: SSAValue, name: str) -> object:
    """Read an available fact from an open value type."""
    state = getattr(getattr(value.type, "state", None), "data", {})
    return state.get(name)


def string_fact(
    operation: Operation,
    name: str,
    *,
    operand: int = 0,
    state_name: str | None = None,
) -> str | None:
    """Read a represented attribute or an available operand-state fact."""
    fact = operation.attributes.get(name)
    if fact is None and operand < len(operation.operands):
        fact = value_state(operation.operands[operand], state_name or name)
    return (
        fact.data
        if isinstance(fact, StringAttr) and fact.data != "unknown"
        else None
    )


def operand_dependencies(
    operation: Operation,
    operands: Sequence[int],
    axes: Mapping[str, DependencyKind],
) -> OperationDependencies:
    """Describe each result's relationship to the selected data operands."""
    return OperationDependencies(
        tuple(
            ValueDependency(result, operand, dict(axes))
            for result in range(len(operation.results))
            for operand in operands
            if operand < len(operation.operands)
        )
    )


def _promote(kind: DependencyKind) -> DependencyKind:
    """Promote a mapped read that reaches one result through two operands."""
    return "mixing" if kind == "reindexed" else kind


def _join(
    left: Mapping[str, DependencyKind], right: Mapping[str, DependencyKind]
) -> Mapping[str, DependencyKind]:
    """Combine two axis maps along every axis either of them describes."""
    return {
        axis: _combine(left.get(axis, UNKNOWN), right.get(axis, UNKNOWN))
        for axis in left.keys() | right.keys()
    }


def _promoted(
    axes: Mapping[str, DependencyKind],
) -> Mapping[str, DependencyKind]:
    """Promote every mapped axis of a merged path."""
    return {axis: _promote(kind) for axis, kind in axes.items()}


def _combine(left: DependencyKind, right: DependencyKind) -> DependencyKind:
    if "unknown" in (left, right):
        return "unknown"
    if "mixing" in (left, right):
        return "mixing"
    if "reindexed" in (left, right):
        return "reindexed"
    return "element"


def region_dependencies(operation: Operation) -> OperationDependencies:
    """Compose a single-block region's dataflow into outer relationships.

    Block arguments correspond to outer operands and the terminator yields
    outer results. A missing relationship on a contributing path leaves that
    path unknown. Multiple paths with mapped reads conservatively mix.
    """
    if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        return OperationDependencies()
    block = operation.regions[0].block
    if len(block.args) != len(operation.operands) or block.last_op is None:
        return OperationDependencies()
    # None is an identity path from a block argument, for every queried axis.
    paths: dict[SSAValue, dict[int, Mapping[str, DependencyKind] | None]] = {
        value: {index: None} for index, value in enumerate(block.args)
    }
    for child in block.ops:
        if child is block.last_op:
            break
        dependencies = operation_dependencies(child)
        for result_index, result in enumerate(child.results):
            combined: dict[int, Mapping[str, DependencyKind] | None] = {}
            for operand_index, operand in enumerate(child.operands):
                edge = dependencies.axes(result_index, operand_index)
                for external, prefix in paths.get(operand, {}).items():
                    merged: Mapping[str, DependencyKind] = (
                        edge if prefix is None else _join(edge, prefix)
                    )
                    previous = combined.get(external)
                    if previous is not None:
                        merged = _promoted(_join(previous, merged))
                    combined[external] = merged
            paths[result] = combined
    relations = []
    for result_index, value in enumerate(block.last_op.operands):
        for operand_index, axes in paths.get(value, {}).items():
            # Identity yields use the axes represented by the public value type.
            if axes is None:
                axes = identity_axes(value)
            relations.append(ValueDependency(result_index, operand_index, axes))
    return OperationDependencies(tuple(relations))


def identity_axes(value: SSAValue) -> Mapping[str, DependencyKind]:
    """Identify preserved coordinates from the represented value kind."""
    name = value.type.name
    state = getattr(getattr(value.type, "state", None), "data", {})
    if (
        name
        in {
            "fhelium_rns.bundle",
            "fhelium_ckks.ciphertext",
            "fhelium_ckks.compressed_plaintext",
        }
        or "polynomial_domain" in state
    ):
        axes = ("coefficient", "limb", "component", "batch")
    elif name.startswith(("fhelium_logical.", "fhelium_semantic.")):
        axes = (
            ("slot", "batch")
            if name.startswith("fhelium_logical.")
            else ("tensor_position",)
        )
    else:
        axes = ("tensor_position",)
    return {axis: "element" for axis in axes}


def identity_dependencies(operation: Operation) -> OperationDependencies:
    """Describe a single-value storage cast preserving its coordinate type."""
    if len(operation.operands) != 1 or len(operation.results) != 1:
        return OperationDependencies()
    if operation.operands[0].type != operation.results[0].type:
        return OperationDependencies()
    return OperationDependencies(
        (ValueDependency(0, 0, identity_axes(operation.operands[0])),)
    )


def operand_relations(
    operation: Operation,
    axes: Mapping[int, Mapping[str, DependencyKind]],
) -> OperationDependencies:
    """Describe each result against the selected operand relationships."""
    return OperationDependencies(
        tuple(
            ValueDependency(result_index, operand_index, dict(operand_axes))
            for result_index in range(len(operation.results))
            for operand_index, operand_axes in axes.items()
            if operand_index < len(operation.operands)
        )
    )
