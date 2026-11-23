"""Represent bounded SSA port graphs and reusable builtin matcher recipes.

An occurrence binds ordered operand and result ports to the current Program.
Plans retain structure and operand-role rules; current types, material values,
uses and scheduling state are evaluated by the Backend matcher. Source hints
identify candidate origins without participating in semantic equivalence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any

from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue

from fhelium.backend.triton._matching import (
    OperationRecipe,
    operation_recipe,
    recipe_dependencies,
)

from ._materials import MATERIAL_REQUEST_ATTRIBUTE
from ._origins import get_origin
from ._selection_plan import _encode_attribute

if TYPE_CHECKING:
    from ._selection_plan import SelectionPlans


def _type_name(value_type: object) -> tuple[str, str]:
    cls = type(value_type)
    return cls.__module__, cls.__qualname__


@dataclass(frozen=True)
class PortGraph:
    """Ordered local edges and external producer-sharing relationships.

    Every result is a potential output port. Which ports currently escape is
    evaluated on the occurrence, so changes to outside uses remain visible.
    """

    nodes: tuple
    inputs: tuple
    outputs: tuple

    def key(self) -> tuple:
        return self.nodes, self.inputs, self.outputs


@dataclass(frozen=True)
class Occurrence:
    """A bounded candidate mapped to current operations and external values."""

    port_graph: PortGraph
    nodes: tuple[Operation, ...]
    inputs: tuple[SSAValue, ...]
    origin: dict[str, Any] | None = None

    @classmethod
    def from_operations(
        cls,
        operations: Sequence[Operation],
        *,
        origin: dict[str, Any] | None = None,
    ) -> Occurrence:
        nodes = tuple(operations)
        if not nodes or len(nodes) > 8:
            raise ValueError(
                "Local candidates contain between one and eight operations"
            )
        if any(op.regions or op.successors for op in nodes):
            raise ValueError("Local candidates require flat dataflow")
        positions = {op: index for index, op in enumerate(nodes)}
        inputs: dict[SSAValue, int] = {}
        producers: dict[Operation, int] = {}
        input_graph = []

        def edge(value: SSAValue) -> tuple:
            if value.owner in positions:
                return "result", positions[value.owner], value.index
            index = inputs.get(value)
            if index is None:
                index = len(inputs)
                inputs[value] = index
                if isinstance(value, BlockArgument):
                    producer = ("argument",)
                else:
                    producer = (
                        "producer",
                        producers.setdefault(value.owner, len(producers)),
                        value.index,
                    )
                input_graph.append((producer, _type_name(value.type)))
            return "input", index

        graph_nodes = tuple(
            (
                type(op).__module__,
                type(op).__qualname__,
                op.name,
                tuple(edge(value) for value in op.operands),
                tuple(_type_name(value.type) for value in op.results),
            )
            for op in nodes
        )
        outputs = tuple(
            (index, result)
            for index, op in enumerate(nodes)
            for result in range(len(op.results))
        )
        return cls(
            PortGraph(graph_nodes, tuple(input_graph), outputs),
            nodes,
            tuple(inputs),
            origin,
        )

    def escaping_outputs(self) -> tuple[SSAValue, ...]:
        selected = set(self.nodes)
        return tuple(
            result
            for operation in self.nodes
            for result in operation.results
            if any(use.operation not in selected for use in result.uses)
        )


@dataclass(frozen=True)
class LocalPlan:
    """Content-addressed port structure, fixed facts and matcher rules."""

    port_graph: PortGraph
    fixed_facts: tuple
    matcher_recipe: OperationRecipe

    def to_dict(self) -> dict[str, Any]:
        return {
            "port_graph": self.port_graph.key(),
            "fixed_facts": self.fixed_facts,
            "matcher_recipe": self.matcher_recipe.to_dict(),
        }


class LocalPlanScope:
    """Lookup local recipes during one pass without retaining prior SSA values.

    Single-operation neighborhoods provide a bounded DAG fallback. Captured and
    lowering-origin occurrences use the same port extraction and fixed-fact
    guard. Instance evaluation follows the recipe's current operand positions.
    """

    def __init__(self, plans: SelectionPlans, workspace, policy):
        self.plans = plans
        self.workspace = workspace
        self.policy = policy
        self.stats = workspace.setdefault("local_plan_stats", {})
        for name in (
            "occurrences",
            "recipe_hits",
            "recipe_misses",
            "facts_hits",
            "facts_misses",
            "guard_bypasses",
        ):
            self.stats.setdefault(name, 0)
        self.stats.setdefault("sources", {})
        self.recipes: dict[tuple, OperationRecipe] = {}
        # Retaining each Attribute for this scope prevents Python identity reuse.
        self.attributes: dict[int, tuple[Attribute, tuple]] = {}

    def _attribute(self, attribute: Attribute) -> tuple:
        cached = self.attributes.get(id(attribute))
        if cached is not None:
            return cached[1]
        definitions = {}
        reference = _encode_attribute(attribute, {}, definitions)
        result = reference, tuple(definitions)
        self.attributes[id(attribute)] = attribute, result
        return result

    def _fixed_facts(self, operation: Operation) -> tuple:
        return (
            tuple(
                (key, self._attribute(value))
                for key, value in sorted(operation.attributes.items())
                if key != MATERIAL_REQUEST_ATTRIBUTE
            ),
            tuple(
                (key, self._attribute(value))
                for key, value in sorted(operation.properties.items())
            ),
            recipe_dependencies(operation),
        )

    def matcher_recipe(self, operation: Operation) -> OperationRecipe:
        origin = get_origin(operation, self.workspace)
        source = origin.get("source", "dag") if origin else "dag"
        self.stats["occurrences"] += 1
        sources = self.stats["sources"]
        sources[source] = sources.get(source, 0) + 1
        try:
            occurrence = Occurrence.from_operations((operation,), origin=origin)
            fixed = self._fixed_facts(operation)
        except ValueError:
            self.stats["guard_bypasses"] += 1
            return operation_recipe(operation)
        key = occurrence.port_graph.key(), fixed
        recipe = self.recipes.get(key)
        if recipe is not None:
            self.stats["recipe_hits"] += 1
            return recipe
        address = hashlib.blake2b(
            json.dumps(
                ("rns-local-v1", self.policy, key),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            digest_size=24,
        ).hexdigest()
        saved = self.plans.local_entries.get(address)
        if saved is None:
            recipe = operation_recipe(operation)
            self.plans.local_entries[address] = LocalPlan(
                occurrence.port_graph, fixed, recipe
            ).to_dict()
            self.stats["recipe_misses"] += 1
        else:
            recipe = OperationRecipe.from_dict(saved["matcher_recipe"])
            self.stats["recipe_hits"] += 1
        self.recipes[key] = recipe
        return recipe


def local_plan_scope(workspace, policy) -> LocalPlanScope | None:
    """Enable declared builtin rules when the source-family plan store is present."""
    from ._selection_plan import SelectionPlans

    plans = workspace.get(SelectionPlans)
    if (
        policy is None
        or not isinstance(plans, SelectionPlans)
        or not workspace.get("local_plan_reuse", True)
    ):
        return None
    return LocalPlanScope(plans, workspace, policy)
