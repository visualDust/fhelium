"""Recognize pure RNS expressions and NTT dependencies from Program facts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
import triton
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import BlockArgument, Operation, SSAValue

from fhelium.ir import (
    DEFAULT_OPERATION_SPECS,
    EXECUTION_IMPLEMENTATION_ATTRIBUTE,
    operation_dependencies,
)
from fhelium.ir.dialects import ckks, core, fusion, ntt


def _transform(operation: Operation) -> bool:
    structure = generated_structure(operation)
    return structure is not None and structure.layout == "transform"


_INVERSE_TRANSFORMS = (
    ntt.NttMontgomeryToCoefficientStandardOp,
    ntt.NttMontgomeryToCoefficientMontgomeryOp,
)

# The LLVM SLP vectorizer bundled with Triton 3.7 miscompiles radix-62 NTT
# endpoint kernels for CUDA devices below compute capability 9.0. Triton 3.8
# generates correct code; with earlier Triton these transforms run natively.
_INT64_NTT_MISCOMPILE = tuple(
    int(part) for part in triton.__version__.split(".")[:2]
) < (3, 8)

from ._expressions import (
    component_axis,
)
from ._layout import (
    generated_structure,
    Shape,
    check_ntt_extent,
    check_ntt_layouts,
    result_layout,
)


def _outer(value: SSAValue) -> SSAValue:
    if isinstance(value, BlockArgument):
        parent = value.owner.parent_op()
        if isinstance(parent, fusion.FusedOp):
            return _outer(parent.inputs[value.index])
    return value


def _state(value: SSAValue) -> dict:
    return getattr(getattr(value.type, "state", None), "data", {})


def _integers(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, ArrayAttr):
        return None
    result: list[int] = []
    for item in value:
        if not isinstance(item, IntegerAttr):
            return None
        result.append(int(item.value.data))
    return tuple(result)


@dataclass(frozen=True)
class _TypeFacts:
    """Region-compatibility facts read from one value type's state."""

    facts: tuple[tuple[str, object], ...]
    prime_ids: tuple[int, ...] | None
    device: torch.device | None


_TYPE_FACTS: dict[int, tuple[object, _TypeFacts | None]] = {}


def _type_facts(value_type: object) -> _TypeFacts | None:
    """Return the basis, prime, ring, dtype and device facts of ``value_type``.

    ``None`` marks a type that cannot enter a region: a non-integer residue dtype, an empty prime basis, or a declared non-CUDA device. Results are cached by type identity because one ciphertext type is shared by many operands.
    """
    entry = _TYPE_FACTS.get(id(value_type))
    if entry is not None and entry[0] is value_type:
        return entry[1]
    state = getattr(getattr(value_type, "state", None), "data", {})
    facts: list[tuple[str, object]] = []
    result: _TypeFacts | None = None
    for field in ("basis", "prime_ids", "ring_dimension", "dtype"):
        fact = state.get(field)
        if fact is None or fact == StringAttr("unknown"):
            continue
        if field == "dtype" and isinstance(fact, StringAttr):
            fact = fact.data.removeprefix("torch.")
            if fact not in ("int32", "int64"):
                break
        if field == "prime_ids":
            ids = _integers(fact)
            if ids is not None and not ids:
                break
        facts.append((field, fact))
    else:
        device = None
        declared = state.get("device")
        valid = True
        if isinstance(declared, StringAttr) and declared.data != "unknown":
            try:
                device = torch.device(declared.data)
            except RuntimeError:
                valid = False
            else:
                valid = device.type == "cuda"
        if valid:
            result = _TypeFacts(
                tuple(facts), _integers(state.get("prime_ids")), device
            )
    _TYPE_FACTS[id(value_type)] = (value_type, result)
    return result


_TYPE_SHAPES: dict[int, tuple[object, Shape]] = {}


def _type_shape(value_type: object) -> Shape:
    """Return the declared storage shape of ``value_type``, cached by type identity."""
    entry = _TYPE_SHAPES.get(id(value_type))
    if entry is not None and entry[0] is value_type:
        return entry[1]
    state = getattr(getattr(value_type, "state", None), "data", {})
    shape = _integers(state.get("shape"))
    _TYPE_SHAPES[id(value_type)] = (value_type, shape)
    return shape


def _layout_resolver():
    @cache
    def layout(value: SSAValue) -> tuple[Shape, bool]:
        value = _outer(value)
        owner = value.owner
        if (
            type(owner) is UnrealizedConversionCastOp
            and len(owner.inputs) == len(owner.outputs) == 1
        ):
            shape, _ = layout(owner.inputs[0])
            # Storage is shared, but the result type determines which axis
            # represents ciphertext components rather than ordinary batches.
            return shape, component_axis(value.type)
        if (
            isinstance(owner, Operation)
            and (structure := generated_structure(owner)) is not None
        ):
            facts = [
                layout(owner.operands[i])
                for i in structure.operands("data", "compact")
            ]
            return result_layout(
                structure, [f[0] for f in facts], [f[1] for f in facts]
            )
        return _type_shape(value.type), component_axis(value.type)

    return layout


def _boundary(value: SSAValue, layout):
    try:
        shape, axis = layout(value)
        if shape is not None:
            check_ntt_layouts([(shape, axis)])
            return shape[-2:], shape[1:-2] if axis else shape[:-2], False
        state = _state(value)
        dimension = state.get("ring_dimension")
        check_ntt_extent(
            int(dimension.value.data)
            if isinstance(dimension, IntegerAttr)
            else None
        )
        return None, _integers(state.get("batch_shape")), False
    except (ValueError, TypeError, IndexError):
        return None, None, True


def _supported_dependencies(operation: Operation) -> bool:
    spec = DEFAULT_OPERATION_SPECS.get(operation.name)
    if (
        spec is None
        or spec.operation_type is not type(operation)
        or spec.effect != "pure"
    ):
        return False
    structure = generated_structure(operation)
    # Generated key products address the active Q rows as one stored slice.
    if structure is None or "key_q_rows" in operation.attributes:
        return False
    dependencies = operation_dependencies(operation)
    payload = structure.operands("data", "compact", "key")
    for result_index in range(len(operation.results)):
        for operand_index in payload:
            role = structure.roles[operand_index]
            position = dependencies.kind(
                result_index, operand_index, "coefficient"
            )
            limb = dependencies.kind(result_index, operand_index, "limb")
            if structure.layout == "transform":
                allowed_positions = ("mixing",)
            elif role == "compact":
                allowed_positions = ("element", "reindexed")
            else:
                allowed_positions = ("element",)
            allowed_limbs = (
                ("element", "reindexed") if role == "key" else ("element",)
            )
            if position not in allowed_positions or limb not in allowed_limbs:
                return False
    return True


@dataclass(frozen=True)
class OperationRecipe:
    """Position-based RNS matcher rules, independent of a particular SSA graph."""

    kind: str
    numeric_indices: tuple[int, ...] = ()
    parameter_index: int | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "numeric_indices": list(self.numeric_indices),
            "parameter_index": self.parameter_index,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> OperationRecipe:
        return cls(
            payload["kind"],
            tuple(payload["numeric_indices"]),
            payload["parameter_index"],
        )


def recipe_dependencies(operation: Operation) -> tuple:
    """Return state reads used while constructing the dependency recipe."""
    structure = generated_structure(operation)
    if structure is not None and "compact" in structure.roles:
        from fhelium.ir._dependencies import string_fact

        return (string_fact(operation, "compression_layout"),)
    return ()


def operation_recipe(operation: Operation) -> OperationRecipe:
    """Resolve operand roles and dependency support for a builtin operation."""
    if isinstance(operation, core.MaterialRefOp):
        return OperationRecipe("material")
    if type(operation) is UnrealizedConversionCastOp:
        if len(operation.operands) == len(operation.results) == 1:
            return OperationRecipe("cast", (0,))
        return OperationRecipe("reject")
    structure = generated_structure(operation)
    if (
        structure is None
        or not structure.accepted
        or not _supported_dependencies(operation)
    ):
        return OperationRecipe("reject")
    parameters = structure.operands("parameters")
    return OperationRecipe(
        _RECIPE_KINDS[structure.layout],
        structure.operands("data", "compact"),
        parameters[0] if parameters else None,
    )


_RECIPE_KINDS = {
    "same": "arithmetic",
    "select": "structural",
    "pack": "structural",
    "key_product": "digit",
    "component_product": "arithmetic",
    "transform": "ntt",
}


@dataclass(frozen=True)
class OperationFacts:
    """Evaluated rules bound to current values in one read-only graph window."""

    valid: bool
    count: int = 0
    anchors: int = 0
    parameter: SSAValue | None = None
    known: tuple[tuple[str, object], ...] = ()
    table_devices: tuple[torch.device, ...] = ()
    numeric_devices: tuple[torch.device, ...] = ()


class OperationAnalysis:
    """Share immutable current-value analysis across probes and builtin matchers.

    Recipes contain operand indices. Evaluated facts, layout memoization and
    compact-value uses belong to this read-only window and are discarded before
    graph rewriting. Region counts, parameter identity and escaping uses remain
    in each live matcher.
    """

    def __init__(
        self,
        recipe: Callable[[Operation], OperationRecipe] = operation_recipe,
    ):
        self.recipe = recipe
        self.layout = _layout_resolver()
        self.boundary = cache(lambda value: _boundary(value, self.layout))
        self.facts: dict[Operation, OperationFacts] = {}
        self.reader = RegionMatcher(include_ntt=True)
        self.reader.layout = self.layout
        self.reader.compact = cache(self._compact)

    @staticmethod
    def _compact(value: SSAValue) -> bool:
        if isinstance(value.type, ckks.CompressedPlaintextType):
            return True
        # Compressed rows are plaintexts; a component axis rules them out
        # without visiting every consumer of a widely shared ciphertext.
        if component_axis(value.type):
            return False
        for use in value.uses:
            structure = generated_structure(use.operation)
            if (
                structure is not None
                and use.index < len(structure.roles)
                and structure.roles[use.index] == "compact"
            ):
                return True
        return False

    def operation(self, operation: Operation) -> OperationFacts:
        facts = self.facts.get(operation)
        if facts is not None:
            return facts
        facts = self._facts(operation)
        self.facts[operation] = facts
        return facts

    def _facts(self, operation: Operation) -> OperationFacts:
        try:
            return self._evaluate(operation)
        except (ValueError, TypeError, IndexError):
            return OperationFacts(False)

    def _evaluate(self, operation: Operation) -> OperationFacts:
        if EXECUTION_IMPLEMENTATION_ATTRIBUTE in operation.attributes:
            return OperationFacts(False)
        if (
            not isinstance(
                operation, (core.MaterialRefOp, UnrealizedConversionCastOp)
            )
            and generated_structure(operation) is None
        ):
            return OperationFacts(False)
        if isinstance(operation, core.MaterialRefOp):
            return OperationFacts(True)
        recipe = self.recipe(operation)
        if recipe.kind == "reject":
            return OperationFacts(False)
        self.reader.reset()
        table_devices: list[torch.device] = []
        numeric_devices: list[torch.device] = []
        if not self.reader._check_operation_uncached(
            operation, recipe=recipe, devices=(table_devices, numeric_devices)
        ):
            return OperationFacts(False)
        return OperationFacts(
            True,
            self.reader.count,
            self.reader.anchors,
            self.reader.parameter,
            tuple(self.reader.known.items()),
            tuple(dict.fromkeys(table_devices)),
            tuple(dict.fromkeys(numeric_devices)),
        )


class RegionMatcher:
    """Analyze growing fusion candidates during one read-only scheduling window.

    Probes update only the proposed operation and its affected escaping results,
    then roll back. Committing a candidate retains those facts for the next probe.
    Layout memoization belongs to this matcher and is discarded after scheduling.
    Rejections whose conditions cannot weaken during append are retained until
    reset; missing facts and changing escaping boundaries remain retryable.
    """

    def __init__(self, *, include_ntt: bool):
        self.include_ntt = include_ntt

        self.analysis: OperationAnalysis | None = None
        self.layout = _layout_resolver()
        self.boundary = cache(self._boundary)
        self.compact = OperationAnalysis._compact
        self.reset()

    def use_analysis(self, analysis: OperationAnalysis) -> None:
        """Share current-graph facts during one read-only scheduling window."""
        self.analysis = analysis
        self.layout = analysis.layout
        self.boundary = analysis.boundary

    def _check_facts(self, facts: OperationFacts) -> bool:
        if not facts.valid or facts.anchors and not self.include_ntt:
            self._permanent_rejection = True
            return False
        self.count += facts.count
        self.anchors += facts.anchors
        for current in facts.table_devices:
            if self.device is not None and current != self.device:
                # An unspecified index may still refine to the candidate index.
                if self.device.index is not None:
                    self._permanent_rejection = True
                return False
            self.device = current
        if facts.parameter is not None:
            self._parameter(facts.parameter)
        for field, fact in facts.known:
            if field in self.known and self.known[field] != fact:
                self._permanent_rejection = True
                return False
            self.known[field] = fact
        for current in facts.numeric_devices:
            if (
                self.device is not None
                and self.device.index is not None
                and current.index is not None
                and current.index != self.device.index
            ):
                self._permanent_rejection = True
                return False
            if self.device is None or current.index is not None:
                self.device = current
        return True

    def reset(self) -> None:
        """Start a new region while retaining immutable per-value layout facts."""
        self._rejected: set[Operation] = set()
        self._permanent_rejection = False
        self.count = 0
        self.anchors = 0
        self.device: torch.device | None = None
        self.known: dict[str, object] = {}
        self.parameter: SSAValue | None = None
        self.invalid = False
        self.bad_boundaries = 0
        self.selected: dict[Operation, bool] = {}
        self.remaining: dict[SSAValue, int] = {}
        self.extents: dict[tuple[int, ...], int] = {}
        self.batches: dict[tuple[int, ...], int] = {}
        self.undo: list[tuple[dict[Any, Any], Any, bool, Any]] | None = None

    def _set(self, mapping: dict[Any, Any], key: Any, value: Any) -> None:
        if self.undo is not None:
            self.undo.append((mapping, key, key in mapping, mapping.get(key)))
        if value is None:
            mapping.pop(key, None)
        else:
            mapping[key] = value

    def _parameter(self, value: SSAValue) -> None:
        if self.parameter is not None and self.parameter is not value:
            self._permanent_rejection = True
            self.invalid = True
        self.parameter = value

    def _boundary(self, value: SSAValue):
        return _boundary(value, self.layout)

    def _change_boundary(self, value: SSAValue, delta: int) -> None:
        extent, batch, invalid = self.boundary(value)
        self.bad_boundaries += delta * invalid
        for mapping, key in ((self.extents, extent), (self.batches, batch)):
            if key is not None:
                count = mapping.get(key, 0) + delta
                self._set(mapping, key, count if count else None)

    def _check_operation(self, operation: Operation) -> bool:
        if self.analysis is not None:
            return self._check_facts(self.analysis.operation(operation))
        return self._check_operation_uncached(operation)

    def _check_operation_uncached(
        self,
        operation: Operation,
        *,
        recipe: OperationRecipe | None = None,
        devices: tuple[list[torch.device], list[torch.device]] | None = None,
    ) -> bool:
        if EXECUTION_IMPLEMENTATION_ATTRIBUTE in operation.attributes:
            return False
        if type(operation) is UnrealizedConversionCastOp:
            if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                return False
            if component_axis(operation.inputs[0].type) != component_axis(
                operation.outputs[0].type
            ):
                # The expression emitter forwards internal casts unchanged.
                # A component reinterpretation must instead supply a typed
                # region input or consume a typed region output.
                return False
        elif isinstance(operation, core.MaterialRefOp):
            return True
        elif (structure := generated_structure(operation)) is not None and (
            self.include_ntt or structure.layout != "transform"
        ):
            if recipe is None:
                recipe = operation_recipe(operation)
            if recipe.kind == "reject":
                return False
            # Natively a tensor product takes five kernels, so one product
            # alone meets the minimum region size.
            self.count += 2 if structure.layout == "component_product" else 1
            if structure.layout == "transform":
                self.anchors += 1
        else:
            return False
        # Table placement is part of the numerical ABI as well. Do not
        # infer a CUDA table from the placement of the polynomial payload.
        for operand in operation.operands:
            table_device = _state(operand).get("device")
            if (
                isinstance(table_device, StringAttr)
                and table_device.data != "unknown"
            ):
                current = torch.device(table_device.data)
                if devices is not None:
                    devices[0].append(current)
                if current.type != "cuda" or (
                    self.device is not None and current != self.device
                ):
                    return False
                self.device = current
        numeric_values = (
            tuple(operation.operands[index] for index in recipe.numeric_indices)
            if recipe is not None
            else operation.operands
        )
        if recipe is not None and recipe.parameter_index is not None:
            self._parameter(_outer(operation.operands[recipe.parameter_index]))
        for value in (*numeric_values, *operation.results):
            type_facts = _type_facts(value.type)
            if type_facts is None:
                return False
            for field, fact in type_facts.facts:
                if field in self.known and self.known[field] != fact:
                    return False
                self.known[field] = fact
            current = type_facts.device
            if current is not None:
                if devices is not None:
                    devices[1].append(current)
                if (
                    self.device is not None
                    and self.device.index is not None
                    and current.index is not None
                    and current.index != self.device.index
                ):
                    return False
                if self.device is None or current.index is not None:
                    self.device = current
            shape, _ = self.layout(value)
            if shape is not None:
                if len(shape) < 2:
                    return False
                is_compact = self.compact(value)
                extent = shape[-2:]
                if is_compact:
                    continue
                if "extent" in self.known and self.known["extent"] != extent:
                    return False
                self.known["extent"] = extent
                ids = type_facts.prime_ids
                if ids is not None and len(ids) != shape[-2]:
                    return False
        return True

    def _append(self, operation: Operation) -> None:
        if self.invalid:
            return
        self._set(self.selected, operation, True)
        if not self._check_operation(operation):
            self.invalid = True
            return
        # Each use leaving the region is removed once its consumer joins it.
        # Repeated operands are distinct uses and therefore decrement separately.
        for operand in operation.operands:
            remaining = self.remaining.get(operand, 0)
            if remaining:
                self._set(self.remaining, operand, remaining - 1)
                if remaining == 1:
                    self._change_boundary(operand, -1)
        if _transform(operation):
            self._change_boundary(operation.operands[0], 1)
        if not isinstance(operation, core.MaterialRefOp):
            for value in operation.results:
                remaining = sum(
                    use.operation not in self.selected for use in value.uses
                )
                self._set(self.remaining, value, remaining)
                if remaining:
                    self._change_boundary(value, 1)

    def _result(self) -> int | None:
        if self.invalid or self.count and self.parameter is None:
            return None
        if self.anchors and (
            self.bad_boundaries
            or len(self.extents) > 1
            or len(self.batches) > 1
            or _INT64_NTT_MISCOMPILE
            and self.known.get("dtype") == "int64"
            and self.device is not None
            and torch.cuda.get_device_capability(self.device) < (9, 0)
        ):
            return None
        return (
            self.count if self.count == 0 or self.device is not None else None
        )

    def append(self, operation: Operation) -> None:
        """Commit the operation chosen by the scheduler, including nonmatches."""
        try:
            self._append(operation)
        except (ValueError, TypeError, IndexError):
            self.invalid = True

    def probe(self, operation: Operation) -> int | None:
        """Check one extension without changing the committed region."""
        if self.invalid or operation in self._rejected:
            return None
        self._permanent_rejection = False
        saved = (
            self.count,
            self.anchors,
            self.device,
            self.known.copy(),
            self.parameter,
            self.invalid,
            self.bad_boundaries,
        )
        self.undo = []
        try:
            self.append(operation)
            result = self._result()
            if result is None and self._permanent_rejection:
                self._rejected.add(operation)
            return result
        finally:
            for mapping, key, existed, value in reversed(self.undo):
                if existed:
                    mapping[key] = value
                else:
                    mapping.pop(key, None)
            self.undo = None
            (
                self.count,
                self.anchors,
                self.device,
                self.known,
                self.parameter,
                self.invalid,
                self.bad_boundaries,
            ) = saved


def match_region(
    operations: Sequence[Operation], *, include_ntt: bool
) -> int | None:
    """Return supported computational-op count, zero for plumbing, or None."""
    matcher = RegionMatcher(include_ntt=include_ntt)
    for operation in operations:
        matcher.append(operation)
    return matcher._result()
