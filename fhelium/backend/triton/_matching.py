"""Recognize pure RNS expressions and NTT dependencies from Program facts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
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
from fhelium.ir.dialects import ckks, core, fusion, rns

from ._expressions import (
    NTT_OPS,
    ARITHMETIC_OPS,
    SUPPORTED_OPERATIONS,
    component_axis,
)
from ._layout import (
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
        if isinstance(parent, ckks.RepeatOp) and value.index:
            return _outer(
                (*parent.shared, *parent.row(0), *parent.carried)[
                    value.index - 1
                ]
            )
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
        if isinstance(owner, SUPPORTED_OPERATIONS + NTT_OPS):
            operands = (
                owner.operands[:1]
                if isinstance(owner, NTT_OPS + (rns.KeySwitchDigitProductOp,))
                else owner.operands[:-1]
                if isinstance(owner, ARITHMETIC_OPS)
                else owner.operands
            )
            facts = [layout(v) for v in operands]
            return result_layout(
                type(owner),
                [f[0] for f in facts],
                [f[1] for f in facts],
                component=(
                    int(owner.component.value.data)
                    if isinstance(owner, rns.ExtractComponentOp)
                    else None
                ),
            )
        return _integers(_state(value).get("shape")), component_axis(value.type)

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
    dependencies = operation_dependencies(operation)
    if isinstance(operation, NTT_OPS):
        payload_indices = (0,)
    elif isinstance(operation, rns.KeySwitchDigitProductOp):
        payload_indices = (0, 1)
    elif isinstance(operation, ARITHMETIC_OPS):
        payload_indices = tuple(range(len(operation.operands) - 1))
    else:
        payload_indices = tuple(range(len(operation.operands)))
    for result_index in range(len(operation.results)):
        for operand_index in payload_indices:
            position = dependencies.kind(
                result_index, operand_index, "coefficient"
            )
            limb = dependencies.kind(result_index, operand_index, "limb")
            if isinstance(operation, NTT_OPS):
                allowed_positions = ("mixing",)
            elif (
                isinstance(
                    operation,
                    (
                        ckks.AddCompressedPlaintextOp,
                        ckks.MultiplyCompressedPlaintextOp,
                    ),
                )
                and operand_index > 0
            ):
                allowed_positions = ("element", "reindexed")
            else:
                allowed_positions = ("element",)
            allowed_limbs = (
                ("element", "reindexed")
                if isinstance(operation, rns.KeySwitchDigitProductOp)
                and operand_index == 1
                else ("element",)
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
    if isinstance(
        operation,
        (ckks.AddCompressedPlaintextOp, ckks.MultiplyCompressedPlaintextOp),
    ):
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
    if not isinstance(operation, SUPPORTED_OPERATIONS + NTT_OPS):
        return OperationRecipe("reject")
    if not _supported_dependencies(operation):
        return OperationRecipe("reject")
    if isinstance(operation, NTT_OPS):
        return OperationRecipe("ntt", (0,), 1)
    if isinstance(operation, rns.KeySwitchDigitProductOp):
        return OperationRecipe("digit", (0,), len(operation.operands) - 1)
    if isinstance(operation, ARITHMETIC_OPS):
        return OperationRecipe(
            "arithmetic",
            tuple(range(len(operation.operands) - 1)),
            len(operation.operands) - 1,
        )
    return OperationRecipe("structural", tuple(range(len(operation.operands))))


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
        stats: dict | None = None,
    ):
        self.recipe = recipe
        self.stats = stats
        self.layout = _layout_resolver()
        self.boundary = cache(lambda value: _boundary(value, self.layout))
        self.facts: dict[Operation, OperationFacts] = {}
        self.reader = RegionMatcher(include_ntt=True)
        self.reader.layout = self.layout
        self.reader.compact = cache(self._compact)

    @staticmethod
    def _compact(value: SSAValue) -> bool:
        return isinstance(value.type, ckks.CompressedPlaintextType) or any(
            isinstance(
                use.operation,
                (
                    ckks.AddCompressedPlaintextOp,
                    ckks.MultiplyCompressedPlaintextOp,
                ),
            )
            and use.index in (1, 2)
            and use.index < len(use.operation.operands) - 1
            for use in value.uses
        )

    def operation(self, operation: Operation) -> OperationFacts:
        facts = self.facts.get(operation)
        if facts is not None:
            if self.stats is not None:
                self.stats["facts_hits"] += 1
            return facts
        if self.stats is not None:
            self.stats["facts_misses"] += 1
        try:
            facts = self._evaluate(operation)
        except (ValueError, TypeError, IndexError):
            facts = OperationFacts(False)
        self.facts[operation] = facts
        return facts

    def _evaluate(self, operation: Operation) -> OperationFacts:
        if EXECUTION_IMPLEMENTATION_ATTRIBUTE in operation.attributes:
            return OperationFacts(False)
        if not isinstance(
            operation,
            SUPPORTED_OPERATIONS
            + NTT_OPS
            + (core.MaterialRefOp, UnrealizedConversionCastOp),
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
        self.supported = SUPPORTED_OPERATIONS + (NTT_OPS if include_ntt else ())

        self.analysis: OperationAnalysis | None = None
        self.layout = _layout_resolver()
        self.boundary = cache(self._boundary)
        self.whole_transform = cache(self._whole_transform)
        self.check_mac = True
        self.compact = OperationAnalysis._compact
        self.reset()

    def use_analysis(self, analysis: OperationAnalysis) -> None:
        """Share current-graph facts during one read-only scheduling window."""
        self.analysis = analysis
        self.layout = analysis.layout
        self.boundary = analysis.boundary

    def _check_facts(self, facts: OperationFacts) -> bool:
        if (
            not facts.valid
            or facts.anchors
            and NTT_OPS[0] not in self.supported
        ):
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

    def _whole_transform(self, operation):
        from fhelium.backend.ntt._selection import select_operation_policy
        from fhelium.backend.ntt._stages import endpoint_stages

        shape, _ = self.layout(operation.operands[0])
        log_n = shape[-1].bit_length() - 1 if shape is not None else None
        policy = select_operation_policy(operation, log_n=log_n)
        stages = (
            endpoint_stages(policy, log_n, isinstance(operation, NTT_OPS[2:]))
            if policy is not None
            else None
        )
        return stages is None or stages.whole

    def _check_operation(self, operation: Operation) -> bool:
        if (
            self.check_mac
            and isinstance(operation, NTT_OPS)
            and self.whole_transform(operation)
        ):
            if self.known.get("carried_mac"):
                return False
            self.known["whole_transform"] = True
        if isinstance(operation, ckks.RepeatOp):
            from ._mac import mac_body

            if (
                operation.carried
                and mac_body(operation) is None
                or not operation.carried
                and operation.body.block.args[0].uses
            ):
                return False
            if operation.carried:
                if self.known.get("whole_transform"):
                    return False
                self.known["carried_mac"] = True
            before_count, before_anchors = self.count, self.anchors
            for op in tuple(operation.body.block.ops)[:-1]:
                if not self._check_operation(op):
                    return False
            count = operation.count.value.data - 1
            self.count += count * (self.count - before_count)
            self.anchors += count * (self.anchors - before_anchors)
            return True
        if isinstance(operation, ckks.SelectBatchOp):
            # Round-indexed rows of a MAC body become indexed loads.
            from ._mac import round_rows

            parent = operation.parent_op()
            return isinstance(parent, ckks.RepeatOp) and operation in (
                round_rows(parent) or ()
            )
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
        elif isinstance(operation, self.supported):
            if recipe is None and not _supported_dependencies(operation):
                return False
            # The generated plaintext-add expression accepts NTT operands.
            if (
                isinstance(operation, rns.AddPlaintextOp)
                and operation.polynomial_domain.data != "ntt"
            ):
                return False
            self.count += 1
            if isinstance(operation, NTT_OPS):
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
            else operation.operands[:1]
            if isinstance(operation, NTT_OPS + (rns.KeySwitchDigitProductOp,))
            else operation.operands[:-1]
            if isinstance(operation, ARITHMETIC_OPS)
            else operation.operands
        )
        if recipe is not None:
            if recipe.parameter_index is not None:
                self._parameter(
                    _outer(operation.operands[recipe.parameter_index])
                )
        elif isinstance(operation, NTT_OPS):
            self._parameter(_outer(operation.operands[1]))
        elif isinstance(operation, ARITHMETIC_OPS):
            self._parameter(_outer(operation.operands[-1]))
        elif isinstance(operation, rns.KeySwitchDigitProductOp):
            self._parameter(_outer(operation.parameters))
        for value in (*numeric_values, *operation.results):
            state = _state(value)
            for field in ("basis", "prime_ids", "ring_dimension", "dtype"):
                fact = state.get(field)
                if fact is None or fact == StringAttr("unknown"):
                    continue
                if field == "dtype" and isinstance(fact, StringAttr):
                    fact = fact.data.removeprefix("torch.")
                    if fact not in ("int32", "int64"):
                        return False
                if field == "prime_ids":
                    ids = _integers(fact)
                    if ids is not None and (
                        not ids
                        or ids != tuple(range(ids[0], ids[0] + len(ids)))
                    ):
                        return False
                if field in self.known and self.known[field] != fact:
                    return False
                self.known[field] = fact
            declared_device = state.get("device")
            if (
                isinstance(declared_device, StringAttr)
                and declared_device.data != "unknown"
            ):
                try:
                    current = torch.device(declared_device.data)
                    if devices is not None:
                        devices[1].append(current)
                except RuntimeError:
                    return False
                if current.type != "cuda" or (
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
                ids = _integers(state.get("prime_ids"))
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
        if isinstance(operation, NTT_OPS):
            self._change_boundary(operation.operands[0], 1)
        elif isinstance(operation, ckks.RepeatOp):
            for op in operation.body.block.ops:
                if isinstance(op, NTT_OPS):
                    self._change_boundary(op.operands[0], 1)
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
    matcher.check_mac = any(
        isinstance(operation, ckks.RepeatOp)
        and any(
            isinstance(nested, ckks.RepeatOp) and nested.carried
            for nested in operation.walk()
        )
        for operation in operations
    )
    for operation in operations:
        matcher.append(operation)
    return matcher._result()
