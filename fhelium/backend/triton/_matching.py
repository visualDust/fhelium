"""Recognize pure RNS expressions and NTT dependencies from Program facts."""

from __future__ import annotations

from collections.abc import Sequence
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


class RegionMatcher:
    """Analyze growing fusion candidates during one read-only scheduling window.

    Probes update only the proposed operation and its affected escaping results,
    then roll back. Committing a candidate retains those facts for the next probe.
    Layout memoization belongs to this matcher and is discarded after scheduling.
    """

    def __init__(self, *, include_ntt: bool):
        self.supported = SUPPORTED_OPERATIONS + (NTT_OPS if include_ntt else ())

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
                    if isinstance(
                        owner, NTT_OPS + (rns.KeySwitchDigitProductOp,)
                    )
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
            return _integers(_state(value).get("shape")), component_axis(
                value.type
            )

        self.layout = layout
        self.boundary = cache(self._boundary)
        self.reset()

    def reset(self) -> None:
        """Start a new region while retaining immutable per-value layout facts."""
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
            self.invalid = True
        self.parameter = value

    def _boundary(self, value: SSAValue):
        try:
            shape, axis = self.layout(value)
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

    def _change_boundary(self, value: SSAValue, delta: int) -> None:
        extent, batch, invalid = self.boundary(value)
        self.bad_boundaries += delta * invalid
        for mapping, key in ((self.extents, extent), (self.batches, batch)):
            if key is not None:
                count = mapping.get(key, 0) + delta
                self._set(mapping, key, count if count else None)

    def _check_operation(self, operation: Operation) -> bool:
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
                    limb = dependencies.kind(
                        result_index, operand_index, "limb"
                    )
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
                    if (
                        position not in allowed_positions
                        or limb not in allowed_limbs
                    ):
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
                if current.type != "cuda" or (
                    self.device is not None and current != self.device
                ):
                    return False
                self.device = current
        numeric_values = (
            operation.operands[:1]
            if isinstance(operation, NTT_OPS + (rns.KeySwitchDigitProductOp,))
            else operation.operands[:-1]
            if isinstance(operation, ARITHMETIC_OPS)
            else operation.operands
        )
        if isinstance(operation, NTT_OPS):
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
                is_compact = isinstance(
                    value.type, ckks.CompressedPlaintextType
                ) or any(
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
            return self._result()
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
