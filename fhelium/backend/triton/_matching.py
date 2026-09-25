"""Match known Program facts against the generated RNS/NTT implementation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import BlockArgument, Operation, SSAValue

from fhelium.ir import DEFAULT_OPERATION_SPECS, operation_dependencies
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
    compact_policy,
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


def match_region(
    operations: Sequence[Operation], *, include_ntt: bool
) -> int | None:
    """Return supported computational-op count, zero for plumbing, or None.

    Resource/layout binding supplies missing facts. Known contradictions reject
    the candidate before Compile selects it.
    """
    supported = SUPPORTED_OPERATIONS + (NTT_OPS if include_ntt else ())
    count = 0
    roles: dict[object, set[object]] = {}
    known: dict[str, object] = {}
    device: torch.device | None = None
    anchors: list[Operation] = []
    selected = set(operations)

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

    try:
        for operation in operations:
            if type(operation) is UnrealizedConversionCastOp:
                if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                    return None
                if component_axis(operation.inputs[0].type) != component_axis(
                    operation.outputs[0].type
                ):
                    # The expression emitter forwards internal casts unchanged.
                    # A component reinterpretation must instead supply a typed
                    # region input or consume a typed region output.
                    return None
            elif isinstance(operation, core.MaterialRefOp):
                continue
            elif isinstance(operation, supported):
                spec = DEFAULT_OPERATION_SPECS.get(operation.name)
                if (
                    spec is None
                    or spec.operation_type is not type(operation)
                    or spec.effect != "pure"
                ):
                    return None
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
                            if isinstance(
                                operation, rns.KeySwitchDigitProductOp
                            )
                            and operand_index == 1
                            else ("element",)
                        )
                        if (
                            position not in allowed_positions
                            or limb not in allowed_limbs
                        ):
                            return None
                # The generated plaintext-add expression accepts NTT operands.
                if (
                    isinstance(operation, rns.AddPlaintextOp)
                    and operation.polynomial_domain.data != "ntt"
                ):
                    return None
                count += 1
                if isinstance(operation, NTT_OPS):
                    declared = operation.attributes.get("ntt_backend")
                    if (
                        isinstance(declared, StringAttr)
                        and declared.data != "unknown"
                        and not compact_policy(declared.data)
                    ):
                        return None
                    anchors.append(operation)
            else:
                return None
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
                        device is not None and current != device
                    ):
                        return None
                    device = current
            numeric_values = (
                operation.operands[:1]
                if isinstance(
                    operation, NTT_OPS + (rns.KeySwitchDigitProductOp,)
                )
                else operation.operands[:-1]
                if isinstance(operation, ARITHMETIC_OPS)
                else operation.operands
            )
            if isinstance(operation, NTT_OPS):
                roles.setdefault("rns_parameters", set()).add(
                    _outer(operation.operands[1])
                )
            elif isinstance(operation, ARITHMETIC_OPS):
                roles.setdefault("rns_parameters", set()).add(
                    _outer(operation.operands[-1])
                )
            elif isinstance(operation, rns.KeySwitchDigitProductOp):
                roles.setdefault("rns_parameters", set()).add(
                    _outer(operation.parameters)
                )
            for value in (*numeric_values, *operation.results):
                state = _state(value)
                for field in ("basis", "prime_ids", "ring_dimension", "dtype"):
                    fact = state.get(field)
                    if fact is None or fact == StringAttr("unknown"):
                        continue
                    if field == "dtype" and isinstance(fact, StringAttr):
                        fact = fact.data.removeprefix("torch.")
                        if fact not in ("int32", "int64"):
                            return None
                    if field == "prime_ids":
                        ids = _integers(fact)
                        if ids is not None and (
                            not ids
                            or ids != tuple(range(ids[0], ids[0] + len(ids)))
                        ):
                            return None
                    if field in known and known[field] != fact:
                        return None
                    known[field] = fact
                declared_device = state.get("device")
                if (
                    isinstance(declared_device, StringAttr)
                    and declared_device.data != "unknown"
                ):
                    try:
                        current = torch.device(declared_device.data)
                    except RuntimeError:
                        return None
                    if current.type != "cuda" or (
                        device is not None
                        and device.index is not None
                        and current.index is not None
                        and current.index != device.index
                    ):
                        return None
                    if device is None or current.index is not None:
                        device = current
                shape, _ = layout(value)
                if shape is not None:
                    if len(shape) < 2:
                        return None
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
                    if "extent" in known and known["extent"] != extent:
                        return None
                    known["extent"] = extent
                    ids = _integers(state.get("prime_ids"))
                    if ids is not None and len(ids) != shape[-2]:
                        return None
        if any(len(symbols) != 1 for symbols in roles.values()):
            return None
        if count and not roles:
            return None
        if anchors:
            # As in concrete NttPlan binding, only escaping results (plus the
            # transform sources) constrain the common transform batch layout.
            outputs = [
                value
                for operation in operations
                for value in operation.results
                if not isinstance(operation, core.MaterialRefOp)
                and any(use.operation not in selected for use in value.uses)
            ]
            ntt_values = [op.operands[0] for op in anchors] + outputs
            check_ntt_layouts([layout(value) for value in ntt_values])
            batches: set[tuple[int, ...]] = set()
            for value in ntt_values:
                shape, axis = layout(value)
                state = _state(value)
                if shape is None:
                    dimension = state.get("ring_dimension")
                    check_ntt_extent(
                        int(dimension.value.data)
                        if isinstance(dimension, IntegerAttr)
                        else None
                    )
                    batch = _integers(state.get("batch_shape"))
                else:
                    batch = shape[1:-2] if axis else shape[:-2]
                if batch is not None:
                    batches.add(batch)
            if len(batches) > 1:
                return None
        return count if count == 0 or device is not None else None
    except (ValueError, TypeError, IndexError):
        # These failures describe known input/shape/policy contradictions. The
        # execution binder uses the same layout constraints for concrete data.
        return None
