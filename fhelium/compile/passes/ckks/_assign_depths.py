"""Assign CKKS depths from the rescale and modulus-switch operations in a Program."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass, field

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.config import CkksConfig
from fhelium.ir import Program, value_role
from fhelium.ir.dialects import ckks
from fhelium.ir.dialects._common import OpenStateType
from ._transition_state import is_same_dialect_ckks_cast


_PREPARATION_TYPES = (
    ckks.PrepareAddMessageOp,
    ckks.PrepareAddPlaintextOp,
    ckks.PrepareAddStaticOp,
    ckks.PrepareMultiplyMessageOp,
    ckks.PrepareMultiplyPlaintextOp,
    ckks.PrepareMultiplyStaticOp,
)
_PRESERVING_TYPES = (
    ckks.NegateOp,
    ckks.RotateOp,
    ckks.RotateManyOp,
    ckks.ToNttOp,
    ckks.FromNttOp,
    ckks.ToMontgomeryResiduesOp,
    ckks.ToStandardResiduesOp,
    ckks.RelinearizeOp,
    ckks.SwitchKeyOp,
    ckks.ConjugateOp,
    ckks.ReinterpretScaleOp,
    ckks.AddScalarOp,
    ckks.MultiplyScalarOp,
    ckks.MultiplyIntegerScalarOp,
)
_SAME_LEVEL_TYPES = (
    ckks.MultiplyOp,
    ckks.MultiplyPlaintextOp,
    ckks.AddPlaintextOp,
)


def _prime_ids(config: CkksConfig, depth: int) -> tuple[int, ...]:
    if not 0 <= depth <= config.max_depth:
        raise ValueError(f"CKKS public depth {depth} is unavailable")
    start = sum(len(group) for group in config.q_depth_groups[:depth])
    return tuple(range(start, config.num_q_primes))


def _depth_state(config: CkksConfig, depth: int) -> dict[str, Attribute]:
    return {
        "depth": IntegerAttr(depth, 64),
        "prime_ids": ArrayAttr(
            IntegerAttr(prime_id, 64) for prime_id in _prime_ids(config, depth)
        ),
    }


def _type_with_depth(
    value_type: Attribute,
    config: CkksConfig,
    depth: int,
    *,
    copied_state: dict[str, Attribute] | None = None,
) -> OpenStateType:
    if not isinstance(value_type, OpenStateType):
        raise TypeError(
            f"CKKS depth cannot be represented on {value_type.name!r}"
        )
    return value_type.with_state(
        {
            **({} if copied_state is None else copied_state),
            **_depth_state(config, depth),
        }
    )


def _replace_depth(
    value: SSAValue,
    config: CkksConfig,
    depth: int,
    *,
    copied_state: dict[str, Attribute] | None = None,
) -> tuple[SSAValue, bool]:
    result_type = _type_with_depth(
        value.type,
        config,
        depth,
        copied_state=copied_state,
    )
    if (
        isinstance(value.type, OpenStateType)
        and result_type.state == value.type.state
    ):
        return value, False
    return Rewriter.replace_value_with_new_type(value, result_type), True


def _represented_depth(value: SSAValue) -> int | None:
    state = getattr(getattr(value.type, "state", None), "data", {})
    attribute = state.get("depth")
    return (
        int(attribute.value.data)
        if isinstance(attribute, IntegerAttr)
        else None
    )


@dataclass(frozen=True)
class AssignCkksDepthsPass:
    """Assign depths and align add/sub joins after rescale placement.

    Existing rescale nodes advance one public depth. At add/sub fan-in, only
    the lower-depth consumer edge is advanced through a real ``ModSwitchOp``.
    Repeated application does not duplicate either transition.
    """

    entry_depth: int
    name: str = field(default="assign-ckks-depths", init=False)

    def __post_init__(self) -> None:
        if type(self.entry_depth) is not int:
            raise TypeError("entry_depth must be an integer")

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        config = shared_data.get(CkksConfig)
        if not isinstance(config, CkksConfig):
            raise ValueError(
                "CKKS depth assignment requires CkksConfig in the Compile "
                "workspace"
            )
        _prime_ids(config, self.entry_depth)
        if len(program.functions) != 1:
            raise ValueError("CKKS depth assignment requires one function")
        function = program.function("main")
        block = program.single_block("main")
        if any(
            operation.regions or operation.successors for operation in block.ops
        ):
            raise ValueError(
                "CKKS depth assignment supports a flat single-block SSA DAG"
            )
        depths: dict[SSAValue, int] = {}
        matched = transformed = inserted = 0

        def record(
            value: SSAValue,
            depth: int,
            *,
            copied_state: dict[str, Attribute] | None = None,
        ) -> SSAValue:
            nonlocal transformed
            concrete, changed = _replace_depth(
                value,
                config,
                depth,
                copied_state=copied_state,
            )
            depths[concrete] = depth
            transformed += int(changed)
            return concrete

        def require(value: SSAValue, *, operation: str) -> int:
            depth = depths.get(value)
            if depth is None:
                depth = _represented_depth(value)
                if depth is not None:
                    _prime_ids(config, depth)
                    depths[value] = depth
            if depth is None:
                raise ValueError(
                    f"{operation} requires an assigned operand depth"
                )
            return depth

        for argument in tuple(block.args):
            if value_role(argument) == "encrypted":
                record(argument, self.entry_depth)
            else:
                represented = _represented_depth(argument)
                if represented is not None:
                    _prime_ids(config, represented)
                    depths[argument] = represented
        if not any(
            value_role(argument) == "encrypted" for argument in block.args
        ):
            raise ValueError(
                "CKKS depth assignment requires an encrypted entry"
            )

        for operation in tuple(block.ops):
            if isinstance(operation, ReturnOp):
                continue
            if isinstance(operation, UnrealizedConversionCastOp):
                if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                    raise ValueError(
                        "stateful conversion casts must be one-to-one"
                    )
                source_depth = depths.get(operation.inputs[0])
                if source_depth is None:
                    if value_role(operation.outputs[0]) == "encrypted":
                        raise ValueError(
                            "encrypted conversion cast lacks an assigned depth"
                        )
                    continue
                source_state = dict(
                    getattr(
                        getattr(operation.inputs[0].type, "state", None),
                        "data",
                        {},
                    )
                )
                target_state = dict(
                    getattr(
                        getattr(operation.outputs[0].type, "state", None),
                        "data",
                        {},
                    )
                )
                record(
                    operation.outputs[0],
                    source_depth,
                    copied_state=(
                        target_state
                        if is_same_dialect_ckks_cast(operation)
                        else {**source_state, **target_state}
                    ),
                )
                matched += 1
                continue
            if not operation.results:
                continue

            depth: int | None = None
            if isinstance(operation, _PREPARATION_TYPES):
                depth = require(operation.ciphertext, operation=operation.name)
            elif isinstance(operation, (ckks.AddOp, ckks.SubtractOp)):
                operand_depths = [
                    require(operand, operation=operation.name)
                    for operand in operation.operands
                ]
                target_depth = max(operand_depths)
                for index, source_depth in enumerate(operand_depths):
                    if source_depth == target_depth:
                        continue
                    source = operation.operands[index]
                    switched = ckks.ModSwitchOp(
                        source,
                        _type_with_depth(
                            source.type,
                            config,
                            target_depth,
                        ),
                        attributes={
                            "target_depth": IntegerAttr(target_depth, 64)
                        },
                    )
                    switched.result.name_hint = (
                        f"{source.name_hint or f'operand{index}'}_depth"
                        f"{target_depth}"
                    )
                    owner = operation.parent_block()
                    if owner is None:
                        raise ValueError(
                            "join operation is not attached to a block"
                        )
                    owner.insert_op_before(switched, operation)
                    operation.operands[index] = switched.result
                    depths[switched.result] = target_depth
                    inserted += 1
                depth = target_depth
            elif isinstance(operation, _SAME_LEVEL_TYPES):
                operand_depths = [
                    require(operand, operation=operation.name)
                    for operand in operation.operands[:2]
                ]
                if operand_depths[0] != operand_depths[1]:
                    raise ValueError(
                        f"{operation.name} requires equal operand depths, got "
                        f"{operand_depths[0]} and {operand_depths[1]}"
                    )
                depth = operand_depths[0]
            elif isinstance(operation, ckks.GroupedRotationWeightedSumOp):
                depth = require(operation.value, operation=operation.name)
                plaintext_depths = {
                    require(plaintext, operation=operation.name)
                    for plaintext in operation.plaintexts
                }
                if plaintext_depths != {depth}:
                    raise ValueError(
                        f"{operation.name} requires one ciphertext/plaintext depth"
                    )
            elif isinstance(operation, ckks.RescaleOp):
                source_depth = require(
                    operation.value,
                    operation=operation.name,
                )
                if source_depth >= config.max_depth:
                    raise ValueError(
                        f"{operation.name} leaves the public modulus chain "
                        "without a bootstrapping schedule"
                    )
                depth = source_depth + 1
            elif isinstance(operation, ckks.ModSwitchOp):
                source_depth = require(
                    operation.value, operation=operation.name
                )
                target_depth = int(operation.target_depth.value.data)
                if target_depth < source_depth:
                    raise ValueError(
                        "modulus switch cannot restore dropped Q primes"
                    )
                _prime_ids(config, target_depth)
                depth = target_depth
            elif isinstance(operation, _PRESERVING_TYPES):
                depth = require(operation.operands[0], operation=operation.name)
            elif any(
                value_role(result) == "encrypted"
                for result in operation.results
            ):
                raise ValueError(
                    f"CKKS depth assignment does not define {operation.name!r}"
                )

            if depth is None:
                continue
            matched += 1
            for result in tuple(operation.results):
                if isinstance(result.type, OpenStateType):
                    record(result, depth)

        previous_type = function.function_type
        function.update_function_type()
        transformed += int(function.function_type != previous_type)
        if transformed == 0 and inserted == 0:
            return PassResult.unchanged(program, matched=matched)
        return PassResult(
            program,
            PassStats(
                matched=max(matched, transformed),
                transformed=transformed,
                inserted=inserted,
            ),
        )


__all__ = ["AssignCkksDepthsPass"]
