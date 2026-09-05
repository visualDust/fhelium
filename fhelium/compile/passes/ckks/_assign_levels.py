"""Assign CKKS levels from the rescale and modulus-switch operations in a Program."""

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


def _prime_ids(config: CkksConfig, level: int) -> tuple[int, ...]:
    if not 0 <= level < config.num_scale_primes:
        raise ValueError(f"CKKS public level {level} is unavailable")
    return tuple(range(level, config.num_q_primes))


def _level_state(config: CkksConfig, level: int) -> dict[str, Attribute]:
    return {
        "level": IntegerAttr(level, 64),
        "prime_ids": ArrayAttr(
            IntegerAttr(prime_id, 64) for prime_id in _prime_ids(config, level)
        ),
    }


def _type_with_level(
    value_type: Attribute,
    config: CkksConfig,
    level: int,
    *,
    copied_state: dict[str, Attribute] | None = None,
) -> OpenStateType:
    if not isinstance(value_type, OpenStateType):
        raise TypeError(
            f"CKKS level cannot be represented on {value_type.name!r}"
        )
    return value_type.with_state(
        {
            **({} if copied_state is None else copied_state),
            **_level_state(config, level),
        }
    )


def _replace_level(
    value: SSAValue,
    config: CkksConfig,
    level: int,
    *,
    copied_state: dict[str, Attribute] | None = None,
) -> tuple[SSAValue, bool]:
    result_type = _type_with_level(
        value.type,
        config,
        level,
        copied_state=copied_state,
    )
    if (
        isinstance(value.type, OpenStateType)
        and result_type.state == value.type.state
    ):
        return value, False
    return Rewriter.replace_value_with_new_type(value, result_type), True


def _represented_level(value: SSAValue) -> int | None:
    state = getattr(getattr(value.type, "state", None), "data", {})
    attribute = state.get("level")
    return (
        int(attribute.value.data)
        if isinstance(attribute, IntegerAttr)
        else None
    )


@dataclass(frozen=True)
class AssignCkksLevelsPass:
    """Assign levels and align add/sub joins after rescale placement.

    Existing rescale nodes advance one public level. At add/sub fan-in, only
    the lower-level consumer edge is advanced through a real ``ModSwitchOp``.
    Repeated application does not duplicate either transition.
    """

    entry_level: int
    name: str = field(default="assign-ckks-levels", init=False)

    def __post_init__(self) -> None:
        if type(self.entry_level) is not int:
            raise TypeError("entry_level must be an integer")

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        config = shared_data.get(CkksConfig)
        if not isinstance(config, CkksConfig):
            raise ValueError(
                "CKKS level assignment requires CkksConfig in the Compile "
                "workspace"
            )
        _prime_ids(config, self.entry_level)
        if len(program.functions) != 1:
            raise ValueError("CKKS level assignment requires one function")
        function = program.function("main")
        block = program.single_block("main")
        if any(
            operation.regions or operation.successors for operation in block.ops
        ):
            raise ValueError(
                "CKKS level assignment supports a flat single-block SSA DAG"
            )
        levels: dict[SSAValue, int] = {}
        matched = transformed = inserted = 0

        def record(
            value: SSAValue,
            level: int,
            *,
            copied_state: dict[str, Attribute] | None = None,
        ) -> SSAValue:
            nonlocal transformed
            concrete, changed = _replace_level(
                value,
                config,
                level,
                copied_state=copied_state,
            )
            levels[concrete] = level
            transformed += int(changed)
            return concrete

        def require(value: SSAValue, *, operation: str) -> int:
            level = levels.get(value)
            if level is None:
                level = _represented_level(value)
                if level is not None:
                    _prime_ids(config, level)
                    levels[value] = level
            if level is None:
                raise ValueError(
                    f"{operation} requires an assigned operand level"
                )
            return level

        for argument in tuple(block.args):
            if value_role(argument) == "encrypted":
                record(argument, self.entry_level)
            else:
                represented = _represented_level(argument)
                if represented is not None:
                    _prime_ids(config, represented)
                    levels[argument] = represented
        if not any(
            value_role(argument) == "encrypted" for argument in block.args
        ):
            raise ValueError(
                "CKKS level assignment requires an encrypted entry"
            )

        for operation in tuple(block.ops):
            if isinstance(operation, ReturnOp):
                continue
            if isinstance(operation, UnrealizedConversionCastOp):
                if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                    raise ValueError(
                        "stateful conversion casts must be one-to-one"
                    )
                source_level = levels.get(operation.inputs[0])
                if source_level is None:
                    if value_role(operation.outputs[0]) == "encrypted":
                        raise ValueError(
                            "encrypted conversion cast lacks an assigned level"
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
                    source_level,
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

            level: int | None = None
            if isinstance(operation, _PREPARATION_TYPES):
                level = require(operation.ciphertext, operation=operation.name)
            elif isinstance(operation, (ckks.AddOp, ckks.SubtractOp)):
                operand_levels = [
                    require(operand, operation=operation.name)
                    for operand in operation.operands
                ]
                target_level = max(operand_levels)
                for index, source_level in enumerate(operand_levels):
                    if source_level == target_level:
                        continue
                    source = operation.operands[index]
                    switched = ckks.ModSwitchOp(
                        source,
                        _type_with_level(
                            source.type,
                            config,
                            target_level,
                        ),
                        attributes={
                            "target_level": IntegerAttr(target_level, 64)
                        },
                    )
                    switched.result.name_hint = (
                        f"{source.name_hint or f'operand{index}'}_level"
                        f"{target_level}"
                    )
                    owner = operation.parent_block()
                    if owner is None:
                        raise ValueError(
                            "join operation is not attached to a block"
                        )
                    owner.insert_op_before(switched, operation)
                    operation.operands[index] = switched.result
                    levels[switched.result] = target_level
                    inserted += 1
                level = target_level
            elif isinstance(operation, _SAME_LEVEL_TYPES):
                operand_levels = [
                    require(operand, operation=operation.name)
                    for operand in operation.operands[:2]
                ]
                if operand_levels[0] != operand_levels[1]:
                    raise ValueError(
                        f"{operation.name} requires equal operand levels, got "
                        f"{operand_levels[0]} and {operand_levels[1]}"
                    )
                level = operand_levels[0]
            elif isinstance(operation, ckks.RescaleOp):
                source_level = require(
                    operation.value,
                    operation=operation.name,
                )
                if source_level + 1 >= config.num_scale_primes:
                    raise ValueError(
                        f"{operation.name} leaves the public modulus chain "
                        "without a bootstrapping schedule"
                    )
                level = source_level + 1
            elif isinstance(operation, ckks.ModSwitchOp):
                source_level = require(
                    operation.value, operation=operation.name
                )
                target_level = int(operation.target_level.value.data)
                if target_level < source_level:
                    raise ValueError(
                        "modulus switch cannot restore dropped Q primes"
                    )
                _prime_ids(config, target_level)
                level = target_level
            elif isinstance(operation, _PRESERVING_TYPES):
                level = require(operation.operands[0], operation=operation.name)
            elif any(
                value_role(result) == "encrypted"
                for result in operation.results
            ):
                raise ValueError(
                    f"CKKS level assignment does not define {operation.name!r}"
                )

            if level is None:
                continue
            matched += 1
            for result in tuple(operation.results):
                if isinstance(result.type, OpenStateType):
                    record(result, level)

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


__all__ = ["AssignCkksLevelsPass"]
