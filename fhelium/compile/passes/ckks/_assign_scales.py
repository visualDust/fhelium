"""Assign concrete actual scales to a scheduled CKKS Program."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

import math
from dataclasses import dataclass, field

from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerAttr
from xdsl.dialects.func import ReturnOp
from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import Attribute, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.config import CkksConfig
from fhelium.ir import Program, value_role
from fhelium.ir.dialects import ckks
from fhelium.ir.dialects._common import OpenStateType
from ._transition_state import is_same_dialect_ckks_cast


_PREPARE_ADD_TYPES = (
    ckks.PrepareAddMessageOp,
    ckks.PrepareAddPlaintextOp,
    ckks.PrepareAddStaticOp,
)
_PREPARE_MULTIPLY_TYPES = (
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
    ckks.ModSwitchOp,
    ckks.AddScalarOp,
    ckks.MultiplyIntegerScalarOp,
)


def _positive_scale(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return scale


def _scale_attribute(scale: float) -> FloatAttr:
    return FloatAttr(scale, Float64Type())


def _replace_scale(
    value: SSAValue,
    scale: float,
    *,
    copied_state: dict[str, Attribute] | None = None,
) -> tuple[SSAValue, bool]:
    value_type = value.type
    if not isinstance(value_type, OpenStateType):
        raise TypeError(
            f"CKKS scale cannot be represented on {value_type.name!r}"
        )
    result_type = value_type.with_state(
        {
            **({} if copied_state is None else copied_state),
            "scale": _scale_attribute(scale),
        }
    )
    if result_type.state == value_type.state:
        return value, False
    return Rewriter.replace_value_with_new_type(value, result_type), True


def _represented_scale(value: SSAValue) -> float | None:
    state = getattr(getattr(value.type, "state", None), "data", {})
    attribute = state.get("scale")
    if not isinstance(attribute, FloatAttr):
        return None
    return _positive_scale(attribute.value.data, label="represented CKKS scale")


def _represented_level(value: SSAValue, *, operation: str) -> int:
    state = getattr(getattr(value.type, "state", None), "data", {})
    attribute = state.get("level")
    if not isinstance(attribute, IntegerAttr):
        raise ValueError(
            f"{operation} requires a scheduled level before scale assignment"
        )
    return int(attribute.value.data)


@dataclass(frozen=True)
class AssignCkksScalesPass:
    """Propagate actual scales through an already scheduled CKKS Program.

    The pass never inserts arithmetic or metadata reinterpretation operations.
    Add/sub joins require exact binary64 scale equality. Rescale divides by the
    actual Q prime selected by its input level.
    """

    entry_scale: float
    name: str = field(default="assign-ckks-scales", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entry_scale",
            _positive_scale(self.entry_scale, label="entry_scale"),
        )

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        config = shared_data.get(CkksConfig)
        if not isinstance(config, CkksConfig):
            raise ValueError(
                "CKKS scale assignment requires CkksConfig in the Compile "
                "workspace"
            )
        if len(program.functions) != 1:
            raise ValueError("CKKS scale assignment requires one function")
        function = program.function("main")
        block = program.single_block("main")
        if any(
            operation.regions or operation.successors for operation in block.ops
        ):
            raise ValueError(
                "CKKS scale assignment supports a flat single-block SSA DAG"
            )

        scales: dict[SSAValue, float] = {}
        matched = transformed = 0

        def record(
            value: SSAValue,
            scale: float,
            *,
            copied_state: dict[str, Attribute] | None = None,
        ) -> SSAValue:
            nonlocal transformed
            concrete_scale = _positive_scale(scale, label="CKKS scale")
            concrete, changed = _replace_scale(
                value,
                concrete_scale,
                copied_state=copied_state,
            )
            scales[concrete] = concrete_scale
            transformed += int(changed)
            return concrete

        def require(value: SSAValue, *, operation: str) -> float:
            scale = scales.get(value)
            if scale is None:
                scale = _represented_scale(value)
                if scale is not None:
                    scales[value] = scale
            if scale is None:
                raise ValueError(
                    f"{operation} requires an assigned operand scale"
                )
            return scale

        for argument in tuple(block.args):
            if value_role(argument) == "encrypted":
                record(argument, self.entry_scale)
            else:
                represented = _represented_scale(argument)
                if represented is not None:
                    scales[argument] = represented

        if not any(
            value_role(argument) == "encrypted" for argument in block.args
        ):
            raise ValueError(
                "CKKS scale assignment requires an encrypted entry"
            )

        for operation in tuple(block.ops):
            if isinstance(operation, ReturnOp):
                continue
            if isinstance(operation, UnrealizedConversionCastOp):
                if len(operation.inputs) != 1 or len(operation.outputs) != 1:
                    raise ValueError(
                        "stateful conversion casts must be one-to-one"
                    )
                source_scale = scales.get(operation.inputs[0])
                if source_scale is None:
                    if value_role(operation.outputs[0]) == "encrypted":
                        raise ValueError(
                            "encrypted conversion cast lacks an assigned scale"
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
                    source_scale,
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

            scale: float | None = None
            if isinstance(operation, _PREPARE_ADD_TYPES):
                ciphertext_scale = require(
                    operation.ciphertext,
                    operation=operation.name,
                )
                mode = operation.scale_mode
                if mode is None or mode.data != "ciphertext_scale":
                    raise ValueError(
                        f"{operation.name} requires ciphertext_scale mode"
                    )
                scale = ciphertext_scale
            elif isinstance(operation, _PREPARE_MULTIPLY_TYPES):
                mode = operation.scale_mode
                if mode is None:
                    raise ValueError(f"{operation.name} lacks scale_mode")
                if mode.data == "default_scale":
                    scale = config.default_scale
                elif mode.data == "runtime_plaintext_scale":
                    scale = require(operation.public, operation=operation.name)
                else:
                    raise ValueError(
                        f"{operation.name} has unsupported scale mode "
                        f"{mode.data!r}"
                    )
            elif isinstance(operation, ckks.MultiplyPlaintextOp):
                scale = require(
                    operation.ciphertext,
                    operation=operation.name,
                ) * require(operation.plaintext, operation=operation.name)
            elif isinstance(operation, ckks.MultiplyOp):
                scale = require(
                    operation.lhs, operation=operation.name
                ) * require(
                    operation.rhs,
                    operation=operation.name,
                )
            elif isinstance(operation, (ckks.AddOp, ckks.SubtractOp)):
                lhs = require(operation.lhs, operation=operation.name)
                rhs = require(operation.rhs, operation=operation.name)
                if lhs != rhs:
                    raise ValueError(
                        f"{operation.name} has irreconcilable scales "
                        f"{lhs!r} and {rhs!r}"
                    )
                scale = lhs
            elif isinstance(operation, ckks.AddPlaintextOp):
                ciphertext_scale = require(
                    operation.ciphertext,
                    operation=operation.name,
                )
                plaintext_scale = require(
                    operation.plaintext,
                    operation=operation.name,
                )
                if ciphertext_scale != plaintext_scale:
                    raise ValueError(
                        f"{operation.name} has irreconcilable scales "
                        f"{ciphertext_scale!r} and {plaintext_scale!r}"
                    )
                scale = ciphertext_scale
            elif isinstance(operation, ckks.MultiplyScalarOp):
                scale = require(
                    operation.ciphertext,
                    operation=operation.name,
                ) * _positive_scale(
                    operation.scalar_scale.value.data,
                    label="scalar_scale",
                )
            elif isinstance(operation, ckks.RescaleOp):
                source = operation.value
                source_scale = require(source, operation=operation.name)
                level = _represented_level(source, operation=operation.name)
                scale = source_scale / float(config.q_moduli[level])
            elif isinstance(operation, ckks.ReinterpretScaleOp):
                scale = _positive_scale(
                    operation.scale.value.data,
                    label="reinterpreted scale",
                )
            elif isinstance(operation, _PRESERVING_TYPES):
                scale = require(operation.operands[0], operation=operation.name)
            elif any(
                value_role(result) == "encrypted"
                for result in operation.results
            ):
                raise ValueError(
                    f"CKKS scale assignment does not define {operation.name!r}"
                )

            if scale is None:
                continue
            matched += 1
            for result in tuple(operation.results):
                if isinstance(result.type, OpenStateType):
                    record(result, scale)

        previous_type = function.function_type
        function.update_function_type()
        transformed += int(function.function_type != previous_type)
        if transformed == 0:
            return PassResult.unchanged(program, matched=matched)
        return PassResult(
            program,
            PassStats(
                matched=max(matched, transformed),
                transformed=transformed,
            ),
        )


__all__ = ["AssignCkksScalesPass"]
