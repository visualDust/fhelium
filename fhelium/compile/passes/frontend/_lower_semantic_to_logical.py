"""Lower captured semantic operations to role-explicit logical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass
from typing import cast
import json

from xdsl.dialects.builtin import UnrealizedConversionCastOp, IntegerAttr
from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import Rewriter


from fhelium.ir.dialects import logical, semantic
from fhelium.ir.dialects import torch as torch_dialect
from .._operation_transforms import (
    cast_before,
    display_name,
    operand_role,
    program_operations,
    replacement_operation,
    result_role,
)

_BINARY_LOGICAL_TYPES: dict[
    tuple[type[Operation], tuple[str, str]], type[Operation]
] = {
    (
        semantic.AddOp,
        ("encrypted", "encrypted"),
    ): logical.AddEncryptedEncryptedOp,
    (semantic.AddOp, ("encrypted", "public")): logical.AddEncryptedPublicOp,
    (semantic.AddOp, ("public", "encrypted")): logical.AddPublicEncryptedOp,
    (
        semantic.SubtractOp,
        ("encrypted", "encrypted"),
    ): logical.SubtractEncryptedEncryptedOp,
    (
        semantic.SubtractOp,
        ("encrypted", "public"),
    ): logical.SubtractEncryptedPublicOp,
    (
        semantic.SubtractOp,
        ("public", "encrypted"),
    ): logical.SubtractPublicEncryptedOp,
    (
        semantic.MultiplyOp,
        ("encrypted", "encrypted"),
    ): logical.MultiplyEncryptedEncryptedOp,
    (
        semantic.MultiplyOp,
        ("encrypted", "public"),
    ): logical.MultiplyEncryptedPublicOp,
    (
        semantic.MultiplyOp,
        ("public", "encrypted"),
    ): logical.MultiplyPublicEncryptedOp,
}


def _logical_type(
    value: SSAValue, role: str
) -> logical.EncryptedType | logical.PublicType:
    state = dict(getattr(getattr(value.type, "state", None), "data", {}))
    return (
        logical.EncryptedType().with_state(state)
        if role == "encrypted"
        else logical.PublicType().with_state(state)
    )


def _replace_with_result_cast(
    operation: Operation, replacement: Operation
) -> None:
    cast, result = UnrealizedConversionCastOp.cast_one(
        replacement.results[0], operation.results[0].type
    )
    result.name_hint = operation.results[0].name_hint
    Rewriter.replace_op(
        operation,
        (replacement, cast),
        new_results=(result,),
    )


@dataclass(frozen=True)
class LowerSemanticToLogicalPass:
    """Classify registered encrypted semantic operations by operand roles.

    Other dialects, public-only arithmetic, malformed operations, and unknown
    value roles remain structurally intact so partial mixed-level Programs stay
    valid inputs and outputs of the pass.
    """

    name: str = "lower-semantic-to-logical"

    def run(self, compilation: "Compilation") -> PassResult:
        """Lower matching operation classes and report unchanged candidates."""
        program = compilation.program
        workspace = compilation.workspace

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        for operation in program_operations(program):
            if result_role(operation) != "encrypted":
                targets = {
                    semantic.AddOp: "torch.add",
                    semantic.SubtractOp: "torch.sub",
                    semantic.MultiplyOp: "torch.mul",
                    semantic.NegateOp: "torch.neg",
                    semantic.RollOp: "torch.roll",
                }
                target = targets.get(type(operation))
                if target is None:
                    continue
                arguments = [
                    {"kind": "ssa", "operand": index}
                    for index in range(len(operation.operands))
                ]
                kwargs = []
                if isinstance(operation, semantic.RollOp):
                    kwargs = [
                        [
                            "shifts",
                            {
                                "kind": "literal",
                                "value": int(
                                    cast(
                                        IntegerAttr, operation.shift
                                    ).value.data
                                ),
                            },
                        ],
                        [
                            "dims",
                            {
                                "kind": "literal",
                                "value": int(operation.dimension.value.data)
                                if operation.dimension is not None
                                else None,
                            },
                        ],
                    ]
                descriptor = {
                    "args": {"kind": "tuple", "items": arguments},
                    "kwargs": {"kind": "mapping", "entries": kwargs},
                }
                replacement = torch_dialect.TensorCallOp(
                    tuple(operation.operands),
                    operation.results[0].type,
                    kind="function",
                    target=target,
                    argument_descriptor=json.dumps(descriptor),
                    role="message",
                )
                Rewriter.replace_op(operation, replacement)
                matched += 1
                transformed += 1
                inserted += 1
                continue
            if isinstance(operation, (semantic.NegateOp, semantic.RollOp)):
                matched += 1
                if (
                    len(operation.operands) != 1
                    or operand_role(operation.operands[0]) != "encrypted"
                ):
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: semantic unary operation "
                        "requires one encrypted operand"
                    )
                    continue
                operation_type = (
                    logical.NegateEncryptedOp
                    if isinstance(operation, semantic.NegateOp)
                    else logical.RollEncryptedOp
                )
                operand = operation.operands[0]
                typed_operand = cast_before(
                    operation,
                    operand,
                    _logical_type(operand, "encrypted"),
                    name_hint=f"{display_name(operation)}_logical_operand",
                )
                replacement = replacement_operation(
                    operation,
                    operation_type,
                    operands=(typed_operand,),
                    result_types=(logical.EncryptedType(),),
                )
                _replace_with_result_cast(operation, replacement)
                transformed += 1
                inserted += 1 + (typed_operand is not operand)
                continue
            if not isinstance(
                operation,
                (semantic.AddOp, semantic.SubtractOp, semantic.MultiplyOp),
            ):
                continue
            matched += 1
            if len(operation.operands) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: semantic binary operation "
                    "requires exactly two operands"
                )
                continue
            roles = tuple(operand_role(value) for value in operation.operands)
            classes = tuple(
                "encrypted" if role == "encrypted" else "public"
                for role in roles
                if role in {"encrypted", "message", "plaintext", "static"}
            )
            if len(classes) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: semantic operation has "
                    "unknown operand roles"
                )
                continue
            operation_type = _BINARY_LOGICAL_TYPES.get(
                (type(operation), classes)  # type: ignore[arg-type]
            )
            if operation_type is None:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: semantic operation requires "
                    "at least one encrypted operand"
                )
                continue
            original_operands = tuple(operation.operands)
            typed_operands = tuple(
                cast_before(
                    operation,
                    operand,
                    _logical_type(operand, role),
                    name_hint=(
                        f"{display_name(operation)}_logical_operand{index}"
                    ),
                )
                for index, (operand, role) in enumerate(
                    zip(original_operands, classes, strict=True)
                )
            )
            replacement = replacement_operation(
                operation,
                operation_type,
                operands=typed_operands,
                result_types=(logical.EncryptedType(),),
            )
            _replace_with_result_cast(operation, replacement)
            transformed += 1
            inserted += 1 + sum(
                typed is not original
                for typed, original in zip(
                    typed_operands, original_operands, strict=True
                )
            )
        if transformed == 0:
            return PassResult.unchanged(
                program,
                matched=matched,
                skipped=skipped,
                diagnostics=tuple(diagnostics),
            )
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                inserted=inserted,
                skipped=skipped,
            ),
            tuple(diagnostics),
        )
