"""Insert operation-specific plaintext preparation operations."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr, UnrealizedConversionCastOp

from fhelium.ir import Program

from fhelium.ir.dialects import ckks, logical
from .._operation_transforms import (
    cast_before,
    ciphertext_type,
    display_name,
    operand_role,
    program_operations,
)

_MIXED_LOGICAL: dict[type[object], tuple[str, int, int]] = {
    logical.AddEncryptedPublicOp: ("add", 0, 1),
    logical.AddPublicEncryptedOp: ("add", 1, 0),
    logical.SubtractEncryptedPublicOp: ("subtract", 0, 1),
    logical.SubtractPublicEncryptedOp: ("subtract", 1, 0),
    logical.MultiplyEncryptedPublicOp: ("multiply", 0, 1),
    logical.MultiplyPublicEncryptedOp: ("multiply", 1, 0),
}
_PREPARE_TYPES: dict[tuple[str, str], type[object]] = {
    ("add", "message"): ckks.PrepareAddMessageOp,
    ("add", "plaintext"): ckks.PrepareAddPlaintextOp,
    ("add", "static"): ckks.PrepareAddStaticOp,
    ("multiply", "message"): ckks.PrepareMultiplyMessageOp,
    ("multiply", "plaintext"): ckks.PrepareMultiplyPlaintextOp,
    ("multiply", "static"): ckks.PrepareMultiplyStaticOp,
}


@dataclass(frozen=True)
class InsertPlaintextPreparationPass:
    """Insert typed plaintext preparation at registered mixed logical ops.

    Existing CKKS plaintext operands make repeated application a legal no-op.
    The visible unrealized cast on the ciphertext edge records the still-open
    logical-to-CKKS type conversion without mutating its producer.
    """

    name: str = "insert-plaintext-preparation"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Prepare supported public operands across all function blocks."""

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        for operation in program_operations(program):
            match = _MIXED_LOGICAL.get(type(operation))
            if match is None:
                continue
            arithmetic, encrypted_index, public_index = match
            if len(operation.operands) != 2:
                matched += 1
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: mixed logical operation "
                    "requires exactly two operands"
                )
                continue
            if isinstance(
                operation.operands[public_index].type, ckks.PlaintextType
            ):
                continue
            matched += 1
            public = operation.operands[public_index]
            public_role = operand_role(public)
            if public_role not in {"message", "plaintext", "static"}:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: public operand has an "
                    "unknown extension role"
                )
                continue
            prepare_kind = (
                "add" if arithmetic in {"add", "subtract"} else "multiply"
            )
            if prepare_kind == "add":
                scale_mode = "ciphertext_scale"
                plaintext_state = {
                    "basis": StringAttr("Q"),
                    "polynomial_domain": StringAttr("coefficient"),
                    "representation": StringAttr("rns"),
                    "residue_representation": StringAttr("standard"),
                }
            else:
                scale_mode = (
                    "runtime_plaintext_scale"
                    if public_role == "plaintext"
                    else "default_scale"
                )
                plaintext_state = {
                    "basis": StringAttr("Q"),
                    "polynomial_domain": StringAttr("ntt"),
                    "representation": StringAttr("rns"),
                    "residue_representation": StringAttr("montgomery"),
                }
            ciphertext = operation.operands[encrypted_index]
            typed_ciphertext = cast_before(
                operation,
                ciphertext,
                ciphertext_type(
                    ciphertext,
                    domain="coefficient",
                    residues="standard",
                    components=2,
                ),
                name_hint=f"{display_name(operation)}_ciphertext",
            )
            if typed_ciphertext is not ciphertext:
                inserted += 1
            prepare_type = _PREPARE_TYPES[(prepare_kind, public_role)]
            prepare = prepare_type.create(  # type: ignore[attr-defined]
                operands=(public, typed_ciphertext),
                result_types=(
                    ckks.PlaintextType().with_state(plaintext_state),
                ),
                attributes={
                    "operation": StringAttr(prepare_kind),
                    "source_role": StringAttr(public_role),
                    "scale_mode": StringAttr(scale_mode),
                },
            )
            prepare.results[
                0
            ].name_hint = f"{display_name(operation)}_plaintext"
            block = operation.parent_block()
            if block is None:
                raise ValueError("logical operation is not attached to a block")
            block.insert_op_before(prepare, operation)
            logical_state = {
                **plaintext_state,
                "role": StringAttr(public_role),
            }
            cast, logical_plaintext = UnrealizedConversionCastOp.cast_one(
                prepare.results[0],
                logical.PublicType().with_state(logical_state),
            )
            logical_plaintext.name_hint = prepare.results[0].name_hint
            block.insert_op_before(cast, operation)
            operation.operands[public_index] = logical_plaintext
            operation.attributes["scale_mode"] = StringAttr(scale_mode)
            transformed += 1
            inserted += 2
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
