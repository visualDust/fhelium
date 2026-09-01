"""Lower role-explicit logical operations to explicit CKKS primitives."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.ir import Program

from fhelium.ir.dialects import ckks, logical
from .._operation_transforms import (
    cast_before,
    ciphertext_type,
    display_name,
    program_operations,
)

_CT_CT_TYPES: dict[type[Operation], type[Operation]] = {
    logical.AddEncryptedEncryptedOp: ckks.AddOp,
    logical.SubtractEncryptedEncryptedOp: ckks.SubtractOp,
    logical.MultiplyEncryptedEncryptedOp: ckks.MultiplyOp,
}
_MIXED_TYPES: dict[type[Operation], tuple[str, int, int]] = {
    logical.AddEncryptedPublicOp: ("add", 0, 1),
    logical.AddPublicEncryptedOp: ("add", 1, 0),
    logical.SubtractEncryptedPublicOp: ("subtract", 0, 1),
    logical.SubtractPublicEncryptedOp: ("subtract", 1, 0),
    logical.MultiplyEncryptedPublicOp: ("multiply", 0, 1),
    logical.MultiplyPublicEncryptedOp: ("multiply", 1, 0),
}


def _ntt_montgomery(value: SSAValue) -> bool:
    state = getattr(getattr(value.type, "state", None), "data", {})
    domain = state.get("polynomial_domain")
    residues = state.get("residue_representation")
    return (
        isinstance(domain, StringAttr)
        and domain.data == "ntt"
        and isinstance(residues, StringAttr)
        and residues.data == "montgomery"
    )


def _replace_with_result_cast(
    operation: Operation,
    replacements: tuple[Operation, ...],
    result: SSAValue,
) -> None:
    """Replace a logical op and preserve its external type through a cast."""

    if len(operation.results) != 1:
        raise ValueError("logical-to-CKKS replacement requires one result")
    cast, cast_result = UnrealizedConversionCastOp.cast_one(
        result, operation.results[0].type
    )
    cast_result.name_hint = operation.results[0].name_hint
    Rewriter.replace_op(
        operation,
        (*replacements, cast),
        new_results=(cast_result,),
    )


@dataclass(frozen=True)
class LowerLogicalToCkksPass:
    """Lower registered logical arithmetic through visible typed CKKS edges.

    Unrealized conversion casts preserve mixed-level result types for consumers
    that have not been lowered. Locally unresolved patterns remain legal no-ops
    and leave their original operations unchanged with diagnostics.
    """

    name: str = "lower-logical-to-ckks"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Lower locally ready registered operations and report blockers."""

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        for operation in program_operations(program):
            if isinstance(operation, logical.NegateEncryptedOp):
                matched += 1
                if len(operation.operands) != 1:
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: logical unary operation "
                        "requires one encrypted operand"
                    )
                    continue
                source = operation.operands[0]
                typed = cast_before(
                    operation,
                    source,
                    ciphertext_type(
                        source,
                        domain="coefficient",
                        residues="standard",
                        components=2,
                    ),
                    name_hint=f"{display_name(operation)}_ciphertext",
                )
                inserted += typed is not source
                replacement = ckks.NegateOp(typed, typed.type)
                replacement.results[0].name_hint = operation.results[
                    0
                ].name_hint
                _replace_with_result_cast(
                    operation, (replacement,), replacement.results[0]
                )
                transformed += 1
                inserted += 1
                continue

            target_type = _CT_CT_TYPES.get(type(operation))
            if target_type is not None:
                matched += 1
                if len(operation.operands) != 2:
                    skipped += 1
                    diagnostics.append(
                        f"{display_name(operation)}: logical binary operation "
                        "requires exactly two operands"
                    )
                    continue
                is_multiply = isinstance(
                    operation, logical.MultiplyEncryptedEncryptedOp
                )
                typed_operands: list[SSAValue] = []
                for index, operand in enumerate(operation.operands):
                    if is_multiply and not _ntt_montgomery(operand):
                        skipped += 1
                        diagnostics.append(
                            f"{display_name(operation)}: CT×CT multiply lacks "
                            "typed NTT/Montgomery operands"
                        )
                        break
                    typed = cast_before(
                        operation,
                        operand,
                        ciphertext_type(
                            operand,
                            domain="ntt" if is_multiply else "coefficient",
                            residues=(
                                "montgomery" if is_multiply else "standard"
                            ),
                            components=2,
                        ),
                        name_hint=f"{display_name(operation)}_operand{index}",
                    )
                    inserted += typed is not operand
                    typed_operands.append(typed)
                if len(typed_operands) != 2:
                    continue
                result_type = ciphertext_type(
                    typed_operands[0],
                    domain="ntt" if is_multiply else "coefficient",
                    residues="montgomery" if is_multiply else "standard",
                    components=3 if is_multiply else 2,
                )
                attributes: dict[str, Attribute] = {}
                replacement = target_type.create(
                    operands=typed_operands,
                    result_types=(result_type,),
                    attributes=attributes,
                )
                replacement.results[0].name_hint = operation.results[
                    0
                ].name_hint
                _replace_with_result_cast(
                    operation, (replacement,), replacement.results[0]
                )
                transformed += 1
                inserted += 1
                continue

            mixed = _MIXED_TYPES.get(type(operation))
            if mixed is None:
                continue
            matched += 1
            arithmetic, encrypted_index, public_index = mixed
            if len(operation.operands) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: mixed logical operation "
                    "requires exactly two operands"
                )
                continue
            plaintext = operation.operands[public_index]
            plaintext_state = dict(
                getattr(getattr(plaintext.type, "state", None), "data", {})
            )
            if not all(
                isinstance(plaintext_state.get(key), StringAttr)
                for key in ("polynomial_domain", "residue_representation")
            ):
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: mixed operation lacks typed "
                    "plaintext preparation"
                )
                continue
            typed_plaintext = cast_before(
                operation,
                plaintext,
                ckks.PlaintextType().with_state(plaintext_state),
                name_hint=f"{display_name(operation)}_prepared_plaintext",
            )
            inserted += typed_plaintext is not plaintext
            ciphertext = operation.operands[encrypted_index]
            is_multiply = arithmetic == "multiply"
            if is_multiply and not _ntt_montgomery(ciphertext):
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: plaintext multiply lacks a "
                    "typed NTT/Montgomery ciphertext"
                )
                continue
            typed_ciphertext = cast_before(
                operation,
                ciphertext,
                ciphertext_type(
                    ciphertext,
                    domain="ntt" if is_multiply else "coefficient",
                    residues="montgomery" if is_multiply else "standard",
                    components=2,
                ),
                name_hint=f"{display_name(operation)}_ciphertext",
            )
            inserted += typed_ciphertext is not ciphertext
            attributes: dict[str, Attribute] = {}
            if is_multiply:
                replacement = ckks.MultiplyPlaintextOp.create(
                    operands=(typed_ciphertext, typed_plaintext),
                    result_types=(typed_ciphertext.type,),
                )
                replacements = (replacement,)
            elif arithmetic == "add":
                replacement = ckks.AddPlaintextOp(
                    typed_ciphertext, typed_plaintext, typed_ciphertext.type
                )
                replacements = (replacement,)
            else:
                negated = ckks.NegateOp(typed_ciphertext, typed_ciphertext.type)
                if encrypted_index == 1:
                    replacement = ckks.AddPlaintextOp(
                        negated, typed_plaintext, typed_ciphertext.type
                    )
                    replacements = (negated, replacement)
                    inserted += 1
                else:
                    summed = ckks.AddPlaintextOp(
                        negated, typed_plaintext, typed_ciphertext.type
                    )
                    replacement = ckks.NegateOp(summed, typed_ciphertext.type)
                    replacements = (negated, summed, replacement)
                    inserted += 2
            replacement.results[0].name_hint = operation.results[0].name_hint
            _replace_with_result_cast(
                operation, replacements, replacement.results[0]
            )
            transformed += 1
            inserted += 1
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
