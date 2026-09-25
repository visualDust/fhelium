"""Lower role-explicit logical operations to explicit CKKS primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.rewriter import Rewriter


from fhelium.ir.dialects import ckks, logical
from fhelium.ir.dialects._common import OpenStateType
from ._transition_state import infer_logical_representation
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
    result_type = operation.results[0].type
    if isinstance(result_type, OpenStateType) and isinstance(
        result.type, OpenStateType
    ):
        result_type = result_type.with_state(result.type.state.data)
    cast, cast_result = UnrealizedConversionCastOp.cast_one(result, result_type)
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

    def run(self, compilation: "Compilation") -> PassResult:
        """Lower locally ready registered operations and report blockers."""
        program = compilation.program
        workspace = compilation.workspace

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        for operation in program_operations(program):
            representation_cast = isinstance(
                operation, UnrealizedConversionCastOp
            )
            if (
                representation_cast
                and len(operation.operands) == len(operation.results) == 1
            ) or isinstance(operation, (ckks.ToNttOp, ckks.FromNttOp)):
                source_type = operation.operands[0].type
                target_type = operation.results[0].type
                if isinstance(source_type, OpenStateType) and isinstance(
                    target_type, OpenStateType
                ):
                    additions = {
                        name: value
                        for name, value in source_type.state.data.items()
                        if (representation_cast or name != "strides")
                        and (
                            name not in target_type.state.data
                            or target_type.state.data[name]
                            == StringAttr("unknown")
                        )
                    }
                    if additions:
                        Rewriter.replace_value_with_new_type(
                            operation.results[0],
                            target_type.with_state(additions),
                        )
                        matched += 1
                        transformed += 1
                continue
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
                    ciphertext_type(source),
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
                representations = []
                for operand in operation.operands:
                    try:
                        representations.append(
                            infer_logical_representation(operand, {})
                        )
                    except ValueError:
                        representations.append((None, None))
                align_coefficient = (
                    not is_multiply
                    and (None, None) not in representations
                    and len(set(representations)) > 1
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
                            domain=representations[index][0],
                            residues=representations[index][1],
                            components=2 if is_multiply else None,
                        ),
                        name_hint=f"{display_name(operation)}_operand{index}",
                    )
                    inserted += typed is not operand
                    if align_coefficient and representations[index][0] == "ntt":
                        converted = ckks.FromNttOp(
                            typed,
                            ciphertext_type(
                                typed, domain="coefficient", residues="standard"
                            ),
                        )
                        owner = operation.parent_block()
                        assert owner is not None
                        owner.insert_op_before(converted, operation)
                        typed = converted.result
                        inserted += 1
                    typed_operands.append(typed)
                if len(typed_operands) != 2:
                    continue
                result_type = ciphertext_type(
                    typed_operands[0],
                    domain="ntt" if is_multiply else None,
                    residues="montgomery" if is_multiply else None,
                    components=3 if is_multiply else None,
                )
                if is_multiply:
                    # Scale propagation derives s_left * s_right; the input
                    # scale is not the product's scale.
                    result_type = result_type.with_state({"scale": None})
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
            try:
                input_domain, input_residues = infer_logical_representation(
                    ciphertext, {}
                )
            except ValueError:
                input_domain = input_residues = None
            typed_ciphertext = cast_before(
                operation,
                ciphertext,
                ciphertext_type(
                    ciphertext, domain=input_domain, residues=input_residues
                ),
                name_hint=f"{display_name(operation)}_ciphertext",
            )
            inserted += typed_ciphertext is not ciphertext
            if not is_multiply and input_domain == "ntt":
                converted = ckks.FromNttOp(
                    typed_ciphertext,
                    ciphertext_type(
                        typed_ciphertext,
                        domain="coefficient",
                        residues="standard",
                    ),
                )
                owner = operation.parent_block()
                assert owner is not None
                owner.insert_op_before(converted, operation)
                typed_ciphertext = converted.result
                inserted += 1
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
        for function in program.functions:
            function.update_function_type()
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
