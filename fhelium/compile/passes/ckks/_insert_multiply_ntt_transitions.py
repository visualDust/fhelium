"""Insert NTT transitions required by encrypted multiplication."""

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
    program_operations,
)

_MULTIPLY_ENCRYPTED_INDICES: dict[type[object], tuple[int, ...]] = {
    logical.MultiplyEncryptedEncryptedOp: (0, 1),
    logical.MultiplyEncryptedPublicOp: (0,),
    logical.MultiplyPublicEncryptedOp: (1,),
}


def _is_ntt_montgomery(value_type: object) -> bool:
    state = getattr(getattr(value_type, "state", None), "data", {})
    domain = state.get("polynomial_domain")
    residues = state.get("residue_representation")
    return (
        isinstance(domain, StringAttr)
        and domain.data == "ntt"
        and isinstance(residues, StringAttr)
        and residues.data == "montgomery"
    )


@dataclass(frozen=True)
class InsertMultiplyNttTransitionsPass:
    """Insert typed CKKS NTT transitions at registered logical multiplies.

    The transformed operand type records NTT/Montgomery state, so repeated
    application is idempotent without a marker attribute. Unrealized casts
    preserve visible type-changing edges from logical values to CKKS values.
    """

    name: str = "insert-multiply-ntt-transitions"

    def run(
        self,
        program: Program,
        workspace: dict[object, object],
    ) -> PassResult:
        """Insert missing transitions or return a legal no-op report."""

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        for operation in program_operations(program):
            encrypted_indices = _MULTIPLY_ENCRYPTED_INDICES.get(type(operation))
            if encrypted_indices is None:
                continue
            pending = tuple(
                index
                for index in encrypted_indices
                if not _is_ntt_montgomery(operation.operands[index].type)
            )
            if not pending:
                continue
            matched += 1
            if len(operation.operands) != 2:
                skipped += 1
                diagnostics.append(
                    f"{display_name(operation)}: logical multiply requires "
                    "exactly two operands"
                )
                continue
            block = operation.parent_block()
            if block is None:
                raise ValueError(
                    "multiply operation is not attached to a block"
                )
            for index in pending:
                operand = operation.operands[index]
                side = "lhs" if index == 0 else "rhs"
                coefficient_type = ciphertext_type(
                    operand,
                    domain="coefficient",
                    residues="standard",
                    components=2,
                )
                typed_operand = cast_before(
                    operation,
                    operand,
                    coefficient_type,
                    name_hint=f"{display_name(operation)}_{side}_ciphertext",
                )
                if typed_operand is not operand:
                    inserted += 1
                transition = ckks.ToNttOp(
                    typed_operand,
                    ciphertext_type(
                        typed_operand,
                        domain="ntt",
                        residues="montgomery",
                        components=2,
                    ),
                )
                transition.results[
                    0
                ].name_hint = f"{display_name(operation)}_{side}_ntt"
                block.insert_op_before(transition, operation)
                logical_state = dict(
                    getattr(
                        getattr(transition.results[0].type, "state", None),
                        "data",
                        {},
                    )
                )
                cast, logical_result = UnrealizedConversionCastOp.cast_one(
                    transition.results[0],
                    logical.EncryptedType().with_state(logical_state),
                )
                logical_result.name_hint = transition.results[0].name_hint
                block.insert_op_before(cast, operation)
                operation.operands[index] = logical_result
                inserted += 2
            transformed += 1
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
