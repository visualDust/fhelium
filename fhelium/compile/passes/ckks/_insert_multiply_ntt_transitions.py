"""Insert NTT transitions required by encrypted multiplication."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fhelium.compile._compilation import Compilation


from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import UnrealizedConversionCastOp
from xdsl.ir import SSAValue


from fhelium.ir.dialects import ckks, logical
from .._operation_transforms import (
    cast_before,
    ciphertext_type,
    display_name,
    program_operations,
)
from ._transition_state import infer_logical_representation

_MULTIPLY_ENCRYPTED_INDICES: dict[type[object], tuple[int, ...]] = {
    logical.MultiplyEncryptedEncryptedOp: (0, 1),
    logical.MultiplyEncryptedPublicOp: (0,),
    logical.MultiplyPublicEncryptedOp: (1,),
}


@dataclass(frozen=True)
class InsertMultiplyNttTransitionsPass:
    """Insert typed CKKS NTT transitions at registered logical multiplies.

    The transformed operand type records NTT/Montgomery state, so repeated
    application is idempotent without a marker attribute. Unrealized casts
    preserve visible type-changing edges from logical values to CKKS values.
    """

    name: str = "insert-multiply-ntt-transitions"

    def run(self, compilation: "Compilation") -> PassResult:
        """Insert missing transitions or return a legal no-op report."""
        program = compilation.program
        workspace = compilation.workspace

        del workspace
        matched = transformed = inserted = skipped = 0
        diagnostics: list[str] = []
        converted_operands: dict[SSAValue, SSAValue] = {}
        inferred_representations: dict[SSAValue, tuple[str, str]] = {}
        for operation in program_operations(program):
            encrypted_indices = _MULTIPLY_ENCRYPTED_INDICES.get(type(operation))
            if encrypted_indices is None:
                continue
            invalid: list[tuple[int, str]] = []
            representations: dict[int, tuple[str, str]] = {}
            for index in encrypted_indices:
                try:
                    representation = infer_logical_representation(
                        operation.operands[index],
                        inferred_representations,
                    )
                except ValueError as error:
                    invalid.append((index, str(error)))
                    continue
                representations[index] = representation
                if representation not in {
                    ("coefficient", "standard"),
                    ("ntt", "montgomery"),
                }:
                    invalid.append(
                        (
                            index,
                            f"unsupported representation {representation!r}",
                        )
                    )
            if invalid:
                matched += 1
                skipped += 1
                details = "; ".join(
                    f"operand {index}: {message}" for index, message in invalid
                )
                diagnostics.append(f"{display_name(operation)}: {details}")
                continue
            pending = tuple(
                index
                for index in encrypted_indices
                if representations[index] == ("coefficient", "standard")
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
                cached = converted_operands.get(operand)
                if cached is not None:
                    operation.operands[index] = cached
                    continue
                side = "lhs" if index == 0 else "rhs"
                coefficient_type = ciphertext_type(
                    operand,
                    domain=representations[index][0],
                    residues=representations[index][1],
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
                converted_operands[operand] = logical_result
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
