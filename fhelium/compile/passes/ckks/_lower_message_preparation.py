"""Lower concrete message preparation into CKKS codec and representation steps."""

from __future__ import annotations

from ..._pipeline import (
    PassResult,
    PassStats,
)

from dataclasses import dataclass

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.rewriter import Rewriter

from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core
from .._operation_transforms import display_name, program_operations

_MESSAGE_PREPARATION_TYPES = (
    ckks.PrepareAddMessageOp,
    ckks.PrepareMultiplyMessageOp,
)


def _state_attribute(
    value_type: object,
    name: str,
    attribute_type: type[Attribute],
    *,
    operation_name: str,
) -> Attribute:
    state = getattr(getattr(value_type, "state", None), "data", {})
    attribute = state.get(name)
    if not isinstance(attribute, attribute_type):
        raise ValueError(
            f"{operation_name} requires a concrete plaintext {name!r} state"
        )
    return attribute


def _message_value(value: SSAValue) -> tuple[tuple[Operation, ...], SSAValue]:
    if isinstance(value.type, core.MessageType):
        return (), value
    state = getattr(getattr(value.type, "state", None), "data", {})
    cast, message = UnrealizedConversionCastOp.cast_one(
        value,
        core.MessageType().with_state(dict(state)),
    )
    message.name_hint = value.name_hint
    return (cast,), message


@dataclass(frozen=True)
class LowerMessagePlaintextPreparationPass:
    """Expand concrete message preparation into visible CKKS operations.

    Depth, scale, ordered prime ids, and modulus basis must already be assigned
    by a caller-selected CKKS scheduling pass. Missing state is an error rather
    than a request for this pass to choose a depth or scale.
    """

    name: str = "lower-message-plaintext-preparation"

    def run(
        self,
        program: Program,
        shared_data: dict[object, object],
    ) -> PassResult:
        del shared_data
        matched = transformed = inserted = 0
        for operation in program_operations(program):
            if not isinstance(operation, _MESSAGE_PREPARATION_TYPES):
                continue
            matched += 1
            if len(operation.operands) != 2 or len(operation.results) != 1:
                raise ValueError(
                    f"{display_name(operation)} requires message and "
                    "ciphertext operands with one plaintext result"
                )
            result_type = operation.result.type
            if not isinstance(result_type, ckks.PlaintextType):
                raise TypeError(
                    "Message preparation result must be CKKS plaintext"
                )
            operation_name = display_name(operation)
            depth = _state_attribute(
                result_type,
                "depth",
                IntegerAttr,
                operation_name=operation_name,
            )
            scale = _state_attribute(
                result_type,
                "scale",
                FloatAttr,
                operation_name=operation_name,
            )
            prime_ids = _state_attribute(
                result_type,
                "prime_ids",
                ArrayAttr,
                operation_name=operation_name,
            )
            basis = _state_attribute(
                result_type,
                "basis",
                StringAttr,
                operation_name=operation_name,
            )
            assert isinstance(depth, IntegerAttr)
            assert isinstance(scale, FloatAttr)
            assert isinstance(prime_ids, ArrayAttr)
            assert isinstance(basis, StringAttr)
            if basis.data != "Q":
                raise ValueError(
                    f"{operation_name} supports public Q-basis preparation, "
                    f"got {basis.data!r}"
                )
            if any(not isinstance(item, IntegerAttr) for item in prime_ids):
                raise TypeError(
                    f"{operation_name} requires integer plaintext prime ids"
                )

            message_operations, message = _message_value(operation.public)
            coefficient_type = ckks.PlaintextType().with_state(
                {
                    "depth": depth,
                    "scale": scale,
                    "representation": StringAttr("integer_coefficients"),
                    "polynomial_domain": StringAttr("coefficient"),
                    "role": StringAttr("message"),
                }
            )
            encode = ckks.EncodeOp.create(
                operands=(message,),
                result_types=(coefficient_type,),
                attributes={"depth": depth, "scale": scale},
            )
            encode.result.name_hint = f"{operation_name}_coefficients"

            rns_state: dict[str, Attribute] = {
                "depth": depth,
                "scale": scale,
                "prime_ids": prime_ids,
                "basis": basis,
                "representation": StringAttr("rns"),
                "polynomial_domain": StringAttr("coefficient"),
                "residue_representation": StringAttr("standard"),
                "role": StringAttr("message"),
            }
            rns_type = ckks.PlaintextType().with_state(rns_state)
            to_rns = ckks.IntegerCoefficientsToRnsOp.create(
                operands=(encode.result,),
                result_types=(rns_type,),
                attributes={"modulus_basis": basis, "depth": depth},
            )
            to_rns.result.name_hint = f"{operation_name}_rns"
            replacements: list[Operation] = [
                *message_operations,
                encode,
                to_rns,
            ]
            prepared = to_rns.result

            if isinstance(operation, ckks.PrepareMultiplyMessageOp):
                montgomery_type = rns_type.with_state(
                    {"residue_representation": StringAttr("montgomery")}
                )
                to_montgomery = ckks.ToMontgomeryResiduesOp(
                    prepared,
                    montgomery_type,
                )
                to_montgomery.result.name_hint = f"{operation_name}_montgomery"
                to_ntt = ckks.ToNttOp(to_montgomery.result, result_type)
                to_ntt.result.name_hint = operation.result.name_hint
                replacements.extend((to_montgomery, to_ntt))
                prepared = to_ntt.result
            else:
                expected_domain = result_type.state.data.get(
                    "polynomial_domain"
                )
                expected_residues = result_type.state.data.get(
                    "residue_representation"
                )
                if expected_domain != StringAttr(
                    "coefficient"
                ) or expected_residues != StringAttr("standard"):
                    raise ValueError(
                        f"{operation_name} requires coefficient/standard "
                        "addition plaintext state"
                    )
                if rns_type.state != result_type.state:
                    result_cast, prepared = UnrealizedConversionCastOp.cast_one(
                        prepared,
                        result_type,
                    )
                    prepared.name_hint = operation.result.name_hint
                    replacements.append(result_cast)
                else:
                    prepared.name_hint = operation.result.name_hint

            Rewriter.replace_op(
                operation,
                tuple(replacements),
                new_results=(prepared,),
            )
            transformed += 1
            inserted += len(replacements)

        if transformed == 0:
            return PassResult.unchanged(program, matched=matched)
        return PassResult(
            program,
            PassStats(
                matched=matched,
                transformed=transformed,
                inserted=inserted,
                removed=transformed,
            ),
        )


__all__ = ["LowerMessagePlaintextPreparationPass"]
