"""Public CKKS values at the Backend Program boundary."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
    f64,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Attribute, Block, Operation

from fhelium.backend.execution import OperationBackend
from fhelium.backend.implementation import (
    OperationImplementation,
    OperationImplementationRegistry,
    OperationInvocation,
)
from fhelium.backend.resources import (
    BoundResource,
    ResourceRequirement,
)
from fhelium.compile import Compilation
from fhelium.ir import Program
from fhelium.ir.dialects import ckks, core, semantic
from fhelium.values import Ciphertext, Plaintext


def _array(values: tuple[int, ...]) -> ArrayAttr[IntegerAttr]:
    return ArrayAttr(IntegerAttr(value, 64) for value in values)


def _ciphertext_state(
    *,
    depth: int,
    scale: float,
    prime_ids: tuple[int, ...],
) -> dict[str, Attribute]:
    return {
        "depth": IntegerAttr(depth, 64),
        "scale": FloatAttr(scale, f64),
        "prime_ids": _array(prime_ids),
        "polynomial_domain": StringAttr("coefficient"),
        "basis": StringAttr("Q"),
        "residue_representation": StringAttr("standard"),
        "components": IntegerAttr(2, 64),
    }


def _build(
    program: Program,
    *implementations: OperationImplementation,
):
    registry = OperationImplementationRegistry(implementations)
    backend = OperationBackend(registry)
    return backend.link(Compilation(program))


@dataclass
class _TensorOnlyNegate:
    """Record the Tensor payload received from a public-value boundary."""

    name: str = "test-tensor-only-negate"
    operation_types: tuple[type[Operation], ...] = (ckks.NegateOp,)
    supports_in_place: bool = False
    inputs: list[torch.Tensor] = field(default_factory=list)

    def resource_requirements(
        self,
        invocation: OperationInvocation,
    ) -> tuple[ResourceRequirement, ...]:
        del invocation
        return ()

    def execute(
        self,
        invocation: OperationInvocation,
        inputs: tuple[torch.Tensor, ...],
        resources: tuple[BoundResource, ...],
        *,
        in_place: bool,
    ) -> tuple[torch.Tensor, ...]:
        del invocation, resources, in_place
        self.inputs.append(inputs[0])
        return (-inputs[0][..., 1:, :],)


def test_encrypted_role_boundary_unwraps_and_reconstructs_from_result_ir() -> (
    None
):
    input_state = _ciphertext_state(
        depth=0,
        scale=32.0,
        prime_ids=(0, 1),
    )
    result_state = _ciphertext_state(
        depth=1,
        scale=32.0,
        prime_ids=(1,),
    )
    input_boundary = semantic.SecretType().with_state(input_state)
    input_ckks = ckks.CiphertextType().with_state(input_state)
    result_ckks = ckks.CiphertextType().with_state(result_state)
    result_boundary = semantic.SecretType().with_state(result_state)
    block = Block(arg_types=(input_boundary,))
    enter, entered = UnrealizedConversionCastOp.cast_one(
        block.args[0], input_ckks
    )
    operation = ckks.NegateOp(entered, result_ckks)
    leave, returned = UnrealizedConversionCastOp.cast_one(
        operation.result, result_boundary
    )
    block.add_ops((enter, operation, leave, ReturnOp(returned)))
    program = Program.from_function(block, (result_boundary,))
    implementation = _TensorOnlyNegate()
    executable = _build(program, implementation)
    data = torch.arange(16, dtype=torch.int64).reshape(2, 2, 4)
    source = Ciphertext(
        data=data,
        depth=0,
        scale=32.0,
        prime_ids=(0, 1),
    )

    result = executable.run(source)

    assert implementation.inputs == [source.data]
    assert isinstance(result, Ciphertext)
    assert result.depth == 1
    assert result.scale == 32.0
    assert result.prime_ids == (1,)
    assert result.polynomial_domain == "coefficient"
    assert result.modulus_basis == "Q"
    assert result.residue_representation == "standard"
    torch.testing.assert_close(result.data, -source.data[..., 1:, :])


def _identity_program(value_type: Attribute) -> Program:
    block = Block(arg_types=(value_type,))
    block.add_op(ReturnOp(block.args[0]))
    return Program.from_function(block, (block.args[0].type,))


@pytest.mark.parametrize(
    ("state", "value", "active_field"),
    (
        (
            {
                "depth": IntegerAttr(0, 64),
                "scale": FloatAttr(8.0, f64),
                "representation": StringAttr("slots"),
            },
            Plaintext(
                message=torch.tensor([0.25, -0.5]),
                depth=0,
                scale=8.0,
            ),
            "message",
        ),
        (
            {
                "depth": IntegerAttr(1, 64),
                "scale": FloatAttr(16.0, f64),
                "representation": StringAttr("integer_coefficients"),
                "polynomial_domain": StringAttr("coefficient"),
            },
            Plaintext(
                message=None,
                depth=1,
                scale=16.0,
                data=torch.tensor([1, -2, 3, -4], dtype=torch.int64),
                representation="integer_coefficients",
                polynomial_domain="coefficient",
            ),
            "data",
        ),
        (
            {
                "depth": IntegerAttr(0, 64),
                "scale": FloatAttr(4.0, f64),
                "representation": StringAttr("rns"),
                "polynomial_domain": StringAttr("coefficient"),
                "basis": StringAttr("Q"),
                "residue_representation": StringAttr("standard"),
                "prime_ids": _array((0, 1)),
            },
            Plaintext(
                message=None,
                depth=0,
                scale=4.0,
                data=torch.arange(8, dtype=torch.int64).reshape(2, 4),
                representation="rns",
                polynomial_domain="coefficient",
                modulus_basis="Q",
                residue_representation="standard",
                prime_ids=(0, 1),
            ),
            "data",
        ),
    ),
)
def test_plaintext_role_boundary_uses_the_declared_representation(
    state: dict[str, Attribute],
    value: Plaintext,
    active_field: str,
) -> None:
    value_type = core.PlaintextType().with_state(state)
    result = _build(_identity_program(value_type)).run(value)

    assert isinstance(result, Plaintext)
    assert result is not value
    assert result.depth == value.depth
    assert result.scale == value.scale
    assert result.representation == value.representation
    assert result.polynomial_domain == value.polynomial_domain
    assert result.modulus_basis == value.modulus_basis
    assert result.residue_representation == value.residue_representation
    assert result.prime_ids == value.prime_ids
    assert getattr(result, active_field) is getattr(value, active_field)


def test_ckks_input_rejects_raw_tensor_and_state_disagreement() -> None:
    value_type = ckks.CiphertextType().with_state(
        _ciphertext_state(depth=0, scale=32.0, prime_ids=(0, 1))
    )
    executable = _build(_identity_program(value_type))
    data = torch.zeros((2, 2, 4), dtype=torch.int64)

    with pytest.raises(
        TypeError, match="Program input 0 requires a Ciphertext"
    ):
        executable.run(data)

    mismatched = Ciphertext(
        data=data,
        depth=0,
        scale=64.0,
        prime_ids=(0, 1),
    )
    with pytest.raises(ValueError, match="declares scale=32.0"):
        executable.run(mismatched)


def test_missing_concrete_result_state_fails_without_input_inference() -> None:
    block = Block(arg_types=(core.MessageType(),))
    result_type = semantic.SecretType()
    boundary, result = UnrealizedConversionCastOp.cast_one(
        block.args[0], result_type
    )
    block.add_ops((boundary, ReturnOp(result)))
    executable = _build(Program.from_function(block, (result_type,)))

    with pytest.raises(
        ValueError,
        match="Program result 0 lacks concrete 'depth' state",
    ):
        executable.run(torch.zeros((2, 2, 4), dtype=torch.int64))


def test_message_tensor_boundary_remains_unwrapped() -> None:
    value = torch.arange(4, dtype=torch.int64)
    executable = _build(_identity_program(core.MessageType()))

    assert executable.run(value) is value
