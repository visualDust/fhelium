from __future__ import annotations

import pytest
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from fhelium.ir import Program, create_dialect_context
from fhelium.ir.dialects import REGISTERED_DIALECTS
from fhelium.ir.dialects.ckks import AddOp, CiphertextType
from fhelium.ir.dialects.distributed import (
    AllReduceAddCiphertextOp,
    AllReduceOp,
    GroupSizeOp,
    GroupType,
    RankOp,
    YieldOp,
)
from fhelium.ir.dialects.memory import DeviceType, TransferOp


def _distributed_program() -> Program:
    ciphertext_type = CiphertextType()
    block = Block(
        arg_types=(ciphertext_type, DeviceType(), GroupType()),
    )
    transfer = TransferOp(
        block.args[0],
        block.args[1],
        memory_space="default",
    )
    rank = RankOp(block.args[2])
    group_size = GroupSizeOp(block.args[2])
    combine = Block(arg_types=(ciphertext_type, ciphertext_type))
    addition = AddOp(combine.args[0], combine.args[1], ciphertext_type)
    combine.add_ops((addition, YieldOp(addition)))
    generic = AllReduceOp(transfer, block.args[2], Region(combine))
    specialized = AllReduceAddCiphertextOp(generic, block.args[2])
    block.add_ops(
        (
            transfer,
            rank,
            group_size,
            generic,
            specialized,
            ReturnOp(specialized),
        )
    )
    return Program.from_function(block, (ciphertext_type,))


def test_memory_and_distributed_dialects_are_registered() -> None:
    dialects = {dialect.name: dialect for dialect in REGISTERED_DIALECTS}
    assert tuple(
        operation.name for operation in dialects["fhelium_memory"].operations
    ) == ("fhelium_memory.transfer",)
    assert tuple(
        operation.name for operation in dialects["fhelium_dist"].operations
    ) == (
        "fhelium_dist.rank",
        "fhelium_dist.group_size",
        "fhelium_dist.broadcast",
        "fhelium_dist.all_reduce",
        "fhelium_dist.all_reduce_add_ciphertext",
        "fhelium_dist.yield",
    )


def test_structured_distributed_program_round_trips() -> None:
    program = _distributed_program()
    text = program.to_text()
    parsed = Program.parse(text, source_name="distributed-round-trip")

    parsed.verify_structure()
    assert parsed.to_text() == text
    assert "fhelium_memory.transfer" in text
    assert "fhelium_dist.all_reduce" in text
    assert "fhelium_dist.all_reduce_add_ciphertext" in text


def test_context_loads_standard_arithmetic_and_structured_control_flow() -> (
    None
):
    context = create_dialect_context()

    assert context.get_op("arith.addi").name == "arith.addi"
    assert context.get_op("scf.for").name == "scf.for"
    assert context.get_op("scf.if").name == "scf.if"


def test_transfer_rejects_a_different_result_type() -> None:
    block = Block(arg_types=(CiphertextType(), DeviceType()))
    operation = TransferOp(
        block.args[0],
        block.args[1],
        result_type=DeviceType(),
    )

    with pytest.raises(VerifyException, match="source and result types"):
        operation.verify()


def test_transfer_rejects_an_unknown_memory_space() -> None:
    block = Block(arg_types=(CiphertextType(), DeviceType()))
    operation = TransferOp(
        block.args[0],
        block.args[1],
        memory_space="managed",
    )

    with pytest.raises(VerifyException, match="memory_space"):
        operation.verify()


def test_generic_all_reduce_requires_matching_combine_types() -> None:
    ciphertext_type = CiphertextType()
    block = Block(arg_types=(ciphertext_type, GroupType()))
    combine = Block(arg_types=(ciphertext_type, DeviceType()))
    combine.add_op(YieldOp(combine.args[0]))
    operation = AllReduceOp(block.args[0], block.args[1], Region(combine))

    with pytest.raises(VerifyException, match="combine arguments"):
        operation.verify()
