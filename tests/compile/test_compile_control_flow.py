"""Behavior checks for Compile transformations in structured regions."""

from __future__ import annotations

import pytest

from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.dialects.test import TestOp as VendorRegionOp
from xdsl.dialects.test import TestTermOp as VendorYieldOp
from xdsl.ir import Block, Region

from fhelium.compile.passes import (
    AssignImplementationsPass,
    LowerCkksToRnsNttPass,
    LowerSpecializedCollectivesPass,
)
from fhelium.config import CkksConfig
from fhelium.ir import EXECUTION_IMPLEMENTATION_ATTRIBUTE, Program
from fhelium.ir.dialects import ckks, distributed, rns


def _scf_negate_program() -> tuple[Program, ckks.NegateOp]:
    ciphertext_type = ckks.CiphertextType()
    entry = Block(arg_types=(ciphertext_type,))
    zero = arith.ConstantOp.from_int_and_width(0, IndexType())
    one = arith.ConstantOp.from_int_and_width(1, IndexType())
    body = Block(arg_types=(IndexType(), ciphertext_type))
    negate = ckks.NegateOp(body.args[1], ciphertext_type)
    body.add_ops((negate, scf.YieldOp(negate)))
    loop = scf.ForOp(
        zero,
        one,
        one,
        (entry.args[0],),
        Region(body),
    )
    entry.add_ops((zero, one, loop, ReturnOp(loop)))
    return Program.from_function(entry, (ciphertext_type,)), negate


def test_ckks_lowering_enters_known_scf_region() -> None:
    program, _ = _scf_negate_program()

    result = LowerCkksToRnsNttPass().run(
        program,
        {CkksConfig: CkksConfig(enforce_security_budget=False)},
    )

    assert result.stats.matched == 1
    assert result.stats.transformed == 1
    assert not any(
        isinstance(operation, ckks.NegateOp) for operation in program.walk()
    )
    assert any(
        isinstance(operation, rns.NegateStandardOp)
        for operation in program.walk()
    )


def test_implementation_assignment_enters_known_scf_region() -> None:
    program, negate = _scf_negate_program()

    result = AssignImplementationsPass({ckks.NegateOp.name: "test-negate"}).run(
        program, {}
    )

    assert result.stats.transformed == 1
    assigned = negate.attributes[EXECUTION_IMPLEMENTATION_ATTRIBUTE]
    assert isinstance(assigned, StringAttr)
    assert assigned.data == "test-negate"


def test_compile_does_not_enter_unknown_vendor_region() -> None:
    ciphertext_type = ckks.CiphertextType()
    entry = Block(arg_types=(ciphertext_type,))
    nested = Block()
    negate = ckks.NegateOp(entry.args[0], ciphertext_type)
    nested.add_ops((negate, VendorYieldOp()))
    vendor_region = VendorRegionOp(regions=(Region(nested),))
    entry.add_ops((vendor_region, ReturnOp(entry.args[0])))
    program = Program.from_function(entry, (ciphertext_type,))

    result = AssignImplementationsPass(
        {
            ckks.NegateOp.name: "must-not-enter",
            VendorRegionOp.name: "vendor-owner",
        }
    ).run(program, {})

    assert result.stats.matched == 1
    assert EXECUTION_IMPLEMENTATION_ATTRIBUTE not in negate.attributes
    assigned = vendor_region.attributes[EXECUTION_IMPLEMENTATION_ATTRIBUTE]
    assert isinstance(assigned, StringAttr)
    assert assigned.data == "vendor-owner"


def _specialized_collective_program() -> Program:
    ciphertext_type = ckks.CiphertextType()
    entry = Block(
        arg_types=(ciphertext_type, distributed.GroupType()),
    )
    reduction = distributed.AllReduceAddCiphertextOp(
        entry.args[0],
        entry.args[1],
    )
    entry.add_ops((reduction, ReturnOp(reduction)))
    return Program.from_function(entry, (ciphertext_type,))


def test_specialized_collective_lowering_exposes_combine_region() -> None:
    program = _specialized_collective_program()

    result = LowerSpecializedCollectivesPass().run(program, {})

    assert result.stats.transformed == 1
    assert not any(
        isinstance(operation, distributed.AllReduceAddCiphertextOp)
        for operation in program.walk()
    )
    reduction = next(
        operation
        for operation in program.walk()
        if isinstance(operation, distributed.AllReduceOp)
    )
    assert any(
        isinstance(operation, ckks.AddOp) for operation in reduction.combine.ops
    )


def test_ckks_lowering_enters_collective_combine_region() -> None:
    program = _specialized_collective_program()
    LowerSpecializedCollectivesPass().run(program, {})

    result = LowerCkksToRnsNttPass().run(
        program,
        {CkksConfig: CkksConfig(enforce_security_budget=False)},
    )

    assert result.stats.transformed == 1
    assert any(
        isinstance(operation, rns.AddStandardOp) for operation in program.walk()
    )


def test_collective_lowering_respects_implementation_constraint() -> None:
    program = _specialized_collective_program()
    AssignImplementationsPass(
        {distributed.AllReduceAddCiphertextOp.name: ("specialized-provider")}
    ).run(program, {})

    with pytest.raises(ValueError, match="preserve the operation"):
        LowerSpecializedCollectivesPass().run(program, {})
