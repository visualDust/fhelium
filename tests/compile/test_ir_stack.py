"""Essential neutral-IR behavior tests."""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import Block

from fhelium import ir

_MIXED_PROGRAM = r'''
builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.2"
} {
  func.func @main(%secret: !fhelium_ckks.ciphertext<{components = 2 : i64, level = 3 : i64}>) -> !fhelium_ckks.ciphertext<{}> {
    %product = "fhelium_ckks.multiply"(%secret, %secret) : (!fhelium_ckks.ciphertext<{components = 2 : i64, level = 3 : i64}>, !fhelium_ckks.ciphertext<{components = 2 : i64, level = 3 : i64}>) -> !fhelium_ckks.ciphertext<{components = 3 : i64, level = 3 : i64}>
    %result = "vendor.experimental.refresh"(%product) : (!fhelium_ckks.ciphertext<{components = 3 : i64, level = 3 : i64}>) -> !fhelium_ckks.ciphertext<{}>
    func.return %result : !fhelium_ckks.ciphertext<{}>
  }
}
'''


def test_program_preserves_registered_partial_state_and_unknown_dialects() -> (
    None
):
    program = ir.parse(_MIXED_PROGRAM, source_name="mixed.mlir")
    inventory = ir.inventory_program(program)

    assert inventory.operation_counts["fhelium_ckks.multiply"] == 1
    assert inventory.operation_counts["vendor.experimental.refresh"] == 1
    assert inventory.dialects == frozenset({"func", "fhelium_ckks", "vendor"})
    exported = program.to_text()
    assert ir.parse(exported).to_text() == exported


def test_ckks_multiply_rejects_invalid_or_missing_component_state() -> None:
    ct2 = ir.dialects.ckks.CiphertextType().with_state(
        components=IntegerAttr(2, 64)
    )
    ct3 = ir.dialects.ckks.CiphertextType().with_state(
        components=IntegerAttr(3, 64)
    )
    block = Block(arg_types=(ct2, ct3))
    valid = ir.dialects.ckks.MultiplyOp(block.args[0], block.args[0], ct3)
    invalid = ir.dialects.ckks.MultiplyOp(block.args[1], block.args[0], ct2)
    specification = ir.DEFAULT_OPERATION_SPECS.require(valid.name)
    assert specification.diagnostics(valid) == ()
    assert specification.diagnostics(invalid)

    ciphertext_type = ir.dialects.ckks.CiphertextType()
    untyped_block = Block(arg_types=(ciphertext_type, ciphertext_type))
    missing = ir.dialects.ckks.MultiplyOp(
        untyped_block.args[0], untyped_block.args[1], ciphertext_type
    )
    assert specification.diagnostics(missing)


def test_ckks_rotation_output_domain_matches_represented_result() -> None:
    coefficient = ir.dialects.ckks.CiphertextType().with_state(
        components=IntegerAttr(2, 64),
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("standard"),
    )
    ntt = coefficient.with_state(
        polynomial_domain=StringAttr("ntt"),
        residue_representation=StringAttr("montgomery"),
    )
    block = Block(arg_types=(coefficient, ir.dialects.ckks.EvaluationKeyType()))
    valid = ir.dialects.ckks.RotateOp(
        block.args[0],
        block.args[1],
        ntt,
        attributes={"output_domain": StringAttr("ntt")},
    )
    invalid = ir.dialects.ckks.RotateOp(
        block.args[0],
        block.args[1],
        coefficient,
        attributes={"output_domain": StringAttr("ntt")},
    )
    specification = ir.DEFAULT_OPERATION_SPECS.require(valid.name)
    assert specification.diagnostics(valid) == ()
    assert specification.diagnostics(invalid)
