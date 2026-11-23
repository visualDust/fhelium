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
  func.func @main(%secret: !fhelium_ckks.ciphertext<{components = 2 : i64, depth = 3 : i64}>) -> !fhelium_ckks.ciphertext<{}> {
    %product = "fhelium_ckks.multiply"(%secret, %secret) : (!fhelium_ckks.ciphertext<{components = 2 : i64, depth = 3 : i64}>, !fhelium_ckks.ciphertext<{components = 2 : i64, depth = 3 : i64}>) -> !fhelium_ckks.ciphertext<{components = 3 : i64, depth = 3 : i64}>
    %result = "vendor.experimental.refresh"(%product) : (!fhelium_ckks.ciphertext<{components = 3 : i64, depth = 3 : i64}>) -> !fhelium_ckks.ciphertext<{}>
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
        rotation_step=1,
        attributes={"output_domain": StringAttr("ntt")},
    )
    invalid = ir.dialects.ckks.RotateOp(
        block.args[0],
        block.args[1],
        coefficient,
        rotation_step=1,
        attributes={"output_domain": StringAttr("ntt")},
    )
    specification = ir.DEFAULT_OPERATION_SPECS.require(valid.name)
    assert specification.diagnostics(valid) == ()
    assert specification.diagnostics(invalid)


def test_value_type_equality_includes_mathematical_state():
    ciphertext = ir.dialects.ckks.CiphertextType()
    first = ciphertext.with_state(depth=IntegerAttr(0, 64))
    second = ciphertext.with_state(depth=IntegerAttr(1, 64))
    assert first == ciphertext.with_state(depth=IntegerAttr(0, 64))
    assert first != second
    assert len({first, second}) == 2


def test_late_material_facts_refine_fusion_interfaces():
    import torch
    from xdsl.dialects.func import ReturnOp
    from xdsl.ir import Block
    from fhelium.compile import Compilation, ResolveTensorPlaceholdersPass
    from fhelium.ir import Program
    from fhelium.ir.dialects import core, fusion

    typ = core.MessageType()
    material = core.MaterialRefOp(typ, symbol="table")
    body = Block(arg_types=[typ])
    body.add_op(fusion.YieldOp([body.args[0]]))
    grouped = fusion.FusedOp([material.value], [typ], body)
    entry = Block([material, grouped, ReturnOp(grouped.outputs[0])])
    compilation = Compilation(
        Program.from_function(entry, [typ]),
        material_bindings={"table": torch.ones(4, 2)},
    )
    ResolveTensorPlaceholdersPass().run(compilation)
    compilation.program.verify_structure()
    assert grouped.body.block.args[0].type == material.value.type
    assert grouped.outputs[0].type == material.value.type


def test_generated_material_requests_reuse_references_after_program_restore():
    from xdsl.dialects.func import ReturnOp

    from fhelium import Preset
    from fhelium.compile import Compilation
    from fhelium.compile._materials import (
        declare_material,
        parameter_description,
        parameter_request_identity,
        rns_parameter_identity,
    )
    from fhelium.config import CkksConfig
    from fhelium.ir import Program
    from fhelium.ir.dialects import core

    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    physical = {"dtype": StringAttr("torch.int64"), "device": StringAttr("cpu")}
    typ = core.MessageType().with_state(physical)
    rns_description = parameter_description(
        config, "rns_parameters", prime_ids=[0, 1]
    )
    ntt_description = parameter_description(
        config,
        "ntt_table",
        prime_ids=[0, 1],
        direction="forward",
        index=1,
        ntt_backend="radix2_compact_group16_smem8",
    )
    for description, identity in (
        (rns_description, rns_parameter_identity(config, (0, 1), physical)),
        (
            ntt_description,
            parameter_request_identity(ntt_description, physical),
        ),
    ):
        caller = core.MaterialRefOp(typ, symbol="caller")
        block = Block([caller, ReturnOp(caller.value)])
        program = Program.from_function(block, [typ])
        program.set_material_description("caller", description)
        compilation = Compilation(program)
        created, value = declare_material(
            compilation,
            block.last_op,
            "generated",
            typ,
            description,
            identity=identity,
        )
        assert len(created) == 1
        assert value is not caller.value
        block.insert_op_before(created[0], block.last_op)

        for restored in (
            Compilation(program.clone(), compilation.workspace),
            Compilation(Program.parse(program.to_text())),
        ):
            restored_block = restored.program.single_block()
            refs = {
                op.symbol.data: op
                for op in restored_block.ops
                if isinstance(op, core.MaterialRefOp)
            }
            created, value = declare_material(
                restored,
                restored_block.last_op,
                "another_generated",
                typ,
                description,
                identity=identity,
            )
            assert created == ()
            assert value is refs["generated"].value
            assert value is not refs["caller"].value
            restored.program.verify_structure()
