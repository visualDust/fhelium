"""CKKS polynomial-domain and residue-representation state tests."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block, Operation

from fhelium import compile as fh_compile
from fhelium import ir
from fhelium.config import CkksConfig, Preset


def _config() -> CkksConfig:
    source = CkksConfig.parse(Preset.slots16384_scale50_depth12_int64)
    return CkksConfig(
        default_scale=source.default_scale,
        q_depth_groups=source.q_depth_groups[:6],
        p_moduli=source.p_moduli,
        logN=12,
        enforce_security_budget=False,
    )


def _ciphertext_type(
    domain: str,
    residues: str,
    *,
    components: int = 2,
) -> ir.dialects.ckks.CiphertextType:
    return ir.dialects.ckks.CiphertextType().with_state(
        depth=IntegerAttr(0, 64),
        scale=FloatAttr(4.0, Float64Type()),
        prime_ids=ArrayAttr(IntegerAttr(index, 64) for index in range(5)),
        basis=StringAttr("Q"),
        polynomial_domain=StringAttr(domain),
        residue_representation=StringAttr(residues),
        components=IntegerAttr(components, 64),
        ring_dimension=IntegerAttr(4096, 64),
    )


def _plaintext_type(
    domain: str,
    residues: str,
) -> ir.dialects.ckks.PlaintextType:
    return ir.dialects.ckks.PlaintextType().with_state(
        depth=IntegerAttr(0, 64),
        scale=FloatAttr(4.0, Float64Type()),
        prime_ids=ArrayAttr(IntegerAttr(index, 64) for index in range(5)),
        basis=StringAttr("Q"),
        representation=StringAttr("rns"),
        polynomial_domain=StringAttr(domain),
        residue_representation=StringAttr(residues),
    )


def test_depth_and_scale_assignment_preserve_cast_target_representation() -> (
    None
):
    coefficient = _ciphertext_type("coefficient", "standard")
    ntt = ir.dialects.ckks.CiphertextType().with_state(
        components=IntegerAttr(2, 64),
        polynomial_domain=StringAttr("ntt"),
        residue_representation=StringAttr("montgomery"),
    )
    block = Block(arg_types=(coefficient,))
    cast, result = UnrealizedConversionCastOp.cast_one(block.args[0], ntt)
    block.add_ops((cast, ReturnOp(result)))
    program = ir.Program.from_function(block, (ntt,))
    workspace: dict[object, object] = {CkksConfig: _config()}

    fh_compile.AssignCkksDepthsPass(0).run(program, workspace)
    fh_compile.AssignCkksScalesPass(4.0).run(program, workspace)

    state = cast.outputs[0].type.state.data  # type: ignore[attr-defined]
    assert state["polynomial_domain"] == StringAttr("ntt")
    assert state["residue_representation"] == StringAttr("montgomery")
    assert "basis" not in state


def test_representation_defaults_and_state_analysis_match_plaintext_inverse() -> (
    None
):
    ntt = _plaintext_type("ntt", "montgomery")
    block = Block(arg_types=(ntt,))
    inverse = ir.dialects.ckks.FromNttOp(block.args[0])
    block.add_ops((inverse, ReturnOp(inverse.result)))
    program = ir.Program.from_function(block, (inverse.result.type,))

    result_state = inverse.result.type.state.data  # type: ignore[attr-defined]
    assert result_state["polynomial_domain"] == StringAttr("coefficient")
    assert result_state["residue_representation"] == StringAttr("montgomery")
    inferred = ir.analyze_state_flow(program)[inverse.result]
    assert inferred.field("polynomial_domain") == ir.StateFact.known(
        "coefficient"
    )
    assert inferred.field("residue_representation") == ir.StateFact.known(
        "montgomery"
    )


def test_rescale_analysis_preserves_ntt_montgomery_output() -> None:
    ntt = _ciphertext_type("ntt", "montgomery")
    rescaled_type = ntt.with_state(
        depth=IntegerAttr(1, 64),
        prime_ids=ArrayAttr(IntegerAttr(index, 64) for index in range(1, 5)),
    )
    block = Block(arg_types=(ntt,))
    rescale = ir.dialects.ckks.RescaleOp(
        block.args[0],
        rescaled_type,
        polynomial_domain="ntt",
    )
    block.add_ops((rescale, ReturnOp(rescale.result)))
    program = ir.Program.from_function(block, (rescaled_type,))

    inferred = ir.analyze_state_flow(program)[rescale.result]
    assert inferred.field("polynomial_domain") == ir.StateFact.known("ntt")
    assert inferred.field("residue_representation") == ir.StateFact.known(
        "montgomery"
    )
    fh_compile.ValidateExecutionRepresentationsPass().run(program, {})
    fh_compile.LowerCkksToRnsNttPass().run(
        program,
        {CkksConfig: _config()},
    )
    lowered = next(
        operation
        for operation in program.single_block().ops
        if isinstance(operation, ir.dialects.rns.RescaleDropLeadingPrimesOp)
    )
    assert lowered.input_domain == StringAttr("ntt")
    assert lowered.output_domain == StringAttr("ntt")


def test_rescale_lowering_preserves_the_requested_prime_group() -> None:
    source = _config()
    grouped = CkksConfig(
        default_scale=source.default_scale,
        q_depth_groups=(
            (*source.q_depth_groups[0], *source.q_depth_groups[1]),
            *source.q_depth_groups[2:],
        ),
        p_moduli=source.p_moduli,
        logN=source.logN,
        enforce_security_budget=False,
    )
    input_type = _ciphertext_type("coefficient", "standard").with_state(
        depth=IntegerAttr(0, 64),
        prime_ids=ArrayAttr(
            IntegerAttr(index, 64) for index in range(grouped.num_q_primes)
        ),
    )
    output_type = input_type.with_state(
        depth=IntegerAttr(1, 64),
        prime_ids=ArrayAttr(
            IntegerAttr(index, 64)
            for index in range(2, grouped.num_q_primes)
        ),
    )
    block = Block(arg_types=(input_type,))
    rescale = ir.dialects.ckks.RescaleOp(block.args[0], output_type)
    block.add_ops((rescale, ReturnOp(rescale.result)))
    program = ir.Program.from_function(block, (output_type,))

    fh_compile.LowerCkksToRnsNttPass().run(
        program, {CkksConfig: grouped}
    )

    drops = [
        operation
        for operation in program.single_block().ops
        if isinstance(operation, ir.dialects.rns.RescaleDropLeadingPrimesOp)
    ]
    assert sum(drop.drop_count.value.data for drop in drops) == 2


def test_multiply_transition_reuses_one_ntt_value_across_consumers() -> None:
    state = _ciphertext_type("coefficient", "standard").state.data
    encrypted = ir.dialects.logical.EncryptedType().with_state(state)
    block = Block(arg_types=(encrypted, encrypted, encrypted))
    first = ir.dialects.logical.MultiplyEncryptedEncryptedOp(
        block.args[0], block.args[1]
    )
    second = ir.dialects.logical.MultiplyEncryptedEncryptedOp(
        block.args[0], block.args[2]
    )
    block.add_ops((first, second, ReturnOp(first.result, second.result)))
    program = ir.Program.from_function(
        block, (first.result.type, second.result.type)
    )

    fh_compile.InsertMultiplyNttTransitionsPass().run(program, {})

    transitions = tuple(
        operation
        for operation in program.single_block().ops
        if isinstance(operation, ir.dialects.ckks.ToNttOp)
    )
    assert len(transitions) == 3
    assert first.lhs is second.lhs


def test_captured_input_can_declare_executable_representation() -> None:
    def identity(value: object) -> object:
        return value

    captured = fh_compile.capture(
        identity,
        inputs={
            "value": fh_compile.encrypted(
                polynomial_domain="ntt",
                residue_representation="montgomery",
            )
        },
    )

    argument_type = captured.program.single_block().args[0].type
    state = argument_type.state.data  # type: ignore[attr-defined]
    assert state["polynomial_domain"] == StringAttr("ntt")
    assert state["residue_representation"] == StringAttr("montgomery")


def test_ntt_relinearization_lowering_keeps_real_ntt_dataflow() -> None:
    source_type = _ciphertext_type("ntt", "montgomery", components=3)
    result_type = _ciphertext_type("ntt", "montgomery", components=2)
    block = Block(arg_types=(source_type,))
    relinearize = ir.dialects.ckks.RelinearizeOp(
        block.args[0], result_type, output_domain="ntt"
    )
    block.add_ops((relinearize, ReturnOp(relinearize.result)))
    program = ir.Program.from_function(block, (result_type,))

    fh_compile.LowerCkksToRnsNttPass().run(program, {CkksConfig: _config()})
    fh_compile.ValidateExecutionRepresentationsPass().run(program, {})

    assert any(
        isinstance(operation, ir.dialects.rns.ModDownNttQpToQOp)
        for operation in program.single_block().ops
    )


@pytest.mark.parametrize(
    ("value_type", "expected_operation"),
    (
        (
            _plaintext_type("coefficient", "standard"),
            ir.dialects.ntt.CoefficientStandardToNttMontgomeryOp,
        ),
        (
            _plaintext_type("coefficient", "montgomery"),
            ir.dialects.ntt.CoefficientMontgomeryToNttMontgomeryOp,
        ),
    ),
)
def test_ntt_lowering_selects_input_residue_conversion(
    value_type: ir.dialects.ckks.PlaintextType,
    expected_operation: type[Operation],
) -> None:
    result_type = value_type.with_state(
        polynomial_domain=StringAttr("ntt"),
        residue_representation=StringAttr("montgomery"),
    )
    block = Block(arg_types=(value_type,))
    transform = ir.dialects.ckks.ToNttOp(block.args[0], result_type)
    block.add_ops((transform, ReturnOp(transform.result)))
    program = ir.Program.from_function(block, (result_type,))

    fh_compile.LowerCkksToRnsNttPass().run(
        program,
        {CkksConfig: _config()},
    )

    assert any(
        isinstance(operation, expected_operation)
        for operation in program.single_block().ops
    )


def test_ntt_lowering_and_execution_gate_reject_unknown_representation() -> (
    None
):
    unknown = ir.dialects.ckks.CiphertextType().with_state(
        components=IntegerAttr(3, 64),
        depth=IntegerAttr(0, 64),
    )
    block = Block(arg_types=(unknown,))
    relinearize = ir.dialects.ckks.RelinearizeOp(
        block.args[0],
        _ciphertext_type("coefficient", "standard"),
    )
    block.add_ops((relinearize, ReturnOp(relinearize.result)))
    program = ir.Program.from_function(block, (relinearize.result.type,))

    with pytest.raises(ValueError, match="requires concrete"):
        fh_compile.ValidateExecutionRepresentationsPass().run(program, {})

    transform_block = Block(arg_types=(unknown,))
    transform = ir.dialects.ckks.ToNttOp(transform_block.args[0])
    transform_block.add_ops((transform, ReturnOp(transform.result)))
    transform_program = ir.Program.from_function(
        transform_block, (transform.result.type,)
    )
    with pytest.raises(ValueError, match="requires concrete"):
        fh_compile.LowerCkksToRnsNttPass().run(
            transform_program,
            {CkksConfig: _config()},
        )


def test_multiply_transition_does_not_cast_unknown_state_to_coefficients() -> (
    None
):
    encrypted = ir.dialects.logical.EncryptedType()
    block = Block(arg_types=(encrypted, encrypted))
    multiply = ir.dialects.logical.MultiplyEncryptedEncryptedOp(
        block.args[0], block.args[1]
    )
    block.add_ops((multiply, ReturnOp(multiply.result)))
    program = ir.Program.from_function(block, (multiply.result.type,))

    result = fh_compile.InsertMultiplyNttTransitionsPass().run(program, {})

    assert result.stats.skipped == 1
    assert result.diagnostics
    assert not any(
        isinstance(operation, ir.dialects.ckks.ToNttOp)
        for operation in program.single_block().ops
    )


def test_decrypt_input_domain_is_registered_and_checked_by_execution_gate() -> (
    None
):
    coefficient = _ciphertext_type("coefficient", "standard")
    plaintext = _plaintext_type("coefficient", "standard")
    block = Block(arg_types=(coefficient,))
    decrypt = ir.dialects.ckks.DecryptOp.create(
        operands=(block.args[0],),
        result_types=(plaintext,),
        attributes={
            "key_symbol": StringAttr("secret-key"),
            "input_domain": StringAttr("coefficient"),
        },
    )
    specification = ir.DEFAULT_OPERATION_SPECS.require(decrypt.name)
    assert specification.diagnostics(decrypt) == ()
    block.add_ops((decrypt, ReturnOp(decrypt.result)))
    program = ir.Program.from_function(block, (plaintext,))
    fh_compile.ValidateExecutionRepresentationsPass().run(
        program,
        {},
    )

    decrypt.attributes["input_domain"] = StringAttr("ntt")
    assert specification.diagnostics(decrypt) == ()
    with pytest.raises(ValueError, match="expected one of"):
        fh_compile.ValidateExecutionRepresentationsPass().run(
            program,
            {},
        )
