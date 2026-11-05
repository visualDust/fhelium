"""CKKS polynomial-domain and residue-representation state tests."""

from __future__ import annotations

from fhelium import compile as _compile

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

    fh_compile.AssignCkksDepthsPass(0).run(
        _compile.Compilation(program, _compile.CompileWorkspace(workspace))
    )
    fh_compile.AssignCkksScalesPass(4.0).run(
        _compile.Compilation(program, _compile.CompileWorkspace(workspace))
    )

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
    fh_compile.ValidateExecutionRepresentationsPass().run(
        _compile.Compilation(program, _compile.CompileWorkspace({}))
    )
    fh_compile.LowerCkksToRnsNttPass().run(
        _compile.Compilation(
            program, _compile.CompileWorkspace({CkksConfig: _config()})
        )
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
            IntegerAttr(index, 64) for index in range(2, grouped.num_q_primes)
        ),
    )
    block = Block(arg_types=(input_type,))
    rescale = ir.dialects.ckks.RescaleOp(block.args[0], output_type)
    block.add_ops((rescale, ReturnOp(rescale.result)))
    program = ir.Program.from_function(block, (output_type,))

    fh_compile.LowerCkksToRnsNttPass().run(
        _compile.Compilation(
            program, _compile.CompileWorkspace({CkksConfig: grouped})
        )
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

    fh_compile.InsertMultiplyNttTransitionsPass().run(
        _compile.Compilation(program, _compile.CompileWorkspace({}))
    )

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


@pytest.mark.parametrize("rounding", [None, "floor", "nearest"])
@pytest.mark.parametrize("domain", ["coefficient", "ntt"])
def test_relinearization_lowering_preserves_moddown_rounding(rounding, domain):
    source_type = _ciphertext_type("ntt", "montgomery", components=3)
    result_type = _ciphertext_type(
        domain, "montgomery" if domain == "ntt" else "standard", components=2
    )
    qids = ArrayAttr(IntegerAttr(i, 64) for i in range(len(_config().q_moduli)))
    source_type = source_type.with_state(prime_ids=qids)
    result_type = result_type.with_state(prime_ids=qids)
    block = Block(arg_types=(source_type,))
    key = ir.dialects.core.MaterialRefOp(
        ir.dialects.ckks.EvaluationKeyType().with_state(
            prime_ids=ArrayAttr(
                IntegerAttr(i, 64) for i in range(len(_config().moduli))
            ),
            digit_prime_ids=ArrayAttr(ArrayAttr((qid,)) for qid in qids),
        ),
        symbol="relinearization-key",
    )
    inferred = ir.dialects.ckks.RelinearizeOp(
        block.args[0],
        key.value,
        attributes={"output_domain": StringAttr(domain)},
    )
    assert inferred.result.type.state.data["polynomial_domain"] == StringAttr(
        domain
    )
    assert inferred.result.type.state.data[
        "residue_representation"
    ] == StringAttr("montgomery" if domain == "ntt" else "standard")
    inferred.erase()
    relinearize = ir.dialects.ckks.RelinearizeOp(
        block.args[0], key.value, result_type, output_domain=domain
    )
    if rounding is None:
        relinearize.attributes.pop("moddown_rounding", None)
    else:
        relinearize.attributes["moddown_rounding"] = StringAttr(rounding)
    block.add_ops((key, relinearize, ReturnOp(relinearize.result)))
    program = ir.Program.from_function(block, (result_type,))
    program = ir.Program.parse(program.to_text())
    fh_compile.LowerCkksToRnsNttPass().run(
        _compile.Compilation(
            program, _compile.CompileWorkspace({CkksConfig: _config()})
        )
    )
    fh_compile.ValidateExecutionRepresentationsPass().run(
        _compile.Compilation(program, _compile.CompileWorkspace({}))
    )
    cls = (
        ir.dialects.rns.ModDownNttQpToQOp
        if domain == "ntt"
        else ir.dialects.rns.ModDownQpToQOp
    )
    moddown = next(
        op for op in program.single_block().ops if isinstance(op, cls)
    )
    if rounding is None:
        moddown.attributes.pop("rounding", None)
        program = ir.Program.parse(program.to_text())
        moddown = next(
            op for op in program.single_block().ops if isinstance(op, cls)
        )
    assert moddown.rounding.data == (rounding or "nearest")
    names = [name.data for name in moddown.attributes["parameter_names"]]
    assert ("moddown_half" in names) == (rounding != "floor")


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
        _compile.Compilation(
            program, _compile.CompileWorkspace({CkksConfig: _config()})
        )
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
    key = ir.dialects.core.MaterialRefOp(
        ir.dialects.ckks.EvaluationKeyType(), symbol="relinearization-key"
    )
    relinearize = ir.dialects.ckks.RelinearizeOp(
        block.args[0],
        key.value,
        _ciphertext_type("coefficient", "standard"),
    )
    block.add_ops((key, relinearize, ReturnOp(relinearize.result)))
    program = ir.Program.from_function(block, (relinearize.result.type,))

    with pytest.raises(ValueError, match="requires concrete"):
        fh_compile.ValidateExecutionRepresentationsPass().run(
            _compile.Compilation(program, _compile.CompileWorkspace({}))
        )

    transform_block = Block(arg_types=(unknown,))
    transform = ir.dialects.ckks.ToNttOp(transform_block.args[0])
    transform_block.add_ops((transform, ReturnOp(transform.result)))
    transform_program = ir.Program.from_function(
        transform_block, (transform.result.type,)
    )
    with pytest.raises(ValueError, match="requires concrete"):
        fh_compile.LowerCkksToRnsNttPass().run(
            _compile.Compilation(
                transform_program,
                _compile.CompileWorkspace({CkksConfig: _config()}),
            )
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

    result = fh_compile.InsertMultiplyNttTransitionsPass().run(
        _compile.Compilation(program, _compile.CompileWorkspace({}))
    )

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
    block = Block(arg_types=(coefficient, ir.dialects.core.MessageType()))
    decrypt = ir.dialects.ckks.DecryptOp.create(
        operands=tuple(block.args),
        result_types=(plaintext,),
        attributes={
            "input_domain": StringAttr("coefficient"),
        },
    )
    specification = ir.DEFAULT_OPERATION_SPECS.require(decrypt.name)
    assert specification.diagnostics(decrypt) == ()
    block.add_ops((decrypt, ReturnOp(decrypt.result)))
    program = ir.Program.from_function(block, (plaintext,))
    fh_compile.ValidateExecutionRepresentationsPass().run(
        _compile.Compilation(program, _compile.CompileWorkspace({}))
    )

    decrypt.attributes["input_domain"] = StringAttr("ntt")
    assert specification.diagnostics(decrypt) == ()
    with pytest.raises(ValueError, match="expected one of"):
        fh_compile.ValidateExecutionRepresentationsPass().run(
            _compile.Compilation(program, _compile.CompileWorkspace({}))
        )


def test_rotation_hoisting_preserves_distinct_rounding_groups():
    """Rotations sharing an input retain each group's requested quotient rule."""
    ckks = ir.dialects.ckks
    value_type = _ciphertext_type("coefficient", "standard")
    key_types = [
        ckks.EvaluationKeyType().with_state(
            rotation_step=IntegerAttr(step, 64),
            prime_ids=ArrayAttr(IntegerAttr(i, 64) for i in range(6)),
            digit_prime_ids=ArrayAttr(
                ArrayAttr((IntegerAttr(i, 64),)) for i in range(5)
            ),
        )
        for step in (1, 2)
    ]
    block = Block(arg_types=(value_type, *key_types))
    rotations = [
        ckks.RotateOp(block.args[0], key, moddown_rounding=rounding)
        for rounding in ("floor", "nearest")
        for key in block.args[1:]
    ]
    block.add_ops((*rotations, ReturnOp(*(op.result for op in rotations))))
    program = ir.Program.from_function(block, (value_type,) * 4)
    fh_compile.RotationHoistingPass().run(
        _compile.Compilation(program, _compile.CompileWorkspace({}))
    )
    groups = [op for op in program.walk() if isinstance(op, ckks.RotateManyOp)]
    assert sorted(
        (op.moddown_rounding.data, len(op.outputs)) for op in groups
    ) == [
        ("floor", 2),
        ("nearest", 2),
    ]


@pytest.mark.parametrize("rescale", [False, True])
def test_rotation_sum_assigns_complete_depth_group_and_key_requirements(
    rescale,
):
    source = _config()
    config = CkksConfig(
        logN=source.logN,
        q_depth_groups=(source.q_moduli[:2], *source.q_depth_groups[2:]),
        p_moduli=source.p_moduli,
        default_scale=source.default_scale,
        enforce_security_budget=False,
    )
    ciphertext = _ciphertext_type("coefficient", "standard").with_state(
        prime_ids=ArrayAttr(
            IntegerAttr(i, 64) for i in range(config.num_q_primes)
        )
    )
    key = ir.dialects.ckks.EvaluationKeyType()
    block = Block(arg_types=(ciphertext, ciphertext, key, key))
    result_type = ir.dialects.ckks.CiphertextType().with_state(
        components=IntegerAttr(2, 64),
        basis=StringAttr("Q"),
        polynomial_domain=StringAttr("coefficient"),
        residue_representation=StringAttr("standard"),
    )
    operation = ir.dialects.ckks.RotationSumOp(
        block.args[:2], block.args[2:], (1, -3), result_type, rescale=rescale
    )
    block.add_ops((operation, ReturnOp(operation.result)))
    program = ir.Program.from_function(block, (result_type,))
    compilation = _compile.Compilation(
        program, _compile.CompileWorkspace({CkksConfig: config})
    )
    fh_compile.AssignCkksDepthsPass().run(compilation)
    fh_compile.AssignCkksScalesPass().run(compilation)
    state = operation.result.type.state.data
    assert state["depth"] == IntegerAttr(int(rescale), 64)
    assert state["scale"] == FloatAttr(
        4.0 / config.rescale_divisor(0) if rescale else 4.0, Float64Type()
    )
    assert tuple(i.value.data for i in state["prime_ids"]) == tuple(
        range(2 if rescale else 0, config.num_q_primes)
    )
    requirements = ir.analyze_evaluation_key_requirements(program)
    assert requirements.rotation_steps == frozenset((1, -3))
