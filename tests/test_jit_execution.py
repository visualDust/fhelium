"""JIT execution readiness and handler tests."""

from __future__ import annotations

import json

import pytest
import torch
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import CkksEngine, EvaluationKeySet, Preset, RotationKeySet
from fhelium.experimental.jit._analysis import analyze_requirements
from fhelium.experimental.jit._capture import capture
from fhelium.experimental.jit._dialect import (
    EncryptedType,
    MaterialRefOp,
    MessageType,
    create_ir_context,
    create_operation,
)
from fhelium.experimental.jit._execution import (
    ProgramNotReadyError,
    check_readiness,
    run_program,
)
from fhelium.experimental.jit._program import Program
from fhelium.experimental.jit._specs import encrypted, message
from fhelium.experimental.jit._workspace import Workspace


def _torch_neg_program() -> Program:
    block = Block(arg_types=[MessageType()])
    block.args[0].name_hint = "x"
    operation = create_operation(
        create_ir_context(),
        "torch.call",
        operands=[block.args[0]],
        result_types=[MessageType()],
        attributes={
            "fhelium.call.kind": StringAttr("function"),
            "fhelium.call.target": StringAttr(
                "torch._VariableFunctionsClass.neg"
            ),
            "fhelium.call.arguments": StringAttr(
                json.dumps(
                    {
                        "args": {
                            "kind": "tuple",
                            "items": [{"kind": "ssa", "operand": 0}],
                        },
                        "kwargs": {"kind": "mapping", "entries": []},
                    }
                )
            ),
        },
    )
    block.add_ops((operation, ReturnOp(operation)))
    return Program.from_function(block, (MessageType(),))


def test_pure_torch_program_runs_only_at_explicit_request() -> None:
    program = _torch_neg_program()
    workspace = Workspace({"caller-note": "retained"})

    requirements = analyze_requirements(program)
    assert requirements.operations == frozenset({"torch.call"})
    assert requirements.torch_targets == frozenset(
        {"torch._VariableFunctionsClass.neg"}
    )
    assert check_readiness(program, workspace).runnable

    value = torch.tensor([1.0, -2.0])
    torch.testing.assert_close(
        run_program(program, value, workspace=workspace), -value
    )
    assert workspace["caller-note"] == "retained"


def test_unknown_operation_is_preserved_until_handler_binding() -> None:
    block = Block(arg_types=[MessageType()])
    operation = create_operation(
        create_ir_context(),
        "vendor.double",
        operands=[block.args[0]],
        result_types=[MessageType()],
    )
    block.add_ops((operation, ReturnOp(operation)))
    program = Program.from_function(block, (MessageType(),))

    report = check_readiness(program)
    assert not report.runnable
    assert report.missing_operations == frozenset({"vendor.double"})
    with pytest.raises(ProgramNotReadyError, match="vendor.double"):
        run_program(program, 4)

    workspace = Workspace(
        {
            "handlers": {
                "vendor.double": lambda op, operands, workspace: operands[0] * 2
            }
        }
    )
    assert run_program(program, 4, workspace=workspace) == 8


def test_extension_handlers_cannot_override_core_operation_semantics() -> None:
    block = Block()
    constant = create_operation(
        create_ir_context(),
        "fhelium.constant",
        result_types=[MessageType()],
        attributes={"fhelium.literal": StringAttr("3")},
    )
    block.add_ops((constant, ReturnOp(constant)))
    program = Program.from_function(block, (MessageType(),))
    workspace = Workspace(
        {
            "handlers": {
                "fhelium.constant": (
                    lambda operation, operands, workspace: "overridden"
                )
            }
        }
    )

    assert run_program(program, workspace=workspace) == 3


def test_material_reference_stays_external_to_program() -> None:
    block = Block()
    material = MaterialRefOp(MessageType(), symbol="weights/q", kind="tensor")
    block.add_ops((material, ReturnOp(material)))
    program = Program.from_function(block, (MessageType(),))

    report = check_readiness(program)
    assert not report.runnable
    assert report.missing_materials == frozenset({"weights/q"})
    assert "weights/q" in program.to_text()

    tensor = torch.tensor([2.0, 3.0])
    workspace = Workspace({"materials": {"weights/q": tensor}})
    assert run_program(program, workspace=workspace) is tensor


def test_incomplete_ckks_program_imports_but_run_reports_requirements() -> None:
    block = Block(arg_types=[EncryptedType()])
    rotate = create_operation(
        create_ir_context(),
        "fhelium.ckks.rotate",
        operands=[block.args[0]],
        result_types=[EncryptedType()],
        attributes={"shift": IntegerAttr(3, 64)},
    )
    block.add_ops((rotate, ReturnOp(rotate)))
    program = Program.from_function(block, (EncryptedType(),))

    reparsed = Program.parse(program.to_text())
    requirements = analyze_requirements(reparsed)
    assert requirements.rotation_steps == frozenset({3})
    assert requirements.requires_engine

    report = check_readiness(reparsed)
    assert not report.runnable
    assert {item.code for item in report.diagnostics} == {
        "missing-engine",
        "missing-evaluation-keys",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_readiness_rejects_evaluation_keys_incompatible_with_engine() -> None:
    block = Block(arg_types=[EncryptedType()])
    rotate = create_operation(
        create_ir_context(),
        "fhelium.ckks.rotate",
        operands=[block.args[0]],
        result_types=[EncryptedType()],
        attributes={"shift": IntegerAttr(3, 64)},
    )
    block.add_ops((rotate, ReturnOp(rotate)))
    program = Program.from_function(block, (EncryptedType(),))

    engine = CkksEngine(Preset.slots8192_scale40_levels7_int64, device="cuda:0")
    incompatible = engine.rotation_key(3).clone()
    incompatible.context_id = "0" * 64
    workspace = Workspace(
        {
            "engine": engine,
            "evaluation_keys": EvaluationKeySet(
                rotations=RotationKeySet({3: incompatible})
            ),
        }
    )

    report = program.readiness(workspace)
    assert not report.runnable
    assert any(
        item.code == "incompatible-evaluation-keys"
        for item in report.diagnostics
    )

    malformed = engine.rotation_key(3).clone()
    malformed.data = malformed.data.to_sparse()
    workspace["evaluation_keys"] = EvaluationKeySet(
        rotations=RotationKeySet({3: malformed})
    )
    report = program.readiness(workspace)
    assert not report.runnable
    assert any(
        item.code == "incompatible-evaluation-keys"
        for item in report.diagnostics
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_run_rejects_public_key_incompatible_with_engine() -> None:
    block = Block(arg_types=[EncryptedType()])
    block.add_op(ReturnOp(block.args[0]))
    program = Program.from_function(block, (EncryptedType(),))

    engine = CkksEngine(Preset.slots8192_scale40_levels7_int64, device="cuda:0")
    incompatible = engine.public_key.clone()
    incompatible.context_id = "0" * 64
    workspace = Workspace(
        {
            "engine": engine,
            "public_key": incompatible,
        }
    )

    with pytest.raises(ProgramNotReadyError) as caught:
        program.run(torch.zeros(engine.num_slots), workspace=workspace)
    assert any(
        item.code == "incompatible-public-key"
        for item in caught.value.report.diagnostics
    )

    malformed = engine.public_key.clone()
    malformed.data = malformed.data[:1]
    workspace["public_key"] = malformed
    with pytest.raises(ProgramNotReadyError) as caught:
        program.run(torch.zeros(engine.num_slots), workspace=workspace)
    assert any(
        item.code == "incompatible-public-key"
        for item in caught.value.report.diagnostics
    )


def test_selected_entry_ignores_unreachable_function_requirements() -> None:
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
            func.return %x : !fhelium.message<{}>
          }
          func.func @unused(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
            %result = "vendor.unused"(%x) : (!fhelium.message<{}>) -> !fhelium.message<{}>
            func.return %result : !fhelium.message<{}>
          }
        }
        '''
    )

    assert program.readiness(entry="main").runnable


def test_unsupported_ckks_name_requires_an_extension_handler() -> None:
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main(%x: !fhelium.encrypted<{}>) -> !fhelium.encrypted<{}> {
            %result = "fhelium.ckks.bootstrap"(%x) : (!fhelium.encrypted<{}>) -> !fhelium.encrypted<{}>
            func.return %result : !fhelium.encrypted<{}>
          }
        }
        '''
    )

    report = program.readiness()
    assert not report.runnable
    assert "fhelium.ckks.bootstrap" in report.missing_operations


def test_bound_resource_reference_is_executable() -> None:
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main() -> !fhelium.resource<{}> {
            %resource = "fhelium.resource.ref"() {symbol = "stream/main", kind = "stream"} : () -> !fhelium.resource<{}>
            func.return %resource : !fhelium.resource<{}>
          }
        }
        '''
    )
    resource = object()
    workspace = Workspace({"resources": {"stream/main": resource}})

    assert program.readiness(workspace).runnable
    assert program.run(workspace=workspace) is resource


def test_fhe_torch_call_requires_explicit_target_handler() -> None:
    captured = capture(
        lambda secret: torch.sin(secret), inputs={"secret": encrypted()}
    )

    report = captured.program.readiness(captured.workspace)
    assert not report.runnable
    assert any(
        item.code == "missing-torch-handler" for item in report.diagnostics
    )


def test_mutating_method_is_not_in_the_safe_public_surface() -> None:
    captured = capture(lambda value: value.clear(), inputs={"value": message()})

    report = captured.program.readiness(captured.workspace)
    assert not report.runnable
    assert any(
        item.code == "missing-torch-handler" for item in report.diagnostics
    )
