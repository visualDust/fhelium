"""Essential source-oriented compilation tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.dialects.func import ReturnOp
from xdsl.ir import Block

from fhelium import Preset
from fhelium.config import CkksConfig
from fhelium import compile as fh_compile
from fhelium import ir

_CAPTURE_WEIGHT = torch.tensor([1.25, -0.5], dtype=torch.float64)


def _operation_names(program: ir.Program) -> tuple[str, ...]:
    return tuple(ir.operation_name(operation) for operation in program.walk())


def _semantic_to_ckks_passes() -> tuple[ir.Pass, ...]:
    return (
        fh_compile.EliminateDeadValuesPass(),
        fh_compile.LowerSemanticToLogicalPass(),
        fh_compile.InsertPlaintextPreparationPass(),
        fh_compile.InsertMultiplyNttTransitionsPass(),
        fh_compile.LowerLogicalToCkksPass(),
        fh_compile.InsertRelinearizationPass(),
        fh_compile.InsertRescalePass(),
    )


def test_capture_retains_the_python_reference_and_emits_registered_semantics() -> (
    None
):
    def weighted(
        secret: torch.Tensor,
        bias: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        value = secret * _CAPTURE_WEIGHT + bias
        return -value if enabled else value

    captured = fh_compile.capture(
        weighted,
        inputs={
            "secret": fh_compile.encrypted(),
            "bias": fh_compile.message(),
            "enabled": fh_compile.static(True),
        },
    )

    captured_callable = captured.workspace[fh_compile.CapturedCallable]
    assert isinstance(captured_callable, fh_compile.CapturedCallable)
    assert tuple(captured_callable.runtime_signature.parameters) == (
        "secret",
        "bias",
    )
    assert any(
        isinstance(operation, ir.dialects.semantic.MultiplyOp)
        for operation in captured.program.walk()
    )
    expected = weighted(
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        torch.tensor([0.25, 0.5], dtype=torch.float64),
        True,
    )
    actual = captured_callable.reference(
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        torch.tensor([0.25, 0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(actual, expected)


def test_pipeline_retains_workspace_and_appends_reports() -> None:
    captured = fh_compile.capture(
        lambda value: -value,
        inputs={"value": fh_compile.encrypted()},
    )
    captured_callable = captured.workspace[fh_compile.CapturedCallable]

    first = fh_compile.Pipeline((fh_compile.EliminateDeadValuesPass(),)).run(
        captured
    )
    second = fh_compile.Pipeline(
        (fh_compile.LowerSemanticToLogicalPass(),)
    ).run(first)

    assert first is not captured
    assert first.program is not captured.program
    assert second.workspace is captured.workspace
    assert second.workspace[fh_compile.CapturedCallable] is captured_callable
    assert tuple(report.name for report in second.reports) == (
        "eliminate-dead-values",
        "lower-semantic-to-logical",
    )


def test_compile_partially_lowers_understood_operations() -> None:
    def mixed(secret: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        return torch.sin(secret) + public

    result = fh_compile.compile(
        mixed,
        inputs={
            "secret": fh_compile.encrypted(),
            "public": fh_compile.message(),
        },
        pipeline=fh_compile.Pipeline(_semantic_to_ckks_passes()),
    )

    names = _operation_names(result.program)
    assert "torch.call" in names
    assert "fhelium_ckks.prepare.add.message" in names
    assert "fhelium_ckks.add_plaintext" in names


def test_lower_ckks_pass_waits_for_config_in_compile_workspace() -> None:
    def add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right

    pipeline = fh_compile.Pipeline(
        (
            *_semantic_to_ckks_passes(),
            fh_compile.LowerCkksToRnsNttPass(),
        )
    )
    workspace = fh_compile.CompileWorkspace()

    result = fh_compile.compile(
        add,
        inputs={
            "left": fh_compile.encrypted(),
            "right": fh_compile.encrypted(),
        },
        pipeline=pipeline,
        workspace=workspace,
    )

    assert result.workspace is workspace
    assert "fhelium_ckks.add" in _operation_names(result.program)
    assert "fhelium_rns.add_standard" not in _operation_names(result.program)


def test_pass_can_select_ckks_config_after_ckks_ir_exists() -> None:
    def add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right

    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    caller_value = object()

    @dataclass(frozen=True)
    class SelectConfigFromCkksProgram:
        name: str = "select-config-from-ckks-program"

        def run(
            self,
            program: ir.Program,
            workspace: dict[object, object],
        ) -> fh_compile.PassResult:
            assert "fhelium_ckks.add" in _operation_names(program)
            assert workspace["caller/value"] is caller_value
            workspace[CkksConfig] = config
            return fh_compile.PassResult.unchanged(program, matched=1)

    workspace = fh_compile.CompileWorkspace({"caller/value": caller_value})
    pipeline = fh_compile.Pipeline(
        (
            *_semantic_to_ckks_passes(),
            SelectConfigFromCkksProgram(),
            fh_compile.LowerCkksToRnsNttPass(),
        )
    )

    result = fh_compile.compile(
        add,
        inputs={
            "left": fh_compile.encrypted(),
            "right": fh_compile.encrypted(),
        },
        pipeline=pipeline,
        workspace=workspace,
    )

    assert result.workspace is workspace
    assert workspace["caller/value"] is caller_value
    assert "fhelium_rns.add_standard" in _operation_names(result.program)


def test_backend_assignment_is_partial_and_persists_in_program_text() -> None:
    source = _rotation_program((1,))
    result = fh_compile.compile(
        source,
        pipeline=fh_compile.Pipeline(
            (
                fh_compile.AssignImplementationsPass(
                    {"fhelium_ckks.rotate": "native-direct-rotate"}
                ),
            )
        ),
    )

    rotation = next(
        operation
        for operation in result.program.walk()
        if isinstance(operation, ir.dialects.ckks.RotateOp)
    )
    assignment = rotation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE]
    assert isinstance(assignment, StringAttr)
    assert assignment.data == "native-direct-rotate"
    reparsed = ir.parse(result.program.to_text())
    reparsed_rotation = next(
        operation
        for operation in reparsed.walk()
        if isinstance(operation, ir.dialects.ckks.RotateOp)
    )
    reparsed_assignment = reparsed_rotation.attributes[
        ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE
    ]
    assert isinstance(reparsed_assignment, StringAttr)
    assert reparsed_assignment.data == "native-direct-rotate"


def test_backend_assignment_rejects_conflict_unless_overwrite_is_selected() -> (
    None
):
    source = _rotation_program((1,))
    first = fh_compile.compile(
        source,
        pipeline=fh_compile.Pipeline(
            (
                fh_compile.AssignImplementationsPass(
                    {"fhelium_ckks.rotate": "first"}
                ),
            )
        ),
    ).program
    with pytest.raises(ValueError):
        fh_compile.compile(
            first,
            pipeline=fh_compile.Pipeline(
                (
                    fh_compile.AssignImplementationsPass(
                        {"fhelium_ckks.rotate": "second"}
                    ),
                )
            ),
        )

    replaced = fh_compile.compile(
        first,
        pipeline=fh_compile.Pipeline(
            (
                fh_compile.AssignImplementationsPass(
                    {"fhelium_ckks.rotate": "second"},
                    overwrite=True,
                ),
            )
        ),
    )
    rotation = next(
        operation
        for operation in replaced.program.walk()
        if isinstance(operation, ir.dialects.ckks.RotateOp)
    )
    assignment = rotation.attributes[ir.EXECUTION_IMPLEMENTATION_ATTRIBUTE]
    assert isinstance(assignment, StringAttr)
    assert assignment.data == "second"


def _rotation_program(
    shifts: tuple[int, ...],
    *,
    key_symbol_index: int | None = None,
    ring_dimension: int | None = 32,
) -> ir.Program:
    result_names = ("rotation_one", "rotation_two", "rotation_three")
    ciphertext_type = ir.dialects.ckks.CiphertextType()
    if ring_dimension is not None:
        ciphertext_type = ciphertext_type.with_state(
            {"ring_dimension": IntegerAttr(ring_dimension, 64)}
        )
    block = Block(arg_types=(ciphertext_type,))
    rotations = []
    key_references = []
    for index, shift in enumerate(shifts):
        symbol = (
            "caller-key"
            if index == key_symbol_index
            else f"rotation-key:{shift}"
        )
        key_type = ir.dialects.ckks.EvaluationKeyType().with_state(
            {"rotation_step": IntegerAttr(shift, 64)}
        )
        key_reference = ir.dialects.core.ResourceRefOp(
            key_type,
            symbol=symbol,
            kind="rotation-key",
        )
        rotation = ir.dialects.ckks.RotateOp(
            block.args[0],
            key_reference.value,
            ciphertext_type,
        )
        rotation.result.name_hint = (
            result_names[index]
            if index < len(result_names)
            else f"rotation_extra_{index}"
        )
        key_references.append(key_reference)
        rotations.append(rotation)
    block.add_ops(
        (
            *(
                operation
                for pair in zip(key_references, rotations, strict=True)
                for operation in pair
            ),
            ReturnOp(*(item.result for item in rotations)),
        )
    )
    return ir.Program.from_function(
        block,
        tuple(ciphertext_type for _ in rotations),
    )
