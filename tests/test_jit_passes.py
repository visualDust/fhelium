"""Unified xDSL lowering, scheduling, and validation pass tests."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import pytest
from xdsl.dialects.builtin import StringAttr

from fhelium.experimental.jit._capture import capture
from fhelium.experimental.jit._dialect import (
    MessageType,
    create_ir_context,
    create_operation,
    operation_name,
)
from fhelium.experimental.jit._errors import JitPassError
from fhelium.experimental.jit._program import Program
from fhelium.experimental.jit._specs import encrypted, message
from fhelium.experimental.jit._workspace import Workspace
from fhelium.experimental.jit.passes import (
    PassPipeline,
    PassResult,
    ValidateExecutableGraphPass,
    default_pipeline,
)


@dataclass(frozen=True)
class AppendAfterTerminatorPass:
    name: str = "append-after-terminator"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[Any, Any],
    ) -> PassResult:
        del workspace
        trailing = create_operation(
            create_ir_context(),
            "fhelium.constant",
            result_types=(MessageType(),),
            attributes={"fhelium.literal": StringAttr("1")},
        )
        program.entry_block().add_op(trailing)
        return PassResult.unchanged(program)


def _entry_names(program: Program) -> tuple[str, ...]:
    return tuple(
        operation_name(operation)
        for operation in program.entry_block().ops
        if operation_name(operation) != "func.return"
    )


def test_default_pipeline_lowers_ct_ct_multiply_explicitly() -> None:
    def square(secret):
        return secret * secret

    captured = capture(square, inputs={"secret": encrypted()})
    workspace = captured.workspace

    result = default_pipeline().run(captured.program, workspace)

    names = _entry_names(result.program)
    assert names.count("fhelium.ckks.to_ntt") == 2
    assert "fhelium.ckks.multiply" in names
    assert "fhelium.ckks.relinearize" in names
    assert "fhelium.ckks.rescale" in names
    assert not any("semantic" in name or "logical" in name for name in names)
    assert result.workspace is workspace
    assert tuple(report.name for report in result.reports) == (
        "eliminate-dead-values",
        "lower-semantic-to-logical",
        "insert-plaintext-preparation",
        "insert-multiply-ntt-transitions",
        "lower-logical-to-ckks",
        "insert-relinearization",
        "insert-rescale",
        "late-rescale",
        "late-relinearization",
    )

    ValidateExecutableGraphPass().run(result.program, workspace)


def test_default_pipeline_prepares_mixed_addition() -> None:
    def add_public(secret, public):
        return secret + public

    captured = capture(
        add_public,
        inputs={"secret": encrypted(), "public": message()},
    )
    result = default_pipeline().run(
        captured.program,
        captured.workspace,
    )

    names = _entry_names(result.program)
    assert "fhelium.ckks.prepare.add.message" in names
    assert "fhelium.ckks.add_plaintext" in names
    assert not any("semantic" in name or "logical" in name for name in names)
    ValidateExecutableGraphPass().run(result.program, result.workspace)


def test_default_pipeline_makes_plaintext_multiply_transitions_explicit() -> (
    None
):
    def multiply_public(secret, public):
        return secret * public

    captured = capture(
        multiply_public,
        inputs={"secret": encrypted(), "public": message()},
    )
    result = default_pipeline().run(captured.program, captured.workspace)

    names = _entry_names(result.program)
    assert names.count("fhelium.ckks.to_ntt") == 1
    assert names.count("fhelium.ckks.multiply_plaintext") == 1
    assert names.count("fhelium.ckks.from_ntt") == 1
    assert names.count("fhelium.ckks.rescale") == 1
    assert names.index("fhelium.ckks.to_ntt") < names.index(
        "fhelium.ckks.multiply_plaintext"
    )
    assert names.index("fhelium.ckks.multiply_plaintext") < names.index(
        "fhelium.ckks.from_ntt"
    )
    ValidateExecutableGraphPass().run(result.program, result.workspace)


def test_default_pipeline_is_structurally_idempotent() -> None:
    def square(secret):
        return secret * secret

    captured = capture(square, inputs={"secret": encrypted()})
    first = default_pipeline().run(captured.program, captured.workspace)
    second = default_pipeline().run(first.program, first.workspace)

    assert second.program.to_text() == first.program.to_text()
    assert second.workspace is first.workspace


def test_pipeline_rejects_structurally_invalid_extension_pass_result() -> None:
    captured = capture(lambda value: value, inputs={"value": message()})

    with pytest.raises(JitPassError, match="structurally invalid xDSL"):
        PassPipeline((AppendAfterTerminatorPass(),)).run(captured.program)


def test_validator_rejects_unknown_operation_until_handler_is_supplied() -> (
    None
):
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
            %result = "vendor.custom"(%x) : (!fhelium.message<{}>) -> !fhelium.message<{}>
            func.return %result : !fhelium.message<{}>
          }
        }
        '''
    )

    try:
        ValidateExecutableGraphPass().run(program, Workspace())
    except Exception as error:
        assert "no handler" in str(error)
    else:
        raise AssertionError("validator unexpectedly accepted vendor.custom")

    workspace = Workspace(
        {
            "handlers": {
                "vendor.custom": lambda operation, operands, retained: None
            }
        }
    )
    result = PassPipeline((ValidateExecutableGraphPass(),)).run(
        program, workspace
    )
    assert result.workspace is workspace
    assert "vendor.custom" in result.program.to_text()


def test_unknown_effectful_operation_survives_default_dce() -> None:
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
            %effect = "vendor.effect"(%x) : (!fhelium.message<{}>) -> !fhelium.message<{}>
            func.return %x : !fhelium.message<{}>
          }
        }
        '''
    )

    result = default_pipeline().run(program)

    assert "vendor.effect" in result.program.to_text()


def test_unknown_extension_type_is_not_reclassified_as_message() -> None:
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main(%x: !fhelium.encrypted<{}>, %m: !vendor.unknown) -> !fhelium.encrypted<{}> {
            %result = "fhelium.semantic.add"(%x, %m) : (!fhelium.encrypted<{}>, !vendor.unknown) -> !fhelium.encrypted<{}>
            func.return %result : !fhelium.encrypted<{}>
          }
        }
        '''
    )

    result = default_pipeline().run(program)
    text = result.program.to_text()

    assert '"fhelium.semantic.add"' in text
    assert "fhelium.ckks.prepare.add.message" not in text


def test_fractional_static_multiplication_uses_scaled_plaintext_and_rescale() -> (
    None
):
    captured = capture(
        lambda secret: secret * 0.5,
        inputs={"secret": encrypted()},
    )

    result = default_pipeline().run(captured.program, captured.workspace)
    text = result.program.to_text()

    assert 'scale_mode = "default_scale"' in text
    assert "unit_scale" not in text
    assert '"fhelium.ckks.rescale"' in text
