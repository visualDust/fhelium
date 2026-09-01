"""PyTorch capture into neutral mixed-level IR tests."""

from __future__ import annotations

import json

import torch
from xdsl.dialects.builtin import StringAttr

from fhelium import compile as fh_compile
from fhelium import ir


def _operation_names(program: ir.Program) -> tuple[str, ...]:
    return tuple(ir.operation_name(operation) for operation in program.walk())


def _captured_callable(
    compilation: fh_compile.Compilation,
) -> fh_compile.CapturedCallable[object]:
    captured = compilation.workspace[fh_compile.CapturedCallable]
    assert isinstance(captured, fh_compile.CapturedCallable)
    return captured


def test_capture_accepts_a_pure_public_torch_graph() -> None:
    def public_map(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 1.5

    result = fh_compile.capture(public_map, inputs={"x": fh_compile.message()})

    names = _operation_names(result.program)
    assert names.count("torch.call") == 2
    assert not any(name.startswith("fhelium_semantic.") for name in names)
    captured = _captured_callable(result)
    assert captured.runtime_signature == captured.signature
    torch.testing.assert_close(
        captured.reference(torch.tensor([0.25, -0.5])),
        public_map(torch.tensor([0.25, -0.5])),
    )
    text = result.program.to_text()
    assert ir.parse(text).to_text() == text


def test_capture_preserves_unknown_encrypted_calls_for_later_handlers() -> None:
    def mixed_map(secret: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        transformed = torch.sin(secret)
        return transformed + public

    result = fh_compile.capture(
        mixed_map,
        inputs={
            "secret": fh_compile.encrypted(scale=None),
            "public": fh_compile.message(),
        },
    )

    names = _operation_names(result.program)
    assert "torch.call" in names
    assert "fhelium_semantic.add" in names
    torch_call = next(
        operation
        for operation in result.program.walk()
        if ir.operation_name(operation) == "torch.call"
    )
    target = torch_call.attributes["fhelium.call.target"]
    assert isinstance(target, StringAttr)
    assert isinstance(torch_call, ir.dialects.torch.CallOp)
    assert target.data == "torch.sin"
    assert "!fhelium_semantic.secret" in result.program.to_text()


_CAPTURE_WEIGHT = torch.tensor([1.25, 2.5], dtype=torch.float64)


def test_static_specialization_is_frontend_state_and_outputs_may_be_tuples() -> (
    None
):
    def select(
        x: torch.Tensor,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if enabled:
            return x, torch.neg(x)
        return x, x

    result = fh_compile.capture(
        select,
        inputs={"x": fh_compile.message(), "enabled": fh_compile.static(True)},
    )

    captured = _captured_callable(result)
    assert tuple(captured.runtime_signature.parameters) == ("x",)
    assert len(result.program.single_block().args) == 1
    assert len(result.program.function().function_type.outputs) == 2
    output_structure = result.program.module.attributes[
        "fhelium.output_structure"
    ]
    assert isinstance(output_structure, StringAttr)
    assert json.loads(output_structure.data)["kind"] == "tuple"
    expected = select(torch.tensor([2.0]), True)
    actual = captured.reference(torch.tensor([2.0]))
    assert isinstance(actual, tuple)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
