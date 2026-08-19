"""Direct FX capture tests for JIT's canonical unified Program."""

from __future__ import annotations

import json

import torch
from xdsl.dialects.builtin import StringAttr

from fhelium.experimental.jit._capture import capture
from fhelium.experimental.jit._dialect import operation_name
from fhelium.experimental.jit._program import Program
from fhelium.experimental.jit._specs import encrypted, message, static
from fhelium.experimental.jit._workspace import Workspace


def _operation_names(program: Program) -> tuple[str, ...]:
    return tuple(operation_name(operation) for operation in program.walk())


def test_capture_accepts_a_pure_public_torch_graph() -> None:
    def public_map(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + 1.5

    result = capture(public_map, inputs={"x": message()})

    names = _operation_names(result.program)
    assert names.count("torch.call") == 2
    assert not any(name.startswith("fhelium.semantic.") for name in names)
    assert result.runtime_signature == result.signature
    torch.testing.assert_close(
        result.reference(torch.tensor([0.25, -0.5])),
        public_map(torch.tensor([0.25, -0.5])),
    )

    text = result.program.to_text()
    assert Program.parse(text).to_text() == text


def test_capture_preserves_unknown_encrypted_calls_for_later_handlers() -> None:
    def mixed_map(secret: torch.Tensor, public: torch.Tensor) -> torch.Tensor:
        transformed = torch.sin(secret)
        return transformed + public

    result = capture(
        mixed_map,
        inputs={"secret": encrypted(scale=None), "public": message()},
    )

    names = _operation_names(result.program)
    assert "torch.call" in names
    assert "fhelium.semantic.add" in names
    torch_call = next(
        operation
        for operation in result.program.walk()
        if operation_name(operation) == "torch.call"
    )
    target = torch_call.attributes["fhelium.call.target"]
    assert isinstance(target, StringAttr)
    assert target.data == "torch.sin"
    assert "!fhelium.encrypted" in result.program.to_text()


_CAPTURE_WEIGHT = torch.tensor([1.25, 2.5], dtype=torch.float64)


def test_capture_externalizes_tensor_constants_as_symbolic_materials() -> None:
    def weighted(secret: torch.Tensor) -> torch.Tensor:
        return secret * _CAPTURE_WEIGHT

    workspace = Workspace({"caller-policy": object()})
    result = capture(
        weighted,
        inputs={"secret": encrypted()},
        workspace=workspace,
    )

    assert result.workspace is workspace
    assert "caller-policy" in workspace
    materials = workspace["materials"]
    assert len(materials) == 1
    symbol, captured = next(iter(materials.items()))
    assert symbol.startswith("capture/tensor/")
    assert isinstance(captured, torch.Tensor)
    assert captured is not _CAPTURE_WEIGHT
    torch.testing.assert_close(captured, _CAPTURE_WEIGHT)

    text = result.program.to_text()
    assert "fhelium.material.ref" in text
    assert symbol in text
    assert "1.25" not in text
    assert "2.5" not in text
    assert "fhelium.semantic.multiply" in text


def test_static_specialization_is_graph_external_and_outputs_may_be_tuples() -> (
    None
):
    def select(
        x: torch.Tensor,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if enabled:
            return x, torch.neg(x)
        return x, x

    result = capture(
        select,
        inputs={"x": message(), "enabled": static(True)},
    )

    assert tuple(result.runtime_signature.parameters) == ("x",)
    assert len(result.program.entry_block().args) == 1
    assert len(result.program.entry_function().function_type.outputs) == 2
    output_structure = result.program.module.attributes[
        "fhelium.output_structure"
    ]
    assert isinstance(output_structure, StringAttr)
    assert json.loads(output_structure.data)["kind"] == "tuple"
    expected = select(torch.tensor([2.0]), True)
    actual = result.reference(torch.tensor([2.0]))
    assert isinstance(actual, tuple)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
