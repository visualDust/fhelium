"""Public single-package API tests for JIT programs."""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from fhelium.experimental import jit


def test_legacy_sugar_package_has_no_compatibility_alias() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fhelium.experimental.sugar")


@dataclass(frozen=True)
class RecordDepthPass:
    name: str = "record-depth"

    def run(
        self,
        program: jit.Program,
        workspace: MutableMapping[Any, Any],
    ) -> jit.PassResult:
        workspace["analysis/multiplication-depth"] = 1
        return jit.PassResult.unchanged(program)


def test_trace_transform_and_run_use_one_program_and_workspace() -> None:
    def public_map(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + bias

    capture = jit.trace(
        public_map,
        inputs={"x": jit.message(), "bias": jit.message()},
    )
    source = capture.program
    transformed = source.transform(
        RecordDepthPass(), workspace=capture.workspace
    )

    assert isinstance(source, jit.Program)
    assert isinstance(transformed.program, jit.Program)
    assert transformed.program is not source
    assert transformed.workspace is capture.workspace
    assert transformed.workspace["analysis/multiplication-depth"] == 1

    x = torch.tensor([0.1, -0.2])
    bias = torch.tensor([0.3, 0.4])
    torch.testing.assert_close(
        transformed.program.run(x, bias, workspace=transformed.workspace),
        public_map(x, bias),
    )


def test_load_incomplete_ckks_then_apply_default_pipeline(
    tmp_path: Path,
) -> None:
    text = r'''
    builtin.module attributes {
      fhelium.schema_version = "1",
      fhelium.dialect_version = "0.1"
    } {
      func.func @main(%x: !fhelium.encrypted<{}>) -> !fhelium.encrypted<{}> {
        %product = "fhelium.semantic.multiply"(%x, %x)
          : (!fhelium.encrypted<{}>, !fhelium.encrypted<{}>)
         -> !fhelium.encrypted<{}>
        func.return %product : !fhelium.encrypted<{}>
      }
    }
    '''
    path = tmp_path / "program.mlir"
    path.write_text(text, encoding="utf-8")

    program = jit.load(str(path))
    assert "fhelium.semantic.multiply" in program.to_text()
    assert not program.readiness().runnable

    workspace = jit.Workspace({"programmer/policy": "explicit"})
    lowered = jit.default_pipeline().run(program, workspace)
    lowered_text = lowered.program.to_text()

    assert lowered.workspace is workspace
    assert lowered.workspace["programmer/policy"] == "explicit"
    assert "fhelium.semantic.multiply" not in lowered_text
    assert "fhelium.ckks.multiply" in lowered_text
    assert "fhelium.ckks.relinearize" in lowered_text
    assert "fhelium.ckks.rescale" in lowered_text

    requirements = lowered.program.requirements()
    assert requirements.requires_engine
    assert requirements.requires_relinearization
    report = lowered.program.readiness(lowered.workspace)
    assert not report.runnable
    assert {item.code for item in report.diagnostics} == {
        "missing-engine",
        "missing-evaluation-keys",
    }


def test_top_level_parse_and_run_accept_custom_extension_handler() -> None:
    program = jit.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @main(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
            %result = "application.double"(%x)
              : (!fhelium.message<{}>) -> !fhelium.message<{}>
            func.return %result : !fhelium.message<{}>
          }
        }
        '''
    )
    workspace = jit.Workspace(
        {
            "handlers": {
                "application.double": (
                    lambda operation, operands, retained: operands[0] * 2
                )
            }
        }
    )

    assert program.readiness(workspace).runnable
    assert jit.run(program, 7, workspace=workspace) == 14
