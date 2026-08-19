"""Unified mixed-dialect Program and retained pass-workspace tests."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass

import pytest
from xdsl.dialects.builtin import StringAttr

from fhelium.experimental.jit._dialect import operation_name
from fhelium.experimental.jit._program import Program
from fhelium.experimental.jit._workspace import Workspace
from fhelium.experimental.jit.passes._base import (
    PassPipeline,
    PassResult,
    PassStats,
)

_INCOMPLETE_MIXED_PROGRAM = r'''
builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.1"
} {
  func.func @main(%secret: !fhelium.encrypted<{level = 0 : i64}>) -> !fhelium.encrypted<{}> {
    %relin = "fhelium.material.ref"() {symbol = "keys/relinearization", kind = "relinearization_key"} : () -> !fhelium.material<{}>
    %product = "fhelium.ckks.multiply"(%secret, %secret) : (!fhelium.encrypted<{level = 0 : i64}>, !fhelium.encrypted<{level = 0 : i64}>) -> !fhelium.encrypted<{scale = 9.99 : f64}>
    %view = "torch.aten.view.default"(%product) {shape = [4 : i64, 8 : i64]} : (!fhelium.encrypted<{scale = 9.99 : f64}>) -> !third_party.tensor<"opaque-layout">
    %result = "vendor.ckks.bootstrap"(%view, %relin) : (!third_party.tensor<"opaque-layout">, !fhelium.material<{}>) -> !fhelium.encrypted<{}>
    func.return %result : !fhelium.encrypted<{}>
  }
}
'''


def test_program_round_trips_incomplete_mixed_dialect_graph() -> None:
    program = Program.parse(_INCOMPLETE_MIXED_PROGRAM, source_name="mixed.mlir")

    names = tuple(operation_name(operation) for operation in program.walk())
    assert names == (
        "func.func",
        "fhelium.material.ref",
        "fhelium.ckks.multiply",
        "torch.aten.view.default",
        "vendor.ckks.bootstrap",
        "func.return",
    )

    # Missing rescale, unresolved material, unknown Torch/extension operations,
    # and partial CKKS state do not block structural interchange.
    exported = program.to_text()
    assert "fhelium.ckks.rescale" not in exported
    assert '"keys/relinearization"' in exported
    assert '"torch.aten.view.default"' in exported
    assert '"vendor.ckks.bootstrap"' in exported
    assert Program.parse(exported).to_text() == exported


def test_program_rejects_malformed_registered_structure() -> None:
    with pytest.raises(Exception, match="Expected 1 result"):
        Program.parse('builtin.module { "fhelium.material.ref"() : () -> () }')


def test_program_load_save_and_entry_helpers(tmp_path) -> None:
    program = Program.parse(_INCOMPLETE_MIXED_PROGRAM)
    path = tmp_path / "program.mlir"

    program.save(path)
    loaded = Program.load(path)

    assert loaded.to_text() == program.to_text()
    assert loaded.entry_function().sym_name.data == "main"
    assert len(loaded.entry_block().args) == 1
    assert loaded.clone().to_text() == loaded.to_text()


@dataclass(frozen=True)
class RecordCountPass:
    name: str = "record-count"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[object, object],
    ) -> PassResult:
        workspace["operation_count"] = sum(1 for _ in program.walk())
        return PassResult.unchanged(program)


@dataclass(frozen=True)
class ObserveAndMarkPass:
    name: str = "observe-and-mark"

    def run(
        self,
        program: Program,
        workspace: MutableMapping[object, object],
    ) -> PassResult:
        workspace["observed_count"] = workspace["operation_count"]
        program.module.attributes["test.marker"] = StringAttr("changed")
        return PassResult(
            program,
            PassStats(matched=1, transformed=1),
        )


def test_pipeline_shares_and_retains_unrestricted_workspace() -> None:
    source = Program.parse(_INCOMPLETE_MIXED_PROGRAM)
    marker = object()
    workspace = Workspace({marker: lambda: "caller-owned"})
    pipeline = PassPipeline((RecordCountPass(), ObserveAndMarkPass()))

    result = pipeline.run(source, workspace)

    assert result.workspace is workspace
    assert result.workspace[marker]() == "caller-owned"
    assert result.workspace["observed_count"] == 6
    assert result.program is not source
    assert "test.marker" in result.program.module.attributes
    assert "test.marker" not in source.module.attributes
    assert tuple(report.name for report in result.reports) == (
        "record-count",
        "observe-and-mark",
    )
    assert tuple(report.stats.transformed for report in result.reports) == (
        0,
        1,
    )

    workspace["after"] = True
    assert result.workspace["after"] is True


def test_pipeline_accepts_and_preserves_plain_dictionary() -> None:
    plain_workspace: dict[object, object] = {"caller": 1}
    result = PassPipeline((RecordCountPass(),)).run(
        Program.parse(_INCOMPLETE_MIXED_PROGRAM), plain_workspace
    )

    assert result.workspace is plain_workspace
    assert plain_workspace["operation_count"] == 6


def test_default_pipeline_is_not_hard_coded_to_main() -> None:
    program = Program.parse(
        r'''
        builtin.module attributes {
          fhelium.schema_version = "1",
          fhelium.dialect_version = "0.1"
        } {
          func.func @helper(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
            %result = "torch.call"(%x) {
              fhelium.call.kind = "function",
              fhelium.call.target = "torch.neg",
              fhelium.call.arguments = "{\22args\22:{\22items\22:[{\22kind\22:\22ssa\22,\22operand\22:0}],\22kind\22:\22tuple\22},\22kwargs\22:{\22entries\22:[],\22kind\22:\22mapping\22}}"
            } : (!fhelium.message<{}>) -> !fhelium.message<{}>
            func.return %result : !fhelium.message<{}>
          }
        }
        '''
    )

    from fhelium.experimental.jit.passes import default_pipeline

    result = default_pipeline().run(program)
    assert "@helper" in result.program.to_text()
