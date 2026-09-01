"""Neutral mixed-dialect Program behavior tests."""

from __future__ import annotations

import pytest

from fhelium import ir

_INCOMPLETE_MIXED_PROGRAM = r'''
builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.2"
} {
  func.func @main(%secret: !fhelium_ckks.ciphertext<{level = 0 : i64}>) -> !fhelium_ckks.ciphertext<{}> {
    %relin = "fhelium.material.ref"() {symbol = "keys/relinearization", kind = "relinearization_key"} : () -> !fhelium.material<{}>
    %product = "fhelium_ckks.multiply"(%secret, %secret) : (!fhelium_ckks.ciphertext<{level = 0 : i64}>, !fhelium_ckks.ciphertext<{level = 0 : i64}>) -> !fhelium_ckks.ciphertext<{scale = 9.99 : f64}>
    %view = "torch.aten.view.default"(%product) {shape = [4 : i64, 8 : i64]} : (!fhelium_ckks.ciphertext<{scale = 9.99 : f64}>) -> !third_party.tensor<"opaque-layout">
    %result = "vendor.ckks.bootstrap"(%view, %relin) : (!third_party.tensor<"opaque-layout">, !fhelium.material<{}>) -> !fhelium_ckks.ciphertext<{}>
    func.return %result : !fhelium_ckks.ciphertext<{}>
  }
}
'''


def test_program_round_trips_incomplete_mixed_dialect_graph() -> None:
    program = ir.parse(_INCOMPLETE_MIXED_PROGRAM, source_name="mixed.mlir")

    names = {ir.operation_name(operation) for operation in program.walk()}
    assert "fhelium_ckks.multiply" in names
    assert "torch.aten.view.default" in names
    assert "vendor.ckks.bootstrap" in names

    exported = program.to_text()
    assert ir.parse(exported).to_text() == exported


def test_program_rejects_malformed_registered_structure() -> None:
    with pytest.raises(Exception):
        ir.parse('builtin.module { "fhelium.material.ref"() : () -> () }')


def test_program_load_save_round_trip(tmp_path) -> None:
    program = ir.parse(_INCOMPLETE_MIXED_PROGRAM)
    path = tmp_path / "program.mlir"

    program.save(path)
    loaded = ir.load(str(path))

    assert loaded.to_text() == program.to_text()
