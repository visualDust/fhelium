#!/usr/bin/env python3

"""Parse, inspect, transform, and serialize a textual mixed-level Program.

Textual IR is the same Program representation accepted by Compile. This
example performs a stable parse/print round trip, inserts one caller-defined
analysis pass into the standard compile pipeline, and stops before execution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from common import print_table

from fhelium import compile as fh_compile
from fhelium import ir

_PROGRAM_TEXT = r'''
builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.2"
} {
  func.func @main(
    %x: !fhelium_semantic.secret<{role = "encrypted"}>,
    %gain: !fhelium_semantic.public<{role = "message"}>
  ) -> !fhelium_semantic.secret<{role = "encrypted"}> {
    %rotated = "fhelium_semantic.roll"(%x) {shift = 2 : i64, dimension = -1 : i64}
      : (!fhelium_semantic.secret<{role = "encrypted"}>)
     -> !fhelium_semantic.secret<{role = "encrypted"}>
    %mixed = "fhelium_semantic.add"(%x, %rotated)
      : (!fhelium_semantic.secret<{role = "encrypted"}>,
         !fhelium_semantic.secret<{role = "encrypted"}>)
     -> !fhelium_semantic.secret<{role = "encrypted"}>
    %result = "fhelium_semantic.multiply"(%mixed, %gain)
      : (!fhelium_semantic.secret<{role = "encrypted"}>,
         !fhelium_semantic.public<{role = "message"}>)
     -> !fhelium_semantic.secret<{role = "encrypted"}>
    func.return %result : !fhelium_semantic.secret<{role = "encrypted"}>
  }
}
'''


@dataclass(frozen=True)
class RecordDialectInventoryPass:
    """Publish operation counts by dialect without rewriting the Program."""

    name: str = "record-dialect-inventory"

    def run(
        self,
        program: ir.Program,
        shared_data: dict[object, object],
    ) -> fh_compile.PassResult:
        counts = Counter(
            operation.name.split(".", maxsplit=1)[0]
            for operation in program.walk()
        )
        shared_data["textual-ir/dialect-counts"] = dict(sorted(counts.items()))
        return fh_compile.PassResult.unchanged(
            program,
            matched=sum(counts.values()),
            diagnostics=(f"observed {len(counts)} dialect namespaces",),
        )


def main() -> None:
    imported = ir.parse(
        _PROGRAM_TEXT, source_name="inline-compile-example.mlir"
    )
    serialized = imported.to_text()
    round_tripped = ir.parse(
        serialized,
        source_name="round-tripped-compile-example.mlir",
    )
    if round_tripped.to_text() != serialized:
        raise RuntimeError("textual Program round trip was not stable")

    workspace = fh_compile.CompileWorkspace()
    pipeline = fh_compile.Pipeline(
        (
            RecordDialectInventoryPass(),
            fh_compile.EliminateDeadValuesPass(),
            fh_compile.LowerSemanticToLogicalPass(),
            fh_compile.InsertPlaintextPreparationPass(),
            fh_compile.InsertMultiplyNttTransitionsPass(),
            fh_compile.LowerLogicalToCkksPass(),
            fh_compile.InsertRelinearizationPass(),
            fh_compile.InsertRescalePass(),
        )
    )
    compiled = fh_compile.compile(
        round_tripped,
        pipeline=pipeline,
        workspace=workspace,
    )
    requirements = ir.analyze_evaluation_key_requirements(compiled.program)

    print_table(
        ["textual Program", "value"],
        [
            ["stable round trip", True],
            ["dialect counts", workspace["textual-ir/dialect-counts"]],
            ["rotation steps", sorted(requirements.rotation_steps)],
            ["relinearization key", requirements.requires_relinearization],
        ],
    )
    print()
    print_table(
        ["pass", "matched", "transformed", "diagnostics"],
        [
            [
                report.name,
                report.stats.matched,
                report.stats.transformed,
                "; ".join(report.diagnostics) or "none",
            ]
            for report in compiled.reports
        ],
    )
    print()
    print("--- stable imported Program ---")
    print(serialized)
    print()
    print("--- compiled Program ---")
    print(ir.format_program(compiled.program))


if __name__ == "__main__":
    main()
