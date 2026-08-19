#!/usr/bin/env python3

"""Import, inspect, bind, and execute a textual mixed-dialect JIT Program.

The example is IR-first and CPU-only. It imports FHElium, Torch, and
application-extension operations from text, runs a composed pass
pipeline with a custom analysis pass, retains caller-owned Workspace state,
binds a custom operation handler, checks readiness, executes, and verifies a
stable textual round trip.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import torch
from common import print_table
from xdsl.ir import Operation

from fhelium.experimental import jit

_PROGRAM_TEXT = r'''
builtin.module attributes {
  fhelium.schema_version = "1",
  fhelium.dialect_version = "0.1"
} {
  func.func @main(%x: !fhelium.message<{}>) -> !fhelium.message<{}> {
    %bias = "fhelium.constant"() {fhelium.literal = "0.75"}
      : () -> !fhelium.message<{}>
    %negated = "torch.call"(%x) {
      fhelium.call.kind = "function",
      fhelium.call.target = "torch.neg",
      fhelium.call.arguments = "{\22args\22:{\22items\22:[{\22kind\22:\22ssa\22,\22operand\22:0}],\22kind\22:\22tuple\22},\22kwargs\22:{\22entries\22:[],\22kind\22:\22mapping\22}}"
    } : (!fhelium.message<{}>) -> !fhelium.message<{}>
    %result = "application.scale_and_shift"(%negated, %bias)
      : (!fhelium.message<{}>, !fhelium.message<{}>)
     -> !fhelium.message<{}>
    func.return %result : !fhelium.message<{}>
  }
}
'''


@dataclass(frozen=True)
class RecordRequirementsPass:
    """Publish the imported operation surface into the retained Workspace."""

    name: str = "record-imported-requirements"

    def run(
        self,
        program: jit.Program,
        workspace: MutableMapping[Any, Any],
    ) -> jit.PassResult:
        requirements = program.requirements()
        operation_surface = tuple(sorted(requirements.operations))
        workspace["analysis/imported-operation-surface"] = operation_surface
        return jit.PassResult.unchanged(
            program,
            matched=len(operation_surface),
            diagnostics=(
                f"recorded {len(operation_surface)} executable operation names",
            ),
        )


def scale_and_shift(
    operation: Operation,
    operands: tuple[object, ...],
    workspace: MutableMapping[Any, Any],
) -> object:
    """Execute ``application.scale_and_shift`` from graph-external policy."""

    del operation
    if len(operands) != 2:
        raise ValueError("application.scale_and_shift requires two operands")
    source, bias = operands
    if not isinstance(source, torch.Tensor) or not isinstance(
        bias, (int, float)
    ):
        raise TypeError("scale_and_shift expects Tensor and real operands")
    gain = workspace.get("application/gain")
    if not isinstance(gain, (int, float)):
        raise TypeError("workspace['application/gain'] must be real")
    return source * gain + bias


def main() -> None:
    imported = jit.parse(_PROGRAM_TEXT, source_name="inline-application.mlir")
    canonical_text = imported.to_text()
    round_tripped = jit.parse(
        canonical_text,
        source_name="round-tripped-application.mlir",
    )
    if round_tripped.to_text() != canonical_text:
        raise RuntimeError("textual JIT Program round trip was not stable")

    requirements = round_tripped.requirements()
    unbound = round_tripped.readiness()
    if unbound.runnable:
        raise RuntimeError(
            "the extension operation was ready without a handler"
        )

    workspace = jit.Workspace(
        {
            "application/gain": 0.5,
            "handlers": {
                "application.scale_and_shift": scale_and_shift,
            },
        }
    )
    pipeline = jit.PassPipeline(
        (
            RecordRequirementsPass(),
            jit.ValidateExecutableGraphPass(),
        )
    )
    transformed = pipeline.run(round_tripped, workspace)
    if transformed.workspace is not workspace:
        raise RuntimeError("the custom pipeline did not retain its Workspace")

    ready = transformed.program.readiness(transformed.workspace)
    if not ready.runnable:
        detail = "; ".join(item.message for item in ready.diagnostics)
        raise RuntimeError(f"the bound textual Program is not ready: {detail}")

    value = torch.tensor([1.0, -2.0, 0.25], dtype=torch.float64)
    actual = jit.run(
        transformed.program,
        value,
        workspace=transformed.workspace,
    )
    expected = -value * 0.5 + 0.75
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    print_table(
        ["requirement", "value"],
        [
            ["operations", sorted(requirements.operations)],
            ["unknown operations", sorted(requirements.unknown_operations)],
            ["Torch targets", sorted(requirements.torch_targets)],
            ["requires CKKS engine", requirements.requires_engine],
        ],
    )
    print()
    print_table(
        ["readiness", "runnable", "diagnostics"],
        [
            [
                "without extension handler",
                unbound.runnable,
                ", ".join(item.code for item in unbound.diagnostics),
            ],
            [
                "with retained Workspace",
                ready.runnable,
                ", ".join(item.code for item in ready.diagnostics) or "none",
            ],
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
            for report in transformed.reports
        ],
    )
    print()
    print_table(
        ["input", "output", "expected"],
        [[value.tolist(), actual.tolist(), expected.tolist()]],
    )
    print()
    print("--- stable round-tripped textual Program ---")
    print(canonical_text)


if __name__ == "__main__":
    main()
