#!/usr/bin/env python3

"""Compose built-in Compile passes, link the Program, and execute it.

The first part captures a Python function, runs a caller-selected pass sequence, links the backend-ready Program, and executes it. The second part parses a textual mixed-level Program, prints it back, and runs the front of the same pipeline with an added analysis pass that records operation counts by dialect.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import cast

import torch
from common import print_table

from fhelium import Preset
from fhelium import compile as fh_compile
from fhelium import ir
from fhelium.backend import OperationBackend, OperationImplementationRegistry
from fhelium.backend.ckks import CkksDeviceResources
from fhelium.config import CkksConfig
from fhelium.eager import Engine
from fhelium.values import Ciphertext

_ROTATION = 3
_LOGICAL_SLOTS = 16


def rotated_quadratic(
    x: torch.Tensor,
    rotation: int,
) -> torch.Tensor:
    """Square the sum of one vector and its cyclic rotation."""

    mixed = x + torch.roll(x, shifts=rotation, dims=-1)
    return mixed * mixed


def _dialect_counts(program: ir.Program) -> Counter[str]:
    """Count operation namespaces in one Program."""

    return Counter(
        operation.name.split(".", maxsplit=1)[0] for operation in program.walk()
    )


def captured_pipeline() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7)
    device = "cpu"
    backend = OperationBackend()
    workspace = fh_compile.CompileWorkspace(
        {
            CkksConfig: config,
            OperationImplementationRegistry: backend.registry,
        }
    )

    captured = fh_compile.capture(
        rotated_quadratic,
        inputs={
            "x": fh_compile.encrypted(
                slots=_LOGICAL_SLOTS,
                polynomial_domain="coefficient",
                residue_representation="standard",
            ),
            "rotation": fh_compile.static(_ROTATION),
        },
        workspace=workspace,
    )

    # Compose the built-in passes required by this workload.
    pipeline = fh_compile.Pipeline(
        (
            fh_compile.EliminateDeadValuesPass(),
            fh_compile.LowerSemanticToLogicalPass(),
            fh_compile.InsertMultiplyNttTransitionsPass(),
            fh_compile.LowerLogicalToCkksPass(),
            fh_compile.ResolveRotationKeyOperandsPass(),
            fh_compile.InsertRelinearizationPass(),
            fh_compile.InsertRescalePass(),
            fh_compile.AssignCkksDepthsPass(entry_depth=0),
            fh_compile.AssignCkksScalesPass(entry_scale=config.scale_at(0)),
            fh_compile.AssignNttImplementationPass(
                "native-ntt", ntt_backend="radix2_indexed"
            ),
            fh_compile.LowerCkksToRnsNttPass(),
            fh_compile.SelectNttImplementationsPass(),
        )
    )
    compiled = pipeline.run(captured)

    requirements = ir.analyze_evaluation_key_requirements(compiled.program)
    engine = Engine(config, rng_seed=17)
    secret_key = engine.create_secret_key(device=device)
    public_key = engine.create_public_key(secret_key, device=device)
    rotation_keys = tuple(
        engine.create_rotation_key(step, secret_key, device=device)
        for step in sorted(requirements.rotation_steps)
    )
    relinearization_key = (
        engine.create_relinearization_key(secret_key, device=device)
        if requirements.requires_relinearization
        else None
    )
    resources = CkksDeviceResources(
        config=config,
        device=device,
        rng_seed=29,
    )
    fh_compile.prepare_material_bindings(
        compiled,
        resources=resources,
        keys=(
            *rotation_keys,
            *(
                (relinearization_key,)
                if relinearization_key is not None
                else ()
            ),
        ),
    )
    executable = backend.link(compiled)

    logical_x = torch.linspace(
        -0.04,
        0.04,
        _LOGICAL_SLOTS,
        dtype=torch.float64,
    )
    clear_x = logical_x.repeat(config.num_slots // _LOGICAL_SLOTS)
    captured_callable = cast(
        fh_compile.CapturedCallable[torch.Tensor],
        captured.workspace[fh_compile.CapturedCallable],
    )
    expected = captured_callable.reference(clear_x)

    encrypted_x = engine.encrypt_message(
        clear_x,
        public_key,
        depth=0,
        scale=config.scale_at(0),
        device=device,
    )
    result = cast(
        Ciphertext,
        executable.run(encrypted_x),
    )
    decoded = engine.decrypt_message(result, secret_key)
    captured_inventory = ir.inventory_program(captured.program)
    compiled_inventory = ir.inventory_program(compiled.program)
    print_table(
        ["stage", "operations", "dialects"],
        [
            [
                "captured",
                sum(captured_inventory.operation_counts.values()),
                dict(sorted(_dialect_counts(captured.program).items())),
            ],
            [
                "backend-ready",
                sum(compiled_inventory.operation_counts.values()),
                dict(sorted(_dialect_counts(compiled.program).items())),
            ],
        ],
    )
    print()
    print_table(
        ["execution", "value"],
        [
            ["rotation steps", sorted(requirements.rotation_steps)],
            ["relinearization key", requirements.requires_relinearization],
            ["result depth", result.depth],
            ["result scale", f"{result.scale:.6e}"],
            [
                "maximum error",
                f"{float((decoded - expected).abs().max()):.6e}",
            ],
        ],
    )
    print()
    print_table(
        [
            "pass",
            "matched",
            "transformed",
            "inserted",
            "removed",
            "skipped",
            "host seconds",
        ],
        [
            [
                report.name,
                report.stats.matched,
                report.stats.transformed,
                report.stats.inserted,
                report.stats.removed,
                report.stats.skipped,
                f"{report.seconds:.6f}",
            ]
            for report in compiled.reports
        ],
    )


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

    def run(self, compilation: fh_compile.Compilation) -> fh_compile.PassResult:
        program = compilation.program
        shared_data = compilation.workspace
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


def textual_pipeline() -> None:
    imported = ir.parse(
        _PROGRAM_TEXT, source_name="inline-compile-example.mlir"
    )
    serialized = imported.to_text()
    round_tripped = ir.parse(
        serialized,
        source_name="round-tripped-compile-example.mlir",
    )

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
    compiled = pipeline.run(fh_compile.Compilation(round_tripped, workspace))
    requirements = ir.analyze_evaluation_key_requirements(compiled.program)

    print_table(
        ["textual Program", "value"],
        [
            ["stable round trip", round_tripped.to_text() == serialized],
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
    print(serialized)


def main() -> None:
    captured_pipeline()
    print()
    textual_pipeline()


if __name__ == "__main__":
    main()
