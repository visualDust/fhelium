#!/usr/bin/env python3

"""Compose and run a pass-controlled encrypted quadratic JIT Program.

The default lowering pipeline is extended with a custom, non-rewriting CKKS
audit pass and the executable-graph validator. The example then plans and binds
evaluation keys, checks readiness before and after binding, and validates
the actual CKKS result against one fixed two-rescale validation threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import torch
from common import add_engine_args, error_stats, make_engine, print_table

import fhelium as fh
from fhelium.experimental import jit

_ROTATION = 3
_GAIN = 0.625
_INPUT_ABS_BOUND = 0.03
_BIAS_ABS_BOUND = 0.004
_CLEAR_OUTPUT_BOUND = (2.0 * _INPUT_ABS_BOUND * _GAIN) ** 2 + _BIAS_ABS_BOUND

# The default scale-40 circuit has two explicit rescale stages and a cleartext
# magnitude bounded above by _CLEAR_OUTPUT_BOUND. This non-configurable limit
# reserves less than 0.4% of that bound for aggregate CKKS approximation,
# encryption, key-switch, NTT, and rescale error; it is not adjusted at runtime.
_VALIDATION_ATOL = 2e-5


def rotated_quadratic(
    x: torch.Tensor,
    gain: float,
    bias: torch.Tensor,
    rotation: int,
) -> torch.Tensor:
    """Square a gain-scaled sum of the input and one cyclic rotation."""

    mixed = (x + torch.roll(x, shifts=rotation, dims=-1)) * gain
    return mixed * mixed + bias


@dataclass(frozen=True)
class AuditExplicitCkksPass:
    """Audit unresolved arithmetic and record the lowered CKKS surface.

    The pass leaves the Program and rescale/relinearization placement
    unchanged.
    """

    name: str = "audit-explicit-ckks"

    def run(
        self,
        program: jit.Program,
        workspace: MutableMapping[Any, Any],
    ) -> jit.PassResult:
        requirements = program.requirements()
        unresolved = sorted(
            operation
            for operation in requirements.operations
            if operation.startswith(("fhelium.semantic.", "fhelium.logical."))
        )
        required = {
            "fhelium.ckks.rotate",
            "fhelium.ckks.multiply",
            "fhelium.ckks.relinearize",
            "fhelium.ckks.rescale",
        }
        missing = sorted(required - requirements.operations)
        if unresolved or missing:
            raise jit.JitPassError(
                "explicit CKKS audit failed: "
                f"unresolved={unresolved}, missing={missing}"
            )

        operation_surface = tuple(sorted(requirements.operations))
        workspace["analysis/explicit-ckks-operation-surface"] = (
            operation_surface
        )
        return jit.PassResult.unchanged(
            program,
            matched=len(operation_surface),
            diagnostics=(
                "audited explicit CKKS operations without rewriting them",
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-levels7-int64")
    args = parser.parse_args()

    engine = make_engine(args)

    workspace = jit.Workspace(
        {
            "programmer/pipeline-policy": (
                "default lowering plus explicit CKKS audit and validation"
            )
        }
    )
    captured = jit.trace(
        rotated_quadratic,
        inputs={
            "x": jit.encrypted(),
            "gain": jit.message(),
            "bias": jit.message(),
            "rotation": jit.static(_ROTATION),
        },
        workspace=workspace,
    )

    # Insert a caller-owned audit after the named default step, then
    # append the structural execution gate. No pass here pretends to perform a
    # backend-specific late-rescale or late-relinearization optimization.
    pipeline = (
        jit.default_pipeline()
        .after("late-relinearization", AuditExplicitCkksPass())
        .then(jit.ValidateExecutableGraphPass())
    )
    lowered = pipeline.run(captured.program, captured.workspace)
    if lowered.workspace is not workspace:
        raise RuntimeError("the custom pipeline did not retain its Workspace")
    program = lowered.program

    # Key planning is a pure scan of the explicit lowered operations. Before
    # binding, readiness exposes the absent runtime capabilities.
    key_plan = jit.analyze_evaluation_key_requirements(program)
    before_bindings = program.readiness(workspace)
    if before_bindings.runnable:
        raise RuntimeError(
            "the CKKS Program was ready without runtime bindings"
        )

    evaluation_keys = fh.EvaluationKeySet(
        rotations=fh.RotationKeySet(
            {
                step: engine.rotation_key(step)
                for step in key_plan.rotation_steps
            }
        ),
        relinearization=(
            engine.relinearization_key
            if key_plan.requires_relinearization
            else None
        ),
    )
    workspace.update(
        {
            "engine": engine,
            "evaluation_keys": evaluation_keys,
        }
    )
    ready = program.readiness(workspace)
    if not ready.runnable:
        detail = "; ".join(item.message for item in ready.diagnostics)
        raise RuntimeError(f"the bound JIT Program is not ready: {detail}")

    index = torch.arange(engine.num_slots, dtype=torch.float64)
    clear_x = 0.02 * torch.sin(0.013 * index) + 0.01 * torch.cos(0.031 * index)
    bias = _BIAS_ABS_BOUND * torch.sin(0.007 * index + 0.2)
    reference = captured.reference(clear_x, _GAIN, bias)
    if float(torch.abs(reference).max()) > _CLEAR_OUTPUT_BOUND:
        raise RuntimeError("the analytical cleartext bound was violated")

    encrypted_x = engine.encrypt_message(clear_x, engine.public_key)
    encrypted_result = jit.run(
        program,
        encrypted_x,
        _GAIN,
        bias,
        workspace=workspace,
    )
    decoded = engine.decrypt_message(
        encrypted_result,
        engine.secret_key,
        is_real=True,
    )
    statistics = error_stats(decoded, reference)
    if statistics["max_abs"] > _VALIDATION_ATOL:
        raise RuntimeError(
            "custom-pipeline execution exceeded its fixed CKKS validation threshold: "
            f"max_abs_error={statistics['max_abs']:.3e}, "
            f"atol={_VALIDATION_ATOL:.3e}"
        )

    print_table(
        ["pipeline position", "pass"],
        [[index, name] for index, name in enumerate(pipeline.names)],
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
            "diagnostics",
        ],
        [
            [
                report.name,
                report.stats.matched,
                report.stats.transformed,
                report.stats.inserted,
                report.stats.removed,
                report.stats.skipped,
                "; ".join(report.diagnostics) or "none",
            ]
            for report in lowered.reports
        ],
    )
    print()
    print_table(
        ["planning/readiness", "value"],
        [
            ["rotation steps", sorted(key_plan.rotation_steps)],
            ["relinearization key", key_plan.requires_relinearization],
            [
                "before bindings",
                ", ".join(item.code for item in before_bindings.diagnostics),
            ],
            ["after bindings", "runnable" if ready.runnable else "blocked"],
        ],
    )
    print()
    print_table(
        [
            "result level",
            "result scale",
            "clear |max|",
            "validation atol",
            "max abs error",
            "rms error",
        ],
        [
            [
                encrypted_result.level,
                f"{encrypted_result.scale:.6e}",
                f"{float(torch.abs(reference).max()):.3e}",
                f"{_VALIDATION_ATOL:.3e}",
                f"{statistics['max_abs']:.3e}",
                f"{statistics['rms']:.3e}",
            ]
        ],
    )


if __name__ == "__main__":
    main()
