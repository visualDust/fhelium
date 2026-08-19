#!/usr/bin/env python3

"""Trace, lower, check, and run a dense encrypted matrix-vector program.

This example follows the JIT's trace-first model: tracing produces one mixed-
dialect Program, the default pipeline lowers its recognized encrypted
operations, readiness compares the lowered Program with retained runtime
bindings, and only ``Program.run(...)`` executes it. Public matrix preparation stays
as Torch operations while encrypted arithmetic becomes explicit CKKS operations.
"""

from __future__ import annotations

import argparse

import torch
from common import add_engine_args, make_engine, print_table

import fhelium as fh
from fhelium.experimental import jit

_MATRIX_SIZE = 8
_VALIDATION_ATOL = 3e-5


def square_matvec_quadratic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output_gain: torch.Tensor,
    matrix_size: int,
    repeats: int,
) -> torch.Tensor:
    """Evaluate a repeated packed affine map followed by a quadratic."""

    main_diagonal = torch.diagonal(weight).repeat(repeats)
    affine = x * main_diagonal

    for shift in range(1, matrix_size):
        wrapped_head = torch.diagonal(
            weight,
            offset=matrix_size - shift,
        )
        ordinary_tail = torch.diagonal(weight, offset=-shift)
        cyclic_diagonal = torch.cat((wrapped_head, ordinary_tail))
        packed_diagonal = cyclic_diagonal.repeat(repeats)
        rotated = torch.roll(x, shifts=shift, dims=-1)
        affine = affine + rotated * packed_diagonal

    affine = affine + bias.repeat(repeats)
    quadratic = (affine + 0.25) * (affine - 0.5)
    return quadratic * output_gain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser, default_preset="slots8192-scale40-levels7-int64")
    args = parser.parse_args()

    engine = make_engine(args)
    if engine.num_slots % _MATRIX_SIZE:
        parser.error("the engine slot count must be divisible by matrix size")
    repeats = engine.num_slots // _MATRIX_SIZE

    # 1. Trace into the canonical mixed-dialect Program. Tracing does not run a
    # lowering pipeline and does not establish execution readiness.
    captured = jit.trace(
        square_matvec_quadratic,
        inputs={
            "x": jit.encrypted(),
            "weight": jit.message(),
            "bias": jit.message(),
            "output_gain": jit.plaintext(),
            "matrix_size": jit.static(_MATRIX_SIZE),
            "repeats": jit.static(repeats),
        },
    )

    traced_readiness = captured.program.readiness(captured.workspace)
    if traced_readiness.runnable:
        raise RuntimeError("the traced semantic Program was unexpectedly ready")

    # 2. Select and run the ordinary lowering policy. A pipeline
    # transforms one clone of the canonical xDSL Program. The same retained
    # Workspace carries caller policy, graph-external materials, and runtime
    # capabilities without embedding live objects in the graph.
    lowered = jit.default_pipeline().run(
        captured.program,
        captured.workspace,
    )
    program = lowered.program
    workspace = lowered.workspace
    if workspace is not captured.workspace:
        raise RuntimeError("the pass pipeline did not retain its Workspace")

    # 3. Analyze requirements without materializing keys. Readiness is a
    # separate readiness check and is expected to fail before runtime bindings exist.
    key_requirements = jit.analyze_evaluation_key_requirements(program)
    before_bindings = program.readiness(workspace)
    if before_bindings.runnable:
        raise RuntimeError(
            "the lowered CKKS Program was ready without an engine"
        )

    evaluation_keys = fh.EvaluationKeySet(
        rotations=fh.RotationKeySet(
            {
                step: engine.rotation_key(step)
                for step in key_requirements.rotation_steps
            }
        ),
        relinearization=(
            engine.relinearization_key
            if key_requirements.requires_relinearization
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

    index = torch.arange(_MATRIX_SIZE, dtype=torch.float64)
    application_x = 0.025 * torch.sin(0.7 * index) + 0.01 * torch.cos(
        1.3 * index
    )
    rows = index[:, None]
    columns = index[None, :]
    weight = (
        0.06 * torch.sin((rows + 1.0) * (columns + 2.0))
        + 0.03 * torch.cos(rows - 2.0 * columns)
        + 0.28 * torch.eye(_MATRIX_SIZE, dtype=torch.float64)
    )
    bias = 0.012 * torch.cos(0.9 * index)
    packed_x = application_x.repeat(repeats)
    output_gain_value = 0.75
    output_gain_plaintext = engine.plaintext(
        output_gain_value,
        level=2,
        scale=engine.config.default_scale,
    )

    reference = captured.reference(
        packed_x,
        weight,
        bias,
        output_gain_value,
    )
    clear_affine = application_x @ weight.T + bias
    clear_block = (
        (clear_affine + 0.25) * (clear_affine - 0.5) * output_gain_value
    )
    torch.testing.assert_close(
        reference[:_MATRIX_SIZE],
        clear_block,
        rtol=0.0,
        atol=0.0,
    )

    encrypted_x = engine.encrypt_message(packed_x, engine.public_key)
    # 4. Run exactly the Program whose readiness was checked. Supplying an
    # already encrypted value keeps online encryption outside this run request.
    result = jit.run(
        program,
        encrypted_x,
        weight,
        bias,
        output_gain_plaintext,
        workspace=workspace,
    )
    decoded = engine.decrypt_message(
        result,
        engine.secret_key,
        is_real=True,
    )
    error = torch.abs(decoded - reference)
    max_abs_error = float(error.max())
    if max_abs_error > _VALIDATION_ATOL:
        raise RuntimeError(
            "JIT execution exceeded its fixed CKKS validation threshold: "
            f"max_abs_error={max_abs_error:.3e}, "
            f"atol={_VALIDATION_ATOL:.3e}"
        )

    print_table(
        ["stage", "operations", "runnable", "diagnostics"],
        [
            [
                "trace",
                len(traced_readiness.requirements.operations),
                traced_readiness.runnable,
                ", ".join(item.code for item in traced_readiness.diagnostics),
            ],
            [
                "lowered, unbound",
                len(before_bindings.requirements.operations),
                before_bindings.runnable,
                ", ".join(item.code for item in before_bindings.diagnostics),
            ],
            [
                "lowered, bound",
                len(ready.requirements.operations),
                ready.runnable,
                ", ".join(item.code for item in ready.diagnostics) or "none",
            ],
        ],
    )
    print()
    print_table(
        ["planned evaluation keys", "value"],
        [
            ["rotation steps", sorted(key_requirements.rotation_steps)],
            ["relinearization", key_requirements.requires_relinearization],
        ],
    )
    print()
    print("--- lowered mixed-dialect Program ---")
    print(program.to_text())
    print()
    print_table(
        [
            "pass",
            "matched",
            "transformed",
            "inserted",
            "removed",
            "skipped",
        ],
        [
            [
                report.name,
                report.stats.matched,
                report.stats.transformed,
                report.stats.inserted,
                report.stats.removed,
                report.stats.skipped,
            ]
            for report in lowered.reports
        ],
    )
    print()
    print_table(
        ["execution", "level", "scale", "max abs error"],
        [
            [
                "trace -> passes -> run",
                result.level,
                f"{result.scale:.6e}",
                f"{max_abs_error:.3e}",
            ]
        ],
    )


if __name__ == "__main__":
    main()
