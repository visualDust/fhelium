#!/usr/bin/env python3

"""Compose built-in Compile passes, link the Program, and execute it."""

from __future__ import annotations

from collections import Counter
from typing import cast

import torch
from common import print_table

from fhelium import Preset
from fhelium import compile as fh_compile
from fhelium import ir
from fhelium.backend import OperationBackend
from fhelium.backend.ckks import CkksDeviceResources
from fhelium.backend.rns.chain import RnsChain
from fhelium.backend.rns.decomposition import HybridRnsDecomposition
from fhelium.backend.rns.layout import RnsLayout
from fhelium.config import CkksConfig
from fhelium.eager import Engine
from fhelium.ir.dialects import ckks
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


def main() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_levels7_int64)
    device = "cpu"
    workspace = fh_compile.CompileWorkspace({CkksConfig: config})

    captured = fh_compile.capture(
        rotated_quadratic,
        inputs={
            "x": fh_compile.encrypted(slots=_LOGICAL_SLOTS),
            "rotation": fh_compile.static(_ROTATION),
        },
        workspace=workspace,
    )

    # This workload selects one concrete composition from the built-in pass
    # catalog. FHElium does not impose this sequence as a global default.
    pipeline = fh_compile.Pipeline(
        (
            fh_compile.EliminateDeadValuesPass(),
            fh_compile.LowerSemanticToLogicalPass(),
            fh_compile.InsertMultiplyNttTransitionsPass(),
            fh_compile.LowerLogicalToCkksPass(),
            fh_compile.ResolveRotationKeyOperandsPass(),
            fh_compile.InsertRelinearizationPass(),
            fh_compile.InsertRescalePass(),
            fh_compile.AssignCkksLevelsPass(entry_level=0),
            fh_compile.AssignCkksScalesPass(
                entry_scale=config.default_scale,
            ),
            fh_compile.LowerCkksToRnsNttPass(
                preserve=frozenset({ckks.RotateOp.name})
            ),
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
    evaluation_keys = (
        *rotation_keys,
        *((relinearization_key,) if relinearization_key is not None else ()),
    )

    chain = RnsChain(config.num_q_primes, config.num_p_primes)
    layout = RnsLayout(chain, HybridRnsDecomposition(chain))
    resources = CkksDeviceResources(
        config=config,
        rns_layout=layout,
        device=device,
        rng_seed=29,
    )
    executable = OperationBackend(
        keys=evaluation_keys,
        materializer=resources,
    ).link(compiled)

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
        level=0,
        scale=config.default_scale,
        device=device,
    )
    result = cast(
        Ciphertext,
        executable.run(encrypted_x),
    )
    decoded = engine.decrypt_message(result, secret_key)
    torch.testing.assert_close(
        decoded.real,
        expected,
        rtol=5e-6,
        atol=5e-6,
    )
    torch.testing.assert_close(
        decoded.imag,
        torch.zeros_like(decoded.imag),
        rtol=0.0,
        atol=5e-6,
    )

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
            ["result level", result.level],
            ["result scale", f"{result.scale:.6e}"],
            [
                "maximum error",
                f"{float((decoded.real - expected).abs().max()):.6e}",
            ],
        ],
    )
    print()
    print_table(
        ["pass", "matched", "transformed", "inserted", "removed", "skipped"],
        [
            [
                report.name,
                report.stats.matched,
                report.stats.transformed,
                report.stats.inserted,
                report.stats.removed,
                report.stats.skipped,
            ]
            for report in compiled.reports
        ],
    )


if __name__ == "__main__":
    main()
