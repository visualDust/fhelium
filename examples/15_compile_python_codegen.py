#!/usr/bin/env python3

"""Emit editable Eager and low-level Backend Python from two Program stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from common import print_table

from fhelium import Preset
from fhelium import compile as fh_compile
from fhelium.backend import OperationBackend
from fhelium.backend.ckks import CkksDeviceResources
from fhelium.config import CkksConfig
from fhelium.eager import Engine
from fhelium.values import Ciphertext

_ROTATION = 3
_LOGICAL_SLOTS = 16


def rotated_sum(x: torch.Tensor, rotation: int) -> torch.Tensor:
    """Add one vector to its cyclic rotation."""

    return x + torch.roll(x, shifts=rotation, dims=-1)


def _load_function(
    source: fh_compile.GeneratedPythonSource,
) -> Callable[..., object]:
    namespace: dict[str, object] = {}
    exec(source.source, namespace)
    return cast(Callable[..., object], namespace[source.entry_point])


def main() -> None:
    config = CkksConfig.parse(Preset.slots8192_scale40_depth7_int64)
    workspace = fh_compile.CompileWorkspace({CkksConfig: config})
    captured = fh_compile.capture(
        rotated_sum,
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

    # Emit Eager Python from a CKKS-depth Program. The emitter is a normal
    # optional pass and does not prescribe the passes before or after it.
    ckks_compilation = fh_compile.Pipeline(
        (
            fh_compile.EliminateDeadValuesPass(),
            fh_compile.LowerSemanticToLogicalPass(),
            fh_compile.LowerLogicalToCkksPass(),
            fh_compile.ResolveRotationKeyOperandsPass(),
            fh_compile.AssignCkksDepthsPass(entry_depth=0),
            fh_compile.AssignCkksScalesPass(
                entry_scale=config.default_scale,
            ),
            fh_compile.EmitEagerPythonPass(),
        )
    ).run(captured)
    eager_source = cast(
        fh_compile.EagerPythonSource,
        ckks_compilation.workspace[fh_compile.EagerPythonSource],
    )

    # Continue lowering, then independently emit direct Backend implementation
    # calls. Moving either emitter to an unsupported stage makes that emitter
    # report the first operation it cannot convert.
    backend_compilation = fh_compile.Pipeline(
        (
            fh_compile.AssignNttImplementationPass(
                "native-ntt", ntt_backend="radix2_indexed"
            ),
            fh_compile.LowerCkksToRnsNttPass(),
            fh_compile.SelectNttImplementationsPass(
                OperationBackend().registry
            ),
            fh_compile.PrepareOperationOperandsPass(
                OperationBackend().registry
            ),
            fh_compile.EmitBackendPythonPass(),
        )
    ).run(ckks_compilation)
    backend_source = cast(
        fh_compile.BackendPythonSource,
        backend_compilation.workspace[fh_compile.BackendPythonSource],
    )

    engine = Engine(config, rng_seed=23)
    secret_key = engine.create_secret_key(device="cpu")
    public_key = engine.create_public_key(secret_key, device="cpu")
    rotation_key = engine.create_rotation_key(
        _ROTATION,
        secret_key,
        device="cpu",
    )
    logical_x = torch.linspace(
        -0.04,
        0.04,
        _LOGICAL_SLOTS,
        dtype=torch.float64,
    )
    clear_x = logical_x.repeat(config.num_slots // _LOGICAL_SLOTS)
    encrypted_x = engine.encrypt_message(
        clear_x,
        public_key,
        depth=0,
        scale=config.default_scale,
        device="cpu",
    )

    eager_function = _load_function(eager_source)
    eager_result = eager_function(
        engine,
        encrypted_x,
        materials={f"rotation-key:{_ROTATION}": rotation_key},
        resources={},
    )
    assert isinstance(eager_result, Ciphertext)

    materializer = CkksDeviceResources(
        config=config,
        device="cpu",
        rng_seed=29,
    )
    fh_compile.prepare_material_bindings(
        backend_compilation, resources=materializer, keys=(rotation_key,)
    )
    backend_function = _load_function(backend_source)
    backend_result = backend_function(
        encrypted_x.data,
        materials=backend_compilation.material_bindings,
        resources={},
    )
    assert isinstance(backend_result, torch.Tensor)
    assert torch.equal(backend_result, eager_result.data)

    decoded = engine.decrypt_message(eager_result, secret_key)
    expected = clear_x + torch.roll(clear_x, shifts=_ROTATION, dims=-1)
    torch.testing.assert_close(
        decoded.real,
        expected,
        rtol=5e-6,
        atol=5e-6,
    )

    print("--- Eager Python ---")
    print(eager_source.source)
    print("--- Backend Python ---")
    print(backend_source.source)
    print_table(
        ["generated target", "operations", "resources"],
        [
            [
                "Eager",
                eager_source.operation_count,
                eager_source.resource_symbols,
            ],
            [
                "Backend",
                backend_source.operation_count,
                backend_source.resource_symbols,
            ],
        ],
    )


if __name__ == "__main__":
    main()
