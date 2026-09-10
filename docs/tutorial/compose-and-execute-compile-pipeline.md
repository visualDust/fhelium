# Compose and execute a pipeline from built-in Compile passes

**Example source:** [`examples/17_compose_and_execute.py`](https://github.com/VisualDust/fhelium/blob/main/examples/17_compose_and_execute.py)

Example 17 is the ordinary end-to-end Compile workflow. It captures a supported PyTorch computation, composes existing FHElium passes, lowers the Program to operations covered by the Backend, links live keys and arithmetic resources, executes encrypted inputs, and checks the decrypted result.

## Run the example

From the repository root:

```bash
python examples/17_compose_and_execute.py
```

The example runs on CPU and prints compact captured/backend-ready Program inventories, evaluation-key requirements, result state, numerical error, and pass reports.

## Source computation

The source adds a vector to one cyclic rotation and squares the sum:

```python
def rotated_quadratic(x, rotation):
    mixed = x + torch.roll(x, shifts=rotation, dims=-1)
    return mixed * mixed
```

Capture assigns `x` the encrypted role and treats `rotation` as a compile-time integer. The logical 16-slot input is tiled across the complete CKKS slot ring before encryption, so a full-ring rotation implements the same periodic layout used by the clear computation.

## Caller-selected pipeline

FHElium does not provide a global default pipeline. The example chooses built-in passes for this workload:

```python
from fhelium import compile as fh_compile

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
        fh_compile.AssignCkksScalesPass(
            entry_scale=config.default_scale,
        ),
        fh_compile.LowerCkksToRnsNttPass(
            preserve=frozenset({ckks.RotateOp.name})
        ),
    )
)
compiled = pipeline.run(captured)
```

This sequence records the following caller-selected pipeline choices:

- it retains rotation as a registered whole CKKS operation;
- it inserts immediate relinearization and rescale operations for the ciphertext product;
- it assigns concrete depths and per-value actual scales after transition placement;
- it lowers the remaining CKKS arithmetic into RNS and NTT operations.

Another caller may choose late transition placement, additional analyses, a different lowering route, or a pipeline that intentionally stops at an intermediate Program.

## Keys, resources, and linking

`analyze_evaluation_key_requirements(...)` finds one rotation step and a relinearization requirement in the transformed Program. The example creates only those evaluation keys. `CkksDeviceResources` materializes the requested CPU RNS, NTT, rescale, and key-switch resources while linking:

```python
backend = OperationBackend(
    keys=evaluation_keys,
    materializer=resources,
)
executable = backend.link(compiled)
```

The Compilation retains Program-external materials and pass reports. The Backend workspace supplies live keys and arithmetic resources. `link(...)` matches those two sources and returns a `ProgramExecutable` without changing the source Compilation.

## Encrypted execution

The example encrypts one tiled input, invokes the executable through the public `Ciphertext` boundary, decrypts the result, and compares it with the retained `CapturedCallable` reference:

```python
result = executable.run(encrypted_x)
decoded = engine.decrypt_message(result, secret_key)
expected = captured_callable.reference(clear_x)
```

Registered Backend implementations remain Tensor-oriented internally. `ProgramExecutable` performs the public value adaptation at the outer execution interface.

## Why this differs from Example 19

Example 17 uses only operations and passes already supplied by FHElium. [Example 19](customize-compile-pass-and-pipeline.md) defines a new pass that rewrites captured `torch.matmul` into a baby-step/giant-step schedule before composing, linking, and executing its pipeline. [Example 18](ir-textual-program.md) instead focuses on textual Program IR and analysis-pass composition and intentionally stops before Backend execution.

::: details Source

<<< @/../examples/17_compose_and_execute.py

:::
