# Compose and execute a pipeline from built-in Compile passes

**Example source:** [`examples/12_compile_pipeline.py`](https://github.com/VisualDust/fhelium/blob/main/examples/12_compile_pipeline.py)

Example 12 is the ordinary end-to-end Compile workflow. It captures a supported PyTorch computation, composes existing FHElium passes, lowers the Program to operations covered by the Backend, links live keys and arithmetic resources, executes encrypted inputs, and checks the decrypted result.

## Run the example

From the repository root:

```bash
python examples/12_compile_pipeline.py
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

Pipelines express workload-specific choices. The example supplies the configuration and implementation registry through the capture workspace:

```python
from fhelium.backend import OperationBackend, OperationImplementationRegistry
from fhelium import compile as fh_compile

backend = OperationBackend()
workspace = fh_compile.CompileWorkspace({
    CkksConfig: config,
    OperationImplementationRegistry: backend.registry,
})
```

Pass the workspace to `capture`. The pipeline then composes the built-in transformations:

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
        fh_compile.AssignNttImplementationPass("native-ntt", ntt_backend="radix2_indexed"),
        fh_compile.LowerCkksToRnsNttPass(),
        fh_compile.SelectNttImplementationsPass(),
    )
)
compiled = pipeline.run(captured)
```

This sequence records the following caller-selected pipeline choices:

- it inserts immediate relinearization and rescale operations for the ciphertext product;
- it assigns concrete depths and per-value actual scales after transition placement;
- it declares the configured key layout;
- it selects the indexed radix-2 NTT schedule and lowers CKKS arithmetic to RNS/NTT operations.

Another caller may choose late transition placement, additional analyses, a different lowering route, or a pipeline that returns an intermediate Program.

Lowering and implementation selection declare their required materials using the workspace registry. Custom key layouts can be supplied through `workspace["key_layouts"]`, keyed by material symbol or label; see [Materials and preparation](../developer/materials-and-preparation.md#evaluation-key-layout-declarations) for the fields and selection rules.

## Keys, resources, and linking

`analyze_evaluation_key_requirements(...)` finds the required rotation and relinearization uses in the compiled Program. The example creates compatible keys and uses `prepare_material_bindings` to supply the numerical materials for execution:

```python
fh_compile.prepare_material_bindings(
    compiled,
    resources=resources,
    keys=(*rotation_keys, *((relinearization_key,) if relinearization_key is not None else ())),
)
executable = backend.link(compiled)
```

`prepare_material_bindings` supplies Tensor data for the declared symbols, and linking builds an executable from the resulting bindings.

## Encrypted execution

The example encrypts one tiled input, invokes the executable through the public `Ciphertext` boundary, decrypts the result, and compares it with the retained `CapturedCallable` reference:

```python
result = executable.run(encrypted_x)
decoded = engine.decrypt_message(result, secret_key)
expected = captured_callable.reference(clear_x)
```

Registered Backend implementations remain Tensor-oriented internally. `ProgramExecutable` performs the public value adaptation at the outer execution interface.

## Why this differs from Example 14

Example 12 uses only operations and passes already supplied by FHElium. [Example 14](customize-compile-pass-and-pipeline.md) defines a new pass that rewrites captured `torch.matmul` into a baby-step/giant-step schedule before composing, linking, and executing its pipeline. [Example 13](ir-textual-program.md) instead focuses on textual Program IR and analysis-pass composition and returns the analyzed Program for inspection.

::: details Source

<<< @/../examples/12_compile_pipeline.py

:::

## Leaving NTT choices open

Logical NTT operations can leave their algorithm unassigned. Add `SelectNttImplementationsPass()` where the required facts become available, then supply any newly introduced table materials before linking. `AssignNttImplementationPass` can constrain an algorithm, group width or fixed radix without assigning every execution detail. See [NTT semantics and selection](../developer/compiled-execution-and-kernels#ntt-semantics-and-selection) for the selection order and table-data responsibilities.
