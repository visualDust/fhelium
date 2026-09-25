# Choose operation implementations and lowerings

Implementation selection chooses the Backend code that executes an operation. Lowering selection chooses the operations that represent the calculation. Use these controls to retain a whole CKKS operation, expose RNS/NTT work, or request a particular implementation for a measured workload.

## Prerequisites

Have a correct Eager evaluator or a Compilation and a clear-message reference. Keep the input state, parameter set, and measurement scope fixed while comparing candidates. See [Build and transform a Program](build-program-pipeline.md) for a manual compilation and [Compile a callable](compile-callable.md) for a callable interface.

## 1. Inspect the available implementations

```python
from fhelium.backend import OperationBackend

backend = OperationBackend()
for implementation in backend.registry.implementations:
    print(implementation.name, [op.name for op in implementation.operation_types])
```

An implementation name is resolved together with an operation class. The presence of a name does not imply support for every operation, representation, device, or in-place call. `OperationBackend` owns the selected registry and its resource workspace; placement follows Tensor operands and the execution implementation's requirements.

## 2. Record a per-operation requirement

An NTT assignment can request the native implementation and its transform policy:

```python
from fhelium import compile as fc

ntt_selection = fc.AssignNttImplementationPass(
    "native-ntt", ntt_backend="radix2_indexed"
)
```

For other operations, `AssignImplementationsPass(selections={operation_name: implementation_name})` records the requested name on matching operations. Obtain both names from the registry inventory. `overwrite=True` permits replacing a prior assignment; otherwise a conflicting assignment fails. This pass records a constraint and does not establish executable coverage.

Insert assignment where the intended operations exist and before a transformation that could replace them. Inspect pass decisions and the transformed Program to confirm that the requirement reached the expected operations. The default recipe performs local lowering and fusion selection; it does not benchmark alternatives or retry an execution failure with another implementation.

## 3. Preserve a whole operation or expose its numerical decomposition

```python
lowering = fc.LowerCkksToRnsNttPass(
    selections={"fhelium_ckks.rescale": "rns-drop-leading-prime"},
    preserve={"fhelium_ckks.multiply"},
)
```

This pass preserves whole CKKS multiplication and selects the named rescale lowering. The recipe name is an implementation identifier; the mathematical rescale divisor still follows the configured dropped depth group. The resulting mixed-level Program can contain both whole-operation implementations and RNS/NTT operations.

An assigned whole operation must be preserved when applying this lowering pass. Lowering rejects a selected operation if the rewrite would erase its recorded implementation assignment. Unhandled operations can remain in the Program for another pass or an implementation at their current level.

## 4. Apply a transform policy in Eager

```python
import fhelium as fh
from fhelium.eager import Engine

engine = Engine(
    fh.Preset.slots8192_scale40_depth7_int64,
    ntt_backend="radix2_indexed",
)
```

This selects the Engine's NTT policy while the actual device is chosen by factories or operands. Keep key material on the operation device. For CUDA alternatives, [screen compatible NTT backends](screen-ntt-backends.md), then [interpret the timing evidence](choose-ntt-backend.md). The retained legacy screening runner measures its own evaluator, so confirm the result with the current execution path being deployed.

## 5. Validate and record the selected execution

Prepare materials, link with the same Backend, and inspect `executable.manifest` for bound implementation and resource information. Compare decoded results against the clear-message reference before accepting performance evidence. Record whether timing includes transformation, preparation, linking, first device-code compilation, input staging, and steady execution.

See [Operation declaration and implementation selection](../developer/operation-registration-and-selection.md) for registration, [RNS and NTT](../developer/rns-and-ntt.md) for numerical resources, and [Generated kernels and fusion](../developer/compiled-execution-and-kernels.md) for fused execution.
