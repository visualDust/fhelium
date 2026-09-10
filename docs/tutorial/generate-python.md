# Generate editable Python from a Program

**Example source:** [`examples/20_generate_python.py`](https://github.com/VisualDust/fhelium/blob/main/examples/20_generate_python.py)

Example 20 uses two optional Compile passes to emit Python at two different Program stages:

- `EmitEagerPythonPass` translates CKKS operations into calls on the public Eager `Engine`;
- `EmitBackendPythonPass` resolves the operations visible at its position and emits direct calls to their Backend implementation classes.

Both passes scan the Program supplied at their own pipeline position, leave it unchanged, and publish source in the shared `CompileWorkspace`. Neither pass inserts other transformations or requires a particular predecessor. If an operation at the selected stage cannot be represented, that emitter reports the operation and stops.

## Run the example

```bash
python examples/20_generate_python.py
```

The example emits both source forms, executes them on CPU, checks that their Tensor payloads are identical, and compares the decrypted Eager result with the clear rotated sum.

## Eager Python emission

The example first transforms a captured `torch.roll` plus addition into CKKS operations and places the Eager emitter at that point:

```python
from fhelium import compile as fh_compile

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

eager_source = ckks_compilation.workspace[
    fh_compile.EagerPythonSource
]
```

The emitted function contains ordinary Eager calls:

```python
def generated_eager(engine, x, *, materials, resources):
    rotation_key = resources["rotation-key:3"]
    rotated = engine.rotate_with_key(x, rotation_key)
    result = engine.add(x, rotated)
    return result
```

The complete emitted source retains SSA aliases introduced by the current Program rather than reconstructing the original Python spelling. Materials and keys remain function inputs. The source does not contain Tensor payloads, secret-key data, devices, RNS contexts, or buffers.

The Eager emitter accepts flat CKKS operations that have a public Engine equivalent, one-to-one boundary casts, material/resource references, constants, and `func.return`. Semantic, logical, RNS, NTT, distributed, unknown, and region-owning operations are outside this emitter's current surface. A caller chooses a CKKS stage when Eager-style source is desired; the pass itself does not choose or enforce that stage.

## Backend Python emission

The example continues from the CKKS Compilation, lowers addition into RNS arithmetic while retaining rotation as a whole CKKS operation, and invokes the Backend emitter:

```python
backend_compilation = fh_compile.Pipeline(
    (
        fh_compile.LowerCkksToRnsNttPass(
            preserve=frozenset({ckks.RotateOp.name})
        ),
        fh_compile.EmitBackendPythonPass(),
    )
).run(ckks_compilation)

backend_source = backend_compilation.workspace[
    fh_compile.BackendPythonSource
]
```

The generated module imports and instantiates the selected implementation classes, reconstructs literal `OperationInvocation` descriptors, and calls each implementation directly:

```python
implementation_0 = NativeRnsLinearImplementation()
invocation_0 = OperationInvocation(
    operation_type=AddStandardOp,
    operand_count=2,
    result_count=1,
    ...
)

def generated_backend(x, y, *, materials, resources):
    rns = BoundResource(
        "active-rns-parameters",
        "rns-parameters",
        resources["active-rns-parameters"],
    )
    result, = implementation_0.execute(
        invocation_0,
        (x, y),
        (rns,),
        in_place=False,
    )
    return result
```

This source does not load a Program, call `ProgramExecutable`, or run another SSA interpreter. Its inputs and outputs are low-level Tensor payloads. The developer can edit implementation construction, resource use, call order, temporary values, or replace a call with experimental code.

`EmitBackendPythonPass` uses the built-in implementation registry by default. A developer using a custom registry supplies it directly:

```python
fh_compile.EmitBackendPythonPass(registry=my_registry)
```

Resolution happens while that pass scans the current Program. It does not consume a `ProgramDispatchTable` produced by another pass and therefore does not impose a pipeline order. Unsupported or ambiguous operations are reported by the emitter instead of being lowered or silently assigned.

Implementation classes must be importable at module scope. A zero-state custom implementation can be constructed directly. The emitter refuses to copy constructor state from an external custom implementation into source; a developer can first replace it with an importable source-level implementation or emit at another stage. FHElium-owned implementation constructor fields contain implementation-selection data and may be reproduced when needed.

## Materials and live resources

Both source forms use ordinary mappings:

```python
generated(
    input_tensor,
    materials={"weight": weight_tensor},
    resources={
        "rotation-key:3": rotation_key,
        "active-rns-parameters": rns_context,
    },
)
```

`EagerPythonSource` records the referenced material and resource symbols. `BackendPythonSource.resource_requirements` also records `(symbol, kind)` pairs so the caller can ask an existing resource materializer to construct the non-key resources. The generated module wraps those ordinary objects as `BoundResource` only at the Backend execution ABI.

The source emitted by one pass describes the Program snapshot seen at that position. Later passes may continue transforming the Compilation; they do not rewrite or invalidate an earlier source artifact. The caller retains the source corresponding to the stage it intends to edit or execute.

Generated Python is executable code. Execute only source emitted from trusted Programs and implementation registries, and review edited source before running it with live keys or resources.

::: details Source

<<< @/../examples/20_generate_python.py

:::
