# Customize and audit a JIT pass pipeline

**Example source:** [`examples/21_jit_custom_pipeline.py`](https://github.com/VisualDust/fhelium/blob/main/examples/21_jit_custom_pipeline.py)

Example 21 traces an encrypted rotated quadratic, inserts a caller-defined audit
into the default JIT pipeline, retains the audit result in the shared
`Workspace`, provisions exactly the rotation and relinearization keys required
by the lowered program, and executes the program with CUDA CKKS evaluation.

The example inserts a non-rewriting policy/audit pass that verifies required
CKKS operations and records their final surface after the default scheduling
passes.

## Run the complete example

From the repository root:

```bash
python examples/21_jit_custom_pipeline.py \
  --preset slots8192-scale40-levels7-int64
```

The script runs on the selected CPU or CUDA engine and prints:

- every pass in its pipeline position;
- match, transformation, insertion, removal, and skip counts;
- evaluation-key requirements;
- readiness before and after runtime binding;
- the result level and actual scale;
- the analytical cleartext bound, fixed validation threshold, maximum absolute error,
  and root-mean-square error.

## 1. Define the encrypted computation

The captured function combines one rotation, a public gain, a
ciphertext-ciphertext multiplication, and a public bias:

```python
def rotated_quadratic(
    x: torch.Tensor,
    gain: float,
    bias: torch.Tensor,
    rotation: int,
) -> torch.Tensor:
    mixed = (x + torch.roll(x, shifts=rotation, dims=-1)) * gain
    return mixed * mixed + bias
```

For slot vector $x$, rotation $r$, gain $g$, and bias $b$, the cleartext
function is

$$
m = g\left(x + \operatorname{roll}(x,r)\right),
\qquad
 y = m^2 + b.
$$

Example 21 fixes

$$
r=3,\qquad g=0.625,
$$

while `x`, `gain`, and `bias` remain runtime arguments.

## 2. Declare capture roles and retain one workspace

The caller creates the workspace before capture:

```python
workspace = jit.Workspace(
    {
        "programmer/pipeline-policy": (
            "default lowering plus explicit CKKS audit and validation"
        )
    }
)
```

It then assigns one role to every function parameter:

```python
captured = jit.trace(
    rotated_quadratic,
    inputs={
        "x": jit.encrypted(),
        "gain": jit.message(),
        "bias": jit.message(),
        "rotation": jit.static(3),
    },
    workspace=workspace,
)
```

| Parameter | Role | Consequence |
| --- | --- | --- |
| `x` | `encrypted()` | Runtime input is a compatible `Ciphertext`, or a Tensor encrypted online when a compatible public key is bound |
| `gain` | `message()` | Runtime scalar remains public data until an encrypted consumer requires plaintext preparation |
| `bias` | `message()` | Runtime Tensor remains public and is prepared for encrypted addition by lowering |
| `rotation` | `static(3)` | Capture specializes the integer and omits it from the runtime signature |

`captured.program` is the source-independent xDSL `Program`.
`captured.workspace` is the same mapping object supplied by the caller. Program text
contains operations and symbolic identities; live engines, keys, policies,
handlers, and analysis results remain outside the graph.

## 3. Implement a non-rewriting audit pass

A JIT pass has a stable `name` and a
`run(program, workspace) -> PassResult` method. Example 21 defines:

```python
@dataclass(frozen=True)
class AuditExplicitCkksPass:
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
```

The pass has two independent effects:

1. it rejects a lowered surface that still contains recognized semantic or
   logical arithmetic, or that lacks an operation required by this workload;
2. it publishes the observed operation surface under a caller-owned
   workspace key.

The pass deliberately returns `PassResult.unchanged(...)`. `matched` records
what it inspected; no transformation count is fabricated. A successful audit
is evidence about the current program, not a numerical optimization.

The workspace does not assign a schema or invalidation policy to the custom
analysis key. The producer and every consumer of
`"analysis/explicit-ckks-operation-surface"` must define and maintain its
schema, semantics, and invalidation policy.

## 4. Insert the pass at one named position

The example extends the default pipeline without copying its pass tuple:

```python
pipeline = (
    jit.default_pipeline()
    .after("late-relinearization", AuditExplicitCkksPass())
    .then(jit.ValidateExecutableGraphPass())
)
```

`after(...)` requires one uniquely named target. This makes the insertion point
part of the caller's inspectable policy rather than an implicit callback. The
audit runs after all default lowering and scheduling-report passes. The final
validator checks the executable schema independently.

The late-rescale and late-relinearization passes in the default pipeline are
conservative reporting passes and leave placement unchanged. Example 21's
audit verifies the resulting operation surface.

Run the pipeline:

```python
lowered = pipeline.run(captured.program, captured.workspace)
if lowered.workspace is not workspace:
    raise RuntimeError("the custom pipeline did not retain its Workspace")
program = lowered.program
```

The pipeline clones the source program once, gives every pass the same
workspace object, structurally verifies every returned program, and records one
report per pass. A pass may legally report an unchanged result when its local
pattern is absent or intentionally retained.

## 5. Inspect pass evidence

The ordered names are available before execution:

```python
for position, name in enumerate(pipeline.names):
    print(position, name)
```

After execution, each `PassReport` contains:

```python
for report in lowered.reports:
    print(
        report.name,
        report.stats.matched,
        report.stats.transformed,
        report.stats.inserted,
        report.stats.removed,
        report.stats.skipped,
        report.diagnostics,
    )
```

For this circuit, lowering introduces explicit plaintext preparation, NTT
transitions around ciphertext multiplication, relinearization, and two rescale
operations. The custom audit reports the final operation surface without
rewriting it. `ValidateExecutableGraphPass` provides a separate executable
schema gate; it does not replace the runtime capability check.

## 6. Plan the required evaluation keys

Key planning scans the explicit lowered operations:

```python
key_plan = jit.analyze_evaluation_key_requirements(program)
```

For this program:

```text
rotation steps       [3]
relinearization key  True
```

Provision exactly those capabilities:

```python
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
```

The requirement analysis is pure: it does not generate keys, inspect their
storage, or mutate the workspace. Readiness subsequently checks both capability
presence and compatibility of every required key with the selected engine,
including context, device, dtype, ring dimension, prime layout, NTT/Montgomery
state, hybrid digit structure, and canonical rotation step.

## 7. Compare readiness before and after binding

Before adding runtime services, the example expects the program to be blocked:

```python
before_bindings = program.readiness(workspace)
if before_bindings.runnable:
    raise RuntimeError("the CKKS Program was ready without runtime bindings")
```

The maintained workload reports:

```text
missing-engine
missing-evaluation-keys
```

Bind the engine and planned key inventory to the same retained workspace:

```python
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
```

Readiness is observational. It validates the selected entry, schemas,
obligations, handlers, resources, engine, and key compatibility without
executing operations or invoking resolvers. `jit.run(...)` repeats this gate at
the execution requirements.

## 8. Construct bounded inputs and the semantic reference

The deterministic input construction is

```python
index = torch.arange(engine.num_slots, dtype=torch.float64)
clear_x = 0.02 * torch.sin(0.013 * index) + 0.01 * torch.cos(0.031 * index)
bias = 0.004 * torch.sin(0.007 * index + 0.2)
reference = captured.reference(clear_x, 0.625, bias)
```

It establishes

$$
\lVert x\rVert_\infty \le 0.03,
\qquad
\lVert b\rVert_\infty \le 0.004.
$$

Because a cyclic rotation preserves the infinity norm,

$$
\lVert m\rVert_\infty
\le 2(0.03)(0.625)=0.0375.
$$

Therefore

$$
\lVert y\rVert_\infty
\le 0.0375^2+0.004
=0.00540625.
$$

The script checks this bound before encrypted evaluation. A violation indicates
that the deterministic input fixture or circuit definition changed; it is not hidden by
a numerical tolerance adjustment.

## 9. Execute and enforce the two-rescale validation threshold

Encrypt the secret input and execute the lowered program:

```python
encrypted_x = engine.encrypt_message(clear_x, engine.public_key)
encrypted_result = jit.run(
    program,
    encrypted_x,
    0.625,
    bias,
    workspace=workspace,
)
decoded = engine.decrypt_message(
    encrypted_result,
    engine.secret_key,
    is_real=True,
)
```

The default scale-40 lowering has two explicit rescale stages:

1. multiplication of the encrypted sum by the prepared public gain;
2. ciphertext-ciphertext squaring followed by relinearization.

The example fixes

```python
_VALIDATION_ATOL = 2e-5
```

This value is not configurable from the command line and is not adjusted from
the observed output. It reserves less than 0.4% of the analytical cleartext
bound for aggregate CKKS approximation, encryption, key switching, NTT, and
rescale error.

A representative maintained CUDA validation produced:

```text
result level     2
result scale     1.099511e+12
clear |max|      5.294e-03
validation atol  2.000e-05
max abs error    1.637e-08
rms error        2.782e-09
```

Observed error is evidence for this preset, implementation, and deterministic
input. The fixed `2e-5` threshold is the maintained example acceptance criterion; the
observed value is not a replacement tolerance and does not define a general
JIT accuracy guarantee.

## Complete source

<<< @/../examples/21_jit_custom_pipeline.py

## Continue

- [JIT programs](../concepts/unified-jit-programs.md) defines Program,
  Workspace, pass, analysis, readiness, and handler responsibilities.
- [JIT internals](../developer/unified-jit-internals.md) specifies canonical
  schemas, pass scope, structural verification, and runtime trust decisions.
- [Trace, transform, and run a JIT program](unified-jit.md) covers the
  trace-first baseline used by Example 19.
- [Import and execute textual JIT IR](jit-textual-ir.md) covers the IR-first,
  CPU-only workflow in Example 20.
