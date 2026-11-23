# Diagnose Compile preparation failures

Compile preparation transforms a calculation, supplies numerical operands and live resources, and links implementations for execution. Identify the failing stage and the missing requirement before changing CKKS state or provisioning resources.

## Prerequisites

Retain the source calculation, representative input metadata, selected pipeline, and full exception. Record material symbols and descriptions without printing secret payloads. For a manual Program, keep the last successful Compilation; for a callable, keep successful specializations and identify the input conditions that triggered the failure.

## 1. Locate the failing stage

| Stage | Evidence | First action |
| --- | --- | --- |
| Python capture | Unsupported Python construct, argument type, or captured call | Isolate the expression and choose a supported frontend or describe the operation directly in a Program. |
| Transformation | Pass name, diagnostics, before/after operation inventory | Inspect the facts and operation family required by that pass. |
| Material preparation | Unresolved material symbols | Supply the missing Tensor or the caller-owned data provider and keys. |
| Linking | Unsupported operation, assigned implementation, representation, or live-resource error | Compare the transformed operations with the selected Backend registry and resource bindings. |
| Callable invocation | `SpecializationMiss` with `on_miss="error"` | Prepare the admitted input conditions before invocation. |
| Numerical execution | Native/device failure or incorrect decoded result | Check placement, ABI, input state, key relation, and the mathematical schedule. |

Manual Programs skip Python capture. `capture` and `capture_eager` produce Compilations that can be inspected before execution. `fc.compile` performs capture where needed and then prepares the same kinds of Program facts and bindings behind a function interface.

Preparation errors propagate from the call or from `compiled.prepare(*inputs)`. A successful CPU call does not establish Graph capture: `stats()["graphs"]` reports the prepared fallback and its reason. A disk-cache miss or ineligible source still permits ordinary compilation.

## 2. Inspect the Program and reports

```python
print(compilation.program.to_text())
for report in compilation.reports:
    print(report.name, report.seconds, report.stats, report.diagnostics, report.decisions)
print(compilation.program.material_descriptions)
print(sorted(compilation.material_bindings))
```

This block uses the Compilation at the last successful stage. [Visualize mixed-level IR](visualize-mixed-level-ir.md) helps trace the producer of a mismatched value. Unknown shape, depth, scale, device, or representation can be valid in an intermediate Program; determine which execution consumer now requires the missing fact. Supply a known entry state, a representative input for specialization, or the appropriate assignment pass. Do not guess prime rows or scales.

## 3. Resolve missing materials without changing their meaning

Call `fc.prepare_material_bindings(compilation, resources=resources, keys=keys)` with the providers and evaluation keys selected for the task. Inspect the returned unresolved symbols. This helper fills only missing entries; an existing incorrect assignment remains the caller's responsibility.

If multiple keys could satisfy a description, bind the intended key under the Program symbol or assign its `.data` directly in `compilation.material_bindings`. Verify its cryptographic relation outside the Tensor layout. Key generation belongs in the application's provisioning stage; material lookup and linking do not generate missing keys. See [Bind and deploy a Program](persist-compiled-program.md).

## 4. Resolve operation and implementation requirements

Inventory `backend.registry.implementations` and match both operation class and requested implementation name. A partial lowering can leave mixed-level operations, but each executed operation needs supported execution or a further lowering. An assignment that selects a whole operation must survive later transformations; use the lowering pass's `preserve` control rather than erasing the assignment accidentally.

`backend.diagnostics(operation)` returns a tuple of diagnostic strings for implementation or named-resource resolution failures on the selected operation; an empty tuple means those checks succeeded. For device-specific failure, verify the installed native backend and the placement of every numerical operand and key; selecting a Backend does not transfer them.

## 5. Distinguish cache misses from changed captured state

A callable can prepare a new specialization for changed shapes, value state, placement, or immutable scalar parameters. With `on_miss="error"`, call `prepare` for intended new conditions during setup. Changing ordinary input Tensor contents does not require a new specialization when metadata stays compatible.

A new input signature can miss the resident and optimized-instance caches while reusing a source-family template. Inspect the specialization’s `structure` status and `structure_layer`: a template hit avoids recapture, while a guard miss means current metadata or material state no longer matches an observed capture condition. `unsupported` means the captured operations lack complete template rules and use ordinary compilation. Lowering and fusion plan misses can be local to changed key or digit layouts; they do not imply that every operation was reselected.

Inspect `local_plans` when a structural hit still spends time in selection or fusion:

```python
for record in compiled.stats()["specializations"]:
    local = record.get("local_plans", {})
    print("Local recipe requests:", local.get("occurrences", 0))
    print("Recipe hits:", local.get("recipe_hits", 0),
          "misses:", local.get("recipe_misses", 0))
    print("Window facts hits:", local.get("facts_hits", 0),
          "misses:", local.get("facts_misses", 0))
```

The current local consumer covers one operation and its direct operand/result ports. `occurrences` counts recipe requests, including requests in different pass scopes. Recipe hits reuse a local structural recipe; facts hits reuse the current operation’s evaluated facts within one read-only analysis window. Repeated probes can produce many facts hits for one recipe request. Global scheduling and final fusion checks still run on this path. Use pass and preparation timings to judge its cost: hit counts measure reuse frequency, not elapsed-time savings. See [Prepared host execution](../developer/prepared-host-execution.md) for counter lifetimes.

Changing a captured Python constant requires clearing the callable; changing its source Program or pass policy requires a new callable. Rebinding a material dictionary after linking is not a supported way to retarget a live executable. Prepare and link again with the intended assignments.

## 6. Confirm the repair at the same scope

Repeat the failed preparation with unchanged input conditions, inspect the resulting reports and executable manifest, and compare decoded outputs with the original clear-message criterion. If a different algebraic schedule was required, account for its rounding and scale transitions rather than requiring ciphertext residue equality. Use [Value-state diagnosis](diagnose-value-state-mismatch.md) for CKKS mismatches and [Prepared host execution](../developer/prepared-host-execution.md) for linking and execution internals.
