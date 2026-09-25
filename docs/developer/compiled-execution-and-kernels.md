# Generated kernels, fusion, and NTT execution

Compile constructs fusion regions from compatible represented operations, and selected Backend implementations generate their device execution. The resulting Program can be linked and run manually or reused through callable Compile. This page owns fusion representation, modular code generation, and generated NTT execution; [Compilation and passes](compilation-and-passes.md) owns the surrounding recipe and [prepared host execution](prepared-host-execution.md) owns invocation and caches.

## From represented operations to kernels

`compile/passes/fusion.py` selects regions through implementation-provided matching rules. `backend/triton/_expressions.py` represents component arithmetic, `_pointwise.py` generates modular kernels, and `_ntt.py` plus `_ntt_codegen.py` prepare transform regions. `_tensor.py` separately handles ordinary Tensor fusion, while `_launch.py` owns prepared launch support.

Whole native CKKS implementations remain selectable alongside lowered and generated regions. A generated region can execute several kernels, and native NTT middle stages can participate in that sequence. Fusion changes the execution grouping while retaining equations, effects, external outputs, and the selected numerical representations.

## NTT semantics and selection

The logical NTT operations specify forward or normalized inverse negacyclic transforms and their source/destination residue representations. Their RNS parameter operand and optional table operands are numerical Tensors. Algorithm selection preserves the transform semantics and the ordering consumed by other polynomial operations.

An NTT may have no schedule assignment, partial algorithm constraints, or a complete schedule and execution implementation. `AssignNttImplementationPass` can express partial choices without filling the remaining fields:

```python
from fhelium import compile as fc
from fhelium.backend import OperationBackend
from fhelium.compile.passes.backend import (
    AssignNttImplementationPass,
    SelectNttImplementationsPass,
)

backend = OperationBackend()
ntt_selection = fc.Pipeline((
    AssignNttImplementationPass(algorithm="radix2_compact", group_width=8),
    SelectNttImplementationsPass(backend.registry),
))
```

This example constrains compact radix-2 transforms. The other native algorithm families are `radix2_indexed` and `fixed_radix`; `radix` constrains the latter. A concrete `implementation` assignment applies to logical NTT operations, so place that assignment after CKKS lowering. A complete `ntt_backend` schedule can also be assigned to CKKS operations that contain internal transforms.

`SelectNttImplementationsPass` consults the selected implementation's schedule selection function. Existing assignments remain constraints. Native selection uses represented placement, transform size, algorithm choices and table layout; it applies a stable default rather than timing candidates. Missing facts can leave a choice unresolved. It does not fill an unknown device or move operands.

When a selected transform requires additional tables, the pass introduces ordinary material references. Supply these separately through `Compilation.material_bindings`, optionally using `prepare_material_bindings` with an existing resource provider. Supplied table operands are not discarded or regenerated. Compact radix-2 group sizes share a table layout, so selecting a different group can retain the same table Tensors. Other layouts are not silently substituted.

Capture retains the table layout chosen by an Eager data provider. An Engine's implicit default does not force that exact group size in Compile; an explicitly supplied `ntt_backend` remains a constraint. Context transforms and registered native operations use the same prepared numerical calls. The Context owns table provision, while execution consumes the supplied Tensor operands.

## Fusion representation

A fusion region retains the operations it combines and exposes its external operands and results. Its body records the represented computation for implementation matching and code generation.

`FuseOperationsPass` groups supported arithmetic and transforms subject to SSA dependencies, compatible resources/layouts, and effects. Representation casts that only change IR types do not require a device kernel. Externally used intermediate results must remain results of the fused region, so fusion does not discard fan-out.

A fused region must not assume that distinct input values have distinct storage. For pure out-of-place arithmetic, aliasing inputs is valid. Mutation, transfers, unsupported operations, and incompatible resource requirements constrain grouping.

The pass receives caller-selected implementations. Each supplies its operation and layout matching rules through the Backend fusion interface; the pass owns SSA region construction and preserves explicit implementation assignments. It incrementally groups consecutive compatible operations and uses caller order to choose among matching implementations. The match count records the number of computational operations. Binding supplies missing layout facts.

Triton source templates and launch arithmetic remain in Backend. Native operations outside the selected groups remain ordinary Backend calls. The generated implementation owns its component expression representation.

Compile's Python source emitters export a Program stage as Eager calls or direct Backend calls. They have a different product from the Backend kernel generator. Backend Python export and linked execution share host control-flow emission. Supported structured regions become Python control flow or prepared combine callables; fusion regions are prepared by their selected Backend implementation.

## Triton implementations

`fhelium.backend.triton` supplies individual-operation and region implementations:

- `TritonRnsImplementation`, named `triton-rns-pointwise`, implements individual supported RNS arithmetic operations.
- `TritonFusionImplementation`, named `triton-rns-fused`, generates a component-indexed arithmetic kernel.
- `TritonFusionImplementation(name="triton-ntt-fused", include_ntt=True)` supplies the region implementation selected for transform schedules.

`TritonTensorFusionImplementation`, named `triton-tensor-fused`, separately implements ordinary float32/float64 elementwise expressions and last-axis rolls. Automatic selection requires known CUDA layouts. Input strides, broadcasting and separate escaping outputs are preserved. Floating-point contraction is disabled; autograd inputs retain ordinary Torch execution.

All use the existing Backend implementation registry. Importing or registering them does not globally change native selection. Explicit implementation assignments remain authoritative.

### Component expressions and modular arithmetic

Each represented polynomial bundle maps to component expressions evaluated at a batch, prime-row, and polynomial index. Component extraction selects expressions; packing specifies output indexing. These operations need not allocate intermediate tensors. Plaintext broadcasting and input strides are part of the index mapping. Common arithmetic expressions can reuse register values while independently returned arrays retain separate storage.

The generator supports modular addition, subtraction, negation, Montgomery multiplication, plaintext arithmetic, and standard/Montgomery conversion. Ciphertext multiplication lowers to the component expressions

$$
(a_0b_0,\;a_0b_1+a_1b_0,\;a_1b_1),
$$

where products use Montgomery reduction in each prime row. Later compatible CKKS operations continue the same expression graph rather than forcing these three components through a separate intermediate tensor.

For radix $R$ and $k=-q^{-1}\bmod R$, reduction forms

$$
m=(abk)\bmod R,\qquad u=(ab+mq)/R.
$$

The int32 implementation uses $R=2^{32}$ and $q<2^{30}$; int64 uses $R=2^{62}$ and $q<2^{60}$. Integer high/low products and carries implement the wide arithmetic. The result preserves the native operation's canonical or lazy representative, rather than merely agreeing modulo $q$. Lazy additions and subsequent canonicalization retain their defined order.

Key-switch digit products use the same multiplication expressions with a live evaluation-key pair. Key data arrives as Tensor operands; generated-code caches contain layout and expression information, not key contents.

### Compact radix-2 execution

A transform is an anchor in the expression graph. Its input stage evaluates producer expressions at the butterfly's input indices. Its output stage evaluates consumers using the transformed values and any side inputs. Compact radix-2 middle stages remain native synchronized kernels. A region can contain successive transforms, with storage materialized where a transform dependency requires it.

Forward standard-input conversion is performed during input loading. Inverse output normalization and standard-residue conversion occur in the final stage in the same mathematical order as the native transform. The selected NTT schedule must use a compact radix-2 policy. Indexed and fixed-radix plans are not silently converted to that policy.

The compact implementation prepares a direct Python execution function with fixed native calls, generated endpoint calls and temporary lifetimes. Results and scratch arrays are allocated for each execution.

Generated kernels obtain a compiled-kernel launcher on their first use. Warm execution supplies current Tensor pointers and the current stream to that launcher rather than repeating Triton's generic argument specialization. Pointer-alignment specialization is disabled for these generated signatures, allowing valid offset views to retain the same prepared layout. Launch hooks remain active, and changed debug or instrumentation settings trigger preparation under those settings.

Basis extension, rescale/ModDown rounding, communication, and unsupported operations delimit regions. This generator does not rewrite those algorithms or move their rounding steps. Existing native key-switch and rescale fused paths remain independently selectable implementations.

## Reuse and intermediate lifetimes

`ReuseIntermediatesPass` combines identical Tensor/resource references and type-cast paths, then shares matching internal NTT or residue-conversion results. Reuse requires the same SSA inputs, represented result state, attributes, and resources. It does not merge independently returned arrays or cancel arbitrary transform/inverse pairs.

`EliminateDeadValuesPass` uses registered operation effects, including low-level arithmetic and pure fusion bodies. Random draws, unknown operations, and other effects remain liveness roots. Removing an IR cast reduces execution plumbing; eliminating a dead or repeated NTT removes actual transform work.

`RotationHoistingPass` groups coefficient-domain rotations by shared input and parameter Tensor operands across interleaved pure consumers. It places each multi-result group after its operands are available and before its results are used. Unknown effects, control flow, and assigned implementations stop cross-operation grouping; supported nested regions are processed independently. The pass preserves key operands and caller assignments, and does not estimate kernel costs. Numerical rescale/relinearization placement remains caller-selected and is not inserted into captured Eager schedules.

## Preparation and measurement

Backend operation preparation consumes a region once to construct its execution plan. Known layouts can be prepared from physical IR facts; missing layout facts may defer that step until execution. Device-code compilation can remain lazy after host linking. Prepare and warm the intended workload before timing or CUDA Graph capture.

For integer fusion, compare the specified residue representatives as well as agreement modulo each prime. Whole and lowered algorithms can legitimately choose different lazy representatives when their contracts permit it. Check component indexing, broadcast and offset views, externally used intermediates, and rounding transitions using the owning numerical contract. Measure cold preparation, warmed execution, and launch counts separately to evaluate the effect of a lowering.

[Execution buffers and CUDA Graphs](execution-buffers-and-cuda-graphs.md) describes stable-address capture of native and generated kernels together. The [Compile tutorial](../tutorial/compile-jit.md) provides a runnable callable example, and [manual Program execution](../how-to/build-program-pipeline.md) uses the same Pipeline and Backend mechanisms directly.
