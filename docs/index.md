---
layout: home
---

<HomeHero />

<HomeOpeningRail />

## Choose your level of control

FHElium provides one hierarchical CKKS programming model with four entry levels: direct evaluator calls, tuned key and arithmetic schedules, mixed-dialect JIT `Program` objects, and GPU-rank partitioning. Every level reaches the same `CkksEngine`, core `Ciphertext` values, and CKKS state invariants.

<HomeControlDeck />

Move between these levels as the workload matures: prototype semantic tensor code, inspect the mixed-dialect program, tune the evaluator schedule, and compose rank-local values with typed collectives; all levels share the same encrypted value model. Continue to the [Quickstart](tutorial/tutorials.md) for direct CKKS evaluation, the [JIT tutorial](tutorial/unified-jit.md) for capture, passes, and readiness, or the [SPMD model](concepts/distributed/spmd-model.md) for rank-local multi-GPU ownership and collective semantics.

## Build through the stack

Start with ordinary Python or typed PyTorch semantics, expose the encrypted graph and its CKKS mechanics, then carry the same evaluator into repeated and distributed execution. FHElium exposes the joints between layers: graph policy, ciphertext state, evaluator keys, resident artifacts, CUDA Graph capture, and rank-local placement remain independently inspectable and selectable.

<HomeStackBuilder />

Use the route builder to compare which decisions belong to authoring, graph policy, execution, and application-owned placement.

## JIT, with the graph in your hands

Write the evaluator beside ordinary Python, trace or import one mixed-dialect xDSL program, transform it with selected local passes, and execute only after an independent readiness check. The `Program` remains inspectable while its live materials, engines, keys, handlers, and caches remain in a retained workspace.

<HomeGraphXray />

The resulting `Program` can be printed as textual IR, passed through another selected pipeline, or executed eagerly with bound runtime capabilities. PyTorch capture, textual import, custom passes, and backend handlers all use the same program class.

## Measured performance

The same conventional cyclic-diagonal BSGS formulation measures both packed
plaintext-matrix × ciphertext-vector (**PT×CT**) and ciphertext-matrix ×
ciphertext-vector (**CT×CT**) evaluation. <a href="/assets/fhelium-workload.py" download="fhelium-workload.py">View source</a>

<BsgsMatvecPerformance />

## Continue by task

<DocGrid>
  <DocCard
    title="Run a first computation"
    description="Install against a selected PyTorch stack and execute the complete CPU or CUDA quickstart."
    href="/tutorial/"
  />
  <DocCard
    title="Build a correct evaluator"
    description="Plan levels, scales, domains, key material, and CKKS state transitions."
    href="/concepts/"
  />
  <DocCard
    title="Diagnose or optimize a workload"
    description="Apply focused procedures for correctness, memory, execution, distribution, and performance."
    href="/how-to/"
  />
  <DocCard
    title="Work on FHElium internals"
    description="Follow calls across the Python engine, dispatcher schemas, C++, and native CUDA operators."
    href="/developer/"
  />
  <DocCard
    title="Look up an API"
    description="Browse signatures and API documentation generated directly from the current Python source."
    href="/api/"
  />
</DocGrid>
