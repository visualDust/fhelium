---
layout: home
---

<HomeHero />

<InstallCommand />

## Choose your level of control

Build through the full FHE stack while choosing the level of control that fits each workload. Eager execution and Compile Programs use the same Backend, arithmetic resources, and native CPU/CUDA implementations.

<HomeControlExplorer />

## Usage models

Choose a Program workflow, immediate execution with runtime mechanisms, or a
rank-local distributed program.

<HomeUsageTabs />

## Measured performance

The same conventional cyclic-diagonal BSGS formulation measures both packed
plaintext-matrix × ciphertext-vector (**PT×CT**) and ciphertext-matrix ×
ciphertext-vector (**CT×CT**) evaluation. <a href="/assets/fhelium-workload.py" download="fhelium-workload.py">View source</a>

<BsgsMatvecPerformance />

## Continue by task

<DocGrid class="home-task-grid">
  <DocCard
    title="Run a first computation"
    description="Install FHElium and execute an encrypted computation on CPU or CUDA."
    href="/tutorial/"
  />
  <DocCard
    title="Understand the architecture"
    description="Follow values, CKKS state, Programs, Backend implementations, runtime mechanisms, and native execution."
    href="/concepts/"
  />
  <DocCard
    title="Solve a workload problem"
    description="Use focused procedures for parameters, compilation, placement, memory, distribution, and performance."
    href="/how-to/"
  />
  <DocCard
    title="Develop FHElium"
    description="Trace Eager and Compile through Backend resources, PyTorch schemas, and native CPU/CUDA operators."
    href="/developer/"
  />
  <DocCard
    title="Browse the API"
    description="Look up current public modules, classes, functions, and signatures generated from source."
    href="/api/"
  />
</DocGrid>
