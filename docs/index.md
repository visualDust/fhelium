---
layout: home
---

<HomeHero />

<InstallCommand />

## Use FHElium with agents

The `fhelium` agent skill gives coding agents the FHElium workflow for writing, compiling, measuring, and deploying CKKS programs. Install it into a project for Claude Code, Codex, Cursor, and other agents that read Agent Skills:

```bash
npx skills add visualDust/fhelium --skill fhelium
```

## Execution routes

Eager runs each operation immediately; the other routes run one compiled schedule.

<HomeControlExplorer />

## Usage models

The Compile, Deploy, and Serve snippets reuse the Engine, keys, and ciphertext from the Eager snippet.

<HomeUsageTabs />

## Just-in-time optimization

`fhelium.compile` captures the compute flow and optimizes it at runtime: one encrypted matrix-vector product takes 3.49 ms instead of 5.20 ms in eager execution.

<HomeCompileTimeline />

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
