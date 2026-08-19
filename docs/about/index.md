---
title: About FHElium
description: FHElium's motivation, characteristics, work in progress, and history.
---

# About FHElium

FHElium is an open-source, research-oriented framework for fully homomorphic encryption (FHE), with a current implementation focused on CKKS. It connects program representation, cryptographic state, execution, runtime resource management, distributed systems, and hardware acceleration so that they can be studied independently or together.

## Why FHElium exists

FHE inference is supported by an increasingly capable full stack. Compiler frontends and intermediate representations translate application operators into legal encrypted programs. Algorithm and compiler research improves packing, approximation, scale management, and circuit planning. Software backends and optimized kernels accelerate the resulting primitives, while runtimes, distributed systems, and specialized hardware provide additional execution capacity. These advances are complementary and together form the modern FHE inference ecosystem.

The resulting system, however, is not the sum of independent local optima. Program representation and packing determine rotations, keys, and data layout. CKKS state determines numerical behavior, storage volume, and operation cost. Execution order changes temporary memory, data movement, and communication. Workload and runtime conditions change which decisions are effective, while hardware characteristics feed back into algorithm, compiler, and scheduling choices. Optimizing only one layer, or isolating every layer behind an opaque interface, can miss opportunities that appear only when these effects are considered together.

FHElium provides a modular research framework in which cross-layer designs can be expressed, transformed, executed, measured, and validated. Users choose their level of control, specialized components connect through replaceable interfaces, and runtime evidence can guide optimization while remaining inspectable.

## What characterizes FHElium

### 1. Across the stack and modular

FHElium provides coordinated layers across the encrypted-execution stack:

```text
express → transform → execute → observe → validate
```

Each layer retains a precise responsibility and a replaceable interface. A researcher can study one component independently or use shared value semantics and evidence to trace its numerical and system-level consequences across the framework.

### 2. Multiple levels of control

The same computation can be approached through high-level tensor programs, program transformation, direct evaluator operations, or runtime orchestration without changing its underlying value semantics. Program representation does not require every execution decision to become static, so runtime information can still influence behavior where it is useful.

### 3. Orthogonal and explainable

FHElium represents computation, cryptographic state, resource ownership, placement, lifetime, communication, and runtime evidence as related but distinct concerns. Each component states what it owns, requires, and changes, allowing local reasoning while preserving an inspectable explanation of cross-layer effects.

## Work in progress

### Multiple execution backends

The current implementation provides native CPU and CUDA execution backends through common value semantics, operator schemas, and validation requirements. Each backend retains platform-specific execution policies: CPU uses PyTorch intra-op parallelism and an indexed radix-2 NTT, while CUDA adds tuned NTT families, CUDA Graph execution, and application-owned multi-GPU composition. Future backends should preserve the shared semantics while exposing their own kernels, memory hierarchies, and optimization opportunities. This would support research across NVIDIA and AMD GPUs, TPUs, and future encrypted-computing hardware without hiding hardware evidence behind a lowest-common-denominator interface. We are seeking donated hardware or sustained remote access to representative datacenter accelerators, to explore different micro-architectures. Interested organizations are invited to contact us.

### Compiler and ecosystem interoperability

FHElium's versioned CKKS operation vocabulary and extensible representation support interoperability with tensor frontends, external compilers, program-analysis tools, transformation pipelines, code generators, and deployment systems. Focused adapters extend this model to runtimes, persistence, profiling, and benchmarking while each subsystem retains its own policy.

### Agentic AI ready

FHElium's CKKS state, structured requirements, diagnostics, and reproducible evidence are intended to give agentic systems feedback they can act on. The goal is for an agent to translate a cleartext program into an encrypted program or tune an encrypted workload's performance while keeping each transformation, assumption, and validation result observable and debuggable.

## History

FHElium emerged from **Slackoffhe**, an internal redesign of [Tiberate-FHE](https://github.com/visualDust/tiberate-fhe). As the intended programming model, runtime responsibilities, and cross-layer research goals became clearer, the redesign expanded into a near-complete reconstruction of the system and was established as **FHElium**, with a new abstraction hierarchy for encrypted execution and resource management.

Tiberate-FHE had itself begun as an independently maintained continuation of the archived [Liberate-FHE](https://github.com/Desilo/liberate-fhe). That earlier connection is development background rather than FHElium's architectural baseline. FHElium does not preserve API or architecture compatibility with Tiberate-FHE or Liberate-FHE and is not a drop-in continuation of either project.

The name places **F** before *helium*, making FHE visible while reflecting the goal of a light, modular, and extensible system. Tiberate-FHE, Slackoffhe, and FHElium are research-oriented projects initially developed by [Gavin Gong](https://github.com/VisualDust) while working under the supervision of [Dr. Wujie Wen](https://wenwujie.github.io/) at NC State University.
