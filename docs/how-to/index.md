# How-to guides

How-to guides solve calculation, deployment, execution, and diagnosis tasks using FHElium’s current interfaces. Choose the starting representation that fits the application: [Eager operations](../tutorial/basic-ckks-workflow.md) execute CKKS steps immediately; [a constructed or captured Program](build-program-pipeline.md) supports caller-composed transformation and linking; [a compiled callable](compile-callable.md) adds specialization, CUDA Graph execution, and export over a calculation. The two Compile entry paths share Programs, passes, materials, and Backend execution.

Each guide states the required setup, the procedure, and the evidence that establishes its outcome. For responsibilities and implementation mechanisms, use the [Developer architecture overview](../developer/engine-native-stack.md).

## Describe and execute calculations

<DocGrid>
  <DocCard
    title="Encode, encrypt, and evaluate CKKS data"
    description="Follow the basic Eager workflow: keys, packed values, arithmetic transitions, and decoded error."
    href="/tutorial/basic-ckks-workflow"
  />
  <DocCard
    title="Build a Program pipeline and write passes"
    description="Describe IR directly or capture Python, compose passes, and write analysis or rewrite passes."
    href="/how-to/build-program-pipeline"
  />
  <DocCard
    title="Compile a callable"
    description="Set the options of fc.compile, prepare specializations, and read compile statistics."
    href="/how-to/compile-callable"
  />
  <DocCard
    title="Choose compile boundaries"
    description="Decide what a compiled function contains: inlining, loops, scalar specialization, methods, Graph segments, and cache-friendly source."
    href="/how-to/choose-compile-boundaries"
  />
  <DocCard
    title="Inspect Program transformations"
    description="Render selected SSA dataflow, compare transformation snapshots, and emit Eager Python."
    href="/how-to/visualize-mixed-level-ir"
  />
</DocGrid>

## Parameters and implementations

<DocGrid>
  <DocCard
    title="Choose a preset and chain depth"
    description="Select a built-in or bootstrap preset, plan rescale transitions, actual scales, magnitude range, and precision."
    href="/how-to/choose-preset-and-depth"
  />
  <DocCard
    title="Choose an NTT implementation"
    description="Measure compatible NTT backends on the target GPU and explain the ranking through launch, traffic, and resource costs."
    href="/how-to/choose-ntt-implementation"
  />
  <DocCard
    title="Choose operation implementations and lowerings"
    description="Retain whole operations or expose RNS/NTT work and inspect Backend choices."
    href="/how-to/select-operation-implementation"
  />
</DocGrid>

## Materials and deployment

<DocGrid>
  <DocCard
    title="Provision the minimum required keyset"
    description="Derive the evaluation-key inventory and bind keys to Eager, compiled, and loaded execution."
    href="/how-to/provision-keyset"
  />
  <DocCard
    title="Save and relink a Program"
    description="Persist a Program with selected materials, supply missing bindings, and link it in another process."
    href="/how-to/save-and-relink-program"
  />
  <DocCard
    title="Deploy a compiled callable"
    description="Export linked schedules, kernel images, and tables to one artifact and load it in Python or C++ with caller keys."
    href="/how-to/deploy-compiled-callable"
  />
  <DocCard
    title="Store named values"
    description="Store values under logical names with checked immutable generations and defined local durability."
    href="/how-to/manage-value-store"
  />
  <DocCard
    title="Choose and switch a local execution device"
    description="Place factories, values, and keys on CPU or CUDA; compiled execution follows the placement of its operands."
    href="/how-to/switch-cpu-cuda"
  />
  <DocCard
    title="Call the native library from C++"
    description="Link libfhelium through its CMake package and call CKKS, RNS, and NTT Tensor functions."
    href="/how-to/call-native-library-from-cpp"
  />
</DocGrid>

## Repeated execution and memory

<DocGrid>
  <DocCard
    title="Use CUDA Graphs"
    description="Read Graph segment choices of compiled callables and capture Eager code manually with CudaGraphProgram."
    href="/how-to/use-cuda-graphs"
  />
  <DocCard
    title="Choose a homogeneous batch size"
    description="Compare batch throughput, latency, and working sets across active depths."
    href="/how-to/choose-homogeneous-batch-size"
  />
  <DocCard
    title="Serve compiled requests"
    description="Prepare admitted signatures at startup, feed requests through buffers, and keep results and Residency decisions outside compiled calls."
    href="/how-to/serve-compiled-requests"
  />
  <DocCard
    title="Manage CUDA memory with buffers and Residency"
    description="Stream tiles through reusable buffers and place managed values between compiled calls within a device budget."
    href="/how-to/manage-cuda-memory"
  />
</DocGrid>

## Distributed execution

<DocGrid>
  <DocCard
    title="Choose a multi-GPU partition"
    description="Partition independent inputs, additive terms, or compatible RNS-row computation."
    href="/how-to/choose-multi-gpu-partition"
  />
</DocGrid>

## Diagnosis and performance

<DocGrid>
  <DocCard
    title="Diagnose Compile failures"
    description="Resolve capture errors, specialization misses, export and load errors, and cache fallbacks."
    href="/how-to/diagnose-compile-failures"
  />
  <DocCard
    title="Diagnose a value-state mismatch"
    description="Reconcile configuration, key relation, depth, scale, representation, and placement."
    href="/how-to/diagnose-value-state-mismatch"
  />
  <DocCard
    title="Diagnose a Residency failure"
    description="Inspect endpoint availability, budgets, protections, stale decisions, and partial execution."
    href="/how-to/diagnose-residency-failure"
  />
  <DocCard
    title="Diagnose a distributed hang"
    description="Find divergent collective order, rank membership, placement, and local failures."
    href="/how-to/diagnose-distributed-hang"
  />
  <DocCard
    title="Inspect runtime, memory, and CUDA topology"
    description="Record installed backend capability and process-visible memory and topology."
    href="/how-to/inspect-runtime-and-cuda"
  />
  <DocCard
    title="Measure and optimize a workload"
    description="Separate first-call and steady-state latency, read compile and segment statistics, and account for device memory."
    href="/how-to/measure-and-optimize"
  />
</DocGrid>

For the complete benchmark suite and report submission format, see [Run and submit a benchmark](/benchmarks/run-and-submit).

## Experimental features

<DocGrid>
  <DocCard
    title="Compose a bootstrap callable"
    description="Assemble full-slot transforms and periodic reduction with defined coordinates."
    href="/how-to/compose-bootstrap-circuit"
  />
  <DocCard
    title="Implement a bootstrap component"
    description="Supply approximation, polynomial, transform, or periodic-reduction strategies."
    href="/how-to/implement-bootstrap-component"
  />
  <DocCard
    title="Use multiparty CKKS"
    description="Exercise collective arithmetic on synthetic data under the documented security limitations."
    href="/how-to/use-multiparty-ckks"
  />
</DocGrid>
