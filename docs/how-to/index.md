# How-to guides

How-to guides solve calculation, deployment, execution, and diagnosis tasks using FHElium’s current interfaces. Choose the starting representation that fits the application: [Eager operations](evaluate-ckks-data.md) execute CKKS steps immediately; [a constructed or captured Program](build-program-pipeline.md) supports caller-composed transformation and linking; [a compiled callable](compile-callable.md) adds specialization and a function interface over a calculation. The two Compile entry paths share Programs, passes, materials, and Backend execution.

Each guide states the required setup, the procedure, and the evidence that establishes its outcome. For responsibilities and implementation mechanisms, use the [Developer architecture overview](../developer/engine-native-stack.md).

## Describe and execute calculations

<DocGrid>
  <DocCard
    title="Encode, encrypt, and evaluate CKKS data"
    description="Prepare keys and packed values, express arithmetic transitions, and measure decoded error."
    href="/how-to/evaluate-ckks-data"
  />
  <DocCard
    title="Build and transform a Program"
    description="Describe IR directly or capture Python, compose passes, and choose an execution product."
    href="/how-to/build-program-pipeline"
  />
  <DocCard
    title="Compile and reuse a callable"
    description="Specialize a function or Program and control preparation, pipelines, and Backend binding."
    href="/how-to/compile-callable"
  />
  <DocCard
    title="Write and insert a Compilation pass"
    description="Collect analysis or implement a numerical rewrite using the current Compilation."
    href="/how-to/write-compilation-pass"
  />
  <DocCard
    title="Visualize mixed-level IR"
    description="Render selected SSA dataflow and compare transformation snapshots."
    href="/how-to/visualize-mixed-level-ir"
  />
</DocGrid>

## Parameters and implementations

<DocGrid>
  <DocCard
    title="Choose a preset and chain depth"
    description="Plan physical slots, rescale groups, actual scales, magnitude range, and precision."
    href="/how-to/choose-preset-and-depth"
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
    description="Derive the evaluation-key inventory and preserve parameter identity and key custody."
    href="/how-to/provision-keyset"
  />
  <DocCard
    title="Bind, save, load, and deploy a Program"
    description="Supply numerical materials, persist selected payloads, and link on the destination."
    href="/how-to/persist-compiled-program"
  />
  <DocCard
    title="Manage artifacts by logical name"
    description="Store values under checked immutable generations with defined local durability."
    href="/how-to/manage-artifacts"
  />
  <DocCard
    title="Choose and switch a local execution device"
    description="Place factories, values, keys, and compiled execution resources on CPU or CUDA."
    href="/how-to/switch-cpu-cuda"
  />
</DocGrid>

## Repeated execution and memory

<DocGrid>
  <DocCard
    title="Capture a repeated evaluator"
    description="Capture a fixed CUDA schedule, stage changing inputs, and manage borrowed output."
    href="/how-to/capture-repeated-evaluator"
  />
  <DocCard
    title="Choose a homogeneous batch size"
    description="Compare batch throughput, latency, and working sets across active depths."
    href="/how-to/choose-homogeneous-batch-size"
  />
  <DocCard
    title="Choose a Residency control level"
    description="Select direct movement, manager primitives, plans, or admission policy."
    href="/how-to/choose-residency-control-level"
  />
  <DocCard
    title="Stream resources with bounded CUDA memory"
    description="Choose fixed buffers and managed residency windows with valid copy and use lifetimes."
    href="/how-to/stream-bounded-memory"
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
    title="Diagnose Compile preparation failures"
    description="Locate missing facts, materials, implementations, or live execution resources."
    href="/how-to/diagnose-compile-preparation"
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
    title="Screen NTT backends on the target GPU"
    description="Run the retained diagnostic recommender and confirm candidates on the current evaluator."
    href="/how-to/screen-ntt-backends"
  />
  <DocCard
    title="Interpret NTT backend performance"
    description="Explain kernel-versus-primitive differences through launch, traffic, and resource costs."
    href="/how-to/choose-ntt-backend"
  />
  <DocCard
    title="Optimize a workload systematically"
    description="Measure a correct execution path and isolate changes without changing its error criterion."
    href="/how-to/optimize-workload"
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
