# How-to guides

How-to guides provide procedures for application, deployment, diagnosis, and performance tasks using FHElium's public interfaces.

## CKKS

<DocGrid>
  <DocCard
    title="Choose a preset and chain depth"
    description="Plan slots, depth, scale, range, and validation before building an evaluator."
    href="/how-to/choose-preset-and-depth"
  />
  <DocCard
    title="Provision the minimum required keyset"
    description="Derive keys from the operation schedule and place only what each worker needs."
    href="/how-to/provision-keyset"
  />
  <DocCard
    title="Diagnose a value-state mismatch"
    description="Compare stored value/key state, physical placement, and external key relations in order."
    href="/how-to/diagnose-value-state-mismatch"
  />
</DocGrid>

## Distributed execution

<DocGrid>
  <DocCard
    title="Choose a multi-GPU partition"
    description="Decide between independent data, additive-term, and RNS-limb parallelism."
    href="/how-to/choose-multi-gpu-partition"
  />
  <DocCard
    title="Diagnose a distributed hang"
    description="Localize collective-order, rank, validation, and device mismatches."
    href="/how-to/diagnose-distributed-hang"
  />
</DocGrid>

## Experimental public interfaces

<DocGrid>
  <DocCard
    title="Compose a bootstrap callable"
    description="Select built-in approximation, polynomial evaluation, transform, and periodic-reduction components."
    href="/how-to/compose-bootstrap-circuit"
  />
  <DocCard
    title="Implement a CKKS bootstrap component"
    description="Implement an approximation, polynomial DAG, linear compiler/evaluator, or periodic reduction as a direct Python object."
    href="/how-to/implement-bootstrap-component"
  />
  <DocCard
    title="Use multiparty CKKS"
    description="Run the application-owned state machine for collective key material and synthetic secret-dependent output operations."
    href="/how-to/use-multiparty-ckks"
  />
  <DocCard
    title="Visualize and inspect a JIT Program"
    description="Render selected SSA, type, attribute, obligation, and user evidence for pattern analysis and pass comparison."
    href="/how-to/visualize-jit-program"
  />
</DocGrid>

## Execution and lifecycle

<DocGrid>
  <DocCard
    title="Choose and switch a local execution device"
    description="Select CPU or CUDA at engine construction, move values with .to(...), and recreate device-owned runtime state safely."
    href="/how-to/switch-cpu-cuda"
  />
  <DocCard
    title="Manage exact artifacts by logical name"
    description="Publish, replace, validate, and consume checked ArtifactStore generations with local durability guarantees."
    href="/how-to/manage-exact-artifacts"
  />
  <DocCard
    title="Capture a repeated evaluator"
    description="Separate static and dynamic state, then capture one rank-local schedule."
    href="/how-to/capture-repeated-evaluator"
  />
  <DocCard
    title="Choose a Residency control level"
    description="Choose functional movement, strict manager primitives, a manual plan, or deterministic automatic admission."
    href="/how-to/choose-residency-control-level"
  />
  <DocCard
    title="Stream resources with bounded CUDA memory"
    description="Choose reusable buffers or residency windows and define their lifetimes."
    href="/how-to/stream-bounded-memory"
  />
  <DocCard
    title="Diagnose a Residency failure"
    description="Reconcile endpoints, budgets, protections, decision evidence, stale state, and committed partial execution."
    href="/how-to/diagnose-residency-failure"
  />
</DocGrid>

## Performance

<DocGrid>
  <DocCard
    title="Inspect runtime and CUDA topology"
    description="Record the package/runtime environment and inspect devices and peer access before multi-GPU work."
    href="/how-to/inspect-runtime-and-cuda"
  />
  <DocCard
    title="Screen NTT backends"
    description="Run the focused recommendation command after a correct representative workload exists."
    href="/how-to/screen-ntt-backends"
  />
  <DocCard
    title="Analyze and choose an NTT backend"
    description="Explain kernel-versus-primitive results through launch, traffic, occupancy, RNS, and CKKS composition costs."
    href="/how-to/choose-ntt-backend"
  />
  <DocCard
    title="Benchmark a workload correctly"
    description="Define the timed work, synchronize CUDA, and retain a correctness oracle."
    href="/how-to/benchmark-a-workload"
  />
  <DocCard
    title="Choose a homogeneous batch size"
    description="Compare unbatched, B1, and larger batches across active levels, latency, and peak memory."
    href="/how-to/choose-homogeneous-batch-size"
  />
  <DocCard
    title="Optimize a workload systematically"
    description="Profile the evaluator, isolate the dominant cost, and validate each change."
    href="/how-to/optimize-workload"
  />
</DocGrid>
