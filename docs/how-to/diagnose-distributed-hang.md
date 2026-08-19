# Diagnose a distributed hang

A distributed hang usually means ranks entered incompatible communication
phases, not that NCCL randomly stopped. Localize the first divergent rank and
collective before changing timeouts or algorithms.

## 1. Reproduce with the smallest launch

Use:

- world size two;
- the smallest valid preset/problem;
- one process per GPU;
- deterministic inputs;
- no CUDA Graph, prefetch, or background policy;
- a finite process-group timeout;
- per-rank line-buffered logs.

Confirm the same worker succeeds with world size one.

## 2. Log rank and device identity

At worker start, record:

```text
global rank
world size
local rank
selected CUDA device
process group backend
group membership
context ID
```

Check that global ranks are not confused with group-relative ranks and that
each process selected the intended device before allocating values.

## 3. Number every collective phase

Log immediately before and after each collective:

```text
phase number and name
rank
collective function
src/dst/root
value type and shape
level, prime IDs, polynomial domain, modulus basis
```

The last phase entered by all ranks and the first phase entered by only some
ranks usually identify the control-flow divergence.

## 4. Check collective ordering

All ranks in a process group must execute compatible collectives in the same
order. Common causes include:

- a rank with no local arithmetic skips a collective;
- a local validation error raises before peers enter error exchange;
- one rank returns early after an empty partition;
- root uses gather while peers use reduce;
- one branch broadcasts a key and another branch proceeds to ciphertext data;
- a different loop count causes an extra collective on one rank.

Make empty-work ranks contribute a valid neutral value or participate in the
same transport/control phases.

## 5. Check typed descriptors before payloads

Compare the exact descriptor on every rank:

- concrete value/key type;
- tensor shapes and dtypes;
- context;
- level and scale;
- prime IDs;
- polynomial domain, modulus basis, and residue representation;
- rotation step or key specialization.

If one rank rejects a descriptor locally while peers begin a payload transfer,
the program can hang. Use the typed APIs' group-consistent validation path
rather than open-coding one-sided checks.

## 6. Check CUDA stream and asynchronous failures

An earlier CUDA error may surface at a collective or synchronization call.
Temporarily add synchronization after narrowly defined phases to locate the first
failing kernel, but remove debugging synchronization after finding the cause.

Check:

- device tensors belong to the process's selected GPU;
- no tensor from another local device is passed accidentally;
- producers complete before NCCL reads payloads;
- buffer storage is not overwritten while communication or kernels read it.

## 7. Enable focused diagnostics

Useful environment-level diagnostics include:

```bash
NCCL_DEBUG=INFO
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

Use a finite timeout and preserve each rank's complete log. Avoid enabling so
much tracing that the first semantic divergence becomes invisible in noise.

## 8. Reduce the collective

Replace the hanging phase temporarily with the smallest equivalent:

- broadcast one ordinary tensor;
- broadcast one typed ciphertext;
- gather one independent value;
- reduce one transparent/additive-compatible ciphertext;
- scatter/gather one limb range.

This separates process-group health from descriptor allocation, payload
transport, and CKKS reduction logic.

## 9. Restore mechanisms incrementally

After the eager two-rank path succeeds, restore:

1. full problem size;
2. full key distribution;
3. asynchronous copies/streams;
4. rank-local graph replay;
5. residency and prefetch;
6. target rank count.

Keep phase numbers and timeouts as regression diagnostics.

## Related documentation

- [Rank-local SPMD](../concepts/distributed/spmd-model.md)
- [Communication semantics](../concepts/distributed/communication-semantics.md)
- [Choose a multi-GPU partition](choose-multi-gpu-partition.md)
- [Independent ciphertext tutorial](../tutorial/spmd-independent-ciphertexts.md)
