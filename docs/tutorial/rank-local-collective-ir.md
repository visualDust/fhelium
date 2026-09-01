# Transform rank-local collective IR

**Example source:** [`examples/21_rank_local_collective_ir.py`](https://github.com/VisualDust/fhelium/blob/main/examples/21_rank_local_collective_ir.py)

Example 21 constructs a rank-local Program containing collective operations and applies a caller-selected Compile pass to expose a generic reduction region. It compares two valid representations of ciphertext-add all-reduce without launching distributed execution.

## Run the example

```bash
python examples/21_rank_local_collective_ir.py
```

The script requires neither a process group nor multiple devices because it only builds and transforms IR.

## Rank-local Program

The function receives:

- one rank-local CKKS ciphertext;
- one launch-bound `fhelium_dist.group` value.

It reads rank and group size, broadcasts the ciphertext from rank zero, and applies `fhelium_dist.all_reduce_add_ciphertext`.

Every operation describes the work observed by one rank. A later SPMD launch supplies device count, group rank, process-group lifecycle, and transport resources.

## Specialized collective form

`fhelium_dist.all_reduce_add_ciphertext` states that the local ciphertext values are combined by ciphertext addition. A Backend with a whole-operation implementation may consume this representation directly.

The specialized operation provides one inspectable reduction form available to passes and execution owners.

## Generic combine-region form

`LowerSpecializedCollectivesPass` can replace the specialized operation with:

```text
fhelium_dist.all_reduce ... ({
  ^bb0(%left: !fhelium_ckks.ciphertext,
       %right: !fhelium_ckks.ciphertext):
    %sum = fhelium_ckks.add %left, %right
    fhelium_dist.yield %sum
})
```

The generic operation exposes the local binary combine computation as a region, allowing later passes to inspect or transform the CKKS addition.

The example compiles the same source Program twice:

```python
LowerSpecializedCollectivesPass(lower_ciphertext_add=False)
LowerSpecializedCollectivesPass(lower_ciphertext_add=True)
```

The first request records a `preserve` decision. The second records a `generic-combine-region` decision and rewrites the IR. Both choices appear in pass reports.

## Protocol checks beyond Program structure

Structural construction and transformation establish local IR validity.
Distributed execution additionally requires checks for:

- that all ranks enter the same collective;
- that collectives occur in the same cross-rank order;
- that region control flow is rank-uniform;
- that the combine operation is associative under the selected arithmetic;
- that execution is free of deadlock;
- that process-group resources match the Program.

A caller may compose diagnostic passes for a particular protocol and run them
before linking or execution.

## Why preserve both forms

The two forms serve different transformation and execution choices:

- the specialized operation gives a provider one whole-operation unit;
- the generic operation exposes the local combine arithmetic;
- a distributed pass can choose according to a concrete workload and resource plan;
- the selected representation remains visible in Program text and pass decisions.

Neither choice becomes a permanent framework-wide lowering policy.

::: details Source

<<< @/../examples/21_rank_local_collective_ir.py

:::

## Next steps

- [SPMD execution model](../concepts/distributed/spmd-model.md) defines rank-local ownership and launch responsibilities.
- [Communication semantics](../concepts/distributed/communication-semantics.md) distinguishes typed ciphertext reduction from residue-row reconstruction.
- [Neutral IR Programs](../concepts/neutral-ir-programs.md) explains regions and permissive mixed-level structure.
