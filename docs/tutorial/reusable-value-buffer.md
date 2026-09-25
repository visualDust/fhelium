# Double-buffered execution

**Example source:** [`examples/17_runtime_double_buffer.py`](https://github.com/VisualDust/fhelium/blob/main/examples/17_runtime_double_buffer.py)

ReusableValueBuffer owns fixed storage for an ordinary Tensor/value tree. Example 17 alternates two CUDA buffers while streaming operation-ready plaintext tiles from pinned host memory. It demonstrates data-transfer ordering, not a performance comparison or an automatic memory-admission policy.

```bash
python examples/17_runtime_double_buffer.py --device cuda:0
python examples/17_runtime_double_buffer.py --num-tiles 4 --plaintexts-per-tile 4 --message-size 32
```

The example keeps its established depth-20, logN-16 CKKS configuration and absolute error threshold. The default uses four tiles of four plaintexts, rather than a multi-GiB all-resident benchmark. Two device tiles occupy about 60 MiB; evaluator resources and temporaries are additional allocations.

## Prepare the data and storage

Each tile has a different scalar weight sum, at most 0.125. Distinct tile data makes a missed or incorrectly ordered transfer observable. Host plaintexts are independently pinned, and `ReusableValueBuffer.like` initializes two independent CUDA storage trees from the first tile. The example retains each buffer's ordinary `value` view and records its Tensor addresses.

## Order transfer and computation

At iteration i:

1. Enqueue tile i+1 into the other buffer on the transfer stream.
2. Its copy waits for the event marking that buffer's previous reader.
3. The compute stream waits on the current tile's CopyHandle.
4. Evaluate the current tile using its retained value view.
5. Record a read-completion event before that buffer can be overwritten.

`CopyHandle.wait_on(stream)` orders device work without a host synchronization at each iteration. The application retains the host source values while copies are in flight. Streams are synchronized before releasing buffers.

## Check the result

The example validates every tile, checks that target addresses did not change, and prints host-weight and fixed-buffer byte counts. It does not include a second all-resident execution mode, allocator flushing, or timing statistics.

[Example 18](cuda-graph-matvec.md) uses CUDA Graph capture/replay instead of manually scheduling a streaming workload. [Example 19](explicit-residency.md) adds managed placement and leases, which are separate from the fixed storage and copy handles demonstrated here.

::: details Source
<<< @/../examples/17_runtime_double_buffer.py
:::
