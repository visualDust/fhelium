# Double-buffered serving

**Example source:** [`examples/16_runtime_double_buffer.py`](https://github.com/VisualDust/fhelium/blob/main/examples/16_runtime_double_buffer.py)

`DoubleBuffer` owns `count` ReusableValueBuffers (two by default) with fixed device storage for an ordinary Tensor/value tree and a transfer stream that fills them in turn. Example 16 serves a queue of requests with a compiled layer: while the layer evaluates one request's operation-ready plaintext tile from one buffer, the next tile is copied from pinned host memory into the other buffer.

```bash
python examples/16_runtime_double_buffer.py --device cuda:0
python examples/16_runtime_double_buffer.py --num-requests 6 --message-size 32
```

The example uses a depth-20, logN-16 CKKS configuration and checks an absolute error threshold. The default serves six requests of four plaintexts each. Two device tiles occupy 42 MiB; evaluator resources and temporaries are additional allocations.

## Prepare the data and the layer

Each request has a different scalar weight sum, at most 0.125, so a missed or incorrectly ordered transfer is observable. Host plaintexts are independently pinned. The compiled `layer` takes the ciphertext and the four plaintexts as arguments and sums their products in the NTT domain before one inverse NTT and rescale. A producer thread places `(index, tile)` requests on a `queue.Queue` and ends the stream with `None`.

## Order transfer and computation

`DoubleBuffer(host_tiles[0], device=device)` allocates both buffers from the first tile's structure. `feed(requests, value=...)` reads the queue and keeps one request in flight ahead of the one being consumed:

1. `put(tile)` calls `ReusableValueBuffer.load`, which records an event on the current stream and starts the copy on the transfer stream after that event. Work already issued on the current stream, including the previous reader of that buffer, therefore completes before the buffer is overwritten.
2. When a request is yielded, the current stream waits for its copy; the host does not synchronize at each iteration.
3. The layer runs on the current stream with the loaded values.

The layer captures one CUDA Graph on its second call and replays it for the following requests; each replay copies its arguments from the current buffer into the Graph's input storage on the device. The application retains the host tiles while copies are in flight, and leaving the `with` block waits for outstanding copies before the buffers are released.

## Consume each result before the next call

A compiled result is stored in the layer's output region. If the caller still references the previous result when the next call starts, that call first copies the result into separate storage. The example decrypts each result and drops it with `del output` before the next request, so no copy is needed. Accumulating results in a list is also correct; each retained result then costs one device copy.

## Check the result

The example validates every request against the clear product, prints host-weight and device-buffer byte counts, and reports the Graph state, replay count, and capture variants.

[Example 17](compiled-cuda-graphs.md) examines CUDA Graph segments in detail, and [Serve compiled requests](../how-to/serve-compiled-requests.md) lists the serving decisions. [Example 18](manual-residency.md) adds managed placement and leases, which are separate from the fixed storage demonstrated here.

::: details Source
<<< @/../examples/16_runtime_double_buffer.py
:::
