# CUDA Graph matrix-vector multiplication

**Example source:** [`examples/11_cuda_graph_matrix_vector.py`](https://github.com/VisualDust/fhelium/blob/main/examples/11_cuda_graph_matrix_vector.py)

This example captures a fixed packed CKKS matrix-vector evaluator
$\mathbf{y}=A\mathbf{x}$ while keeping each request ciphertext as a dynamic input. The tutorial explains captured state, replay storage, and the
steady-state measured region.

## Run the example

```bash
python examples/11_cuda_graph_matrix_vector.py \
  --device cuda:0 \
  --preset slots8192-scale40-levels7-int64 \
  --size 8 \
  --capture-warmup 3 \
  --benchmark-warmup 10 \
  --runs 100
```

## 1. Identify static and dynamic state

The workload computes `y = A @ x` with cyclic diagonals. Its state divides
into:

| Static across replays | Dynamic for each replay |
| --- | --- |
| engine and CKKS context | encrypted input vector |
| encoded matrix diagonals | ciphertext payload |
| exact rotation keys | request-specific values |
| operation schedule | matching exact-value input signature |

Encryption and decryption remain outside the graph.

## 2. Prepare operation-ready constants

```python
diagonals = [
    engine.prepare_plaintext_for_multiplication(
        engine.encode(
            cyclic_diagonal_slots(matrix, step, engine.num_slots),
            level=0,
        )
    )
    for step in range(matrix.size(0))
]
rotation_keys = {
    step: engine.rotation_key(step)
    for step in range(1, matrix.size(0))
}
```

These values are captured as callable state. Changing their object identity,
storage address, level, or shape after capture would invalidate the captured
schedule.

## 3. Bind static state

```python
from functools import partial

schedule = partial(
    matrix_vector,
    engine=engine,
    diagonals=diagonals,
    rotation_keys=rotation_keys,
)
```

The resulting callable has one dynamic argument: the source ciphertext. This
is preferable to a hidden global cache because the captured resources are visible
through direct Python calls.

## 4. Capture from a prototype

```python
program = CudaGraphProgram.capture(
    schedule,
    example_inputs=(prototype,),
    warmup=capture_warmup,
)
```

[`CudaGraphProgram`](../api/fhelium/execution/cuda_graph.md#cudagraphprogram) performs side-stream
warmup, allocates fixed dynamic-input storage, captures the evaluator, records
the output storage, and derives an exact input signature.

The prototype determines structure, including:

- value-tree shape;
- exact value type;
- tensor shape and dtype;
- CKKS context and level;
- polynomial domain, modulus basis, residue representation, scale, and prime IDs.

## 5. Replay with changing ciphertexts

```python
result = program.replay(
    encrypted,
    synchronize=True,
)
```

Replay validates the new value before staging its payload into the fixed
input allocation. A mismatched level or representation is rejected rather
than silently converted inside the graph wrapper.

The example uses three different encrypted vectors and verifies all three
against `matrix @ vector`.

## 6. Understand borrowed output storage

```python
borrowed_pointer = result.data.data_ptr()
```

The default result references graph-owned output storage. The next replay can
overwrite it. This avoids an extra device copy when the caller consumes the
output immediately.

Use:

```python
owned = program.replay(encrypted, copy_output=True)
```

when a result must survive a later replay or escape into another asynchronous
lifetime.

## 7. Separate construction and steady-state cost

The example reports:

- warmup time;
- graph capture time;
- first replay time;
- eager mean latency;
- graph replay mean latency.

Capture is a one-time program-construction cost. Compare steady-state replay
only when the same static schedule will execute enough times to amortize it.

## Close the program

```python
program.close()
```

Closing releases graph-owned inputs, outputs, and capture state. Do not treat
a captured program as an unbounded global singleton when different contexts,
models, or input signatures require independent storage.

::: danger Application-owned serving policy
A serving extension assigns models and users, admits requests, manages
per-user keys and eviction, and composes those policies around one or more
fixed captured programs.
:::

::: details Complete runnable source
<<< @/../examples/11_cuda_graph_matrix_vector.py
:::

## Related concepts and guides

- [CUDA Graph execution model](../concepts/execution/cuda-graph-model.md)
- [Exact signatures and buffers](../concepts/execution/exact-signatures-and-buffers.md)
- [Capture a repeated evaluator](../how-to/capture-repeated-evaluator.md)
