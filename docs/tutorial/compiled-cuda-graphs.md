# Compiled CUDA Graphs

**Example source:** [`examples/17_runtime_cuda_graph.py`](https://github.com/VisualDust/fhelium/blob/main/examples/17_runtime_cuda_graph.py)

A compiled callable on CUDA executes its specialization's exec schedule through a Runner, and with Graphs enabled the Runner captures the schedule's capture-safe segments as CUDA Graphs and replays them. Example 17 compiles two functions that form a pipeline, a packed matrix-vector product $y=Ax$ and a square activation $y^2$, and shows the three `graph` settings, segment statistics, in-place reads between compiled functions, and the storage of results the caller keeps.

## Run the example

```bash
python examples/17_runtime_cuda_graph.py
python examples/17_runtime_cuda_graph.py --size 16 --runs 100
```

The example requires CUDA and uses `slots8192-scale40-depth7`. It prints the latency and segment choices of each `graph` setting, then the pipeline's error, Graph variants, staged copies, and compiled execution memory, and finally the result-ownership counters.

## The two functions

```python
def matvec(x):
    rotated = Ciphertext.stack_batch([x, *engine.rotate_many_by_steps(x, steps)])
    products = engine.sum_plaintext_products(
        engine.coefficient_domain_to_ntt_domain(rotated), diagonals)
    return engine.rescale_to_next_depth(engine.ntt_domain_to_coefficient_domain(products))

def square(y):
    ntt = engine.coefficient_domain_to_ntt_domain(y)
    return engine.relinearize(engine.multiply(ntt, ntt), relinearization_key, rescale=True)
```

`matvec` evaluates $\sum_{k} d_{-k}\odot\operatorname{roll}(x,-k)$ over an 8-slot period with hoisted rotations and one fused plaintext-product sum; `square` multiplies, relinearizes, and rescales. The pipeline moves the input from depth 0 to depth 2.

## Graph settings

```python
for graph in ("auto", True, False):
    first = fc.compile(matvec, graph=graph)
    second = fc.compile(square, graph=graph)
    second(first(x))
```

The first execution of a specialization runs its schedule directly, the second captures each Graph segment, and later executions replay it. A segment is a maximal run of capture-safe instructions at the top level of the schedule; kernel launches, copies, views, and operator calls on one CUDA device are capture-safe, while Python calls and CPU operations end a segment. With `"auto"`, each segment compares the device time of its first execution with the median of three replays after capture and keeps its Graph only when replay is faster; `True` keeps every Graph, and `False` runs every call directly. Each record in `compiled.stats()["graphs"]` lists its segments with `choice` equal to `"graph"`, `"direct"`, or `"measuring"` before the decision. For this small pipeline both segments keep their Graphs under `"auto"`, and replay reduces latency relative to `graph=False`.

## Reading a predecessor's output in place

Compiled functions on one device share one arena for their temporaries and the Graph pool of their Runners. The output of `first` lies in its Runner's output region, and `second` reads it there: a segment that reads another Runner's storage keeps up to four captured variants, one for each set of input addresses, instead of copying its inputs into its own storage. The example reports one captured variant and no staged input copies for the activation. `fhelium.backend.runner.compiled_execution_bytes(device)` reports the device memory that the shared arena and the Graph pool hold.

## Results the caller keeps

A Graph result is returned in the Runner's output region without a copy. When the caller still references a result and the next execution would overwrite that region, the Runner first moves the result to separate storage, so every result stays valid while the caller holds it. The example keeps three results in a list and finds three distinct storages, with the moves counted as `detached_outputs` in the Graph statistics; when the loop instead drops each result with `del` before the next call, no further moves occur apart from the one result still referenced when the loop starts. Keeping results is correct and costs one device copy per kept result.

## Clear the callables

`clear()` releases the specializations, their Graphs, and their Runners. Call it after the last result has been consumed.

[Use CUDA Graphs](../how-to/use-cuda-graphs.md) covers the `graph` option for an existing function and the manual `CudaGraphProgram` capture of Eager code. [Compiled execution](../concepts/execution/compiled-execution.md) defines schedules, storage classes, and segments.

::: details Source
<<< @/../examples/17_runtime_cuda_graph.py
:::
