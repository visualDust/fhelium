# Choose and switch a local execution device

Select a CPU or CUDA device when constructing `CkksEngine`. The same public
CKKS methods and backend-neutral native schemas are used on both devices;
PyTorch dispatch selects an implementation from the tensor device.

## Check the installed native backends

Inspect the immutable native status before selecting a device dynamically:

```python
from fhelium.native import native_backend_available, native_status

status = native_status()
print(status.backends)

if native_backend_available("cuda"):
    device = "cuda:0"
elif native_backend_available("cpu"):
    device = "cpu"
else:
    raise RuntimeError(status.reason)
```

Backend inclusion is a source-build choice. A visible GPU does not add CUDA to
a CPU-only `_ops` library. Rebuild according to the
[installation guide](../tutorial/installation.md#select-native-backends)
when the required backend is absent.

## Construct an engine on one device

```python
import fhelium as fh

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_levels7_int64)

cpu_engine = fh.CkksEngine(config, device="cpu")
cuda_engine = fh.CkksEngine(config, device="cuda:0")
```

An unindexed CPU device is canonicalized to `cpu`. An unindexed `cuda` device
uses `torch.cuda.current_device()`. Supplying `device=None` selects the current
CUDA device only when CUDA is visible and the installed extension includes the
CUDA backend; otherwise it selects CPU when the extension includes CPU. Prefer
an indexed device in reproducible programs and experiment records.

Engine construction creates device-owned RNS/NTT parameter tensors, prepared
constants, and a CSPRNG for that device. An engine therefore does not have an
in-place `to()` operation.

## Move a stored value

Tensor-backed core values expose direct value movement:

```python
cuda_ciphertext = ciphertext.to("cuda:0")
cpu_ciphertext = cuda_ciphertext.to("cpu")
```

Movement changes tensor placement but does not change the value's context,
level, scale, active prime IDs, polynomial domain, modulus basis, residue
representation, or component count. It also does not move or recreate the
engine, key mappings, prepared plaintexts, resources, Residency handles, JIT
workspace objects, or CUDA Graph state.

Use the moved value only with an engine and required keys on the same device:

```python
cuda_engine = fh.CkksEngine(config, device="cuda:0")
cuda_secret_key = secret_key.to("cuda:0")
cuda_ciphertext = ciphertext.to("cuda:0")

decoded = cuda_engine.decrypt_message(
    cuda_ciphertext,
    secret_key=cuda_secret_key,
    is_real=True,
)
```

Operations reject mixed-device operands rather than performing a hidden
transfer.

## Recreate device-owned evaluator state

For a complete application transition between CPU and CUDA:

1. synchronize unfinished CUDA work before consuming its result on CPU;
2. create a new engine with the same `CkksConfig` and context parameters on the
   destination device;
3. move or regenerate the required key values;
4. move input `Plaintext` and `Ciphertext` values;
5. recreate prepared plaintexts, evaluation-key sets, JIT workspaces, and other
   device-owned runtime objects for the destination engine;
6. rebuild CUDA-only resources such as CUDA Graph programs rather than moving
   them.

Context identity, value state, dtype, device, and complete RNS layout are
validated after movement. A destination engine created from the same
configuration can consume a moved core value when the operation's required keys
and operands have also been moved or regenerated. Engine-owned caches,
installed defaults, resources, buffers, Residency handles, JIT workspaces, and
CUDA Graph state are not part of that value movement.

## Account for backend-specific capabilities

The local CPU and CUDA engines share CKKS arithmetic methods, but not every
optimization exists on both devices:

- CPU uses the indexed radix-2 NTT backend and Torch intra-op parallelism;
- CUDA supports indexed and production compact/fixed-radix NTT policies;
- CUDA Graph capture, CUDA streams/events, multi-GPU collectives, and CUDA
  topology inspection are CUDA-specific;
- local JIT and experimental multiparty arithmetic can execute on CPU or CUDA
  when their selected operation path uses supported engine primitives.

A CPU-only PyTorch distribution may also omit its pinned-host allocator. Use
ordinary pageable CPU values for local CPU evaluation; pinned-host staging is
an accelerator transfer mechanism rather than a requirement for CPU CKKS.

Do not select an unsupported CUDA-tuned NTT policy for a CPU engine. When
switching devices for performance evaluation, preserve the mathematical
configuration, input values, operation schedule, correctness oracle, and
numerical thresholds; then report backend-specific NTT policy and timing
separately.
