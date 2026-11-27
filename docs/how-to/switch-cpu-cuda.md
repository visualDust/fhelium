# Choose and switch a local execution device

Execution placement determines where numerical operands, keys, and device-local resources live. Eager factories accept a destination device, arithmetic follows placed Tensor operands, and compiled execution consumes the data and resources prepared for its selected implementations.

## Prerequisites

Have a working CPU calculation or saved values and an installed native backend for the target device. Retain configuration provenance and key relations while moving data. An Eager Engine owns its CKKS context and constructs device resources lazily; factory defaults follow PyTorch rather than an Engine-owned default device.

## Check the installed native backends

Inspect native status before choosing a device:

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

Backend inclusion is a source-build choice. A visible GPU does not add CUDA to a CPU-only native library. Rebuild according to the [installation guide](../tutorial/installation.md#select-native-backends) when the required backend is absent.

## Select placement for created values

```python
import fhelium as fh
import torch
from fhelium.eager import Engine

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7)
engine = Engine(config)

torch.set_default_device("cuda:0")
plaintext = engine.encode(message)
secret_key = engine.create_secret_key()
```

Source and material factories use `torch.get_default_device()` when `device` is omitted. A caller may instead select placement on each call:

```python
plaintext = engine.encode(message, device="cuda:0")
secret_key = engine.create_secret_key(device="cuda:0")
public_key = engine.create_public_key(secret_key)
ciphertext = engine.encrypt(plaintext, public_key)
```

The first operation on a device constructs and caches that device's RNS/NTT tables, random stream, operation bindings, and direct-dispatch state. Later calls reuse them. Random streams on different devices use separate nonce domains.

## Dispatch ordinary operations from values

Homomorphic operations dispatch from Tensor operand placement:

```python
cuda_sum = engine.add(cuda_left, cuda_right)
```

They do not consult PyTorch's default device and do not move operands. Invalid mixed-device calls fail in Torch or native execution rather than triggering an Engine copy:

```python
engine.add(cpu_ciphertext, cuda_ciphertext)
```

## Request a boundary transfer

Tensor-backed values expose direct movement:

```python
cuda_ciphertext = ciphertext.to("cuda:0")
cpu_ciphertext = cuda_ciphertext.to("cpu")
```

A boundary operation may also receive `device`. The public adapter moves its plaintext or ciphertext to that device before execution:

```python
cpu_slots = engine.decode(
    engine.decrypt(cuda_ciphertext, cpu_secret_key, device="cpu"),
    device="cpu",
)
```

Without `device`, encrypt, decrypt, and decode inherit placement from their materialized plaintext or ciphertext. Movement preserves depth, scale, prime IDs, polynomial domain, modulus basis, residue representation, and component count.

Key placement is governed separately. By default, Engine does not copy key material between devices. A key required by an operation must already be on the operation device:

```python
cuda_secret_key = cpu_secret_key.to("cuda:0")
plaintext_cuda = engine.decrypt(cuda_ciphertext, cuda_secret_key)
```

To opt into cached, lazy replicas of installed keys, construct the Engine with `allow_automatic_key_replication=True`. Enabling this setting permits secret, public, and evaluation-key tensors to be copied to devices selected by operands. The original key remains on its source device; this setting does not provide secure memory erasure or a device trust policy.

## Preserve one key relation across devices

Generating a new secret key independently on another device creates a different cryptographic relation. To use the same relation elsewhere, move that key or pass a destination device to a derived-key factory:

```python
cuda_secret_key = cpu_secret_key.to("cuda:0")
cuda_public_key = engine.create_public_key(cuda_secret_key)

# Equivalent placement request; the input relation is copied first.
cuda_rotation_key = engine.create_rotation_key(
    1,
    cpu_secret_key,
    device="cuda:0",
)
```

The Engine does not silently resample a destination secret key. Unless automatic key replication is enabled, it also does not silently copy key material for an operation.

## Compiled execution placement

Placement in a compiled function is part of its dataflow. A `.to(device)`, `.cpu()`, or `.cuda()` call on a value inside the function is captured as a transfer of its Tensors and executes at that position on every call; every other operation executes on the device of its operands, as in Eager mode. Compile neither inserts nor removes transfers, so a function that mixes devices states each transfer where it occurs. Argument devices are part of the call signature: calling the same function with CPU and CUDA inputs prepares two specializations. Materials such as keys and plaintexts captured from the function's closure or instance keep the device they had at capture; place them on the execution device before the first call.

A loaded artifact runs on the device passed to `fc.load(..., device=...)`, which places its CUDA materials and execution there; keys supplied to it are read on their own device and should be on that device. For a manual Program, prepare the materials on the intended device before linking with `OperationBackend`; `prepare_material_bindings` fills missing symbols from caller-supplied device-local providers and keys and never transfers or generates them. See [Save and relink a Program](save-and-relink-program.md).

## Account for backend-specific capabilities

CPU and CUDA share public CKKS methods, but not every optimization exists on both devices:

- CPU uses the indexed radix-2 NTT backend and Torch intra-op parallelism;
- CUDA may provide indexed radix-2 and configuration-selected compact NTT implementations;
- CUDA Graph execution of compiled callables, CUDA streams, and fixed device buffers are CUDA-specific; compiled callables on CPU run their schedules directly.

Callers that depend on one implementation should check the installed backend and requested implementation rather than infer capability from the public method name.

## Verify the outcome

Execute the same clear-message calculation after placement changes, inspect actual operand/key devices, and compare decoded error under the original criterion. Account for copies separately from steady arithmetic. Ciphertext residues need not match across independently randomized encryptions on different devices.
