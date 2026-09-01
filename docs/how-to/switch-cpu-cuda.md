# Choose and switch a local execution device

`fhelium.eager.Engine` constructs execution resources lazily for each concrete
CPU or CUDA device used by an operation. It does not own a default device. The
same public CKKS methods and registered Tensor implementations serve both
backends.

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

Backend inclusion is a source-build choice. A visible GPU does not add CUDA to
a CPU-only `_ops` library. Rebuild according to the
[installation guide](../tutorial/installation.md#select-native-backends) when
the required backend is absent.

## Select placement for created values

```python
import fhelium as fh
import torch
from fhelium.eager import Engine

config = fh.CkksConfig.parse(fh.Preset.slots8192_scale40_levels7_int64)
engine = Engine(config)

torch.set_default_device("cuda:0")
plaintext = engine.encode(message)
secret_key = engine.create_secret_key()
```

Source and material factories use `torch.get_default_device()` when `device`
is omitted. A caller may instead select placement on each call:

```python
plaintext = engine.encode(message, device="cuda:0")
secret_key = engine.create_secret_key(device="cuda:0")
public_key = engine.create_public_key(secret_key)
ciphertext = engine.encrypt(plaintext, public_key)
```

The first operation on a device constructs and caches that device's RNS/NTT
tables, random stream, operation bindings, and direct-dispatch state. Later calls
reuse them. Random streams on different devices use separate nonce domains.

## Dispatch ordinary operations from values

Homomorphic operations dispatch from Tensor operand placement:

```python
cuda_sum = engine.add(cuda_left, cuda_right)
```

They do not consult PyTorch's default device and do not move operands. Invalid
mixed-device calls fail in Torch or native execution rather than triggering an
Engine copy:

```python
engine.add(cpu_ciphertext, cuda_ciphertext)
```

## Request a boundary transfer

Tensor-backed values expose direct movement:

```python
cuda_ciphertext = ciphertext.to("cuda:0")
cpu_ciphertext = cuda_ciphertext.to("cpu")
```

A boundary operation may also receive `device`. Supplying it authorizes the
public adapter to move its plaintext or ciphertext before executing on the
target:

```python
cpu_slots = engine.decode(
    engine.decrypt(cuda_ciphertext, cpu_secret_key, device="cpu"),
    device="cpu",
)
```

Without `device`, encrypt, decrypt, and decode inherit placement from their
materialized plaintext or ciphertext. Movement preserves level, scale, prime
IDs, polynomial domain, modulus basis, residue representation,
and component count.

Key placement is governed separately. By default, Engine does not copy key
material between devices. A key required by an operation must already be on
the operation device:

```python
cuda_secret_key = cpu_secret_key.to("cuda:0")
plaintext_cuda = engine.decrypt(cuda_ciphertext, cuda_secret_key)
```

To opt into cached, lazy replicas of installed keys, construct the Engine with
`allow_automatic_key_replication=True`. Enabling this setting permits secret,
public, and evaluation-key tensors to be copied to devices selected by
operands. The original key remains on its source device; this setting does not
provide secure memory erasure or a device trust policy.

## Preserve one key relation across devices

Generating a new secret key independently on another device creates a different
cryptographic relation. To use the same relation elsewhere, move that key or
pass a destination device to a derived-key factory:

```python
cuda_secret_key = cpu_secret_key.to("cuda:0")
cuda_public_key = engine.create_public_key(cuda_secret_key)

# Equivalent explicit placement request; the input relation is copied first.
cuda_rotation_key = engine.create_rotation_key(
    1,
    cpu_secret_key,
    device="cuda:0",
)
```

The Engine does not silently resample a destination secret key. Unless
automatic key replication is enabled, it also does not silently copy key
material for an operation.

## Compile and Experimental JIT placement

A transformed `Program` does not allocate device resources. Backend linking or
JIT specialization selects resources before execution and can
reuse them across `run()` calls.
The caller selects a Backend pipeline and execution device. Its resource pass
materializes Program-wide arithmetic resources once, while the caller supplies
keys, process groups, and other externally owned resources. Assignment then
builds the executable used across compatible inputs. Compile and Backend own
the stable Program execution interface; Eager owns automatic key-replication
policy.

A creation operation may define its own static or dynamic device parameter when
it allocates a new Tensor without a placed Tensor operand. Operations over
existing Tensor values dispatch from operand placement. A pass that changes
placement must insert a real transfer operation before or after the numerical
operation, rather than attaching a generic device attribute to every operation.
`fhelium_memory.transfer` represents an actual placement change; mixed-device
arithmetic regions still require callers to insert and schedule those transfers.

## Account for backend-specific capabilities

CPU and CUDA share public CKKS methods, but not every optimization exists on
both devices:

- CPU uses the indexed radix-2 NTT backend and Torch intra-op parallelism;
- CUDA may provide indexed radix-2 and configuration-selected compact NTT
  implementations;
- CUDA Graph capture, CUDA streams, and device-resident graph buffers remain
  CUDA-specific execution mechanisms.

Callers that depend on one implementation should check the installed backend
and requested implementation rather than infer capability from the public
method name.
