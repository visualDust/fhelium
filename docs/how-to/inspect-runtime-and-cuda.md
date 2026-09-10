# Inspect runtime, memory, and CUDA topology

Use the `fhelium` command-line interface to record the installed package
version, inspect CUDA devices, and verify peer topology before running a
benchmark or distributed workload.

FHElium exposes related observations with different responsibilities:

- `fhelium.runtime.CpuTopology` reads the host CPU model and processor counts;
- `fhelium.runtime.CudaTopology` reads process-visible CUDA devices and
  directional peer-access capability;
- `fhelium.runtime.MemorySnapshot` reads current host or CUDA memory capacity
  and availability;
- `fhelium.native.native_status()` reports native-extension discovery, ABI
  validation, and loading;
- `fhelium.native.cuda` uses the compiled CUDA extension for device inspection
  and may run an opt-in peer-bandwidth probe.

## Inspect CPU and CUDA topology

```python
from fhelium.runtime import CpuTopology, CudaTopology

cpu = CpuTopology.probe()
cuda = CudaTopology.probe()

print(cpu.as_dict())
print(cuda.as_dict())
```

`CudaTopology` contains all process-visible CUDA devices and the directional
peer-access matrix:

```python
for device in cuda.devices:
    print(device.device, device.name, device.compute_capability)

print(cuda.peer_access)
```

CUDA indices follow the process-visible order after variables such as
`CUDA_VISIBLE_DEVICES` are applied. `CpuTopology.probe()` does not inspect or
initialize CUDA. Call `CudaTopology.probe()` only when CUDA inventory is
needed. A JIT Session performs the observation required by its selected
`device`; ordinary JIT callers do not probe topology first.

In `peer_access[source][destination]`, the row is the source device and the
column is the destination device. FHElium records diagonal entries as `True`;
off-diagonal entries come from `torch.cuda.can_device_access_peer(source,
destination)`. The matrix does not allocate buffers or measure bandwidth.

## Read current memory availability

Hardware topology and live memory availability have different lifetimes. Read
a new memory snapshot whenever current availability matters:

```python
from fhelium.runtime import MemorySnapshot

cpu_memory = MemorySnapshot.read("cpu")
gpu_memory = MemorySnapshot.read("cuda:0")

print(cpu_memory.available_bytes)
print(gpu_memory.available_bytes)
print(gpu_memory.torch_allocated_bytes)
print(gpu_memory.torch_reserved_bytes)
```

CPU snapshots use psutil's operating-system memory report on Linux, Windows,
and macOS. `available_bytes` estimates memory that can be supplied without
swapping. CUDA snapshots use `torch.cuda.mem_get_info`: their availability
covers the whole device and therefore reflects other processes and non-PyTorch
allocations.
`torch_allocated_bytes` and `torch_reserved_bytes` describe only the current
process's PyTorch caching allocator and are `None` for CPU snapshots.

Each object is an immutable point-in-time reading. Call `read` again to obtain
new counters. Global available memory is a process-wide capacity observation;
Residency reports values owned by its manager against caller-supplied budgets.

## Record the installed version

```bash
fhelium version
```

Run the command in the same environment that will import FHElium. Package
version is one part of the native application binary interface (ABI) record;
also preserve the Python,
PyTorch, CUDA, driver, and GPU information reported by the relevant tools.

## Inspect CUDA devices

Print the device summary:

```bash
fhelium cuda info
```

Request details for one zero-based CUDA device:

```bash
fhelium cuda info --device 0
```

Use machine-readable output in an experiment record or admission check:

```bash
fhelium cuda info --json > results/cuda-info.json
```

The same device inspection is available to Python applications through the
separate `fhelium.native.cuda` module:

```python
from fhelium.native.cuda import get_cuda_device_properties, get_cuda_info

devices = get_cuda_device_properties()
topology = get_cuda_info(test_p2p_bandwidth=False)
```

A visible GPU does not establish that the installed FHElium Torch operator
extension matches the runtime ABI. If import or operator loading fails,
compare the complete source-build environment in the
[installation guide](../tutorial/installation.md) and inspect the
[native runtime status API](../api/fhelium/native.md).

## Inspect peer topology

For a multi-GPU SPMD schedule, inspect peer-access relationships before
assuming that direct device-to-device transfer is available:

```bash
fhelium cuda topo
```

Add the opt-in bandwidth probe only on an idle system where allocating and
copying test buffers is acceptable:

```bash
fhelium cuda topo --bandwidth
```

Topology describes a transfer capability; it does not select a distributed
schedule, partition values, or establish the ownership of cryptographic
material. Use [Choose a multi-GPU partition](choose-multi-gpu-partition.md) for
those partitioning requirements.

## Discover benchmark commands

```bash
fhelium benchmark list
fhelium benchmark --help
```

Running `fhelium benchmark` without a subcommand opens the interactive terminal
interface. For reproducible automation, use a configured non-interactive
subcommand and preserve structured JSON output. See
[Benchmark methodology](/benchmarks/methodology).
