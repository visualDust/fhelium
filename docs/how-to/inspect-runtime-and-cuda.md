# Inspect the runtime and CUDA topology

Use the `fhelium` command-line interface to record the installed package
version, inspect CUDA devices, and verify peer topology before running a
benchmark or distributed workload.

## Record the installed version

```bash
fhelium version
```

Run the command in the same environment that will import FHElium. Package
version alone is not a complete native application binary interface (ABI)
record; also preserve the Python,
PyTorch, CUDA, driver, and GPU information reported by the environment.

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
[Benchmark a workload correctly](benchmark-a-workload.md).
