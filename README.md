# FHElium: FHE, built from the tensor up

> Pronounced **“philium”** /ˈfɪliəm/ or **“F-helium”** /ˌɛf ˈhiːliəm/.

[![PyPI version](https://img.shields.io/pypi/v/fhelium)](https://pypi.org/project/fhelium/) [![Python versions](https://img.shields.io/pypi/pyversions/fhelium)](https://pypi.org/project/fhelium/) [![PyPI downloads](https://img.shields.io/pypi/dw/fhelium)](https://pypi.org/project/fhelium/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Documentation:** [fhelium.550w.host](https://fhelium.550w.host)

FHElium is a full-stack CKKS framework for Python and PyTorch. It provides
native CPU and NVIDIA CUDA execution, tensor-backed encrypted values, immediate
Eager evaluation, inspectable Compile Programs, runtime resource management,
and rank-local distributed execution.

## Install

FHElium supports CPython 3.12 and 3.13 on Linux x86-64, Windows x86-64, and
macOS Apple Silicon. It builds against the PyTorch installation selected by the
user. CUDA builds require a compatible CUDA toolkit and C++17 compiler; macOS
uses the CPU implementation rather than PyTorch MPS.

Use the [installation selector](https://fhelium.550w.host/#install-fhelium) to
choose the operating system, Python version, PyTorch version, and CPU or CUDA
configuration. The selector provides a prebuilt-wheel command when that exact
combination is published and a source-build command otherwise.

For a source installation, install the intended PyTorch package first, then
build FHElium in the same environment:

```bash
python -m pip install "scikit-build-core==1.0.3" "cmake>=3.18" ninja
python -m pip install \
  --no-binary=fhelium \
  --no-build-isolation --no-cache-dir --verbose \
  fhelium
```

`--no-build-isolation` keeps the selected Torch build available to CMake, and
`--no-cache-dir` prevents reuse of a wheel compiled for another Torch or CUDA
environment. Set `CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU"` for a CPU-only
build, `CUDA` for CUDA-only, or `CPU+CUDA` for both implementations.

See the [installation guide](https://fhelium.550w.host/tutorial/installation)
for compiler requirements, CUDA architecture selection, and troubleshooting.

## Programming models

FHElium provides two first-class ways to execute CKKS computations:

- `fhelium.eager.Engine` executes each requested operation immediately;
- `fhelium.compile` constructs and transforms source-independent
  `fhelium.ir.Program` objects.

Both use the same `fhelium.backend` operation implementations and arithmetic
resources. Eager follows the public state transition of each called operation
and dispatches from operand placement. Compile callers select the passes that
assign CKKS state, preserve or lower operations, bind materials and keys, and
construct a `ProgramExecutable`.

### Eager execution

```python
import torch
import fhelium as fh
from fhelium.eager import Engine

# Change this to "cuda:0" to create inputs and keys on CUDA.
torch.set_default_device("cpu")
engine = Engine(fh.Preset.slots8192_scale40_depth7_int64)

x = torch.linspace(-0.05, 0.05, 32, dtype=torch.float64)
y = torch.linspace(0.02, -0.02, 32, dtype=torch.float64)

ct_x = engine.encrypt_message(x)
ct_y = engine.encrypt_message(y)

ct_sum = engine.add(ct_x, ct_y)

x_ntt = engine.coefficient_domain_to_ntt_domain(ct_x)
y_ntt = engine.coefficient_domain_to_ntt_domain(ct_y)
product_triplet = engine.multiply(x_ntt, y_ntt)
ct_product = engine.rescale_to_next_depth(
    engine.relinearize(product_triplet)
)

rotation_key = engine.rotation_key(1)
ct_rotated = engine.rotate_with_key(ct_x, rotation_key)

sum_clear = engine.decrypt_message(ct_sum, is_real=True)[: x.numel()]
product_clear = engine.decrypt_message(ct_product, is_real=True)[: x.numel()]

torch.testing.assert_close(sum_clear, x + y, atol=2e-5, rtol=0)
torch.testing.assert_close(product_clear, x * y, atol=2e-5, rtol=0)
```

Each encrypted value records its CKKS level, actual scale, active primes,
polynomial domain, modulus basis, residue representation, and component count.
Multiplication does not implicitly relinearize or rescale, and operations do
not silently move Tensor payloads between devices.

### Compile Programs

Compile captures or imports a source-independent Program and applies the pass
sequence selected by the caller:

```python
from fhelium import compile as fh_compile


def workload(secret, public):
    return secret + public


captured = fh_compile.capture(
    workload,
    inputs={
        "secret": fh_compile.encrypted(),
        "public": fh_compile.message(),
    },
)

pipeline = fh_compile.Pipeline(
    (
        fh_compile.EliminateDeadValuesPass(),
        fh_compile.LowerSemanticToLogicalPass(),
    )
)
compiled = pipeline.run(captured)

print(compiled.program.to_text())
print(compiled.reports)
```

A caller can continue from the same `Compilation` with CKKS state assignment,
transition placement, CKKS-to-RNS/NTT lowering, key analysis, Backend linking,
or custom passes. See
[`examples/17_compose_and_execute.py`](./examples/17_compose_and_execute.py)
for an encrypted end-to-end execution and
[`examples/19_customize_compile_pass.py`](./examples/19_customize_compile_pass.py)
for a caller-defined BSGS transformation.

## Runtime and distributed execution

`fhelium.runtime` provides device observations, reusable value buffers, and
CUDA Graph execution. `fhelium.residency` manages live value placement and
lifetime accounting. `fhelium.distributed` provides rank-local value transport
and CKKS-aware collectives. Bootstrapping, multiparty CKKS, and JIT interfaces
are available under `fhelium.experimental`.

The runtime components can be used independently around an evaluator:

```python
from fhelium.runtime import (
    CpuTopology,
    CudaGraphProgram,
    MemorySnapshot,
    ReusableValueBuffer,
)
from fhelium.residency import PAGEABLE_HOST, ResidencyManager

cpu = CpuTopology.probe()
memory = MemorySnapshot.read("cuda:0")
buffer = ReusableValueBuffer.like(prototype, device="cuda:0")
program = CudaGraphProgram.capture(evaluator, example_inputs=(prototype,))
result = program.replay(next_input, synchronize=True)

residency = ResidencyManager()
weight_handle = residency.adopt(weight, at=PAGEABLE_HOST)
snapshot = residency.snapshot()
```

FHElium uses a single-program, multiple-data (SPMD) model. Each process owns a
local device and an Eager Engine; the application chooses its data partition,
keys, process groups, and communication schedule.

```python
import torch
import fhelium as fh
import fhelium.distributed as dist
from fhelium.eager import Engine


def main():
    dist.init()
    torch.set_default_device(dist.local_device())
    engine = Engine(fh.Preset.slots32768_scale40_depth34_int64)

    weight = (
        engine.encode(torch.ones(16, dtype=torch.float64))
        if dist.get_rank() == 0
        else None
    )
    weight = dist.broadcast_plaintext(weight, src=0)
    print(
        f"rank={dist.get_rank()} device={weight.device} depth={weight.depth}"
    )
    dist.shutdown()


if __name__ == "__main__":
    main()
```

Save the program as `distributed_example.py` and launch one process per local
GPU:

```bash
torchrun --standalone --nproc-per-node=2 distributed_example.py
```

The distributed API supports transport of independent values, modular
reduction of additive ciphertext partials, and partitioning/reconstruction of
residue-number-system limbs.

## Command-line tools

The `fhelium` command reports the installed version, inspects CUDA devices and
peer topology, runs benchmarks, and evaluates NTT backend candidates:

```bash
fhelium version
fhelium cuda info
fhelium cuda topo --bandwidth
fhelium benchmark list
fhelium benchmark run --device cpu --output results/benchmark.json
fhelium benchmark recommend ntt --suite kernel --device cuda:0
```

Running `fhelium benchmark` without a subcommand opens the interactive
benchmark interface. See [Inspect runtime and CUDA](https://fhelium.550w.host/how-to/inspect-runtime-and-cuda)
and [Benchmark methodology](https://fhelium.550w.host/benchmarks/methodology)
for the complete command options and output schemas.

## Documentation and examples

- [Tutorials](https://fhelium.550w.host/tutorial/)
- [Concepts](https://fhelium.550w.host/concepts/)
- [How-to guides](https://fhelium.550w.host/how-to/)
- [Architecture](https://fhelium.550w.host/concepts/architecture/system-overview)
- [API reference](https://fhelium.550w.host/api/)
- [Developer guide](https://fhelium.550w.host/developer/)

Start with [`examples/01_basic_ckks_flow.py`](./examples/01_basic_ckks_flow.py)
or browse [`examples/README.md`](./examples/README.md) for Eager, Compile,
CUDA Graph, distributed, persistence, Residency, batching, Bootstrap, and
multiparty examples.

## Development

Create the locked development environment and run the project checks:

```bash
git clone https://github.com/VisualDust/fhelium.git
cd fhelium
uv sync --locked
source .venv/bin/activate
just check
```

The [contributor guide](https://fhelium.550w.host/developer/contributing)
documents custom Torch environments, native builds, editor configuration, and
the validation workflow.

## Citation

If you use FHElium in research or software, cite the project as:

```bibtex
@software{fhelium2026,
  author  = {Zhaoting Gong and Jiaming Liang and Ran Ran and Wujie Wen},
  title   = {FHElium: A Cross-Stack CKKS Research Framework for CPU and CUDA},
  year    = {2026},
  version = {0.25.0},
  url     = {https://github.com/VisualDust/fhelium}
}
```

The same metadata is available in [`CITATION.cff`](./CITATION.cff).

## License

FHElium is licensed under the [MIT License](./LICENSE).
