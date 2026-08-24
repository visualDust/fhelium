# FHElium: FHE, built from the tensor up

> Pronounced **“philium”** /ˈfɪliəm/ or **“F-helium”** /ˌɛf ˈhiːliəm/.

[![PyPI version](https://img.shields.io/pypi/v/fhelium)](https://pypi.org/project/fhelium/) [![Python versions](https://img.shields.io/pypi/pyversions/fhelium)](https://pypi.org/project/fhelium/) [![PyPI downloads](https://img.shields.io/pypi/dw/fhelium)](https://pypi.org/project/fhelium/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Website:** [fhelium.550w.host](https://fhelium.550w.host)

FHElium is a CPU- and CUDA-accelerated homomorphic-encryption library for Python and PyTorch. It integrates encrypted arithmetic with PyTorch's tensor programming and execution infrastructure. Its tensor-first design represents CKKS values, state, and operations through familiar tensor-oriented APIs.

> FHElium is under active development. The API may change between releases.

## Install

FHElium currently supports Linux x86-64, Windows x86-64, and macOS Apple Silicon with Python 3.12 or 3.13, PyTorch `>=2.10,<2.14`, and a C++17 host compiler. The default build follows the target Torch package: CPU-only Torch produces a CPU-only extension, while CUDA-enabled Torch produces one extension with CPU and CUDA implementations. CUDA builds on Linux or Windows additionally require a matching CUDA toolkit and CUDA C++17 compiler. macOS execution uses the native CPU backend, not PyTorch MPS.

Use the [installation selector](https://fhelium.550w.host/#install-fhelium) for a compatible prebuilt Linux wheel or a source-build command for the selected Torch environment. Prebuilt wheels are complete `fhelium` wheels served from FHElium's static release store; PyPI provides the source distribution.

For a source build, install the intended PyTorch build first, following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then build FHElium in the same Python environment:

```bash
python -m pip install "scikit-build-core==1.0.3" "cmake>=3.18" ninja
python -m pip install \
  --no-binary=fhelium \
  --no-build-isolation --no-cache-dir --verbose \
  fhelium
```

The build uses the installed Torch stack. `--no-build-isolation` keeps that stack available while compiling the native code, and `--no-cache-dir` prevents reuse of a locally compiled wheel in another Torch environment.

Set `CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU"` for a CPU-only build even
with CUDA-enabled Torch, `CUDA` for CUDA-only, or `CPU+CUDA` for an explicit
combined build.

The [installation guide](https://fhelium.550w.host/tutorial/installation) covers CUDA architecture selection and build troubleshooting.

## Quick start

```python
import torch
import fhelium as fh

engine = fh.CkksEngine(
    fh.Preset.slots8192_scale40_levels7_int64,
    device="cpu",  # use "cuda:0" to dispatch the same API to CUDA
)

x = torch.linspace(-0.05, 0.05, 32, dtype=torch.float64)
y = torch.linspace(0.02, -0.02, 32, dtype=torch.float64)

ct_x = engine.encrypt_message(x)
ct_y = engine.encrypt_message(y)

ct_sum = engine.add(ct_x, ct_y)

x_ntt = engine.coefficient_domain_to_ntt_domain(ct_x)
y_ntt = engine.coefficient_domain_to_ntt_domain(ct_y)
triplet = engine.multiply(x_ntt, y_ntt)
product = engine.rescale_to_next_level(engine.relinearize(triplet))

rotated = engine.rotate_by_step(ct_x, 1)

sum_clear = engine.decrypt_message(ct_sum, is_real=True)[: x.numel()]
product_clear = engine.decrypt_message(product, is_real=True)[: x.numel()]

assert rotated.level == ct_x.level
torch.testing.assert_close(sum_clear, x + y, atol=2e-5, rtol=0)
torch.testing.assert_close(product_clear, x * y, atol=2e-5, rtol=0)
```

`Preset` member names record complex slot capacity, default scale bits,
public-level count, and integral tensor dtype. Maintained int32 and int64
families are listed in the
[preset and chain-depth guide](https://fhelium.550w.host/how-to/choose-preset-and-depth).

Each value records its CKKS level, actual scale, active primes, polynomial domain, modulus basis, and residue representation. The program chooses when to change those states: multiplication does not silently relinearize or rescale, and addition requires matching scales. Methods ending in `_` mutate their first value, following the PyTorch naming convention.

## Across GPUs

FHElium uses a single-program, multiple-data (SPMD) model. Each process owns one `CkksEngine` and one local device; `fhelium.distributed` handles communication for tensor-backed encrypted values:

```python
import fhelium as fh
import fhelium.distributed as dist

dist.init()
engine = fh.CkksEngine(
    fh.Preset.slots32768_scale40_levels34_int64,
    device=dist.local_device(),
    allow_sk_gen=False,
)
```

The distributed API separates three different relationships between ranks:

- independent plaintexts or ciphertexts moved with scatter, gather, or broadcast;
- additive ciphertext partials combined with CKKS modular addition;
- residue-number-system limbs partitioned from one ciphertext and reconstructed later.

Data partitioning, key movement, and communication schedules remain part of the application. The [distributed examples](./examples/README.md) show one- and two-GPU programs using the same API.

## JIT programs

`fhelium.experimental.jit` traces typed PyTorch callables or imports textual xDSL into one mixed-dialect `Program`. Selected pass pipelines transform the program, while live materials, engines, keys, handlers, resources, and caches remain in a retained workspace. Execution begins with an independent readiness check:

```python
from fhelium.experimental import jit

captured = jit.trace(
    lambda secret, public: secret + public,
    inputs={"secret": jit.encrypted(), "public": jit.message()},
)
lowered = jit.default_pipeline().run(
    captured.program,
    captured.workspace,
)
print(lowered.program.to_text())
```

See the [JIT tutorial](https://fhelium.550w.host/tutorial/unified-jit) for runtime provisioning and encrypted execution.

FHElium is also moving toward a multi-backend kernel architecture, including an additional TileLang backend.

## Learn more

- [Tutorials](https://fhelium.550w.host/tutorial/)
- [Programming model](https://fhelium.550w.host/concepts/programming-model)
- [CKKS concepts](https://fhelium.550w.host/concepts/ckks/context-and-modulus-chain)
- [How-to guides](https://fhelium.550w.host/how-to/)
- [API reference](https://fhelium.550w.host/api/)
- [Developer guide](https://fhelium.550w.host/developer/)

Start with [`examples/01_basic_ckks_flow.py`](./examples/01_basic_ckks_flow.py) or browse [`examples/README.md`](./examples/README.md) for key management, scale and state transitions, bootstrapping, rotation hoisting, distributed execution, CUDA Graphs, serialization, residency, batching, compressed plaintexts, and JIT workflows.

## Development

Development uses an editable checkout. Clone the repository, then choose one
of two environment paths:

```bash
git clone https://github.com/VisualDust/fhelium.git
cd fhelium
```

- Use **uv** for the maintained default environment. uv resolves the tracked
  lock, creates `.venv`, installs the development dependencies, and builds
  FHElium as an editable package.
- Use **pip** when the environment must retain a selected Torch build, such as
  a different CUDA variant. Install that Torch build first; pip then builds
  FHElium against the active environment without replacing its Torch choice.

These paths are alternatives for one environment. Use separate virtual
environments when testing both: `uv sync --locked` restores the default locked
Torch line, while the pip path preserves the Torch package already selected.

The repository files have separate responsibilities:

- `pyproject.toml` is the package-manager interface for dependency metadata. It
  declares the build and development groups and identifies the provider for
  dynamic runtime dependencies.
- `uv.lock` records one tested resolution of those declarations for the
  default developer environment, currently using Torch 2.13.
- `packaging/release_matrix.json` declares the Python, Torch, CUDA, and
  operating-system configurations for published artifacts. The developer lock
  does not replace this matrix.
- `justfile` provides optional shortcuts over the commands below. It does not
  define dependencies or supported release configurations.

### Default environment with uv

```bash
uv --preview-features extra-build-dependencies sync --locked
```

Activate the resulting environment with `source .venv/bin/activate` on Linux
or macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. The repository
requires a validated minimum uv version because uv 0.9 labels package-specific
build dependencies as a preview feature.

### Environment with a selected Torch build

After installing the intended Torch package in the active environment:

```bash
python -m pip install --group build
python -m pip install \
  --editable . --no-build-isolation --no-cache-dir --verbose
python -m pip install --group dev
```

`--no-build-isolation` makes the selected Torch ABI visible to the native
build. `--no-cache-dir` prevents reuse of a wheel compiled for another Torch
environment.

### Validate a change

Install the repository hooks once from either environment:

```bash
pre-commit install
```

Then run the project checks:

```bash
python -m pytest -q
ruff check .
ruff format --check .
pyright
```

If `just` is installed, `just check` runs the Ruff, Pyright, and pytest commands
above. Use `just NATIVE_BACKENDS=CPU build-uv` in the uv environment or
`just NATIVE_BACKENDS=CPU build-pip` in the pip environment to rebuild through
the corresponding path. Select `CPU+CUDA` only when that environment contains
a CUDA-enabled Torch package and matching toolkit. uv and `just` remain
optional: pip is the public installation and release-validation path.

The [developer guide](https://fhelium.550w.host/developer/) follows calls across the Python API, PyTorch dispatcher, C++, and CUDA kernels.

## Citation

If you use FHElium in research or software, cite the project as:

```bibtex
@software{fhelium2026,
  author  = {Zhaoting Gong and Jiaming Liang and Ran Ran and Wujie Wen},
  title   = {FHElium: A Cross-Stack CKKS Research Framework for CPU and CUDA},
  year    = {2026},
  version = {0.10.0},
  url     = {https://github.com/VisualDust/fhelium}
}
```

The same software citation metadata is available in [`CITATION.cff`](./CITATION.cff).

## License

FHElium is licensed under the [MIT License](./LICENSE).
