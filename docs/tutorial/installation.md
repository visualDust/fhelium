# Installation

FHElium builds its native extension against the PyTorch installation in the
active environment. Use Python 3.12 or 3.13, install the intended PyTorch
distribution from the [official PyTorch selector](https://pytorch.org/get-started/locally/),
and provide a C++17-capable compiler.

<InstallCommand :show-details-link="false" />

Binary compatibility includes the Python ABI, pinned Torch version, Torch CUDA
version, C++ ABI, and compiled GPU architectures. Prebuilt Linux wheels are
complete `fhelium` wheels served by FHElium's static release store. The
installer above selects the configuration-specific wheel index corresponding
to the chosen Torch and compute platform. PyPI provides the source distribution
for supported source-build environments.

The following command builds from the PyPI source distribution against the
preinstalled target Torch environment:

```bash
python -m pip install "scikit-build-core>=1.0.3" "cmake>=3.18" ninja
CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU+CUDA" \
  python -m pip install \
    --no-binary=fhelium \
    --no-build-isolation --no-cache-dir --verbose \
    fhelium
```

CUDA source builds require a toolkit with the same major version as the
preinstalled Torch CUDA build; using the same minor version is preferred.
CPU-only builds set `CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU"`.
`CMAKE_CUDA_ARCHITECTURES` optionally overrides the generated architecture
list.

## Verify

```bash
python -c "import torch, fhelium; print(torch.__version__, fhelium.__version__)"
python -c "from fhelium.native import native_status; print(native_status())"
```

For a CUDA build, inspect visible devices with `fhelium cuda info`. If the
selected Torch, CUDA, compiler, or cached native binary is incompatible,
`native_status()` reports the detected build metadata and a targeted rebuild
instruction.
