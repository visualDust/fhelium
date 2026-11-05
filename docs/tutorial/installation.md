# Installation

FHElium builds its native extension against the PyTorch installation in the active environment. Use Python 3.12 or 3.13, install the intended PyTorch distribution from the [official PyTorch selector](https://pytorch.org/get-started/locally/), and provide a compiler supporting the C++ standard required by that Torch build: C++17 or C++20.

<InstallCommand :show-details-link="false" />

Binary compatibility includes the Python ABI, pinned Torch version, Torch CUDA or HIP version, C++ ABI, and compiled GPU architectures. Published prebuilt wheels cover CPU and NVIDIA CUDA configurations. AMD ROCm builds currently use the source-checkout procedure below. Prebuilt Linux wheels are complete `fhelium` wheels served by FHElium's static release store. The installer above selects the configuration-specific wheel index corresponding to the chosen Torch and compute platform. PyPI provides the source distribution for supported source-build environments.

The following command builds from the PyPI source distribution against the preinstalled target Torch environment:

```bash
python -m pip install "scikit-build-core==1.0.3" "cmake>=3.18" ninja
CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU+CUDA" \
  python -m pip install \
    --no-binary=fhelium \
    --no-build-isolation --no-cache-dir --verbose \
    fhelium
```

CUDA source builds require a toolkit with the same major version as the preinstalled Torch CUDA build; using the same minor version is preferred. CPU-only builds set `CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU"`. `CMAKE_CUDA_ARCHITECTURES` optionally overrides the generated architecture list. The generated list emits native images for its default targets and PTX for the newest target recognized by the selected toolkit.

## Experimental AMD ROCm source builds

On Linux, install the intended ROCm PyTorch distribution and its compatible HIP development toolkit. PyTorch identifies AMD GPU Tensors with `device="cuda"`; `torch.version.hip` identifies the HIP runtime. Keep the Triton distribution supplied with that Torch environment.

From a source checkout, build CPU and HIP execution:

```bash
python -m pip install "scikit-build-core==1.0.3" "cmake>=3.21" ninja
CMAKE_ARGS="-DFHELIUM_NATIVE_BACKENDS=CPU+HIP -DCMAKE_HIP_ARCHITECTURES=gfx942" \
  python -m pip install --no-build-isolation --no-cache-dir --verbose .
```

`gfx942` targets MI300X. `CMAKE_HIP_ARCHITECTURES` selects the AMD code objects included in the extension; omitting it uses HIP toolchain discovery. `AUTO` chooses CPU plus the GPU backend of the selected Torch build. CMake compiles the checked-in HIP implementation directly and links Torch's HIP libraries. CUDA and HIP builds use separate Torch environments and produce distinct native ABI identities.

## Verify

```bash
python -c "import torch, fhelium; print(torch.__version__, fhelium.__version__)"
python -c "from fhelium.native import native_status; print(native_status())"
```

For a CUDA or HIP build, inspect visible devices with `fhelium cuda info` and peer connectivity with `fhelium cuda topo`. Native loading checks the Python, Torch, CUDA/HIP, and C++ ABI identities recorded at build time. Import errors include the detected mismatch and rebuild guidance; `native_status()` reports the loaded extension and its enabled backends.
