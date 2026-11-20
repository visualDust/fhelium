# Call the native library from C++

`libfhelium` is the native library behind the `torch.ops.fhelium_*` operators. A C++ program links it through the `FHElium` CMake package and calls the same CKKS, RNS, and NTT Tensor functions that FHElium's Python Backend uses.

## Prerequisites

Install FHElium with a native backend in a Python environment, as described in the [installation guide](../tutorial/installation.md). The installed package contains the C++ SDK: `libfhelium` (`fhelium.dll` with its import library `fhelium.lib` on Windows), the public headers, and the CMake package. The C++ program compiles against the Torch headers and libraries of the same environment, with a C++20 compiler compatible with that Torch build.

## Locate the CMake packages

`fhelium.native.cmake_prefix_path` and `torch.utils.cmake_prefix_path` give the two CMake prefixes:

```bash
PREFIX="$(python -c 'import fhelium.native, torch.utils; print(fhelium.native.cmake_prefix_path + ";" + torch.utils.cmake_prefix_path)')"
```

The `FHElium` package accepts only its exact FHElium version and the Torch build, including its CUDA or ROCm variant, that `libfhelium` was compiled against. It defines the imported target `FHElium::fhelium`, which carries the include directory and the `torch` link dependency, and the variables `FHElium_NATIVE_BACKENDS` and `FHElium_TORCH_VERSION`.

## Link the library

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_application LANGUAGES CXX)

find_package(FHElium CONFIG REQUIRED)

add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE FHElium::fhelium)
```

Configure with the prefixes from the previous step. For a CUDA build of Torch, Torch's CMake package enables the CUDA language and locates the CUDA Toolkit from `CMAKE_CUDA_COMPILER`:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"
cmake --build build
```

Omit `CMAKE_CUDA_COMPILER` for a CPU build of Torch.

On Windows, add the directory that contains `fhelium.dll` (`fhelium/native/lib` in the installed package) and the Torch `lib` directory to `PATH` before running the program.

## Call functions

`<fhelium/fhelium.h>` declares the complete interface:

| Header | Namespace | Content |
| --- | --- | --- |
| `fhelium/rns.h` | `fhelium::rns` | Modular arithmetic, Montgomery conversion, and basis conversion on residue Tensors |
| `fhelium/ntt.h` | `fhelium::ntt` | Forward and inverse negacyclic NTTs and indexed-schedule validation |
| `fhelium/ckks.h` | `fhelium::ckks` | Ciphertext multiplication, plaintext addition, rescale, Galois automorphisms, and key-switch primitives |
| `fhelium/ntt_diagnostic.h` | `fhelium::ntt::diagnostic` | NTT variants with an explicit shared-memory transform size, for profiling |
| `fhelium/rns_parameters.h` | `fhelium::rns` | Row layout of the `rns_params` operand |

Each function has the name and arguments of the Torch operator with the same name. Mutated schema arguments are `at::Tensor&`, and a trailing underscore marks a function that mutates its first argument in place. A call executes on the device of its Tensor arguments: all-CPU arguments select the CPU implementation, and a CUDA argument selects the GPU implementation. A function that has no implementation for that device in the installed build raises `c10::NotImplementedError`.

```cpp
#include <fhelium/fhelium.h>
#include <torch/torch.h>

// residues: [*batch, limb, N] int64 standard residues
// params:   [fhelium::rns::RNS_PARAMETER_COUNT, limb] on the same device
at::Tensor square(const at::Tensor& residues, const at::Tensor& params) {
  auto x = residues.clone();
  fhelium::rns::to_montgomery_(x, params);
  auto y = fhelium::rns::montgomery_mul(x, x, params);
  fhelium::rns::from_montgomery_(y, params);
  return y;  // x^2 mod q_j in [0, 2 q_j)
}
```

The functions consume Tensors only; they do not construct parameter tables, NTT twiddles, schedules, or keys. Obtain these Tensors from the FHElium Python contexts, for example `RnsContext.rns_parameters_for`, and transfer them with Torch serialization, or compute them in the application according to the layout in `fhelium/rns_parameters.h`.

The repository example [`examples/cpp`](https://github.com/VisualDust/fhelium/tree/main/examples/cpp) computes an RNS parameter table in C++, runs Montgomery multiplication, standard addition, and two-component ciphertext multiplication on CPU and on CUDA when available, and compares each result with 128-bit integer arithmetic.
