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
| `fhelium/rns_tables.h` | `fhelium::rns` | Parameter tables and the product, inverse, mixed-radix, CRT, and quotient tables of basis conversion, rescale, and ModDown |
| `fhelium/ntt_tables.h` | `fhelium::ntt` | Negacyclic roots, compact and fixed-radix twiddles, indexed radix-2 schedules, and NTT-domain Galois permutations |
| `fhelium/ckks_tables.h` | `fhelium::ckks` | Rotation Galois elements, coefficient Galois gathers, slot permutations and twists, and periodic-message positions |
| `fhelium/ckks_parameters.h` | `fhelium::ckks` | Built-in presets, the prime catalog, moduli validation, the security budget table, and the hybrid key-digit partition |
| `fhelium/random.h` | `fhelium::random` | ChaCha20 key streams and uniform, discrete Gaussian, centered binomial, and ternary samplers |
| `fhelium/tensor_table.h` | `fhelium` | `TensorTable`, the named prepared tables that whole operations read |
| `fhelium/ntt_operations.h` | `fhelium::ntt` | Forward and inverse transforms selected by NTT policy name |
| `fhelium/rns_operations.h` | `fhelium::rns` | Hybrid ModUp, basis extension, ModDown, and rescale |
| `fhelium/ckks_operations.h` | `fhelium::ckks` | Encoding and decoding, encryption and decryption, key switching, relinearization, rotation, rotation sums, modulus raising, compressed-plaintext arithmetic, and scalar arithmetic |
| `fhelium/ckks_keys.h` | `fhelium::ckks` | Secret, public, key-switching, relinearization, and Galois key generation, and secret-key basis conversion |

Each function has the name and arguments of the Torch operator with the same name; `preset_parameters` returns a `CkksParameters` structure, and its operator returns the same values as a tuple. A whole operation or key-generation function takes a `TensorTable` and an options structure such as `KeySwitchOptions` or `KeySwitchKeyLayout`; its operator receives the table as parallel `str[] table_names` and `Tensor[] tables` arguments and the options as separate attributes. Mutated schema arguments are `at::Tensor&`, and a trailing underscore marks a function that mutates its first argument in place. A call executes on the device of its Tensor arguments: all-CPU arguments select the CPU implementation, and a CUDA argument selects the GPU implementation. A function that has no implementation for that device in the installed build raises `c10::NotImplementedError`.

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

The table functions compute on the host and return CPU Tensors in standard residues; move them to the execution device and convert twiddles to Montgomery form with `to_montgomery_` where the consuming function requires it. The Python contexts build their tables with the same functions. Evaluation keys are generated in Python and can be transferred with Torch serialization.

The repository example [`examples/cpp`](https://github.com/VisualDust/fhelium/tree/main/examples/cpp) builds an RNS parameter table with `parameter_table`, runs Montgomery multiplication, standard addition, and two-component ciphertext multiplication on CPU and on CUDA when available, compares each result with host modular arithmetic, and checks one ChaCha20 key-stream block against the RFC 8439 test vector.
