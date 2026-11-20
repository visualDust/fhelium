# C++ reference

`libfhelium` exports the native CKKS, RNS, and NTT functions as C++ functions on ATen Tensors. Each function is also registered as the Torch operator of the same name, which FHElium's Python Backend calls, so one entry documents both interfaces. [Call the native library from C++](../../how-to/call-native-library-from-cpp.md) describes how to link the library through the `FHElium` CMake package.

| Header | Namespace | Contents |
| --- | --- | --- |
| [`fhelium/ckks.h`](ckks.md) | `fhelium::ckks` | Ciphertext products, plaintext addition, rescaling, Galois automorphisms, and key-switching steps |
| [`fhelium/rns.h`](rns.md) | `fhelium::rns` | Modular arithmetic, Montgomery conversion, and basis conversion on residue Tensors |
| [`fhelium/ntt.h`](ntt.md) | `fhelium::ntt` | Forward and inverse negacyclic NTTs and indexed-schedule validation |
| [`fhelium/ntt_diagnostic.h`](ntt-diagnostic.md) | `fhelium::ntt::diagnostic` | NTT functions with a selectable shared-memory transform size |
| [`fhelium/rns_parameters.h`](rns-parameters.md) | `fhelium::rns` | Row layout of the `rns_params` Tensor |

`fhelium/fhelium.h` includes every header.

## Devices

A function selects its implementation from the devices of its Tensor arguments: any CUDA Tensor selects the GPU implementation, and CPU Tensors alone select the CPU implementation. The Devices column lists the implementations a function provides. A build contains the CPU implementation and the GPU implementation of its configured backend, CUDA or HIP; calling a function whose implementation is absent from the build raises an error. Undefined optional Tensors do not take part in the selection.

## Generation rule

`scripts/generate_cpp_api_docs.py` generates these pages from the public headers in `csrc/include/fhelium`. A `//!` comment before a header's namespace describes the header, and a `///` comment documents the declaration that follows it. Torch operator names and schemas come from the registrations in `csrc/torchops`, and device implementations come from the function routers in `csrc/ops`. Generation fails when a registered function has no documented declaration.
