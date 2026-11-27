# libfhelium sources

`csrc` holds libfhelium, the C++ library that implements FHElium's numerical work, and the PyTorch registration module that exposes it to Python. Every public function is declared in `include/fhelium`, implemented in one of the source directories below, and bound once in `torchops` as the Torch operator of the same name.

## Layout

```text
csrc/
├── include/fhelium/   public headers; the only interface shared across directories
├── ops/               operations on Tensor data
│   ├── rns/           residue arithmetic primitives
│   ├── ntt/           NTT primitives
│   ├── ckks/          CKKS Tensor primitives (automorphisms, key-switch products, ...)
│   ├── random/        ChaCha20 key streams and samplers
│   ├── operations/    whole RNS, NTT, and CKKS operations composed from the primitives
│   └── common/        device routing, validation, and arithmetic shared by the primitives
├── parameters/        CKKS presets, prime catalog, security table, key-digit partition
├── tables/            host construction of RNS, NTT, and CKKS numerical tables
├── keys/              CKKS key generation composed from the primitives
├── runtime/           CUDA and HIP device and peer inspection (separate modules)
└── torchops/          Torch schemas and registrations of the public functions (`_ops`)
```

The directories separate code by role:

- `ops` contains operations: functions that transform Tensor data and that a Backend implementation executes for an IR operation. Primitives (`ops/<family>`) run on a device; whole operations (`ops/operations`) compose primitives and read their numerical operands from the table structs of `fhelium/operation_tables.h`.
- `parameters` and `tables` run on the host before execution. Parameter functions return host values such as moduli and digit sizes; table functions return CPU Tensors that callers move to the execution device.
- `keys` composes primitives into key generation. Keys are prepared data for execution, so key generation has no IR operation; its functions are ordinary C++ functions with Torch bindings for Python.
- `torchops` is the Python binding of every public function. Registration in `torchops` does not make a function an operation.

## Dependencies

Code in one directory uses other directories only through `include/fhelium`. A private header (for example `ops/operations/composition.h` or `tables/table_support.h`) is included only from its own directory. Primitives, `parameters`, and `tables` call no other source directory; `ops/operations` and `keys` call primitives; `torchops` binds every public function. A function never calls a directory that depends on it.

## Primitives

A primitive family `ops/<family>` has the same structure in every family:

| File | Contents |
| --- | --- |
| `include/fhelium/<family>.h` | `FHELIUM_API` declaration with a `///` comment stating the computed equations, shapes, domains, representations, and ranges |
| `ops/<family>/backends.h` | `cpu` and `gpu` backend entry points with the public signature |
| `ops/<family>/api.cpp` | Public definition that routes with `FHELIUM_DISPATCH_CPU_GPU`, `FHELIUM_DISPATCH_GPU`, or `FHELIUM_DISPATCH_CUDA` (`ops/common/backend_dispatch.h`) |
| `ops/<family>/{cpu,cuda,hip}/` | Backend definitions and kernels, compiled only for the enabled backends |
| `torchops/<family>_ops.cpp` | `m.def` schema and an `m.impl` for each supported dispatch key |

## Compositions

Whole operations (`ops/operations`) and key generation (`keys`) call public functions only and run on any device that their primitives support. They take numerical operands as table structs (`fhelium/operation_tables.h`) whose fields are documented with each function, and options as typed structs (`KeySwitchOptions`, `KeySwitchKeyLayout`). Their Torch schemas pass each table struct as one positional `Tensor?[] tables` argument in the field order of its `visit_tables` definition, Galois automorphisms as `Tensor?[] galois` with three slots per step, and the option structs as scalar fields. The whole operations are registered as `CompositeExplicitAutograd` kernels in `torchops/operation_ops.cpp`, so tracing records each as one operator with its fake implementation; key generation is registered as `CompositeImplicitAutograd` kernels in `torchops/key_ops.cpp`. The table builders and the `table_slot_names` layout operator are registered in `torchops/operation_table_ops.cpp`.

Randomized compositions read the stream `seed` at its current nonce and use consecutive nonces internally without modifying `seed`; each function documents how many nonces it uses, and the caller advances the stream by that count.

Code with policy decisions, such as default digit layouts or prime selection, belongs in `parameters` as host functions that return plain values. Compositions receive those values as arguments rather than choosing them.

## Adding code

| New code | Location |
| --- | --- |
| Device kernel for an existing operation | `ops/<family>/{cpu,cuda,hip}/` and the router in `ops/<family>/api.cpp` |
| New primitive | All five files of the primitive structure |
| Operation composed from primitives | `ops/operations/`, declared in `include/fhelium/<family>_operations.h`, registered in `torchops/operation_ops.cpp` |
| Parameter or prime policy | `parameters/`, declared in `include/fhelium/ckks_parameters.h` or `rns_parameters.h` |
| Precomputed table | `tables/`, declared in `include/fhelium/<family>_tables.h` |
| Key material | `keys/`, declared in `include/fhelium/ckks_keys.h`, registered in `torchops/key_ops.cpp` |

`parameters/prime_catalog_data.inc` is generated by `scripts/generate_prime_catalog.py`. After a schema change, regenerate the typed Python wrappers in `fhelium/native/wrapper` with `scripts/generate_native_wrappers.py`. The C++ API reference is generated from the `///` comments of the headers listed in `scripts/generate_cpp_api_docs.py`, which fails when a registered function lacks a documented declaration.

## Build

`CMakeLists.txt` builds libfhelium from `ops`, `parameters`, `tables`, and `keys` with `add_fhelium_library` (`cmake/FHEliumNativeTargets.cmake`), which compiles every `.cpp` file under these directories and the `cpu`, `cuda`, and `hip` subdirectories of the enabled backends. A new source file in an existing directory needs no CMake change; a new top-level source directory is added to the `add_fhelium_library` call. `torchops` builds the `_ops` module against libfhelium, and `runtime` builds the device inspection modules.
