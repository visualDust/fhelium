# FHElium examples

Each numbered file demonstrates one usage model or method; the [tutorial catalog](../docs/tutorial/index.md) lists them with a walkthrough page each. Run them from the repository root with the Python/PyTorch environment used to build FHElium:

```bash
python examples/01_eager_basics.py
python examples/01_eager_basics.py --device cuda:0 --preset slots8192-scale40-depth7
```

Scripts with configurable workloads expose `--help`. Eager scripts accept `--preset` and `--device`. Examples 16–19 require CUDA, and the bootstrap example 09 requires CUDA and a substantially larger working set. Example 15 writes its files to a temporary directory, or to `--output-dir`, and prints the matching `cpp/load_artifact` command.

## Distributed examples

Examples 20–22 run with world size one and with one process per GPU; `dist.init` selects CUDA when available and CPU otherwise:

```bash
python examples/20_distributed_batch_inputs.py --batch-size 7
torchrun --standalone --nproc-per-node=2 examples/20_distributed_batch_inputs.py --batch-size 7
torchrun --standalone --nproc-per-node=2 examples/21_distributed_partial_results.py --size 8
torchrun --standalone --nproc-per-node=2 examples/22_distributed_rns_shards.py
```

Example 23 transforms collective IR without launching distributed execution.

## C++ examples

[`cpp/`](cpp) is a CMake project that links the installed `libfhelium` through `find_package(FHElium)`. Build it with the Python environment that provides FHElium and Torch:

```bash
cmake -S examples/cpp -B build/cpp-example \
  -DCMAKE_PREFIX_PATH="$(python -c 'import fhelium.native, torch.utils; print(fhelium.native.cmake_prefix_path + ";" + torch.utils.cmake_prefix_path)')"
cmake --build build/cpp-example
build/cpp-example/ckks_matrix_vector
```

| Program | Demonstrates |
| --- | --- |
| `ckks_matrix_vector` | An encrypted 8 × 8 matrix-vector product in C++: key generation, encoding and encryption, hybrid key-switching rotations, a weighted sum of encoded diagonals, rescale, and decryption, on CPU and on the first CUDA device when available |
| `runner_schedule` | An exec schedule built with the FHElium Runner: a host loop with a parity branch over `fhelium_rns_ops` operators |
| `dispatcher_operators` | Operator calls through `c10::Dispatcher` |
| `load_artifact [--seed SEED] ARTIFACT OUTPUT INPUT KEY...` | Loads a file written by `fhelium.compile.export`, binds the evaluation keys from value files, runs the first recorded signature on the input value file, and writes the result as a value file; with `--seed`, plaintexts encoded while loading round as in `fhelium.compile.load` with `Engine(rng_seed=SEED)` |

See [Call the native library from C++](../docs/how-to/call-native-library-from-cpp.md) and [Deploy a compiled callable](../docs/how-to/deploy-compiled-callable.md).

## Experimental scope

Example 24 remains last and opt-in. The bootstrap example 09 does not establish an application's encrypted input range. The multiparty example uses synthetic data and throwaway keys; its output protocols have no supported privacy or production security guarantee. Read the linked tutorials before running them.
