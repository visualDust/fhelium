# Deploy a compiled callable

`fhelium.compile.export` writes the linked execution of a compiled callable for a set of call signatures to one artifact file: planned Runner schedules, kernel images, link-time constants, the builder calls that produce its tables, and the materials it stores. `fhelium.compile.load` in Python and `fhelium::Artifact` in C++ run that file in another process without capture, compilation, linking, or Triton compilation. Keys are not stored in the artifact; the loading process supplies them.

## Prerequisites

The exporting and loading processes use the same FHElium native library and Triton version; the artifact records both, and loading an artifact with a different identity raises an error naming the differing field. The loading device has the CUDA architecture of the exporting device, or the artifact contains kernel images for it (`targets` below). The loading process has the evaluation keys of the computation, for example as value files written with `fhelium.save_value`.

## 1. Compile and export

```python
import torch
import fhelium as fh
from fhelium import compile as fc
from fhelium.eager import Engine

engine = Engine(fh.CkksConfig.parse(fh.Preset.slots8192_scale40_depth7))
secret = engine.create_secret_key(device="cuda")
public = engine.create_public_key(secret, device="cuda")
rotation = engine.create_rotation_key(1, secret, device="cuda")
weight = torch.linspace(0.1, 0.2, 16, dtype=torch.float64)
x = engine.encrypt_message(torch.linspace(-0.02, 0.03, 16), public, device="cuda")

@fc.compile
def layer(value):
    plain = engine.prepare_compressed_plaintext(weight, depth=value.depth, device=value.device)
    rotated = engine.add(value, engine.rotate_with_key(value, rotation))
    return engine.multiply_plaintext(engine.coefficient_domain_to_ntt_domain(rotated), plain)

fc.export(layer, "layer.fhelium", signatures=[(x,)])
fh.save_value(rotation, "key-rotation-1.fhelium")
```

`signatures` lists example calls, as positional-argument sequences or keyword mappings. Each one is prepared and stored as entry `call<i>` together with entry `call<i>.graphs`, the same schedule planned with CUDA Graph segments; export runs each example call once to record the value type and metadata of its results.

| Option | Effect |
| --- | --- |
| `weights="raw"` (default) | Plaintexts prepared inside the callable are stored as their raw inputs with a prepare entry that encodes them when the artifact is loaded. |
| `weights="embedded"` | The encoded plaintexts are stored; loading is faster and the file is larger. |
| `targets=("sm_90",)` | Adds kernel images compiled for further CUDA architectures; every kernel also stores the image of the exporting device and its PTX. |
| `overwrite=True` | Replaces an existing file. |

NTT, RNS, and key-switching tables are stored as the calls of registered builder operators that produce them, and the loader runs those calls once on its device. When the callable encodes plaintexts from its own arguments, such as weights passed per call, the prepare entry is marked `per_call`; the loaded callable runs it before a call whose argument Tensors differ from the previous call's by identity or version counter.

Schedules with copy streams, communication streams, events, and CPU-lane tasks export like other schedules. Export raises `fhelium.compile.ExportError` when a schedule calls Python, when an executable is lowered separately per runtime input layout, or when the callable uses Engines of different CKKS configurations.

## 2. Load and call in Python

```python
keys = [fh.load_value("key-rotation-1.fhelium", device="cuda")]
loaded = fc.load("layer.fhelium", keys=keys)
result = loaded(x)
```

`keys` supplies key objects, which are matched to the key descriptions recorded in the artifact by value type and metadata, or a mapping from the artifact's key symbols to keys; loading fails when a declared key has no match. A compact key read from a value file is expanded on its device before binding. `engine` supplies the Engine whose random stream rounds plaintext encoding during loading; without it, `load` creates an Engine for the recorded CKKS configuration. `device` loads the artifact onto another CUDA device, `graph` takes the values of the `graph` option of `fc.compile` (default `"auto"`), and `groups` maps the process-group symbols of an artifact with collectives to `torch.distributed` process groups. The default group is recorded as `world` and binds to the `torch.distributed` WORLD group without an entry in `groups`; `torch.distributed` is initialized before such an artifact is loaded. An artifact records a process rank only when the exported function read `torch.distributed.get_rank()` during capture or its reads could not be identified; loading it on another rank raises `RuntimeError`, and other artifacts load on every rank of the group.

Loading runs the artifact's `tables` entry and its prepare entries once. A call selects the stored signature whose argument conditions match and raises `SpecializationMiss` when none matches.

## 3. Load and call in C++

A C++ program links `FHElium::torch` as described in [Call the native library from C++](call-native-library-from-cpp.md) and loads the same file with `fhelium::Artifact` from `<fhelium/artifact.h>`:

```cpp
#include <fhelium/artifact.h>
#include <fhelium/values.h>

const torch::Device device(torch::kCUDA, 0);
auto artifact = c10::make_intrusive<fhelium::Artifact>("layer.fhelium", device);
for (const auto& path : key_paths)
  artifact->bind_key(fhelium::io::load_value(path, device));
artifact->prepare();
const auto signature = artifact->signatures().at(0);
auto runner = artifact->runner(signature.entry, std::nullopt, "auto", false);

const auto input = fhelium::io::load_value("input.fhelium", device);
auto result = signature.results.at(0);
result.tensors["data"] = runner->run({input.tensors.at("data")}, {}, {}).at(0);
fhelium::io::save_value("output.fhelium", result, false, true);
```

The constructor reads the file for one device and rejects files written by a library with a different FHElium version, operator schema hash, Torch ABI, or CUDA runtime. `bind_key` binds a key value to every declared key material whose role is the value type and whose layout (prime rows, digit rows, basis, domain, residue representation, and, for rotation keys, the step) equals the value's metadata; a compact key is expanded on the device first, and a key the artifact does not use binds nothing. `prepare` builds the recorded tables and the plaintexts of stored weights once. Plaintexts encoded while loading round with the artifact's rounding stream, which `prepare` seeds from the operating system's random source; `seed_rounding(seed)` before `prepare` derives it from `seed` as `fhelium.eager.Engine(rng_seed=seed)` derives its stream, so the load reproduces the plaintexts of a Python load with that Engine. `signatures()` lists each recorded call with its entry and the value type and metadata of its results, and `runner(entry, arena, graphs, profile)` returns a finalized Runner whose inputs are the argument Tensors in signature order. `refresh(name, inputs)` reruns a `per_call` prepare entry with new argument Tensors and writes its plaintexts in place. `bind_group(symbol, name)` or `bind_group(symbol, group)` binds a process-group symbol of an artifact with collectives to a registered c10d group name or a `c10d::ProcessGroup` before Runners are built.

The repository example [`examples/cpp/load_artifact.cpp`](https://github.com/VisualDust/fhelium/tree/main/examples/cpp/load_artifact.cpp) takes an optional `--seed SEED`, the artifact, an output path, the input value file, and the key value files, runs the first signature with automatic CUDA Graph selection, and writes the result as a value file that `fhelium.load_value` reads. [Example 15](../tutorial/export-and-load.md) writes a matching artifact, key file, and input.

## Verify the outcome

Decrypt the loaded callable's result with the client's secret key and compare it with the result of the compiled callable in the exporting process. [Compiled artifact format](../developer/compiled-artifact-format.md) describes the manifest, entries, and material bindings, and [Save and relink a Program](save-and-relink-program.md) covers Program files for applications that link materials themselves.
