# Compiled artifact format

A compiled artifact stores the linked executables of a compiled callable as Runner schedules that `libfhelium` loads without Python. `fhelium.compile.export` writes artifacts, `fhelium.compile.load` and C++ `fhelium::Artifact` read them, and the persistent compile cache stores its entries in the same format. The format is specified in `csrc/include/fhelium/artifact.h` and `csrc/include/fhelium/container.h`; this page summarizes it and the Python components that produce it.

## Container

An artifact is a container file in the safetensors layout: an 8-byte little-endian header length, a JSON header that maps each tensor name to its dtype, shape, and `[begin, end)` byte range, and the raw tensor bytes. The reserved header key `__metadata__` holds a string-to-string map. Writers order tensors by decreasing element alignment and then by name, pad the header to a multiple of 8 bytes, and write metadata keys in sorted order, so the same tensors and metadata produce the same bytes. Files are written through a temporary file that is renamed over the destination and are readable by their owner only. complex64 and complex128 use the dtypes `C64` and `C128`, which only FHElium reads. FHElium value files use the same container with their own format metadata.

An artifact's metadata has three keys: `fhelium.format` (`"fhelium-artifact"`), `fhelium.schema_version` (`1`), and `fhelium.manifest`, a JSON object.

## Manifest

| Field | Content |
| --- | --- |
| `identity` | `fhelium_version`, `native_schema_hash` (SHA-256 of the sorted registered `fhelium_*` operator schemas), `torch_abi` (Torch version and C++11 ABI flag), `cuda_runtime`, and the producing Triton version. A loader rejects a file whose identity differs from its own. |
| `config` | The CKKS configuration text, or `null` when the schedules read no tables. |
| `signatures` | The exported call signatures, the entry each one runs, its material working set, and, as `results`, the value type and metadata of each output in output order. |
| `materials` | `{keys, tables, weights}`: records `{symbol, binding, shape, dtype, device, ...}`. |
| `prepare` | `{name: {entry, inputs, outputs, per_call}}`: entries that produce materials from other materials. |
| `entries` | `{name: entry}`: Runner schedules. |
| `kernels` | Kernel records with launch parameters, scratch sizes, the Triton version, and `images` (`{<arch>: tensor, ptx: tensor}` as uint8 tensors). |
| `constants` | `[{tensor, sizes, strides, offset, device}]`: link-time constant Tensors, each a strided view of one stored tensor. |
| `rank` | The process rank the schedules were captured for when the source reads `torch.distributed.get_rank()` or its reads cannot be identified, otherwise `null`. `fc.load` rejects a file with a different rank. |

## Material bindings

Each material record names how the loader obtains its Tensor:

| Binding | Source | Examples |
| --- | --- | --- |
| `caller` | Bound by the loading process (`bind_key`, the rounding seed) | Evaluation keys with their key description; the rounding seed of plaintext encoding |
| `built` | Output `output` of entry `entry`, normally `tables` | NTT, RNS, and key-switching tables produced by registered builder operators |
| `prepare` | Output of a prepare entry | Plaintexts encoded at load from stored raw weights (`weights="raw"`) |
| `embedded` | Stored container tensor `tensor`, optionally as a view `{sizes, strides, offset}` | Encoded plaintexts with `weights="embedded"`, tables produced by Python code, plaintexts encoded before capture |

Records that name one stored tensor share its storage. `device` (`"cpu"` or `"cuda"`, default `"cuda"`) selects the host or the artifact's device. A prepare entry marked `per_call` also names call arguments (`{argument: position}`) among its inputs; the caller reruns it with `refresh` when those arguments change.

## Entries

`fhelium.backend._artifact.export_executable` stores one linked executable as two entries: its planned schedule `<name>` and the same schedule planned with CUDA Graph segments `<name>.graphs`. Called schedules become entries of their own. An entry holds a string table, the names of two int64 tensors `buffers` and `code`, the integer and float register counts, `inputs` and `outputs`, `output_buffers`, `bindings` (`[buffer, {material: symbol}]` or `[buffer, {constant: index}]`), `graphs`, `callees`, `arena_bytes`, `output_bytes`, `working_set`, `groups` (the process-group symbol and size of each Runner group id; the default group is named `world`), and `streams` (the role of each Runner stream id, such as `h2d`, `d2h`, or `comm`, or null without streams). Events and CPU-lane tasks of a planned schedule are builder calls of the entry.

`buffers` encodes each buffer's storage class, sizes, strides, dtype, and device as string-table indices and integers. `code` encodes the Runner builder calls: the method name, the argument count, and each argument as a tag and payload, where the tags cover `None`, int, float (IEEE-754 bits), bool, string, list, dtype, device, kernel index, callee entry, memory format, and layout. Loading replays the builder calls into a `fhelium::exec::Runner`, so an entry reproduces the Runner that linking built. CUDA devices in an entry refer to the device the artifact is loaded on.

`fhelium.native.artifact.ArtifactWriter` encodes the entries, stores each kernel as the cubin of the exporting device, the cubins of the requested `targets`, and PTX, and raises `ExportError` for schedules with `python_call` instructions. Material and state buffers are stored by symbol; a state buffer is named `state.<requirement>.<field>` after the resource requirement whose live value owns it.

## Loading

`fhelium::Artifact` reads the container onto one device, checks the identity, binds caller materials and process groups (`bind_group`), seeds the rounding stream (`seed_rounding`), runs the `tables` entry and the prepare entries once in `prepare()`, and builds Runners for entries on request. Tables with equal builder calls are shared across artifacts loaded in one process. The Python loader `fhelium.compile.load` wraps the same class: it matches keys, creates an Engine of the recorded configuration for the rounding stream when none is supplied, binds process groups, and returns a `LoadedFunction` that selects the stored signature matching each call.

## Compile cache entries

`fc.compile(..., cache=directory)` stores each prepared specialization as an artifact. A cache entry additionally stores the captured Program and records every relocated material, including keys, tables, and closure plaintexts, as `caller` with the path by which the current source provides it, so a cache hit binds the current Tensors of the source. An entry is keyed by a source key, computed from the function source within the statically checked subset, or by a trace key, computed from the captured Program; both include the call signature, the layouts of materials and supplied keys, and the execution environment. An exported artifact stores all materials other than keys and does not depend on the source that produced it. [Program files and the compile cache](compilation-persistence.md) describes the cache and Program files.
