#pragma once

// Compiled FHElium artifacts.
//
// An artifact is a container (see fhelium/container.h) with the metadata keys
// `fhelium.format` ("fhelium-artifact"), `fhelium.schema_version` and
// `fhelium.manifest`, a JSON object with these fields:
//
//   identity   fhelium_version, native_schema_hash, torch_abi, cuda_runtime
//              of the producing library (see `io::native_identity`), and the
//              producing Triton version.
//   config     the CKKS configuration text; signatures: the exported call
//              signatures and the entry each one runs.
//   materials  {keys, tables, weights}: records {symbol, binding, shape,
//              dtype, ...}. `binding` is `caller` (bound with `bind`),
//              `embedded` (stored as container tensor `tensor`, or with
//              `view: {sizes, strides, offset}` in elements as that view of
//              it; records naming one tensor share its storage), `built`
//              (output `output` of entry `entry`, normally `tables`) or
//              `prepare` (an output of a prepare entry). `device` ("cpu" or
//              "cuda", default "cuda") selects the host or the artifact's
//              device.
//   prepare    {name: {entry, inputs, outputs, per_call}}: entries whose
//              inputs are material symbols and whose outputs are the material
//              symbols they produce, such as encoded plaintexts of raw
//              weights. With `per_call`, inputs also name call arguments
//              ({argument: position}); the caller runs such an entry with
//              `refresh` when those arguments change.
//   entries    {name: entry}: Runner schedules (see fhelium/runner.h).
//   kernels    [{name, num_warps, warp_size, num_ctas, shared,
//              parameter_types, scratch: [global size, global alignment,
//              profile size, profile alignment], triton_version, images:
//              {<arch>: tensor, ptx: tensor}}], with images as uint8 tensors.
//   constants  [{tensor, sizes, strides, offset, device}]: link-time constant
//              Tensors, each a strided view of one stored tensor, on the host
//              for device "cpu" and otherwise on the artifact's device.
//
// An entry holds `strings` (a string table), `buffers` and `code` (names of
// int64 tensors), `int_registers`, `float_registers`, `inputs` ([kind, id]
// with kind buffer, int or float), `read_in_place` (input buffers that Graph
// segments read at their address), `output_buffers`, `outputs` (the result
// structure for the caller), `bindings` ([buffer, {material: symbol} or
// {constant: index}]), `graphs` (whether it has Graph segments), `callees`,
// `arena_bytes`, `output_bytes`, `working_set`, `groups` ([{symbol, size}]:
// the process group of each Runner group id, bound by symbol with
// `bind_group`, and the group size the entry was linked for) and `streams`
// (the role of each Runner stream id, or null without streams).
//
// `buffers` encodes, for each buffer in id order after a leading count:
// storage (string index), the size count or -1 followed by the sizes, the
// stride count or -1 followed by the strides, the dtype (string index of a
// Torch dtype name such as "torch.int32", or -1) and the device (string index
// or -1). `code` encodes the builder calls after a leading count: the method
// name (string index), the argument count and the arguments, each a tag and
// its payload:
//
//   0 None             1 int: value          2 float: IEEE-754 bits
//   3 bool: value      4 str: string index   5 list: count, then items
//   6 dtype: string    7 device: string      8 kernel: index into kernels
//   9 entry: string index of a callee entry name
//   10 memory format: string ("contiguous_format", "preserve_format",
//      "channels_last", "channels_last_3d")
//   11 layout: string ("strided", "sparse_coo", "_mkldnn")
//
// An artifact is device-agnostic within one device type: CUDA devices of
// buffers, operator arguments, launches and segments refer to the device it
// is loaded on.

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fhelium/runner.h"
#include "fhelium/torch_library.h"
#include "fhelium/values.h"

namespace fhelium {

// A recorded call signature: the entry that executes it and the value type
// and metadata of each result, in entry output order, without tensors.
struct ArtifactSignature {
  std::string name;
  std::string entry;
  std::vector<io::Value> results;
};

class FHELIUM_TORCH_API Artifact : public torch::CustomClassHolder {
 public:
  // Reads the artifact at `path` for execution on `device`. The identity of
  // the library that wrote it must match the loaded library's. Constants and
  // embedded materials are placed on `device`, and each kernel selects the
  // image of the device's architecture, or its PTX.
  Artifact(const std::string& path, c10::Device device);
  ~Artifact() override;

  std::string manifest() const;
  std::vector<std::string> entries() const;
  std::vector<ArtifactSignature> signatures() const;
  // Binds material `symbol`; `value` must match the declared shape and dtype
  // and lie on the record's device (see `device` above).
  void bind(const std::string& symbol, const at::Tensor& value);
  // Binds key value `key` (see `io::load_value`) to every caller material
  // whose record names the value type as its role and whose layout fields
  // equal the value's metadata fields. A compact key switching key is expanded
  // first when the record declares the expanded shape. Returns the bound
  // symbols, none for a key the artifact does not use.
  std::vector<std::string> bind_key(const io::Value& key);
  at::Tensor material(const std::string& symbol) const;
  // Binds each rounding-state material (records of kind `rounding_state`,
  // the random stream of stochastic plaintext rounding) to a new stream per
  // device: derived from `seed` and the device as fhelium.Engine(rng_seed=
  // seed) derives it, or seeded from the operating system's random source
  // without `seed`. `prepare` seeds unbound rounding states from the
  // operating system.
  void seed_rounding(std::optional<int64_t> seed);
  // Binds process group `symbol` of the entries to the c10d group registered
  // as `name`, or to `group`, for the Runners built afterwards.
  void bind_group(const std::string& symbol, const std::string& name);
  void bind_group(const std::string& symbol,
                  c10::intrusive_ptr<c10d::ProcessGroup> group);
  // Releases the groups of every Runner the artifact built (see
  // Runner::release_groups) and forgets the group bindings.
  void release_groups();
  // Caller materials not yet bound, and built and prepared materials before
  // `prepare`.
  std::vector<std::string> unbound() const;
  // Runs the entries of built materials and every prepare entry without
  // `per_call` once and binds their outputs. Later calls return immediately.
  void prepare();
  // Runs prepare entry `name` with `inputs` in place of its input materials
  // and writes its outputs into the bound materials.
  void refresh(const std::string& name, const std::vector<at::Tensor>& inputs);
  // A finalized Runner of `entry` with its materials and constants bound.
  // `graphs` is the segment mode (off, on or auto); `arena` holds its arena
  // buffers instead of a Runner-owned region. Called entries run on Runners
  // that this artifact keeps, with private regions.
  c10::intrusive_ptr<exec::Runner> runner(
      const std::string& entry,
      const std::optional<c10::intrusive_ptr<exec::Arena>>& arena,
      const std::string& graphs,
      bool profile_segments);

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace fhelium
