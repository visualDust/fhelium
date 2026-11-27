#pragma once

// Single-file tensor containers in the safetensors layout.
//
// A container file holds an 8-byte little-endian header length, a JSON header
// and the raw bytes of its tensors. The header maps each tensor name to its
// dtype, shape and [begin, end) byte range in the data section, and the
// reserved key `__metadata__` to a string-to-string map. Writers order the
// tensors by decreasing element alignment and then by name, pad the header
// with spaces to a multiple of 8 bytes and escape strings as the safetensors
// reference implementation does. Metadata keys are written in sorted order;
// the reference implementation writes them in an unspecified order and
// otherwise produces the same bytes for the same tensors and metadata. FHElium
// value files and compiled artifacts are containers whose metadata names their
// format.

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "fhelium/torch_library.h"

namespace fhelium::io {

// Contents of one container: the `__metadata__` map and the tensors in file
// order.
struct Container {
  std::map<std::string, std::string> metadata;
  std::vector<std::pair<std::string, at::Tensor>> tensors;
};

// Writes `tensors` and `metadata` to `path` through a temporary file in the
// same directory that is renamed over `path` once complete, so readers never
// observe a partial file. The file is readable and writable by its owner only.
// Tensors on a device are copied to the host. complex64 is stored as the
// safetensors dtype C64 and complex128 as C128, which only FHElium reads.
// Without `overwrite`, an existing `path` is an error.
FHELIUM_TORCH_API void save_container(
    const std::string& path,
    const std::vector<std::pair<std::string, at::Tensor>>& tensors,
    const std::map<std::string, std::string>& metadata,
    bool overwrite);

// Reads a container. The tensors are views of one buffer on `device` that
// holds the data section: on the host the buffer the file was read into, on a
// device one copy of it.
FHELIUM_TORCH_API Container load_container(const std::string& path,
                                           const c10::Device& device);

// Reads the metadata and the tensor header JSON (name to dtype, shape and
// data_offsets, without `__metadata__`) without reading tensor data.
FHELIUM_TORCH_API std::pair<std::map<std::string, std::string>, std::string>
read_container_header(const std::string& path);

// JSON object describing the build of the loaded library: fhelium_version,
// native_schema_hash (SHA-256 of the sorted registered `fhelium_*` operator
// schemas joined by newlines), torch_abi (Torch version and C++11 ABI flag)
// and cuda_runtime (CUDA or HIP runtime version, or "none").
FHELIUM_TORCH_API std::string native_identity();

}  // namespace fhelium::io
