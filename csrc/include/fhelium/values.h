#pragma once

// FHElium value files: one ciphertext, plaintext or key per container.
//
// A value is a tensor bundle with metadata. `type` names the value kind
// (Ciphertext, Plaintext, CompressedPlaintext, SecretKey, PublicKey,
// RelinearizationKey, RotationKey, ConjugationKey, EncapsulationKey and the
// other key kinds of `fhelium.values`); `metadata` holds the value's fields as
// a JSON object, such as `depth`, `scale`, `prime_ids`, `polynomial_domain`,
// `residue_representation` and, for key-switching keys, `digit_prime_ids` and
// `rotation_step`; `tensors` holds its payloads by name, `data` for the
// residues of a ciphertext or key. The file stores the container metadata
// keys `fhelium.format` ("fhelium-value"), `fhelium.schema_version` and
// `fhelium.manifest`, a JSON object with the value type, its metadata, the
// shape and dtype of each tensor, and their byte count. Complex tensors are
// stored as their real and imaginary parts.

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "fhelium/torch_library.h"

namespace fhelium::io {

struct FHELIUM_TORCH_API Value {
  std::string type;
  std::string metadata = "{}";  // JSON object
  std::map<std::string, at::Tensor> tensors;

  bool has(const std::string& field) const;
  int64_t integer(const std::string& field) const;
  double number(const std::string& field) const;
  std::string string(const std::string& field) const;
  std::vector<int64_t> integers(const std::string& field) const;
  // The rows of a list of integer lists, such as `digit_prime_ids`.
  std::vector<std::vector<int64_t>> integer_rows(
      const std::string& field) const;
};

// Reads a value file and places its tensors on `device`.
FHELIUM_TORCH_API Value load_value(const std::string& path,
                                   const c10::Device& device);

// Writes `value` to `path` (see `save_container`). A SecretKey requires
// `allow_secret`; the file stores the key data unencrypted.
FHELIUM_TORCH_API void save_value(const std::string& path,
                                  const Value& value,
                                  bool allow_secret,
                                  bool overwrite);

}  // namespace fhelium::io
