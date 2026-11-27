#pragma once

// Dtype names shared by the container, value-file and artifact codecs.

#include <c10/core/ScalarType.h>

#include <string>
#include <string_view>

namespace fhelium::io::detail {

// Scalar type of a safetensors dtype name such as "I64".
c10::ScalarType container_dtype(const std::string& name);
// Safetensors dtype name of a scalar type.
const char* container_dtype_name(c10::ScalarType type);
// Lowercase hexadecimal SHA-256 digest of `bytes`.
std::string sha256_hex(std::string_view bytes);

}  // namespace fhelium::io::detail
