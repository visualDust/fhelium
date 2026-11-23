#pragma once

// Conversions shared by the host table builders.

#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace fhelium::detail_tables {

inline int64_t montgomery_radix_bits(at::ScalarType dtype) {
  TORCH_CHECK_TYPE(dtype == at::kInt || dtype == at::kLong,
                   "RNS tables require an int32 or int64 dtype");
  return dtype == at::kInt ? 32 : 62;
}

inline std::vector<uint64_t> to_u64(at::IntArrayRef values) {
  std::vector<uint64_t> result;
  result.reserve(values.size());
  for (const int64_t value : values) {
    TORCH_CHECK_VALUE(value > 0, "moduli must be positive, got ", value);
    result.push_back(static_cast<uint64_t>(value));
  }
  return result;
}

// Odd moduli above two that fit the lazy range of the residue dtype.
inline void check_moduli(const std::vector<uint64_t>& moduli,
                         int64_t radix_bits,
                         const char* operation) {
  const uint64_t limit = uint64_t{1} << (radix_bits == 32 ? 30 : 60);
  for (const uint64_t modulus : moduli) {
    TORCH_CHECK_VALUE(modulus > 2 && (modulus & 1) != 0 && modulus < limit,
                      operation,
                      " requires odd moduli above two below 2^",
                      radix_bits == 32 ? 30 : 60, ", got ", modulus);
  }
}

template <typename Writer>
at::Tensor fill_tensor(at::IntArrayRef shape,
                       at::ScalarType dtype,
                       const Writer& writer) {
  auto out = at::empty(shape, at::TensorOptions().dtype(dtype));
  if (dtype == at::kInt) {
    writer(static_cast<int32_t*>(out.data_ptr()));
  } else {
    TORCH_CHECK_TYPE(dtype == at::kLong, "unsupported table dtype ", dtype);
    writer(static_cast<int64_t*>(out.data_ptr()));
  }
  return out;
}

inline at::Tensor to_tensor(const std::vector<uint64_t>& values,
                            at::ScalarType dtype) {
  return fill_tensor({static_cast<int64_t>(values.size())}, dtype,
                     [&](auto* data) {
                       using T = std::remove_pointer_t<decltype(data)>;
                       for (size_t i = 0; i < values.size(); ++i) {
                         data[i] = static_cast<T>(values[i]);
                       }
                     });
}

inline at::Tensor to_tensor(const std::vector<std::vector<uint64_t>>& rows,
                            at::ScalarType dtype,
                            int64_t columns) {
  return fill_tensor(
      {static_cast<int64_t>(rows.size()), columns}, dtype, [&](auto* data) {
        using T = std::remove_pointer_t<decltype(data)>;
        for (size_t r = 0; r < rows.size(); ++r) {
          for (int64_t c = 0; c < columns; ++c) {
            data[r * columns + c] = static_cast<T>(rows[r][c]);
          }
        }
      });
}

inline at::Tensor to_tensor(const std::vector<std::vector<uint64_t>>& rows,
                            at::ScalarType dtype) {
  return to_tensor(rows, dtype,
                   rows.empty() ? 0 : static_cast<int64_t>(rows[0].size()));
}

}  // namespace fhelium::detail_tables
