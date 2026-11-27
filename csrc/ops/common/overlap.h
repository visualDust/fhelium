#pragma once

#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>

#include <cstdint>
#include <utility>

namespace fhelium::detail {

// Byte interval [first, last) spanned by the elements of a nonempty tensor.
inline std::pair<uintptr_t, uintptr_t> byte_extent(const at::Tensor& tensor) {
  int64_t low = 0;
  int64_t high = 0;
  for (int64_t axis = 0; axis < tensor.dim(); ++axis) {
    const int64_t reach = (tensor.size(axis) - 1) * tensor.stride(axis);
    (reach < 0 ? low : high) += reach;
  }
  const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  const auto item = static_cast<int64_t>(tensor.element_size());
  return {base + low * item, base + (high + 1) * item};
}

// True when no element of `a` shares memory with an element of `b`. Views of
// one storage whose overlap status is unknown are disjoint when their byte
// extents are.
inline bool storage_disjoint(const at::Tensor& a, const at::Tensor& b) {
  if (!a.defined() || !b.defined() || a.numel() == 0 || b.numel() == 0 ||
      !a.is_alias_of(b)) {
    return true;
  }
  const auto status = at::get_overlap_status(a, b);
  if (status != at::MemOverlapStatus::TooHard) {
    return status == at::MemOverlapStatus::No;
  }
  const auto [a_first, a_last] = byte_extent(a);
  const auto [b_first, b_last] = byte_extent(b);
  return a_last <= b_first || b_last <= a_first;
}

// True when `a` and `b` view the same elements in the same order.
inline bool same_elements(const at::Tensor& a, const at::Tensor& b) {
  return a.is_same(b) ||
         (a.defined() && b.defined() && a.is_alias_of(b) &&
          a.data_ptr() == b.data_ptr() && a.sizes() == b.sizes() &&
          a.strides() == b.strides() && a.dtype() == b.dtype());
}

}  // namespace fhelium::detail
