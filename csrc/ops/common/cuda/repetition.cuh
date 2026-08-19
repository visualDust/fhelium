#pragma once

#include <torch/torch.h>

// Private compile-time kernel specialization. Public operator schemas are
// layout-specific and never expose this enum as a runtime mode.
enum class RepetitionLayout : int { kCyclic, kContiguous };

inline void check_compressed_rns_binary_3d(const torch::Tensor& lhs,
                                           const torch::Tensor& compressed_rhs,
                                           const char* operation) {
  TORCH_CHECK(
      lhs.dim() == 3 && compressed_rhs.dim() == 3,
      operation,
      " requires canonical lhs [batch, limb, coefficient] and compressed_rhs "
      "[batch, limb, unique] operands");
  TORCH_CHECK(lhs.size(1) == compressed_rhs.size(1),
              operation,
              " operand limb counts differ: ",
              lhs.size(1),
              " vs ",
              compressed_rhs.size(1));
  TORCH_CHECK(lhs.scalar_type() == compressed_rhs.scalar_type(),
              operation,
              " operand dtypes differ: ",
              lhs.scalar_type(),
              " vs ",
              compressed_rhs.scalar_type());
  TORCH_CHECK(lhs.device() == compressed_rhs.device(),
              operation,
              " operand devices differ: ",
              lhs.device(),
              " vs ",
              compressed_rhs.device());
  TORCH_CHECK(
      compressed_rhs.size(0) == lhs.size(0) || compressed_rhs.size(0) == 1,
      operation,
      " operand batch counts differ: ",
      lhs.size(0),
      " vs ",
      compressed_rhs.size(0));
  const auto ring_dimension = lhs.size(2);
  const auto unique_count = compressed_rhs.size(2);
  TORCH_CHECK(
      ring_dimension > 0 && (ring_dimension & (ring_dimension - 1)) == 0,
      operation,
      " ring dimension must be a positive power of two: ",
      ring_dimension);
  TORCH_CHECK(unique_count > 0 && unique_count < ring_dimension,
              operation,
              " compressed unique count must be positive and smaller than "
              "ring dimension: ",
              unique_count,
              " vs ",
              ring_dimension);
  TORCH_CHECK((unique_count & (unique_count - 1)) == 0,
              operation,
              " compressed unique count must be a power of two: ",
              unique_count);
  TORCH_CHECK(ring_dimension % unique_count == 0,
              operation,
              " compressed unique count must divide ring dimension: ",
              unique_count,
              " does not divide ",
              ring_dimension);
}

template <RepetitionLayout layout>
__device__ __forceinline__ int repeated_rhs_index(int coefficient,
                                                  int unique_mask,
                                                  int repeat_shift) {
  if constexpr (layout == RepetitionLayout::kCyclic) {
    return coefficient & unique_mask;
  }
  return coefficient >> repeat_shift;
}

inline int compressed_repeat_shift(int64_t ring_dimension,
                                   int64_t unique_count) {
  int shift = 0;
  for (int64_t repeat_count = ring_dimension / unique_count; repeat_count > 1;
       repeat_count >>= 1) {
    ++shift;
  }
  return shift;
}
