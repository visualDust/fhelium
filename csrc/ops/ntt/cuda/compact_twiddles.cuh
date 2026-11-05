#pragma once

#include "../../common/montgomery.hpp"

// Compact tables are either [prime, N] Montgomery residues or int32
// [prime, N, 2] pairs (standard root, floor(root * 2^32 / q) word).
// Both multiply Montgomery payloads without changing their representation.
template <typename scalar_t>
__device__ __forceinline__ scalar_t compact_twiddle_multiply(
    const CudaTensorAccessor32<scalar_t, 2>& table, int row, int index,
    scalar_t value, scalar_t q_lo, scalar_t q_hi,
    scalar_t neg_inverse_lo, scalar_t neg_inverse_hi) {
  return montgomery_mul(table[row][index], value, q_lo, q_hi,
                        neg_inverse_lo, neg_inverse_hi);
}

__device__ __forceinline__ int32_t compact_twiddle_multiply(
    const CudaTensorAccessor32<int32_t, 3>& table, int row, int index,
    int32_t value, int32_t q_lo, int32_t q_hi,
    int32_t neg_inverse_lo, int32_t neg_inverse_hi) {
  const uint32_t root = static_cast<uint32_t>(table[row][index][0]);
  const uint32_t quotient = static_cast<uint32_t>(table[row][index][1]);
  const uint32_t q = static_cast<uint32_t>(q_lo) |
                     (static_cast<uint32_t>(q_hi) << 16);
  const uint32_t x = static_cast<uint32_t>(value);
  // q < 2^30, root < q, x < 2q. The quotient approximation is at most
  // one below floor(x*root/q), giving a nonnegative remainder below 2q.
  return static_cast<int32_t>(x * root - __umulhi(x, quotient) * q);
}

template <typename scalar_t, typename Function>
void with_compact_twiddles(const torch::Tensor& twiddles, Function&& function) {
  if constexpr (std::is_same_v<scalar_t, int32_t>) {
    if (twiddles.dim() == 3) {
      function(FHELIUM_CUDA_ACCESSOR32(twiddles, int32_t, 3));
      return;
    }
  }
  function(FHELIUM_CUDA_ACCESSOR32(twiddles, scalar_t, 2));
}
