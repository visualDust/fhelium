#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

#include "../rns_parameters.h"

namespace fhelium::cpu {

// Chunk a flat parallel task space into roughly twice the worker count so
// small single-row transforms still spread work across every worker.
inline int64_t adaptive_grain(const int64_t elements) {
  const int64_t threads = std::max<int64_t>(at::get_num_threads(), 1);
  return std::max<int64_t>(1, elements / (threads * 2));
}

template <typename scalar_t>
struct MontgomeryConstants {
  scalar_t twice_modulus;
  scalar_t modulus_lo;
  scalar_t modulus_hi;
  scalar_t neg_inv_modulus_lo;
  scalar_t neg_inv_modulus_hi;
  scalar_t r2;
  scalar_t scaled_r2;
  scalar_t n_inverse_montgomery;
};

template <typename scalar_t>
C10_ALWAYS_INLINE MontgomeryConstants<scalar_t> load_constants(
    const scalar_t* parameters,
    const int64_t parameter_row_stride,
    const int64_t parameter_limb_stride,
    const int64_t limb) {
  const scalar_t* limb_parameters = parameters + limb * parameter_limb_stride;
  return MontgomeryConstants<scalar_t>{
      limb_parameters[RNS_PARAM_TWICE_MODULUS * parameter_row_stride],
      limb_parameters[RNS_PARAM_MODULUS_LO * parameter_row_stride],
      limb_parameters[RNS_PARAM_MODULUS_HI * parameter_row_stride],
      limb_parameters[RNS_PARAM_NEG_INV_MODULUS_LO * parameter_row_stride],
      limb_parameters[RNS_PARAM_NEG_INV_MODULUS_HI * parameter_row_stride],
      limb_parameters[RNS_PARAM_R2 * parameter_row_stride],
      limb_parameters[RNS_PARAM_SCALED_R2 * parameter_row_stride],
      limb_parameters[RNS_PARAM_N_INV_MONTGOMERY * parameter_row_stride],
  };
}

template <typename scalar_t>
C10_ALWAYS_INLINE MontgomeryConstants<scalar_t> load_constants(
    const torch::TensorAccessor<scalar_t, 2>& parameters, const int64_t limb) {
  return MontgomeryConstants<scalar_t>{
      parameters[RNS_PARAM_TWICE_MODULUS][limb],
      parameters[RNS_PARAM_MODULUS_LO][limb],
      parameters[RNS_PARAM_MODULUS_HI][limb],
      parameters[RNS_PARAM_NEG_INV_MODULUS_LO][limb],
      parameters[RNS_PARAM_NEG_INV_MODULUS_HI][limb],
      parameters[RNS_PARAM_R2][limb],
      parameters[RNS_PARAM_SCALED_R2][limb],
      parameters[RNS_PARAM_N_INV_MONTGOMERY][limb],
  };
}

template <typename scalar_t>
C10_ALWAYS_INLINE MontgomeryConstants<scalar_t> load_constants(
    const scalar_t* modulus_lo,
    const int64_t modulus_lo_stride,
    const scalar_t* modulus_hi,
    const int64_t modulus_hi_stride,
    const scalar_t* neg_inv_modulus_lo,
    const int64_t neg_inv_modulus_lo_stride,
    const scalar_t* neg_inv_modulus_hi,
    const int64_t neg_inv_modulus_hi_stride,
    const int64_t limb) {
  return MontgomeryConstants<scalar_t>{
      0,
      modulus_lo[limb * modulus_lo_stride],
      modulus_hi[limb * modulus_hi_stride],
      neg_inv_modulus_lo[limb * neg_inv_modulus_lo_stride],
      neg_inv_modulus_hi[limb * neg_inv_modulus_hi_stride],
      0,
      0,
      0,
  };
}

template <typename scalar_t>
C10_ALWAYS_INLINE MontgomeryConstants<scalar_t> load_constants(
    const torch::TensorAccessor<scalar_t, 1>& modulus_lo,
    const torch::TensorAccessor<scalar_t, 1>& modulus_hi,
    const torch::TensorAccessor<scalar_t, 1>& neg_inv_modulus_lo,
    const torch::TensorAccessor<scalar_t, 1>& neg_inv_modulus_hi,
    const int64_t limb) {
  return MontgomeryConstants<scalar_t>{
      0,
      modulus_lo[limb],
      modulus_hi[limb],
      neg_inv_modulus_lo[limb],
      neg_inv_modulus_hi[limb],
      0,
      0,
      0,
  };
}

template <typename scalar_t>
C10_ALWAYS_INLINE uint64_t
modulus(const MontgomeryConstants<scalar_t>& constants) {
  constexpr int64_t kHalfRadixBits = sizeof(scalar_t) * 4 - 1;
  return static_cast<uint64_t>(constants.modulus_lo) +
         (static_cast<uint64_t>(constants.modulus_hi) << kHalfRadixBits);
}

template <typename scalar_t>
C10_ALWAYS_INLINE uint64_t
neg_inverse(const MontgomeryConstants<scalar_t>& constants) {
  constexpr int64_t kHalfRadixBits = sizeof(scalar_t) * 4 - 1;
  return static_cast<uint64_t>(constants.neg_inv_modulus_lo) +
         (static_cast<uint64_t>(constants.neg_inv_modulus_hi)
          << kHalfRadixBits);
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t
multiply(const scalar_t lhs,
         const scalar_t rhs,
         const MontgomeryConstants<scalar_t>& constants) {
  constexpr int64_t kRadixBits = sizeof(scalar_t) * 8 - 2;
  const uint64_t q = modulus(constants);
#if defined(_MSC_VER) && defined(_M_X64)
  // MSVC x64 exposes the same modulo-2^128 product and carry operations used
  // by the unsigned __int128 implementation below.
  uint64_t product_hi = 0;
  const uint64_t product_lo = _umul128(
      static_cast<uint64_t>(lhs), static_cast<uint64_t>(rhs), &product_hi);
  const uint64_t radix_mask = (uint64_t{1} << kRadixBits) - 1;
  const uint64_t correction =
      (product_lo * neg_inverse(constants)) & radix_mask;
  uint64_t correction_product_hi = 0;
  const uint64_t correction_product_lo =
      _umul128(correction, q, &correction_product_hi);
  uint64_t sum_lo = 0;
  const unsigned char carry =
      _addcarry_u64(0, product_lo, correction_product_lo, &sum_lo);
  const uint64_t sum_hi = product_hi + correction_product_hi + carry;
  const uint64_t result =
      (sum_lo >> kRadixBits) | (sum_hi << (64 - kRadixBits));
#else
  const unsigned __int128 product =
      static_cast<unsigned __int128>(static_cast<uint64_t>(lhs)) *
      static_cast<uint64_t>(rhs);
  const uint64_t radix_mask = (uint64_t{1} << kRadixBits) - 1;
  const uint64_t correction = static_cast<uint64_t>(
      (static_cast<unsigned __int128>(static_cast<uint64_t>(product)) *
       neg_inverse(constants)) &
      radix_mask);
  const unsigned __int128 reduced =
      (product + static_cast<unsigned __int128>(correction) * q) >> kRadixBits;
  const uint64_t result = static_cast<uint64_t>(reduced);
#endif
  return static_cast<scalar_t>(result < q ? result : result - q);
}

// Canonicalize signed or lazy operands before split-word Montgomery
// multiplication so CPU and CUDA implement the same residue map.
template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t canonicalize_operand(
    const scalar_t value, const MontgomeryConstants<scalar_t>& constants) {
  const int64_t q = static_cast<int64_t>(modulus(constants));
  int64_t canonical = static_cast<int64_t>(value) % q;
  if (canonical < 0) canonical += q;
  return static_cast<scalar_t>(canonical);
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t canonicalize_lazy_operand(
    const scalar_t value, const MontgomeryConstants<scalar_t>& constants) {
  const scalar_t q = static_cast<scalar_t>(modulus(constants));
  return value < q ? value : value - q;
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t
multiply_split(const scalar_t a,
               const scalar_t b,
               const MontgomeryConstants<scalar_t>& constants) {
  return multiply(canonicalize_operand(a, constants),
                  canonicalize_operand(b, constants),
                  constants);
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t
reduce(const scalar_t value, const MontgomeryConstants<scalar_t>& constants) {
  return multiply(value, static_cast<scalar_t>(1), constants);
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t add_lazy(const scalar_t lhs,
                                    const scalar_t rhs,
                                    const scalar_t twice_modulus) {
  const int64_t sum = static_cast<int64_t>(lhs) + rhs;
  return static_cast<scalar_t>(sum < twice_modulus ? sum : sum - twice_modulus);
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t subtract_lazy(const scalar_t lhs,
                                         const scalar_t rhs,
                                         const scalar_t twice_modulus) {
  const int64_t difference = static_cast<int64_t>(lhs) + twice_modulus - rhs;
  return static_cast<scalar_t>(
      difference < twice_modulus ? difference : difference - twice_modulus);
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t canonicalize(const scalar_t value,
                                        const scalar_t twice_modulus) {
  const scalar_t q = twice_modulus >> 1;
  return value < q ? value : value - q;
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t center(const scalar_t value,
                                  const scalar_t twice_modulus) {
  const scalar_t q = twice_modulus >> 1;
  return value <= (q >> 1) ? value : value - q;
}

template <typename scalar_t>
C10_ALWAYS_INLINE scalar_t shift_positive(const scalar_t value,
                                          const scalar_t twice_modulus) {
  return value + (twice_modulus >> 1);
}

}  // namespace fhelium::cpu
