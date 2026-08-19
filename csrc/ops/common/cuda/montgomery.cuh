#pragma once

#include <torch/torch.h>
#include <type_traits>

// ------------------------------------------------------------------
// mont scalar cuda kernels
// ------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ scalar_t
montgomery_mul(const scalar_t a,
               const scalar_t b,
               const scalar_t modulus_lo,
               const scalar_t modulus_hi,
               const scalar_t neg_inv_modulus_lo,
               const scalar_t neg_inv_modulus_hi) {
  // Masks.
  constexpr scalar_t one = 1;
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  constexpr scalar_t fb_mask = ((one << nbits) - one);
  constexpr scalar_t lb_mask = (one << half_nbits) - one;

  const scalar_t al = a & lb_mask;
  const scalar_t ah = a >> half_nbits;
  const scalar_t bl = b & lb_mask;
  const scalar_t bh = b >> half_nbits;

  const scalar_t alpha = ah * bh;
  const scalar_t beta = ah * bl + al * bh;
  const scalar_t gamma = al * bl;

  // s = xk mod R
  const scalar_t gammal = gamma & lb_mask;
  const scalar_t gammah = gamma >> half_nbits;
  const scalar_t betal = beta & lb_mask;
  const scalar_t betah = beta >> half_nbits;

  scalar_t upper = gammal * neg_inv_modulus_hi;
  upper = upper + (gammah + betal) * neg_inv_modulus_lo;
  upper = upper << half_nbits;
  scalar_t s = upper + gammal * neg_inv_modulus_lo;
  s = upper + gammal * neg_inv_modulus_lo;
  s = s & fb_mask;

  // t = x + sq
  // u = t/R
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * modulus_lo + sl * modulus_hi;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;

  scalar_t carry = (gamma + sl * modulus_lo) >> half_nbits;
  carry = (carry + betal + sqbl) >> half_nbits;

  return alpha + betah + sqbh + carry + sh * modulus_hi;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
canonicalize_montgomery_operand(const scalar_t value,
                                const scalar_t modulus_lo,
                                const scalar_t modulus_hi) {
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  const scalar_t modulus = modulus_lo + (modulus_hi << half_nbits);
  scalar_t canonical = value % modulus;
  if constexpr (std::is_signed_v<scalar_t>) {
    if (canonical < 0) canonical += modulus;
  }
  return canonical;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
canonicalize_lazy_montgomery_operand(const scalar_t value,
                                     const scalar_t modulus_lo,
                                     const scalar_t modulus_hi) {
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  const scalar_t modulus = modulus_lo + (modulus_hi << half_nbits);
  return value < modulus ? value : value - modulus;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
montgomery_mul_split(const scalar_t a,
                     const scalar_t b,
                     const scalar_t modulus_lo,
                     const scalar_t modulus_hi,
                     const scalar_t neg_inv_modulus_lo,
                     const scalar_t neg_inv_modulus_hi) {
  return montgomery_mul(
      canonicalize_montgomery_operand(a, modulus_lo, modulus_hi),
      canonicalize_montgomery_operand(b, modulus_lo, modulus_hi),
      modulus_lo,
      modulus_hi,
      neg_inv_modulus_lo,
      neg_inv_modulus_hi);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
canonicalize_lazy_residue(const scalar_t x, const scalar_t twice_modulus) {
  constexpr scalar_t one = 1;
  const scalar_t q = twice_modulus >> one;
  // Reduce the lazy interval [0, 2q) to the canonical interval [0, q).
  return (x < q) ? x : x - q;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t add_lazy_residues(
    const scalar_t a, const scalar_t b, const scalar_t twice_modulus) {
  // Add.
  const scalar_t aplusb = a + b;
  // Reduce the lazy interval [0, 2q) to the canonical interval [0, q).
  return (aplusb < twice_modulus) ? aplusb : aplusb - twice_modulus;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sub_lazy_residues(
    const scalar_t a, const scalar_t b, const scalar_t twice_modulus) {
  // Inputs are lazy representatives in [0, 2q), so their difference lies in
  // (-2q, 2q). One signed correction returns the same residue in [0, 2q).
  const scalar_t aminusb = a - b;
  if (aminusb < 0) return aminusb + twice_modulus;
  return (aminusb < twice_modulus) ? aminusb : aminusb - twice_modulus;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
montgomery_reduce(const scalar_t a,
                  const scalar_t modulus_lo,
                  const scalar_t modulus_hi,
                  const scalar_t neg_inv_modulus_lo,
                  const scalar_t neg_inv_modulus_hi) {
  // Masks.
  constexpr scalar_t one = 1;
  // nbits is set to 2 bits less than the full width for overflow safety
  constexpr scalar_t nbits = sizeof(scalar_t) * 8 - 2;
  constexpr scalar_t half_nbits = sizeof(scalar_t) * 4 - 1;
  // lb_mask: mask for the lower half (like 0xFFFF)
  constexpr scalar_t lb_mask = (one << half_nbits) - one;
  // fb_mask: mask to chop the result below 2^nbits to avoid overflow
  constexpr scalar_t fb_mask = ((one << nbits) - one);

  // s = (a * k) mod R, with k = -q^{-1} mod R
  const scalar_t xl = a & lb_mask;
  const scalar_t xh = a >> half_nbits;
  const scalar_t xkb = xh * neg_inv_modulus_lo + xl * neg_inv_modulus_hi;
  scalar_t s = (xkb << half_nbits) + xl * neg_inv_modulus_lo;
  s = s & fb_mask;

  // t = a + s * q, then u = t / R
  // Note that x gets erased in t/R operation if x < R.
  const scalar_t sl = s & lb_mask;
  const scalar_t sh = s >> half_nbits;
  const scalar_t sqb = sh * modulus_lo + sl * modulus_hi;
  const scalar_t sqbl = sqb & lb_mask;
  const scalar_t sqbh = sqb >> half_nbits;
  scalar_t carry = (a + sl * modulus_lo) >> half_nbits;
  carry = (carry + sqbl) >> half_nbits;

  // The final result is approximated by skipping an actual full a + s * q and
  // >> r division Assume we have satisfied the condition 4*q < R. Return the
  // calculated value directly without conditional subtraction.
  return sqbh + carry + sh * modulus_hi;
  // result within [0,2q)
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
center_residue(const scalar_t a, const scalar_t twice_modulus) {
  // Masks.
  constexpr scalar_t one = 1;
  const scalar_t q = twice_modulus >> one;
  const scalar_t q_half = q >> one;
  return (a <= q_half) ? a : a - q;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
shift_residue_positive(const scalar_t a, const scalar_t twice_modulus) {
  // Masks.
  constexpr scalar_t one = 1;
  const scalar_t q = twice_modulus >> one;
  return a + q;
}
