#pragma once

// Host modular arithmetic for table construction. Every modulus is below
// 2^63; products use a 128-bit intermediate.

#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace fhelium::detail_host {

inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
#if defined(_MSC_VER) && !defined(__clang__)
  uint64_t high = 0;
  const uint64_t low = _umul128(a % m, b % m, &high);
  uint64_t remainder = 0;
  (void)_udiv128(high, low, m, &remainder);
  return remainder;
#else
  return static_cast<uint64_t>(static_cast<unsigned __int128>(a % m) *
                               (b % m) % m);
#endif
}

inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t m) {
  a %= m;
  b %= m;
  return a >= m - b ? a - (m - b) : a + b;
}

inline uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t m) {
  a %= m;
  b %= m;
  return a >= b ? a - b : a + (m - b);
}

inline uint64_t pow_mod(uint64_t base, uint64_t exponent, uint64_t m) {
  uint64_t result = 1 % m;
  base %= m;
  while (exponent != 0) {
    if (exponent & 1) {
      result = mul_mod(result, base, m);
    }
    base = mul_mod(base, base, m);
    exponent >>= 1;
  }
  return result;
}

// Inverse of a modulo m for gcd(a, m) = 1 (extended Euclid).
inline uint64_t inv_mod(uint64_t a, uint64_t m) {
  int64_t t = 0;
  int64_t next_t = 1;
  uint64_t r = m;
  uint64_t next_r = a % m;
  while (next_r != 0) {
    const uint64_t quotient = r / next_r;
    const int64_t updated_t = t - static_cast<int64_t>(quotient) * next_t;
    t = next_t;
    next_t = updated_t;
    const uint64_t updated_r = r - quotient * next_r;
    r = next_r;
    next_r = updated_r;
  }
  TORCH_CHECK_VALUE(r == 1, "value ", a, " is not invertible modulo ", m);
  return t < 0 ? static_cast<uint64_t>(t + static_cast<int64_t>(m))
               : static_cast<uint64_t>(t);
}

// 2^bits mod m.
inline uint64_t power_of_two_mod(int64_t bits, uint64_t m) {
  return pow_mod(2, static_cast<uint64_t>(bits), m);
}

// Product of factors modulo m.
inline uint64_t product_mod(const std::vector<uint64_t>& factors,
                            uint64_t m) {
  uint64_t result = 1 % m;
  for (const uint64_t factor : factors) {
    result = mul_mod(result, factor, m);
  }
  return result;
}

// floor(prod(factors) / 2) modulo m for an odd product and odd m.
inline uint64_t half_product_mod(const std::vector<uint64_t>& factors,
                                 uint64_t m) {
  const uint64_t product = product_mod(factors, m);
  return mul_mod(sub_mod(product, 1, m), inv_mod(2, m), m);
}

// Mixed-radix digits of x given its residues modulo moduli (Garner).
inline std::vector<uint64_t> mixed_radix_digits(
    const std::vector<uint64_t>& residues,
    const std::vector<uint64_t>& moduli) {
  std::vector<uint64_t> digits(moduli.size());
  for (size_t i = 0; i < moduli.size(); ++i) {
    const uint64_t q = moduli[i];
    uint64_t value = residues[i] % q;
    uint64_t radix = 1 % q;
    uint64_t accumulated = 0;
    for (size_t j = 0; j < i; ++j) {
      accumulated = add_mod(accumulated, mul_mod(digits[j], radix, q), q);
      radix = mul_mod(radix, moduli[j], q);
    }
    digits[i] = mul_mod(sub_mod(value, accumulated, q), inv_mod(radix, q), q);
  }
  return digits;
}

}  // namespace fhelium::detail_host
