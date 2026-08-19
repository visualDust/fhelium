#pragma once

// Radix-specific cyclic butterflies used by the genuine power-of-two radix
// family. These routines do not iterate over the global radix-2 stage
// algorithm. A radix-R negacyclic digit first applies its beta^k outer twist
// and then evaluates one cyclic R-point NTT with the fixed primitive R-th root.

template <typename scalar_t>
__device__ __forceinline__ scalar_t power_of_two_radix_add(
    const scalar_t a, const scalar_t b, const scalar_t twice_modulus) {
  const scalar_t sum = a + b;
  return (sum < twice_modulus) ? sum : sum - twice_modulus;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t power_of_two_radix_sub(
    const scalar_t a, const scalar_t b, const scalar_t twice_modulus) {
  const scalar_t difference = a + twice_modulus - b;
  return (difference < twice_modulus) ? difference : difference - twice_modulus;
}

template <int BIT_COUNT>
__device__ __forceinline__ int power_of_two_radix_bit_reverse(const int value) {
  int reversed = 0;
#pragma unroll
  for (int bit = 0; bit < BIT_COUNT; ++bit) {
    reversed = (reversed << 1) | ((value >> bit) & 1);
  }
  return reversed;
}

template <typename scalar_t>
__device__ __forceinline__ void power_of_two_radix_cyclic_radix4(
    scalar_t values[4],
    const scalar_t primitive_fourth_root,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  // Positive-exponent four-point NTT:
  //   Y_m = sum_k X_k * zeta_4^(m*k).
  // This uses one multiplication by zeta_4, not two generic radix-2 stages.
  const scalar_t even_sum =
      power_of_two_radix_add(values[0], values[2], twice_modulus);
  const scalar_t even_difference =
      power_of_two_radix_sub(values[0], values[2], twice_modulus);
  const scalar_t odd_sum =
      power_of_two_radix_add(values[1], values[3], twice_modulus);
  const scalar_t odd_difference =
      power_of_two_radix_sub(values[1], values[3], twice_modulus);
  const scalar_t rotated_odd_difference = montgomery_mul(primitive_fourth_root,
                                                         odd_difference,
                                                         modulus_lo,
                                                         modulus_hi,
                                                         neg_inv_modulus_lo,
                                                         neg_inv_modulus_hi);

  values[0] = power_of_two_radix_add(even_sum, odd_sum, twice_modulus);
  values[2] = power_of_two_radix_sub(even_sum, odd_sum, twice_modulus);
  values[1] = power_of_two_radix_add(
      even_difference, rotated_odd_difference, twice_modulus);
  values[3] = power_of_two_radix_sub(
      even_difference, rotated_odd_difference, twice_modulus);
}

template <typename scalar_t>
__device__ __forceinline__ void power_of_two_radix_cyclic_radix16(
    scalar_t values[16],
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int root_row,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  // A dedicated radix-16 Cooley--Tukey butterfly factored as 4x4. The first
  // pass transforms k = r + 4s, the fixed radix-16 twiddle matrix couples the
  // two dimensions, and the second pass produces m = u + 4t. This is one
  // radix-16 digit, not a loop over four global radix-2 stages.
  const scalar_t primitive_fourth_root = radix_root_powers_acc[root_row][4];
  scalar_t work[16];

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    scalar_t column[4] = {
        values[r], values[r + 4], values[r + 8], values[r + 12]};
    power_of_two_radix_cyclic_radix4(column,
                                     primitive_fourth_root,
                                     twice_modulus,
                                     modulus_lo,
                                     modulus_hi,
                                     neg_inv_modulus_lo,
                                     neg_inv_modulus_hi);
#pragma unroll
    for (int u = 0; u < 4; ++u) work[r * 4 + u] = column[u];
  }

#pragma unroll
  for (int u = 0; u < 4; ++u) {
    scalar_t row[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      scalar_t value = work[r * 4 + u];
      const int exponent = u * r;
      if (exponent != 0) {
        value = montgomery_mul(radix_root_powers_acc[root_row][exponent],
                               value,
                               modulus_lo,
                               modulus_hi,
                               neg_inv_modulus_lo,
                               neg_inv_modulus_hi);
      }
      row[r] = value;
    }
    power_of_two_radix_cyclic_radix4(row,
                                     primitive_fourth_root,
                                     twice_modulus,
                                     modulus_lo,
                                     modulus_hi,
                                     neg_inv_modulus_lo,
                                     neg_inv_modulus_hi);
#pragma unroll
    for (int t = 0; t < 4; ++t) values[u + 4 * t] = row[t];
  }
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void power_of_two_radix_cyclic_radix8(
    scalar_t values[8],
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int root_row,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  static_assert(ROOT_ORDER == 8,
                "strict radix-8 digits require radix-8 root powers");
  // Dedicated 2x4 Cooley--Tukey radix-8 butterfly. Transform the even and
  // odd coefficient subsequences with radix-4 butterflies, couple the odd
  // spectrum by zeta_8^u, then combine the two frequency halves. This is one
  // radix-8 digit and does not iterate over three global radix-2 stages.
  constexpr int EIGHTH_ROOT_STRIDE = 1;
  constexpr int FOURTH_ROOT_INDEX = 2;
  const scalar_t primitive_fourth_root =
      radix_root_powers_acc[root_row][FOURTH_ROOT_INDEX];
  scalar_t even[4] = {values[0], values[2], values[4], values[6]};
  scalar_t odd[4] = {values[1], values[3], values[5], values[7]};
  power_of_two_radix_cyclic_radix4(even,
                                   primitive_fourth_root,
                                   twice_modulus,
                                   modulus_lo,
                                   modulus_hi,
                                   neg_inv_modulus_lo,
                                   neg_inv_modulus_hi);
  power_of_two_radix_cyclic_radix4(odd,
                                   primitive_fourth_root,
                                   twice_modulus,
                                   modulus_lo,
                                   modulus_hi,
                                   neg_inv_modulus_lo,
                                   neg_inv_modulus_hi);

#pragma unroll
  for (int frequency = 0; frequency < 4; ++frequency) {
    scalar_t coupled_odd = odd[frequency];
    if (frequency != 0) {
      coupled_odd = montgomery_mul(
          radix_root_powers_acc[root_row][frequency * EIGHTH_ROOT_STRIDE],
          coupled_odd,
          modulus_lo,
          modulus_hi,
          neg_inv_modulus_lo,
          neg_inv_modulus_hi);
    }
    values[frequency] =
        power_of_two_radix_add(even[frequency], coupled_odd, twice_modulus);
    values[frequency + 4] =
        power_of_two_radix_sub(even[frequency], coupled_odd, twice_modulus);
  }
}

template <typename scalar_t, int RADIX_BITS, int ROOT_ORDER>
__device__ __forceinline__ void power_of_two_radix_cyclic_ntt(
    scalar_t values[1 << RADIX_BITS],
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int root_row,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  static_assert(RADIX_BITS >= 2 && RADIX_BITS <= 4,
                "supported strict radix digits are radix 4, 8, and 16");
  static_assert(ROOT_ORDER == (1 << RADIX_BITS),
                "strict radix digits must match their root-table order");

  if constexpr (RADIX_BITS == 2) {
    power_of_two_radix_cyclic_radix4(values,
                                     radix_root_powers_acc[root_row][1],
                                     twice_modulus,
                                     modulus_lo,
                                     modulus_hi,
                                     neg_inv_modulus_lo,
                                     neg_inv_modulus_hi);
  } else if constexpr (RADIX_BITS == 3) {
    power_of_two_radix_cyclic_radix8<scalar_t, ROOT_ORDER>(
        values,
        radix_root_powers_acc,
        root_row,
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
  } else {
    power_of_two_radix_cyclic_radix16(values,
                                      radix_root_powers_acc,
                                      root_row,
                                      twice_modulus,
                                      modulus_lo,
                                      modulus_hi,
                                      neg_inv_modulus_lo,
                                      neg_inv_modulus_hi);
  }
}
