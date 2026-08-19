#pragma once

#include "power_of_two_radix_butterfly.cuh"

// Genuine power-of-two radix inverse DIT digit kernels and host launch
// schedule.

template <typename scalar_t, int RADIX_BITS, int ROOT_ORDER>
__global__ void inverse_ntt_power_of_two_radix_digit_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage,
    const int outer_twiddle_offset) {
  constexpr int RADIX = 1 << RADIX_BITS;
  const int prime_row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tuple = blockIdx.y * kCudaBlockSize + threadIdx.x;
  const int N = static_cast<int>(a_acc.size(2));
  const int coefficient_stride = 1 << start_stage;
  if (tuple >= N / RADIX) return;

  const int initial_group = tuple >> start_stage;
  const int within_group = tuple & (coefficient_stride - 1);
  const int base = (initial_group << (RADIX_BITS + start_stage)) + within_group;

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][prime_row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][prime_row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][prime_row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][prime_row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][prime_row];

  // Undo the digit-bit-reversed DIF lane order before evaluating the cyclic
  // inverse-root NTT. Normalization remains one final N^-1 epilogue.
  scalar_t values[RADIX];
#pragma unroll
  for (int natural_frequency = 0; natural_frequency < RADIX;
       ++natural_frequency) {
    const int stored_lane =
        power_of_two_radix_bit_reverse<RADIX_BITS>(natural_frequency);
    values[natural_frequency] =
        a_acc[batch][prime_row][base + stored_lane * coefficient_stride];
  }

  power_of_two_radix_cyclic_ntt<scalar_t, RADIX_BITS, ROOT_ORDER>(
      values,
      radix_root_powers_acc,
      prime_row,
      twice_modulus,
      modulus_lo,
      modulus_hi,
      neg_inv_modulus_lo,
      neg_inv_modulus_hi);

  const int group_twiddle_offset =
      outer_twiddle_offset + initial_group * (RADIX - 1);
  a_acc[batch][prime_row][base] = values[0];
#pragma unroll
  for (int lane = 1; lane < RADIX; ++lane) {
    a_acc[batch][prime_row][base + lane * coefficient_stride] = montgomery_mul(
        values[lane],
        outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
  }
}

// Cooperative genuine-radix16 prefix. This is the exact DIT dual of the
// forward shared-memory tail: every digit first restores its own
// digit-bit-reversed lane order, evaluates the same 4x4 cyclic transform, and
// only then applies the inverse outer twist.
template <typename scalar_t>
__device__ __forceinline__ void inverse_ntt_radix16_smem_digit(
    scalar_t* data,
    scalar_t* scratch,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int start_stage,
    const int outer_twiddle_offset,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  constexpr int RADIX_BITS = 4;
  constexpr int RADIX = 16;
  const int worker = threadIdx.x & 3;
  const int tuple_local = threadIdx.x >> 2;
  const int coefficient_stride = 1 << start_stage;
  const int tuple = (tile_base >> RADIX_BITS) + tuple_local;
  const int initial_group = tuple >> start_stage;
  const int within_group = tuple & (coefficient_stride - 1);
  const int base = (initial_group << (RADIX_BITS + start_stage)) + within_group;
  const int shared_base = base - tile_base;
  const scalar_t primitive_fourth_root = radix_root_powers_acc[prime_row][4];

  if (threadIdx.x < 64) {
    scalar_t column[4];
#pragma unroll
    for (int column_lane = 0; column_lane < 4; ++column_lane) {
      const int natural_frequency = worker + 4 * column_lane;
      const int stored_lane =
          power_of_two_radix_bit_reverse<RADIX_BITS>(natural_frequency);
      column[column_lane] =
          data[shared_base + stored_lane * coefficient_stride];
    }
    power_of_two_radix_cyclic_radix4(column,
                                     primitive_fourth_root,
                                     twice_modulus,
                                     modulus_lo,
                                     modulus_hi,
                                     neg_inv_modulus_lo,
                                     neg_inv_modulus_hi);
#pragma unroll
    for (int frequency = 0; frequency < 4; ++frequency) {
      scratch[shared_base + (worker * 4 + frequency) * coefficient_stride] =
          column[frequency];
    }
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    scalar_t row[4];
#pragma unroll
    for (int radix4_lane = 0; radix4_lane < 4; ++radix4_lane) {
      scalar_t value = scratch[shared_base +
                               (radix4_lane * 4 + worker) * coefficient_stride];
      const int exponent = worker * radix4_lane;
      if (exponent != 0) {
        value = montgomery_mul(radix_root_powers_acc[prime_row][exponent],
                               value,
                               modulus_lo,
                               modulus_hi,
                               neg_inv_modulus_lo,
                               neg_inv_modulus_hi);
      }
      row[radix4_lane] = value;
    }
    power_of_two_radix_cyclic_radix4(row,
                                     primitive_fourth_root,
                                     twice_modulus,
                                     modulus_lo,
                                     modulus_hi,
                                     neg_inv_modulus_lo,
                                     neg_inv_modulus_hi);

    const int group_twiddle_offset =
        outer_twiddle_offset + initial_group * (RADIX - 1);
#pragma unroll
    for (int frequency_group = 0; frequency_group < 4; ++frequency_group) {
      const int lane = worker + 4 * frequency_group;
      scalar_t value = row[frequency_group];
      if (lane != 0) {
        value = montgomery_mul(
            value,
            outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
            modulus_lo,
            modulus_hi,
            neg_inv_modulus_lo,
            neg_inv_modulus_hi);
      }
      data[shared_base + lane * coefficient_stride] = value;
    }
  }
  __syncthreads();
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void inverse_ntt_radix4_smem_digit(
    scalar_t* data,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int start_stage,
    const int outer_twiddle_offset,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  if (threadIdx.x < 64) {
    constexpr int RADIX_BITS = 2;
    constexpr int RADIX = 4;
    constexpr int FOURTH_ROOT_INDEX = ROOT_ORDER / 4;
    const int coefficient_stride = 1 << start_stage;
    const int tuple = (tile_base >> RADIX_BITS) + threadIdx.x;
    const int initial_group = tuple >> start_stage;
    const int within_group = tuple & (coefficient_stride - 1);
    const int base =
        (initial_group << (RADIX_BITS + start_stage)) + within_group;
    const int shared_base = base - tile_base;
    scalar_t values[RADIX];
#pragma unroll
    for (int natural_frequency = 0; natural_frequency < RADIX;
         ++natural_frequency) {
      const int stored_lane =
          power_of_two_radix_bit_reverse<RADIX_BITS>(natural_frequency);
      values[natural_frequency] =
          data[shared_base + stored_lane * coefficient_stride];
    }
    power_of_two_radix_cyclic_radix4(
        values,
        radix_root_powers_acc[prime_row][FOURTH_ROOT_INDEX],
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
    const int group_twiddle_offset =
        outer_twiddle_offset + initial_group * (RADIX - 1);
    data[shared_base] = values[0];
#pragma unroll
    for (int lane = 1; lane < RADIX; ++lane) {
      data[shared_base + lane * coefficient_stride] = montgomery_mul(
          values[lane],
          outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
          modulus_lo,
          modulus_hi,
          neg_inv_modulus_lo,
          neg_inv_modulus_hi);
    }
  }
  __syncthreads();
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void inverse_ntt_radix8_smem_digit(
    scalar_t* data,
    scalar_t* scratch,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int start_stage,
    const int outer_twiddle_offset,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  static_assert(ROOT_ORDER == 8);
  constexpr int RADIX_BITS = 3;
  constexpr int RADIX = 8;
  constexpr int EIGHTH_ROOT_STRIDE = 1;
  constexpr int FOURTH_ROOT_INDEX = 2;
  const int worker = threadIdx.x & 3;
  const int tuple_local = threadIdx.x >> 2;
  const int coefficient_stride = 1 << start_stage;
  const int tuple = (tile_base >> RADIX_BITS) + tuple_local;
  const int initial_group = tuple >> start_stage;
  const int within_group = tuple & (coefficient_stride - 1);
  const int base = (initial_group << (RADIX_BITS + start_stage)) + within_group;
  const int shared_base = base - tile_base;

  if (worker < 2) {
    scalar_t values[4];
#pragma unroll
    for (int lane4 = 0; lane4 < 4; ++lane4) {
      const int natural_frequency = worker + 2 * lane4;
      const int stored_lane =
          power_of_two_radix_bit_reverse<RADIX_BITS>(natural_frequency);
      values[lane4] = data[shared_base + stored_lane * coefficient_stride];
    }
    power_of_two_radix_cyclic_radix4(
        values,
        radix_root_powers_acc[prime_row][FOURTH_ROOT_INDEX],
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
#pragma unroll
    for (int frequency = 0; frequency < 4; ++frequency) {
      scratch[shared_base + (worker * 4 + frequency) * coefficient_stride] =
          values[frequency];
    }
  }
  __syncthreads();

  scalar_t even = scratch[shared_base + worker * coefficient_stride];
  scalar_t odd = scratch[shared_base + (4 + worker) * coefficient_stride];
  if (worker != 0) {
    odd = montgomery_mul(
        radix_root_powers_acc[prime_row][worker * EIGHTH_ROOT_STRIDE],
        odd,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
  }
  scalar_t outputs[2] = {power_of_two_radix_add(even, odd, twice_modulus),
                         power_of_two_radix_sub(even, odd, twice_modulus)};
  const int group_twiddle_offset =
      outer_twiddle_offset + initial_group * (RADIX - 1);
#pragma unroll
  for (int output_half = 0; output_half < 2; ++output_half) {
    const int lane = worker + 4 * output_half;
    scalar_t value = outputs[output_half];
    if (lane != 0) {
      value = montgomery_mul(
          value,
          outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
          modulus_lo,
          modulus_hi,
          neg_inv_modulus_lo,
          neg_inv_modulus_hi);
    }
    data[shared_base + lane * coefficient_stride] = value;
  }
  __syncthreads();
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void inverse_ntt_fixed_radix_smem_digit(
    scalar_t* data,
    scalar_t* scratch,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int start_stage,
    const int outer_twiddle_offset,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  if constexpr (ROOT_ORDER == 4) {
    inverse_ntt_radix4_smem_digit<scalar_t, ROOT_ORDER>(data,
                                                        outer_twiddles_acc,
                                                        radix_root_powers_acc,
                                                        prime_row,
                                                        tile_base,
                                                        start_stage,
                                                        outer_twiddle_offset,
                                                        twice_modulus,
                                                        modulus_lo,
                                                        modulus_hi,
                                                        neg_inv_modulus_lo,
                                                        neg_inv_modulus_hi);
  } else if constexpr (ROOT_ORDER == 8) {
    inverse_ntt_radix8_smem_digit<scalar_t, ROOT_ORDER>(data,
                                                        scratch,
                                                        outer_twiddles_acc,
                                                        radix_root_powers_acc,
                                                        prime_row,
                                                        tile_base,
                                                        start_stage,
                                                        outer_twiddle_offset,
                                                        twice_modulus,
                                                        modulus_lo,
                                                        modulus_hi,
                                                        neg_inv_modulus_lo,
                                                        neg_inv_modulus_hi);
  } else {
    static_assert(ROOT_ORDER == 16);
    inverse_ntt_radix16_smem_digit(data,
                                   scratch,
                                   outer_twiddles_acc,
                                   radix_root_powers_acc,
                                   prime_row,
                                   tile_base,
                                   start_stage,
                                   outer_twiddle_offset,
                                   twice_modulus,
                                   modulus_lo,
                                   modulus_hi,
                                   neg_inv_modulus_lo,
                                   neg_inv_modulus_hi);
  }
}

template <typename scalar_t>
__global__ void inverse_ntt_radix16_smem_two_digit_prefix_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int first_outer_twiddle_offset,
    const int second_outer_twiddle_offset) {
  constexpr int TILE_SIZE = fhelium::ntt::kNttSharedMemoryTileSize;
  const int prime_row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tile_base = blockIdx.y * TILE_SIZE;
  extern __shared__ char shared_bytes[];
  auto* data = reinterpret_cast<scalar_t*>(shared_bytes);
  auto* scratch = data + TILE_SIZE;

  for (int coefficient = threadIdx.x; coefficient < TILE_SIZE;
       coefficient += blockDim.x) {
    data[coefficient] = a_acc[batch][prime_row][tile_base + coefficient];
  }
  __syncthreads();

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][prime_row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][prime_row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][prime_row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][prime_row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][prime_row];

  inverse_ntt_radix16_smem_digit(data,
                                 scratch,
                                 outer_twiddles_acc,
                                 radix_root_powers_acc,
                                 prime_row,
                                 tile_base,
                                 0,
                                 first_outer_twiddle_offset,
                                 twice_modulus,
                                 modulus_lo,
                                 modulus_hi,
                                 neg_inv_modulus_lo,
                                 neg_inv_modulus_hi);
  inverse_ntt_radix16_smem_digit(data,
                                 scratch,
                                 outer_twiddles_acc,
                                 radix_root_powers_acc,
                                 prime_row,
                                 tile_base,
                                 4,
                                 second_outer_twiddle_offset,
                                 twice_modulus,
                                 modulus_lo,
                                 modulus_hi,
                                 neg_inv_modulus_lo,
                                 neg_inv_modulus_hi);

  for (int coefficient = threadIdx.x; coefficient < TILE_SIZE;
       coefficient += blockDim.x) {
    a_acc[batch][prime_row][tile_base + coefficient] = data[coefficient];
  }
}

template <typename scalar_t>
void launch_inverse_ntt_radix16_smem_two_digit_prefix_cuda(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int first_outer_twiddle_offset,
    const int second_outer_twiddle_offset,
    cudaStream_t stream) {
  constexpr int TILE_SIZE = fhelium::ntt::kNttSharedMemoryTileSize;
  constexpr int THREAD_COUNT = 64;
  const int rows = static_cast<int>(a_acc.size(1));
  const int N = static_cast<int>(a_acc.size(2));
  const dim3 grid(rows, N / TILE_SIZE, a_acc.size(0));
  const size_t shared_bytes = 2 * TILE_SIZE * sizeof(scalar_t);
  inverse_ntt_radix16_smem_two_digit_prefix_kernel<scalar_t>
      <<<grid, THREAD_COUNT, shared_bytes, stream>>>(
          a_acc,
          outer_twiddles_acc,
          radix_root_powers_acc,
          params_acc,
          first_outer_twiddle_offset,
          second_outer_twiddle_offset);
}

template <typename scalar_t, int ROOT_ORDER>
__global__ void inverse_ntt_power_of_two_radix_smem_prefix_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage,
    const int digit_count,
    const int outer_offset0,
    const int outer_offset1,
    const int outer_offset2,
    const int outer_offset3) {
  constexpr int TILE_SIZE = fhelium::ntt::kNttSharedMemoryTileSize;
  constexpr int RADIX_BITS = ROOT_ORDER == 16 ? 4 : (ROOT_ORDER == 8 ? 3 : 2);
  const int prime_row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tile_base = blockIdx.y * TILE_SIZE;
  extern __shared__ char shared_bytes[];
  auto* data = reinterpret_cast<scalar_t*>(shared_bytes);
  auto* scratch = data + TILE_SIZE;

  for (int coefficient = threadIdx.x; coefficient < TILE_SIZE;
       coefficient += blockDim.x) {
    data[coefficient] = a_acc[batch][prime_row][tile_base + coefficient];
  }
  __syncthreads();

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][prime_row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][prime_row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][prime_row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][prime_row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][prime_row];

  int digit_start_stage = start_stage;
  if (digit_count > 0) {
    inverse_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        digit_start_stage,
        outer_offset0,
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
    digit_start_stage += RADIX_BITS;
  }
  if (digit_count > 1) {
    inverse_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        digit_start_stage,
        outer_offset1,
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
    digit_start_stage += RADIX_BITS;
  }
  if (digit_count > 2) {
    inverse_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        digit_start_stage,
        outer_offset2,
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
    digit_start_stage += RADIX_BITS;
  }
  if (digit_count > 3) {
    inverse_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        digit_start_stage,
        outer_offset3,
        twice_modulus,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
  }

  for (int coefficient = threadIdx.x; coefficient < TILE_SIZE;
       coefficient += blockDim.x) {
    a_acc[batch][prime_row][tile_base + coefficient] = data[coefficient];
  }
}

template <typename scalar_t, int ROOT_ORDER>
void launch_inverse_ntt_fixed_radix_smem_prefix_cuda(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage,
    const int digit_count,
    const int outer_offset0,
    const int outer_offset1,
    const int outer_offset2,
    const int outer_offset3,
    cudaStream_t stream) {
  constexpr int TILE_SIZE = fhelium::ntt::kNttSharedMemoryTileSize;
  constexpr int THREAD_COUNT = 128;
  const int rows = static_cast<int>(a_acc.size(1));
  const int N = static_cast<int>(a_acc.size(2));
  const dim3 grid(rows, N / TILE_SIZE, a_acc.size(0));
  const size_t shared_bytes = 2 * TILE_SIZE * sizeof(scalar_t);
  inverse_ntt_power_of_two_radix_smem_prefix_kernel<scalar_t, ROOT_ORDER>
      <<<grid, THREAD_COUNT, shared_bytes, stream>>>(a_acc,
                                                     outer_twiddles_acc,
                                                     radix_root_powers_acc,
                                                     params_acc,
                                                     start_stage,
                                                     digit_count,
                                                     outer_offset0,
                                                     outer_offset1,
                                                     outer_offset2,
                                                     outer_offset3);
}

template <typename scalar_t, int RADIX_BITS, int ROOT_ORDER>
void launch_inverse_ntt_fixed_radix_digit_cuda(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage,
    const int outer_twiddle_offset,
    cudaStream_t stream) {
  static_assert(ROOT_ORDER == (1 << RADIX_BITS));
  const int rows = static_cast<int>(a_acc.size(1));
  const int N = static_cast<int>(a_acc.size(2));
  const int tuple_count = N >> RADIX_BITS;
  const dim3 grid(
      rows, (tuple_count + kCudaBlockSize - 1) / kCudaBlockSize, a_acc.size(0));
  inverse_ntt_power_of_two_radix_digit_kernel<scalar_t, RADIX_BITS, ROOT_ORDER>
      <<<grid, kCudaBlockSize, 0, stream>>>(a_acc,
                                            outer_twiddles_acc,
                                            radix_root_powers_acc,
                                            params_acc,
                                            start_stage,
                                            outer_twiddle_offset);
}

template <typename scalar_t, int RADIX_BITS, int ROOT_ORDER>
void launch_inverse_ntt_fixed_radix_cuda(torch::Tensor a,
                                         const torch::Tensor outer_twiddles,
                                         const torch::Tensor radix_root_powers,
                                         const torch::Tensor rns_params,
                                         const int shared_memory_log_n,
                                         cudaStream_t stream) {
  static_assert(ROOT_ORDER == (1 << RADIX_BITS));
  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  auto outer_twiddles_acc =
      FHELIUM_CUDA_ACCESSOR32(outer_twiddles, scalar_t, 2);
  auto radix_root_powers_acc =
      FHELIUM_CUDA_ACCESSOR32(radix_root_powers, scalar_t, 2);
  auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);

  const int N = static_cast<int>(a.size(2));
  const int logN = __builtin_ctz(static_cast<unsigned>(N));
  const int digit_count = logN / RADIX_BITS;
  const int shared_digit_count =
      std::min(digit_count, shared_memory_log_n / RADIX_BITS);
  TORCH_CHECK(
      shared_digit_count <= fhelium::ntt::kPowerOfTwoRadixMaxSharedDigits,
      "Fixed-radix shared prefix exceeds the supported digit count");

  int start_stage = 0;
  int outer_twiddle_offset = 0;
  std::array<int, fhelium::ntt::kPowerOfTwoRadixMaxSharedDigits>
      shared_outer_offsets{};
  for (int shared_index = 0; shared_index < shared_digit_count;
       ++shared_index) {
    shared_outer_offsets[shared_index] = outer_twiddle_offset;
    const int forward_stage_start = logN - start_stage - RADIX_BITS;
    outer_twiddle_offset += (1 << forward_stage_start) * (ROOT_ORDER - 1);
    start_stage += RADIX_BITS;
  }

  if (shared_digit_count > 0) {
    if constexpr (ROOT_ORDER == 16) {
      if (shared_digit_count == 2) {
        launch_inverse_ntt_radix16_smem_two_digit_prefix_cuda(
            a_acc,
            outer_twiddles_acc,
            radix_root_powers_acc,
            params_acc,
            shared_outer_offsets[0],
            shared_outer_offsets[1],
            stream);
      } else {
        launch_inverse_ntt_fixed_radix_smem_prefix_cuda<scalar_t, ROOT_ORDER>(
            a_acc,
            outer_twiddles_acc,
            radix_root_powers_acc,
            params_acc,
            0,
            shared_digit_count,
            shared_outer_offsets[0],
            shared_outer_offsets[1],
            shared_outer_offsets[2],
            shared_outer_offsets[3],
            stream);
      }
    } else {
      launch_inverse_ntt_fixed_radix_smem_prefix_cuda<scalar_t, ROOT_ORDER>(
          a_acc,
          outer_twiddles_acc,
          radix_root_powers_acc,
          params_acc,
          0,
          shared_digit_count,
          shared_outer_offsets[0],
          shared_outer_offsets[1],
          shared_outer_offsets[2],
          shared_outer_offsets[3],
          stream);
    }
  }

  for (int digit_index = shared_digit_count; digit_index < digit_count;
       ++digit_index) {
    launch_inverse_ntt_fixed_radix_digit_cuda<scalar_t, RADIX_BITS, ROOT_ORDER>(
        a_acc,
        outer_twiddles_acc,
        radix_root_powers_acc,
        params_acc,
        start_stage,
        outer_twiddle_offset,
        stream);
    const int forward_stage_start = logN - start_stage - RADIX_BITS;
    outer_twiddle_offset += (1 << forward_stage_start) * (ROOT_ORDER - 1);
    start_stage += RADIX_BITS;
  }
}

template <typename scalar_t>
void launch_inverse_ntt_power_of_two_radix_cuda(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int radix,
    const int shared_memory_log_n,
    cudaStream_t stream) {
  if (radix == 16) {
    launch_inverse_ntt_fixed_radix_cuda<scalar_t, 4, 16>(a,
                                                         outer_twiddles,
                                                         radix_root_powers,
                                                         rns_params,
                                                         shared_memory_log_n,
                                                         stream);
  } else if (radix == 8) {
    launch_inverse_ntt_fixed_radix_cuda<scalar_t, 3, 8>(a,
                                                        outer_twiddles,
                                                        radix_root_powers,
                                                        rns_params,
                                                        shared_memory_log_n,
                                                        stream);
  } else {
    launch_inverse_ntt_fixed_radix_cuda<scalar_t, 2, 4>(a,
                                                        outer_twiddles,
                                                        radix_root_powers,
                                                        rns_params,
                                                        shared_memory_log_n,
                                                        stream);
  }
}
