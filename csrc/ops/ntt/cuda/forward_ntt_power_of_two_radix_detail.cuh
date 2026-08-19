#pragma once

#include "power_of_two_radix_butterfly.cuh"

// Genuine power-of-two radix forward DIF digit kernels and host launch
// schedule.

template <typename scalar_t, int RADIX_BITS, int ROOT_ORDER>
__global__ void forward_ntt_power_of_two_radix_digit_kernel(
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
  const int logN = __ffs(N) - 1;
  const int final_stride_log = logN - start_stage - RADIX_BITS;
  const int final_stride = 1 << final_stride_log;
  if (tuple >= N / RADIX) return;

  const int initial_group = tuple >> final_stride_log;
  const int within_group = tuple & (final_stride - 1);
  const int base =
      (initial_group << (RADIX_BITS + final_stride_log)) + within_group;

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][prime_row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][prime_row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][prime_row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][prime_row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][prime_row];

  scalar_t values[RADIX];
  values[0] = a_acc[batch][prime_row][base];
  const int group_twiddle_offset =
      outer_twiddle_offset + initial_group * (RADIX - 1);
#pragma unroll
  for (int lane = 1; lane < RADIX; ++lane) {
    values[lane] = montgomery_mul(
        a_acc[batch][prime_row][base + lane * final_stride],
        outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
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

  // Radix-2 DIF produces digit-bit-reversed output order. Preserve that exact
  // domain layout so every family remains bit-exact with the indexed oracle.
#pragma unroll
  for (int lane = 0; lane < RADIX; ++lane) {
    const int natural_frequency =
        power_of_two_radix_bit_reverse<RADIX_BITS>(lane);
    a_acc[batch][prime_row][base + lane * final_stride] =
        values[natural_frequency];
  }
}

// Cooperative genuine-radix16 tail. Each four-thread worker group evaluates
// the same 4x4 Cooley--Tukey digit as power_of_two_radix_cyclic_radix16, but
// keeps only one radix-4 vector per worker. Two complete radix16 digits remain
// distinct: the first digit writes its required digit-bit-reversed layout to
// shared memory before the second digit consumes it.
template <typename scalar_t>
__device__ __forceinline__ void forward_ntt_radix16_smem_digit(
    scalar_t* data,
    scalar_t* scratch,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int logN,
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
  const int final_stride_log = logN - start_stage - RADIX_BITS;
  const int final_stride = 1 << final_stride_log;
  const int tuple = (tile_base >> RADIX_BITS) + tuple_local;
  const int initial_group = tuple >> final_stride_log;
  const int within_group = tuple & (final_stride - 1);
  const int base =
      (initial_group << (RADIX_BITS + final_stride_log)) + within_group;
  const int shared_base = base - tile_base;
  const int group_twiddle_offset =
      outer_twiddle_offset + initial_group * (RADIX - 1);
  const scalar_t primitive_fourth_root = radix_root_powers_acc[prime_row][4];

  if (threadIdx.x < 64) {
    scalar_t column[4];
#pragma unroll
    for (int column_lane = 0; column_lane < 4; ++column_lane) {
      const int lane = worker + 4 * column_lane;
      scalar_t value = data[shared_base + lane * final_stride];
      if (lane != 0) {
        value = montgomery_mul(
            value,
            outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
            modulus_lo,
            modulus_hi,
            neg_inv_modulus_lo,
            neg_inv_modulus_hi);
      }
      column[column_lane] = value;
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
      scratch[shared_base + (worker * 4 + frequency) * final_stride] =
          column[frequency];
    }
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    scalar_t row[4];
#pragma unroll
    for (int radix4_lane = 0; radix4_lane < 4; ++radix4_lane) {
      scalar_t value =
          scratch[shared_base + (radix4_lane * 4 + worker) * final_stride];
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
#pragma unroll
    for (int frequency_group = 0; frequency_group < 4; ++frequency_group) {
      const int natural_frequency = worker + 4 * frequency_group;
      const int stored_lane =
          power_of_two_radix_bit_reverse<RADIX_BITS>(natural_frequency);
      data[shared_base + stored_lane * final_stride] = row[frequency_group];
    }
  }
  __syncthreads();
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void forward_ntt_radix4_smem_digit(
    scalar_t* data,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int logN,
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
    const int final_stride_log = logN - start_stage - RADIX_BITS;
    const int final_stride = 1 << final_stride_log;
    const int tuple = (tile_base >> RADIX_BITS) + threadIdx.x;
    const int initial_group = tuple >> final_stride_log;
    const int within_group = tuple & (final_stride - 1);
    const int base =
        (initial_group << (RADIX_BITS + final_stride_log)) + within_group;
    const int shared_base = base - tile_base;
    const int group_twiddle_offset =
        outer_twiddle_offset + initial_group * (RADIX - 1);
    scalar_t values[RADIX];
    values[0] = data[shared_base];
#pragma unroll
    for (int lane = 1; lane < RADIX; ++lane) {
      values[lane] = montgomery_mul(
          data[shared_base + lane * final_stride],
          outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
          modulus_lo,
          modulus_hi,
          neg_inv_modulus_lo,
          neg_inv_modulus_hi);
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
    for (int lane = 0; lane < RADIX; ++lane) {
      const int natural_frequency =
          power_of_two_radix_bit_reverse<RADIX_BITS>(lane);
      data[shared_base + lane * final_stride] = values[natural_frequency];
    }
  }
  __syncthreads();
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void forward_ntt_radix8_smem_digit(
    scalar_t* data,
    scalar_t* scratch,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int logN,
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
  const int final_stride_log = logN - start_stage - RADIX_BITS;
  const int final_stride = 1 << final_stride_log;
  const int tuple = (tile_base >> RADIX_BITS) + tuple_local;
  const int initial_group = tuple >> final_stride_log;
  const int within_group = tuple & (final_stride - 1);
  const int base =
      (initial_group << (RADIX_BITS + final_stride_log)) + within_group;
  const int shared_base = base - tile_base;
  const int group_twiddle_offset =
      outer_twiddle_offset + initial_group * (RADIX - 1);

  if (worker < 2) {
    scalar_t values[4];
#pragma unroll
    for (int lane4 = 0; lane4 < 4; ++lane4) {
      const int lane = worker + 2 * lane4;
      scalar_t value = data[shared_base + lane * final_stride];
      if (lane != 0) {
        value = montgomery_mul(
            value,
            outer_twiddles_acc[prime_row][group_twiddle_offset + lane - 1],
            modulus_lo,
            modulus_hi,
            neg_inv_modulus_lo,
            neg_inv_modulus_hi);
      }
      values[lane4] = value;
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
      scratch[shared_base + (worker * 4 + frequency) * final_stride] =
          values[frequency];
    }
  }
  __syncthreads();

  scalar_t even = scratch[shared_base + worker * final_stride];
  scalar_t odd = scratch[shared_base + (4 + worker) * final_stride];
  if (worker != 0) {
    odd = montgomery_mul(
        radix_root_powers_acc[prime_row][worker * EIGHTH_ROOT_STRIDE],
        odd,
        modulus_lo,
        modulus_hi,
        neg_inv_modulus_lo,
        neg_inv_modulus_hi);
  }
  const scalar_t low = power_of_two_radix_add(even, odd, twice_modulus);
  const scalar_t high = power_of_two_radix_sub(even, odd, twice_modulus);
  const int low_lane = power_of_two_radix_bit_reverse<RADIX_BITS>(worker);
  const int high_lane = power_of_two_radix_bit_reverse<RADIX_BITS>(worker + 4);
  data[shared_base + low_lane * final_stride] = low;
  data[shared_base + high_lane * final_stride] = high;
  __syncthreads();
}

template <typename scalar_t, int ROOT_ORDER>
__device__ __forceinline__ void forward_ntt_fixed_radix_smem_digit(
    scalar_t* data,
    scalar_t* scratch,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const int prime_row,
    const int tile_base,
    const int logN,
    const int start_stage,
    const int outer_twiddle_offset,
    const scalar_t twice_modulus,
    const scalar_t modulus_lo,
    const scalar_t modulus_hi,
    const scalar_t neg_inv_modulus_lo,
    const scalar_t neg_inv_modulus_hi) {
  if constexpr (ROOT_ORDER == 4) {
    forward_ntt_radix4_smem_digit<scalar_t, ROOT_ORDER>(data,
                                                        outer_twiddles_acc,
                                                        radix_root_powers_acc,
                                                        prime_row,
                                                        tile_base,
                                                        logN,
                                                        start_stage,
                                                        outer_twiddle_offset,
                                                        twice_modulus,
                                                        modulus_lo,
                                                        modulus_hi,
                                                        neg_inv_modulus_lo,
                                                        neg_inv_modulus_hi);
  } else if constexpr (ROOT_ORDER == 8) {
    forward_ntt_radix8_smem_digit<scalar_t, ROOT_ORDER>(data,
                                                        scratch,
                                                        outer_twiddles_acc,
                                                        radix_root_powers_acc,
                                                        prime_row,
                                                        tile_base,
                                                        logN,
                                                        start_stage,
                                                        outer_twiddle_offset,
                                                        twice_modulus,
                                                        modulus_lo,
                                                        modulus_hi,
                                                        neg_inv_modulus_lo,
                                                        neg_inv_modulus_hi);
  } else {
    static_assert(ROOT_ORDER == 16);
    forward_ntt_radix16_smem_digit(data,
                                   scratch,
                                   outer_twiddles_acc,
                                   radix_root_powers_acc,
                                   prime_row,
                                   tile_base,
                                   logN,
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
__global__ void forward_ntt_radix16_smem_two_digit_tail_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage,
    const int first_outer_twiddle_offset,
    const int second_outer_twiddle_offset) {
  constexpr int TILE_SIZE = fhelium::ntt::kNttSharedMemoryTileSize;
  const int prime_row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tile_base = blockIdx.y * TILE_SIZE;
  const int N = static_cast<int>(a_acc.size(2));
  const int logN = __ffs(N) - 1;
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

  forward_ntt_radix16_smem_digit(data,
                                 scratch,
                                 outer_twiddles_acc,
                                 radix_root_powers_acc,
                                 prime_row,
                                 tile_base,
                                 logN,
                                 start_stage,
                                 first_outer_twiddle_offset,
                                 twice_modulus,
                                 modulus_lo,
                                 modulus_hi,
                                 neg_inv_modulus_lo,
                                 neg_inv_modulus_hi);
  forward_ntt_radix16_smem_digit(data,
                                 scratch,
                                 outer_twiddles_acc,
                                 radix_root_powers_acc,
                                 prime_row,
                                 tile_base,
                                 logN,
                                 start_stage + 4,
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
void launch_forward_ntt_radix16_smem_two_digit_tail_cuda(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    CudaTensorAccessor32<scalar_t, 2> outer_twiddles_acc,
    CudaTensorAccessor32<scalar_t, 2> radix_root_powers_acc,
    CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage,
    const int first_outer_twiddle_offset,
    const int second_outer_twiddle_offset,
    cudaStream_t stream) {
  constexpr int TILE_SIZE = fhelium::ntt::kNttSharedMemoryTileSize;
  constexpr int THREAD_COUNT = 64;
  const int rows = static_cast<int>(a_acc.size(1));
  const int N = static_cast<int>(a_acc.size(2));
  const dim3 grid(rows, N / TILE_SIZE, a_acc.size(0));
  const size_t shared_bytes = 2 * TILE_SIZE * sizeof(scalar_t);
  forward_ntt_radix16_smem_two_digit_tail_kernel<scalar_t>
      <<<grid, THREAD_COUNT, shared_bytes, stream>>>(
          a_acc,
          outer_twiddles_acc,
          radix_root_powers_acc,
          params_acc,
          start_stage,
          first_outer_twiddle_offset,
          second_outer_twiddle_offset);
}

template <typename scalar_t, int ROOT_ORDER>
__global__ void forward_ntt_power_of_two_radix_smem_tail_kernel(
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
  const int N = static_cast<int>(a_acc.size(2));
  const int logN = __ffs(N) - 1;
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
    forward_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        logN,
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
    forward_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        logN,
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
    forward_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        logN,
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
    forward_ntt_fixed_radix_smem_digit<scalar_t, ROOT_ORDER>(
        data,
        scratch,
        outer_twiddles_acc,
        radix_root_powers_acc,
        prime_row,
        tile_base,
        logN,
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
void launch_forward_ntt_fixed_radix_smem_tail_cuda(
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
  forward_ntt_power_of_two_radix_smem_tail_kernel<scalar_t, ROOT_ORDER>
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
void launch_forward_ntt_fixed_radix_digit_cuda(
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
  forward_ntt_power_of_two_radix_digit_kernel<scalar_t, RADIX_BITS, ROOT_ORDER>
      <<<grid, kCudaBlockSize, 0, stream>>>(a_acc,
                                            outer_twiddles_acc,
                                            radix_root_powers_acc,
                                            params_acc,
                                            start_stage,
                                            outer_twiddle_offset);
}

template <typename scalar_t, int RADIX_BITS, int ROOT_ORDER>
void launch_forward_ntt_fixed_radix_cuda(torch::Tensor a,
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
  const int shared_digit_begin = digit_count - shared_digit_count;
  TORCH_CHECK(
      shared_digit_count <= fhelium::ntt::kPowerOfTwoRadixMaxSharedDigits,
      "Fixed-radix shared tail exceeds the supported digit count");

  int start_stage = 0;
  int outer_twiddle_offset = 0;
  for (int digit_index = 0; digit_index < shared_digit_begin; ++digit_index) {
    launch_forward_ntt_fixed_radix_digit_cuda<scalar_t, RADIX_BITS, ROOT_ORDER>(
        a_acc,
        outer_twiddles_acc,
        radix_root_powers_acc,
        params_acc,
        start_stage,
        outer_twiddle_offset,
        stream);
    outer_twiddle_offset += (1 << start_stage) * (ROOT_ORDER - 1);
    start_stage += RADIX_BITS;
  }

  if (shared_digit_count == 0) return;

  std::array<int, fhelium::ntt::kPowerOfTwoRadixMaxSharedDigits>
      shared_outer_offsets{};
  int shared_start_stage = start_stage;
  int shared_outer_offset = outer_twiddle_offset;
  for (int shared_index = 0; shared_index < shared_digit_count;
       ++shared_index) {
    shared_outer_offsets[shared_index] = shared_outer_offset;
    shared_outer_offset += (1 << shared_start_stage) * (ROOT_ORDER - 1);
    shared_start_stage += RADIX_BITS;
  }

  if constexpr (ROOT_ORDER == 16) {
    if (shared_digit_count == 2) {
      launch_forward_ntt_radix16_smem_two_digit_tail_cuda(
          a_acc,
          outer_twiddles_acc,
          radix_root_powers_acc,
          params_acc,
          start_stage,
          shared_outer_offsets[0],
          shared_outer_offsets[1],
          stream);
      return;
    }
  }

  launch_forward_ntt_fixed_radix_smem_tail_cuda<scalar_t, ROOT_ORDER>(
      a_acc,
      outer_twiddles_acc,
      radix_root_powers_acc,
      params_acc,
      start_stage,
      shared_digit_count,
      shared_outer_offsets[0],
      shared_outer_offsets[1],
      shared_outer_offsets[2],
      shared_outer_offsets[3],
      stream);
}

template <typename scalar_t>
void launch_forward_ntt_power_of_two_radix_cuda(
    torch::Tensor a,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int radix,
    const int shared_memory_log_n,
    cudaStream_t stream) {
  if (radix == 16) {
    launch_forward_ntt_fixed_radix_cuda<scalar_t, 4, 16>(a,
                                                         outer_twiddles,
                                                         radix_root_powers,
                                                         rns_params,
                                                         shared_memory_log_n,
                                                         stream);
  } else if (radix == 8) {
    launch_forward_ntt_fixed_radix_cuda<scalar_t, 3, 8>(a,
                                                        outer_twiddles,
                                                        radix_root_powers,
                                                        rns_params,
                                                        shared_memory_log_n,
                                                        stream);
  } else {
    launch_forward_ntt_fixed_radix_cuda<scalar_t, 2, 4>(a,
                                                        outer_twiddles,
                                                        radix_root_powers,
                                                        rns_params,
                                                        shared_memory_log_n,
                                                        stream);
  }
}
