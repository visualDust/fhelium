#pragma once

#include "compact_twiddles.cuh"

// Private compact INTT kernels and launch helpers.

// Complete inverse transforms apply their representation transition while
// storing the final butterfly. Partial stage ranges remain unnormalized.
enum class CompactInverseOutput {
  Unnormalized,
  Montgomery,
  StandardLazy,
  Standard,
  Centered,
};

template <CompactInverseOutput OUTPUT, typename scalar_t>
__device__ __forceinline__ scalar_t compact_inverse_output(
    scalar_t value, int row, const CudaTensorAccessor32<scalar_t, 2>& params) {
  if constexpr (OUTPUT == CompactInverseOutput::Unnormalized) {
    return value;
  } else if constexpr (OUTPUT == CompactInverseOutput::Montgomery) {
    return inverse_ntt_normalize(value, row, params);
  } else {
    value = inverse_ntt_to_standard_lazy(value, row, params);
    if constexpr (OUTPUT == CompactInverseOutput::StandardLazy) {
      return value;
    } else {
      value = reduce_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
      if constexpr (OUTPUT == CompactInverseOutput::Centered) {
        return center_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
      } else {
        return value;
      }
    }
  }
}


template <typename scalar_t, CompactInverseOutput OUTPUT = CompactInverseOutput::Unnormalized,
          typename TwiddleAccessor>
__global__ void inverse_ntt_compact_stage_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const TwiddleAccessor inverse_twiddles_compact_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int stage) {
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int j = blockIdx.y * kCudaBlockSize + threadIdx.x;
  const int N = static_cast<int>(a_acc.size(2));
  if (j >= N / 2) return;
  const int q = 1 << stage;
  const int group = j >> stage;
  const int rr = j & (q - 1);
  const int even_j = (group << (stage + 1)) + rr;
  const int odd_j = even_j + q;

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][row];

  const scalar_t U = a_acc[batch][row][even_j];
  const int twiddle_group = j >> stage;
  const scalar_t V = a_acc[batch][row][odd_j];

  const scalar_t O = sub_lazy_residues(U, V, twice_modulus);
  const scalar_t W = compact_twiddle_multiply(inverse_twiddles_compact_acc, row, (N >> (stage + 1)) + twiddle_group, O, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);

  a_acc[batch][row][odd_j] = compact_inverse_output<OUTPUT>(W, row, params_acc);
  a_acc[batch][row][even_j] = compact_inverse_output<OUTPUT>(
      add_lazy_residues(U, V, twice_modulus), row, params_acc);
}

template <typename scalar_t, int GROUPED_STAGE_COUNT,
          CompactInverseOutput OUTPUT = CompactInverseOutput::Unnormalized,
          typename TwiddleAccessor>
__global__ void inverse_ntt_compact_grouped_stages_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const TwiddleAccessor inverse_twiddles_compact_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage) {
  static_assert(GROUPED_STAGE_COUNT == 2 || GROUPED_STAGE_COUNT == 3 ||
                    GROUPED_STAGE_COUNT == 4,
                "supported compact inverse kernels: 4/8/16");
  constexpr int GROUP = 1 << GROUPED_STAGE_COUNT;

  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tuple = blockIdx.y * kCudaBlockSize + threadIdx.x;
  const int N = static_cast<int>(a_acc.size(2));
  const int q = 1 << start_stage;
  if (tuple >= (N >> GROUPED_STAGE_COUNT)) return;

  const int initial_group = tuple >> start_stage;
  const int j = tuple & (q - 1);
  const int base = (initial_group << (GROUPED_STAGE_COUNT + start_stage)) + j;

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const int twiddle_row = row;

  scalar_t x[GROUP];
#pragma unroll
  for (int k = 0; k < GROUP; ++k) x[k] = a_acc[batch][row][base + k * q];

#pragma unroll
  for (int r = 0; r < GROUPED_STAGE_COUNT; ++r) {
    const int stage = start_stage + r;
    const int stride = 1 << r;
    const int t_global_log = start_stage + r;
    const int groups_inside_initial_log = GROUPED_STAGE_COUNT - r - 1;
#pragma unroll
    for (int kk = 0; kk < GROUP; kk += (2 * stride)) {
      const int subgroup_within_initial_group = kk >> (r + 1);
      const int global_group = (initial_group << groups_inside_initial_log) +
                               subgroup_within_initial_group;
#pragma unroll
      for (int rr = 0; rr < stride; ++rr) {
        const int lo = kk + rr;
        const int hi = lo + stride;
        const int flat =
            (global_group << t_global_log) + j + (rr << start_stage);

        const scalar_t U = x[lo];
        const int twiddle_group = flat >> stage;
        const scalar_t V = x[hi];

        const scalar_t O = sub_lazy_residues(U, V, twice_modulus);
        const scalar_t W = compact_twiddle_multiply(inverse_twiddles_compact_acc, twiddle_row, (N >> (stage + 1)) + twiddle_group, O,
                                          modulus_lo,
                                          modulus_hi,
                                          neg_inv_modulus_lo,
                                          neg_inv_modulus_hi);

        x[hi] = W;
        x[lo] = add_lazy_residues(U, V, twice_modulus);
      }
    }
  }

#pragma unroll
  for (int k = 0; k < GROUP; ++k) {
    a_acc[batch][row][base + k * q] =
        compact_inverse_output<OUTPUT>(x[k], row, params_acc);
  }
}

template <typename scalar_t, CompactInverseOutput OUTPUT = CompactInverseOutput::Unnormalized,
          typename TwiddleAccessor>
__global__ void inverse_ntt_compact_smem_prefix_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const TwiddleAccessor inverse_twiddles_compact_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int end_stage) {
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tile = blockIdx.y;
  const int thread_offset = threadIdx.x;
  constexpr int HALF_TILE_COEFFICIENT_COUNT = kCudaBlockSize / 2;
  const int N = static_cast<int>(a_acc.size(2));
  const int tile_base = tile * kCudaBlockSize;
  extern __shared__ unsigned char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const int twiddle_row = row;

  smem[thread_offset] = a_acc[batch][row][tile_base + thread_offset];
  smem[thread_offset + HALF_TILE_COEFFICIENT_COUNT] =
      a_acc[batch][row]
           [tile_base + thread_offset + HALF_TILE_COEFFICIENT_COUNT];
  __syncthreads();

  for (int stage = 0; stage < end_stage; ++stage) {
    const int t = 1 << stage;
    const int span_id = thread_offset >> stage;
    const int rr = thread_offset & (t - 1);
    const int lo = (span_id << (stage + 1)) + rr;
    const int hi = lo + t;
    const int global_lo = tile_base + lo;
    const scalar_t U = smem[lo];
    const int twiddle_group = global_lo >> (stage + 1);
    const scalar_t V = smem[hi];
    const scalar_t O = sub_lazy_residues(U, V, twice_modulus);
    const scalar_t W = compact_twiddle_multiply(inverse_twiddles_compact_acc, twiddle_row, (N >> (stage + 1)) + twiddle_group, O, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);

    smem[hi] = W;
    smem[lo] = add_lazy_residues(U, V, twice_modulus);
    __syncthreads();
  }

  a_acc[batch][row][tile_base + thread_offset] =
      compact_inverse_output<OUTPUT>(smem[thread_offset], row, params_acc);
  a_acc[batch][row][tile_base + thread_offset + HALF_TILE_COEFFICIENT_COUNT] =
      compact_inverse_output<OUTPUT>(
          smem[thread_offset + HALF_TILE_COEFFICIENT_COUNT], row, params_acc);
}

template <typename scalar_t, CompactInverseOutput OUTPUT = CompactInverseOutput::Unnormalized,
          typename TwiddleAccessor>
void launch_inverse_ntt_compact_grouped_stages_cuda(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    TwiddleAccessor inverse_twiddles_compact_acc,
    CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int stage,
    const int grouped_stage_count,
    const dim3 dim_grid_stage,
    cudaStream_t stream) {
  if (grouped_stage_count == 1) {
    inverse_ntt_compact_stage_kernel<scalar_t, OUTPUT>
        <<<dim_grid_stage, kCudaBlockSize, 0, stream>>>(
            a_acc, inverse_twiddles_compact_acc, params_acc, stage);
    return;
  }

  const int C = dim_grid_stage.x;
  const int N = static_cast<int>(a_acc.size(2));
  const int tuples_per_prime = N >> grouped_stage_count;
  const dim3 dim_grid(C,
                      (tuples_per_prime + kCudaBlockSize - 1) / kCudaBlockSize,
                      dim_grid_stage.z);
  if (grouped_stage_count == 4) {
    inverse_ntt_compact_grouped_stages_kernel<scalar_t, 4, OUTPUT>
        <<<dim_grid, kCudaBlockSize, 0, stream>>>(
            a_acc, inverse_twiddles_compact_acc, params_acc, stage);
  } else if (grouped_stage_count == 3) {
    inverse_ntt_compact_grouped_stages_kernel<scalar_t, 3, OUTPUT>
        <<<dim_grid, kCudaBlockSize, 0, stream>>>(
            a_acc, inverse_twiddles_compact_acc, params_acc, stage);
  } else {
    inverse_ntt_compact_grouped_stages_kernel<scalar_t, 2, OUTPUT>
        <<<dim_grid, kCudaBlockSize, 0, stream>>>(
            a_acc, inverse_twiddles_compact_acc, params_acc, stage);
  }
}

template <typename scalar_t, CompactInverseOutput OUTPUT = CompactInverseOutput::Unnormalized>
void launch_inverse_ntt_compact_grouped_stage_range_cuda(
    torch::Tensor a,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int transform_rows,
    const int start_stage,
    const int end_stage,
    cudaStream_t stream) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  const auto N_half = a.size(2) / 2;
  dim3 dim_grid_stage(transform_rows, (N_half + kCudaBlockSize - 1) / kCudaBlockSize, a.size(0));

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  with_compact_twiddles<scalar_t>(
      inverse_twiddles, [&](auto inverse_twiddles_compact_acc) {

    for (int stage = start_stage; stage < end_stage;) {
      const int remaining = end_stage - stage;
      const int grouped_stages = std::min(grouped_stage_count, remaining);
      if (stage + grouped_stages == end_stage) {
        launch_inverse_ntt_compact_grouped_stages_cuda<scalar_t, OUTPUT>(
            a_acc, inverse_twiddles_compact_acc, params_acc, stage,
            grouped_stages, dim_grid_stage, stream);
      } else {
        launch_inverse_ntt_compact_grouped_stages_cuda<scalar_t>(
            a_acc, inverse_twiddles_compact_acc, params_acc, stage,
            grouped_stages, dim_grid_stage, stream);
      }
      stage += grouped_stages;
    }
  });
}

template <typename scalar_t, CompactInverseOutput OUTPUT = CompactInverseOutput::Unnormalized>
void launch_inverse_ntt_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count,
    const int transform_rows,
    cudaStream_t stream) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  // Execute the initial DIT stages from a shared-memory-resident tile, then run
  // the remaining global stages with computed indices.
  const int N = static_cast<int>(a.size(2));
  int logN = 0;
  for (int n = N; n > 1; n >>= 1) ++logN;
  const int effective_smem_stage_count =
      std::min({smem_stage_count, logN, fhelium::ntt::kNttMaxSharedMemoryLogN});

  if (effective_smem_stage_count > 0) {
    auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
    with_compact_twiddles<scalar_t>(
        inverse_twiddles, [&](auto inverse_twiddles_compact_acc) {
      dim3 dim_grid(transform_rows, N / kCudaBlockSize, a.size(0));
      if (effective_smem_stage_count == logN) {
        inverse_ntt_compact_smem_prefix_kernel<scalar_t, OUTPUT>
            <<<dim_grid, kCudaBlockSize / 2,
               kCudaBlockSize * sizeof(scalar_t), stream>>>(
                a_acc, inverse_twiddles_compact_acc, params_acc,
                effective_smem_stage_count);
      } else {
        inverse_ntt_compact_smem_prefix_kernel<scalar_t>
            <<<dim_grid, kCudaBlockSize / 2,
               kCudaBlockSize * sizeof(scalar_t), stream>>>(
                a_acc, inverse_twiddles_compact_acc, params_acc,
                effective_smem_stage_count);
      }
    });
  }

  launch_inverse_ntt_compact_grouped_stage_range_cuda<scalar_t, OUTPUT>(
      a,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      transform_rows,
      effective_smem_stage_count,
      logN,
      stream);
}
