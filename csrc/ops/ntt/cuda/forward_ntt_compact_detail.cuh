#pragma once

// Private compact NTT kernels and launch helpers.

template <typename scalar_t>
__global__ void forward_ntt_compact_stage_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> forward_twiddles_compact_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int stage) {
  // Direct-index single DIF stage with compact bit-reversed twiddle rows.
  // Compact backends compute indices from the stage and compact twiddle rows.
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int j = blockIdx.y * kCudaBlockSize + threadIdx.x;
  const int N = static_cast<int>(a_acc.size(2));
  const int logN = __ffs(N) - 1;
  const int t_log = logN - stage - 1;
  const int t = 1 << t_log;
  const int span_id = j >> t_log;
  const int rr = j & (t - 1);
  const int even_j = (span_id << (t_log + 1)) + rr;
  const int odd_j = even_j + t;

  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][row];

  const scalar_t U = a_acc[batch][row][even_j];
  const int twiddle_group = j >> t_log;
  const scalar_t S =
      forward_twiddles_compact_acc[row][(1 << stage) + twiddle_group];
  const scalar_t O = a_acc[batch][row][odd_j];
  const scalar_t V = montgomery_mul(
      S, O, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);

  const scalar_t UplusV = U + V;
  const scalar_t UminusV = U + twice_modulus - V;

  a_acc[batch][row][even_j] =
      (UplusV < twice_modulus) ? UplusV : UplusV - twice_modulus;
  a_acc[batch][row][odd_j] =
      (UminusV < twice_modulus) ? UminusV : UminusV - twice_modulus;
}

template <typename scalar_t, int GROUPED_STAGE_COUNT>
__global__ void forward_ntt_compact_grouped_stages_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> forward_twiddles_compact_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage) {
  static_assert(GROUPED_STAGE_COUNT == 2 || GROUPED_STAGE_COUNT == 3 ||
                    GROUPED_STAGE_COUNT == 4,
                "supported compact forward kernels: 4/8/16");
  constexpr int GROUP = 1 << GROUPED_STAGE_COUNT;

  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tuple = blockIdx.y * kCudaBlockSize + threadIdx.x;
  const int N = static_cast<int>(a_acc.size(2));
  const int logN = __ffs(N) - 1;
  const int final_t_log = logN - start_stage - GROUPED_STAGE_COUNT;
  const int final_t = 1 << final_t_log;
  if (tuple >= (N >> GROUPED_STAGE_COUNT)) return;

  const int initial_group = tuple >> final_t_log;
  const int j = tuple & (final_t - 1);
  const int base = (initial_group << (GROUPED_STAGE_COUNT + final_t_log)) + j;

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
  for (int k = 0; k < GROUP; ++k) x[k] = a_acc[batch][row][base + k * final_t];

#pragma unroll
  for (int r = 0; r < GROUPED_STAGE_COUNT; ++r) {
    const int stage = start_stage + r;
    const int stride_log = GROUPED_STAGE_COUNT - r - 1;
    const int stride = 1 << stride_log;
#pragma unroll
    for (int kk = 0; kk < GROUP; kk += (2 * stride)) {
      const int subgroup_within_initial_group = kk >> (stride_log + 1);
      const int global_group =
          (initial_group << r) + subgroup_within_initial_group;
#pragma unroll
      for (int rr = 0; rr < stride; ++rr) {
        const int lo = kk + rr;
        const int hi = lo + stride;
        const scalar_t U = x[lo];
        const int twiddle_group = global_group;
        const scalar_t S =
            forward_twiddles_compact_acc[twiddle_row]
                                        [(1 << stage) + twiddle_group];
        const scalar_t O = x[hi];
        const scalar_t V = montgomery_mul(S,
                                          O,
                                          modulus_lo,
                                          modulus_hi,
                                          neg_inv_modulus_lo,
                                          neg_inv_modulus_hi);

        const scalar_t UplusV = U + V;
        const scalar_t UminusV = U + twice_modulus - V;
        x[lo] = (UplusV < twice_modulus) ? UplusV : UplusV - twice_modulus;
        x[hi] = (UminusV < twice_modulus) ? UminusV : UminusV - twice_modulus;
      }
    }
  }

#pragma unroll
  for (int k = 0; k < GROUP; ++k) a_acc[batch][row][base + k * final_t] = x[k];
}

template <typename scalar_t>
__global__ void forward_ntt_compact_smem_tail_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<scalar_t, 2> forward_twiddles_compact_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int start_stage) {
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

  const int logN = __ffs(N) - 1;
  for (int stage = start_stage; stage < logN; ++stage) {
    const int t_log = logN - stage - 1;
    const int t = 1 << t_log;
    const int span_id = thread_offset >> t_log;
    const int rr = thread_offset & (t - 1);
    const int lo = (span_id << (t_log + 1)) + rr;
    const int hi = lo + t;
    const int global_lo = tile_base + lo;
    const scalar_t U = smem[lo];
    const int twiddle_group = global_lo >> (t_log + 1);
    const scalar_t S =
        forward_twiddles_compact_acc[twiddle_row][(1 << stage) + twiddle_group];
    const scalar_t O = smem[hi];
    const scalar_t V = montgomery_mul(
        S, O, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
    const scalar_t UplusV = U + V;
    const scalar_t UminusV = U + twice_modulus - V;
    smem[lo] = (UplusV < twice_modulus) ? UplusV : UplusV - twice_modulus;
    smem[hi] = (UminusV < twice_modulus) ? UminusV : UminusV - twice_modulus;
    __syncthreads();
  }

  a_acc[batch][row][tile_base + thread_offset] = smem[thread_offset];
  a_acc[batch][row][tile_base + thread_offset + HALF_TILE_COEFFICIENT_COUNT] =
      smem[thread_offset + HALF_TILE_COEFFICIENT_COUNT];
}

template <typename scalar_t>
void launch_forward_ntt_compact_grouped_stages_cuda(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    CudaTensorAccessor32<scalar_t, 2> forward_twiddles_compact_acc,
    CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int stage,
    const int grouped_stage_count,
    const dim3 dim_grid_stage,
    cudaStream_t stream) {
  if (grouped_stage_count == 1) {
    forward_ntt_compact_stage_kernel<scalar_t>
        <<<dim_grid_stage, kCudaBlockSize, 0, stream>>>(
            a_acc, forward_twiddles_compact_acc, params_acc, stage);
    return;
  }

  const int C = dim_grid_stage.x;
  const int N = static_cast<int>(a_acc.size(2));
  const int tuples_per_prime = N >> grouped_stage_count;
  const dim3 dim_grid(C,
                      (tuples_per_prime + kCudaBlockSize - 1) / kCudaBlockSize,
                      dim_grid_stage.z);
  if (grouped_stage_count == 4) {
    forward_ntt_compact_grouped_stages_kernel<scalar_t, 4>
        <<<dim_grid, kCudaBlockSize, 0, stream>>>(
            a_acc, forward_twiddles_compact_acc, params_acc, stage);
  } else if (grouped_stage_count == 3) {
    forward_ntt_compact_grouped_stages_kernel<scalar_t, 3>
        <<<dim_grid, kCudaBlockSize, 0, stream>>>(
            a_acc, forward_twiddles_compact_acc, params_acc, stage);
  } else {
    forward_ntt_compact_grouped_stages_kernel<scalar_t, 2>
        <<<dim_grid, kCudaBlockSize, 0, stream>>>(
            a_acc, forward_twiddles_compact_acc, params_acc, stage);
  }
}

template <typename scalar_t>
void launch_forward_ntt_compact_grouped_stage_range_cuda(
    torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int transform_rows,
    const int start_stage,
    const int end_stage,
    cudaStream_t stream) {
  const auto N = a.size(2);
  dim3 dim_grid_stage(transform_rows, (N / 2) / kCudaBlockSize, a.size(0));

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  const auto forward_twiddles_compact_acc =
      FHELIUM_CUDA_ACCESSOR32(forward_twiddles, scalar_t, 2);
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);

  for (int stage = start_stage; stage < end_stage;) {
    const int remaining = end_stage - stage;
    const int grouped_stages = std::min(grouped_stage_count, remaining);
    launch_forward_ntt_compact_grouped_stages_cuda<scalar_t>(
        a_acc,
        forward_twiddles_compact_acc,
        params_acc,
        stage,
        grouped_stages,
        dim_grid_stage,
        stream);
    stage += grouped_stages;
  }
}

template <typename scalar_t>
void launch_forward_ntt_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count,
    const int transform_rows,
    cudaStream_t stream) {
  // Run global DIF stages with computed indices, then execute the tail stages
  // from a shared-memory-resident tile.
  const int N = static_cast<int>(a.size(2));
  int logN = 0;
  for (int n = N; n > 1; n >>= 1) ++logN;
  const int effective_smem_stage_count =
      std::min({smem_stage_count, logN, fhelium::ntt::kNttMaxSharedMemoryLogN});
  const int smem_start_stage = logN - effective_smem_stage_count;

  launch_forward_ntt_compact_grouped_stage_range_cuda<scalar_t>(
      a,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      transform_rows,
      0,
      smem_start_stage,
      stream);

  if (effective_smem_stage_count > 0) {
    auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
    const auto forward_twiddles_compact_acc =
        FHELIUM_CUDA_ACCESSOR32(forward_twiddles, scalar_t, 2);
    const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
    dim3 dim_grid(transform_rows, N / kCudaBlockSize, a.size(0));
    forward_ntt_compact_smem_tail_kernel<scalar_t>
        <<<dim_grid,
           kCudaBlockSize / 2,
           kCudaBlockSize * sizeof(scalar_t),
           stream>>>(
            a_acc, forward_twiddles_compact_acc, params_acc, smem_start_stage);
  }
}
