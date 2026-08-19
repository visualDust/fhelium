#pragma once

// C2 compact radix-2 consumer fusion. The one-digit coefficient-domain QP
// scratch is disposable: global kernels execute the prefix DIF stages, while
// the final eight stages remain in a 256-coefficient shared-memory tile and
// feed the unbatched key-switch-key table directly. Final digit NTT values are
// never written to global memory.
template <typename scalar_t>
__global__ void forward_ntt_compact_smem_tail_keyswitch_accumulate_kernel(
    CudaTensorAccessor32<scalar_t, 3> digit,
    const CudaTensorAccessor32<scalar_t, 2> forward_twiddles,
    const CudaTensorAccessor32<scalar_t, 2> params,
    const CudaTensorAccessor32<scalar_t, 3> key_digit,
    CudaTensorAccessor32<scalar_t, 3> accumulator0,
    CudaTensorAccessor32<scalar_t, 3> accumulator1,
    const int start_stage,
    const int key_row_start) {
  const int row = blockIdx.x;
  const int tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int thread_offset = threadIdx.x;
  constexpr int HALF_TILE = kCudaBlockSize / 2;
  const int N = static_cast<int>(digit.size(2));
  const int tile_base = tile * kCudaBlockSize;
  extern __shared__ unsigned char shared_raw[];
  scalar_t* values = reinterpret_cast<scalar_t*>(shared_raw);

  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];

  values[thread_offset] = digit[batch][row][tile_base + thread_offset];
  values[thread_offset + HALF_TILE] =
      digit[batch][row][tile_base + thread_offset + HALF_TILE];
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
    const int twiddle_group = global_lo >> (t_log + 1);
    const scalar_t U = values[lo];
    const scalar_t V =
        montgomery_mul(forward_twiddles[row][(1 << stage) + twiddle_group],
                       values[hi],
                       modulus_lo,
                       modulus_hi,
                       neg_inv_modulus_lo,
                       neg_inv_modulus_hi);
    const scalar_t sum = U + V;
    const scalar_t difference = U + twice_modulus - V;
    values[lo] = sum < twice_modulus ? sum : sum - twice_modulus;
    values[hi] =
        difference < twice_modulus ? difference : difference - twice_modulus;
    __syncthreads();
  }

  const int key_row = key_row_start + row;
#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    const int local_coefficient = thread_offset + lane * HALF_TILE;
    const int coefficient = tile_base + local_coefficient;
    const scalar_t value = values[local_coefficient];
    const scalar_t product0 = montgomery_mul(value,
                                             key_digit[0][key_row][coefficient],
                                             modulus_lo,
                                             modulus_hi,
                                             neg_inv_modulus_lo,
                                             neg_inv_modulus_hi);
    const scalar_t product1 = montgomery_mul(value,
                                             key_digit[1][key_row][coefficient],
                                             modulus_lo,
                                             modulus_hi,
                                             neg_inv_modulus_lo,
                                             neg_inv_modulus_hi);
    accumulator0[batch][row][coefficient] = add_lazy_residues(
        accumulator0[batch][row][coefficient], product0, twice_modulus);
    accumulator1[batch][row][coefficient] = add_lazy_residues(
        accumulator1[batch][row][coefficient], product1, twice_modulus);
  }
}

template <typename scalar_t>
void launch_forward_ntt_compact_keyswitch_accumulate_cuda(
    torch::Tensor coefficient_digit,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const torch::Tensor key_digit,
    torch::Tensor accumulator0,
    torch::Tensor accumulator1,
    const int key_row_start,
    cudaStream_t stream) {
  constexpr int kVerifiedGroupedStageCount = 4;
  constexpr int kVerifiedTailStageCount = 8;
  const int N = static_cast<int>(coefficient_digit.size(2));
  int logN = 0;
  for (int extent = N; extent > 1; extent >>= 1) ++logN;
  const int tail_start_stage = logN - kVerifiedTailStageCount;
  const int row_count = static_cast<int>(coefficient_digit.size(1));

  launch_forward_ntt_compact_grouped_stage_range_cuda<scalar_t>(
      coefficient_digit,
      forward_twiddles,
      rns_params,
      kVerifiedGroupedStageCount,
      row_count,
      0,
      tail_start_stage,
      stream);

  const dim3 grid(row_count, N / kCudaBlockSize, coefficient_digit.size(0));
  forward_ntt_compact_smem_tail_keyswitch_accumulate_kernel<scalar_t>
      <<<grid, kCudaBlockSize / 2, kCudaBlockSize * sizeof(scalar_t), stream>>>(
          FHELIUM_CUDA_ACCESSOR32(coefficient_digit, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(forward_twiddles, scalar_t, 2),
          FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2),
          FHELIUM_CUDA_ACCESSOR32(key_digit, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(accumulator0, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(accumulator1, scalar_t, 3),
          tail_start_stage,
          key_row_start);
}

void forward_ntt_montgomery_compact_keyswitch_accumulate_inplace_cuda(
    torch::Tensor coefficient_digit_qp,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const torch::Tensor key_digit_qp,
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    const int64_t key_row_start) {
  auto digit = view_rns_batch_3d(coefficient_digit_qp, "coefficient_digit_qp");
  auto accumulator0 = view_rns_batch_3d(accumulator0_qp, "accumulator0_qp");
  auto accumulator1 = view_rns_batch_3d(accumulator1_qp, "accumulator1_qp");
  check_rns_binary_3d(
      accumulator0, digit, "forward_ntt_compact_keyswitch_accumulate", false);
  check_rns_binary_3d(
      accumulator1, digit, "forward_ntt_compact_keyswitch_accumulate", false);
  TORCH_CHECK(key_digit_qp.dim() == 3 && key_digit_qp.size(0) == 2,
              "key_digit_qp must have [2, QP row, coefficient] layout");
  TORCH_CHECK(key_row_start >= 0 &&
                  key_row_start + digit.size(1) <= key_digit_qp.size(1),
              "active rows exceed key_digit_qp row extent");
  TORCH_CHECK(key_digit_qp.size(2) == digit.size(2),
              "key digit coefficient extent mismatch");

  const int device = coefficient_digit_qp.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  AT_DISPATCH_INTEGRAL_TYPES(
      coefficient_digit_qp.scalar_type(),
      "forward_ntt_montgomery_compact_keyswitch_accumulate",
      [&] {
        launch_forward_ntt_compact_keyswitch_accumulate_cuda<scalar_t>(
            digit,
            forward_twiddles,
            rns_params,
            key_digit_qp,
            accumulator0,
            accumulator1,
            static_cast<int>(key_row_start),
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
