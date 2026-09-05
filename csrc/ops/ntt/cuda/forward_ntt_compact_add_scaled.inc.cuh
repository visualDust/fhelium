#pragma once

// Forward NTT of a disposable coefficient/standard correction, with the final
// write forming NTT(correction) + multiplier * NTT_addend in Montgomery form.
// The full Q addend remains in NTT form; neither it nor the row multipliers is
// modified. This epilogue serves the represented NTT-domain ModDown operation.
template <typename scalar_t>
__global__ void forward_ntt_compact_add_scaled_tail_kernel(
    CudaTensorAccessor32<scalar_t, 3> digit,
    const CudaTensorAccessor32<scalar_t, 2> forward_twiddles,
    const CudaTensorAccessor32<scalar_t, 2> params,
    const CudaTensorAccessor32<scalar_t, 3> q_source,
    const CudaTensorAccessor32<scalar_t, 1> p_inverse,
    const int start_stage) {
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

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    const int local = thread_offset + lane * HALF_TILE;
    const int coefficient = tile_base + local;
    const scalar_t scaled = montgomery_mul(
        q_source[batch][row][coefficient], p_inverse[row],
        modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
    digit[batch][row][coefficient] =
        add_lazy_residues(values[local], scaled, twice_modulus);
  }
}

void forward_ntt_to_montgomery_compact_add_scaled_inplace_cuda(
    torch::Tensor correction,
    const torch::Tensor q,
    const torch::Tensor inv,
    const torch::Tensor tw,
    const torch::Tensor params,
    const int64_t group) {
  fhelium::ntt::validate_compact_tables(correction, tw, params, group);
  TORCH_CHECK(q.device() == correction.device() &&
                  q.scalar_type() == correction.scalar_type() &&
                  q.sizes() == correction.sizes(),
              "NTT addend must match source shape, dtype and device");
  TORCH_CHECK(inv.device() == correction.device() &&
                  inv.scalar_type() == correction.scalar_type() &&
                  inv.dim() == 1 && inv.size(0) == correction.size(-2),
              "NTT multiplier must match prime rows, dtype and device");
  fhelium::ntt::validate_no_residue_overlap(correction, q, "NTT addend");
  fhelium::ntt::validate_no_residue_overlap(correction, inv, "NTT multiplier");
  auto destination = view_rns_batch_3d(correction, "correction");
  auto source = view_rns_batch_3d(q, "q");
  cudaSetDevice(correction.device().index());
  auto stream = at::cuda::getCurrentCUDAStream(correction.device().index());
  const int N = static_cast<int>(destination.size(2));
  int logN = 0;
  for (int extent = N; extent > 1; extent >>= 1) ++logN;
  AT_DISPATCH_INTEGRAL_TYPES(
      correction.scalar_type(), "forward_ntt_compact_add_scaled", [&] {
        auto data = FHELIUM_CUDA_ACCESSOR32(destination, scalar_t, 3);
        auto parameters = FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2);
        dim3 grid(destination.size(1), N / kCudaBlockSize,
                  destination.size(0));
        ntt_to_montgomery_inplace_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(data, parameters);
        launch_forward_ntt_compact_grouped_stage_range_cuda<scalar_t>(
            destination, tw, params, static_cast<int>(group),
            static_cast<int>(destination.size(1)), 0, logN - 8, stream);
        forward_ntt_compact_add_scaled_tail_kernel<scalar_t>
            <<<grid, kCudaBlockSize / 2,
               kCudaBlockSize * sizeof(scalar_t), stream>>>(
                data, FHELIUM_CUDA_ACCESSOR32(tw, scalar_t, 2), parameters,
                FHELIUM_CUDA_ACCESSOR32(source, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(inv, scalar_t, 1), logN - 8);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
