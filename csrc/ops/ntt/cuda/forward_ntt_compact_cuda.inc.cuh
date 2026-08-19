#pragma once

// Compact-table grouped NTT entry points. Stage grouping comes from the
// immutable algorithm policy. Shared-memory coverage comes from the native
// production default or a separate diagnostic operator.

template <typename scalar_t>
void launch_forward_ntt_montgomery_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  launch_forward_ntt_compact_grouped_smem_cuda<scalar_t>(a,
                                                         forward_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);
}

void forward_ntt_montgomery_compact_grouped_smem_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count,
    const int64_t smem_stage_count) {
  auto batch_rows = view_rns_batch_3d(a, "montgomery_residues");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(),
      "forward_ntt_montgomery_compact_grouped_smem_cuda",
      ([&] {
        launch_forward_ntt_montgomery_compact_grouped_smem_cuda<scalar_t>(
            batch_rows,
            forward_twiddles,
            rns_params,
            static_cast<int>(grouped_stage_count),
            static_cast<int>(smem_stage_count));
      }));
}

template <typename scalar_t>
void launch_forward_ntt_to_montgomery_compact_grouped_smem_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  const int N = static_cast<int>(a.size(2));
  dim3 to_montgomery_grid(C, N / kCudaBlockSize, a.size(0));

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  ntt_to_montgomery_inplace_kernel<scalar_t>
      <<<to_montgomery_grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);

  launch_forward_ntt_compact_grouped_smem_cuda<scalar_t>(a,
                                                         forward_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);
}

template <typename scalar_t>
void launch_forward_ntt_to_montgomery_compact_grouped_smem_out_cuda(
    const torch::Tensor a,
    torch::Tensor out,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  const int N = static_cast<int>(a.size(2));
  dim3 to_montgomery_grid(C, N / kCudaBlockSize, a.size(0));

  const auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  auto out_acc = FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3);
  ntt_to_montgomery_out_kernel<scalar_t>
      <<<to_montgomery_grid, kCudaBlockSize, 0, stream>>>(
          out_acc, a_acc, params_acc);

  launch_forward_ntt_compact_grouped_smem_cuda<scalar_t>(out,
                                                         forward_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);
}

void forward_ntt_to_montgomery_compact_grouped_smem_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count,
    const int64_t smem_stage_count) {
  auto batch_rows = view_rns_batch_3d(a, "standard_residues");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(),
      "forward_ntt_to_montgomery_compact_grouped_smem_cuda",
      ([&] {
        launch_forward_ntt_to_montgomery_compact_grouped_smem_inplace_cuda<
            scalar_t>(batch_rows,
                      forward_twiddles,
                      rns_params,
                      static_cast<int>(grouped_stage_count),
                      static_cast<int>(smem_stage_count));
      }));
}

torch::Tensor forward_ntt_to_montgomery_compact_grouped_smem_cuda(
    const torch::Tensor a,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count,
    const int64_t smem_stage_count) {
  torch::Tensor out = torch::empty_like(a);
  const auto batch_rows = view_rns_batch_3d(a, "standard_residues");
  auto out_batch_rows = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(),
      "forward_ntt_to_montgomery_compact_grouped_smem_cuda",
      ([&] {
        launch_forward_ntt_to_montgomery_compact_grouped_smem_out_cuda<
            scalar_t>(batch_rows,
                      out_batch_rows,
                      forward_twiddles,
                      rns_params,
                      static_cast<int>(grouped_stage_count),
                      static_cast<int>(smem_stage_count));
      }));
  return out;
}
