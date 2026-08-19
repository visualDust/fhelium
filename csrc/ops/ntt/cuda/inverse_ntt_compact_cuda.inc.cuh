#pragma once

// Compact-table grouped INTT entry points. Stage grouping comes from the
// immutable algorithm policy. Shared-memory coverage comes from the native
// production default or a separate diagnostic operator.

template <typename scalar_t>
void launch_inverse_ntt_montgomery_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  const auto N = a.size(2);
  dim3 normalization_grid(C, N / kCudaBlockSize, a.size(0));

  launch_inverse_ntt_compact_grouped_smem_cuda<scalar_t>(a,
                                                         inverse_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  inverse_ntt_normalize_montgomery_kernel<scalar_t>
      <<<normalization_grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
}

template <typename scalar_t>
void launch_inverse_ntt_to_standard_lazy_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  const auto N = a.size(2);
  dim3 normalization_grid(C, N / kCudaBlockSize, a.size(0));

  launch_inverse_ntt_compact_grouped_smem_cuda<scalar_t>(a,
                                                         inverse_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  inverse_ntt_normalize_to_standard_lazy_kernel<scalar_t>
      <<<normalization_grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
}

template <typename scalar_t>
void launch_inverse_ntt_to_standard_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  const auto N = a.size(2);
  dim3 normalization_grid(C, N / kCudaBlockSize, a.size(0));

  launch_inverse_ntt_compact_grouped_smem_cuda<scalar_t>(a,
                                                         inverse_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  inverse_ntt_normalize_to_standard_kernel<scalar_t>
      <<<normalization_grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
}

template <typename scalar_t>
void launch_inverse_ntt_to_centered_compact_grouped_smem_cuda(
    torch::Tensor a,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int grouped_stage_count,
    const int smem_stage_count) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const int C = static_cast<int>(a.size(1));
  const auto N = a.size(2);
  dim3 normalization_grid(C, N / kCudaBlockSize, a.size(0));

  launch_inverse_ntt_compact_grouped_smem_cuda<scalar_t>(a,
                                                         inverse_twiddles,
                                                         rns_params,
                                                         grouped_stage_count,
                                                         smem_stage_count,
                                                         C,
                                                         stream);

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  inverse_ntt_normalize_to_centered_kernel<scalar_t>
      <<<normalization_grid, kCudaBlockSize, 0, stream>>>(a_acc, params_acc);
}

#define DEFINE_COMPACT_INVERSE_ENTRY(ENTRY_NAME, LAUNCH_NAME)               \
  void ENTRY_NAME(torch::Tensor a,                                          \
                  const torch::Tensor inverse_twiddles,                     \
                  const torch::Tensor rns_params,                           \
                  const int64_t grouped_stage_count,                        \
                  const int64_t smem_stage_count) {                         \
    auto batch_rows = view_rns_batch_3d(a, "montgomery_residues");          \
    AT_DISPATCH_INTEGRAL_TYPES(a.scalar_type(), #ENTRY_NAME, ([&] {         \
                                 LAUNCH_NAME<scalar_t>(                     \
                                     batch_rows,                            \
                                     inverse_twiddles,                      \
                                     rns_params,                            \
                                     static_cast<int>(grouped_stage_count), \
                                     static_cast<int>(smem_stage_count));   \
                               }));                                         \
  }

DEFINE_COMPACT_INVERSE_ENTRY(
    inverse_ntt_montgomery_compact_grouped_smem_inplace_cuda,
    launch_inverse_ntt_montgomery_compact_grouped_smem_cuda)
DEFINE_COMPACT_INVERSE_ENTRY(
    inverse_ntt_to_standard_lazy_compact_grouped_smem_inplace_cuda,
    launch_inverse_ntt_to_standard_lazy_compact_grouped_smem_cuda)
DEFINE_COMPACT_INVERSE_ENTRY(
    inverse_ntt_to_standard_compact_grouped_smem_inplace_cuda,
    launch_inverse_ntt_to_standard_compact_grouped_smem_cuda)
DEFINE_COMPACT_INVERSE_ENTRY(
    inverse_ntt_to_centered_compact_grouped_smem_inplace_cuda,
    launch_inverse_ntt_to_centered_compact_grouped_smem_cuda)

#undef DEFINE_COMPACT_INVERSE_ENTRY
