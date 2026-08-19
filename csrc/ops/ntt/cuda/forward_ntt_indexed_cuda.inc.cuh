#pragma once

// Public transform-only indexed radix-2 NTT entry points.

template <typename scalar_t>
void launch_forward_ntt_montgomery_indexed_cuda(
    torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  const auto C = a.size(1);
  const auto B = a.size(0);
  const auto logN = even_indices.size(0);
  const auto N = even_indices.size(1);

  int dim_block = kCudaBlockSize;
  dim3 dim_grid(C, N / kCudaBlockSize, B);

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  const auto even_acc = FHELIUM_CUDA_ACCESSOR32(even_indices, int, 2);
  const auto odd_acc = FHELIUM_CUDA_ACCESSOR32(odd_indices, int, 2);
  const auto forward_twiddles_acc =
      FHELIUM_CUDA_ACCESSOR32(forward_twiddles, scalar_t, 3);
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);

  for (int i = 0; i < logN; ++i) {
    forward_ntt_indexed_stage_kernel<scalar_t>
        <<<dim_grid, dim_block, 0, stream>>>(
            a_acc, even_acc, odd_acc, forward_twiddles_acc, params_acc, i);
  }
}

void forward_ntt_montgomery_indexed_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  auto batch_rows = view_rns_batch_3d(a, "montgomery_residues");
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "forward_ntt_montgomery_indexed_cuda", ([&] {
        launch_forward_ntt_montgomery_indexed_cuda<scalar_t>(batch_rows,
                                                             even_indices,
                                                             odd_indices,
                                                             forward_twiddles,
                                                             rns_params);
      }));
}
