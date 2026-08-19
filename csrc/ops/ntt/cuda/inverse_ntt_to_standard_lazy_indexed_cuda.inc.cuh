#pragma once

// Public indexed radix-2 inverse NTT lazy-standard-output entry points.

// -------------------------------------------------------------------
// Inverse NTT with lazy standard output
// -------------------------------------------------------------------

template <typename scalar_t>
void launch_inverse_ntt_to_standard_lazy_indexed_cuda(
    torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  // Retrieve the device index, then set the corresponding device and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  // Be careful. even_indices and odd_indices has half the length of the a.
  const auto C = a.size(1);
  const auto B = a.size(0);
  // printf("modulus_lo.size(0), a.size(0) = %ld, %ld = %d\n",
  //          modulus_lo.size(0), a.size(0));
  const auto logN = even_indices.size(0);
  const auto N_half = even_indices.size(1);
  const auto N = a.size(2);

  int dim_block = kCudaBlockSize;
  dim3 dim_grid_ntt(C, N_half / kCudaBlockSize, B);
  dim3 to_montgomery_grid(C, N / kCudaBlockSize, B);

  // make the packed accessors.
  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  const auto even_acc = FHELIUM_CUDA_ACCESSOR32(even_indices, int, 2);
  const auto odd_acc = FHELIUM_CUDA_ACCESSOR32(odd_indices, int, 2);
  const auto inverse_twiddles_acc =
      FHELIUM_CUDA_ACCESSOR32(inverse_twiddles, scalar_t, 3);

  for (int i = 0; i < logN; ++i) {
    inverse_ntt_indexed_stage_kernel<scalar_t>
        <<<dim_grid_ntt, dim_block, 0, stream>>>(
            a_acc, even_acc, odd_acc, inverse_twiddles_acc, params_acc, i);
  }

  // Normalize and Exit
  inverse_ntt_normalize_to_standard_lazy_kernel<scalar_t>
      <<<to_montgomery_grid, dim_block, 0, stream>>>(a_acc, params_acc);
}

void inverse_ntt_to_standard_lazy_indexed_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params) {
  auto batch_rows = view_rns_batch_3d(a, "montgomery_residues");
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "inverse_ntt_to_standard_lazy_indexed_cuda", ([&] {
        launch_inverse_ntt_to_standard_lazy_indexed_cuda<scalar_t>(
            batch_rows,
            even_indices,
            odd_indices,
            inverse_twiddles,
            rns_params);
      }));
}
