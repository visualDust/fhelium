#pragma once

// Public standard-to-Montgomery conversion plus indexed radix-2 NTT entry
// points.

//------------------------------------------------------------------
// Standard-to-Montgomery forward NTT
//------------------------------------------------------------------

template <typename scalar_t>
void launch_forward_ntt_to_montgomery_indexed_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  // Retrieve device idx and stream.
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  // Use a preallocated pytorch stream.
  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  // The problem dimension.
  // Be careful. even_indices and odd_indices has half the length of the a.
  const auto C = a.size(1);
  const auto B = a.size(0);
  const auto logN = even_indices.size(0);
  const auto N_half = even_indices.size(1);
  const auto N = a.size(2);

  int dim_block = kCudaBlockSize;
  dim3 dim_grid_ntt(C, N_half / kCudaBlockSize, B);
  dim3 to_montgomery_grid(C, N / kCudaBlockSize, B);

  auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  const auto even_acc = FHELIUM_CUDA_ACCESSOR32(even_indices, int, 2);
  const auto odd_acc = FHELIUM_CUDA_ACCESSOR32(odd_indices, int, 2);
  const auto forward_twiddles_acc =
      FHELIUM_CUDA_ACCESSOR32(forward_twiddles, scalar_t, 3);

  // Convert the source residues to Montgomery representation.
  ntt_to_montgomery_inplace_kernel<scalar_t>
      <<<to_montgomery_grid, dim_block, 0, stream>>>(a_acc, params_acc);

  // ntt.
  for (int i = 0; i < logN; ++i) {
    forward_ntt_indexed_stage_kernel<scalar_t>
        <<<dim_grid_ntt, dim_block, 0, stream>>>(
            a_acc, even_acc, odd_acc, forward_twiddles_acc, params_acc, i);
  }
}

template <typename scalar_t>
void launch_forward_ntt_to_montgomery_indexed_out_cuda(
    const torch::Tensor a,
    torch::Tensor out,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  const auto params_acc = FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2);
  auto device_id = a.device().index();
  cudaSetDevice(device_id);

  auto stream = at::cuda::getCurrentCUDAStream(device_id);

  const auto C = a.size(1);
  const auto B = a.size(0);
  const auto logN = even_indices.size(0);
  const auto N_half = even_indices.size(1);
  const auto N = a.size(2);

  int dim_block = kCudaBlockSize;
  dim3 dim_grid_ntt(C, N_half / kCudaBlockSize, B);
  dim3 to_montgomery_grid(C, N / kCudaBlockSize, B);

  const auto a_acc = FHELIUM_CUDA_ACCESSOR32(a, scalar_t, 3);
  auto out_acc = FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3);
  const auto even_acc = FHELIUM_CUDA_ACCESSOR32(even_indices, int, 2);
  const auto odd_acc = FHELIUM_CUDA_ACCESSOR32(odd_indices, int, 2);
  const auto forward_twiddles_acc =
      FHELIUM_CUDA_ACCESSOR32(forward_twiddles, scalar_t, 3);

  // First stage is src -> out, so functional standard-to-Montgomery NTT avoids
  // cloning the source ciphertext shard before the subsequent in-place NTT
  // stages.
  ntt_to_montgomery_out_kernel<scalar_t>
      <<<to_montgomery_grid, dim_block, 0, stream>>>(
          out_acc, a_acc, params_acc);

  for (int i = 0; i < logN; ++i) {
    forward_ntt_indexed_stage_kernel<scalar_t>
        <<<dim_grid_ntt, dim_block, 0, stream>>>(
            out_acc, even_acc, odd_acc, forward_twiddles_acc, params_acc, i);
  }
}

void forward_ntt_to_montgomery_indexed_inplace_cuda(
    torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  auto batch_rows = view_rns_batch_3d(a, "standard_residues");
  // Dispatch to the correct data type.
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "forward_ntt_to_montgomery_indexed_inplace_cuda", ([&] {
        launch_forward_ntt_to_montgomery_indexed_inplace_cuda<scalar_t>(
            batch_rows,
            even_indices,
            odd_indices,
            forward_twiddles,
            rns_params);
      }));
}

torch::Tensor forward_ntt_to_montgomery_indexed_cuda(
    const torch::Tensor a,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  torch::Tensor out = torch::empty_like(a);
  const auto batch_rows = view_rns_batch_3d(a, "standard_residues");
  auto out_batch_rows = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(
      a.scalar_type(), "forward_ntt_to_montgomery_indexed_cuda", ([&] {
        launch_forward_ntt_to_montgomery_indexed_out_cuda<scalar_t>(
            batch_rows,
            out_batch_rows,
            even_indices,
            odd_indices,
            forward_twiddles,
            rns_params);
      }));
  return out;
}
