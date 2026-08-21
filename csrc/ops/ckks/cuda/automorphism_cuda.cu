#include "ckks_cuda.h"

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_batch.h"

// Galois automorphism representation requirements. Input is integral CUDA
// [*batch, limb, coefficient_or_ntt_index]; output is newly allocated with the
// same shape, dtype/device, prime rows, domain, representation, and lazy
// or canonical range. source_indices is int32 [N] destination-to-source order.
// The coefficient variant also consumes int8 source_sign [N] and integral
// twice_modulus [limb] to implement $\sigma_g:X\mapsto X^g$ modulo $X^N+1$;
// it returns canonical residues. The NTT variant is a pure gather of NTT
// evaluations. All inputs are read-only and no output aliases an input.
template <typename scalar_t>
__global__ void coefficient_galois_automorphism_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<int32_t, 1> source_indices,
    const CudaTensorAccessor32<int8_t, 1> source_sign,
    const CudaTensorAccessor32<scalar_t, 1> twice_modulus) {
  const int row = blockIdx.x;
  const int destination = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (destination >= residues.size(2)) return;
  scalar_t value = residues[batch][row][source_indices[destination]];
  if (source_sign[destination] == static_cast<int8_t>(-1)) value = -value;
  value = shift_residue_positive(value, twice_modulus[row]);
  out[batch][row][destination] =
      canonicalize_lazy_residue(value, twice_modulus[row]);
}

template <typename scalar_t>
__global__ void ntt_galois_automorphism_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<int32_t, 1> source_indices) {
  const int row = blockIdx.x;
  const int destination = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (destination >= residues.size(2)) return;
  out[batch][row][destination] =
      residues[batch][row][source_indices[destination]];
}

void validate_automorphism(const torch::Tensor& residues,
                           const torch::Tensor& source_indices,
                           const char* operation) {
  TORCH_CHECK(
      source_indices.dim() == 1 && source_indices.size(0) == residues.size(2),
      operation,
      " source index count must match coefficient count");
}

torch::Tensor apply_coefficient_galois_automorphism_cuda(
    const torch::Tensor residues,
    const torch::Tensor source_indices,
    const torch::Tensor source_sign,
    const torch::Tensor twice_modulus) {
  auto out = torch::empty_like(residues);
  const auto input = view_rns_batch_3d(residues, "residues");
  auto output = view_rns_batch_3d(out, "out");
  validate_automorphism(
      input, source_indices, "apply_coefficient_galois_automorphism");
  TORCH_CHECK(source_sign.dim() == 1 && source_sign.size(0) == input.size(2),
              "apply_coefficient_galois_automorphism source sign count must "
              "match coefficient count");
  check_rns_row_vector(twice_modulus,
                       input.size(1),
                       "apply_coefficient_galois_automorphism",
                       "twice_modulus");
  const int device = residues.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(input.size(1),
            (input.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            input.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "apply_coefficient_galois_automorphism", [&] {
        coefficient_galois_automorphism_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(input, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(source_indices, int32_t, 1),
                FHELIUM_CUDA_ACCESSOR32(source_sign, int8_t, 1),
                FHELIUM_CUDA_ACCESSOR32(twice_modulus, scalar_t, 1));
      });
  return out;
}

torch::Tensor apply_ntt_galois_automorphism_cuda(
    const torch::Tensor residues_ntt, const torch::Tensor source_indices) {
  auto out = torch::empty_like(residues_ntt);
  const auto input = view_rns_batch_3d(residues_ntt, "residues_ntt");
  auto output = view_rns_batch_3d(out, "out");
  validate_automorphism(input, source_indices, "apply_ntt_galois_automorphism");
  const int device = residues_ntt.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(input.size(1),
            (input.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            input.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      residues_ntt.scalar_type(), "apply_ntt_galois_automorphism", [&] {
        ntt_galois_automorphism_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(input, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(source_indices, int32_t, 1));
      });
  return out;
}
