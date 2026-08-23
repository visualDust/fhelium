#include <torch/library.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

#include <cstdint>

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../ciphertext_components.h"

namespace {

using fhelium::ckks::ThreeComponentMultiplyOutput;
using fhelium::ckks::TwoComponentMultiplyInputs;

constexpr const char* kOperation = "multiply_two_component_ntt_montgomery";

void check_cuda_status(const cudaError_t status,
                       const char* operation,
                       const char* action) {
  TORCH_CHECK(status == cudaSuccess,
              operation,
              " could not ",
              action,
              ": ",
              cudaGetErrorString(status));
}

template <typename scalar_t>
__global__ void fused_scalar_kernel(
    CudaTensorAccessor32<scalar_t, 3> d0,
    CudaTensorAccessor32<scalar_t, 3> d1,
    CudaTensorAccessor32<scalar_t, 3> d2,
    const CudaTensorAccessor32<scalar_t, 3> lhs0,
    const CudaTensorAccessor32<scalar_t, 3> lhs1,
    const CudaTensorAccessor32<scalar_t, 3> rhs0,
    const CudaTensorAccessor32<scalar_t, 3> rhs1,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs0.size(2)) return;

  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const scalar_t a0 = lhs0[batch][row][coefficient];
  const scalar_t a1 = lhs1[batch][row][coefficient];
  const scalar_t b0 = rhs0[batch][row][coefficient];
  const scalar_t b1 = rhs1[batch][row][coefficient];
  d0[batch][row][coefficient] = montgomery_mul(
      a0, b0, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
  const scalar_t cross01 = montgomery_mul(
      a0, b1, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
  const scalar_t cross10 = montgomery_mul(
      a1, b0, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
  d1[batch][row][coefficient] =
      add_lazy_residues(cross01, cross10, params[RNS_PARAM_TWICE_MODULUS][row]);
  d2[batch][row][coefficient] = montgomery_mul(
      a1, b1, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
}

template <typename scalar_t>
void launch(const TwoComponentMultiplyInputs& inputs,
            const ThreeComponentMultiplyOutput& output,
            const torch::Tensor& params) {
  const int device = inputs.lhs0.device().index();
  check_cuda_status(
      cudaSetDevice(device), kOperation, "select the input CUDA device");
  auto stream = at::cuda::getCurrentCUDAStream(device);
  const dim3 grid(inputs.lhs0.size(1),
                  (inputs.lhs0.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  inputs.lhs0.size(0));
  fused_scalar_kernel<scalar_t><<<grid, kCudaBlockSize, 0, stream>>>(
      FHELIUM_CUDA_ACCESSOR32(output.d0, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(output.d1, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(output.d2, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(inputs.lhs0, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(inputs.lhs1, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(inputs.rhs0, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(inputs.rhs1, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

torch::Tensor multiply_two_component_cuda(const torch::Tensor lhs_components,
                                          const torch::Tensor rhs_components,
                                          const torch::Tensor rns_params) {
  const auto inputs = fhelium::ckks::check_two_component_multiply_inputs(
      lhs_components, rhs_components, rns_params, kOperation);
  const auto output =
      fhelium::ckks::allocate_three_component_output(lhs_components);
  if (lhs_components.scalar_type() == torch::kInt32) {
    launch<int32_t>(inputs, output, rns_params);
  } else {
    launch<int64_t>(inputs, output, rns_params);
  }
  return output.packed;
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CUDA, m) {
  m.impl("multiply_two_component_ntt_montgomery", &multiply_two_component_cuda);
}
