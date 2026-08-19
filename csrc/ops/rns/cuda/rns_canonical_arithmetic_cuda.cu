#include "rns_canonical_arithmetic_cuda.h"

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

enum class CanonicalBinaryOperation : int { kAdd, kSubtract };

template <typename scalar_t, CanonicalBinaryOperation operation>
__global__ void rns_canonical_binary_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> rhs,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;

  const int rhs_batch = rhs.size(0) == lhs.size(0) ? batch : 0;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  scalar_t value;
  if constexpr (operation == CanonicalBinaryOperation::kAdd) {
    value = add_lazy_residues(lhs[batch][row][coefficient],
                              rhs[rhs_batch][row][coefficient],
                              twice_modulus);
  } else {
    value = sub_lazy_residues(lhs[batch][row][coefficient],
                              rhs[rhs_batch][row][coefficient],
                              twice_modulus);
  }
  out[batch][row][coefficient] =
      canonicalize_lazy_residue(value, twice_modulus);
}

template <typename scalar_t, CanonicalBinaryOperation operation>
void launch_canonical_binary(torch::Tensor out,
                             const torch::Tensor lhs,
                             const torch::Tensor rhs,
                             const torch::Tensor params) {
  const int device = lhs.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(lhs.size(1),
            (lhs.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            lhs.size(0));
  rns_canonical_binary_kernel<scalar_t, operation>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(lhs, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(rhs, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

template <CanonicalBinaryOperation operation>
void canonical_binary_into(torch::Tensor out,
                           const torch::Tensor lhs,
                           const torch::Tensor rhs,
                           const torch::Tensor params,
                           const char* operation_name) {
  auto out_rows = view_rns_batch_3d(out, "out");
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(rhs, "rhs");
  check_rns_binary_3d(lhs_rows, rhs_rows, operation_name, true);
  TORCH_CHECK(out_rows.sizes() == lhs_rows.sizes(),
              operation_name,
              " output shape must match lhs");
  check_rns_parameter_rows(lhs_rows, params, operation_name);
  check_mutable_rns_output(out_rows, rhs_rows, params);
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), "rns_canonical_binary", [&] {
    launch_canonical_binary<scalar_t, operation>(
        out_rows, lhs_rows, rhs_rows, params);
  });
}

template <typename scalar_t>
__global__ void rns_montgomery_mul_row_scalars_canonical_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 1> row_scalars,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;

  const scalar_t value =
      montgomery_mul(residues[batch][row][coefficient],
                     row_scalars[row],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  out[batch][row][coefficient] =
      canonicalize_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
}

}  // namespace

torch::Tensor rns_add_canonical_cuda(const torch::Tensor lhs,
                                     const torch::Tensor rhs,
                                     const torch::Tensor rns_params) {
  auto out = torch::empty_like(lhs);
  canonical_binary_into<CanonicalBinaryOperation::kAdd>(
      out, lhs, rhs, rns_params, "rns_add_canonical");
  return out;
}

void rns_add_canonical_inplace_cuda(torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params) {
  canonical_binary_into<CanonicalBinaryOperation::kAdd>(
      lhs, lhs, rhs, rns_params, "rns_add_canonical");
}

torch::Tensor rns_sub_canonical_cuda(const torch::Tensor lhs,
                                     const torch::Tensor rhs,
                                     const torch::Tensor rns_params) {
  auto out = torch::empty_like(lhs);
  canonical_binary_into<CanonicalBinaryOperation::kSubtract>(
      out, lhs, rhs, rns_params, "rns_sub_canonical");
  return out;
}

void rns_sub_canonical_inplace_cuda(torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params) {
  canonical_binary_into<CanonicalBinaryOperation::kSubtract>(
      lhs, lhs, rhs, rns_params, "rns_sub_canonical");
}

torch::Tensor rns_montgomery_mul_row_scalars_canonical_cuda(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  auto out = torch::empty_like(residues);
  const auto residue_rows = view_rns_batch_3d(residues, "residues");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_rns_parameter_rows(
      residue_rows, rns_params, "rns_montgomery_mul_row_scalars_canonical");
  check_rns_row_vector(row_scalars,
                       residue_rows.size(1),
                       "rns_montgomery_mul_row_scalars_canonical",
                       "row_scalars");
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "rns_montgomery_mul_row_scalars_canonical", [&] {
        const int device = residue_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(residue_rows.size(1),
                  (residue_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  residue_rows.size(0));
        rns_montgomery_mul_row_scalars_canonical_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(residue_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(row_scalars, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2));
      });
  return out;
}
