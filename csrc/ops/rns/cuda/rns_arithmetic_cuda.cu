#include "rns_arithmetic_cuda.h"

#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/cuda/repetition.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

enum class BinaryRnsOperation : int { kMontgomeryMul, kAddLazy, kSubLazy };
enum class UnaryRnsOperation : int {
  kToMontgomery,
  kFromMontgomery,
  kCanonicalize,
  kCenter,
  kShiftPositive,
};

template <typename scalar_t, BinaryRnsOperation operation>
__global__ void rns_binary_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> rhs,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;

  const int rhs_batch = rhs.size(0) == lhs.size(0) ? batch : 0;
  const scalar_t a = lhs[batch][row][coefficient];
  const scalar_t b = rhs[rhs_batch][row][coefficient];

  if constexpr (operation == BinaryRnsOperation::kMontgomeryMul) {
    out[batch][row][coefficient] =
        montgomery_mul(a,
                       b,
                       params[RNS_PARAM_MODULUS_LO][row],
                       params[RNS_PARAM_MODULUS_HI][row],
                       params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                       params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  } else if constexpr (operation == BinaryRnsOperation::kAddLazy) {
    out[batch][row][coefficient] =
        add_lazy_residues(a, b, params[RNS_PARAM_TWICE_MODULUS][row]);
  } else {
    out[batch][row][coefficient] =
        sub_lazy_residues(a, b, params[RNS_PARAM_TWICE_MODULUS][row]);
  }
}

template <typename scalar_t, RepetitionLayout layout>
__global__ void rns_montgomery_mul_compressed_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> compressed_rhs,
    const CudaTensorAccessor32<scalar_t, 2> params,
    int unique_mask,
    int repeat_shift) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;

  const int rhs_batch = compressed_rhs.size(0) == lhs.size(0) ? batch : 0;
  const int rhs_index =
      repeated_rhs_index<layout>(coefficient, unique_mask, repeat_shift);
  out[batch][row][coefficient] =
      montgomery_mul(lhs[batch][row][coefficient],
                     compressed_rhs[rhs_batch][row][rhs_index],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t, BinaryRnsOperation operation>
void launch_binary(torch::Tensor out,
                   const torch::Tensor lhs,
                   const torch::Tensor rhs,
                   const torch::Tensor params) {
  const int device = lhs.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(lhs.size(1),
            (lhs.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            lhs.size(0));
  rns_binary_kernel<scalar_t, operation><<<grid, kCudaBlockSize, 0, stream>>>(
      FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(lhs, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(rhs, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

template <BinaryRnsOperation operation>
torch::Tensor binary_rns(const torch::Tensor& lhs,
                         const torch::Tensor& rhs,
                         const torch::Tensor& params,
                         const char* operation_name) {
  auto out = torch::empty_like(lhs);
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(rhs, "rhs");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_rns_binary_3d(lhs_rows, rhs_rows, operation_name, true);
  check_rns_parameter_rows(lhs_rows, params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), "rns_binary", [&] {
    launch_binary<scalar_t, operation>(out_rows, lhs_rows, rhs_rows, params);
  });
  return out;
}

template <typename scalar_t, UnaryRnsOperation operation>
__global__ void rns_unary_inplace_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;

  scalar_t value = residues[batch][row][coefficient];
  if constexpr (operation == UnaryRnsOperation::kToMontgomery) {
    value = montgomery_mul(value,
                           params[RNS_PARAM_R2][row],
                           params[RNS_PARAM_MODULUS_LO][row],
                           params[RNS_PARAM_MODULUS_HI][row],
                           params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                           params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  } else if constexpr (operation == UnaryRnsOperation::kFromMontgomery) {
    value = montgomery_reduce(value,
                              params[RNS_PARAM_MODULUS_LO][row],
                              params[RNS_PARAM_MODULUS_HI][row],
                              params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                              params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  } else if constexpr (operation == UnaryRnsOperation::kCanonicalize) {
    value =
        canonicalize_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  } else if constexpr (operation == UnaryRnsOperation::kCenter) {
    value = center_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  } else {
    value = shift_residue_positive(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  }
  residues[batch][row][coefficient] = value;
}

template <typename scalar_t, UnaryRnsOperation operation>
void launch_unary(torch::Tensor residues, const torch::Tensor params) {
  const int device = residues.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(residues.size(1),
            (residues.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            residues.size(0));
  rns_unary_inplace_kernel<scalar_t, operation>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(residues, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

template <UnaryRnsOperation operation>
void unary_rns_inplace(torch::Tensor residues,
                       const torch::Tensor& params,
                       const char* operand_name,
                       const char* operation_name) {
  auto rows = view_rns_batch_3d(residues, operand_name);
  check_rns_parameter_rows(rows, params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), "rns_unary", [&] {
    launch_unary<scalar_t, operation>(rows, params);
  });
}

template <typename scalar_t>
__global__ void rns_montgomery_mul_row_scalars_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 1> scalars,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  residues[batch][row][coefficient] =
      montgomery_mul(residues[batch][row][coefficient],
                     scalars[row],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t>
__global__ void rns_add_lazy_twice_modulus_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> rhs,
    const CudaTensorAccessor32<scalar_t, 1> twice_modulus) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;
  const int rhs_batch = rhs.size(0) == lhs.size(0) ? batch : 0;
  out[batch][row][coefficient] =
      add_lazy_residues(lhs[batch][row][coefficient],
                        rhs[rhs_batch][row][coefficient],
                        twice_modulus[row]);
}

template <typename scalar_t>
__global__ void rns_lift_centered_coefficients_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 2> coefficients,
    const CudaTensorAccessor32<scalar_t, 1> twice_modulus) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= coefficients.size(1)) return;
  out[batch][row][coefficient] = shift_residue_positive(
      coefficients[batch][coefficient], twice_modulus[row]);
}

}  // namespace

torch::Tensor rns_montgomery_mul_cuda(const torch::Tensor lhs,
                                      const torch::Tensor rhs,
                                      const torch::Tensor rns_params) {
  return binary_rns<BinaryRnsOperation::kMontgomeryMul>(
      lhs, rhs, rns_params, "rns_montgomery_mul");
}

namespace {

template <RepetitionLayout layout>
torch::Tensor rns_montgomery_mul_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params,
    const char* operation_name) {
  auto out = torch::empty_like(lhs);
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(compressed_rhs, "compressed_rhs");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_compressed_rns_binary_3d(lhs_rows, rhs_rows, operation_name);
  check_rns_parameter_rows(lhs_rows, rns_params, operation_name);
  const int unique_mask = rhs_rows.size(2) - 1;
  const int repeat_shift =
      compressed_repeat_shift(lhs_rows.size(2), rhs_rows.size(2));
  AT_DISPATCH_INTEGRAL_TYPES(
      lhs.scalar_type(), "rns_montgomery_mul_compressed", [&] {
        const int device = lhs_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(lhs_rows.size(1),
                  (lhs_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  lhs_rows.size(0));
        rns_montgomery_mul_compressed_kernel<scalar_t, layout>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(lhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2),
                unique_mask,
                repeat_shift);
      });
  return out;
}

}  // namespace

torch::Tensor rns_montgomery_mul_cyclic_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_compressed_cuda<RepetitionLayout::kCyclic>(
      lhs, compressed_rhs, rns_params, "rns_montgomery_mul_cyclic_compressed");
}

torch::Tensor rns_montgomery_mul_contiguous_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_compressed_cuda<RepetitionLayout::kContiguous>(
      lhs,
      compressed_rhs,
      rns_params,
      "rns_montgomery_mul_contiguous_compressed");
}

void rns_montgomery_mul_row_scalars_inplace_cuda(
    torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  auto rows = view_rns_batch_3d(residues, "residues");
  check_rns_parameter_rows(rows, rns_params, "rns_montgomery_mul_row_scalars");
  check_rns_row_vector(row_scalars,
                       rows.size(1),
                       "rns_montgomery_mul_row_scalars",
                       "row_scalars");
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "rns_montgomery_mul_row_scalars", [&] {
        const int device = rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(rows.size(1),
                  (rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  rows.size(0));
        rns_montgomery_mul_row_scalars_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(row_scalars, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2));
      });
}

void rns_to_montgomery_inplace_cuda(torch::Tensor standard_residues,
                                    const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kToMontgomery>(
      standard_residues, rns_params, "standard_residues", "rns_to_montgomery");
}

void rns_from_montgomery_inplace_cuda(torch::Tensor montgomery_residues,
                                      const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kFromMontgomery>(montgomery_residues,
                                                        rns_params,
                                                        "montgomery_residues",
                                                        "rns_from_montgomery");
}

void rns_canonicalize_residues_inplace_cuda(torch::Tensor lazy_residues,
                                            const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kCanonicalize>(
      lazy_residues, rns_params, "lazy_residues", "rns_canonicalize");
}

torch::Tensor rns_add_lazy_cuda(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params) {
  return binary_rns<BinaryRnsOperation::kAddLazy>(
      lhs, rhs, rns_params, "rns_add_lazy");
}

torch::Tensor rns_add_lazy_with_twice_modulus_cuda(
    const torch::Tensor lhs,
    const torch::Tensor rhs,
    const torch::Tensor twice_modulus) {
  auto out = torch::empty_like(lhs);
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(rhs, "rhs");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_rns_binary_3d(
      lhs_rows, rhs_rows, "rns_add_lazy_with_twice_modulus", true);
  check_rns_row_vector(twice_modulus,
                       lhs_rows.size(1),
                       "rns_add_lazy_with_twice_modulus",
                       "twice_modulus");
  AT_DISPATCH_INTEGRAL_TYPES(
      lhs.scalar_type(), "rns_add_lazy_with_twice_modulus", [&] {
        const int device = lhs_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(lhs_rows.size(1),
                  (lhs_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  lhs_rows.size(0));
        rns_add_lazy_twice_modulus_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(lhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(twice_modulus, scalar_t, 1));
      });
  return out;
}

torch::Tensor rns_sub_lazy_cuda(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params) {
  return binary_rns<BinaryRnsOperation::kSubLazy>(
      lhs, rhs, rns_params, "rns_sub_lazy");
}

void rns_center_residues_inplace_cuda(torch::Tensor canonical_residues,
                                      const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kCenter>(canonical_residues,
                                                rns_params,
                                                "canonical_residues",
                                                "rns_center_residues");
}

void rns_shift_residues_positive_inplace_cuda(torch::Tensor centered_residues,
                                              const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kShiftPositive>(
      centered_residues,
      rns_params,
      "centered_residues",
      "rns_shift_residues_positive");
}

torch::Tensor rns_lift_centered_coefficients_cuda(
    const torch::Tensor centered_coefficients,
    const torch::Tensor twice_modulus) {
  const auto coefficient_rows =
      view_coefficient_batch_2d(centered_coefficients, "centered_coefficients");
  check_rns_row_vector(twice_modulus,
                       twice_modulus.size(0),
                       "rns_lift_centered_coefficients",
                       "twice_modulus");
  std::vector<int64_t> output_shape(centered_coefficients.sizes().begin(),
                                    centered_coefficients.sizes().end() - 1);
  output_shape.push_back(twice_modulus.size(0));
  output_shape.push_back(centered_coefficients.size(-1));
  auto out = centered_coefficients.new_empty(output_shape);
  auto out_rows = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(
      centered_coefficients.scalar_type(),
      "rns_lift_centered_coefficients",
      [&] {
        const int device = coefficient_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(out_rows.size(1),
                  (out_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  out_rows.size(0));
        rns_lift_centered_coefficients_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(coefficient_rows, scalar_t, 2),
                FHELIUM_CUDA_ACCESSOR32(twice_modulus, scalar_t, 1));
      });
  return out;
}
