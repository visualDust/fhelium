#include "rns_standard_arithmetic_cuda.h"

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

enum class StandardBinaryOperation : int { kAdd, kSubtract };

template <typename scalar_t, StandardBinaryOperation operation>
__global__ void rns_standard_binary_kernel(
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
  if constexpr (operation == StandardBinaryOperation::kAdd) {
    value = add_lazy_residues(lhs[batch][row][coefficient],
                              rhs[rhs_batch][row][coefficient],
                              twice_modulus);
  } else {
    value = sub_lazy_residues(lhs[batch][row][coefficient],
                              rhs[rhs_batch][row][coefficient],
                              twice_modulus);
  }
  out[batch][row][coefficient] = reduce_lazy_residue(value, twice_modulus);
}

template <typename scalar_t, StandardBinaryOperation operation>
void launch_standard_binary(torch::Tensor out,
                            const torch::Tensor lhs,
                            const torch::Tensor rhs,
                            const torch::Tensor params) {
  const int device = lhs.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(lhs.size(1),
            (lhs.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            lhs.size(0));
  rns_standard_binary_kernel<scalar_t, operation>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(lhs, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(rhs, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

template <StandardBinaryOperation operation>
void standard_binary_into(torch::Tensor out,
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
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), "rns_standard_binary", [&] {
    launch_standard_binary<scalar_t, operation>(
        out_rows, lhs_rows, rhs_rows, params);
  });
}

template <typename scalar_t>
__global__ void rns_montgomery_mul_row_scalars_standard_kernel(
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
      reduce_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
}

}  // namespace

torch::Tensor rns_add_standard_cuda(const torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params) {
  auto out = torch::empty_like(lhs);
  standard_binary_into<StandardBinaryOperation::kAdd>(
      out, lhs, rhs, rns_params, "rns_add_standard");
  return out;
}

void rns_add_standard_inplace_cuda(torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params) {
  standard_binary_into<StandardBinaryOperation::kAdd>(
      lhs, lhs, rhs, rns_params, "rns_add_standard");
}

torch::Tensor rns_sub_standard_cuda(const torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params) {
  auto out = torch::empty_like(lhs);
  standard_binary_into<StandardBinaryOperation::kSubtract>(
      out, lhs, rhs, rns_params, "rns_sub_standard");
  return out;
}

void rns_sub_standard_inplace_cuda(torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params) {
  standard_binary_into<StandardBinaryOperation::kSubtract>(
      lhs, lhs, rhs, rns_params, "rns_sub_standard");
}

torch::Tensor rns_montgomery_mul_row_scalars_standard_cuda(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  auto out = torch::empty_like(residues);
  const auto residue_rows = view_rns_batch_3d(residues, "residues");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_rns_parameter_rows(
      residue_rows, rns_params, "rns_montgomery_mul_row_scalars_standard");
  check_rns_row_vector(row_scalars,
                       residue_rows.size(1),
                       "rns_montgomery_mul_row_scalars_standard",
                       "row_scalars");
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "rns_montgomery_mul_row_scalars_standard", [&] {
        const int device = residue_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(residue_rows.size(1),
                  (residue_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  residue_rows.size(0));
        rns_montgomery_mul_row_scalars_standard_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(residue_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(row_scalars, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2));
      });
  return out;
}

template <typename scalar_t>
__global__ void rns_sum_standard_batch_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> source,
    const CudaTensorAccessor32<scalar_t, 2> params,
    const int64_t count,
    const int64_t inner) {
  const int row = blockIdx.x;
  const int64_t coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  if (coefficient >= source.size(2)) return;
  const int64_t plane = blockIdx.z;
  const int64_t outer = plane / inner;
  const int64_t inner_index = plane - outer * inner;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const int64_t base = (outer * count) * inner + inner_index;
  scalar_t value = source[base][row][coefficient];
  for (int64_t index = 1; index < count; ++index) {
    value = reduce_lazy_residue(
        add_lazy_residues(value,
                          source[base + index * inner][row][coefficient],
                          twice_modulus),
        twice_modulus);
  }
  out[plane][row][coefficient] = value;
}

torch::Tensor rns_sum_standard_batch_cuda(const torch::Tensor source,
                                          const int64_t dim,
                                          const torch::Tensor rns_params) {
  constexpr const char* operation = "rns_sum_standard_batch";
  TORCH_CHECK(dim >= 0 && dim <= source.dim() - 3,
              operation,
              " dim must select a leading batch axis");
  TORCH_CHECK(source.size(dim) > 0,
              operation,
              " cannot reduce an empty batch axis");
  const auto source_rows = view_rns_batch_3d(source, "source");
  check_rns_parameter_rows(source_rows, rns_params, operation);
  auto sizes = source.sizes().vec();
  sizes.erase(sizes.begin() + dim);
  auto out = torch::empty(sizes, source.options());
  auto out_rows = view_rns_batch_3d(out, "out");
  const int64_t count = source.size(dim);
  int64_t inner = 1;
  for (int64_t axis = dim + 1; axis < source.dim() - 2; ++axis) {
    inner *= source.size(axis);
  }
  const int64_t plane_count = source_rows.size(0) / (count * inner) * inner;
  TORCH_CHECK(plane_count <= 65535,
              operation,
              " supports at most 65535 reduced planes");
  AT_DISPATCH_INTEGRAL_TYPES(source.scalar_type(), operation, [&] {
    const int device = source_rows.device().index();
    cudaSetDevice(device);
    auto stream = at::cuda::getCurrentCUDAStream(device);
    dim3 grid(source_rows.size(1),
              (source_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
              plane_count);
    rns_sum_standard_batch_kernel<scalar_t>
        <<<grid, kCudaBlockSize, 0, stream>>>(
            FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
            FHELIUM_CUDA_ACCESSOR32(source_rows, scalar_t, 3),
            FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2),
            count,
            inner);
  });
  return out;
}
