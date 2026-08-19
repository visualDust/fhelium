#include "ckks_cuda.h"

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/cuda/repetition.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

// Prepared-plaintext addition requirements. Ciphertext components are integral
// CUDA [*batch, limb, coefficient] standard residues. Prepared plaintexts use
// the same exact prime rows in coefficient-domain Montgomery representation;
// dense shape is [*batch, limb, coefficient]. Compressed variants replace the
// final extent with repeated power-of-two support, and strided
// form supplies [*batch, limb] implicit values. rns_params is
// [parameter, limb] in the same prime_ids order. The kernel computes
// $c'_{0,i}=c_{0,i}+p_i\bmod q_i$ and returns standard canonical [0, q_i).
// Functional output does not alias; underscore variants mutate only ciphertext
// storage. Plaintext/tables are read-only. Only a genuinely unbatched plaintext
// may broadcast across public batch; limb/coefficient axes never broadcast.

template <typename scalar_t>
__device__ __forceinline__ scalar_t
add_prepared_plaintext_residue(scalar_t ciphertext_value,
                               scalar_t plaintext_value,
                               const CudaTensorAccessor32<scalar_t, 2>& params,
                               int row) {
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];

  scalar_t value = montgomery_mul(ciphertext_value,
                                  params[RNS_PARAM_R2][row],
                                  modulus_lo,
                                  modulus_hi,
                                  neg_inv_modulus_lo,
                                  neg_inv_modulus_hi);
  value = add_lazy_residues(value, plaintext_value, twice_modulus);
  value = montgomery_reduce(
      value, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
  return canonicalize_lazy_residue(value, twice_modulus);
}

// Add one operation-ready coefficient-domain plaintext to a ciphertext
// component. The plaintext may either match the ciphertext's exact batch or be
// the unique allowed broadcast case: one genuinely unbatched RNS plaintext.
template <typename scalar_t>
__global__ void ckks_add_prepared_plaintext_component_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> ciphertext,
    const CudaTensorAccessor32<scalar_t, 3> plaintext,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= ciphertext.size(2)) return;
  const int plaintext_batch =
      plaintext.size(0) == ciphertext.size(0) ? batch : 0;

  out[batch][row][coefficient] = add_prepared_plaintext_residue(
      ciphertext[batch][row][coefficient],
      plaintext[plaintext_batch][row][coefficient],
      params,
      row);
}

template <typename scalar_t, RepetitionLayout layout>
__global__ void ckks_add_compressed_plaintext_component_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> ciphertext,
    const CudaTensorAccessor32<scalar_t, 3> compressed_plaintext,
    const CudaTensorAccessor32<scalar_t, 2> params,
    int unique_mask,
    int repeat_shift) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= ciphertext.size(2)) return;
  const int plaintext_batch =
      compressed_plaintext.size(0) == ciphertext.size(0) ? batch : 0;
  const int plaintext_index =
      repeated_rhs_index<layout>(coefficient, unique_mask, repeat_shift);

  out[batch][row][coefficient] = add_prepared_plaintext_residue(
      ciphertext[batch][row][coefficient],
      compressed_plaintext[plaintext_batch][row][plaintext_index],
      params,
      row);
}

template <typename scalar_t>
__global__ void ckks_add_strided_plaintext_component_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> ciphertext,
    const CudaTensorAccessor32<scalar_t, 3> strided_plaintext,
    const CudaTensorAccessor32<scalar_t, 2> implicit_plaintext,
    const CudaTensorAccessor32<scalar_t, 2> params,
    int support_mask,
    int support_shift) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= ciphertext.size(2)) return;
  const int plaintext_batch =
      strided_plaintext.size(0) == ciphertext.size(0) ? batch : 0;
  scalar_t plaintext_value = implicit_plaintext[plaintext_batch][row];
  if ((coefficient & support_mask) == 0) {
    plaintext_value =
        strided_plaintext[plaintext_batch][row][coefficient >> support_shift];
  }
  out[batch][row][coefficient] = add_prepared_plaintext_residue(
      ciphertext[batch][row][coefficient], plaintext_value, params, row);
}

template <typename scalar_t>
void launch_ckks_add_prepared_plaintext_component_cuda(
    torch::Tensor out,
    const torch::Tensor ciphertext,
    const torch::Tensor plaintext,
    const torch::Tensor params) {
  const int device = ciphertext.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(ciphertext.size(1),
            (ciphertext.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            ciphertext.size(0));
  ckks_add_prepared_plaintext_component_kernel<scalar_t>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(ciphertext, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(plaintext, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

void validate_add_prepared_plaintext(const torch::Tensor& ciphertext,
                                     const torch::Tensor& plaintext,
                                     const torch::Tensor& params) {
  check_rns_binary_3d(
      ciphertext, plaintext, "ckks_add_prepared_plaintext_component", true);
  check_rns_parameter_rows(
      ciphertext, params, "ckks_add_prepared_plaintext_component");
}

torch::Tensor ckks_add_prepared_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor prepared_plaintext,
    const torch::Tensor rns_params) {
  auto out = torch::empty_like(ciphertext_component);
  const auto ciphertext_rows =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext_rows =
      view_rns_batch_3d(prepared_plaintext, "prepared_plaintext");
  auto out_rows = view_rns_batch_3d(out, "out");
  validate_add_prepared_plaintext(ciphertext_rows, plaintext_rows, rns_params);
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(),
      "ckks_add_prepared_plaintext_component",
      [&] {
        launch_ckks_add_prepared_plaintext_component_cuda<scalar_t>(
            out_rows, ciphertext_rows, plaintext_rows, rns_params);
      });
  return out;
}

void ckks_add_prepared_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor prepared_plaintext,
    const torch::Tensor rns_params) {
  auto ciphertext_rows =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext_rows =
      view_rns_batch_3d(prepared_plaintext, "prepared_plaintext");
  validate_add_prepared_plaintext(ciphertext_rows, plaintext_rows, rns_params);
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(),
      "ckks_add_prepared_plaintext_component_inplace",
      [&] {
        launch_ckks_add_prepared_plaintext_component_cuda<scalar_t>(
            ciphertext_rows, ciphertext_rows, plaintext_rows, rns_params);
      });
}

namespace {

void validate_add_compressed_plaintext(
    const torch::Tensor& ciphertext,
    const torch::Tensor& compressed_plaintext,
    const torch::Tensor& params,
    const char* operation_name) {
  check_compressed_rns_binary_3d(
      ciphertext, compressed_plaintext, operation_name);
  check_rns_parameter_rows(ciphertext, params, operation_name);
}

template <typename scalar_t, RepetitionLayout layout>
void launch_ckks_add_compressed_plaintext_component_cuda(
    torch::Tensor out,
    const torch::Tensor ciphertext,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor params) {
  const int device = ciphertext.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(ciphertext.size(1),
            (ciphertext.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            ciphertext.size(0));
  const int unique_mask = compressed_plaintext.size(2) - 1;
  const int repeat_shift =
      compressed_repeat_shift(ciphertext.size(2), compressed_plaintext.size(2));
  ckks_add_compressed_plaintext_component_kernel<scalar_t, layout>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(ciphertext, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(compressed_plaintext, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2),
          unique_mask,
          repeat_shift);
}

template <RepetitionLayout layout>
torch::Tensor add_compressed_plaintext_component(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params,
    const char* operation_name) {
  auto out = torch::empty_like(ciphertext_component);
  const auto ciphertext_rows =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext_rows =
      view_rns_batch_3d(compressed_plaintext, "compressed_plaintext");
  auto out_rows = view_rns_batch_3d(out, "out");
  validate_add_compressed_plaintext(
      ciphertext_rows, plaintext_rows, rns_params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(),
      "ckks_add_compressed_plaintext_component",
      [&] {
        launch_ckks_add_compressed_plaintext_component_cuda<scalar_t, layout>(
            out_rows, ciphertext_rows, plaintext_rows, rns_params);
      });
  return out;
}

template <RepetitionLayout layout>
void add_compressed_plaintext_component_inplace(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params,
    const char* operation_name) {
  auto ciphertext_rows =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext_rows =
      view_rns_batch_3d(compressed_plaintext, "compressed_plaintext");
  validate_add_compressed_plaintext(
      ciphertext_rows, plaintext_rows, rns_params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(),
      "ckks_add_compressed_plaintext_component_inplace",
      [&] {
        launch_ckks_add_compressed_plaintext_component_cuda<scalar_t, layout>(
            ciphertext_rows, ciphertext_rows, plaintext_rows, rns_params);
      });
}

}  // namespace

torch::Tensor ckks_add_cyclic_compressed_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  return add_compressed_plaintext_component<RepetitionLayout::kCyclic>(
      ciphertext_component,
      compressed_plaintext,
      rns_params,
      "ckks_add_cyclic_compressed_plaintext_component");
}

void ckks_add_cyclic_compressed_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  add_compressed_plaintext_component_inplace<RepetitionLayout::kCyclic>(
      ciphertext_component,
      compressed_plaintext,
      rns_params,
      "ckks_add_cyclic_compressed_plaintext_component_inplace");
}

torch::Tensor ckks_add_contiguous_compressed_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  return add_compressed_plaintext_component<RepetitionLayout::kContiguous>(
      ciphertext_component,
      compressed_plaintext,
      rns_params,
      "ckks_add_contiguous_compressed_plaintext_component");
}

void ckks_add_contiguous_compressed_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  add_compressed_plaintext_component_inplace<RepetitionLayout::kContiguous>(
      ciphertext_component,
      compressed_plaintext,
      rns_params,
      "ckks_add_contiguous_compressed_plaintext_component_inplace");
}

namespace {

void validate_add_strided_plaintext(const torch::Tensor& ciphertext,
                                    const torch::Tensor& strided_plaintext,
                                    const torch::Tensor& implicit_plaintext,
                                    const torch::Tensor& params) {
  check_compressed_rns_binary_3d(
      ciphertext, strided_plaintext, "ckks_add_strided_plaintext_component");
  TORCH_CHECK(implicit_plaintext.dim() == 2,
              "ckks_add_strided_plaintext_component requires canonical "
              "implicit_plaintext [batch, limb] storage");
  TORCH_CHECK(implicit_plaintext.size(0) == strided_plaintext.size(0) &&
                  implicit_plaintext.size(1) == strided_plaintext.size(1),
              "ckks_add_strided_plaintext_component implicit plaintext "
              "shape must match compact batch and limb dimensions: ",
              implicit_plaintext.sizes(),
              " vs ",
              strided_plaintext.sizes());
  TORCH_CHECK(
      implicit_plaintext.scalar_type() == strided_plaintext.scalar_type(),
      "ckks_add_strided_plaintext_component compact and implicit "
      "plaintext dtypes differ");
  TORCH_CHECK(implicit_plaintext.device() == strided_plaintext.device(),
              "ckks_add_strided_plaintext_component compact and implicit "
              "plaintext devices differ");
  check_rns_parameter_rows(
      ciphertext, params, "ckks_add_strided_plaintext_component");
}

template <typename scalar_t>
void launch_ckks_add_strided_plaintext_component_cuda(
    torch::Tensor out,
    const torch::Tensor ciphertext,
    const torch::Tensor strided_plaintext,
    const torch::Tensor implicit_plaintext,
    const torch::Tensor params) {
  const int device = ciphertext.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(ciphertext.size(1),
            (ciphertext.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            ciphertext.size(0));
  const int support_stride = ciphertext.size(2) / strided_plaintext.size(2);
  const int support_mask = support_stride - 1;
  const int support_shift =
      compressed_repeat_shift(ciphertext.size(2), strided_plaintext.size(2));
  ckks_add_strided_plaintext_component_kernel<scalar_t>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(ciphertext, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(strided_plaintext, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(implicit_plaintext, scalar_t, 2),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2),
          support_mask,
          support_shift);
}

}  // namespace

torch::Tensor ckks_add_strided_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor strided_plaintext,
    const torch::Tensor implicit_plaintext,
    const torch::Tensor rns_params) {
  auto out = torch::empty_like(ciphertext_component);
  const auto ciphertext_rows =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext_rows =
      view_rns_batch_3d(strided_plaintext, "strided_plaintext");
  const auto implicit_rows =
      view_coefficient_batch_2d(implicit_plaintext, "implicit_plaintext");
  auto out_rows = view_rns_batch_3d(out, "out");
  validate_add_strided_plaintext(
      ciphertext_rows, plaintext_rows, implicit_rows, rns_params);
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(),
      "ckks_add_strided_plaintext_component",
      [&] {
        launch_ckks_add_strided_plaintext_component_cuda<scalar_t>(
            out_rows,
            ciphertext_rows,
            plaintext_rows,
            implicit_rows,
            rns_params);
      });
  return out;
}

void ckks_add_strided_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor strided_plaintext,
    const torch::Tensor implicit_plaintext,
    const torch::Tensor rns_params) {
  auto ciphertext_rows =
      view_rns_batch_3d(ciphertext_component, "ciphertext_component");
  const auto plaintext_rows =
      view_rns_batch_3d(strided_plaintext, "strided_plaintext");
  const auto implicit_rows =
      view_coefficient_batch_2d(implicit_plaintext, "implicit_plaintext");
  validate_add_strided_plaintext(
      ciphertext_rows, plaintext_rows, implicit_rows, rns_params);
  AT_DISPATCH_INTEGRAL_TYPES(
      ciphertext_component.scalar_type(),
      "ckks_add_strided_plaintext_component_inplace",
      [&] {
        launch_ckks_add_strided_plaintext_component_cuda<scalar_t>(
            ciphertext_rows,
            ciphertext_rows,
            plaintext_rows,
            implicit_rows,
            rns_params);
      });
}
