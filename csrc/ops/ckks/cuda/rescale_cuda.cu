#include "ckks_cuda.h"

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

// Rescale kernel input and output requirements. remaining is integral CUDA
// [*batch, remaining_limb, coefficient] and dropped is
// [*batch, coefficient], both standard residues from the same input integer.
// inverse [remaining_limb] stores the dropped Q prime inverse in Montgomery
// form for each destination prime; params [parameter, remaining_limb]
// follows the same prime_ids order. The kernel computes
// $\operatorname{Round}(c/q_{\mathrm{drop}})\bmod q_i$: nearest increments
// when the canonical dropped residue exceeds floor(q_drop/2), while truncate
// omits that increment. Output is coefficient/standard canonical [0, q_i)
// with remaining shape. Functional output does not alias input; underscore
// variants preserve and mutate remaining storage only. Tables are read-only.

template <typename scalar_t, bool nearest>
__global__ void ckks_rescale_drop_leading_prime_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> remaining,
    const CudaTensorAccessor32<scalar_t, 1> inverse,
    const CudaTensorAccessor32<scalar_t, 2> dropped,
    const CudaTensorAccessor32<scalar_t, 2> params,
    const int64_t half_drop_prime) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= remaining.size(2)) return;

  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const scalar_t dropped_value = dropped[batch][coefficient];

  scalar_t quotient = remaining[batch][row][coefficient] - dropped_value;
  quotient = montgomery_mul(quotient,
                            inverse[row],
                            modulus_lo,
                            modulus_hi,
                            neg_inv_modulus_lo,
                            neg_inv_modulus_hi);
  if constexpr (nearest) {
    quotient += dropped_value > half_drop_prime ? 1 : 0;
  }
  out[batch][row][coefficient] =
      canonicalize_lazy_residue(quotient, twice_modulus);
}

void validate_rescale_operands(const torch::Tensor& remaining,
                               const torch::Tensor& inverse,
                               const torch::Tensor& dropped,
                               const torch::Tensor& params,
                               const char* operation) {
  check_rns_parameter_rows(remaining, params, operation);
  check_rns_row_vector(
      inverse, remaining.size(1), operation, "drop_prime_inverse_mont");
  TORCH_CHECK(dropped.dim() == 2 && dropped.size(0) == remaining.size(0) &&
                  dropped.size(1) == remaining.size(2),
              operation,
              " dropped residue batch and coefficient shape must match "
              "remaining residues");
}

template <typename scalar_t, bool nearest>
void launch_rescale(torch::Tensor out,
                    const torch::Tensor remaining,
                    const torch::Tensor inverse,
                    const torch::Tensor dropped,
                    const torch::Tensor params,
                    const int64_t half_drop_prime) {
  const int device = remaining.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(remaining.size(1),
            (remaining.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            remaining.size(0));
  ckks_rescale_drop_leading_prime_kernel<scalar_t, nearest>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(remaining, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(inverse, scalar_t, 1),
          FHELIUM_CUDA_ACCESSOR32(dropped, scalar_t, 2),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2),
          half_drop_prime);
}

template <bool nearest>
torch::Tensor rescale_out(const torch::Tensor& remaining_residues,
                          const torch::Tensor& drop_prime_inverse_mont,
                          const torch::Tensor& dropped_residue,
                          const torch::Tensor& rns_params,
                          const int64_t half_drop_prime,
                          const char* operation) {
  auto out = torch::empty_like(remaining_residues);
  const auto remaining =
      view_rns_batch_3d(remaining_residues, "remaining_residues");
  const auto dropped =
      view_coefficient_batch_2d(dropped_residue, "dropped_residue");
  auto out_rows = view_rns_batch_3d(out, "out");
  validate_rescale_operands(
      remaining, drop_prime_inverse_mont, dropped, rns_params, operation);
  AT_DISPATCH_INTEGRAL_TYPES(
      remaining_residues.scalar_type(), "ckks_rescale", [&] {
        launch_rescale<scalar_t, nearest>(out_rows,
                                          remaining,
                                          drop_prime_inverse_mont,
                                          dropped,
                                          rns_params,
                                          half_drop_prime);
      });
  return out;
}

template <bool nearest>
void rescale_inplace(torch::Tensor remaining_residues,
                     const torch::Tensor& drop_prime_inverse_mont,
                     const torch::Tensor& dropped_residue,
                     const torch::Tensor& rns_params,
                     const int64_t half_drop_prime,
                     const char* operation) {
  auto remaining = view_rns_batch_3d(remaining_residues, "remaining_residues");
  const auto dropped =
      view_coefficient_batch_2d(dropped_residue, "dropped_residue");
  validate_rescale_operands(
      remaining, drop_prime_inverse_mont, dropped, rns_params, operation);
  AT_DISPATCH_INTEGRAL_TYPES(
      remaining_residues.scalar_type(), "ckks_rescale_inplace", [&] {
        launch_rescale<scalar_t, nearest>(remaining,
                                          remaining,
                                          drop_prime_inverse_mont,
                                          dropped,
                                          rns_params,
                                          half_drop_prime);
      });
}

}  // namespace

torch::Tensor ckks_rescale_drop_leading_prime_nearest_cuda(
    const torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params,
    const int64_t half_drop_prime) {
  return rescale_out<true>(remaining_residues,
                           drop_prime_inverse_mont,
                           dropped_residue,
                           rns_params,
                           half_drop_prime,
                           "ckks_rescale_drop_leading_prime_nearest");
}

void ckks_rescale_drop_leading_prime_nearest_inplace_cuda(
    torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params,
    const int64_t half_drop_prime) {
  rescale_inplace<true>(remaining_residues,
                        drop_prime_inverse_mont,
                        dropped_residue,
                        rns_params,
                        half_drop_prime,
                        "ckks_rescale_drop_leading_prime_nearest");
}

torch::Tensor ckks_rescale_drop_leading_prime_truncate_cuda(
    const torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params) {
  return rescale_out<false>(remaining_residues,
                            drop_prime_inverse_mont,
                            dropped_residue,
                            rns_params,
                            0,
                            "ckks_rescale_drop_leading_prime_truncate");
}

void ckks_rescale_drop_leading_prime_truncate_inplace_cuda(
    torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params) {
  rescale_inplace<false>(remaining_residues,
                         drop_prime_inverse_mont,
                         dropped_residue,
                         rns_params,
                         0,
                         "ckks_rescale_drop_leading_prime_truncate");
}
