#include "mixed_radix_cuda.h"

#include <utility>
#include <vector>

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

// Generic mixed-radix requirements (coefficient domain).
//
// source_residues is integral CUDA [*batch, source_limb, coefficient] in
// standard lazy range [0, 2q_i), with source limb j mapped to one exact ordered
// source prime id. Native code collapses only *batch. For source bases b_r, let
// M_0=1 and $M_r=\prod_{t<r}b_t$. `mixed_radix_decompose` returns non-aliasing
// standard lazy digits of the same shape such that
// $x=\sum_r d_rM_r\pmod{\prod_r b_r}$ and $0\le d_r<b_r$ semantically.
// normalizers [source_limb-1] contain $M_r^{-1}R$ for the next source prime;
// propagation [source_limb-1, source_limb] contains $M_rR$ for later primes;
// the four [source_limb] split-word vectors describe those source primes.
//
// `mixed_radix_basis_extend_to_montgomery` consumes standard digits
// [*batch, digit, coefficient]. extension_coefficients has shape
// [digit-1, destination_limb] and stores $M_rR^2$ for r>=1 in exact
// destination-prime order; rns_params is [parameter, destination_limb] in that
// same order. It returns newly allocated
// [*batch, destination_limb, coefficient] Montgomery lazy residues in
// [0, 2p_j), evaluating $\sum_r d_rM_r\bmod p_j$. Inputs are read-only and no
// output aliases them. All table/residue tensors share dtype and CUDA device.
template <typename scalar_t>
__global__ void mixed_radix_decompose_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> source,
    const CudaTensorAccessor32<scalar_t, 1> normalizer,
    const CudaTensorAccessor32<scalar_t, 2> propagation,
    const CudaTensorAccessor32<scalar_t, 1> modulus_lo,
    const CudaTensorAccessor32<scalar_t, 1> modulus_hi,
    const CudaTensorAccessor32<scalar_t, 1> neg_inv_modulus_lo,
    const CudaTensorAccessor32<scalar_t, 1> neg_inv_modulus_hi) {
  constexpr int MAX_DIGIT_ROWS = 8;
  const int coefficient = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  const int row_count = source.size(1);
  if (coefficient >= source.size(2) || row_count > MAX_DIGIT_ROWS) return;

  scalar_t digits[MAX_DIGIT_ROWS];
  const scalar_t first_residue = canonicalize_lazy_montgomery_operand(
      source[batch][0][coefficient], modulus_lo[0], modulus_hi[0]);
  for (int row = 0; row < row_count; ++row) digits[row] = first_residue;

  for (int step = 0; step < row_count - 1; ++step) {
    const int row = step + 1;
    const scalar_t difference = source[batch][row][coefficient] - digits[row];
    const scalar_t twice_modulus =
        (modulus_lo[row] + (modulus_hi[row] << (sizeof(scalar_t) * 4 - 1)))
        << 1;
    const scalar_t digit =
        canonicalize_lazy_residue(montgomery_mul_split(difference,
                                                       normalizer[step],
                                                       modulus_lo[row],
                                                       modulus_hi[row],
                                                       neg_inv_modulus_lo[row],
                                                       neg_inv_modulus_hi[row]),
                                  twice_modulus);
    digits[row] = digit;
    for (int target = row + 1; target < row_count; ++target) {
      digits[target] += montgomery_mul(digit,
                                       propagation[step][target],
                                       modulus_lo[target],
                                       modulus_hi[target],
                                       neg_inv_modulus_lo[target],
                                       neg_inv_modulus_hi[target]);
    }
  }
  for (int row = 0; row < row_count; ++row) {
    out[batch][row][coefficient] = digits[row];
  }
}

torch::Tensor mixed_radix_decompose_cuda(
    const torch::Tensor source_residues,
    const torch::Tensor mixed_radix_normalizers,
    const torch::Tensor mixed_radix_propagation_coefficients,
    const torch::Tensor modulus_lo,
    const torch::Tensor modulus_hi,
    const torch::Tensor neg_inv_modulus_lo,
    const torch::Tensor neg_inv_modulus_hi) {
  auto out = torch::empty_like(source_residues);
  const auto source = view_rns_batch_3d(source_residues, "source_residues");
  auto output = view_rns_batch_3d(out, "out");
  TORCH_CHECK(source.size(1) <= 8,
              "mixed_radix_decompose supports at most 8 digit rows");
  check_rns_row_vector(mixed_radix_normalizers,
                       source.size(1) - 1,
                       "mixed_radix_decompose",
                       "mixed_radix_normalizers");
  for (const auto& item : std::vector<std::pair<torch::Tensor, const char*>>{
           {modulus_lo, "modulus_lo"},
           {modulus_hi, "modulus_hi"},
           {neg_inv_modulus_lo, "neg_inv_modulus_lo"},
           {neg_inv_modulus_hi, "neg_inv_modulus_hi"}}) {
    check_rns_row_vector(
        item.first, source.size(1), "mixed_radix_decompose", item.second);
  }
  TORCH_CHECK(
      mixed_radix_propagation_coefficients.dim() == 2 &&
          mixed_radix_propagation_coefficients.size(0) == source.size(1) - 1 &&
          mixed_radix_propagation_coefficients.size(1) == source.size(1),
      "mixed_radix_decompose propagation table shape mismatch");
  for (const auto& table : {mixed_radix_normalizers,
                            mixed_radix_propagation_coefficients,
                            modulus_lo,
                            modulus_hi,
                            neg_inv_modulus_lo,
                            neg_inv_modulus_hi}) {
    TORCH_CHECK(table.scalar_type() == source_residues.scalar_type(),
                "mixed_radix_decompose table dtype mismatch");
    TORCH_CHECK(table.device() == source_residues.device(),
                "mixed_radix_decompose tables must share the source device");
  }
  const int device = source_residues.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid((source.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            1,
            source.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      source_residues.scalar_type(), "mixed_radix_decompose", [&] {
        mixed_radix_decompose_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(source, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(mixed_radix_normalizers, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(
                    mixed_radix_propagation_coefficients, scalar_t, 2),
                FHELIUM_CUDA_ACCESSOR32(modulus_lo, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(modulus_hi, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(neg_inv_modulus_lo, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(neg_inv_modulus_hi, scalar_t, 1));
      });
  return out;
}

// Evaluate mixed-radix digits on every destination modulus row.
template <typename scalar_t>
__global__ void mixed_radix_basis_extend_to_montgomery_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> digits,
    const CudaTensorAccessor32<scalar_t, 2> extension_coeffs,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int destination_row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= digits.size(2)) return;

  const scalar_t twice_modulus =
      params[RNS_PARAM_TWICE_MODULUS][destination_row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][destination_row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][destination_row];
  const scalar_t neg_inv_modulus_lo =
      params[RNS_PARAM_NEG_INV_MODULUS_LO][destination_row];
  const scalar_t neg_inv_modulus_hi =
      params[RNS_PARAM_NEG_INV_MODULUS_HI][destination_row];
  scalar_t value = montgomery_mul(digits[batch][0][coefficient],
                                  params[RNS_PARAM_R2][destination_row],
                                  modulus_lo,
                                  modulus_hi,
                                  neg_inv_modulus_lo,
                                  neg_inv_modulus_hi);
  for (int row = 1; row < digits.size(1); ++row) {
    const scalar_t contribution =
        montgomery_mul(digits[batch][row][coefficient],
                       extension_coeffs[row - 1][destination_row],
                       modulus_lo,
                       modulus_hi,
                       neg_inv_modulus_lo,
                       neg_inv_modulus_hi);
    value = add_lazy_residues(value, contribution, twice_modulus);
  }
  out[batch][destination_row][coefficient] = value;
}

torch::Tensor mixed_radix_basis_extend_to_montgomery_cuda(
    const torch::Tensor mixed_radix_components,
    const torch::Tensor basis_extension_coefficients,
    const torch::Tensor rns_params,
    const int64_t destination_row_count) {
  TORCH_CHECK(destination_row_count > 0,
              "destination_row_count must be positive");
  auto sizes = mixed_radix_components.sizes().vec();
  TORCH_CHECK(sizes.size() >= 2,
              "mixed_radix_components requires [..., digit row, N] layout");
  sizes[sizes.size() - 2] = destination_row_count;
  auto out = torch::empty(sizes, mixed_radix_components.options());
  const auto digits =
      view_rns_batch_3d(mixed_radix_components, "mixed_radix_components");
  auto output = view_rns_batch_3d(out, "out");
  check_rns_parameter_rows(
      output, rns_params, "mixed_radix_basis_extend_to_montgomery");
  TORCH_CHECK(basis_extension_coefficients.dim() == 2 &&
                  basis_extension_coefficients.size(0) == digits.size(1) - 1 &&
                  (digits.size(1) == 1 ||
                   basis_extension_coefficients.size(1) == output.size(1)),
              "mixed_radix_basis_extend_to_montgomery basis extension table "
              "shape mismatch");
  TORCH_CHECK(
      basis_extension_coefficients.scalar_type() ==
              mixed_radix_components.scalar_type() &&
          rns_params.scalar_type() == mixed_radix_components.scalar_type(),
      "mixed_radix_basis_extend_to_montgomery table dtype mismatch");
  TORCH_CHECK(basis_extension_coefficients.device() ==
                      mixed_radix_components.device() &&
                  rns_params.device() == mixed_radix_components.device(),
              "mixed_radix_basis_extend_to_montgomery tables must share the "
              "input device");
  const int device = mixed_radix_components.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(output.size(1),
            (digits.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            digits.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      mixed_radix_components.scalar_type(),
      "mixed_radix_basis_extend_to_montgomery",
      [&] {
        mixed_radix_basis_extend_to_montgomery_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(digits, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(
                    basis_extension_coefficients, scalar_t, 2),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2));
      });
  return out;
}
