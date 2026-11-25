#include <torch/torch.h>

#include "../../common/hip/kernel_support.hpp"
#include "../../common/montgomery.hpp"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"
#include "../backends.h"

namespace {

// Galois automorphism representation requirements. Input is integral CUDA
// [*batch, limb, coefficient_or_ntt_index]; output is newly allocated with the
// same shape, dtype/device, prime rows, domain, representation, and lazy
// or standard range. source_indices is int32 [N] destination-to-source order.
// The coefficient variant also consumes int8 source_sign [N] and integral
// twice_modulus [limb] to implement $\sigma_g:X\mapsto X^g$ modulo $X^N+1$;
// it returns standard residues. The NTT variant is a pure gather of NTT
// evaluations. All inputs are read-only and no output aliases an input.
template <typename scalar_t>
__global__ void coefficient_galois_automorphism_kernel(
    HipTensorAccessor32<scalar_t, 3> out,
    const HipTensorAccessor32<scalar_t, 3> residues,
    const HipTensorAccessor32<int32_t, 1> source_indices,
    const HipTensorAccessor32<int8_t, 1> source_sign,
    const HipTensorAccessor32<scalar_t, 1> twice_modulus) {
  const int row = blockIdx.x;
  const int destination = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (destination >= residues.size(2)) return;
  scalar_t value = residues[batch][row][source_indices[destination]];
  if (source_sign[destination] == static_cast<int8_t>(-1)) value = -value;
  value = shift_residue_positive(value, twice_modulus[row]);
  out[batch][row][destination] = reduce_lazy_residue(value, twice_modulus[row]);
}

template <typename scalar_t>
__global__ void ntt_galois_automorphism_kernel(
    HipTensorAccessor32<scalar_t, 3> out,
    const HipTensorAccessor32<scalar_t, 3> residues,
    const HipTensorAccessor32<int32_t, 1> source_indices) {
  const int row = blockIdx.x;
  const int destination = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (destination >= residues.size(2)) return;
  out[batch][row][destination] =
      residues[batch][row][source_indices[destination]];
}

template <typename scalar_t>
__global__ void add_ntt_galois_automorphism_kernel(
    HipTensorAccessor32<scalar_t, 3> target,
    const HipTensorAccessor32<scalar_t, 3> residues,
    const HipTensorAccessor32<int32_t, 1> source_indices,
    const HipTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int destination = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (destination >= residues.size(2)) return;
  const int source = source_indices[destination];
  target[batch][row][destination] =
      add_lazy_residues(target[batch][row][destination],
                        residues[batch][row][source],
                        params[RNS_PARAM_TWICE_MODULUS][row]);
}

void validate_automorphism(const torch::Tensor& residues,
                           const torch::Tensor& source_indices,
                           const char* operation) {
  TORCH_CHECK(
      source_indices.dim() == 1 && source_indices.size(0) == residues.size(2),
      operation,
      " source index count must match coefficient count");
}

torch::Tensor apply_coefficient_galois_automorphism_hip(
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
  hipSetDevice(device);
  auto stream = c10::hip::getCurrentHIPStream(device);
  dim3 grid(input.size(1),
            (input.size(2) + kHipBlockSize - 1) / kHipBlockSize,
            input.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "apply_coefficient_galois_automorphism", [&] {
        coefficient_galois_automorphism_kernel<scalar_t>
            <<<grid, kHipBlockSize, 0, stream>>>(
                FHELIUM_HIP_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(input, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(source_indices, int32_t, 1),
                FHELIUM_HIP_ACCESSOR32(source_sign, int8_t, 1),
                FHELIUM_HIP_ACCESSOR32(twice_modulus, scalar_t, 1));
      });
  return out;
}

torch::Tensor apply_ntt_galois_automorphism_hip(
    const torch::Tensor residues_ntt, const torch::Tensor source_indices) {
  auto out = torch::empty_like(residues_ntt);
  const auto input = view_rns_batch_3d(residues_ntt, "residues_ntt");
  auto output = view_rns_batch_3d(out, "out");
  validate_automorphism(input, source_indices, "apply_ntt_galois_automorphism");
  const int device = residues_ntt.device().index();
  hipSetDevice(device);
  auto stream = c10::hip::getCurrentHIPStream(device);
  dim3 grid(input.size(1),
            (input.size(2) + kHipBlockSize - 1) / kHipBlockSize,
            input.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      residues_ntt.scalar_type(), "apply_ntt_galois_automorphism", [&] {
        ntt_galois_automorphism_kernel<scalar_t>
            <<<grid, kHipBlockSize, 0, stream>>>(
                FHELIUM_HIP_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(input, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(source_indices, int32_t, 1));
      });
  return out;
}

void add_ntt_galois_automorphism_hip(torch::Tensor& target,
                                     const torch::Tensor residues_ntt,
                                     const torch::Tensor source_indices,
                                     const torch::Tensor rns_params) {
  const auto input = view_rns_batch_3d(residues_ntt, "residues_ntt");
  auto output = view_rns_batch_3d(target, "target");
  validate_automorphism(input, source_indices, "add_ntt_galois_automorphism_");
  TORCH_CHECK(target.sizes() == residues_ntt.sizes() &&
                  target.scalar_type() == residues_ntt.scalar_type() &&
                  target.device() == residues_ntt.device() &&
                  source_indices.scalar_type() == torch::kInt32 &&
                  source_indices.device() == residues_ntt.device() &&
                  rns_params.device() == residues_ntt.device() &&
                  rns_params.dim() == 2 && rns_params.size(1) == input.size(1),
              "add_ntt_galois_automorphism_ requires matching target and "
              "residues, int32 indices and one parameter column per row");
  const int device = residues_ntt.device().index();
  hipSetDevice(device);
  auto stream = c10::hip::getCurrentHIPStream(device);
  dim3 grid(input.size(1),
            (input.size(2) + kHipBlockSize - 1) / kHipBlockSize,
            input.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      residues_ntt.scalar_type(), "add_ntt_galois_automorphism_", [&] {
        add_ntt_galois_automorphism_kernel<scalar_t>
            <<<grid, kHipBlockSize, 0, stream>>>(
                FHELIUM_HIP_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(input, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(source_indices, int32_t, 1),
                FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2));
      });
}

}  // namespace

namespace fhelium::ckks::gpu {

at::Tensor apply_coefficient_galois_automorphism(
    const at::Tensor& residues,
    const at::Tensor& source_indices,
    const at::Tensor& source_sign,
    const at::Tensor& twice_modulus) {
  return ::apply_coefficient_galois_automorphism_hip(
      residues, source_indices, source_sign, twice_modulus);
}

at::Tensor apply_ntt_galois_automorphism(const at::Tensor& residues_ntt,
                                         const at::Tensor& source_indices) {
  return ::apply_ntt_galois_automorphism_hip(residues_ntt, source_indices);
}


void add_ntt_galois_automorphism_(at::Tensor& target,
                                  const at::Tensor& residues_ntt,
                                  const at::Tensor& source_indices,
                                  const at::Tensor& rns_params) {
  ::add_ntt_galois_automorphism_hip(target, residues_ntt, source_indices, rns_params);
}

}  // namespace fhelium::ckks::gpu
