#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <algorithm>

#include "../../common/cpu/montgomery.h"
#include "../../common/rns_batch.h"
#include "tensor_validation.h"

namespace {

using fhelium::ckks::cpu::check_cpu_peer;

torch::Tensor coefficient_galois_cpu(const torch::Tensor residues,
                                     const torch::Tensor source_indices,
                                     const torch::Tensor source_sign,
                                     const torch::Tensor twice_modulus) {
  constexpr const char* operation = "apply_coefficient_galois_automorphism";
  for (const auto& peer : {source_indices, source_sign, twice_modulus}) {
    TORCH_CHECK(peer.device() == residues.device(),
                operation,
                " requires tables on the operand CPU device");
  }
  TORCH_CHECK(residues.device().is_cpu(), operation, " requires CPU tensors");
  const auto input = view_rns_batch_3d(residues, "residues");
  TORCH_CHECK(source_indices.scalar_type() == torch::kInt32 &&
                  source_sign.scalar_type() == torch::kInt8 &&
                  source_indices.dim() == 1 && source_sign.dim() == 1 &&
                  source_indices.size(0) == input.size(2) &&
                  source_sign.size(0) == input.size(2),
              operation,
              " gather table shape or dtype mismatch");
  check_rns_row_vector(
      twice_modulus, input.size(1), operation, "twice_modulus");
  check_cpu_peer(residues, twice_modulus, operation, "twice_modulus");
  auto out = torch::empty_like(residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), operation, [&] {
    const scalar_t* source = input.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const int32_t* indices = source_indices.data_ptr<int32_t>();
    const int8_t* signs = source_sign.data_ptr<int8_t>();
    const scalar_t* twice = twice_modulus.data_ptr<scalar_t>();
    const int64_t index_stride = source_indices.stride(0);
    const int64_t sign_stride = source_sign.stride(0);
    const int64_t twice_stride = twice_modulus.stride(0);
    const int64_t batch_count = input.size(0);
    const int64_t limbs = input.size(1);
    const int64_t coefficients = input.size(2);
    const int64_t source_stride0 = input.stride(0);
    const int64_t source_stride1 = input.stride(1);
    const int64_t source_stride2 = input.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    const int64_t elements = batch_count * limbs * coefficients;
    for (int64_t coefficient = 0; coefficient < coefficients; ++coefficient) {
      const int64_t source_index = indices[coefficient * index_stride];
      TORCH_CHECK(source_index >= 0 && source_index < coefficients,
                  operation,
                  " source index is out of bounds");
    }
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_limb = index / coefficients;
            const int64_t limb = batch_limb % limbs;
            const int64_t batch = batch_limb / limbs;
            const int64_t coefficient_begin = index - batch_limb * coefficients;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficients, end - batch_limb * coefficients);
            const scalar_t twice_modulus_value = twice[limb * twice_stride];
            const scalar_t* source_rows = source + batch * source_stride0;
            scalar_t* result_row =
                result + batch * result_stride0 + limb * result_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              const int64_t source_index = indices[coefficient * index_stride];
              scalar_t value = source_rows[limb * source_stride1 +
                                           source_index * source_stride2];
              if (signs[coefficient * sign_stride] == -1) value = -value;
              value = fhelium::cpu::shift_positive(value, twice_modulus_value);
              result_row[coefficient * result_stride2] =
                  fhelium::cpu::reduce_to_standard(value, twice_modulus_value);
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

torch::Tensor ntt_galois_cpu(const torch::Tensor residues,
                             const torch::Tensor source_indices) {
  constexpr const char* operation = "apply_ntt_galois_automorphism";
  TORCH_CHECK(residues.device().is_cpu() &&
                  source_indices.device() == residues.device() &&
                  source_indices.scalar_type() == torch::kInt32,
              operation,
              " requires CPU residues and int32 CPU indices");
  const auto input = view_rns_batch_3d(residues, "residues_ntt");
  TORCH_CHECK(
      source_indices.dim() == 1 && source_indices.size(0) == input.size(2),
      operation,
      " source index count mismatch");
  auto out = torch::empty_like(residues);
  auto output = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), operation, [&] {
    const scalar_t* source = input.data_ptr<scalar_t>();
    scalar_t* result = output.data_ptr<scalar_t>();
    const int32_t* indices = source_indices.data_ptr<int32_t>();
    const int64_t index_stride = source_indices.stride(0);
    const int64_t batch_count = input.size(0);
    const int64_t limbs = input.size(1);
    const int64_t coefficients = input.size(2);
    const int64_t source_stride0 = input.stride(0);
    const int64_t source_stride1 = input.stride(1);
    const int64_t source_stride2 = input.stride(2);
    const int64_t result_stride0 = output.stride(0);
    const int64_t result_stride1 = output.stride(1);
    const int64_t result_stride2 = output.stride(2);
    const int64_t elements = batch_count * limbs * coefficients;
    for (int64_t coefficient = 0; coefficient < coefficients; ++coefficient) {
      const int64_t source_index = indices[coefficient * index_stride];
      TORCH_CHECK(source_index >= 0 && source_index < coefficients,
                  operation,
                  " source index is out of bounds");
    }
    at::parallel_for(
        0,
        elements,
        fhelium::cpu::adaptive_grain(elements),
        [&](int64_t begin, int64_t end) {
          int64_t index = begin;
          while (index < end) {
            const int64_t batch_limb = index / coefficients;
            const int64_t limb = batch_limb % limbs;
            const int64_t batch = batch_limb / limbs;
            const int64_t coefficient_begin = index - batch_limb * coefficients;
            const int64_t coefficient_end = std::min<int64_t>(
                coefficients, end - batch_limb * coefficients);
            const scalar_t* source_rows = source + batch * source_stride0;
            scalar_t* result_row =
                result + batch * result_stride0 + limb * result_stride1;
            for (int64_t coefficient = coefficient_begin;
                 coefficient < coefficient_end;
                 ++coefficient) {
              result_row[coefficient * result_stride2] =
                  source_rows[limb * source_stride1 +
                              indices[coefficient * index_stride] *
                                  source_stride2];
            }
            index = batch_limb * coefficients + coefficient_end;
          }
        });
  });
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CPU, m) {
  m.impl("apply_coefficient_galois_automorphism", &coefficient_galois_cpu);
  m.impl("apply_ntt_galois_automorphism", &ntt_galois_cpu);
}
