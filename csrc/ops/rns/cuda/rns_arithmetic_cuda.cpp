#include <torch/torch.h>

#include "../backends.h"
#include "mixed_radix_cuda.h"
#include "rns_arithmetic_cuda.h"

namespace fhelium::rns::gpu {

at::Tensor montgomery_mul(const at::Tensor& lhs,
                          const at::Tensor& rhs,
                          const at::Tensor& rns_params) {
  return rns_montgomery_mul_cuda(lhs, rhs, rns_params);
}

at::Tensor montgomery_weighted_sum(at::TensorList ciphertexts,
                                   at::TensorList plaintexts,
                                   const at::Tensor& rns_params) {
  return rns_montgomery_weighted_sum_cuda(ciphertexts, plaintexts, rns_params);
}

at::Tensor montgomery_weighted_sums(at::TensorList ciphertexts,
                                    at::TensorList plaintexts,
                                    int64_t group_count,
                                    const at::Tensor& rns_params) {
  return rns_montgomery_weighted_sums_cuda(
      ciphertexts, plaintexts, group_count, rns_params);
}

at::Tensor montgomery_mul_cyclic_compressed(const at::Tensor& lhs,
                                            const at::Tensor& compressed_rhs,
                                            const at::Tensor& rns_params) {
  return rns_montgomery_mul_cyclic_compressed_cuda(
      lhs, compressed_rhs, rns_params);
}

at::Tensor montgomery_mul_contiguous_compressed(
    const at::Tensor& lhs,
    const at::Tensor& compressed_rhs,
    const at::Tensor& rns_params) {
  return rns_montgomery_mul_contiguous_compressed_cuda(
      lhs, compressed_rhs, rns_params);
}

void montgomery_mul_row_scalars_(at::Tensor& residues,
                                 const at::Tensor& row_scalars,
                                 const at::Tensor& rns_params) {
  rns_montgomery_mul_row_scalars_inplace_cuda(
      residues, row_scalars, rns_params);
}

void to_montgomery_(at::Tensor& standard_residues,
                    const at::Tensor& rns_params) {
  rns_to_montgomery_inplace_cuda(standard_residues, rns_params);
}

void from_montgomery_(at::Tensor& montgomery_residues,
                      const at::Tensor& rns_params) {
  rns_from_montgomery_inplace_cuda(montgomery_residues, rns_params);
}

at::Tensor add_lazy(const at::Tensor& lhs,
                    const at::Tensor& rhs,
                    const at::Tensor& rns_params) {
  return rns_add_lazy_cuda(lhs, rhs, rns_params);
}

at::Tensor add_lazy_with_twice_modulus(const at::Tensor& lhs,
                                       const at::Tensor& rhs,
                                       const at::Tensor& twice_modulus) {
  return rns_add_lazy_with_twice_modulus_cuda(lhs, rhs, twice_modulus);
}

at::Tensor sub_lazy(const at::Tensor& lhs,
                    const at::Tensor& rhs,
                    const at::Tensor& rns_params) {
  return rns_sub_lazy_cuda(lhs, rhs, rns_params);
}

void reduce_to_standard_(at::Tensor& lazy_residues,
                         const at::Tensor& rns_params) {
  rns_reduce_to_standard_inplace_cuda(lazy_residues, rns_params);
}

void center_residues_(at::Tensor& standard_residues,
                      const at::Tensor& rns_params) {
  rns_center_residues_inplace_cuda(standard_residues, rns_params);
}

void shift_residues_positive_(at::Tensor& centered_residues,
                              const at::Tensor& rns_params) {
  rns_shift_residues_positive_inplace_cuda(centered_residues, rns_params);
}

at::Tensor lift_integer_coefficients(const at::Tensor& coefficients,
                                     const at::Tensor& twice_modulus) {
  return rns_lift_integer_coefficients_cuda(coefficients, twice_modulus);
}

at::Tensor mixed_radix_decompose(
    const at::Tensor& source_residues,
    const at::Tensor& mixed_radix_normalizers,
    const at::Tensor& mixed_radix_propagation_coefficients,
    const at::Tensor& modulus_lo,
    const at::Tensor& modulus_hi,
    const at::Tensor& neg_inv_modulus_lo,
    const at::Tensor& neg_inv_modulus_hi) {
  return mixed_radix_decompose_cuda(source_residues,
                                    mixed_radix_normalizers,
                                    mixed_radix_propagation_coefficients,
                                    modulus_lo,
                                    modulus_hi,
                                    neg_inv_modulus_lo,
                                    neg_inv_modulus_hi);
}

at::Tensor mixed_radix_basis_extend_to_montgomery(
    const at::Tensor& mixed_radix_components,
    const at::Tensor& basis_extension_coefficients,
    const at::Tensor& rns_params,
    int64_t destination_row_count) {
  return mixed_radix_basis_extend_to_montgomery_cuda(
      mixed_radix_components,
      basis_extension_coefficients,
      rns_params,
      destination_row_count);
}

void crt_basis_extend_out_(const at::Tensor& source_residues,
                           const at::Tensor& source_parameters,
                           const at::Tensor& target_parameters,
                           const at::Tensor& matrix,
                           const at::Tensor& target_reciprocals,
                           const std::optional<at::Tensor>& source_normalizers,
                           const std::optional<at::Tensor>& source_reciprocals,
                           const std::optional<at::Tensor>& target_offsets,
                           at::Tensor& output) {
  ::crt_basis_extend_out_cuda(source_residues,
                              source_parameters,
                              target_parameters,
                              matrix,
                              target_reciprocals,
                              source_normalizers,
                              source_reciprocals,
                              target_offsets,
                              output);
}

at::Tensor crt_basis_extend(
    const at::Tensor& source_residues,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const at::Tensor& target_reciprocals,
    const std::optional<at::Tensor>& source_normalizers,
    const std::optional<at::Tensor>& source_reciprocals,
    const std::optional<at::Tensor>& target_offsets) {
  return ::crt_basis_extend_cuda(source_residues,
                                 source_parameters,
                                 target_parameters,
                                 matrix,
                                 target_reciprocals,
                                 source_normalizers,
                                 source_reciprocals,
                                 target_offsets);
}

at::Tensor mixed_radix_modup_tiled(const at::Tensor& source_residues,
                                   at::TensorList decomposition_tables,
                                   const at::Tensor& extension,
                                   const at::Tensor& parameters) {
  return ::mixed_radix_modup_tiled_cuda(
      source_residues, decomposition_tables, extension, parameters);
}

void mixed_radix_modup_tiled_out_(const at::Tensor& source_residues,
                                  at::TensorList decomposition_tables,
                                  const at::Tensor& extension,
                                  const at::Tensor& parameters,
                                  at::Tensor& out) {
  ::mixed_radix_modup_tiled_out_cuda(
      source_residues, decomposition_tables, extension, parameters, out);
}

void mixed_radix_decompose_out_(
    const at::Tensor& source_residues,
    const at::Tensor& mixed_radix_normalizers,
    const at::Tensor& mixed_radix_propagation_coefficients,
    const at::Tensor& modulus_lo,
    const at::Tensor& modulus_hi,
    const at::Tensor& neg_inv_modulus_lo,
    const at::Tensor& neg_inv_modulus_hi,
    at::Tensor& out) {
  ::mixed_radix_decompose_out_cuda(source_residues,
                                   mixed_radix_normalizers,
                                   mixed_radix_propagation_coefficients,
                                   modulus_lo,
                                   modulus_hi,
                                   neg_inv_modulus_lo,
                                   neg_inv_modulus_hi,
                                   out);
}

void mixed_radix_basis_extend_to_montgomery_out_(
    const at::Tensor& mixed_radix_components,
    const at::Tensor& basis_extension_coefficients,
    const at::Tensor& rns_params,
    int64_t destination_row_count,
    at::Tensor& out) {
  ::mixed_radix_basis_extend_to_montgomery_out_cuda(
      mixed_radix_components,
      basis_extension_coefficients,
      rns_params,
      destination_row_count,
      out);
}

}  // namespace fhelium::rns::gpu
