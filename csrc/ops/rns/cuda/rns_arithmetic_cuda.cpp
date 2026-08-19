#include <torch/library.h>
#include <torch/torch.h>

#include "mixed_radix_cuda.h"
#include "rns_arithmetic_cuda.h"

torch::Tensor montgomery_mul(const torch::Tensor lhs,
                             const torch::Tensor rhs,
                             const torch::Tensor rns_params) {
  return rns_montgomery_mul_cuda(lhs, rhs, rns_params);
}

torch::Tensor montgomery_mul_cyclic_compressed(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_cyclic_compressed_cuda(
      lhs, compressed_rhs, rns_params);
}

torch::Tensor montgomery_mul_contiguous_compressed(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_contiguous_compressed_cuda(
      lhs, compressed_rhs, rns_params);
}

void montgomery_mul_row_scalars_(torch::Tensor residues,
                                 const torch::Tensor row_scalars,
                                 const torch::Tensor rns_params) {
  rns_montgomery_mul_row_scalars_inplace_cuda(
      residues, row_scalars, rns_params);
}

void to_montgomery_(torch::Tensor standard_residues,
                    const torch::Tensor rns_params) {
  rns_to_montgomery_inplace_cuda(standard_residues, rns_params);
}

void from_montgomery_(torch::Tensor montgomery_residues,
                      const torch::Tensor rns_params) {
  rns_from_montgomery_inplace_cuda(montgomery_residues, rns_params);
}

torch::Tensor add_lazy(const torch::Tensor lhs,
                       const torch::Tensor rhs,
                       const torch::Tensor rns_params) {
  return rns_add_lazy_cuda(lhs, rhs, rns_params);
}

torch::Tensor add_lazy_with_twice_modulus(const torch::Tensor lhs,
                                          const torch::Tensor rhs,
                                          const torch::Tensor twice_modulus) {
  return rns_add_lazy_with_twice_modulus_cuda(lhs, rhs, twice_modulus);
}

torch::Tensor sub_lazy(const torch::Tensor lhs,
                       const torch::Tensor rhs,
                       const torch::Tensor rns_params) {
  return rns_sub_lazy_cuda(lhs, rhs, rns_params);
}

void canonicalize_residues_(torch::Tensor lazy_residues,
                            const torch::Tensor rns_params) {
  rns_canonicalize_residues_inplace_cuda(lazy_residues, rns_params);
}

void center_residues_(torch::Tensor canonical_residues,
                      const torch::Tensor rns_params) {
  rns_center_residues_inplace_cuda(canonical_residues, rns_params);
}

void shift_residues_positive_(torch::Tensor centered_residues,
                              const torch::Tensor rns_params) {
  rns_shift_residues_positive_inplace_cuda(centered_residues, rns_params);
}

torch::Tensor lift_centered_coefficients(
    const torch::Tensor centered_coefficients,
    const torch::Tensor twice_modulus) {
  return rns_lift_centered_coefficients_cuda(centered_coefficients,
                                             twice_modulus);
}

torch::Tensor mixed_radix_decompose(
    const torch::Tensor source_residues,
    const torch::Tensor mixed_radix_normalizers,
    const torch::Tensor mixed_radix_propagation_coefficients,
    const torch::Tensor modulus_lo,
    const torch::Tensor modulus_hi,
    const torch::Tensor neg_inv_modulus_lo,
    const torch::Tensor neg_inv_modulus_hi) {
  return mixed_radix_decompose_cuda(source_residues,
                                    mixed_radix_normalizers,
                                    mixed_radix_propagation_coefficients,
                                    modulus_lo,
                                    modulus_hi,
                                    neg_inv_modulus_lo,
                                    neg_inv_modulus_hi);
}

torch::Tensor mixed_radix_basis_extend_to_montgomery(
    const torch::Tensor mixed_radix_components,
    const torch::Tensor basis_extension_coefficients,
    const torch::Tensor rns_params,
    const int64_t destination_row_count) {
  return mixed_radix_basis_extend_to_montgomery_cuda(
      mixed_radix_components,
      basis_extension_coefficients,
      rns_params,
      destination_row_count);
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CUDA, m) {
  m.impl("montgomery_mul", &montgomery_mul);
  m.impl("montgomery_mul_cyclic_compressed", &montgomery_mul_cyclic_compressed);
  m.impl("montgomery_mul_contiguous_compressed",
         &montgomery_mul_contiguous_compressed);
  m.impl("montgomery_mul_row_scalars_", &montgomery_mul_row_scalars_);
  m.impl("to_montgomery_", &to_montgomery_);
  m.impl("from_montgomery_", &from_montgomery_);
  m.impl("add_lazy", &add_lazy);
  m.impl("add_lazy_with_twice_modulus", &add_lazy_with_twice_modulus);
  m.impl("sub_lazy", &sub_lazy);
  m.impl("canonicalize_residues_", &canonicalize_residues_);
  m.impl("center_residues_", &center_residues_);
  m.impl("shift_residues_positive_", &shift_residues_positive_);
  m.impl("lift_centered_coefficients", &lift_centered_coefficients);
  m.impl("mixed_radix_decompose", &mixed_radix_decompose);
  m.impl("mixed_radix_basis_extend_to_montgomery",
         &mixed_radix_basis_extend_to_montgomery);
}
