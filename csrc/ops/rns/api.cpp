#include "backends.h"

#include "../common/backend_dispatch.h"

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::rns::"

namespace fhelium::rns {

at::Tensor montgomery_mul(const at::Tensor& lhs,
                          const at::Tensor& rhs,
                          const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(montgomery_mul, lhs, rhs, rns_params);
}

at::Tensor montgomery_weighted_sum(at::TensorList ciphertexts,
                                   at::TensorList plaintexts,
                                   const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      montgomery_weighted_sum, ciphertexts, plaintexts, rns_params);
}

at::Tensor montgomery_weighted_sums(at::TensorList ciphertexts,
                                    at::TensorList plaintexts,
                                    int64_t group_count,
                                    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(montgomery_weighted_sums,
                           ciphertexts,
                           plaintexts,
                           group_count,
                           rns_params);
}

at::Tensor montgomery_mul_cyclic_compressed(const at::Tensor& lhs,
                                            const at::Tensor& compressed_rhs,
                                            const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      montgomery_mul_cyclic_compressed, lhs, compressed_rhs, rns_params);
}

at::Tensor montgomery_mul_contiguous_compressed(
    const at::Tensor& lhs,
    const at::Tensor& compressed_rhs,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      montgomery_mul_contiguous_compressed, lhs, compressed_rhs, rns_params);
}

void montgomery_mul_row_scalars_(at::Tensor& residues,
                                 const at::Tensor& row_scalars,
                                 const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      montgomery_mul_row_scalars_, residues, row_scalars, rns_params);
}

void to_montgomery_(at::Tensor& standard_residues,
                    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(to_montgomery_, standard_residues, rns_params);
}

void from_montgomery_(at::Tensor& montgomery_residues,
                      const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(from_montgomery_, montgomery_residues, rns_params);
}

at::Tensor add_lazy(const at::Tensor& lhs,
                    const at::Tensor& rhs,
                    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_lazy, lhs, rhs, rns_params);
}

at::Tensor add_lazy_with_twice_modulus(const at::Tensor& lhs,
                                       const at::Tensor& rhs,
                                       const at::Tensor& twice_modulus) {
  FHELIUM_DISPATCH_CPU_GPU(
      add_lazy_with_twice_modulus, lhs, rhs, twice_modulus);
}

at::Tensor sub_lazy(const at::Tensor& lhs,
                    const at::Tensor& rhs,
                    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(sub_lazy, lhs, rhs, rns_params);
}

at::Tensor add_standard(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_standard, lhs, rhs, rns_params);
}

void add_standard_(at::Tensor& lhs,
                   const at::Tensor& rhs,
                   const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_standard_, lhs, rhs, rns_params);
}

at::Tensor sum_standard_batch(const at::Tensor& source,
                              int64_t dim,
                              const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(sum_standard_batch, source, dim, rns_params);
}

at::Tensor sub_standard(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(sub_standard, lhs, rhs, rns_params);
}

void sub_standard_(at::Tensor& lhs,
                   const at::Tensor& rhs,
                   const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(sub_standard_, lhs, rhs, rns_params);
}

at::Tensor montgomery_mul_row_scalars_standard(const at::Tensor& residues,
                                               const at::Tensor& row_scalars,
                                               const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      montgomery_mul_row_scalars_standard, residues, row_scalars, rns_params);
}

void reduce_to_standard_(at::Tensor& lazy_residues,
                         const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(reduce_to_standard_, lazy_residues, rns_params);
}

void center_residues_(at::Tensor& standard_residues,
                      const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(center_residues_, standard_residues, rns_params);
}

void shift_residues_positive_(at::Tensor& centered_residues,
                              const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      shift_residues_positive_, centered_residues, rns_params);
}

at::Tensor lift_centered_coefficients(const at::Tensor& centered_coefficients,
                                      const at::Tensor& twice_modulus) {
  FHELIUM_DISPATCH_CPU_GPU(
      lift_centered_coefficients, centered_coefficients, twice_modulus);
}

void crt_basis_extend_out_(const at::Tensor& source_residues,
                           const at::Tensor& source_parameters,
                           const at::Tensor& target_parameters,
                           const at::Tensor& matrix,
                           const at::Tensor& target_reciprocals,
                           const std::optional<at::Tensor>& source_normalizers,
                           const std::optional<at::Tensor>& source_reciprocals,
                           at::Tensor& output) {
  FHELIUM_DISPATCH_CUDA(crt_basis_extend_out_,
                        source_residues,
                        source_parameters,
                        target_parameters,
                        matrix,
                        target_reciprocals,
                        source_normalizers,
                        source_reciprocals,
                        output);
}

at::Tensor crt_basis_extend(
    const at::Tensor& source_residues,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const at::Tensor& target_reciprocals,
    const std::optional<at::Tensor>& source_normalizers,
    const std::optional<at::Tensor>& source_reciprocals) {
  FHELIUM_DISPATCH_CUDA(crt_basis_extend,
                        source_residues,
                        source_parameters,
                        target_parameters,
                        matrix,
                        target_reciprocals,
                        source_normalizers,
                        source_reciprocals);
}

at::Tensor mixed_radix_modup_tiled(const at::Tensor& source_residues,
                                   at::TensorList decomposition_tables,
                                   const at::Tensor& extension,
                                   const at::Tensor& parameters) {
  FHELIUM_DISPATCH_CUDA(mixed_radix_modup_tiled,
                        source_residues,
                        decomposition_tables,
                        extension,
                        parameters);
}

void mixed_radix_modup_tiled_out_(const at::Tensor& source_residues,
                                  at::TensorList decomposition_tables,
                                  const at::Tensor& extension,
                                  const at::Tensor& parameters,
                                  at::Tensor& out) {
  FHELIUM_DISPATCH_CUDA(mixed_radix_modup_tiled_out_,
                        source_residues,
                        decomposition_tables,
                        extension,
                        parameters,
                        out);
}

at::Tensor mixed_radix_decompose(
    const at::Tensor& source_residues,
    const at::Tensor& mixed_radix_normalizers,
    const at::Tensor& mixed_radix_propagation_coefficients,
    const at::Tensor& modulus_lo,
    const at::Tensor& modulus_hi,
    const at::Tensor& neg_inv_modulus_lo,
    const at::Tensor& neg_inv_modulus_hi) {
  FHELIUM_DISPATCH_CPU_GPU(mixed_radix_decompose,
                           source_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(mixed_radix_basis_extend_to_montgomery,
                           mixed_radix_components,
                           basis_extension_coefficients,
                           rns_params,
                           destination_row_count);
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
  FHELIUM_DISPATCH_CPU_GPU(mixed_radix_decompose_out_,
                           source_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(mixed_radix_basis_extend_to_montgomery_out_,
                           mixed_radix_components,
                           basis_extension_coefficients,
                           rns_params,
                           destination_row_count,
                           out);
}

}  // namespace fhelium::rns
