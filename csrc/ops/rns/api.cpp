#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include "backends.h"

#include "../common/backend_dispatch.h"
#include "../common/overlap.h"
#include "../operations/composition.h"

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::rns::"

namespace fhelium::rns {

at::Tensor montgomery_mul(const at::Tensor& lhs,
                          const at::Tensor& rhs,
                          const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(montgomery_mul, lhs, rhs, rns_params);
}

void montgomery_weighted_sum_out(at::TensorList ciphertexts,
                                 at::TensorList plaintexts,
                                 const at::Tensor& rns_params,
                                 at::Tensor& out) {
  TORCH_CHECK_VALUE(!ciphertexts.empty(),
                    "montgomery_weighted_sum requires at least one term");
  bool disjoint = true;
  for (const auto& term : ciphertexts)
    disjoint = disjoint && detail::storage_disjoint(out, term);
  for (const auto& term : plaintexts)
    disjoint = disjoint && detail::storage_disjoint(out, term);
  TORCH_CHECK_VALUE(out.sizes() == ciphertexts[0].sizes() &&
                        out.scalar_type() == ciphertexts[0].scalar_type() &&
                        out.device() == ciphertexts[0].device() && disjoint,
                    "montgomery_weighted_sum_out requires an output with the "
                    "ciphertext shape, dtype and device, disjoint from the "
                    "terms");
  FHELIUM_DISPATCH_CPU_GPU(
      montgomery_weighted_sum_out, ciphertexts, plaintexts, rns_params, out);
}

at::Tensor montgomery_weighted_sum(at::TensorList ciphertexts,
                                   at::TensorList plaintexts,
                                   const at::Tensor& rns_params) {
  TORCH_CHECK_VALUE(!ciphertexts.empty(),
                    "montgomery_weighted_sum requires at least one term");
  auto out = at::empty_like(ciphertexts[0]);
  montgomery_weighted_sum_out(ciphertexts, plaintexts, rns_params, out);
  return out;
}

namespace {

// Checks the output of an elementwise out form: the shape, dtype and device
// of `like`, and either `like` itself or storage-disjoint from it.
void check_elementwise_out(const at::Tensor& out,
                           const at::Tensor& like,
                           const char* operation) {
  TORCH_CHECK_VALUE(
      out.sizes() == like.sizes() && out.scalar_type() == like.scalar_type() &&
          out.device() == like.device() &&
          (detail::same_elements(out, like) ||
           detail::storage_disjoint(out, like)),
      operation,
      " requires an output with the operand shape, dtype and device that is "
      "the operand or disjoint from it");
}

std::vector<int64_t> weighted_sums_shape(at::TensorList ciphertexts,
                                         int64_t group_count) {
  TORCH_CHECK_VALUE(!ciphertexts.empty() && ciphertexts[0].dim() >= 1,
                    "montgomery_weighted_sums requires at least one term");
  std::vector<int64_t> shape{ciphertexts[0].size(0), group_count};
  shape.insert(shape.end(),
               ciphertexts[0].sizes().begin() + 1,
               ciphertexts[0].sizes().end());
  return shape;
}

}  // namespace

void montgomery_weighted_sums_out(at::TensorList ciphertexts,
                                  at::TensorList plaintexts,
                                  int64_t group_count,
                                  const at::Tensor& rns_params,
                                  at::Tensor& out) {
  const auto shape = weighted_sums_shape(ciphertexts, group_count);
  TORCH_CHECK_VALUE(out.sizes() == c10::IntArrayRef(shape) &&
                        out.scalar_type() == ciphertexts[0].scalar_type() &&
                        out.device() == ciphertexts[0].device() &&
                        out.is_contiguous(),
                    "montgomery_weighted_sums_out requires a contiguous "
                    "output of shape ",
                    c10::IntArrayRef(shape),
                    " with the ciphertext dtype and device");
  FHELIUM_DISPATCH_CPU_GPU(montgomery_weighted_sums_out,
                           ciphertexts,
                           plaintexts,
                           group_count,
                           rns_params,
                           out);
}

at::Tensor montgomery_weighted_sums(at::TensorList ciphertexts,
                                    at::TensorList plaintexts,
                                    int64_t group_count,
                                    const at::Tensor& rns_params) {
  auto out = at::empty(weighted_sums_shape(ciphertexts, group_count),
                       ciphertexts[0].options());
  montgomery_weighted_sums_out(
      ciphertexts, plaintexts, group_count, rns_params, out);
  return out;
}

void montgomery_mul_cyclic_compressed_out(const at::Tensor& lhs,
                                          const at::Tensor& compressed_rhs,
                                          const at::Tensor& rns_params,
                                          at::Tensor& out) {
  check_elementwise_out(out, lhs, "montgomery_mul_cyclic_compressed_out");
  TORCH_CHECK_VALUE(detail::storage_disjoint(out, compressed_rhs),
                    "montgomery_mul_cyclic_compressed_out requires an output "
                    "disjoint from compressed_rhs");
  FHELIUM_DISPATCH_CPU_GPU(montgomery_mul_cyclic_compressed_out,
                           lhs,
                           compressed_rhs,
                           rns_params,
                           out);
}

void montgomery_mul_contiguous_compressed_out(const at::Tensor& lhs,
                                              const at::Tensor& compressed_rhs,
                                              const at::Tensor& rns_params,
                                              at::Tensor& out) {
  check_elementwise_out(out, lhs, "montgomery_mul_contiguous_compressed_out");
  TORCH_CHECK_VALUE(detail::storage_disjoint(out, compressed_rhs),
                    "montgomery_mul_contiguous_compressed_out requires an "
                    "output disjoint from compressed_rhs");
  FHELIUM_DISPATCH_CPU_GPU(montgomery_mul_contiguous_compressed_out,
                           lhs,
                           compressed_rhs,
                           rns_params,
                           out);
}

at::Tensor montgomery_mul_cyclic_compressed(const at::Tensor& lhs,
                                            const at::Tensor& compressed_rhs,
                                            const at::Tensor& rns_params) {
  auto out = at::empty_like(lhs);
  montgomery_mul_cyclic_compressed_out(lhs, compressed_rhs, rns_params, out);
  return out;
}

at::Tensor montgomery_mul_contiguous_compressed(
    const at::Tensor& lhs,
    const at::Tensor& compressed_rhs,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(lhs);
  montgomery_mul_contiguous_compressed_out(
      lhs, compressed_rhs, rns_params, out);
  return out;
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

void montgomery_mul_out(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params,
                        at::Tensor& out) {
  FHELIUM_DISPATCH_CPU_GPU(montgomery_mul_out, lhs, rhs, rns_params, out);
}

void add_lazy_out(const at::Tensor& lhs,
                  const at::Tensor& rhs,
                  const at::Tensor& rns_params,
                  at::Tensor& out) {
  FHELIUM_DISPATCH_CPU_GPU(add_lazy_out, lhs, rhs, rns_params, out);
}

at::Tensor add_lazy(const at::Tensor& lhs,
                    const at::Tensor& rhs,
                    const at::Tensor& rns_params) {
  auto out = at::empty_like(lhs);
  add_lazy_out(lhs, rhs, rns_params, out);
  return out;
}

void add_lazy_with_twice_modulus_out(const at::Tensor& lhs,
                                     const at::Tensor& rhs,
                                     const at::Tensor& twice_modulus,
                                     at::Tensor& out) {
  check_elementwise_out(out, lhs, "add_lazy_with_twice_modulus_out");
  FHELIUM_DISPATCH_CPU_GPU(
      add_lazy_with_twice_modulus_out, lhs, rhs, twice_modulus, out);
}

at::Tensor add_lazy_with_twice_modulus(const at::Tensor& lhs,
                                       const at::Tensor& rhs,
                                       const at::Tensor& twice_modulus) {
  auto out = at::empty_like(lhs);
  add_lazy_with_twice_modulus_out(lhs, rhs, twice_modulus, out);
  return out;
}

void sub_lazy_out(const at::Tensor& lhs,
                  const at::Tensor& rhs,
                  const at::Tensor& rns_params,
                  at::Tensor& out) {
  check_elementwise_out(out, lhs, "sub_lazy_out");
  FHELIUM_DISPATCH_CPU_GPU(sub_lazy_out, lhs, rhs, rns_params, out);
}

at::Tensor sub_lazy(const at::Tensor& lhs,
                    const at::Tensor& rhs,
                    const at::Tensor& rns_params) {
  auto out = at::empty_like(lhs);
  sub_lazy_out(lhs, rhs, rns_params, out);
  return out;
}

void add_standard_out(const at::Tensor& lhs,
                      const at::Tensor& rhs,
                      const at::Tensor& rns_params,
                      at::Tensor& out) {
  check_elementwise_out(out, lhs, "add_standard_out");
  FHELIUM_DISPATCH_CPU_GPU(add_standard_out, lhs, rhs, rns_params, out);
}

at::Tensor add_standard(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params) {
  auto out = at::empty_like(lhs);
  add_standard_out(lhs, rhs, rns_params, out);
  return out;
}

void add_standard_(at::Tensor& lhs,
                   const at::Tensor& rhs,
                   const at::Tensor& rns_params) {
  add_standard_out(lhs, rhs, rns_params, lhs);
}

void sum_standard_batch_out(const at::Tensor& source,
                            int64_t dim,
                            const at::Tensor& rns_params,
                            at::Tensor& out) {
  TORCH_CHECK_VALUE(dim >= 0 && dim <= source.dim() - 3,
                    "sum_standard_batch dim must select a leading batch axis");
  auto sizes = source.sizes().vec();
  sizes.erase(sizes.begin() + dim);
  TORCH_CHECK_VALUE(out.sizes() == c10::IntArrayRef(sizes) &&
                        out.scalar_type() == source.scalar_type() &&
                        out.device() == source.device() &&
                        detail::storage_disjoint(out, source),
                    "sum_standard_batch_out requires an output of shape ",
                    c10::IntArrayRef(sizes),
                    " with the source dtype and device, disjoint from it");
  FHELIUM_DISPATCH_CPU_GPU(
      sum_standard_batch_out, source, dim, rns_params, out);
}

at::Tensor sum_standard_batch(const at::Tensor& source,
                              int64_t dim,
                              const at::Tensor& rns_params) {
  TORCH_CHECK_VALUE(dim >= 0 && dim <= source.dim() - 3,
                    "sum_standard_batch dim must select a leading batch axis");
  auto sizes = source.sizes().vec();
  sizes.erase(sizes.begin() + dim);
  auto out = at::empty(sizes, source.options());
  sum_standard_batch_out(source, dim, rns_params, out);
  return out;
}

void sub_standard_out(const at::Tensor& lhs,
                      const at::Tensor& rhs,
                      const at::Tensor& rns_params,
                      at::Tensor& out) {
  check_elementwise_out(out, lhs, "sub_standard_out");
  FHELIUM_DISPATCH_CPU_GPU(sub_standard_out, lhs, rhs, rns_params, out);
}

at::Tensor sub_standard(const at::Tensor& lhs,
                        const at::Tensor& rhs,
                        const at::Tensor& rns_params) {
  auto out = at::empty_like(lhs);
  sub_standard_out(lhs, rhs, rns_params, out);
  return out;
}

void sub_standard_(at::Tensor& lhs,
                   const at::Tensor& rhs,
                   const at::Tensor& rns_params) {
  sub_standard_out(lhs, rhs, rns_params, lhs);
}

void negate_standard_out(const at::Tensor& source,
                         const at::Tensor& rns_params,
                         at::Tensor& out) {
  check_elementwise_out(out, source, "negate_standard_out");
  FHELIUM_DISPATCH_CPU_GPU(negate_standard_out, source, rns_params, out);
}

at::Tensor negate_standard(const at::Tensor& source,
                           const at::Tensor& rns_params) {
  auto out = at::empty_like(source);
  negate_standard_out(source, rns_params, out);
  return out;
}

void negate_standard_(at::Tensor& source, const at::Tensor& rns_params) {
  negate_standard_out(source, rns_params, source);
}

void montgomery_mul_row_scalars_standard_out(const at::Tensor& residues,
                                             const at::Tensor& row_scalars,
                                             const at::Tensor& rns_params,
                                             at::Tensor& out) {
  TORCH_CHECK_VALUE(
      out.sizes() == residues.sizes() &&
          out.scalar_type() == residues.scalar_type() &&
          out.device() == residues.device() &&
          (detail::same_elements(out, residues) ||
           detail::storage_disjoint(out, residues)),
      "montgomery_mul_row_scalars_standard_out requires an output with the "
      "residue "
      "shape, dtype and device that is the input or disjoint from it");
  FHELIUM_DISPATCH_CPU_GPU(montgomery_mul_row_scalars_standard_out,
                           residues,
                           row_scalars,
                           rns_params,
                           out);
}

at::Tensor montgomery_mul_row_scalars_standard(const at::Tensor& residues,
                                               const at::Tensor& row_scalars,
                                               const at::Tensor& rns_params) {
  auto out = at::empty_like(residues);
  montgomery_mul_row_scalars_standard_out(
      residues, row_scalars, rns_params, out);
  return out;
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

at::Tensor lift_integer_coefficients(const at::Tensor& coefficients,
                                     const at::Tensor& twice_modulus) {
  FHELIUM_DISPATCH_CPU_GPU(
      lift_integer_coefficients, coefficients, twice_modulus);
}

#if defined(FHELIUM_WITH_CUDA)
namespace {

// The CUDA kernel stages a 16-target int32 matrix chunk in 48 KiB of shared
// memory, which admits up to 768 source rows; other inputs use the portable
// sum.
bool crt_kernel_applies(const at::Tensor& source, const at::Tensor& matrix) {
  return source.is_cuda() && source.scalar_type() == at::kInt &&
         matrix.size(0) * 16 * source.element_size() <= 48 * 1024;
}

}  // namespace
#endif

void crt_basis_extend_out(const at::Tensor& source_residues,
                          const at::Tensor& source_parameters,
                          const at::Tensor& target_parameters,
                          const at::Tensor& matrix,
                          const at::Tensor& target_reciprocals,
                          const std::optional<at::Tensor>& source_normalizers,
                          const std::optional<at::Tensor>& source_reciprocals,
                          const std::optional<at::Tensor>& target_offsets,
                          at::Tensor& output) {
#if defined(FHELIUM_WITH_CUDA)
  if (crt_kernel_applies(source_residues, matrix)) {
    gpu::crt_basis_extend_out(source_residues,
                              source_parameters,
                              target_parameters,
                              matrix,
                              target_reciprocals,
                              source_normalizers,
                              source_reciprocals,
                              target_offsets,
                              output);
    return;
  }
#endif
  output.copy_(detail_operations::portable_crt_basis_extend(source_residues,
                                                            source_parameters,
                                                            target_parameters,
                                                            matrix,
                                                            source_normalizers,
                                                            target_offsets));
}

at::Tensor crt_basis_extend(const at::Tensor& source_residues,
                            const at::Tensor& source_parameters,
                            const at::Tensor& target_parameters,
                            const at::Tensor& matrix,
                            const at::Tensor& target_reciprocals,
                            const std::optional<at::Tensor>& source_normalizers,
                            const std::optional<at::Tensor>& source_reciprocals,
                            const std::optional<at::Tensor>& target_offsets) {
#if defined(FHELIUM_WITH_CUDA)
  if (crt_kernel_applies(source_residues, matrix)) {
    return gpu::crt_basis_extend(source_residues,
                                 source_parameters,
                                 target_parameters,
                                 matrix,
                                 target_reciprocals,
                                 source_normalizers,
                                 source_reciprocals,
                                 target_offsets);
  }
#endif
  return detail_operations::portable_crt_basis_extend(source_residues,
                                                      source_parameters,
                                                      target_parameters,
                                                      matrix,
                                                      source_normalizers,
                                                      target_offsets);
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

void mixed_radix_modup_tiled_out(const at::Tensor& source_residues,
                                 at::TensorList decomposition_tables,
                                 const at::Tensor& extension,
                                 const at::Tensor& parameters,
                                 at::Tensor& out) {
  FHELIUM_DISPATCH_CUDA(mixed_radix_modup_tiled_out,
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

void mixed_radix_decompose_out(
    const at::Tensor& source_residues,
    const at::Tensor& mixed_radix_normalizers,
    const at::Tensor& mixed_radix_propagation_coefficients,
    const at::Tensor& modulus_lo,
    const at::Tensor& modulus_hi,
    const at::Tensor& neg_inv_modulus_lo,
    const at::Tensor& neg_inv_modulus_hi,
    at::Tensor& out) {
  FHELIUM_DISPATCH_CPU_GPU(mixed_radix_decompose_out,
                           source_residues,
                           mixed_radix_normalizers,
                           mixed_radix_propagation_coefficients,
                           modulus_lo,
                           modulus_hi,
                           neg_inv_modulus_lo,
                           neg_inv_modulus_hi,
                           out);
}

void mixed_radix_basis_extend_to_montgomery_out(
    const at::Tensor& mixed_radix_components,
    const at::Tensor& basis_extension_coefficients,
    const at::Tensor& rns_params,
    int64_t destination_row_count,
    at::Tensor& out) {
  FHELIUM_DISPATCH_CPU_GPU(mixed_radix_basis_extend_to_montgomery_out,
                           mixed_radix_components,
                           basis_extension_coefficients,
                           rns_params,
                           destination_row_count,
                           out);
}

}  // namespace fhelium::rns
