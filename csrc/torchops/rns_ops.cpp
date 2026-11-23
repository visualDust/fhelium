#include <torch/library.h>

#include <fhelium/rns.h>

// Torch operator schemas and backend registrations for the public functions
// of the same names. Each registered kernel is the public function itself.

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def("montgomery_mul(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_weighted_sum(Tensor[] ciphertexts, Tensor[] plaintexts, "
      "Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_weighted_sums(Tensor[] ciphertexts, Tensor[] plaintexts, "
      "int group_count, Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_mul_cyclic_compressed(Tensor lhs, Tensor compressed_rhs, "
      "Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_mul_contiguous_compressed(Tensor lhs, Tensor "
      "compressed_rhs, Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_mul_row_scalars_(Tensor(a!) residues, Tensor row_scalars, "
      "Tensor rns_params) -> ()");
  m.def(
      "to_montgomery_(Tensor(a!) standard_residues, Tensor rns_params) -> ()");
  m.def(
      "from_montgomery_(Tensor(a!) montgomery_residues, Tensor rns_params) -> "
      "()");
  m.def("add_lazy(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def(
      "add_lazy_with_twice_modulus(Tensor lhs, Tensor rhs, Tensor "
      "twice_modulus) -> Tensor");
  m.def("sub_lazy(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def("add_standard(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def("add_standard_(Tensor(a!) lhs, Tensor rhs, Tensor rns_params) -> ()");
  m.def(
      "sum_standard_batch(Tensor source, int dim, Tensor rns_params) -> "
      "Tensor");
  m.def("sub_standard(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def("sub_standard_(Tensor(a!) lhs, Tensor rhs, Tensor rns_params) -> ()");
  m.def(
      "montgomery_mul_row_scalars_standard(Tensor residues, Tensor "
      "row_scalars, Tensor rns_params) -> Tensor");
  m.def(
      "reduce_to_standard_(Tensor(a!) lazy_residues, Tensor rns_params) -> "
      "()");
  m.def(
      "center_residues_(Tensor(a!) standard_residues, Tensor rns_params) -> "
      "()");
  m.def(
      "shift_residues_positive_(Tensor(a!) centered_residues, Tensor "
      "rns_params) -> ()");
  m.def(
      "lift_integer_coefficients(Tensor coefficients, Tensor twice_modulus) "
      "-> Tensor");
  m.def(
      "crt_basis_extend_out_(Tensor source_residues, Tensor source_parameters, "
      "Tensor target_parameters, Tensor matrix, Tensor target_reciprocals, "
      "Tensor? source_normalizers, Tensor? source_reciprocals, Tensor(a!) "
      "output) -> ()");
  m.def(
      "crt_basis_extend(Tensor source_residues, Tensor source_parameters, "
      "Tensor target_parameters, Tensor matrix, Tensor target_reciprocals, "
      "Tensor? source_normalizers=None, Tensor? source_reciprocals=None) -> "
      "Tensor");
  m.def(
      "mixed_radix_modup_tiled(Tensor source_residues, Tensor[] "
      "decomposition_tables, "
      "Tensor extension, Tensor parameters) -> Tensor");
  m.def(
      "mixed_radix_modup_tiled_out_(Tensor source_residues, Tensor[] "
      "decomposition_tables, "
      "Tensor extension, Tensor parameters, Tensor(a!) out) -> ()");
  m.def(
      "mixed_radix_decompose(Tensor source_residues, Tensor "
      "mixed_radix_normalizers, Tensor mixed_radix_propagation_coefficients, "
      "Tensor modulus_lo, Tensor modulus_hi, Tensor neg_inv_modulus_lo, Tensor "
      "neg_inv_modulus_hi) -> Tensor");
  m.def(
      "mixed_radix_basis_extend_to_montgomery(Tensor "
      "mixed_radix_components, Tensor basis_extension_coefficients, Tensor "
      "rns_params, int destination_row_count) -> Tensor");
  m.def(
      "mixed_radix_decompose_out_(Tensor source_residues, Tensor "
      "mixed_radix_normalizers, Tensor mixed_radix_propagation_coefficients, "
      "Tensor modulus_lo, Tensor modulus_hi, Tensor neg_inv_modulus_lo, Tensor "
      "neg_inv_modulus_hi, Tensor(a!) out) -> ()");
  m.def(
      "mixed_radix_basis_extend_to_montgomery_out_(Tensor "
      "mixed_radix_components, Tensor basis_extension_coefficients, Tensor "
      "rns_params, int destination_row_count, Tensor(a!) out) -> ()");
}

#if defined(FHELIUM_WITH_CPU)
TORCH_LIBRARY_IMPL(fhelium_rns_ops, CPU, m) {
  m.impl("montgomery_mul", &fhelium::rns::montgomery_mul);
  m.impl("montgomery_weighted_sum", &fhelium::rns::montgomery_weighted_sum);
  m.impl("montgomery_weighted_sums", &fhelium::rns::montgomery_weighted_sums);
  m.impl("montgomery_mul_cyclic_compressed",
         &fhelium::rns::montgomery_mul_cyclic_compressed);
  m.impl("montgomery_mul_contiguous_compressed",
         &fhelium::rns::montgomery_mul_contiguous_compressed);
  m.impl("montgomery_mul_row_scalars_",
         &fhelium::rns::montgomery_mul_row_scalars_);
  m.impl("to_montgomery_", &fhelium::rns::to_montgomery_);
  m.impl("from_montgomery_", &fhelium::rns::from_montgomery_);
  m.impl("add_lazy", &fhelium::rns::add_lazy);
  m.impl("add_lazy_with_twice_modulus",
         &fhelium::rns::add_lazy_with_twice_modulus);
  m.impl("sub_lazy", &fhelium::rns::sub_lazy);
  m.impl("add_standard", &fhelium::rns::add_standard);
  m.impl("add_standard_", &fhelium::rns::add_standard_);
  m.impl("sum_standard_batch", &fhelium::rns::sum_standard_batch);
  m.impl("sub_standard", &fhelium::rns::sub_standard);
  m.impl("sub_standard_", &fhelium::rns::sub_standard_);
  m.impl("montgomery_mul_row_scalars_standard",
         &fhelium::rns::montgomery_mul_row_scalars_standard);
  m.impl("reduce_to_standard_", &fhelium::rns::reduce_to_standard_);
  m.impl("center_residues_", &fhelium::rns::center_residues_);
  m.impl("shift_residues_positive_", &fhelium::rns::shift_residues_positive_);
  m.impl("lift_integer_coefficients", &fhelium::rns::lift_integer_coefficients);
  m.impl("mixed_radix_decompose", &fhelium::rns::mixed_radix_decompose);
  m.impl("mixed_radix_basis_extend_to_montgomery",
         &fhelium::rns::mixed_radix_basis_extend_to_montgomery);
  m.impl("mixed_radix_decompose_out_",
         &fhelium::rns::mixed_radix_decompose_out_);
  m.impl("mixed_radix_basis_extend_to_montgomery_out_",
         &fhelium::rns::mixed_radix_basis_extend_to_montgomery_out_);
  m.impl("crt_basis_extend_out_", &fhelium::rns::crt_basis_extend_out_);
  m.impl("crt_basis_extend", &fhelium::rns::crt_basis_extend);
}
#endif

#if defined(FHELIUM_WITH_CUDA) || defined(FHELIUM_WITH_HIP)
TORCH_LIBRARY_IMPL(fhelium_rns_ops, CUDA, m) {
  m.impl("montgomery_mul", &fhelium::rns::montgomery_mul);
  m.impl("montgomery_weighted_sum", &fhelium::rns::montgomery_weighted_sum);
  m.impl("montgomery_weighted_sums", &fhelium::rns::montgomery_weighted_sums);
  m.impl("montgomery_mul_cyclic_compressed",
         &fhelium::rns::montgomery_mul_cyclic_compressed);
  m.impl("montgomery_mul_contiguous_compressed",
         &fhelium::rns::montgomery_mul_contiguous_compressed);
  m.impl("montgomery_mul_row_scalars_",
         &fhelium::rns::montgomery_mul_row_scalars_);
  m.impl("to_montgomery_", &fhelium::rns::to_montgomery_);
  m.impl("from_montgomery_", &fhelium::rns::from_montgomery_);
  m.impl("add_lazy", &fhelium::rns::add_lazy);
  m.impl("add_lazy_with_twice_modulus",
         &fhelium::rns::add_lazy_with_twice_modulus);
  m.impl("sub_lazy", &fhelium::rns::sub_lazy);
  m.impl("add_standard", &fhelium::rns::add_standard);
  m.impl("add_standard_", &fhelium::rns::add_standard_);
  m.impl("sum_standard_batch", &fhelium::rns::sum_standard_batch);
  m.impl("sub_standard", &fhelium::rns::sub_standard);
  m.impl("sub_standard_", &fhelium::rns::sub_standard_);
  m.impl("montgomery_mul_row_scalars_standard",
         &fhelium::rns::montgomery_mul_row_scalars_standard);
  m.impl("reduce_to_standard_", &fhelium::rns::reduce_to_standard_);
  m.impl("center_residues_", &fhelium::rns::center_residues_);
  m.impl("shift_residues_positive_", &fhelium::rns::shift_residues_positive_);
  m.impl("lift_integer_coefficients", &fhelium::rns::lift_integer_coefficients);
  m.impl("mixed_radix_decompose", &fhelium::rns::mixed_radix_decompose);
  m.impl("mixed_radix_basis_extend_to_montgomery",
         &fhelium::rns::mixed_radix_basis_extend_to_montgomery);
  m.impl("mixed_radix_decompose_out_",
         &fhelium::rns::mixed_radix_decompose_out_);
  m.impl("mixed_radix_basis_extend_to_montgomery_out_",
         &fhelium::rns::mixed_radix_basis_extend_to_montgomery_out_);
  m.impl("crt_basis_extend_out_", &fhelium::rns::crt_basis_extend_out_);
  m.impl("crt_basis_extend", &fhelium::rns::crt_basis_extend);
#if defined(FHELIUM_WITH_CUDA)
  m.impl("mixed_radix_modup_tiled", &fhelium::rns::mixed_radix_modup_tiled);
  m.impl("mixed_radix_modup_tiled_out_",
         &fhelium::rns::mixed_radix_modup_tiled_out_);
#endif
}
#endif
