#include <torch/library.h>

#include <fhelium/ckks.h>

// Torch operator schemas and backend registrations for the public functions
// of the same names. Each registered kernel is the public function itself.

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "multiply_two_component_ntt_montgomery(Tensor lhs_components, Tensor "
      "rhs_components, Tensor rns_params) -> Tensor");
  m.def(
      "add_prepared_plaintext_component(Tensor ciphertext_component, Tensor "
      "prepared_plaintext, Tensor rns_params) -> Tensor");
  m.def(
      "add_prepared_plaintext_component_(Tensor(a!) ciphertext_component, "
      "Tensor prepared_plaintext, Tensor rns_params) -> ()");
  m.def(
      "add_cyclic_compressed_plaintext_component(Tensor "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> Tensor");
  m.def(
      "add_cyclic_compressed_plaintext_component_(Tensor(a!) "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> ()");
  m.def(
      "add_contiguous_compressed_plaintext_component(Tensor "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> Tensor");
  m.def(
      "add_contiguous_compressed_plaintext_component_(Tensor(a!) "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> ()");
  m.def(
      "add_strided_plaintext_component(Tensor ciphertext_component, Tensor "
      "strided_plaintext, Tensor implicit_plaintext, Tensor rns_params) -> "
      "Tensor");
  m.def(
      "add_strided_plaintext_component_(Tensor(a!) ciphertext_component, "
      "Tensor strided_plaintext, Tensor implicit_plaintext, Tensor "
      "rns_params) -> ()");
  m.def(
      "rescale_drop_leading_prime_nearest(Tensor remaining_residues, Tensor "
      "drop_prime_inverse_mont, Tensor dropped_residue, Tensor rns_params, int "
      "half_drop_prime) -> Tensor");
  m.def(
      "rescale_drop_leading_prime_nearest_(Tensor(a!) remaining_residues, "
      "Tensor drop_prime_inverse_mont, Tensor dropped_residue, Tensor "
      "rns_params, int half_drop_prime) -> ()");
  m.def(
      "rescale_drop_leading_prime_truncate(Tensor remaining_residues, Tensor "
      "drop_prime_inverse_mont, Tensor dropped_residue, Tensor rns_params) -> "
      "Tensor");
  m.def(
      "rescale_drop_leading_prime_truncate_(Tensor(a!) remaining_residues, "
      "Tensor drop_prime_inverse_mont, Tensor dropped_residue, Tensor "
      "rns_params) -> ()");
  m.def(
      "apply_coefficient_galois_automorphism(Tensor residues, Tensor "
      "source_indices, Tensor source_sign, Tensor twice_modulus) -> Tensor");
  m.def(
      "apply_ntt_galois_automorphism(Tensor residues_ntt, Tensor "
      "source_indices) -> Tensor");
  m.def(
      "keyswitch_moddown_qp_to_q(Tensor q_residues, Tensor p_residues, Tensor "
      "moddown_p_drop_inverses_montgomery, Tensor rns_params, "
      "Tensor? moddown_half=None, str rounding=\"nearest\") -> Tensor");
  m.def(
      "keyswitch_accumulate_digit_products_(Tensor(a!) accumulator0_qp, "
      "Tensor(b!) accumulator1_qp, Tensor extended_digit_ntt_qp, Tensor "
      "key_switch_key_digit, Tensor rns_params, int key_digit_row_start, "
      "Tensor? source_indices=None) -> "
      "()");
  m.def(
      "keyswitch_accumulate_products_(Tensor(a!) accumulator0_qp, "
      "Tensor(b!) accumulator1_qp, Tensor[] extended_digits_ntt_qp, Tensor "
      "key_switch_key, Tensor rns_params, int key_digit_row_start, "
      "Tensor? source_indices=None) -> ()");
}

#if defined(FHELIUM_WITH_CPU)
TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CPU, m) {
  m.impl("multiply_two_component_ntt_montgomery",
         &fhelium::ckks::multiply_two_component_ntt_montgomery);
  m.impl("add_prepared_plaintext_component",
         &fhelium::ckks::add_prepared_plaintext_component);
  m.impl("add_prepared_plaintext_component_",
         &fhelium::ckks::add_prepared_plaintext_component_);
  m.impl("add_cyclic_compressed_plaintext_component",
         &fhelium::ckks::add_cyclic_compressed_plaintext_component);
  m.impl("add_cyclic_compressed_plaintext_component_",
         &fhelium::ckks::add_cyclic_compressed_plaintext_component_);
  m.impl("add_contiguous_compressed_plaintext_component",
         &fhelium::ckks::add_contiguous_compressed_plaintext_component);
  m.impl("add_contiguous_compressed_plaintext_component_",
         &fhelium::ckks::add_contiguous_compressed_plaintext_component_);
  m.impl("add_strided_plaintext_component",
         &fhelium::ckks::add_strided_plaintext_component);
  m.impl("add_strided_plaintext_component_",
         &fhelium::ckks::add_strided_plaintext_component_);
  m.impl("rescale_drop_leading_prime_nearest",
         &fhelium::ckks::rescale_drop_leading_prime_nearest);
  m.impl("rescale_drop_leading_prime_nearest_",
         &fhelium::ckks::rescale_drop_leading_prime_nearest_);
  m.impl("rescale_drop_leading_prime_truncate",
         &fhelium::ckks::rescale_drop_leading_prime_truncate);
  m.impl("rescale_drop_leading_prime_truncate_",
         &fhelium::ckks::rescale_drop_leading_prime_truncate_);
  m.impl("apply_coefficient_galois_automorphism",
         &fhelium::ckks::apply_coefficient_galois_automorphism);
  m.impl("apply_ntt_galois_automorphism",
         &fhelium::ckks::apply_ntt_galois_automorphism);
  m.impl("keyswitch_moddown_qp_to_q",
         &fhelium::ckks::keyswitch_moddown_qp_to_q);
  m.impl("keyswitch_accumulate_digit_products_",
         &fhelium::ckks::keyswitch_accumulate_digit_products_);
}
#endif

#if defined(FHELIUM_WITH_CUDA) || defined(FHELIUM_WITH_HIP)
TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CUDA, m) {
  m.impl("multiply_two_component_ntt_montgomery",
         &fhelium::ckks::multiply_two_component_ntt_montgomery);
  m.impl("add_prepared_plaintext_component",
         &fhelium::ckks::add_prepared_plaintext_component);
  m.impl("add_prepared_plaintext_component_",
         &fhelium::ckks::add_prepared_plaintext_component_);
  m.impl("add_cyclic_compressed_plaintext_component",
         &fhelium::ckks::add_cyclic_compressed_plaintext_component);
  m.impl("add_cyclic_compressed_plaintext_component_",
         &fhelium::ckks::add_cyclic_compressed_plaintext_component_);
  m.impl("add_contiguous_compressed_plaintext_component",
         &fhelium::ckks::add_contiguous_compressed_plaintext_component);
  m.impl("add_contiguous_compressed_plaintext_component_",
         &fhelium::ckks::add_contiguous_compressed_plaintext_component_);
  m.impl("add_strided_plaintext_component",
         &fhelium::ckks::add_strided_plaintext_component);
  m.impl("add_strided_plaintext_component_",
         &fhelium::ckks::add_strided_plaintext_component_);
  m.impl("rescale_drop_leading_prime_nearest",
         &fhelium::ckks::rescale_drop_leading_prime_nearest);
  m.impl("rescale_drop_leading_prime_nearest_",
         &fhelium::ckks::rescale_drop_leading_prime_nearest_);
  m.impl("rescale_drop_leading_prime_truncate",
         &fhelium::ckks::rescale_drop_leading_prime_truncate);
  m.impl("rescale_drop_leading_prime_truncate_",
         &fhelium::ckks::rescale_drop_leading_prime_truncate_);
  m.impl("apply_coefficient_galois_automorphism",
         &fhelium::ckks::apply_coefficient_galois_automorphism);
  m.impl("apply_ntt_galois_automorphism",
         &fhelium::ckks::apply_ntt_galois_automorphism);
  m.impl("keyswitch_moddown_qp_to_q",
         &fhelium::ckks::keyswitch_moddown_qp_to_q);
  m.impl("keyswitch_accumulate_digit_products_",
         &fhelium::ckks::keyswitch_accumulate_digit_products_);
  m.impl("keyswitch_accumulate_products_",
         &fhelium::ckks::keyswitch_accumulate_products_);
}
#endif
