#include "backends.h"

#include "../common/backend_dispatch.h"

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::ckks::"

namespace fhelium::ckks {

at::Tensor multiply_two_component_ntt_montgomery(
    const at::Tensor& lhs_components,
    const at::Tensor& rhs_components,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(multiply_two_component_ntt_montgomery,
                           lhs_components,
                           rhs_components,
                           rns_params);
}

at::Tensor add_prepared_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& prepared_plaintext,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_prepared_plaintext_component,
                           ciphertext_component,
                           prepared_plaintext,
                           rns_params);
}

void add_prepared_plaintext_component_(at::Tensor& ciphertext_component,
                                       const at::Tensor& prepared_plaintext,
                                       const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_prepared_plaintext_component_,
                           ciphertext_component,
                           prepared_plaintext,
                           rns_params);
}

at::Tensor add_cyclic_compressed_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_cyclic_compressed_plaintext_component,
                           ciphertext_component,
                           compressed_plaintext,
                           rns_params);
}

void add_cyclic_compressed_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_cyclic_compressed_plaintext_component_,
                           ciphertext_component,
                           compressed_plaintext,
                           rns_params);
}

at::Tensor add_contiguous_compressed_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_contiguous_compressed_plaintext_component,
                           ciphertext_component,
                           compressed_plaintext,
                           rns_params);
}

void add_contiguous_compressed_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_contiguous_compressed_plaintext_component_,
                           ciphertext_component,
                           compressed_plaintext,
                           rns_params);
}

at::Tensor add_strided_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& strided_plaintext,
    const at::Tensor& implicit_plaintext,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_strided_plaintext_component,
                           ciphertext_component,
                           strided_plaintext,
                           implicit_plaintext,
                           rns_params);
}

void add_strided_plaintext_component_(at::Tensor& ciphertext_component,
                                      const at::Tensor& strided_plaintext,
                                      const at::Tensor& implicit_plaintext,
                                      const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_strided_plaintext_component_,
                           ciphertext_component,
                           strided_plaintext,
                           implicit_plaintext,
                           rns_params);
}

at::Tensor rescale_drop_leading_prime_nearest(
    const at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params,
    int64_t half_drop_prime) {
  FHELIUM_DISPATCH_CPU_GPU(rescale_drop_leading_prime_nearest,
                           remaining_residues,
                           drop_prime_inverse_mont,
                           dropped_residue,
                           rns_params,
                           half_drop_prime);
}

void rescale_drop_leading_prime_nearest_(
    at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params,
    int64_t half_drop_prime) {
  FHELIUM_DISPATCH_CPU_GPU(rescale_drop_leading_prime_nearest_,
                           remaining_residues,
                           drop_prime_inverse_mont,
                           dropped_residue,
                           rns_params,
                           half_drop_prime);
}

at::Tensor rescale_drop_leading_prime_truncate(
    const at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(rescale_drop_leading_prime_truncate,
                           remaining_residues,
                           drop_prime_inverse_mont,
                           dropped_residue,
                           rns_params);
}

void rescale_drop_leading_prime_truncate_(
    at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(rescale_drop_leading_prime_truncate_,
                           remaining_residues,
                           drop_prime_inverse_mont,
                           dropped_residue,
                           rns_params);
}

at::Tensor apply_coefficient_galois_automorphism(
    const at::Tensor& residues,
    const at::Tensor& source_indices,
    const at::Tensor& source_sign,
    const at::Tensor& twice_modulus) {
  FHELIUM_DISPATCH_CPU_GPU(apply_coefficient_galois_automorphism,
                           residues,
                           source_indices,
                           source_sign,
                           twice_modulus);
}

at::Tensor apply_ntt_galois_automorphism(const at::Tensor& residues_ntt,
                                         const at::Tensor& source_indices) {
  FHELIUM_DISPATCH_CPU_GPU(
      apply_ntt_galois_automorphism, residues_ntt, source_indices);
}

at::Tensor keyswitch_moddown_qp_to_q(
    const at::Tensor& q_residues,
    const at::Tensor& p_residues,
    const at::Tensor& moddown_p_drop_inverses_montgomery,
    const at::Tensor& rns_params,
    const std::optional<at::Tensor>& moddown_half,
    c10::string_view rounding) {
  FHELIUM_DISPATCH_CPU_GPU(keyswitch_moddown_qp_to_q,
                           q_residues,
                           p_residues,
                           moddown_p_drop_inverses_montgomery,
                           rns_params,
                           moddown_half,
                           rounding);
}

void keyswitch_accumulate_digit_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    const at::Tensor& extended_digit_ntt_qp,
    const at::Tensor& key_switch_key_digit,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    const std::optional<at::Tensor>& source_indices) {
  FHELIUM_DISPATCH_CPU_GPU(keyswitch_accumulate_digit_products_,
                           accumulator0_qp,
                           accumulator1_qp,
                           extended_digit_ntt_qp,
                           key_switch_key_digit,
                           rns_params,
                           key_digit_row_start,
                           source_indices);
}

void keyswitch_accumulate_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    at::TensorList extended_digits_ntt_qp,
    const at::Tensor& key_switch_key,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    const std::optional<at::Tensor>& source_indices) {
  FHELIUM_DISPATCH_GPU(keyswitch_accumulate_products_,
                       accumulator0_qp,
                       accumulator1_qp,
                       extended_digits_ntt_qp,
                       key_switch_key,
                       rns_params,
                       key_digit_row_start,
                       source_indices);
}

}  // namespace fhelium::ckks
