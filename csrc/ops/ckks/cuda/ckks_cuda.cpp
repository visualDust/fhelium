#include <torch/library.h>
#include <torch/torch.h>

#include "ckks_cuda.h"

torch::Tensor add_prepared_plaintext_component(
    const torch::Tensor ciphertext_component,
    const torch::Tensor prepared_plaintext,
    const torch::Tensor rns_params) {
  return ckks_add_prepared_plaintext_component_cuda(
      ciphertext_component, prepared_plaintext, rns_params);
}

void add_prepared_plaintext_component_(torch::Tensor ciphertext_component,
                                       const torch::Tensor prepared_plaintext,
                                       const torch::Tensor rns_params) {
  ckks_add_prepared_plaintext_component_inplace_cuda(
      ciphertext_component, prepared_plaintext, rns_params);
}

torch::Tensor add_cyclic_compressed_plaintext_component(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  return ckks_add_cyclic_compressed_plaintext_component_cuda(
      ciphertext_component, compressed_plaintext, rns_params);
}

void add_cyclic_compressed_plaintext_component_(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  ckks_add_cyclic_compressed_plaintext_component_inplace_cuda(
      ciphertext_component, compressed_plaintext, rns_params);
}

torch::Tensor add_contiguous_compressed_plaintext_component(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  return ckks_add_contiguous_compressed_plaintext_component_cuda(
      ciphertext_component, compressed_plaintext, rns_params);
}

void add_contiguous_compressed_plaintext_component_(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params) {
  ckks_add_contiguous_compressed_plaintext_component_inplace_cuda(
      ciphertext_component, compressed_plaintext, rns_params);
}

torch::Tensor add_strided_plaintext_component(
    const torch::Tensor ciphertext_component,
    const torch::Tensor strided_plaintext,
    const torch::Tensor implicit_plaintext,
    const torch::Tensor rns_params) {
  return ckks_add_strided_plaintext_component_cuda(
      ciphertext_component, strided_plaintext, implicit_plaintext, rns_params);
}

void add_strided_plaintext_component_(torch::Tensor ciphertext_component,
                                      const torch::Tensor strided_plaintext,
                                      const torch::Tensor implicit_plaintext,
                                      const torch::Tensor rns_params) {
  ckks_add_strided_plaintext_component_inplace_cuda(
      ciphertext_component, strided_plaintext, implicit_plaintext, rns_params);
}

torch::Tensor rescale_drop_leading_prime_nearest(
    const torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params,
    const int64_t half_drop_prime) {
  return ckks_rescale_drop_leading_prime_nearest_cuda(remaining_residues,
                                                      drop_prime_inverse_mont,
                                                      dropped_residue,
                                                      rns_params,
                                                      half_drop_prime);
}

void rescale_drop_leading_prime_nearest_(
    torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params,
    const int64_t half_drop_prime) {
  ckks_rescale_drop_leading_prime_nearest_inplace_cuda(remaining_residues,
                                                       drop_prime_inverse_mont,
                                                       dropped_residue,
                                                       rns_params,
                                                       half_drop_prime);
}

torch::Tensor rescale_drop_leading_prime_truncate(
    const torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params) {
  return ckks_rescale_drop_leading_prime_truncate_cuda(
      remaining_residues, drop_prime_inverse_mont, dropped_residue, rns_params);
}

void rescale_drop_leading_prime_truncate_(
    torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params) {
  ckks_rescale_drop_leading_prime_truncate_inplace_cuda(
      remaining_residues, drop_prime_inverse_mont, dropped_residue, rns_params);
}

torch::Tensor apply_coefficient_galois_automorphism(
    const torch::Tensor residues,
    const torch::Tensor source_indices,
    const torch::Tensor source_sign,
    const torch::Tensor twice_modulus) {
  return apply_coefficient_galois_automorphism_cuda(
      residues, source_indices, source_sign, twice_modulus);
}

torch::Tensor apply_ntt_galois_automorphism(
    const torch::Tensor residues_ntt, const torch::Tensor source_indices) {
  return apply_ntt_galois_automorphism_cuda(residues_ntt, source_indices);
}

torch::Tensor keyswitch_moddown_qp_to_q(
    const torch::Tensor q_residues,
    const torch::Tensor p_residues,
    const torch::Tensor moddown_p_drop_inverses_montgomery,
    const torch::Tensor rns_params) {
  return keyswitch_moddown_qp_to_q_cuda(
      q_residues, p_residues, moddown_p_drop_inverses_montgomery, rns_params);
}

void keyswitch_accumulate_digit_products_(
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    const torch::Tensor extended_digit_ntt_qp,
    const torch::Tensor key_switch_key_digit,
    const torch::Tensor rns_params,
    const int64_t key_digit_row_start) {
  keyswitch_accumulate_digit_products_inplace_cuda(accumulator0_qp,
                                                   accumulator1_qp,
                                                   extended_digit_ntt_qp,
                                                   key_switch_key_digit,
                                                   rns_params,
                                                   key_digit_row_start);
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CUDA, m) {
  m.impl("add_prepared_plaintext_component", &add_prepared_plaintext_component);
  m.impl("add_prepared_plaintext_component_",
         &add_prepared_plaintext_component_);
  m.impl("add_cyclic_compressed_plaintext_component",
         &add_cyclic_compressed_plaintext_component);
  m.impl("add_cyclic_compressed_plaintext_component_",
         &add_cyclic_compressed_plaintext_component_);
  m.impl("add_contiguous_compressed_plaintext_component",
         &add_contiguous_compressed_plaintext_component);
  m.impl("add_contiguous_compressed_plaintext_component_",
         &add_contiguous_compressed_plaintext_component_);
  m.impl("add_strided_plaintext_component", &add_strided_plaintext_component);
  m.impl("add_strided_plaintext_component_", &add_strided_plaintext_component_);
  m.impl("rescale_drop_leading_prime_nearest",
         &rescale_drop_leading_prime_nearest);
  m.impl("rescale_drop_leading_prime_nearest_",
         &rescale_drop_leading_prime_nearest_);
  m.impl("rescale_drop_leading_prime_truncate",
         &rescale_drop_leading_prime_truncate);
  m.impl("rescale_drop_leading_prime_truncate_",
         &rescale_drop_leading_prime_truncate_);
  m.impl("apply_coefficient_galois_automorphism",
         &apply_coefficient_galois_automorphism);
  m.impl("apply_ntt_galois_automorphism", &apply_ntt_galois_automorphism);
  m.impl("keyswitch_moddown_qp_to_q", &keyswitch_moddown_qp_to_q);
  m.impl("keyswitch_accumulate_digit_products_",
         &keyswitch_accumulate_digit_products_);
}
