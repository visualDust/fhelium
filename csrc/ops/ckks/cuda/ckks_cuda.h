#pragma once

#include <torch/torch.h>

torch::Tensor ckks_add_prepared_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor prepared_plaintext,
    const torch::Tensor rns_params);

void ckks_add_prepared_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor prepared_plaintext,
    const torch::Tensor rns_params);

torch::Tensor ckks_add_cyclic_compressed_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params);

void ckks_add_cyclic_compressed_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params);

torch::Tensor ckks_add_contiguous_compressed_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params);

void ckks_add_contiguous_compressed_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor compressed_plaintext,
    const torch::Tensor rns_params);

torch::Tensor ckks_add_strided_plaintext_component_cuda(
    const torch::Tensor ciphertext_component,
    const torch::Tensor strided_plaintext,
    const torch::Tensor implicit_plaintext,
    const torch::Tensor rns_params);

void ckks_add_strided_plaintext_component_inplace_cuda(
    torch::Tensor ciphertext_component,
    const torch::Tensor strided_plaintext,
    const torch::Tensor implicit_plaintext,
    const torch::Tensor rns_params);

torch::Tensor ckks_rescale_drop_leading_prime_nearest_cuda(
    const torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params,
    int64_t half_drop_prime);

void ckks_rescale_drop_leading_prime_nearest_inplace_cuda(
    torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params,
    int64_t half_drop_prime);

torch::Tensor ckks_rescale_drop_leading_prime_truncate_cuda(
    const torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params);

void ckks_rescale_drop_leading_prime_truncate_inplace_cuda(
    torch::Tensor remaining_residues,
    const torch::Tensor drop_prime_inverse_mont,
    const torch::Tensor dropped_residue,
    const torch::Tensor rns_params);

torch::Tensor apply_coefficient_galois_automorphism_cuda(
    const torch::Tensor residues,
    const torch::Tensor source_indices,
    const torch::Tensor source_sign,
    const torch::Tensor twice_modulus);

torch::Tensor apply_ntt_galois_automorphism_cuda(
    const torch::Tensor residues_ntt, const torch::Tensor source_indices);

torch::Tensor keyswitch_moddown_qp_to_q_cuda(
    const torch::Tensor q_residues,
    const torch::Tensor p_residues,
    const torch::Tensor moddown_p_drop_inverses_montgomery,
    const torch::Tensor rns_params);

void keyswitch_accumulate_digit_products_inplace_cuda(
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    const torch::Tensor extended_digit_ntt_qp,
    const torch::Tensor key_switch_key_digit,
    const torch::Tensor rns_params,
    int64_t key_digit_row_start);
