#pragma once

#include <torch/torch.h>

torch::Tensor rns_montgomery_mul_hip(const torch::Tensor lhs,
                                      const torch::Tensor rhs,
                                      const torch::Tensor rns_params);

torch::Tensor rns_montgomery_weighted_sum_hip(
    at::TensorList ciphertexts,
    at::TensorList plaintexts,
    const torch::Tensor rns_params);

torch::Tensor rns_montgomery_weighted_sums_hip(
    at::TensorList ciphertexts,
    at::TensorList plaintexts,
    int64_t group_count,
    const torch::Tensor rns_params);

torch::Tensor rns_montgomery_mul_cyclic_compressed_hip(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params);

torch::Tensor rns_montgomery_mul_contiguous_compressed_hip(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params);

void rns_montgomery_mul_row_scalars_inplace_hip(
    torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params);

void rns_to_montgomery_inplace_hip(torch::Tensor standard_residues,
                                    const torch::Tensor rns_params);

void rns_from_montgomery_inplace_hip(torch::Tensor montgomery_residues,
                                      const torch::Tensor rns_params);

void rns_reduce_to_standard_inplace_hip(torch::Tensor lazy_residues,
                                         const torch::Tensor rns_params);

torch::Tensor rns_add_lazy_hip(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params);

torch::Tensor rns_add_lazy_with_twice_modulus_hip(
    const torch::Tensor lhs,
    const torch::Tensor rhs,
    const torch::Tensor twice_modulus);

torch::Tensor rns_sub_lazy_hip(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params);

void rns_center_residues_inplace_hip(torch::Tensor standard_residues,
                                      const torch::Tensor rns_params);

void rns_shift_residues_positive_inplace_hip(torch::Tensor centered_residues,
                                              const torch::Tensor rns_params);

torch::Tensor rns_lift_centered_coefficients_hip(
    const torch::Tensor centered_coefficients,
    const torch::Tensor twice_modulus);
