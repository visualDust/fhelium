#pragma once

#include <torch/torch.h>

torch::Tensor rns_montgomery_mul_cuda(const torch::Tensor lhs,
                                      const torch::Tensor rhs,
                                      const torch::Tensor rns_params);

torch::Tensor rns_montgomery_mul_cyclic_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params);

torch::Tensor rns_montgomery_mul_contiguous_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params);

void rns_montgomery_mul_row_scalars_inplace_cuda(
    torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params);

void rns_to_montgomery_inplace_cuda(torch::Tensor standard_residues,
                                    const torch::Tensor rns_params);

void rns_from_montgomery_inplace_cuda(torch::Tensor montgomery_residues,
                                      const torch::Tensor rns_params);

void rns_canonicalize_residues_inplace_cuda(torch::Tensor lazy_residues,
                                            const torch::Tensor rns_params);

torch::Tensor rns_add_lazy_cuda(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params);

torch::Tensor rns_add_lazy_with_twice_modulus_cuda(
    const torch::Tensor lhs,
    const torch::Tensor rhs,
    const torch::Tensor twice_modulus);

torch::Tensor rns_sub_lazy_cuda(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params);

void rns_center_residues_inplace_cuda(torch::Tensor canonical_residues,
                                      const torch::Tensor rns_params);

void rns_shift_residues_positive_inplace_cuda(torch::Tensor centered_residues,
                                              const torch::Tensor rns_params);

torch::Tensor rns_lift_centered_coefficients_cuda(
    const torch::Tensor centered_coefficients,
    const torch::Tensor twice_modulus);
