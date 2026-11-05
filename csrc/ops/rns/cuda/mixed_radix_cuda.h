#pragma once

#include <torch/torch.h>

torch::Tensor mixed_radix_decompose_cuda(
    const torch::Tensor source_residues,
    const torch::Tensor mixed_radix_normalizers,
    const torch::Tensor mixed_radix_propagation_coefficients,
    const torch::Tensor modulus_lo,
    const torch::Tensor modulus_hi,
    const torch::Tensor neg_inv_modulus_lo,
    const torch::Tensor neg_inv_modulus_hi);

torch::Tensor mixed_radix_basis_extend_to_montgomery_cuda(
    const torch::Tensor mixed_radix_components,
    const torch::Tensor basis_extension_coefficients,
    const torch::Tensor rns_params,
    int64_t destination_row_count);

void mixed_radix_decompose_out_cuda(
    const torch::Tensor source_residues,
    const torch::Tensor mixed_radix_normalizers,
    const torch::Tensor mixed_radix_propagation_coefficients,
    const torch::Tensor modulus_lo,
    const torch::Tensor modulus_hi,
    const torch::Tensor neg_inv_modulus_lo,
    const torch::Tensor neg_inv_modulus_hi,
    torch::Tensor out);

void mixed_radix_basis_extend_to_montgomery_out_cuda(
    const torch::Tensor mixed_radix_components,
    const torch::Tensor basis_extension_coefficients,
    const torch::Tensor rns_params,
    const int64_t destination_row_count,
    torch::Tensor out);

// Fused exact Garner decomposition and Montgomery basis extension.
torch::Tensor mixed_radix_modup_tiled_cuda(
    const torch::Tensor source_residues, at::TensorList decomposition_tables,
    const torch::Tensor extension, const torch::Tensor parameters);

void mixed_radix_modup_tiled_out_cuda(
    const torch::Tensor source_residues, at::TensorList decomposition_tables,
    const torch::Tensor extension, const torch::Tensor parameters,
    torch::Tensor out);

torch::Tensor crt_basis_extend_cuda(
    torch::Tensor source_residues, torch::Tensor source_parameters,
    torch::Tensor target_parameters, torch::Tensor matrix,
    torch::Tensor target_reciprocals,
    std::optional<torch::Tensor> source_normalizers,
    std::optional<torch::Tensor> source_reciprocals);

void crt_basis_extend_out_cuda(
    torch::Tensor source, torch::Tensor source_parameters,
    torch::Tensor target_parameters, torch::Tensor matrix,
    torch::Tensor target_reciprocals,
    std::optional<torch::Tensor> source_normalizers,
    std::optional<torch::Tensor> source_reciprocals, torch::Tensor output);
