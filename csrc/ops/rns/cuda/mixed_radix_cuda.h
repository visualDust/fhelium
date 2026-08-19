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
