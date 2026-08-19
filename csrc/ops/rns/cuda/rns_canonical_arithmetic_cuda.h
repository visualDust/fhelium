#pragma once

#include <torch/torch.h>

torch::Tensor rns_add_canonical_cuda(const torch::Tensor lhs,
                                     const torch::Tensor rhs,
                                     const torch::Tensor rns_params);

void rns_add_canonical_inplace_cuda(torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params);

torch::Tensor rns_sub_canonical_cuda(const torch::Tensor lhs,
                                     const torch::Tensor rhs,
                                     const torch::Tensor rns_params);

void rns_sub_canonical_inplace_cuda(torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params);

torch::Tensor rns_montgomery_mul_row_scalars_canonical_cuda(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params);
