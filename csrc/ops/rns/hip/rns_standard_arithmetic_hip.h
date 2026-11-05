#pragma once

#include <torch/torch.h>

torch::Tensor rns_add_standard_hip(const torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params);

void rns_add_standard_inplace_hip(torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params);

torch::Tensor rns_sub_standard_hip(const torch::Tensor lhs,
                                    const torch::Tensor rhs,
                                    const torch::Tensor rns_params);

void rns_sub_standard_inplace_hip(torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params);

torch::Tensor rns_montgomery_mul_row_scalars_standard_hip(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params);

torch::Tensor rns_sum_standard_batch_hip(const torch::Tensor source,
                                          const int64_t dim,
                                          const torch::Tensor rns_params);
