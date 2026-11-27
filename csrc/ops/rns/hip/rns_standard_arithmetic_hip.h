#pragma once

#include <torch/torch.h>

void rns_montgomery_mul_row_scalars_standard_out_hip(
    const torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params,
    torch::Tensor out);

void rns_add_standard_out_hip(const torch::Tensor lhs,
                              const torch::Tensor rhs,
                              const torch::Tensor rns_params,
                              torch::Tensor out);

void rns_sub_standard_out_hip(const torch::Tensor lhs,
                              const torch::Tensor rhs,
                              const torch::Tensor rns_params,
                              torch::Tensor out);

void rns_negate_standard_out_hip(const torch::Tensor source,
                                 const torch::Tensor rns_params,
                                 torch::Tensor out);

void rns_sum_standard_batch_out_hip(const torch::Tensor source,
                                    const int64_t dim,
                                    const torch::Tensor rns_params,
                                    torch::Tensor out);
