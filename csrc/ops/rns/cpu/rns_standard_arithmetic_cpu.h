#pragma once

#include <torch/torch.h>

torch::Tensor rns_add_standard_cpu(const torch::Tensor lhs,
                                   const torch::Tensor rhs,
                                   const torch::Tensor rns_params);

torch::Tensor rns_sum_standard_batch_cpu(const torch::Tensor source,
                                         const int64_t dim,
                                         const torch::Tensor rns_params);
