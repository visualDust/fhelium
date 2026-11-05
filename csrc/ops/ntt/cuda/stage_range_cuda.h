#pragma once

#include <torch/torch.h>

void forward_ntt_compact_stage_range_cuda(
    torch::Tensor residues, const torch::Tensor twiddles,
    const torch::Tensor parameters, int64_t start_stage, int64_t end_stage,
    int64_t grouped_stage_count);
void inverse_ntt_compact_stage_range_cuda(
    torch::Tensor residues, const torch::Tensor twiddles,
    const torch::Tensor parameters, int64_t start_stage, int64_t end_stage,
    int64_t grouped_stage_count);

void forward_ntt_fixed_radix_stage_range_cuda(
    torch::Tensor residues, const torch::Tensor outer_twiddles,
    const torch::Tensor root_powers, const torch::Tensor parameters,
    int64_t start_stage, int64_t end_stage, int64_t radix);
void inverse_ntt_fixed_radix_stage_range_cuda(
    torch::Tensor residues, const torch::Tensor outer_twiddles,
    const torch::Tensor root_powers, const torch::Tensor parameters,
    int64_t start_stage, int64_t end_stage, int64_t radix);
