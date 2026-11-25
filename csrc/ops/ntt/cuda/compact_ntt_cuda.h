#pragma once

#include <torch/torch.h>

// Compact radix-2 CUDA launchers. Operands use [*batch, prime, N] layouts
// with validated compact tables; see compact_ntt_phases.cuh for the schedule.

enum class CompactForwardInput { Montgomery, Standard };
enum class CompactInverseResult { Montgomery, StandardLazy, Standard, Centered };

void compact_forward_ntt_cuda(const torch::Tensor& source,
                              torch::Tensor destination,
                              const torch::Tensor& forward_twiddles,
                              const torch::Tensor& rns_params,
                              CompactForwardInput input);

void compact_inverse_ntt_cuda(torch::Tensor residues,
                              const torch::Tensor& inverse_twiddles,
                              const torch::Tensor& rns_params,
                              CompactInverseResult result);

void compact_forward_ntt_add_scaled_cuda(torch::Tensor source,
                                         const torch::Tensor& addend,
                                         const torch::Tensor& multiplier,
                                         const torch::Tensor& forward_twiddles,
                                         const torch::Tensor& rns_params);

void compact_forward_ntt_keyswitch_accumulate_cuda(
    torch::Tensor coefficient_digit_qp,
    const torch::Tensor& forward_twiddles,
    const torch::Tensor& rns_params,
    const torch::Tensor& key_digit_qp,
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    int64_t key_row_start,
    int64_t key_p_row_start);
