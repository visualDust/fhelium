#pragma once

#include <torch/torch.h>

void forward_ntt_montgomery_indexed_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params);

void forward_ntt_to_montgomery_indexed_inplace_cuda(
    torch::Tensor standard_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params);

torch::Tensor forward_ntt_to_montgomery_indexed_cuda(
    const torch::Tensor standard_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params);

void forward_ntt_montgomery_compact_grouped_smem_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    int64_t grouped_stage_count,
    int64_t smem_stage_count);

void forward_ntt_to_montgomery_compact_grouped_smem_inplace_cuda(
    torch::Tensor standard_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    int64_t grouped_stage_count,
    int64_t smem_stage_count);

torch::Tensor forward_ntt_to_montgomery_compact_grouped_smem_cuda(
    const torch::Tensor standard_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    int64_t grouped_stage_count,
    int64_t smem_stage_count);

void forward_ntt_montgomery_compact_keyswitch_accumulate_inplace_cuda(
    torch::Tensor coefficient_digit_qp,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const torch::Tensor key_digit_qp,
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    int64_t key_row_start);

void forward_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);

void forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_cuda(
    torch::Tensor standard_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);

torch::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact_cuda(
    const torch::Tensor standard_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);
