#pragma once

#include <torch/torch.h>

void inverse_ntt_montgomery_indexed_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params);

void inverse_ntt_to_standard_lazy_indexed_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params);

void inverse_ntt_to_standard_indexed_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params);

void inverse_ntt_to_centered_indexed_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params);

void inverse_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);

void inverse_ntt_to_standard_lazy_power_of_two_radix_compact_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);

void inverse_ntt_to_standard_power_of_two_radix_compact_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);

void inverse_ntt_to_centered_power_of_two_radix_compact_inplace_cuda(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    int64_t radix,
    int64_t shared_memory_log_n);
