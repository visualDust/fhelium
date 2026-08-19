#include <torch/library.h>
#include <torch/torch.h>

#include "../ntt_validation.h"
#include "inverse_ntt_cuda.h"

void inverse_ntt_montgomery_indexed_(torch::Tensor montgomery_residues,
                                     const torch::Tensor even_indices,
                                     const torch::Tensor odd_indices,
                                     const torch::Tensor inverse_twiddles,
                                     const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        inverse_twiddles,
                                        rns_params);
  inverse_ntt_montgomery_indexed_inplace_cuda(montgomery_residues,
                                              even_indices,
                                              odd_indices,
                                              inverse_twiddles,
                                              rns_params);
}

void inverse_ntt_to_standard_lazy_indexed_(torch::Tensor montgomery_residues,
                                           const torch::Tensor even_indices,
                                           const torch::Tensor odd_indices,
                                           const torch::Tensor inverse_twiddles,
                                           const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        inverse_twiddles,
                                        rns_params);
  inverse_ntt_to_standard_lazy_indexed_inplace_cuda(montgomery_residues,
                                                    even_indices,
                                                    odd_indices,
                                                    inverse_twiddles,
                                                    rns_params);
}

void inverse_ntt_to_standard_indexed_(torch::Tensor montgomery_residues,
                                      const torch::Tensor even_indices,
                                      const torch::Tensor odd_indices,
                                      const torch::Tensor inverse_twiddles,
                                      const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        inverse_twiddles,
                                        rns_params);
  inverse_ntt_to_standard_indexed_inplace_cuda(montgomery_residues,
                                               even_indices,
                                               odd_indices,
                                               inverse_twiddles,
                                               rns_params);
}

void inverse_ntt_to_centered_indexed_(torch::Tensor montgomery_residues,
                                      const torch::Tensor even_indices,
                                      const torch::Tensor odd_indices,
                                      const torch::Tensor inverse_twiddles,
                                      const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        inverse_twiddles,
                                        rns_params);
  inverse_ntt_to_centered_indexed_inplace_cuda(montgomery_residues,
                                               even_indices,
                                               odd_indices,
                                               inverse_twiddles,
                                               rns_params);
}

void inverse_ntt_montgomery_compact_grouped_smem_(
    torch::Tensor montgomery_residues,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, inverse_twiddles, rns_params, grouped_stage_count);
  inverse_ntt_montgomery_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_standard_lazy_compact_grouped_smem_(
    torch::Tensor montgomery_residues,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, inverse_twiddles, rns_params, grouped_stage_count);
  inverse_ntt_to_standard_lazy_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_standard_compact_grouped_smem_(
    torch::Tensor montgomery_residues,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, inverse_twiddles, rns_params, grouped_stage_count);
  inverse_ntt_to_standard_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_centered_compact_grouped_smem_(
    torch::Tensor montgomery_residues,
    const torch::Tensor inverse_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, inverse_twiddles, rns_params, grouped_stage_count);
  inverse_ntt_to_centered_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_montgomery_power_of_two_radix_compact_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  inverse_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_standard_lazy_power_of_two_radix_compact_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  inverse_ntt_to_standard_lazy_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_standard_power_of_two_radix_compact_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  inverse_ntt_to_standard_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_centered_power_of_two_radix_compact_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  inverse_ntt_to_centered_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_montgomery_power_of_two_radix_compact_diagnostic_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int64_t shared_memory_log_n) {
  fhelium::ntt::validate_shared_memory_log_n(shared_memory_log_n);
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  inverse_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      shared_memory_log_n);
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CUDA, m) {
  m.impl("inverse_ntt_montgomery_indexed_", &inverse_ntt_montgomery_indexed_);
  m.impl("inverse_ntt_to_standard_lazy_indexed_",
         &inverse_ntt_to_standard_lazy_indexed_);
  m.impl("inverse_ntt_to_standard_indexed_", &inverse_ntt_to_standard_indexed_);
  m.impl("inverse_ntt_to_centered_indexed_", &inverse_ntt_to_centered_indexed_);
  m.impl("inverse_ntt_montgomery_compact_grouped_smem_",
         &inverse_ntt_montgomery_compact_grouped_smem_);
  m.impl("inverse_ntt_to_standard_lazy_compact_grouped_smem_",
         &inverse_ntt_to_standard_lazy_compact_grouped_smem_);
  m.impl("inverse_ntt_to_standard_compact_grouped_smem_",
         &inverse_ntt_to_standard_compact_grouped_smem_);
  m.impl("inverse_ntt_to_centered_compact_grouped_smem_",
         &inverse_ntt_to_centered_compact_grouped_smem_);
  m.impl("inverse_ntt_montgomery_power_of_two_radix_compact_",
         &inverse_ntt_montgomery_power_of_two_radix_compact_);
  m.impl("inverse_ntt_to_standard_lazy_power_of_two_radix_compact_",
         &inverse_ntt_to_standard_lazy_power_of_two_radix_compact_);
  m.impl("inverse_ntt_to_standard_power_of_two_radix_compact_",
         &inverse_ntt_to_standard_power_of_two_radix_compact_);
  m.impl("inverse_ntt_to_centered_power_of_two_radix_compact_",
         &inverse_ntt_to_centered_power_of_two_radix_compact_);
}

TORCH_LIBRARY_IMPL(fhelium_ntt_diagnostic_ops, CUDA, m) {
  m.impl("inverse_ntt_montgomery_power_of_two_radix_compact_override_",
         &inverse_ntt_montgomery_power_of_two_radix_compact_diagnostic_);
}
