#include <torch/torch.h>

#include "../backends.h"
#include "../ntt_validation.h"
#include "inverse_ntt_cuda.h"

namespace fhelium::ntt::gpu {

void inverse_ntt_montgomery_indexed_(at::Tensor& montgomery_residues,
                                     const at::Tensor& even_indices,
                                     const at::Tensor& odd_indices,
                                     const at::Tensor& inverse_twiddles,
                                     const at::Tensor& rns_params) {
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

void inverse_ntt_to_standard_lazy_indexed_(at::Tensor& montgomery_residues,
                                           const at::Tensor& even_indices,
                                           const at::Tensor& odd_indices,
                                           const at::Tensor& inverse_twiddles,
                                           const at::Tensor& rns_params) {
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

void inverse_ntt_to_standard_indexed_(at::Tensor& montgomery_residues,
                                      const at::Tensor& even_indices,
                                      const at::Tensor& odd_indices,
                                      const at::Tensor& inverse_twiddles,
                                      const at::Tensor& rns_params) {
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

void inverse_ntt_to_centered_indexed_(at::Tensor& montgomery_residues,
                                      const at::Tensor& even_indices,
                                      const at::Tensor& odd_indices,
                                      const at::Tensor& inverse_twiddles,
                                      const at::Tensor& rns_params) {
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
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(montgomery_residues,
                                        inverse_twiddles,
                                        rns_params,
                                        grouped_stage_count,
                                        true);
  inverse_ntt_montgomery_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_standard_lazy_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(montgomery_residues,
                                        inverse_twiddles,
                                        rns_params,
                                        grouped_stage_count,
                                        true);
  inverse_ntt_to_standard_lazy_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_standard_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(montgomery_residues,
                                        inverse_twiddles,
                                        rns_params,
                                        grouped_stage_count,
                                        true);
  inverse_ntt_to_standard_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_to_centered_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(montgomery_residues,
                                        inverse_twiddles,
                                        rns_params,
                                        grouped_stage_count,
                                        true);
  inverse_ntt_to_centered_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      inverse_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void inverse_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
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
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
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
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
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
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
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

}  // namespace fhelium::ntt::gpu

namespace fhelium::ntt::diagnostic::gpu {

void inverse_ntt_montgomery_power_of_two_radix_compact_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    int64_t shared_memory_log_n) {
  ::fhelium::ntt::gpu::
      inverse_ntt_montgomery_power_of_two_radix_compact_diagnostic_(
          montgomery_residues,
          outer_twiddles,
          radix_root_powers,
          rns_params,
          shared_memory_log_n);
}

}  // namespace fhelium::ntt::diagnostic::gpu
