#include <torch/torch.h>

#include "../backends.h"
#include "../ntt_validation.h"
#include "forward_ntt_hip.h"

namespace fhelium::ntt::gpu {

// CUDA indexed execution uses one launch per stage and provides the validation
// baseline for compact grouped CUDA policies. CPU indexed execution is the CPU
// production backend.
void forward_ntt_montgomery_indexed_(at::Tensor& montgomery_residues,
                                     const at::Tensor& even_indices,
                                     const at::Tensor& odd_indices,
                                     const at::Tensor& forward_twiddles,
                                     const at::Tensor& rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  forward_ntt_montgomery_indexed_inplace_hip(montgomery_residues,
                                             even_indices,
                                             odd_indices,
                                             forward_twiddles,
                                             rns_params);
}

void forward_ntt_to_montgomery_indexed_(at::Tensor& standard_residues,
                                        const at::Tensor& even_indices,
                                        const at::Tensor& odd_indices,
                                        const at::Tensor& forward_twiddles,
                                        const at::Tensor& rns_params) {
  fhelium::ntt::validate_indexed_tables(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  forward_ntt_to_montgomery_indexed_inplace_hip(standard_residues,
                                                even_indices,
                                                odd_indices,
                                                forward_twiddles,
                                                rns_params);
}

at::Tensor forward_ntt_to_montgomery_indexed(
    const at::Tensor& standard_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params) {
  fhelium::ntt::validate_indexed_tables(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  return forward_ntt_to_montgomery_indexed_hip(standard_residues,
                                               even_indices,
                                               odd_indices,
                                               forward_twiddles,
                                               rns_params);
}

void forward_ntt_montgomery_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, forward_twiddles, rns_params, grouped_stage_count);
  forward_ntt_montgomery_compact_grouped_smem_inplace_hip(
      montgomery_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_to_montgomery_compact_grouped_smem_(
    at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      standard_residues, forward_twiddles, rns_params, grouped_stage_count);
  forward_ntt_to_montgomery_compact_grouped_smem_inplace_hip(
      standard_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

at::Tensor forward_ntt_to_montgomery_compact_grouped_smem(
    const at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      standard_residues, forward_twiddles, rns_params, grouped_stage_count);
  return forward_ntt_to_montgomery_compact_grouped_smem_hip(
      standard_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_montgomery_compact_keyswitch_accumulate_(
    at::Tensor& coefficient_digit_qp,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    const at::Tensor& key_digit_qp,
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    int64_t key_row_start,
    int64_t grouped_stage_count) {
  // Specialized mutating fusion: coefficient_digit_qp is disposable scratch
  // with coefficient/Montgomery input [*batch, QP limb, N]. Its NTT values
  // are consumed inside the final kernel and not written back to scratch.
  // Products with key_digit_qp[key component, QP limb, N] are added to
  // the two NTT/Montgomery QP accumulators. key_row_start maps local limb j to
  // key limb key_row_start+j. Key/tables/params are read-only; the three schema
  // write aliases are distinct and no implicit broadcast changes public batch.
  fhelium::ntt::validate_compact_tables(
      coefficient_digit_qp, forward_twiddles, rns_params, grouped_stage_count);
  forward_ntt_montgomery_compact_keyswitch_accumulate_inplace_hip(
      coefficient_digit_qp,
      forward_twiddles,
      rns_params,
      key_digit_qp,
      accumulator0_qp,
      accumulator1_qp,
      key_row_start,
      grouped_stage_count);
}

void forward_ntt_montgomery_compact_grouped_smem_diagnostic_(
    torch::Tensor montgomery_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count,
    const int64_t shared_memory_log_n) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, forward_twiddles, rns_params, grouped_stage_count);
  fhelium::ntt::validate_shared_memory_log_n(shared_memory_log_n);
  forward_ntt_montgomery_compact_grouped_smem_inplace_hip(montgomery_residues,
                                                          forward_twiddles,
                                                          rns_params,
                                                          grouped_stage_count,
                                                          shared_memory_log_n);
}

void forward_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_montgomery_power_of_two_radix_compact_inplace_hip(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_to_montgomery_power_of_two_radix_compact_(
    at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      standard_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_hip(
      standard_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

at::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      standard_residues, outer_twiddles, radix_root_powers, rns_params);
  return forward_ntt_to_montgomery_power_of_two_radix_compact_hip(
      standard_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_montgomery_power_of_two_radix_compact_diagnostic_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params,
    const int64_t shared_memory_log_n) {
  fhelium::ntt::validate_shared_memory_log_n(shared_memory_log_n);
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_montgomery_power_of_two_radix_compact_inplace_hip(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      shared_memory_log_n);
}

void forward_ntt_to_montgomery_compact_add_scaled_(
    at::Tensor& source,
    const at::Tensor& addend,
    const at::Tensor& multiplier,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  ::forward_ntt_to_montgomery_compact_add_scaled_inplace_hip(
      source,
      addend,
      multiplier,
      forward_twiddles,
      rns_params,
      grouped_stage_count);
}

}  // namespace fhelium::ntt::gpu

namespace fhelium::ntt::diagnostic::gpu {

void forward_ntt_montgomery_compact_grouped_smem_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count,
    int64_t shared_memory_log_n) {
  ::fhelium::ntt::gpu::forward_ntt_montgomery_compact_grouped_smem_diagnostic_(
      montgomery_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      shared_memory_log_n);
}

void forward_ntt_montgomery_power_of_two_radix_compact_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    int64_t shared_memory_log_n) {
  ::fhelium::ntt::gpu::
      forward_ntt_montgomery_power_of_two_radix_compact_diagnostic_(
          montgomery_residues,
          outer_twiddles,
          radix_root_powers,
          rns_params,
          shared_memory_log_n);
}

}  // namespace fhelium::ntt::diagnostic::gpu
