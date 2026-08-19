#include <torch/library.h>
#include <torch/torch.h>

#include "../ntt_validation.h"
#include "forward_ntt_cuda.h"

// CUDA indexed execution uses one launch per stage and provides the validation
// baseline for compact grouped CUDA policies. CPU indexed execution is the CPU
// production backend.
void forward_ntt_montgomery_indexed_(torch::Tensor montgomery_residues,
                                     const torch::Tensor even_indices,
                                     const torch::Tensor odd_indices,
                                     const torch::Tensor forward_twiddles,
                                     const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(montgomery_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  forward_ntt_montgomery_indexed_inplace_cuda(montgomery_residues,
                                              even_indices,
                                              odd_indices,
                                              forward_twiddles,
                                              rns_params);
}

void forward_ntt_to_montgomery_indexed_(torch::Tensor standard_residues,
                                        const torch::Tensor even_indices,
                                        const torch::Tensor odd_indices,
                                        const torch::Tensor forward_twiddles,
                                        const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  forward_ntt_to_montgomery_indexed_inplace_cuda(standard_residues,
                                                 even_indices,
                                                 odd_indices,
                                                 forward_twiddles,
                                                 rns_params);
}

torch::Tensor forward_ntt_to_montgomery_indexed(
    const torch::Tensor standard_residues,
    const torch::Tensor even_indices,
    const torch::Tensor odd_indices,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params) {
  fhelium::ntt::validate_indexed_tables(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params);
  return forward_ntt_to_montgomery_indexed_cuda(standard_residues,
                                                even_indices,
                                                odd_indices,
                                                forward_twiddles,
                                                rns_params);
}

void forward_ntt_montgomery_compact_grouped_smem_(
    torch::Tensor montgomery_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      montgomery_residues, forward_twiddles, rns_params, grouped_stage_count);
  forward_ntt_montgomery_compact_grouped_smem_inplace_cuda(
      montgomery_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_to_montgomery_compact_grouped_smem_(
    torch::Tensor standard_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      standard_residues, forward_twiddles, rns_params, grouped_stage_count);
  forward_ntt_to_montgomery_compact_grouped_smem_inplace_cuda(
      standard_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

torch::Tensor forward_ntt_to_montgomery_compact_grouped_smem(
    const torch::Tensor standard_residues,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      standard_residues, forward_twiddles, rns_params, grouped_stage_count);
  return forward_ntt_to_montgomery_compact_grouped_smem_cuda(
      standard_residues,
      forward_twiddles,
      rns_params,
      grouped_stage_count,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_montgomery_compact_keyswitch_accumulate_(
    torch::Tensor coefficient_digit_qp,
    const torch::Tensor forward_twiddles,
    const torch::Tensor rns_params,
    const torch::Tensor key_digit_qp,
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    const int64_t key_row_start) {
  // Specialized mutating fusion: coefficient_digit_qp changes in place from
  // coefficient/Montgomery to NTT/Montgomery [*batch, QP limb, N], then its
  // rowwise products with key_digit_qp[key component, QP limb, N] are added to
  // the two NTT/Montgomery QP accumulators. key_row_start maps local limb j to
  // key limb key_row_start+j. Key/tables/params are read-only; the three schema
  // write aliases are distinct and no implicit broadcast changes public batch.
  constexpr int64_t kVerifiedGroupedStageCount = 4;
  fhelium::ntt::validate_compact_tables(coefficient_digit_qp,
                                        forward_twiddles,
                                        rns_params,
                                        kVerifiedGroupedStageCount);
  forward_ntt_montgomery_compact_keyswitch_accumulate_inplace_cuda(
      coefficient_digit_qp,
      forward_twiddles,
      rns_params,
      key_digit_qp,
      accumulator0_qp,
      accumulator1_qp,
      key_row_start);
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
  forward_ntt_montgomery_compact_grouped_smem_inplace_cuda(montgomery_residues,
                                                           forward_twiddles,
                                                           rns_params,
                                                           grouped_stage_count,
                                                           shared_memory_log_n);
}

void forward_ntt_montgomery_power_of_two_radix_compact_(
    torch::Tensor montgomery_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      montgomery_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

void forward_ntt_to_montgomery_power_of_two_radix_compact_(
    torch::Tensor standard_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      standard_residues, outer_twiddles, radix_root_powers, rns_params);
  forward_ntt_to_montgomery_power_of_two_radix_compact_inplace_cuda(
      standard_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      fhelium::ntt::kNttDefaultSharedMemoryLogN);
}

torch::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact(
    const torch::Tensor standard_residues,
    const torch::Tensor outer_twiddles,
    const torch::Tensor radix_root_powers,
    const torch::Tensor rns_params) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      standard_residues, outer_twiddles, radix_root_powers, rns_params);
  return forward_ntt_to_montgomery_power_of_two_radix_compact_cuda(
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
  forward_ntt_montgomery_power_of_two_radix_compact_inplace_cuda(
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      plan.radix,
      shared_memory_log_n);
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CUDA, m) {
  m.impl("forward_ntt_montgomery_indexed_", &forward_ntt_montgomery_indexed_);
  m.impl("forward_ntt_to_montgomery_indexed_",
         &forward_ntt_to_montgomery_indexed_);
  m.impl("forward_ntt_to_montgomery_indexed",
         &forward_ntt_to_montgomery_indexed);
  m.impl("forward_ntt_montgomery_compact_grouped_smem_",
         &forward_ntt_montgomery_compact_grouped_smem_);
  m.impl("forward_ntt_to_montgomery_compact_grouped_smem_",
         &forward_ntt_to_montgomery_compact_grouped_smem_);
  m.impl("forward_ntt_to_montgomery_compact_grouped_smem",
         &forward_ntt_to_montgomery_compact_grouped_smem);
  m.impl("forward_ntt_montgomery_compact_keyswitch_accumulate_",
         &forward_ntt_montgomery_compact_keyswitch_accumulate_);
  m.impl("forward_ntt_montgomery_power_of_two_radix_compact_",
         &forward_ntt_montgomery_power_of_two_radix_compact_);
  m.impl("forward_ntt_to_montgomery_power_of_two_radix_compact_",
         &forward_ntt_to_montgomery_power_of_two_radix_compact_);
  m.impl("forward_ntt_to_montgomery_power_of_two_radix_compact",
         &forward_ntt_to_montgomery_power_of_two_radix_compact);
}

TORCH_LIBRARY_IMPL(fhelium_ntt_diagnostic_ops, CUDA, m) {
  m.impl("forward_ntt_montgomery_compact_grouped_smem_override_",
         &forward_ntt_montgomery_compact_grouped_smem_diagnostic_);
  m.impl("forward_ntt_montgomery_power_of_two_radix_compact_override_",
         &forward_ntt_montgomery_power_of_two_radix_compact_diagnostic_);
}
