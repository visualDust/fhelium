#include "backends.h"

#include "../common/backend_dispatch.h"

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::ntt::"

namespace fhelium::ntt {

void forward_ntt_montgomery_indexed_(at::Tensor& montgomery_residues,
                                     const at::Tensor& even_indices,
                                     const at::Tensor& odd_indices,
                                     const at::Tensor& forward_twiddles,
                                     const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(forward_ntt_montgomery_indexed_,
                           montgomery_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(forward_ntt_to_montgomery_indexed_,
                           standard_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(forward_ntt_to_montgomery_indexed,
                           standard_residues,
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
  FHELIUM_DISPATCH_GPU(forward_ntt_montgomery_compact_grouped_smem_,
                       montgomery_residues,
                       forward_twiddles,
                       rns_params,
                       grouped_stage_count);
}

void forward_ntt_to_montgomery_compact_grouped_smem_(
    at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_compact_grouped_smem_,
                       standard_residues,
                       forward_twiddles,
                       rns_params,
                       grouped_stage_count);
}

at::Tensor forward_ntt_to_montgomery_compact_grouped_smem(
    const at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_compact_grouped_smem,
                       standard_residues,
                       forward_twiddles,
                       rns_params,
                       grouped_stage_count);
}

void forward_ntt_to_montgomery_compact_add_scaled_(
    at::Tensor& source,
    const at::Tensor& addend,
    const at::Tensor& multiplier,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_compact_add_scaled_,
                       source,
                       addend,
                       multiplier,
                       forward_twiddles,
                       rns_params,
                       grouped_stage_count);
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
  FHELIUM_DISPATCH_GPU(forward_ntt_montgomery_compact_keyswitch_accumulate_,
                       coefficient_digit_qp,
                       forward_twiddles,
                       rns_params,
                       key_digit_qp,
                       accumulator0_qp,
                       accumulator1_qp,
                       key_row_start,
                       grouped_stage_count);
}

void forward_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(forward_ntt_montgomery_power_of_two_radix_compact_,
                       montgomery_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

void forward_ntt_to_montgomery_power_of_two_radix_compact_(
    at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_power_of_two_radix_compact_,
                       standard_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

at::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_power_of_two_radix_compact,
                       standard_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

void inverse_ntt_montgomery_indexed_(at::Tensor& montgomery_residues,
                                     const at::Tensor& even_indices,
                                     const at::Tensor& odd_indices,
                                     const at::Tensor& inverse_twiddles,
                                     const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(inverse_ntt_montgomery_indexed_,
                           montgomery_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(inverse_ntt_to_standard_lazy_indexed_,
                           montgomery_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(inverse_ntt_to_standard_indexed_,
                           montgomery_residues,
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
  FHELIUM_DISPATCH_CPU_GPU(inverse_ntt_to_centered_indexed_,
                           montgomery_residues,
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
  FHELIUM_DISPATCH_GPU(inverse_ntt_montgomery_compact_grouped_smem_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params,
                       grouped_stage_count);
}

void inverse_ntt_to_standard_lazy_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_standard_lazy_compact_grouped_smem_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params,
                       grouped_stage_count);
}

void inverse_ntt_to_standard_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_standard_compact_grouped_smem_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params,
                       grouped_stage_count);
}

void inverse_ntt_to_centered_compact_grouped_smem_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_centered_compact_grouped_smem_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params,
                       grouped_stage_count);
}

void inverse_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_montgomery_power_of_two_radix_compact_,
                       montgomery_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

void inverse_ntt_to_standard_lazy_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_standard_lazy_power_of_two_radix_compact_,
                       montgomery_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

void inverse_ntt_to_standard_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_standard_power_of_two_radix_compact_,
                       montgomery_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

void inverse_ntt_to_centered_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_centered_power_of_two_radix_compact_,
                       montgomery_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params);
}

void compact_ntt_stage_range_(at::Tensor& residues,
                              const at::Tensor& compact_twiddles,
                              const at::Tensor& rns_params,
                              bool inverse,
                              int64_t start_stage,
                              int64_t end_stage,
                              int64_t grouped_stage_count) {
  FHELIUM_DISPATCH_GPU(compact_ntt_stage_range_,
                       residues,
                       compact_twiddles,
                       rns_params,
                       inverse,
                       start_stage,
                       end_stage,
                       grouped_stage_count);
}

void fixed_radix_ntt_stage_range_(at::Tensor& residues,
                                  const at::Tensor& outer_twiddles,
                                  const at::Tensor& root_powers,
                                  const at::Tensor& rns_params,
                                  bool inverse,
                                  int64_t start_stage,
                                  int64_t end_stage) {
  FHELIUM_DISPATCH_GPU(fixed_radix_ntt_stage_range_,
                       residues,
                       outer_twiddles,
                       root_powers,
                       rns_params,
                       inverse,
                       start_stage,
                       end_stage);
}

}  // namespace fhelium::ntt

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::ntt::diagnostic::"

namespace fhelium::ntt::diagnostic {

void forward_ntt_montgomery_compact_grouped_smem_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count,
    int64_t shared_memory_log_n) {
  FHELIUM_DISPATCH_GPU(forward_ntt_montgomery_compact_grouped_smem_override_,
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
  FHELIUM_DISPATCH_GPU(
      forward_ntt_montgomery_power_of_two_radix_compact_override_,
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      shared_memory_log_n);
}

void inverse_ntt_montgomery_power_of_two_radix_compact_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    int64_t shared_memory_log_n) {
  FHELIUM_DISPATCH_GPU(
      inverse_ntt_montgomery_power_of_two_radix_compact_override_,
      montgomery_residues,
      outer_twiddles,
      radix_root_powers,
      rns_params,
      shared_memory_log_n);
}

}  // namespace fhelium::ntt::diagnostic
