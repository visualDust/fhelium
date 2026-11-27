#include "backends.h"

#include "../common/backend_dispatch.h"
#include "../common/overlap.h"

#include <ATen/ops/empty_like.h>

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::ntt::"

namespace fhelium::ntt {

namespace {

void check_output(const at::Tensor& out,
                  const at::Tensor& like,
                  const char* operation) {
  TORCH_CHECK_VALUE(out.sizes() == like.sizes() &&
                        out.scalar_type() == like.scalar_type() &&
                        out.device() == like.device(),
                    operation,
                    " requires an output of shape ",
                    like.sizes(),
                    ", dtype ",
                    like.scalar_type(),
                    " and device ",
                    like.device());
}

}  // namespace

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

void forward_ntt_to_montgomery_indexed_out(const at::Tensor& standard_residues,
                                           const at::Tensor& even_indices,
                                           const at::Tensor& odd_indices,
                                           const at::Tensor& forward_twiddles,
                                           const at::Tensor& rns_params,
                                           at::Tensor& out) {
  check_output(out, standard_residues, "forward_ntt_to_montgomery_indexed_out");
  TORCH_CHECK_VALUE(fhelium::detail::storage_disjoint(out, standard_residues),
                    "forward_ntt_to_montgomery_indexed_out requires out "
                    "disjoint from its input");
  FHELIUM_DISPATCH_CPU_GPU(forward_ntt_to_montgomery_indexed_out,
                           standard_residues,
                           even_indices,
                           odd_indices,
                           forward_twiddles,
                           rns_params,
                           out);
}

at::Tensor forward_ntt_to_montgomery_indexed(
    const at::Tensor& standard_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(standard_residues);
  forward_ntt_to_montgomery_indexed_out(standard_residues,
                                        even_indices,
                                        odd_indices,
                                        forward_twiddles,
                                        rns_params,
                                        out);
  return out;
}

void forward_ntt_montgomery_compact_(at::Tensor& montgomery_residues,
                                     const at::Tensor& forward_twiddles,
                                     const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(forward_ntt_montgomery_compact_,
                       montgomery_residues,
                       forward_twiddles,
                       rns_params);
}

void forward_ntt_to_montgomery_compact_(at::Tensor& standard_residues,
                                        const at::Tensor& forward_twiddles,
                                        const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_compact_,
                       standard_residues,
                       forward_twiddles,
                       rns_params);
}

void forward_ntt_to_montgomery_compact_out(const at::Tensor& standard_residues,
                                           const at::Tensor& forward_twiddles,
                                           const at::Tensor& rns_params,
                                           at::Tensor& out) {
  check_output(out, standard_residues, "forward_ntt_to_montgomery_compact_out");
  TORCH_CHECK_VALUE(fhelium::detail::storage_disjoint(out, standard_residues),
                    "forward_ntt_to_montgomery_compact_out requires out "
                    "disjoint from its input");
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_compact_out,
                       standard_residues,
                       forward_twiddles,
                       rns_params,
                       out);
}

at::Tensor forward_ntt_to_montgomery_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(standard_residues);
  forward_ntt_to_montgomery_compact_out(
      standard_residues, forward_twiddles, rns_params, out);
  return out;
}

void forward_ntt_to_montgomery_compact_add_scaled_(
    at::Tensor& source,
    const at::Tensor& addend,
    const at::Tensor& multiplier,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_compact_add_scaled_,
                       source,
                       addend,
                       multiplier,
                       forward_twiddles,
                       rns_params);
}

void forward_ntt_montgomery_compact_keyswitch_accumulate_(
    at::Tensor& coefficient_digit_qp,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    const at::Tensor& key_digit_qp,
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    int64_t key_row_start,
    int64_t key_p_row_start) {
  FHELIUM_DISPATCH_GPU(forward_ntt_montgomery_compact_keyswitch_accumulate_,
                       coefficient_digit_qp,
                       forward_twiddles,
                       rns_params,
                       key_digit_qp,
                       accumulator0_qp,
                       accumulator1_qp,
                       key_row_start,
                       key_p_row_start);
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

void forward_ntt_to_montgomery_power_of_two_radix_compact_out(
    const at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    at::Tensor& out) {
  check_output(out,
               standard_residues,
               "forward_ntt_to_montgomery_power_of_two_radix_compact_out");
  TORCH_CHECK_VALUE(fhelium::detail::storage_disjoint(out, standard_residues),
                    "forward_ntt_to_montgomery_power_of_two_radix_compact_out "
                    "requires out disjoint from its input");
  FHELIUM_DISPATCH_GPU(forward_ntt_to_montgomery_power_of_two_radix_compact_out,
                       standard_residues,
                       outer_twiddles,
                       radix_root_powers,
                       rns_params,
                       out);
}

at::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(standard_residues);
  forward_ntt_to_montgomery_power_of_two_radix_compact_out(
      standard_residues, outer_twiddles, radix_root_powers, rns_params, out);
  return out;
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

void inverse_ntt_montgomery_compact_(at::Tensor& montgomery_residues,
                                     const at::Tensor& inverse_twiddles,
                                     const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_montgomery_compact_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params);
}

void inverse_ntt_to_standard_lazy_compact_(at::Tensor& montgomery_residues,
                                           const at::Tensor& inverse_twiddles,
                                           const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_standard_lazy_compact_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params);
}

void inverse_ntt_to_standard_compact_(at::Tensor& montgomery_residues,
                                      const at::Tensor& inverse_twiddles,
                                      const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_standard_compact_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params);
}

void inverse_ntt_to_centered_compact_(at::Tensor& montgomery_residues,
                                      const at::Tensor& inverse_twiddles,
                                      const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_GPU(inverse_ntt_to_centered_compact_,
                       montgomery_residues,
                       inverse_twiddles,
                       rns_params);
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
