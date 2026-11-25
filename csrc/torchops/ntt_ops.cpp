#include <torch/library.h>

#include <fhelium/ntt.h>
#include <fhelium/ntt_diagnostic.h>

// Torch operator schemas and backend registrations for the public functions
// of the same names. Each registered kernel is the public function itself.

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "forward_ntt_montgomery_indexed_(Tensor(a!) montgomery_residues, Tensor "
      "even_indices, Tensor odd_indices, Tensor forward_twiddles, Tensor "
      "rns_params) -> ()");
  m.def(
      "forward_ntt_to_montgomery_indexed_(Tensor(a!) standard_residues, Tensor "
      "even_indices, Tensor odd_indices, Tensor forward_twiddles, Tensor "
      "rns_params) -> ()");
  m.def(
      "forward_ntt_to_montgomery_indexed(Tensor standard_residues, Tensor "
      "even_indices, Tensor odd_indices, Tensor forward_twiddles, Tensor "
      "rns_params) -> Tensor");
  m.def(
      "forward_ntt_montgomery_compact_(Tensor(a!) "
      "montgomery_residues, Tensor forward_twiddles, Tensor rns_params) -> ()");
  m.def(
      "forward_ntt_to_montgomery_compact_(Tensor(a!) "
      "standard_residues, Tensor forward_twiddles, Tensor rns_params) -> ()");
  m.def(
      "forward_ntt_to_montgomery_compact(Tensor "
      "standard_residues, Tensor forward_twiddles, Tensor rns_params) -> Tensor");
  m.def(
      "forward_ntt_to_montgomery_compact_add_scaled_(Tensor(a!) source, "
      "Tensor addend, Tensor multiplier, Tensor forward_twiddles, "
      "Tensor rns_params) -> ()");
  m.def(
      "forward_ntt_montgomery_compact_keyswitch_accumulate_(Tensor(a!) "
      "coefficient_digit_qp, Tensor forward_twiddles, Tensor rns_params, "
      "Tensor key_digit_qp, Tensor(b!) accumulator0_qp, Tensor(c!) "
      "accumulator1_qp, int key_row_start, int key_p_row_start=-1) -> ()");
  m.def(
      "forward_ntt_montgomery_power_of_two_radix_compact_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> ()");
  m.def(
      "forward_ntt_to_montgomery_power_of_two_radix_compact_(Tensor(a!) "
      "standard_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> ()");
  m.def(
      "forward_ntt_to_montgomery_power_of_two_radix_compact(Tensor "
      "standard_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> Tensor");
}

// Keep tuning overrides out of the production operator ABI. These operators
// exist only for correctness differentials and cross-GPU profiling.
TORCH_LIBRARY_FRAGMENT(fhelium_ntt_diagnostic_ops, m) {
  m.def(
      "forward_ntt_montgomery_power_of_two_radix_compact_override_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params, int shared_memory_log_n) -> ()");
}

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "inverse_ntt_montgomery_indexed_(Tensor(a!) montgomery_residues, Tensor "
      "even_indices, Tensor odd_indices, Tensor inverse_twiddles, Tensor "
      "rns_params) -> ()");
  m.def(
      "inverse_ntt_to_standard_lazy_indexed_(Tensor(a!) montgomery_residues, "
      "Tensor even_indices, Tensor odd_indices, Tensor inverse_twiddles, "
      "Tensor "
      "rns_params) -> ()");
  m.def(
      "inverse_ntt_to_standard_indexed_(Tensor(a!) montgomery_residues, Tensor "
      "even_indices, Tensor odd_indices, Tensor inverse_twiddles, Tensor "
      "rns_params) -> ()");
  m.def(
      "inverse_ntt_to_centered_indexed_(Tensor(a!) montgomery_residues, Tensor "
      "even_indices, Tensor odd_indices, Tensor inverse_twiddles, Tensor "
      "rns_params) -> ()");
  m.def(
      "inverse_ntt_montgomery_compact_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_to_standard_lazy_compact_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_to_standard_compact_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_to_centered_compact_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_montgomery_power_of_two_radix_compact_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_to_standard_lazy_power_of_two_radix_compact_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_to_standard_power_of_two_radix_compact_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> ()");
  m.def(
      "inverse_ntt_to_centered_power_of_two_radix_compact_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params) -> ()");
}

// Keep tuning overrides out of the production operator ABI. These operators
// exist only for correctness differentials and cross-GPU profiling.
TORCH_LIBRARY_FRAGMENT(fhelium_ntt_diagnostic_ops, m) {
  m.def(
      "inverse_ntt_montgomery_power_of_two_radix_compact_override_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params, int shared_memory_log_n) -> ()");
}

// Unnormalized compact radix-2 stages on Montgomery residues. The half-open
// interval indexes forward DIF or inverse DIT stages; kernel launches between
// stage groups provide the required global synchronization. Representation
// conversion and inverse normalization belong to the surrounding schedule.
TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "fixed_radix_ntt_stage_range_(Tensor(a!) residues, Tensor "
      "outer_twiddles, "
      "Tensor root_powers, Tensor rns_params, bool inverse, int start_stage, "
      "int end_stage) -> ()");
}

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "validate_indexed_schedule(Tensor even_indices, Tensor odd_indices, "
      "int ring_dimension) -> ()");
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CompositeExplicitAutograd, m) {
  m.impl("validate_indexed_schedule", &fhelium::ntt::validate_indexed_schedule);
}

#if defined(FHELIUM_WITH_CPU)
TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CPU, m) {
  m.impl("forward_ntt_montgomery_indexed_",
         &fhelium::ntt::forward_ntt_montgomery_indexed_);
  m.impl("forward_ntt_to_montgomery_indexed_",
         &fhelium::ntt::forward_ntt_to_montgomery_indexed_);
  m.impl("forward_ntt_to_montgomery_indexed",
         &fhelium::ntt::forward_ntt_to_montgomery_indexed);
  m.impl("inverse_ntt_montgomery_indexed_",
         &fhelium::ntt::inverse_ntt_montgomery_indexed_);
  m.impl("inverse_ntt_to_standard_lazy_indexed_",
         &fhelium::ntt::inverse_ntt_to_standard_lazy_indexed_);
  m.impl("inverse_ntt_to_standard_indexed_",
         &fhelium::ntt::inverse_ntt_to_standard_indexed_);
  m.impl("inverse_ntt_to_centered_indexed_",
         &fhelium::ntt::inverse_ntt_to_centered_indexed_);
}
#endif

#if defined(FHELIUM_WITH_CUDA) || defined(FHELIUM_WITH_HIP)
TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CUDA, m) {
  m.impl("forward_ntt_montgomery_indexed_",
         &fhelium::ntt::forward_ntt_montgomery_indexed_);
  m.impl("forward_ntt_to_montgomery_indexed_",
         &fhelium::ntt::forward_ntt_to_montgomery_indexed_);
  m.impl("forward_ntt_to_montgomery_indexed",
         &fhelium::ntt::forward_ntt_to_montgomery_indexed);
  m.impl("forward_ntt_montgomery_compact_",
         &fhelium::ntt::forward_ntt_montgomery_compact_);
  m.impl("forward_ntt_to_montgomery_compact_",
         &fhelium::ntt::forward_ntt_to_montgomery_compact_);
  m.impl("forward_ntt_to_montgomery_compact",
         &fhelium::ntt::forward_ntt_to_montgomery_compact);
  m.impl("forward_ntt_to_montgomery_compact_add_scaled_",
         &fhelium::ntt::forward_ntt_to_montgomery_compact_add_scaled_);
  m.impl("forward_ntt_montgomery_compact_keyswitch_accumulate_",
         &fhelium::ntt::forward_ntt_montgomery_compact_keyswitch_accumulate_);
  m.impl("forward_ntt_montgomery_power_of_two_radix_compact_",
         &fhelium::ntt::forward_ntt_montgomery_power_of_two_radix_compact_);
  m.impl("forward_ntt_to_montgomery_power_of_two_radix_compact_",
         &fhelium::ntt::forward_ntt_to_montgomery_power_of_two_radix_compact_);
  m.impl("forward_ntt_to_montgomery_power_of_two_radix_compact",
         &fhelium::ntt::forward_ntt_to_montgomery_power_of_two_radix_compact);
  m.impl("inverse_ntt_montgomery_indexed_",
         &fhelium::ntt::inverse_ntt_montgomery_indexed_);
  m.impl("inverse_ntt_to_standard_lazy_indexed_",
         &fhelium::ntt::inverse_ntt_to_standard_lazy_indexed_);
  m.impl("inverse_ntt_to_standard_indexed_",
         &fhelium::ntt::inverse_ntt_to_standard_indexed_);
  m.impl("inverse_ntt_to_centered_indexed_",
         &fhelium::ntt::inverse_ntt_to_centered_indexed_);
  m.impl("inverse_ntt_montgomery_compact_",
         &fhelium::ntt::inverse_ntt_montgomery_compact_);
  m.impl("inverse_ntt_to_standard_lazy_compact_",
         &fhelium::ntt::inverse_ntt_to_standard_lazy_compact_);
  m.impl("inverse_ntt_to_standard_compact_",
         &fhelium::ntt::inverse_ntt_to_standard_compact_);
  m.impl("inverse_ntt_to_centered_compact_",
         &fhelium::ntt::inverse_ntt_to_centered_compact_);
  m.impl("inverse_ntt_montgomery_power_of_two_radix_compact_",
         &fhelium::ntt::inverse_ntt_montgomery_power_of_two_radix_compact_);
  m.impl(
      "inverse_ntt_to_standard_lazy_power_of_two_radix_compact_",
      &fhelium::ntt::inverse_ntt_to_standard_lazy_power_of_two_radix_compact_);
  m.impl("inverse_ntt_to_standard_power_of_two_radix_compact_",
         &fhelium::ntt::inverse_ntt_to_standard_power_of_two_radix_compact_);
  m.impl("inverse_ntt_to_centered_power_of_two_radix_compact_",
         &fhelium::ntt::inverse_ntt_to_centered_power_of_two_radix_compact_);
  m.impl("fixed_radix_ntt_stage_range_",
         &fhelium::ntt::fixed_radix_ntt_stage_range_);
}
#endif

#if defined(FHELIUM_WITH_CUDA) || defined(FHELIUM_WITH_HIP)
TORCH_LIBRARY_IMPL(fhelium_ntt_diagnostic_ops, CUDA, m) {
  m.impl("forward_ntt_montgomery_power_of_two_radix_compact_override_",
         &fhelium::ntt::diagnostic::
             forward_ntt_montgomery_power_of_two_radix_compact_override_);
  m.impl("inverse_ntt_montgomery_power_of_two_radix_compact_override_",
         &fhelium::ntt::diagnostic::
             inverse_ntt_montgomery_power_of_two_radix_compact_override_);
}
#endif
