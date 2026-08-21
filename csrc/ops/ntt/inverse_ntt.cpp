#include <torch/library.h>

// Inverse negacyclic NTT requirements.
//
// Operands are integral [*batch, limb, ntt_index] tensors on one registered
// execution device with final
// extent N and lazy Montgomery residues in [0, 2q_i). Native code collapses
// only *batch; it does not broadcast or allocate. Twiddle row, rns_params
// column, and operand limb j all refer to the same prime_ids[j]. Indexed
// schedules are int32 [stage, N/2]; compact tables are [limb, N], and strict
// fixed-radix tables are [limb, N-1] plus [limb, radix]. All tables are on the
// operand device, residue tables are Montgomery, and inputs other than the
// operand are read-only.
//
// These kernels invert
// $A_i[k]=\sum_n a_i[n]\psi_i^{(2\operatorname{br}(k)+1)n}\bmod q_i$
// and normalize by $N^{-1}\bmod q_i$. Every schema mutates and preserves the
// operand storage, changing the final axis meaning to coefficient. The
// `montgomery` output remains Montgomery/lazy [0, 2q_i); `standard_lazy` is
// standard/lazy [0, 2q_i); `standard` is standard/canonical [0, q_i); and
// `centered` is standard in the centered representative interval.

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
      "inverse_ntt_montgomery_compact_grouped_smem_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> ()");
  m.def(
      "inverse_ntt_to_standard_lazy_compact_grouped_smem_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> ()");
  m.def(
      "inverse_ntt_to_standard_compact_grouped_smem_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> ()");
  m.def(
      "inverse_ntt_to_centered_compact_grouped_smem_(Tensor(a!) "
      "montgomery_residues, Tensor inverse_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> ()");
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
