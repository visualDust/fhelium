#include <torch/library.h>

// Forward negacyclic NTT requirements.
//
// Operand axes are [*batch, limb, coefficient] with final extent N. Native
// validation collapses only *batch to [batch, limb, N]; it does not broadcast
// or copy. All residue/twiddle/parameter tensors are integral, dense enough for
// that view, and on one registered execution device. Parameter column j and
// twiddle row j
// describe the prime_ids[j] represented by operand limb j. rns_params is
// [parameter, limb] with rows defined by rns_parameters.h. Indexed schedules
// are int32 [stage, N/2]; compact tables are [limb, N], while strict-radix
// tables are [limb, N-1] and [limb, radix]. Twiddles are Montgomery residues.
//
// For primitive 2N-th root $\psi_i$ and bit reversal $\operatorname{br}$,
// each limb computes
// $A_i[k]=\sum_{n=0}^{N-1}a_i[n]
// \psi_i^{(2\operatorname{br}(k)+1)n}\bmod q_i$.
// The `montgomery` variants consume coefficient/Montgomery residues; the
// `to_montgomery` variants consume coefficient/standard residues. All output
// is NTT/Montgomery in lazy range [0, 2q_i). A trailing underscore mutates and
// preserves operand storage. The non-underscore variant allocates output that
// does not alias an input. Tables and parameters are read-only.

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
      "forward_ntt_montgomery_compact_grouped_smem_(Tensor(a!) "
      "montgomery_residues, Tensor forward_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> ()");
  m.def(
      "forward_ntt_to_montgomery_compact_grouped_smem_(Tensor(a!) "
      "standard_residues, Tensor forward_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> ()");
  m.def(
      "forward_ntt_to_montgomery_compact_grouped_smem(Tensor "
      "standard_residues, Tensor forward_twiddles, Tensor rns_params, int "
      "grouped_stage_count) -> Tensor");
  m.def(
      "forward_ntt_montgomery_compact_keyswitch_accumulate_(Tensor(a!) "
      "coefficient_digit_qp, Tensor forward_twiddles, Tensor rns_params, "
      "Tensor key_digit_qp, Tensor(b!) accumulator0_qp, Tensor(c!) "
      "accumulator1_qp, int key_row_start) -> ()");
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
      "forward_ntt_montgomery_compact_grouped_smem_override_(Tensor(a!) "
      "montgomery_residues, Tensor forward_twiddles, Tensor rns_params, int "
      "grouped_stage_count, int shared_memory_log_n) -> ()");
  m.def(
      "forward_ntt_montgomery_power_of_two_radix_compact_override_(Tensor(a!) "
      "montgomery_residues, Tensor outer_twiddles, Tensor radix_root_powers, "
      "Tensor rns_params, int shared_memory_log_n) -> ()");
}
