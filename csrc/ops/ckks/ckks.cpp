#include <torch/library.h>

// CKKS-local native tensor requirements.
//
// Residue operands are integral [*batch, limb, coefficient_or_ntt_index]
// tensors on a registered execution device. Native views flatten only
// homogeneous *batch axes. There is no limb or polynomial-index broadcasting;
// the only batch broadcast is a validated unbatched plaintext. rns_params is
// [parameter, limb], and column j
// describes the exact prime_ids[j] of operand limb j. All tensors in one call
// share dtype/device. Functional output is newly allocated; a trailing
// underscore mutates and preserves only schema-annotated storage. SPMD
// communication and rank iteration remains in application Python code.
//
// Plaintext addition consumes coefficient/standard ciphertext residues and
// coefficient/Montgomery plaintext residues, computes
// $c'_{0,i}=c_{0,i}+p_i\bmod q_i$, and returns coefficient/standard canonical
// output. Rescale consumes standard coefficient rows split as remaining Q(P)
// plus the dropped leading Q residue and computes the selected rounded quotient
// modulo every remaining prime. Galois operations apply $\sigma_g:X\mapsto
// X^g$ by a read-only gather, preserving basis/representation/domain.
// Key-switch multiply-accumulate consumes NTT/Montgomery QP digits and keys and
// mutates two NTT/Montgomery QP accumulators. ModDown consumes separate
// canonical coefficient/standard Q and P rows in each row prime's [0, q_i) or
// [0, p_j) interval, sequentially divides and rounds by every P prime, and
// returns newly allocated canonical coefficient/standard Q rows; its inverse
// table is specifically `moddown_p_drop_inverses_montgomery`.

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "add_prepared_plaintext_component(Tensor ciphertext_component, Tensor "
      "prepared_plaintext, Tensor rns_params) -> Tensor");
  m.def(
      "add_prepared_plaintext_component_(Tensor(a!) ciphertext_component, "
      "Tensor prepared_plaintext, Tensor rns_params) -> ()");
  m.def(
      "add_cyclic_compressed_plaintext_component(Tensor "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> Tensor");
  m.def(
      "add_cyclic_compressed_plaintext_component_(Tensor(a!) "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> ()");
  m.def(
      "add_contiguous_compressed_plaintext_component(Tensor "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> Tensor");
  m.def(
      "add_contiguous_compressed_plaintext_component_(Tensor(a!) "
      "ciphertext_component, Tensor compressed_plaintext, Tensor rns_params) "
      "-> ()");
  m.def(
      "add_strided_plaintext_component(Tensor ciphertext_component, Tensor "
      "strided_plaintext, Tensor implicit_plaintext, Tensor rns_params) -> "
      "Tensor");
  m.def(
      "add_strided_plaintext_component_(Tensor(a!) ciphertext_component, "
      "Tensor strided_plaintext, Tensor implicit_plaintext, Tensor "
      "rns_params) -> ()");
  m.def(
      "rescale_drop_leading_prime_nearest(Tensor remaining_residues, Tensor "
      "drop_prime_inverse_mont, Tensor dropped_residue, Tensor rns_params, int "
      "half_drop_prime) -> Tensor");
  m.def(
      "rescale_drop_leading_prime_nearest_(Tensor(a!) remaining_residues, "
      "Tensor drop_prime_inverse_mont, Tensor dropped_residue, Tensor "
      "rns_params, int half_drop_prime) -> ()");
  m.def(
      "rescale_drop_leading_prime_truncate(Tensor remaining_residues, Tensor "
      "drop_prime_inverse_mont, Tensor dropped_residue, Tensor rns_params) -> "
      "Tensor");
  m.def(
      "rescale_drop_leading_prime_truncate_(Tensor(a!) remaining_residues, "
      "Tensor drop_prime_inverse_mont, Tensor dropped_residue, Tensor "
      "rns_params) -> ()");
  m.def(
      "apply_coefficient_galois_automorphism(Tensor residues, Tensor "
      "source_indices, Tensor source_sign, Tensor twice_modulus) -> Tensor");
  m.def(
      "apply_ntt_galois_automorphism(Tensor residues_ntt, Tensor "
      "source_indices) -> Tensor");
  m.def(
      "keyswitch_moddown_qp_to_q(Tensor q_residues, Tensor p_residues, Tensor "
      "moddown_p_drop_inverses_montgomery, Tensor rns_params) -> Tensor");
  m.def(
      "keyswitch_accumulate_digit_products_(Tensor(a!) accumulator0_qp, "
      "Tensor(b!) accumulator1_qp, Tensor extended_digit_ntt_qp, Tensor "
      "key_switch_key_digit, Tensor rns_params, int key_digit_row_start) -> "
      "()");
}
