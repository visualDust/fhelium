#include <torch/library.h>

// Dense RNS arithmetic requirements.
//
// Residue operands are integral [*batch, limb, coefficient_or_ntt_index]
// tensors on a registered execution device. Native views collapse only
// *batch and never broadcast a non-singleton batch, coefficient, or limb axis.
// rns_params is integral
// [parameter, limb] on the same device; its column j describes the exact
// prime_ids[j] represented by operand limb j. Binary functional outputs have
// lhs shape/state and do not alias inputs. A trailing underscore mutates only
// its schema write argument, preserves storage, and treats tables/other
// operands as read-only.
//
// `montgomery_mul` computes $a_i b_i R^{-1}\bmod q_i$ in lazy [0, 2q_i).
// The cyclic/contiguous compressed variants apply the same product after
// expanding [*compressed_batch, limb, unique_index] to the lhs final extent;
// compact batch count must be one or equal to lhs after *batch is collapsed.
// `add_lazy` and `sub_lazy` compute modulo $2q_i$ from lazy inputs. Conversion
// by `to_montgomery_` maps standard $x_i$ to $x_iR\bmod q_i$;
// `from_montgomery_` applies REDC and returns standard lazy residues. These
// operations preserve coefficient versus NTT domain. `canonicalize_residues_`
// maps [0, 2q_i) to [0, q_i); `center_residues_` maps canonical values to the
// centered interval; `shift_residues_positive_` adds q_i to centered values.
// `lift_centered_coefficients` maps integral [*batch, coefficient] to
// [*batch, limb, coefficient]; the caller aligns twice_modulus[j] with the
// exact prime_ids[j] for limb j. It allocates non-aliasing standard lazy
// output.

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def("montgomery_mul(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_mul_cyclic_compressed(Tensor lhs, Tensor compressed_rhs, "
      "Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_mul_contiguous_compressed(Tensor lhs, Tensor "
      "compressed_rhs, Tensor rns_params) -> Tensor");
  m.def(
      "montgomery_mul_row_scalars_(Tensor(a!) residues, Tensor row_scalars, "
      "Tensor rns_params) -> ()");
  m.def(
      "to_montgomery_(Tensor(a!) standard_residues, Tensor rns_params) -> ()");
  m.def(
      "from_montgomery_(Tensor(a!) montgomery_residues, Tensor rns_params) -> "
      "()");
  m.def("add_lazy(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def(
      "add_lazy_with_twice_modulus(Tensor lhs, Tensor rhs, Tensor "
      "twice_modulus) -> Tensor");
  m.def("sub_lazy(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def(
      "canonicalize_residues_(Tensor(a!) lazy_residues, Tensor rns_params) -> "
      "()");
  m.def(
      "center_residues_(Tensor(a!) canonical_residues, Tensor rns_params) -> "
      "()");
  m.def(
      "shift_residues_positive_(Tensor(a!) centered_residues, Tensor "
      "rns_params) -> ()");
  m.def(
      "lift_centered_coefficients(Tensor centered_coefficients, Tensor "
      "twice_modulus) -> Tensor");
  m.def(
      "mixed_radix_decompose(Tensor source_residues, Tensor "
      "mixed_radix_normalizers, Tensor mixed_radix_propagation_coefficients, "
      "Tensor modulus_lo, Tensor modulus_hi, Tensor neg_inv_modulus_lo, Tensor "
      "neg_inv_modulus_hi) -> Tensor");
  m.def(
      "mixed_radix_basis_extend_to_montgomery(Tensor "
      "mixed_radix_components, Tensor basis_extension_coefficients, Tensor "
      "rns_params, int destination_row_count) -> Tensor");
}
