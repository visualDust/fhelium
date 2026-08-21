#include <torch/library.h>

// Canonical arithmetic consumes integral [*batch, limb, index] operands on a
// registered execution device in [0, q_i) or documented lazy-compatible
// inputs. rns_params has shape
// [parameter, limb], and column j describes the prime_ids[j] of limb j.
// Add/sub compute $(a_i +/- b_i)\bmod q_i$ and return [0, q_i). Montgomery
// row-scalar multiply computes $a_i b_iR^{-1}\bmod q_i$ and canonicalizes.
// Functional results have lhs shape/domain/representation, alias no input, and
// do not mutate tables. Trailing-underscore forms preserve and mutate lhs
// storage only. Batch broadcasting is limited to the validated
// singleton RHS case; limb and final-index axes never broadcast. CPU kernels
// use PyTorch intra-op parallelism behind the same schemas.

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def("add_canonical(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def("add_canonical_(Tensor(a!) lhs, Tensor rhs, Tensor rns_params) -> ()");
  m.def("sub_canonical(Tensor lhs, Tensor rhs, Tensor rns_params) -> Tensor");
  m.def("sub_canonical_(Tensor(a!) lhs, Tensor rhs, Tensor rns_params) -> ()");
  m.def(
      "montgomery_mul_row_scalars_canonical(Tensor residues, Tensor "
      "row_scalars, Tensor rns_params) -> Tensor");
}
