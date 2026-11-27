#include <ATen/MemoryOverlap.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include "../common/overlap.h"
#include "backends.h"

#include "../common/backend_dispatch.h"

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::ckks::"

namespace fhelium::ckks {

namespace {

// An output Tensor matches the shape, dtype and device of `like`.
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

namespace {

// A plaintext-addition output has the shape, dtype and device of the
// ciphertext component, is that component or disjoint from it, and is
// disjoint from the plaintext operands.
void check_plaintext_out(const at::Tensor& out,
                         const at::Tensor& ciphertext_component,
                         std::initializer_list<at::Tensor> plaintexts,
                         const char* operation) {
  check_output(out, ciphertext_component, operation);
  bool disjoint = detail::same_elements(out, ciphertext_component) ||
                  detail::storage_disjoint(out, ciphertext_component);
  for (const auto& plaintext : plaintexts) {
    disjoint = disjoint && detail::storage_disjoint(out, plaintext);
  }
  TORCH_CHECK_VALUE(disjoint,
                    operation,
                    " requires an output that is the ciphertext component or "
                    "disjoint from it and disjoint from the plaintext");
}

}  // namespace

void multiply_two_component_ntt_montgomery_out(const at::Tensor& lhs_components,
                                               const at::Tensor& rhs_components,
                                               const at::Tensor& rns_params,
                                               at::Tensor& out) {
  auto sizes = lhs_components.sizes().vec();
  TORCH_CHECK_VALUE(!sizes.empty(),
                    "multiply_two_component_ntt_montgomery_out requires "
                    "component-major operands");
  sizes[0] = 3;
  TORCH_CHECK_VALUE(out.sizes() == at::IntArrayRef(sizes) &&
                        out.scalar_type() == lhs_components.scalar_type() &&
                        out.device() == lhs_components.device() &&
                        detail::storage_disjoint(out, lhs_components) &&
                        detail::storage_disjoint(out, rhs_components),
                    "multiply_two_component_ntt_montgomery_out requires an "
                    "output of shape ",
                    at::IntArrayRef(sizes),
                    " with the operand dtype and device, disjoint from both "
                    "operands");
  FHELIUM_DISPATCH_CPU_GPU(multiply_two_component_ntt_montgomery_out,
                           lhs_components,
                           rhs_components,
                           rns_params,
                           out);
}

at::Tensor multiply_two_component_ntt_montgomery(
    const at::Tensor& lhs_components,
    const at::Tensor& rhs_components,
    const at::Tensor& rns_params) {
  auto sizes = lhs_components.sizes().vec();
  TORCH_CHECK_VALUE(!sizes.empty(),
                    "multiply_two_component_ntt_montgomery requires "
                    "component-major operands");
  sizes[0] = 3;
  auto out = at::empty(sizes, lhs_components.options());
  multiply_two_component_ntt_montgomery_out(
      lhs_components, rhs_components, rns_params, out);
  return out;
}

void add_prepared_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& prepared_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out) {
  check_plaintext_out(out,
                      ciphertext_component,
                      {prepared_plaintext},
                      "add_prepared_plaintext_component_out");
  FHELIUM_DISPATCH_CPU_GPU(add_prepared_plaintext_component_out,
                           ciphertext_component,
                           prepared_plaintext,
                           rns_params,
                           out);
}

at::Tensor add_prepared_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& prepared_plaintext,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(ciphertext_component);
  add_prepared_plaintext_component_out(
      ciphertext_component, prepared_plaintext, rns_params, out);
  return out;
}

void add_prepared_plaintext_component_(at::Tensor& ciphertext_component,
                                       const at::Tensor& prepared_plaintext,
                                       const at::Tensor& rns_params) {
  add_prepared_plaintext_component_out(ciphertext_component,
                                       prepared_plaintext,
                                       rns_params,
                                       ciphertext_component);
}

void add_cyclic_compressed_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out) {
  check_plaintext_out(out,
                      ciphertext_component,
                      {compressed_plaintext},
                      "add_cyclic_compressed_plaintext_component_out");
  FHELIUM_DISPATCH_CPU_GPU(add_cyclic_compressed_plaintext_component_out,
                           ciphertext_component,
                           compressed_plaintext,
                           rns_params,
                           out);
}

at::Tensor add_cyclic_compressed_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(ciphertext_component);
  add_cyclic_compressed_plaintext_component_out(
      ciphertext_component, compressed_plaintext, rns_params, out);
  return out;
}

void add_cyclic_compressed_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  add_cyclic_compressed_plaintext_component_out(ciphertext_component,
                                                compressed_plaintext,
                                                rns_params,
                                                ciphertext_component);
}

void add_contiguous_compressed_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out) {
  check_plaintext_out(out,
                      ciphertext_component,
                      {compressed_plaintext},
                      "add_contiguous_compressed_plaintext_component_out");
  FHELIUM_DISPATCH_CPU_GPU(add_contiguous_compressed_plaintext_component_out,
                           ciphertext_component,
                           compressed_plaintext,
                           rns_params,
                           out);
}

at::Tensor add_contiguous_compressed_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(ciphertext_component);
  add_contiguous_compressed_plaintext_component_out(
      ciphertext_component, compressed_plaintext, rns_params, out);
  return out;
}

void add_contiguous_compressed_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params) {
  add_contiguous_compressed_plaintext_component_out(ciphertext_component,
                                                    compressed_plaintext,
                                                    rns_params,
                                                    ciphertext_component);
}

void add_strided_plaintext_component_out(const at::Tensor& ciphertext_component,
                                         const at::Tensor& strided_plaintext,
                                         const at::Tensor& implicit_plaintext,
                                         const at::Tensor& rns_params,
                                         at::Tensor& out) {
  check_plaintext_out(out,
                      ciphertext_component,
                      {strided_plaintext, implicit_plaintext},
                      "add_strided_plaintext_component_out");
  FHELIUM_DISPATCH_CPU_GPU(add_strided_plaintext_component_out,
                           ciphertext_component,
                           strided_plaintext,
                           implicit_plaintext,
                           rns_params,
                           out);
}

at::Tensor add_strided_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& strided_plaintext,
    const at::Tensor& implicit_plaintext,
    const at::Tensor& rns_params) {
  auto out = at::empty_like(ciphertext_component);
  add_strided_plaintext_component_out(ciphertext_component,
                                      strided_plaintext,
                                      implicit_plaintext,
                                      rns_params,
                                      out);
  return out;
}

void add_strided_plaintext_component_(at::Tensor& ciphertext_component,
                                      const at::Tensor& strided_plaintext,
                                      const at::Tensor& implicit_plaintext,
                                      const at::Tensor& rns_params) {
  add_strided_plaintext_component_out(ciphertext_component,
                                      strided_plaintext,
                                      implicit_plaintext,
                                      rns_params,
                                      ciphertext_component);
}

void apply_coefficient_galois_automorphism_out(const at::Tensor& residues,
                                               const at::Tensor& source_indices,
                                               const at::Tensor& source_sign,
                                               const at::Tensor& twice_modulus,
                                               at::Tensor& out) {
  check_output(out, residues, "apply_coefficient_galois_automorphism_out");
  TORCH_CHECK_VALUE(fhelium::detail::storage_disjoint(out, residues),
                    "apply_coefficient_galois_automorphism_out requires out "
                    "disjoint from its inputs");
  FHELIUM_DISPATCH_CPU_GPU(apply_coefficient_galois_automorphism_out,
                           residues,
                           source_indices,
                           source_sign,
                           twice_modulus,
                           out);
}

at::Tensor apply_coefficient_galois_automorphism(
    const at::Tensor& residues,
    const at::Tensor& source_indices,
    const at::Tensor& source_sign,
    const at::Tensor& twice_modulus) {
  auto out = at::empty_like(residues);
  apply_coefficient_galois_automorphism_out(
      residues, source_indices, source_sign, twice_modulus, out);
  return out;
}

void apply_ntt_galois_automorphism_out(const at::Tensor& residues_ntt,
                                       const at::Tensor& source_indices,
                                       at::Tensor& out) {
  check_output(out, residues_ntt, "apply_ntt_galois_automorphism_out");
  TORCH_CHECK_VALUE(fhelium::detail::storage_disjoint(out, residues_ntt),
                    "apply_ntt_galois_automorphism_out requires out disjoint "
                    "from its inputs");
  FHELIUM_DISPATCH_CPU_GPU(
      apply_ntt_galois_automorphism_out, residues_ntt, source_indices, out);
}

at::Tensor apply_ntt_galois_automorphism(const at::Tensor& residues_ntt,
                                         const at::Tensor& source_indices) {
  auto out = at::empty_like(residues_ntt);
  apply_ntt_galois_automorphism_out(residues_ntt, source_indices, out);
  return out;
}

void add_ntt_galois_automorphism_(at::Tensor& target,
                                  const at::Tensor& residues_ntt,
                                  const at::Tensor& source_indices,
                                  const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(add_ntt_galois_automorphism_,
                           target,
                           residues_ntt,
                           source_indices,
                           rns_params);
}

void keyswitch_moddown_qp_to_q_out(
    const at::Tensor& q_residues,
    const at::Tensor& p_residues,
    const at::Tensor& moddown_p_drop_inverses_montgomery,
    const at::Tensor& rns_params,
    const std::optional<at::Tensor>& moddown_half,
    c10::string_view rounding,
    at::Tensor& out,
    const std::optional<at::Tensor>& p_scratch) {
  check_output(out, q_residues, "keyswitch_moddown_qp_to_q_out");
  if (p_scratch.has_value()) {
    check_output(*p_scratch, p_residues, "keyswitch_moddown_qp_to_q_out");
    TORCH_CHECK_VALUE(
        fhelium::detail::storage_disjoint(*p_scratch, out) &&
            fhelium::detail::storage_disjoint(*p_scratch, q_residues),
        "keyswitch_moddown_qp_to_q_out requires p_scratch disjoint from out "
        "and q_residues");
  }
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, q_residues),
      "keyswitch_moddown_qp_to_q_out requires out disjoint from its inputs");
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, p_residues),
      "keyswitch_moddown_qp_to_q_out requires out disjoint from its inputs");
  FHELIUM_DISPATCH_CPU_GPU(keyswitch_moddown_qp_to_q_out,
                           q_residues,
                           p_residues,
                           moddown_p_drop_inverses_montgomery,
                           rns_params,
                           moddown_half,
                           rounding,
                           out,
                           p_scratch);
}

at::Tensor keyswitch_moddown_qp_to_q(
    const at::Tensor& q_residues,
    const at::Tensor& p_residues,
    const at::Tensor& moddown_p_drop_inverses_montgomery,
    const at::Tensor& rns_params,
    const std::optional<at::Tensor>& moddown_half,
    c10::string_view rounding) {
  auto out = at::empty_like(q_residues);
  keyswitch_moddown_qp_to_q_out(q_residues,
                                p_residues,
                                moddown_p_drop_inverses_montgomery,
                                rns_params,
                                moddown_half,
                                rounding,
                                out);
  return out;
}

void keyswitch_accumulate_digit_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    const at::Tensor& extended_digit_ntt_qp,
    const at::Tensor& key_switch_key_digit,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    int64_t key_p_row_start,
    const std::optional<at::Tensor>& source_indices,
    const std::optional<at::Tensor>& key_seed,
    int64_t key_rounds) {
  FHELIUM_DISPATCH_CPU_GPU(keyswitch_accumulate_digit_products_,
                           accumulator0_qp,
                           accumulator1_qp,
                           extended_digit_ntt_qp,
                           key_switch_key_digit,
                           rns_params,
                           key_digit_row_start,
                           key_p_row_start,
                           source_indices,
                           key_seed,
                           key_rounds);
}

void keyswitch_accumulate_products_(at::TensorList accumulators0_qp,
                                    at::TensorList accumulators1_qp,
                                    at::TensorList extended_digits_ntt_qp,
                                    at::TensorList key_switch_keys,
                                    const at::Tensor& rns_params,
                                    int64_t key_digit_row_start,
                                    int64_t key_p_row_start,
                                    at::TensorList source_indices,
                                    at::TensorList key_seeds,
                                    int64_t key_rounds) {
  FHELIUM_DISPATCH_GPU(keyswitch_accumulate_products_,
                       accumulators0_qp,
                       accumulators1_qp,
                       extended_digits_ntt_qp,
                       key_switch_keys,
                       rns_params,
                       key_digit_row_start,
                       key_p_row_start,
                       source_indices,
                       key_seeds,
                       key_rounds);
}

}  // namespace fhelium::ckks
