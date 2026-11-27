#include <torch/library.h>

#include <fhelium/ckks_parameters.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt_tables.h>
#include <fhelium/rns_tables.h>

#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// Torch operator schemas for the table builders and slot names of the whole
// operations. A builder receives one RNS basis as its parameter table, the
// moduli of its columns, and log2 of the ring dimension; it returns the names
// and Tensors of the fields it defines, in slot order. These functions run on
// the host and place their results on the device of the parameter table, so
// one kernel serves every dispatch key.

namespace {

std::tuple<std::vector<int64_t>, std::vector<int64_t>> plan_key_layout_op(
    c10::IntArrayRef moduli, c10::IntArrayRef q_prime_ids) {
  auto layout = fhelium::ckks::plan_key_layout(moduli, q_prime_ids);
  std::vector<int64_t> sizes;
  for (const auto& digit : layout.digit_prime_ids) {
    sizes.push_back(static_cast<int64_t>(digit.size()));
  }
  return {std::move(layout.p_prime_ids), std::move(sizes)};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "prepare_tables(Tensor parameters, int[] moduli, int log_n, str "
      "ntt_backend) -> (Tensor[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
}

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def(
      "joint_rescale_table_set(Tensor parameters, int[] moduli, int log_n, "
      "int[] q_prime_ids, int[] p_prime_ids, int[] drop_prime_ids, int[] "
      "target_prime_ids, str moddown_rounding, str rescale_rounding, str "
      "basis_conversion) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "rescale_table_set(Tensor parameters, int[] moduli, int log_n, int[] "
      "source_prime_ids, int[] target_prime_ids, str rounding, bool "
      "ntt_input) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "basis_extension_table_set(Tensor source_parameters, int[] "
      "source_moduli, Tensor target_parameters, int[] target_moduli, int "
      "log_n, int[] source_prime_ids, int[] target_prime_ids) -> (str[], "
      "Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "crt_table_set(Tensor parameters, int[] moduli, int log_n, int[] "
      "source_prime_ids, int[] target_prime_ids, bool normalized) -> (str[], "
      "Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "centered_quotient_table_set(Tensor parameters, int[] moduli, int "
      "log_n, int[] source_prime_ids, int[] target_prime_ids, int[] "
      "divisor_prime_ids) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
}

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "key_switch_table_set(Tensor parameters, int[] moduli, int log_n, int[] "
      "q_prime_ids, int[] p_prime_ids, int[] digits, str basis_conversion, "
      "str moddown_rounding, bool rescale, int[] drop_prime_ids, int[] "
      "target_prime_ids, str rescale_rounding) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "rotation_sum_table_set(Tensor parameters, int[] moduli, int log_n, "
      "int[] q_prime_ids, int[] p_prime_ids, int[] next_prime_ids, str "
      "rounding) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "encapsulation_table_set(Tensor parameters, int[] moduli, int log_n, "
      "str ntt_backend, float scale, int[] source_q_moduli, int[] "
      "low_p_moduli, int return_prime, int[] return_p_moduli, int[] "
      "q_moduli, int[] target_prime_ids, str rounding) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "embedding_table_set(int log_n, int generator, Device device) -> "
      "(str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "periodic_table_set(Tensor parameters, int[] moduli, int log_n, int "
      "period, int generator) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "reconstruction_table_set(Tensor parameters, int[] moduli, int log_n, "
      "int[] prime_ids) -> (str[], Tensor[])",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "galois_table_set(int log_n, int galois_element, Device device) -> "
      "Tensor[]",
      {at::Tag::cudagraph_unsafe});
  m.def("plan_key_layout(int[] moduli, int[] q_prime_ids) -> (int[], int[])",
        {at::Tag::cudagraph_unsafe});
  m.def("table_slot_names(str family, int digit_count) -> str[]",
        {at::Tag::cudagraph_unsafe});
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CompositeImplicitAutograd, m) {
  m.impl("prepare_tables", &fhelium::ntt::prepare_tables);
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CompositeImplicitAutograd, m) {
  m.impl("joint_rescale_table_set", &fhelium::rns::joint_rescale_table_set);
  m.impl("rescale_table_set", &fhelium::rns::rescale_table_set);
  m.impl("basis_extension_table_set", &fhelium::rns::basis_extension_table_set);
  m.impl("crt_table_set", &fhelium::rns::crt_table_set);
  m.impl("centered_quotient_table_set",
         &fhelium::rns::centered_quotient_table_set);
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeImplicitAutograd, m) {
  m.impl("key_switch_table_set", &fhelium::ckks::key_switch_table_set);
  m.impl("rotation_sum_table_set", &fhelium::ckks::rotation_sum_table_set);
  m.impl("encapsulation_table_set", &fhelium::ckks::encapsulation_table_set);
  m.impl("reconstruction_table_set", &fhelium::ckks::reconstruction_table_set);
  m.impl("embedding_table_set", &fhelium::ckks::embedding_table_set);
  m.impl("periodic_table_set", &fhelium::ckks::periodic_table_set);
  m.impl("galois_table_set", &fhelium::ckks::galois_table_set);
  m.impl("plan_key_layout", &plan_key_layout_op);
  m.impl("table_slot_names", &fhelium::ckks::table_slot_names);
}
