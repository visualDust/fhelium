#include <torch/library.h>

#include <fhelium/ckks_keys.h>
#include <fhelium/operation_tables.h>

#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// Torch operator schemas for CKKS key generation. Each adapter unflattens the
// positional `tables` slots into the table struct of the function and the
// KeySwitchKeyLayout from its flattened fields, then calls the public function
// of the same name. Key generation composes registered operators, so it is
// registered as CompositeImplicitAutograd kernels.

namespace {

using fhelium::RingTables;
using fhelium::TableSlots;
using fhelium::unflatten_tables;
using fhelium::ckks::KeySwitchKeyLayout;

#define FHELIUM_KEY_LAYOUT_SCHEMA                                  \
  "Tensor seed, Tensor cdt_table, str ntt_backend, int[] q_rows, " \
  "int[] digit_sizes, int[] p_rows, Tensor? uniform_by_digit, "    \
  "Tensor? error_by_digit"

#define FHELIUM_KEY_LAYOUT_PARAMETERS                        \
  const at::Tensor &seed, const at::Tensor &cdt_table,       \
      std::string_view ntt_backend, c10::IntArrayRef q_rows, \
      c10::IntArrayRef digit_sizes, c10::IntArrayRef p_rows, \
      const std::optional<at::Tensor>&uniform_by_digit,      \
      const std::optional<at::Tensor>&error_by_digit

KeySwitchKeyLayout layout(std::string_view ntt_backend,
                          c10::IntArrayRef q_rows,
                          c10::IntArrayRef digit_sizes,
                          c10::IntArrayRef p_rows) {
  KeySwitchKeyLayout result;
  result.ntt_backend = std::string(ntt_backend);
  result.q_rows = q_rows.vec();
  result.digit_sizes = digit_sizes.vec();
  result.p_rows = p_rows.vec();
  return result;
}

at::Tensor create_secret_key_op(int64_t ring_dimension,
                                const TableSlots& tables,
                                const at::Tensor& seed,
                                std::string_view ntt_backend,
                                std::optional<int64_t> hamming_weight) {
  return fhelium::ckks::create_secret_key(ring_dimension,
                                          unflatten_tables<RingTables>(tables),
                                          seed,
                                          ntt_backend,
                                          hamming_weight);
}

at::Tensor prepare_secret_key_op(const at::Tensor& coefficients,
                                 const TableSlots& tables,
                                 std::string_view ntt_backend) {
  return fhelium::ckks::prepare_secret_key(
      coefficients, unflatten_tables<RingTables>(tables), ntt_backend);
}

at::Tensor create_public_key_op(
    const at::Tensor& secret_key,
    const TableSlots& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    std::string_view ntt_backend,
    const std::optional<at::Tensor>& uniform,
    const std::optional<at::Tensor>& error_coefficients) {
  return fhelium::ckks::create_public_key(secret_key,
                                          unflatten_tables<RingTables>(tables),
                                          seed,
                                          cdt_table,
                                          ntt_backend,
                                          uniform,
                                          error_coefficients);
}

using KeyTuple = std::
    tuple<at::Tensor, std::optional<at::Tensor>, std::vector<int64_t>, int64_t>;

KeyTuple key_tuple(const fhelium::ckks::KeySwitchKey& key) {
  return {key.data, key.seed_rows, key.spec.moduli, key.spec.rounds};
}

fhelium::ckks::KeySwitchKey key_value(const at::Tensor& data,
                                      const at::Tensor& seed_rows,
                                      c10::IntArrayRef moduli,
                                      std::string_view prf,
                                      int64_t rounds) {
  return {data,
          seed_rows,
          {std::string(prf),
           rounds,
           std::vector<int64_t>(moduli.begin(), moduli.end())}};
}

at::Tensor expand_key_switch_key_op(const at::Tensor& data,
                                    const at::Tensor& seed_rows,
                                    c10::IntArrayRef moduli,
                                    std::string_view prf,
                                    int64_t rounds) {
  return fhelium::ckks::expand_key_switch_key(
             key_value(data, seed_rows, moduli, prf, rounds))
      .data;
}

at::Tensor compact_key_switch_key_op(const at::Tensor& data,
                                     const at::Tensor& seed_rows,
                                     c10::IntArrayRef moduli,
                                     std::string_view prf,
                                     int64_t rounds) {
  return fhelium::ckks::compact_key_switch_key(
             key_value(data, seed_rows, moduli, prf, rounds))
      .data;
}

KeyTuple create_key_switch_key_op(const at::Tensor& source_secret_key,
                                  const at::Tensor& destination_secret_key,
                                  const TableSlots& tables,
                                  FHELIUM_KEY_LAYOUT_PARAMETERS) {
  return key_tuple(fhelium::ckks::create_key_switch_key(
      source_secret_key,
      destination_secret_key,
      unflatten_tables<RingTables>(tables),
      seed,
      cdt_table,
      layout(ntt_backend, q_rows, digit_sizes, p_rows),
      uniform_by_digit,
      error_by_digit));
}

KeyTuple create_relinearization_key_op(const at::Tensor& secret_key,
                                       const TableSlots& tables,
                                       FHELIUM_KEY_LAYOUT_PARAMETERS) {
  return key_tuple(fhelium::ckks::create_relinearization_key(
      secret_key,
      unflatten_tables<RingTables>(tables),
      seed,
      cdt_table,
      layout(ntt_backend, q_rows, digit_sizes, p_rows),
      uniform_by_digit,
      error_by_digit));
}

KeyTuple create_galois_key_op(const at::Tensor& secret_key,
                              int64_t galois_element,
                              const TableSlots& tables,
                              FHELIUM_KEY_LAYOUT_PARAMETERS) {
  return key_tuple(fhelium::ckks::create_galois_key(
      secret_key,
      galois_element,
      unflatten_tables<RingTables>(tables),
      seed,
      cdt_table,
      layout(ntt_backend, q_rows, digit_sizes, p_rows),
      uniform_by_digit,
      error_by_digit));
}

at::Tensor convert_secret_key_op(const at::Tensor& secret_key,
                                 const TableSlots& tables,
                                 std::string_view ntt_backend) {
  return fhelium::ckks::convert_secret_key(
      secret_key,
      unflatten_tables<fhelium::SecretConversionTables>(tables),
      ntt_backend);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "create_secret_key(int ring_dimension, Tensor?[] tables, Tensor seed, "
      "str ntt_backend, int? hamming_weight) "
      "-> Tensor",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "prepare_secret_key(Tensor coefficients, Tensor?[] tables, str "
      "ntt_backend) -> Tensor",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "create_public_key(Tensor secret_key, Tensor?[] tables, Tensor seed, "
      "Tensor cdt_table, str ntt_backend, "
      "Tensor? uniform, Tensor? error_coefficients) -> Tensor",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "create_key_switch_key(Tensor source_secret_key, "
      "Tensor destination_secret_key, Tensor?[] "
      "tables, " FHELIUM_KEY_LAYOUT_SCHEMA ") -> (Tensor, Tensor?, int[], int)",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "create_relinearization_key(Tensor secret_key, Tensor?[] "
      "tables, " FHELIUM_KEY_LAYOUT_SCHEMA ") -> (Tensor, Tensor?, int[], int)",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "create_galois_key(Tensor secret_key, int galois_element, "
      "Tensor?[] tables, " FHELIUM_KEY_LAYOUT_SCHEMA
      ") -> (Tensor, Tensor?, int[], int)",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "expand_key_switch_key(Tensor data, Tensor seed_rows, int[] moduli, "
      "str prf, int rounds) -> Tensor",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "compact_key_switch_key(Tensor data, Tensor seed_rows, int[] moduli, "
      "str prf, int rounds) -> Tensor",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "convert_secret_key(Tensor secret_key, Tensor?[] tables, str "
      "ntt_backend) -> Tensor",
      {at::Tag::cudagraph_unsafe});
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeImplicitAutograd, m) {
  m.impl("create_secret_key", &create_secret_key_op);
  m.impl("prepare_secret_key", &prepare_secret_key_op);
  m.impl("create_public_key", &create_public_key_op);
  m.impl("create_key_switch_key", &create_key_switch_key_op);
  m.impl("create_relinearization_key", &create_relinearization_key_op);
  m.impl("create_galois_key", &create_galois_key_op);
  m.impl("expand_key_switch_key", &expand_key_switch_key_op);
  m.impl("compact_key_switch_key", &compact_key_switch_key_op);
  m.impl("convert_secret_key", &convert_secret_key_op);
}
