#include <torch/library.h>

#include <fhelium/ckks_keys.h>
#include <fhelium/tensor_table.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Torch operator schemas for CKKS key generation. Each adapter builds the
// TensorTable from the parallel `table_names` and `tables` arguments and the
// KeySwitchKeyLayout from its flattened fields, then calls the public function
// of the same name. Key generation composes registered operators, so it is
// registered as CompositeImplicitAutograd kernels.

namespace {

using fhelium::TensorTable;
using fhelium::ckks::KeySwitchKeyLayout;

#define FHELIUM_KEY_LAYOUT_SCHEMA                                     \
  "Tensor seed, Tensor cdt_table, str ntt_backend, int q_row_start, " \
  "int[] digit_sizes, int[] p_rows, Tensor? uniform_by_digit, "       \
  "Tensor? error_by_digit"

#define FHELIUM_KEY_LAYOUT_PARAMETERS                        \
  const at::Tensor &seed, const at::Tensor &cdt_table,       \
      std::string_view ntt_backend, int64_t q_row_start,     \
      c10::IntArrayRef digit_sizes, c10::IntArrayRef p_rows, \
      const std::optional<at::Tensor>&uniform_by_digit,      \
      const std::optional<at::Tensor>&error_by_digit

KeySwitchKeyLayout layout(std::string_view ntt_backend,
                          int64_t q_row_start,
                          c10::IntArrayRef digit_sizes,
                          c10::IntArrayRef p_rows) {
  KeySwitchKeyLayout result;
  result.ntt_backend = std::string(ntt_backend);
  result.q_row_start = q_row_start;
  result.digit_sizes = digit_sizes.vec();
  result.p_rows = p_rows.vec();
  return result;
}

at::Tensor create_secret_key_op(int64_t ring_dimension,
                                const std::vector<std::string>& names,
                                at::TensorList tables,
                                const at::Tensor& seed,
                                std::string_view ntt_backend,
                                std::optional<int64_t> hamming_weight) {
  return fhelium::ckks::create_secret_key(ring_dimension,
                                          TensorTable(names, tables),
                                          seed,
                                          ntt_backend,
                                          hamming_weight);
}

at::Tensor prepare_secret_key_op(const at::Tensor& coefficients,
                                 const std::vector<std::string>& names,
                                 at::TensorList tables,
                                 std::string_view ntt_backend) {
  return fhelium::ckks::prepare_secret_key(
      coefficients, TensorTable(names, tables), ntt_backend);
}

at::Tensor create_public_key_op(
    const at::Tensor& secret_key,
    const std::vector<std::string>& names,
    at::TensorList tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    std::string_view ntt_backend,
    const std::optional<at::Tensor>& uniform,
    const std::optional<at::Tensor>& error_coefficients) {
  return fhelium::ckks::create_public_key(secret_key,
                                          TensorTable(names, tables),
                                          seed,
                                          cdt_table,
                                          ntt_backend,
                                          uniform,
                                          error_coefficients);
}

at::Tensor create_key_switch_key_op(const at::Tensor& source_secret_key,
                                    const at::Tensor& destination_secret_key,
                                    const std::vector<std::string>& names,
                                    at::TensorList tables,
                                    FHELIUM_KEY_LAYOUT_PARAMETERS) {
  return fhelium::ckks::create_key_switch_key(
      source_secret_key,
      destination_secret_key,
      TensorTable(names, tables),
      seed,
      cdt_table,
      layout(ntt_backend, q_row_start, digit_sizes, p_rows),
      uniform_by_digit,
      error_by_digit);
}

at::Tensor create_relinearization_key_op(const at::Tensor& secret_key,
                                         const std::vector<std::string>& names,
                                         at::TensorList tables,
                                         FHELIUM_KEY_LAYOUT_PARAMETERS) {
  return fhelium::ckks::create_relinearization_key(
      secret_key,
      TensorTable(names, tables),
      seed,
      cdt_table,
      layout(ntt_backend, q_row_start, digit_sizes, p_rows),
      uniform_by_digit,
      error_by_digit);
}

at::Tensor create_galois_key_op(const at::Tensor& secret_key,
                                int64_t galois_element,
                                const std::vector<std::string>& names,
                                at::TensorList tables,
                                FHELIUM_KEY_LAYOUT_PARAMETERS) {
  return fhelium::ckks::create_galois_key(
      secret_key,
      galois_element,
      TensorTable(names, tables),
      seed,
      cdt_table,
      layout(ntt_backend, q_row_start, digit_sizes, p_rows),
      uniform_by_digit,
      error_by_digit);
}

at::Tensor convert_secret_key_op(const at::Tensor& secret_key,
                                 const std::vector<std::string>& names,
                                 at::TensorList tables,
                                 std::string_view ntt_backend) {
  return fhelium::ckks::convert_secret_key(
      secret_key, TensorTable(names, tables), ntt_backend);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "create_secret_key(int ring_dimension, str[] table_names, "
      "Tensor[] tables, Tensor seed, str ntt_backend, int? hamming_weight) "
      "-> Tensor");
  m.def(
      "prepare_secret_key(Tensor coefficients, str[] table_names, "
      "Tensor[] tables, str ntt_backend) -> Tensor");
  m.def(
      "create_public_key(Tensor secret_key, str[] table_names, "
      "Tensor[] tables, Tensor seed, Tensor cdt_table, str ntt_backend, "
      "Tensor? uniform, Tensor? error_coefficients) -> Tensor");
  m.def(
      "create_key_switch_key(Tensor source_secret_key, "
      "Tensor destination_secret_key, str[] table_names, Tensor[] "
      "tables, " FHELIUM_KEY_LAYOUT_SCHEMA ") -> Tensor");
  m.def(
      "create_relinearization_key(Tensor secret_key, str[] table_names, "
      "Tensor[] tables, " FHELIUM_KEY_LAYOUT_SCHEMA ") -> Tensor");
  m.def(
      "create_galois_key(Tensor secret_key, int galois_element, "
      "str[] table_names, Tensor[] tables, " FHELIUM_KEY_LAYOUT_SCHEMA
      ") -> Tensor");
  m.def(
      "convert_secret_key(Tensor secret_key, str[] table_names, "
      "Tensor[] tables, str ntt_backend) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeImplicitAutograd, m) {
  m.impl("create_secret_key", &create_secret_key_op);
  m.impl("prepare_secret_key", &prepare_secret_key_op);
  m.impl("create_public_key", &create_public_key_op);
  m.impl("create_key_switch_key", &create_key_switch_key_op);
  m.impl("create_relinearization_key", &create_relinearization_key_op);
  m.impl("create_galois_key", &create_galois_key_op);
  m.impl("convert_secret_key", &convert_secret_key_op);
}
