#include <torch/library.h>

#include <fhelium/ckks_operations.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/rns_operations.h>
#include <fhelium/tensor_table.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Torch operator schemas for the whole RNS, NTT, and CKKS operations. Each
// adapter builds the TensorTable from the parallel `table_names` and `tables`
// arguments and the KeySwitchOptions from their flattened fields, then calls
// the public function of the same name. The operations compose registered
// operators, so they are registered as CompositeImplicitAutograd kernels.

namespace {

using fhelium::TensorTable;
using fhelium::ckks::KeySwitchOptions;

#define FHELIUM_KEY_SWITCH_SCHEMA                                        \
  "str ntt_backend, int[] digits, int key_row_start, "                   \
  "int key_p_row_start, int[] key_q_rows, str basis_conversion, "           \
  "int p_count, str moddown_rounding, str input_domain, "                \
  "str output_domain, int galois_generator, Tensor[] key_seeds, "        \
  "int key_rounds"

#define FHELIUM_KEY_SWITCH_PARAMETERS                                        \
  std::string_view ntt_backend, c10::IntArrayRef digits,                     \
      int64_t key_row_start, int64_t key_p_row_start,                        \
      c10::IntArrayRef key_q_rows,                                           \
      std::string_view basis_conversion, int64_t p_count,                        \
      std::string_view moddown_rounding, std::string_view input_domain,      \
      std::string_view output_domain, int64_t galois_generator,              \
      at::TensorList key_seeds, int64_t key_rounds

#define FHELIUM_KEY_SWITCH_OPTIONS     \
  key_switch_options(ntt_backend,      \
                     digits,           \
                     key_row_start,    \
                     key_p_row_start,  \
                     key_q_rows,       \
                     basis_conversion,     \
                     p_count,          \
                     moddown_rounding, \
                     input_domain,     \
                     output_domain,    \
                     galois_generator, \
                     key_seeds,        \
                     key_rounds)

KeySwitchOptions key_switch_options(FHELIUM_KEY_SWITCH_PARAMETERS) {
  KeySwitchOptions options;
  options.ntt_backend = std::string(ntt_backend);
  options.digits = digits.vec();
  options.key_row_start = key_row_start;
  options.key_p_row_start = key_p_row_start;
  options.key_q_rows = key_q_rows.vec();
  options.basis_conversion = std::string(basis_conversion);
  options.p_count = p_count;
  options.moddown_rounding = std::string(moddown_rounding);
  options.input_domain = std::string(input_domain);
  options.output_domain = std::string(output_domain);
  options.galois_generator = galois_generator;
  options.key_seeds = key_seeds.vec();
  options.key_rounds = key_rounds;
  return options;
}

TensorTable table(const std::vector<std::string>& names,
                  at::TensorList tensors) {
  return TensorTable(names, tensors);
}

void transform__op(at::Tensor& residues,
                   const std::vector<std::string>& names,
                   at::TensorList tables,
                   std::string_view ntt_backend,
                   std::string_view basis,
                   std::string_view transition) {
  fhelium::ntt::transform_(
      residues, table(names, tables), ntt_backend, basis, transition);
}

at::Tensor transform_op(const at::Tensor& residues,
                        const std::vector<std::string>& names,
                        at::TensorList tables,
                        std::string_view ntt_backend,
                        std::string_view basis,
                        std::string_view transition) {
  return fhelium::ntt::transform(
      residues, table(names, tables), ntt_backend, basis, transition);
}

at::Tensor hybrid_modup_digit_op(const at::Tensor& source,
                                 const std::vector<std::string>& names,
                                 at::TensorList tables,
                                 int64_t digit_index,
                                 c10::IntArrayRef digits) {
  return fhelium::rns::hybrid_modup_digit(
      source, table(names, tables), digit_index, digits);
}

at::Tensor centered_basis_extend_op(const at::Tensor& source,
                                    const std::vector<std::string>& names,
                                    at::TensorList tables) {
  return fhelium::rns::centered_basis_extend(source, table(names, tables));
}

at::Tensor rounded_trailing_quotient_op(const at::Tensor& source,
                                        const std::vector<std::string>& names,
                                        at::TensorList tables,
                                        int64_t drop_count,
                                        std::string_view rounding) {
  return fhelium::rns::rounded_trailing_quotient(
      source, table(names, tables), drop_count, rounding);
}

at::Tensor moddown_qp_to_q_op(const at::Tensor& source,
                              const std::vector<std::string>& names,
                              at::TensorList tables,
                              int64_t p_count,
                              std::string_view rounding) {
  return fhelium::rns::moddown_qp_to_q(
      source, table(names, tables), p_count, rounding);
}

at::Tensor moddown_ntt_qp_to_q_op(
    const at::Tensor& source,
    const std::optional<at::Tensor>& coefficient_c0,
    const std::vector<std::string>& names,
    at::TensorList tables,
    std::string_view ntt_backend,
    int64_t p_count,
    std::string_view rounding) {
  return fhelium::rns::moddown_ntt_qp_to_q(source,
                                           coefficient_c0,
                                           table(names, tables),
                                           ntt_backend,
                                           p_count,
                                           rounding);
}

at::Tensor moddown_add_rescale_op(const at::Tensor& products,
                                  const at::Tensor& base,
                                  const std::vector<std::string>& names,
                                  at::TensorList tables,
                                  std::string_view ntt_backend,
                                  int64_t q_count,
                                  int64_t joint_drop_count,
                                  bool joint_drop_is_prefix,
                                  std::string_view output_domain) {
  return fhelium::rns::moddown_add_rescale(products,
                                           base,
                                           table(names, tables),
                                           ntt_backend,
                                           q_count,
                                           joint_drop_count,
                                           joint_drop_is_prefix,
                                           output_domain);
}

at::Tensor rational_rescale_op(const at::Tensor& source,
                               const std::vector<std::string>& names,
                               at::TensorList tables,
                               std::string_view input_domain,
                               std::string_view ntt_backend) {
  return fhelium::rns::rational_rescale(
      source, table(names, tables), input_domain, ntt_backend);
}

at::Tensor key_switch_op(const at::Tensor& source,
                         const at::Tensor& key,
                         const std::vector<std::string>& names,
                         at::TensorList tables,
                         FHELIUM_KEY_SWITCH_PARAMETERS,
                         bool conjugate) {
  return fhelium::ckks::key_switch(
      source, key, table(names, tables), FHELIUM_KEY_SWITCH_OPTIONS, conjugate);
}

at::Tensor relinearize_op(const at::Tensor& source,
                          const at::Tensor& key,
                          const std::vector<std::string>& names,
                          at::TensorList tables,
                          FHELIUM_KEY_SWITCH_PARAMETERS,
                          bool rescale,
                          int64_t q_count,
                          int64_t joint_drop_count,
                          bool joint_drop_is_prefix) {
  std::optional<fhelium::ckks::JointRescaleOptions> rescale_options;
  if (rescale) {
    rescale_options = fhelium::ckks::JointRescaleOptions{
        q_count, joint_drop_count, joint_drop_is_prefix};
  }
  return fhelium::ckks::relinearize(source,
                                    key,
                                    table(names, tables),
                                    FHELIUM_KEY_SWITCH_OPTIONS,
                                    rescale_options);
}

at::Tensor rotate_op(const at::Tensor& source,
                     const at::Tensor& key,
                     int64_t step,
                     const std::vector<std::string>& names,
                     at::TensorList tables,
                     FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate(
      source, key, step, table(names, tables), FHELIUM_KEY_SWITCH_OPTIONS);
}

std::vector<at::Tensor> rotate_many_op(const at::Tensor& source,
                                       at::TensorList keys,
                                       c10::IntArrayRef steps,
                                       const std::vector<std::string>& names,
                                       at::TensorList tables,
                                       FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate_many(
      source, keys, steps, table(names, tables), FHELIUM_KEY_SWITCH_OPTIONS);
}

at::Tensor rotation_digits_op(const at::Tensor& component,
                              const std::vector<std::string>& names,
                              at::TensorList tables,
                              FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotation_digits(
      component, table(names, tables), FHELIUM_KEY_SWITCH_OPTIONS);
}

at::Tensor rotate_hoisted_op(const at::Tensor& c0,
                             const at::Tensor& hoisted_digits,
                             const at::Tensor& key,
                             int64_t step,
                             const std::vector<std::string>& names,
                             at::TensorList tables,
                             FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate_hoisted(c0,
                                       hoisted_digits,
                                       key,
                                       step,
                                       table(names, tables),
                                       FHELIUM_KEY_SWITCH_OPTIONS);
}

void rotation_sum_add__op(at::Tensor& ordinary,
                          at::Tensor& products,
                          const at::Tensor& source,
                          const std::optional<at::Tensor>& key,
                          int64_t step,
                          const std::vector<std::string>& names,
                          at::TensorList tables,
                          FHELIUM_KEY_SWITCH_PARAMETERS) {
  fhelium::ckks::rotation_sum_add_(ordinary,
                                   products,
                                   source,
                                   key,
                                   step,
                                   table(names, tables),
                                   FHELIUM_KEY_SWITCH_OPTIONS);
}

at::Tensor rotation_sum_finish__op(at::Tensor& ordinary,
                                   at::Tensor& products,
                                   const std::vector<std::string>& names,
                                   at::TensorList tables,
                                   FHELIUM_KEY_SWITCH_PARAMETERS,
                                   int64_t remaining_q_count) {
  return fhelium::ckks::rotation_sum_finish_(ordinary,
                                             products,
                                             table(names, tables),
                                             FHELIUM_KEY_SWITCH_OPTIONS,
                                             remaining_q_count);
}

at::Tensor rotation_sum_op(at::TensorList sources,
                           at::TensorList keys,
                           c10::IntArrayRef steps,
                           const std::vector<std::string>& names,
                           at::TensorList tables,
                           FHELIUM_KEY_SWITCH_PARAMETERS,
                           int64_t remaining_q_count) {
  return fhelium::ckks::rotation_sum(sources,
                                     keys,
                                     steps,
                                     table(names, tables),
                                     FHELIUM_KEY_SWITCH_OPTIONS,
                                     remaining_q_count);
}

at::Tensor grouped_rotation_weighted_sum_op(
    const at::Tensor& source,
    at::TensorList keys,
    c10::IntArrayRef baby_steps,
    at::TensorList plaintexts,
    int64_t group_count,
    const std::vector<std::string>& names,
    at::TensorList tables,
    FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::grouped_rotation_weighted_sum(
      source,
      keys,
      baby_steps,
      plaintexts,
      group_count,
      table(names, tables),
      FHELIUM_KEY_SWITCH_OPTIONS);
}

at::Tensor encapsulated_modulus_raise_op(const at::Tensor& source,
                                         const at::Tensor& low_key,
                                         const at::Tensor& return_key,
                                         const std::vector<std::string>& names,
                                         at::TensorList tables,
                                         FHELIUM_KEY_SWITCH_PARAMETERS,
                                         int64_t return_drop_count,
                                         int64_t target_row_start,
                                         int64_t q_count) {
  return fhelium::ckks::encapsulated_modulus_raise(source,
                                                   low_key,
                                                   return_key,
                                                   table(names, tables),
                                                   FHELIUM_KEY_SWITCH_OPTIONS,
                                                   return_drop_count,
                                                   target_row_start,
                                                   q_count);
}

at::Tensor encrypt_op(const at::Tensor& coefficients,
                      const at::Tensor& public_key,
                      const std::vector<std::string>& names,
                      at::TensorList tables,
                      const at::Tensor& seed,
                      const at::Tensor& cdt_table,
                      std::string_view ntt_backend,
                      int64_t key_row_start,
                      std::string_view output_domain) {
  return fhelium::ckks::encrypt(coefficients,
                                public_key,
                                table(names, tables),
                                seed,
                                cdt_table,
                                ntt_backend,
                                key_row_start,
                                output_domain);
}

at::Tensor decrypt_phase_op(const at::Tensor& ciphertext,
                            const at::Tensor& secret_key,
                            const std::vector<std::string>& names,
                            at::TensorList tables,
                            std::string_view ntt_backend,
                            int64_t key_row_start,
                            std::string_view input_domain) {
  return fhelium::ckks::decrypt_phase(ciphertext,
                                      secret_key,
                                      table(names, tables),
                                      ntt_backend,
                                      key_row_start,
                                      input_domain);
}

at::Tensor decrypt_op(const at::Tensor& ciphertext,
                      const at::Tensor& secret_key,
                      const std::vector<std::string>& names,
                      at::TensorList tables,
                      std::string_view ntt_backend,
                      int64_t key_row_start,
                      std::string_view input_domain) {
  return fhelium::ckks::decrypt(ciphertext,
                                secret_key,
                                table(names, tables),
                                ntt_backend,
                                key_row_start,
                                input_domain);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "transform_(Tensor(a!) residues, str[] table_names, Tensor[] tables, "
      "str ntt_backend, str basis, str transition) -> ()");
  m.def(
      "transform(Tensor residues, str[] table_names, Tensor[] tables, "
      "str ntt_backend, str basis, str transition) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CompositeImplicitAutograd, m) {
  m.impl("transform_", &transform__op);
  m.impl("transform", &transform_op);
}

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def(
      "hybrid_modup_digit(Tensor source, str[] table_names, Tensor[] tables, "
      "int digit_index, int[] digits) -> Tensor");
  m.def(
      "centered_basis_extend(Tensor source, str[] table_names, "
      "Tensor[] tables) -> Tensor");
  m.def(
      "rounded_trailing_quotient(Tensor source, str[] table_names, "
      "Tensor[] tables, int drop_count, str rounding) -> Tensor");
  m.def(
      "moddown_qp_to_q(Tensor source, str[] table_names, Tensor[] tables, "
      "int p_count, str rounding) -> Tensor");
  m.def(
      "moddown_ntt_qp_to_q(Tensor source, Tensor? coefficient_c0, "
      "str[] table_names, Tensor[] tables, str ntt_backend, int p_count, "
      "str rounding) -> Tensor");
  m.def(
      "moddown_add_rescale(Tensor products, Tensor base, str[] table_names, "
      "Tensor[] tables, str ntt_backend, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix, str output_domain) -> Tensor");
  m.def(
      "rational_rescale(Tensor source, str[] table_names, Tensor[] tables, "
      "str input_domain, str ntt_backend) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CompositeImplicitAutograd, m) {
  m.impl("hybrid_modup_digit", &hybrid_modup_digit_op);
  m.impl("centered_basis_extend", &centered_basis_extend_op);
  m.impl("rounded_trailing_quotient", &rounded_trailing_quotient_op);
  m.impl("moddown_qp_to_q", &moddown_qp_to_q_op);
  m.impl("moddown_ntt_qp_to_q", &moddown_ntt_qp_to_q_op);
  m.impl("rational_rescale", &rational_rescale_op);
  m.impl("moddown_add_rescale", &moddown_add_rescale_op);
}

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "key_switch(Tensor source, Tensor key, str[] table_names, "
      "Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", bool conjugate) -> Tensor");
  m.def(
      "relinearize(Tensor source, Tensor key, str[] table_names, "
      "Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", bool rescale, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix) -> Tensor");
  m.def(
      "rotate(Tensor source, Tensor key, int step, str[] table_names, "
      "Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> Tensor");
  m.def(
      "rotate_many(Tensor source, Tensor[] keys, int[] steps, "
      "str[] table_names, Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ") -> Tensor[]");
  m.def(
      "rotation_digits(Tensor component, str[] table_names, Tensor[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> Tensor");
  m.def(
      "rotate_hoisted(Tensor c0, Tensor hoisted_digits, Tensor key, int step, "
      "str[] table_names, Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ") -> Tensor");
  m.def(
      "rotation_sum_add_(Tensor(a!) ordinary, Tensor(b!) products, "
      "Tensor source, Tensor? key, int step, str[] table_names, "
      "Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> ()");
  m.def(
      "rotation_sum_finish_(Tensor(a!) ordinary, Tensor(b!) products, "
      "str[] table_names, Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count) -> Tensor");
  m.def(
      "rotation_sum(Tensor[] sources, Tensor[] keys, int[] steps, "
      "str[] table_names, Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count) -> Tensor");
  m.def(
      "grouped_rotation_weighted_sum(Tensor source, Tensor[] keys, "
      "int[] baby_steps, Tensor[] plaintexts, int group_count, "
      "str[] table_names, Tensor[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ") -> Tensor");
  m.def(
      "encapsulated_modulus_raise(Tensor source, Tensor low_key, "
      "Tensor return_key, str[] table_names, Tensor[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int return_drop_count, int target_row_start, int q_count) -> Tensor");
  m.def(
      "add_compressed_plaintext(Tensor ciphertext, Tensor compressed, "
      "Tensor? implicit, Tensor parameters, str layout, "
      "str polynomial_domain) -> Tensor");
  m.def(
      "add_compressed_plaintext_(Tensor(a!) ciphertext, Tensor compressed, "
      "Tensor? implicit, Tensor parameters, str layout, "
      "str polynomial_domain) -> Tensor(a!)");
  m.def(
      "multiply_compressed_plaintext(Tensor ciphertext, Tensor compressed, "
      "Tensor? implicit, Tensor parameters, str layout) -> Tensor");
  m.def(
      "multiply_compressed_plaintext_(Tensor(a!) ciphertext, "
      "Tensor compressed, Tensor? implicit, Tensor parameters, "
      "str layout) -> Tensor(a!)");
  m.def(
      "scalar_arithmetic(Tensor ciphertext, Tensor parameters, "
      "Tensor seed, float scalar, float scalar_scale, "
      "bool add) -> Tensor");
  m.def(
      "encode(Tensor message, Tensor pre, Tensor twister, Tensor seed, "
      "float scale) -> Tensor");
  m.def(
      "decode(Tensor coefficients, Tensor post, Tensor skewer, float scale, "
      "bool is_real) -> Tensor");
  m.def(
      "prepare_periodic_plaintext(Tensor message, Tensor pre, "
      "Tensor twister, Tensor seed, Tensor parameters, "
      "Tensor twice_moduli, Tensor even, Tensor odd, Tensor twiddles, "
      "int ring_dimension, float scale, str polynomial_domain) -> Tensor[]");
  m.def(
      "encrypt(Tensor coefficients, Tensor public_key, str[] table_names, "
      "Tensor[] tables, Tensor seed, Tensor cdt_table, str ntt_backend, "
      "int key_row_start, str output_domain) -> Tensor");
  m.def(
      "decrypt_phase(Tensor ciphertext, Tensor secret_key, "
      "str[] table_names, Tensor[] tables, str ntt_backend, "
      "int key_row_start, str input_domain) -> Tensor");
  m.def(
      "reconstruct_q_coefficients(Tensor residues, Tensor source_parameters, "
      "Tensor normalizers, Tensor propagation, Tensor half_digits) -> Tensor");
  m.def(
      "decrypt(Tensor ciphertext, Tensor secret_key, str[] table_names, "
      "Tensor[] tables, str ntt_backend, int key_row_start, "
      "str input_domain) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeImplicitAutograd, m) {
  m.impl("key_switch", &key_switch_op);
  m.impl("relinearize", &relinearize_op);
  m.impl("rotate", &rotate_op);
  m.impl("rotate_many", &rotate_many_op);
  m.impl("rotation_digits", &rotation_digits_op);
  m.impl("rotate_hoisted", &rotate_hoisted_op);
  m.impl("rotation_sum_add_", &rotation_sum_add__op);
  m.impl("rotation_sum_finish_", &rotation_sum_finish__op);
  m.impl("rotation_sum", &rotation_sum_op);
  m.impl("grouped_rotation_weighted_sum", &grouped_rotation_weighted_sum_op);
  m.impl("encapsulated_modulus_raise", &encapsulated_modulus_raise_op);
  m.impl("add_compressed_plaintext", &fhelium::ckks::add_compressed_plaintext);
  m.impl("add_compressed_plaintext_",
         &fhelium::ckks::add_compressed_plaintext_);
  m.impl("multiply_compressed_plaintext",
         &fhelium::ckks::multiply_compressed_plaintext);
  m.impl("multiply_compressed_plaintext_",
         &fhelium::ckks::multiply_compressed_plaintext_);
  m.impl("scalar_arithmetic", &fhelium::ckks::scalar_arithmetic);
  m.impl("encode", &fhelium::ckks::encode);
  m.impl("decode", &fhelium::ckks::decode);
  m.impl("prepare_periodic_plaintext",
         &fhelium::ckks::prepare_periodic_plaintext);
  m.impl("encrypt", &encrypt_op);
  m.impl("decrypt_phase", &decrypt_phase_op);
  m.impl("reconstruct_q_coefficients",
         &fhelium::ckks::reconstruct_q_coefficients);
  m.impl("decrypt", &decrypt_op);
}
