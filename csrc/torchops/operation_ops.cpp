#include <torch/library.h>

#include <fhelium/ckks_operations.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/operation_tables.h>
#include <fhelium/rns_operations.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Torch operator schemas for the whole RNS, NTT, and CKKS operations. Each
// adapter unflattens the positional `tables` (and `galois`) slots into the
// table structs of fhelium/operation_tables.h and the KeySwitchOptions from
// their flattened fields, then calls the public function of the same name. The
// operations are registered as CompositeExplicitAutograd kernels so that
// tracing records each one as a single operator with its fake implementation.

namespace {

using fhelium::TableSlots;
using fhelium::unflatten_tables;
using fhelium::ckks::KeySwitchOptions;

#define FHELIUM_KEY_SWITCH_SCHEMA                                 \
  "str ntt_backend, int[] digits, int key_row_start, "            \
  "int key_p_row_start, int[] key_q_rows, str basis_conversion, " \
  "int p_count, str moddown_rounding, str input_domain, "         \
  "str output_domain, Tensor[] key_seeds, int key_rounds"

#define FHELIUM_KEY_SWITCH_PARAMETERS                                 \
  std::string_view ntt_backend, c10::IntArrayRef digits,              \
      int64_t key_row_start, int64_t key_p_row_start,                 \
      c10::IntArrayRef key_q_rows, std::string_view basis_conversion, \
      int64_t p_count, std::string_view moddown_rounding,             \
      std::string_view input_domain, std::string_view output_domain,  \
      at::TensorList key_seeds, int64_t key_rounds

#define FHELIUM_KEY_SWITCH_OPTIONS     \
  key_switch_options(ntt_backend,      \
                     digits,           \
                     key_row_start,    \
                     key_p_row_start,  \
                     key_q_rows,       \
                     basis_conversion, \
                     p_count,          \
                     moddown_rounding, \
                     input_domain,     \
                     output_domain,    \
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
  options.key_seeds = key_seeds.vec();
  options.key_rounds = key_rounds;
  return options;
}

fhelium::KeySwitchTables key_switch_tables(const TableSlots& tables) {
  return unflatten_tables<fhelium::KeySwitchTables>(tables);
}

// The Galois tables of a single automorphism, or none for an empty list.
std::optional<fhelium::GaloisTables> optional_galois(const TableSlots& slots) {
  if (slots.empty()) return std::nullopt;
  return unflatten_tables<fhelium::GaloisTables>(slots);
}

void transform__op(at::Tensor& residues,
                   const TableSlots& tables,
                   std::string_view ntt_backend,
                   std::string_view transition) {
  fhelium::ntt::transform_(residues,
                           unflatten_tables<fhelium::NttTables>(tables),
                           ntt_backend,
                           transition);
}

at::Tensor transform_op(const at::Tensor& residues,
                        const TableSlots& tables,
                        std::string_view ntt_backend,
                        std::string_view transition) {
  return fhelium::ntt::transform(residues,
                                 unflatten_tables<fhelium::NttTables>(tables),
                                 ntt_backend,
                                 transition);
}

at::Tensor hybrid_modup_digit_op(const at::Tensor& source,
                                 const TableSlots& tables,
                                 int64_t digit_index,
                                 c10::IntArrayRef digits) {
  return fhelium::rns::hybrid_modup_digit(
      source, key_switch_tables(tables), digit_index, digits);
}

at::Tensor centered_basis_extend_op(const at::Tensor& source,
                                    const TableSlots& tables) {
  return fhelium::rns::centered_basis_extend(
      source, unflatten_tables<fhelium::BasisExtensionTables>(tables));
}

at::Tensor rounded_trailing_quotient_op(const at::Tensor& source,
                                        const TableSlots& tables,
                                        int64_t drop_count,
                                        std::string_view rounding) {
  return fhelium::rns::rounded_trailing_quotient(
      source,
      unflatten_tables<fhelium::TrailingQuotientTables>(tables),
      drop_count,
      rounding);
}

at::Tensor moddown_qp_to_q_op(const at::Tensor& source,
                              const TableSlots& tables,
                              int64_t p_count,
                              std::string_view rounding) {
  return fhelium::rns::moddown_qp_to_q(
      source, key_switch_tables(tables), p_count, rounding);
}

at::Tensor moddown_ntt_qp_to_q_op(
    const at::Tensor& source,
    const std::optional<at::Tensor>& coefficient_c0,
    const TableSlots& tables,
    std::string_view ntt_backend,
    int64_t p_count,
    std::string_view rounding) {
  return fhelium::rns::moddown_ntt_qp_to_q(source,
                                           coefficient_c0,
                                           key_switch_tables(tables),
                                           ntt_backend,
                                           p_count,
                                           rounding);
}

at::Tensor moddown_add_rescale_op(const at::Tensor& products,
                                  const at::Tensor& base,
                                  const TableSlots& tables,
                                  std::string_view ntt_backend,
                                  int64_t q_count,
                                  int64_t joint_drop_count,
                                  bool joint_drop_is_prefix,
                                  std::string_view output_domain) {
  return fhelium::rns::moddown_add_rescale(products,
                                           base,
                                           key_switch_tables(tables),
                                           ntt_backend,
                                           q_count,
                                           joint_drop_count,
                                           joint_drop_is_prefix,
                                           output_domain);
}

at::Tensor rational_rescale_op(const at::Tensor& source,
                               const TableSlots& tables,
                               std::string_view input_domain,
                               std::string_view ntt_backend) {
  return fhelium::rns::rational_rescale(
      source,
      unflatten_tables<fhelium::RescaleTables>(tables),
      input_domain,
      ntt_backend);
}

void hybrid_modup_digit_out_op(const at::Tensor& source,
                               const TableSlots& tables,
                               int64_t digit_index,
                               c10::IntArrayRef digits,
                               at::Tensor& out) {
  fhelium::rns::hybrid_modup_digit_out(
      source, key_switch_tables(tables), digit_index, digits, out);
}

void centered_basis_extend_out_op(const at::Tensor& source,
                                  const TableSlots& tables,
                                  at::Tensor& out,
                                  const std::optional<at::Tensor>& workspace) {
  fhelium::rns::centered_basis_extend_out(
      source,
      unflatten_tables<fhelium::BasisExtensionTables>(tables),
      out,
      workspace.value_or(at::Tensor()));
}

int64_t centered_basis_extend_workspace_bytes_op(const at::Tensor& source,
                                                 const TableSlots& tables) {
  return fhelium::rns::centered_basis_extend_workspace_bytes(
      source, unflatten_tables<fhelium::BasisExtensionTables>(tables));
}

void rounded_trailing_quotient_out_op(
    const at::Tensor& source,
    const TableSlots& tables,
    int64_t drop_count,
    std::string_view rounding,
    at::Tensor& out,
    const std::optional<at::Tensor>& workspace) {
  fhelium::rns::rounded_trailing_quotient_out(
      source,
      unflatten_tables<fhelium::TrailingQuotientTables>(tables),
      drop_count,
      rounding,
      out,
      workspace.value_or(at::Tensor()));
}

int64_t rounded_trailing_quotient_workspace_bytes_op(
    const at::Tensor& source,
    const TableSlots& tables,
    int64_t drop_count,
    std::string_view rounding) {
  return fhelium::rns::rounded_trailing_quotient_workspace_bytes(
      source,
      unflatten_tables<fhelium::TrailingQuotientTables>(tables),
      drop_count,
      rounding);
}

void moddown_qp_to_q_out_op(const at::Tensor& source,
                            const TableSlots& tables,
                            int64_t p_count,
                            std::string_view rounding,
                            at::Tensor& out,
                            const std::optional<at::Tensor>& workspace) {
  fhelium::rns::moddown_qp_to_q_out(source,
                                    key_switch_tables(tables),
                                    p_count,
                                    rounding,
                                    out,
                                    workspace.value_or(at::Tensor()));
}

int64_t moddown_qp_to_q_workspace_bytes_op(const at::Tensor& source,
                                           const TableSlots& tables,
                                           int64_t p_count,
                                           std::string_view rounding) {
  return fhelium::rns::moddown_qp_to_q_workspace_bytes(
      source, key_switch_tables(tables), p_count, rounding);
}

void moddown_ntt_qp_to_q_out_op(const at::Tensor& source,
                                const std::optional<at::Tensor>& coefficient_c0,
                                const TableSlots& tables,
                                std::string_view ntt_backend,
                                int64_t p_count,
                                std::string_view rounding,
                                at::Tensor& out) {
  fhelium::rns::moddown_ntt_qp_to_q_out(source,
                                        coefficient_c0,
                                        key_switch_tables(tables),
                                        ntt_backend,
                                        p_count,
                                        rounding,
                                        out);
}

void moddown_add_rescale_out_op(const at::Tensor& products,
                                const at::Tensor& base,
                                const TableSlots& tables,
                                std::string_view ntt_backend,
                                int64_t q_count,
                                int64_t joint_drop_count,
                                bool joint_drop_is_prefix,
                                std::string_view output_domain,
                                at::Tensor& out,
                                const std::optional<at::Tensor>& workspace) {
  fhelium::rns::moddown_add_rescale_out(products,
                                        base,
                                        key_switch_tables(tables),
                                        ntt_backend,
                                        q_count,
                                        joint_drop_count,
                                        joint_drop_is_prefix,
                                        output_domain,
                                        out,
                                        workspace.value_or(at::Tensor()));
}

int64_t moddown_add_rescale_workspace_bytes_op(const at::Tensor& products,
                                               const at::Tensor& base,
                                               const TableSlots& tables,
                                               std::string_view ntt_backend,
                                               int64_t q_count,
                                               int64_t joint_drop_count,
                                               bool joint_drop_is_prefix,
                                               std::string_view output_domain) {
  return fhelium::rns::moddown_add_rescale_workspace_bytes(
      products,
      base,
      key_switch_tables(tables),
      ntt_backend,
      q_count,
      joint_drop_count,
      joint_drop_is_prefix,
      output_domain);
}

void rational_rescale_out_op(const at::Tensor& source,
                             const TableSlots& tables,
                             std::string_view input_domain,
                             std::string_view ntt_backend,
                             at::Tensor& out,
                             const std::optional<at::Tensor>& workspace) {
  fhelium::rns::rational_rescale_out(
      source,
      unflatten_tables<fhelium::RescaleTables>(tables),
      input_domain,
      ntt_backend,
      out,
      workspace.value_or(at::Tensor()));
}

int64_t rational_rescale_workspace_bytes_op(const at::Tensor& source,
                                            const TableSlots& tables,
                                            std::string_view input_domain,
                                            std::string_view ntt_backend) {
  return fhelium::rns::rational_rescale_workspace_bytes(
      source,
      unflatten_tables<fhelium::RescaleTables>(tables),
      input_domain,
      ntt_backend);
}

at::Tensor key_switch_op(const at::Tensor& source,
                         const at::Tensor& key,
                         const TableSlots& tables,
                         const TableSlots& conjugation,
                         FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::key_switch(source,
                                   key,
                                   key_switch_tables(tables),
                                   FHELIUM_KEY_SWITCH_OPTIONS,
                                   optional_galois(conjugation));
}

at::Tensor relinearize_op(const at::Tensor& source,
                          const at::Tensor& key,
                          const TableSlots& tables,
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
                                    key_switch_tables(tables),
                                    FHELIUM_KEY_SWITCH_OPTIONS,
                                    rescale_options);
}

at::Tensor rotate_op(const at::Tensor& source,
                     const at::Tensor& key,
                     const TableSlots& galois,
                     const TableSlots& tables,
                     FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate(source,
                               key,
                               unflatten_tables<fhelium::GaloisTables>(galois),
                               key_switch_tables(tables),
                               FHELIUM_KEY_SWITCH_OPTIONS);
}

std::vector<at::Tensor> rotate_many_op(const at::Tensor& source,
                                       at::TensorList keys,
                                       const TableSlots& galois,
                                       const TableSlots& tables,
                                       FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate_many(source,
                                    keys,
                                    fhelium::unflatten_galois_tables(galois),
                                    key_switch_tables(tables),
                                    FHELIUM_KEY_SWITCH_OPTIONS);
}

at::Tensor rotation_digits_op(const at::Tensor& component,
                              const TableSlots& tables,
                              FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotation_digits(
      component, key_switch_tables(tables), FHELIUM_KEY_SWITCH_OPTIONS);
}

at::Tensor rotate_hoisted_op(const at::Tensor& c0,
                             const at::Tensor& hoisted_digits,
                             const at::Tensor& key,
                             const TableSlots& galois,
                             const TableSlots& tables,
                             FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate_hoisted(
      c0,
      hoisted_digits,
      key,
      unflatten_tables<fhelium::GaloisTables>(galois),
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS);
}

void rotation_sum_add__op(at::Tensor& ordinary,
                          at::Tensor& products,
                          const at::Tensor& source,
                          const std::optional<at::Tensor>& key,
                          const TableSlots& galois,
                          const TableSlots& tables,
                          FHELIUM_KEY_SWITCH_PARAMETERS,
                          const std::optional<at::Tensor>& workspace) {
  fhelium::ckks::rotation_sum_add_(ordinary,
                                   products,
                                   source,
                                   key,
                                   optional_galois(galois),
                                   key_switch_tables(tables),
                                   FHELIUM_KEY_SWITCH_OPTIONS,
                                   workspace.value_or(at::Tensor()));
}

int64_t rotation_sum_add_workspace_bytes_op(
    const at::Tensor& ordinary,
    const at::Tensor& products,
    const at::Tensor& source,
    const std::optional<at::Tensor>& key,
    const TableSlots& galois,
    const TableSlots& tables,
    FHELIUM_KEY_SWITCH_PARAMETERS) {
  (void)ordinary;
  (void)products;
  (void)key;
  return fhelium::ckks::rotation_sum_add_workspace_bytes(
      source,
      optional_galois(galois),
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS);
}

void rotation_sum_finish_out_op(at::Tensor& ordinary,
                                at::Tensor& products,
                                const TableSlots& tables,
                                FHELIUM_KEY_SWITCH_PARAMETERS,
                                int64_t remaining_q_count,
                                at::Tensor& out,
                                const std::optional<at::Tensor>& workspace) {
  fhelium::ckks::rotation_sum_finish_out_(ordinary,
                                          products,
                                          key_switch_tables(tables),
                                          FHELIUM_KEY_SWITCH_OPTIONS,
                                          remaining_q_count,
                                          out,
                                          workspace.value_or(at::Tensor()));
}

int64_t rotation_sum_finish_workspace_bytes_op(const at::Tensor& ordinary,
                                               const at::Tensor& products,
                                               const TableSlots& tables,
                                               FHELIUM_KEY_SWITCH_PARAMETERS,
                                               int64_t remaining_q_count) {
  (void)ordinary;
  return fhelium::ckks::rotation_sum_finish_workspace_bytes(
      products, key_switch_tables(tables), remaining_q_count);
}

at::Tensor rotation_sum_finish__op(at::Tensor& ordinary,
                                   at::Tensor& products,
                                   const TableSlots& tables,
                                   FHELIUM_KEY_SWITCH_PARAMETERS,
                                   int64_t remaining_q_count) {
  return fhelium::ckks::rotation_sum_finish_(ordinary,
                                             products,
                                             key_switch_tables(tables),
                                             FHELIUM_KEY_SWITCH_OPTIONS,
                                             remaining_q_count);
}

at::Tensor rotation_sum_op(at::TensorList sources,
                           at::TensorList keys,
                           c10::IntArrayRef steps,
                           const TableSlots& galois,
                           const TableSlots& tables,
                           FHELIUM_KEY_SWITCH_PARAMETERS,
                           int64_t remaining_q_count) {
  return fhelium::ckks::rotation_sum(sources,
                                     keys,
                                     steps,
                                     fhelium::unflatten_galois_tables(galois),
                                     key_switch_tables(tables),
                                     FHELIUM_KEY_SWITCH_OPTIONS,
                                     remaining_q_count);
}

at::Tensor grouped_rotation_weighted_sum_op(const at::Tensor& source,
                                            at::TensorList keys,
                                            c10::IntArrayRef baby_steps,
                                            const TableSlots& galois,
                                            at::TensorList plaintexts,
                                            int64_t group_count,
                                            const TableSlots& tables,
                                            FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::grouped_rotation_weighted_sum(
      source,
      keys,
      baby_steps,
      fhelium::unflatten_galois_tables(galois),
      plaintexts,
      group_count,
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS);
}

void key_switch_out_op(const at::Tensor& source,
                       const at::Tensor& key,
                       const TableSlots& tables,
                       const TableSlots& conjugation,
                       FHELIUM_KEY_SWITCH_PARAMETERS,
                       at::Tensor& out) {
  fhelium::ckks::key_switch_out(source,
                                key,
                                key_switch_tables(tables),
                                FHELIUM_KEY_SWITCH_OPTIONS,
                                optional_galois(conjugation),
                                out);
}

void relinearize_out_op(const at::Tensor& source,
                        const at::Tensor& key,
                        const TableSlots& tables,
                        FHELIUM_KEY_SWITCH_PARAMETERS,
                        bool rescale,
                        int64_t q_count,
                        int64_t joint_drop_count,
                        bool joint_drop_is_prefix,
                        at::Tensor& out,
                        const std::optional<at::Tensor>& workspace) {
  std::optional<fhelium::ckks::JointRescaleOptions> rescale_options;
  if (rescale) {
    rescale_options = fhelium::ckks::JointRescaleOptions{
        q_count, joint_drop_count, joint_drop_is_prefix};
  }
  fhelium::ckks::relinearize_out(source,
                                 key,
                                 key_switch_tables(tables),
                                 FHELIUM_KEY_SWITCH_OPTIONS,
                                 rescale_options,
                                 out,
                                 workspace.value_or(at::Tensor()));
}

int64_t relinearize_workspace_bytes_op(const at::Tensor& source,
                                       const at::Tensor& key,
                                       const TableSlots& tables,
                                       FHELIUM_KEY_SWITCH_PARAMETERS,
                                       bool rescale,
                                       int64_t q_count,
                                       int64_t joint_drop_count,
                                       bool joint_drop_is_prefix) {
  (void)key;
  std::optional<fhelium::ckks::JointRescaleOptions> rescale_options;
  if (rescale) {
    rescale_options = fhelium::ckks::JointRescaleOptions{
        q_count, joint_drop_count, joint_drop_is_prefix};
  }
  return fhelium::ckks::relinearize_workspace_bytes(source,
                                                    key_switch_tables(tables),
                                                    FHELIUM_KEY_SWITCH_OPTIONS,
                                                    rescale_options);
}

void rotate_out_op(const at::Tensor& source,
                   const at::Tensor& key,
                   const TableSlots& galois,
                   const TableSlots& tables,
                   FHELIUM_KEY_SWITCH_PARAMETERS,
                   at::Tensor& out) {
  fhelium::ckks::rotate_out(source,
                            key,
                            unflatten_tables<fhelium::GaloisTables>(galois),
                            key_switch_tables(tables),
                            FHELIUM_KEY_SWITCH_OPTIONS,
                            out);
}

void rotation_digits_out_op(const at::Tensor& component,
                            const TableSlots& tables,
                            FHELIUM_KEY_SWITCH_PARAMETERS,
                            at::Tensor& out) {
  fhelium::ckks::rotation_digits_out(
      component, key_switch_tables(tables), FHELIUM_KEY_SWITCH_OPTIONS, out);
}

void rotate_hoisted_out_op(const at::Tensor& c0,
                           const at::Tensor& hoisted_digits,
                           const at::Tensor& key,
                           const TableSlots& galois,
                           const TableSlots& tables,
                           FHELIUM_KEY_SWITCH_PARAMETERS,
                           at::Tensor& out,
                           const std::optional<at::Tensor>& workspace) {
  fhelium::ckks::rotate_hoisted_out(
      c0,
      hoisted_digits,
      key,
      unflatten_tables<fhelium::GaloisTables>(galois),
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS,
      out,
      workspace.value_or(at::Tensor()));
}

int64_t rotate_hoisted_workspace_bytes_op(const at::Tensor& c0,
                                          const at::Tensor& hoisted_digits,
                                          const at::Tensor& key,
                                          const TableSlots& galois,
                                          const TableSlots& tables,
                                          FHELIUM_KEY_SWITCH_PARAMETERS) {
  (void)key;
  (void)galois;
  return fhelium::ckks::rotate_hoisted_workspace_bytes(
      c0,
      hoisted_digits,
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS);
}

void rotate_many_out_op(const at::Tensor& source,
                        at::TensorList keys,
                        const TableSlots& galois,
                        const TableSlots& tables,
                        FHELIUM_KEY_SWITCH_PARAMETERS,
                        at::TensorList out,
                        const std::optional<at::Tensor>& workspace) {
  auto buffer = workspace.value_or(at::Tensor());
  fhelium::ckks::rotate_many_out(source,
                                 keys,
                                 fhelium::unflatten_galois_tables(galois),
                                 key_switch_tables(tables),
                                 FHELIUM_KEY_SWITCH_OPTIONS,
                                 out,
                                 buffer);
}

int64_t rotate_many_workspace_bytes_op(const at::Tensor& source,
                                       at::TensorList keys,
                                       const TableSlots& galois,
                                       const TableSlots& tables,
                                       FHELIUM_KEY_SWITCH_PARAMETERS) {
  return fhelium::ckks::rotate_many_workspace_bytes(
      source,
      keys,
      fhelium::unflatten_galois_tables(galois),
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS);
}

void rotation_sum_out_op(at::TensorList sources,
                         at::TensorList keys,
                         c10::IntArrayRef steps,
                         const TableSlots& galois,
                         const TableSlots& tables,
                         FHELIUM_KEY_SWITCH_PARAMETERS,
                         int64_t remaining_q_count,
                         at::Tensor& out) {
  fhelium::ckks::rotation_sum_out(sources,
                                  keys,
                                  steps,
                                  fhelium::unflatten_galois_tables(galois),
                                  key_switch_tables(tables),
                                  FHELIUM_KEY_SWITCH_OPTIONS,
                                  remaining_q_count,
                                  out);
}

void grouped_rotation_weighted_sum_out_op(const at::Tensor& source,
                                          at::TensorList keys,
                                          c10::IntArrayRef baby_steps,
                                          const TableSlots& galois,
                                          at::TensorList plaintexts,
                                          int64_t group_count,
                                          const TableSlots& tables,
                                          FHELIUM_KEY_SWITCH_PARAMETERS,
                                          at::Tensor& out) {
  fhelium::ckks::grouped_rotation_weighted_sum_out(
      source,
      keys,
      baby_steps,
      fhelium::unflatten_galois_tables(galois),
      plaintexts,
      group_count,
      key_switch_tables(tables),
      FHELIUM_KEY_SWITCH_OPTIONS,
      out);
}

at::Tensor encapsulated_modulus_raise_op(const at::Tensor& source,
                                         const at::Tensor& low_key,
                                         const at::Tensor& return_key,
                                         const TableSlots& tables,
                                         FHELIUM_KEY_SWITCH_PARAMETERS,
                                         int64_t return_drop_count,
                                         int64_t target_row_start,
                                         int64_t q_count) {
  return fhelium::ckks::encapsulated_modulus_raise(
      source,
      low_key,
      return_key,
      unflatten_tables<fhelium::EncapsulationTables>(tables),
      FHELIUM_KEY_SWITCH_OPTIONS,
      return_drop_count,
      target_row_start,
      q_count);
}

at::Tensor encrypt_op(const at::Tensor& coefficients,
                      const at::Tensor& public_key,
                      const TableSlots& tables,
                      at::Tensor& seed,
                      const at::Tensor& cdt_table,
                      std::string_view ntt_backend,
                      int64_t key_row_start,
                      std::string_view output_domain) {
  return fhelium::ckks::encrypt(coefficients,
                                public_key,
                                unflatten_tables<fhelium::RingTables>(tables),
                                seed,
                                cdt_table,
                                ntt_backend,
                                key_row_start,
                                output_domain);
}

at::Tensor decrypt_phase_op(const at::Tensor& ciphertext,
                            const at::Tensor& secret_key,
                            const TableSlots& tables,
                            std::string_view ntt_backend,
                            int64_t key_row_start,
                            std::string_view input_domain) {
  return fhelium::ckks::decrypt_phase(
      ciphertext,
      secret_key,
      unflatten_tables<fhelium::RingTables>(tables),
      ntt_backend,
      key_row_start,
      input_domain);
}

at::Tensor decrypt_op(const at::Tensor& ciphertext,
                      const at::Tensor& secret_key,
                      const TableSlots& tables,
                      std::string_view ntt_backend,
                      int64_t key_row_start,
                      std::string_view input_domain) {
  return fhelium::ckks::decrypt(
      ciphertext,
      secret_key,
      unflatten_tables<fhelium::DecryptionTables>(tables),
      ntt_backend,
      key_row_start,
      input_domain);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def(
      "transform_(Tensor(a!) residues, Tensor?[] tables, str ntt_backend, "
      "str transition) -> ()");
  m.def(
      "transform(Tensor residues, Tensor?[] tables, str ntt_backend, "
      "str transition) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CompositeExplicitAutograd, m) {
  m.impl("transform_", &transform__op);
  m.impl("transform", &transform_op);
}

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def(
      "hybrid_modup_digit(Tensor source, Tensor?[] tables, int digit_index, "
      "int[] digits) -> Tensor");
  m.def("centered_basis_extend(Tensor source, Tensor?[] tables) -> Tensor");
  m.def(
      "rounded_trailing_quotient(Tensor source, Tensor?[] tables, "
      "int drop_count, str rounding) -> Tensor");
  m.def(
      "moddown_qp_to_q(Tensor source, Tensor?[] tables, int p_count, "
      "str rounding) -> Tensor");
  m.def(
      "moddown_ntt_qp_to_q(Tensor source, Tensor? coefficient_c0, "
      "Tensor?[] tables, str ntt_backend, int p_count, str rounding) "
      "-> Tensor");
  m.def(
      "moddown_add_rescale(Tensor products, Tensor base, Tensor?[] tables, "
      "str ntt_backend, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix, str output_domain) -> Tensor");
  m.def(
      "rational_rescale(Tensor source, Tensor?[] tables, str input_domain, "
      "str ntt_backend) -> Tensor");
  m.def(
      "hybrid_modup_digit_out(Tensor source, Tensor?[] tables, int "
      "digit_index, int[] digits, Tensor(a!) out) -> ()");
  m.def(
      "centered_basis_extend_out(Tensor source, Tensor?[] tables, Tensor(a!) "
      "out, Tensor(b!)? workspace=None) -> ()");
  m.def(
      "centered_basis_extend_workspace_bytes(Tensor source, Tensor?[] tables) "
      "-> int");
  m.def(
      "rounded_trailing_quotient_out(Tensor source, Tensor?[] tables, int "
      "drop_count, str rounding, Tensor(a!) out, Tensor(b!)? workspace=None) "
      "-> ()");
  m.def(
      "rounded_trailing_quotient_workspace_bytes(Tensor source, Tensor?[] "
      "tables, int drop_count, str rounding) -> int");
  m.def(
      "moddown_qp_to_q_out(Tensor source, Tensor?[] tables, int p_count, str "
      "rounding, Tensor(a!) out, Tensor(b!)? workspace=None) -> ()");
  m.def(
      "moddown_qp_to_q_workspace_bytes(Tensor source, Tensor?[] tables, int "
      "p_count, str rounding) -> int");
  m.def(
      "moddown_ntt_qp_to_q_out(Tensor source, Tensor? coefficient_c0, "
      "Tensor?[] tables, str ntt_backend, int p_count, str rounding, "
      "Tensor(a!) out) -> ()");
  m.def(
      "moddown_add_rescale_out(Tensor products, Tensor base, Tensor?[] tables, "
      "str ntt_backend, int q_count, int joint_drop_count, bool "
      "joint_drop_is_prefix, str output_domain, Tensor(a!) out, "
      "Tensor(b!)? workspace=None) -> ()");
  m.def(
      "moddown_add_rescale_workspace_bytes(Tensor products, Tensor base, "
      "Tensor?[] tables, str ntt_backend, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix, str output_domain) -> int");
  m.def(
      "rational_rescale_out(Tensor source, Tensor?[] tables, str input_domain, "
      "str ntt_backend, Tensor(a!) out, Tensor(b!)? workspace=None) -> ()");
  m.def(
      "rational_rescale_workspace_bytes(Tensor source, Tensor?[] tables, str "
      "input_domain, str ntt_backend) -> int");
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CompositeExplicitAutograd, m) {
  m.impl("hybrid_modup_digit", &hybrid_modup_digit_op);
  m.impl("centered_basis_extend", &centered_basis_extend_op);
  m.impl("rounded_trailing_quotient", &rounded_trailing_quotient_op);
  m.impl("moddown_qp_to_q", &moddown_qp_to_q_op);
  m.impl("moddown_ntt_qp_to_q", &moddown_ntt_qp_to_q_op);
  m.impl("rational_rescale", &rational_rescale_op);
  m.impl("moddown_add_rescale", &moddown_add_rescale_op);
  m.impl("hybrid_modup_digit_out", &hybrid_modup_digit_out_op);
  m.impl("centered_basis_extend_out", &centered_basis_extend_out_op);
  m.impl("rounded_trailing_quotient_out", &rounded_trailing_quotient_out_op);
  m.impl("moddown_qp_to_q_out", &moddown_qp_to_q_out_op);
  m.impl("moddown_ntt_qp_to_q_out", &moddown_ntt_qp_to_q_out_op);
  m.impl("moddown_add_rescale_out", &moddown_add_rescale_out_op);
  m.impl("rational_rescale_out", &rational_rescale_out_op);
}

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def(
      "key_switch(Tensor source, Tensor key, Tensor?[] tables, "
      "Tensor?[] conjugation, " FHELIUM_KEY_SWITCH_SCHEMA ") -> Tensor");
  m.def(
      "relinearize(Tensor source, Tensor key, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", bool rescale, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix) -> Tensor");
  m.def(
      "rotate(Tensor source, Tensor key, Tensor?[] galois, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> Tensor");
  m.def(
      "rotate_many(Tensor source, Tensor[] keys, Tensor?[] galois, "
      "Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> Tensor[]");
  m.def(
      "rotation_digits(Tensor component, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> Tensor");
  m.def(
      "rotate_hoisted(Tensor c0, Tensor hoisted_digits, Tensor key, "
      "Tensor?[] galois, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ") -> Tensor");
  m.def(
      "rotation_sum_add_(Tensor(a!) ordinary, Tensor(b!) products, "
      "Tensor source, Tensor? key, Tensor?[] galois, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", Tensor(c!)? workspace=None) -> ()");
  m.def(
      "rotation_sum_add_workspace_bytes(Tensor ordinary, Tensor products, "
      "Tensor source, Tensor? key, Tensor?[] galois, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> int");
  m.def(
      "rotation_sum_finish_(Tensor(a!) ordinary, Tensor(b!) products, "
      "Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count) -> Tensor");
  m.def(
      "rotation_sum_finish_out_(Tensor(a!) ordinary, Tensor(b!) products, "
      "Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count, Tensor(c!) out, Tensor(d!)? workspace=None) "
      "-> ()");
  m.def(
      "rotation_sum_finish_workspace_bytes(Tensor ordinary, Tensor products, "
      "Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count) -> int");
  m.def(
      "rotation_sum(Tensor[] sources, Tensor[] keys, int[] steps, "
      "Tensor?[] galois, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count) -> Tensor");
  m.def(
      "grouped_rotation_weighted_sum(Tensor source, Tensor[] keys, "
      "int[] baby_steps, Tensor?[] galois, Tensor[] plaintexts, "
      "int group_count, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ") -> Tensor");
  m.def(
      "encapsulated_modulus_raise(Tensor source, Tensor low_key, "
      "Tensor return_key, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
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
      "Tensor(a!) seed, float scalar, float scalar_scale, "
      "bool add) -> Tensor");
  m.def(
      "add_compressed_plaintext_out(Tensor ciphertext, Tensor compressed, "
      "Tensor? implicit, Tensor parameters, str layout, "
      "str polynomial_domain, Tensor(a!) out) -> ()");
  m.def(
      "multiply_compressed_plaintext_out(Tensor ciphertext, "
      "Tensor compressed, Tensor? implicit, Tensor parameters, "
      "str layout, Tensor(a!) out) -> ()");
  m.def(
      "scalar_arithmetic_out(Tensor ciphertext, Tensor parameters, "
      "Tensor(a!) seed, float scalar, float scalar_scale, "
      "bool add, Tensor(b!) out) -> ()");
  m.def(
      "encode(Tensor message, Tensor pre, Tensor twister, Tensor(a!) seed, "
      "float scale) -> Tensor",
      {at::Tag::cudagraph_unsafe});
  m.def(
      "decode(Tensor coefficients, Tensor post, Tensor skewer, float scale, "
      "bool is_real) -> Tensor");
  m.def(
      "prepare_periodic_plaintext(Tensor message, Tensor pre, "
      "Tensor twister, Tensor(a!) seed, Tensor parameters, "
      "Tensor twice_moduli, Tensor even, Tensor odd, Tensor twiddles, "
      "int ring_dimension, float scale, str polynomial_domain) -> Tensor[]");
  m.def(
      "encrypt(Tensor coefficients, Tensor public_key, Tensor?[] tables, "
      "Tensor(a!) seed, Tensor cdt_table, str ntt_backend, "
      "int key_row_start, str output_domain) -> Tensor");
  m.def(
      "decrypt_phase(Tensor ciphertext, Tensor secret_key, Tensor?[] tables, "
      "str ntt_backend, int key_row_start, str input_domain) -> Tensor");
  m.def(
      "reconstruct_q_coefficients(Tensor residues, Tensor source_parameters, "
      "Tensor normalizers, Tensor propagation, Tensor half_digits) -> Tensor");
  m.def(
      "decrypt(Tensor ciphertext, Tensor secret_key, Tensor?[] tables, "
      "str ntt_backend, int key_row_start, str input_domain) -> Tensor");
  m.def(
      "key_switch_out(Tensor source, Tensor key, Tensor?[] tables, "
      "Tensor?[] conjugation, " FHELIUM_KEY_SWITCH_SCHEMA
      ", Tensor(a!) out) -> ()");
  m.def(
      "relinearize_out(Tensor source, Tensor key, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", bool rescale, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix, Tensor(a!) out, Tensor(b!)? workspace=None) "
      "-> ()");
  m.def(
      "relinearize_workspace_bytes(Tensor source, Tensor key, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", bool rescale, int q_count, int joint_drop_count, "
      "bool joint_drop_is_prefix) -> int");
  m.def(
      "rotate_out(Tensor source, Tensor key, Tensor?[] galois, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ", Tensor(a!) out) -> ()");
  m.def(
      "rotation_digits_out(Tensor component, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ", Tensor(a!) out) -> ()");
  m.def(
      "rotate_hoisted_out(Tensor c0, Tensor hoisted_digits, Tensor key, "
      "Tensor?[] galois, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", Tensor(a!) out, Tensor(b!)? workspace=None) -> ()");
  m.def(
      "rotate_hoisted_workspace_bytes(Tensor c0, Tensor hoisted_digits, "
      "Tensor key, Tensor?[] galois, Tensor?[] "
      "tables, " FHELIUM_KEY_SWITCH_SCHEMA ") -> int");
  m.def(
      "rotate_many_out(Tensor source, Tensor[] keys, Tensor?[] galois, "
      "Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", Tensor(a!)[] out, Tensor(b!)? workspace=None) -> ()");
  m.def(
      "rotate_many_workspace_bytes(Tensor source, Tensor[] keys, "
      "Tensor?[] galois, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ") -> int");
  m.def(
      "rotation_sum_out(Tensor[] sources, Tensor[] keys, int[] steps, "
      "Tensor?[] galois, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", int remaining_q_count, Tensor(a!) out) -> ()");
  m.def(
      "grouped_rotation_weighted_sum_out(Tensor source, Tensor[] keys, "
      "int[] baby_steps, Tensor?[] galois, Tensor[] plaintexts, "
      "int group_count, Tensor?[] tables, " FHELIUM_KEY_SWITCH_SCHEMA
      ", Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeExplicitAutograd, m) {
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
  m.impl("add_compressed_plaintext_out",
         &fhelium::ckks::add_compressed_plaintext_out);
  m.impl("multiply_compressed_plaintext_out",
         &fhelium::ckks::multiply_compressed_plaintext_out);
  m.impl("scalar_arithmetic_out", &fhelium::ckks::scalar_arithmetic_out);
  m.impl("encode", &fhelium::ckks::encode);
  m.impl("decode", &fhelium::ckks::decode);
  m.impl("prepare_periodic_plaintext",
         &fhelium::ckks::prepare_periodic_plaintext);
  m.impl("encrypt", &encrypt_op);
  m.impl("decrypt_phase", &decrypt_phase_op);
  m.impl("reconstruct_q_coefficients",
         &fhelium::ckks::reconstruct_q_coefficients);
  m.impl("decrypt", &decrypt_op);
  m.impl("key_switch_out", &key_switch_out_op);
  m.impl("relinearize_out", &relinearize_out_op);
  m.impl("rotate_out", &rotate_out_op);
  m.impl("rotation_digits_out", &rotation_digits_out_op);
  m.impl("rotate_hoisted_out", &rotate_hoisted_out_op);
  m.impl("rotation_sum_finish_out_", &rotation_sum_finish_out_op);
  m.impl("rotate_many_out", &rotate_many_out_op);
  m.impl("rotation_sum_out", &rotation_sum_out_op);
  m.impl("grouped_rotation_weighted_sum_out",
         &grouped_rotation_weighted_sum_out_op);
}

// Workspace size queries read only operand metadata and run on every
// backend, including Meta and fake Tensors.
TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeImplicitAutograd, m) {
  m.impl("rotate_many_workspace_bytes", &rotate_many_workspace_bytes_op);
  m.impl("relinearize_workspace_bytes", &relinearize_workspace_bytes_op);
  m.impl("rotate_hoisted_workspace_bytes", &rotate_hoisted_workspace_bytes_op);
  m.impl("rotation_sum_add_workspace_bytes",
         &rotation_sum_add_workspace_bytes_op);
  m.impl("rotation_sum_finish_workspace_bytes",
         &rotation_sum_finish_workspace_bytes_op);
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CompositeImplicitAutograd, m) {
  m.impl("centered_basis_extend_workspace_bytes",
         &centered_basis_extend_workspace_bytes_op);
  m.impl("rational_rescale_workspace_bytes",
         &rational_rescale_workspace_bytes_op);
  m.impl("moddown_qp_to_q_workspace_bytes",
         &moddown_qp_to_q_workspace_bytes_op);
  m.impl("rounded_trailing_quotient_workspace_bytes",
         &rounded_trailing_quotient_workspace_bytes_op);
  m.impl("moddown_add_rescale_workspace_bytes",
         &moddown_add_rescale_workspace_bytes_op);
}
