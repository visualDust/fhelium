#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include <vector>

#include "composition.h"

namespace fhelium::detail_operations {

at::Tensor portable_crt_basis_extend(
    const at::Tensor& source,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const std::optional<at::Tensor>& source_normalizers,
    const std::optional<at::Tensor>& target_offsets) {
  const auto destinations = matrix.size(1);
  auto values = source.clone();
  rns::reduce_to_standard_(values, source_parameters);
  if (source_normalizers.has_value()) {
    auto normalizers = source_normalizers->unsqueeze(-1).clone();
    rns::to_montgomery_(normalizers, source_parameters);
    values = rns::montgomery_mul_row_scalars_standard(
        values, normalizers.squeeze(-1), source_parameters);
    rns::reduce_to_standard_(values, source_parameters);
    // Shift by (q - 1)/2; target_offsets removes its matrix image, so the
    // lifted digit is the centered residue.
    auto halves = at::floor_divide(source_parameters.select(0, 0), 4)
                      .unsqueeze(-1)
                      .expand_as(values)
                      .contiguous();
    values = rns::add_lazy(values, halves, source_parameters);
    rns::reduce_to_standard_(values, source_parameters);
  }
  // A second factor R lets ordinary Montgomery multiplication map an
  // unencoded source residue directly to the encoded matrix result.
  auto coefficients =
      matrix.transpose(0, 1).clone(at::MemoryFormat::Contiguous);
  rns::to_montgomery_(coefficients, target_parameters);
  std::vector<int64_t> modulus_shape(source.dim() - 2, 1);
  modulus_shape.push_back(-1);
  modulus_shape.push_back(1);
  const auto moduli =
      at::floor_divide(target_parameters.select(0, 0), 2).view(modulus_shape);
  auto shape = source.sizes().vec();
  shape[shape.size() - 2] = destinations;
  auto result =
      target_offsets.has_value()
          ? target_offsets->view(modulus_shape).expand(shape).contiguous()
          : at::zeros(shape, source.options());
  for (int64_t row = 0; row < matrix.size(0); ++row) {
    const auto residues = at::remainder(values.narrow(-2, row, 1), moduli);
    const auto term = rns::montgomery_mul_row_scalars_standard(
        residues, coefficients.select(1, row), target_parameters);
    result = rns::add_lazy(result, term, target_parameters);
  }
  rns::reduce_to_standard_(result, target_parameters);
  return result;
}

}  // namespace fhelium::detail_operations

namespace fhelium::rns {

namespace {

using detail_operations::divide_trailing_rows;

// Centered CRT correction -r/D of `tables` entries named `prefix` plus the
// CRT operand names; see `centered_quotient_tables` in fhelium.backend.rns.
at::Tensor centered_quotient_correction(const at::Tensor& source,
                                        const TensorTable& tables,
                                        const std::string& prefix) {
  return crt_basis_extend(source,
                          tables.at(prefix + "source_parameters"),
                          tables.at(prefix + "target_parameters"),
                          tables.at(prefix + "matrix"),
                          tables.at(prefix + "target_reciprocals"),
                          tables.at(prefix + "source_normalizers"),
                          tables.at(prefix + "source_reciprocals"),
                          tables.at(prefix + "target_offsets"));
}

// Returns `rows` placed at `positions` of a zero tensor with `row_count`
// rows; the other rows hold residue zero.
at::Tensor place_rows(const at::Tensor& rows,
                      const at::Tensor& positions,
                      int64_t row_count) {
  auto shape = rows.sizes().vec();
  shape[shape.size() - 2] = row_count;
  auto placed = at::zeros(shape, rows.options());
  placed.index_copy_(-2, positions, rows);
  return placed;
}

}  // namespace

at::Tensor rational_rescale(const at::Tensor& source,
                            const TensorTable& tables,
                            std::string_view input_domain,
                            std::string_view ntt_backend) {
  const auto& source_parameters = tables.at("source_parameters");
  const auto& target_parameters = tables.at("target_parameters");
  const auto target_rows = target_parameters.size(-1);
  const auto restore = tables.get("restore");
  const auto scaled =
      restore.has_value()
          ? montgomery_mul_row_scalars_standard(
                source, *restore, source_parameters)
          : source;
  auto dropped =
      scaled.index_select(-2, tables.at("drop_positions")).contiguous();
  auto kept = scaled.index_select(-2, tables.at("keep_positions"));
  if (const auto positions = tables.get("target_keep_positions")) {
    kept = place_rows(kept, *positions, target_rows);
  }
  if (input_domain == "ntt") {
    ntt::transform_(
        dropped, tables, ntt_backend, "dropped", "inverse_to_standard");
  }
  add_standard_(
      dropped, tables.at("dropped_offset"), tables.at("dropped_parameters"));
  if (input_domain != "ntt") {
    add_standard_(kept, tables.at("target_offset"), target_parameters);
    return ckks::keyswitch_moddown_qp_to_q(kept,
                                           dropped,
                                           tables.at("inverses"),
                                           tables.at("ordered_parameters"),
                                           std::nullopt,
                                           "floor");
  }
  std::vector<int64_t> offset_shape(dropped.sizes().begin(),
                                    dropped.sizes().end() - 2);
  offset_shape.push_back(-1);
  offset_shape.push_back(-1);
  auto correction = ckks::keyswitch_moddown_qp_to_q(
      tables.at("target_offset").expand(offset_shape),
      dropped,
      tables.at("inverses"),
      tables.at("ordered_parameters"),
      std::nullopt,
      "floor");
  return detail_operations::forward_add_scaled(correction,
                                               kept,
                                               tables.at("product_inverse"),
                                               target_parameters,
                                               tables,
                                               ntt_backend,
                                               "target",
                                               false);
}

at::Tensor hybrid_modup_digit(const at::Tensor& source,
                              const TensorTable& tables,
                              int64_t digit_index,
                              c10::IntArrayRef digits) {
  const auto spans = detail_operations::digit_spans(digits);
  TORCH_CHECK_VALUE(
      digit_index >= 0 && digit_index < static_cast<int64_t>(spans.size()),
      "digit_index is outside the digit list");
  return detail_operations::hybrid_modup_digit(source,
                                               tables,
                                               digit_index,
                                               spans[digit_index],
                                               at::Tensor(),
                                               at::Tensor());
}

at::Tensor centered_basis_extend(const at::Tensor& source,
                                 const TensorTable& tables) {
  const auto& parameters = tables.at("source_parameters");
  auto digits = source.clone();
  reduce_to_standard_(digits, parameters);
  const auto count = source.size(-2);
  if (count > 1) {
    digits = mixed_radix_decompose(digits,
                                   tables.at("normalizers"),
                                   tables.at("propagation"),
                                   tables.at("source_lo"),
                                   tables.at("source_hi"),
                                   tables.at("source_neg_lo"),
                                   tables.at("source_neg_hi"));
    reduce_to_standard_(digits, parameters);
  }
  const auto& target_parameters = tables.at("target_parameters");
  auto output =
      mixed_radix_basis_extend_to_montgomery(digits,
                                             tables.at("extension"),
                                             target_parameters,
                                             target_parameters.size(-1));
  from_montgomery_(output, target_parameters);
  reduce_to_standard_(output, target_parameters);
  auto equal =
      at::ones_like(digits.select(-2, 0), digits.options().dtype(at::kBool));
  auto negative = at::zeros_like(equal);
  const auto& half = tables.at("half_digits");
  for (int64_t row = count - 1; row >= 0; --row) {
    const auto digit = digits.select(-2, row);
    negative.bitwise_or_(equal.bitwise_and(digit.gt(half[row])));
    equal.bitwise_and_(digit.eq(half[row]));
  }
  std::vector<int64_t> shape(output.dim() - 2, 1);
  shape.push_back(-1);
  shape.push_back(1);
  const auto corrected =
      at::where(negative.unsqueeze(-2),
                output - tables.at("source_product").view(shape),
                output);
  return at::remainder(corrected, tables.at("target_moduli").view(shape));
}

at::Tensor rounded_trailing_quotient(const at::Tensor& source,
                                     const TensorTable& tables,
                                     int64_t drop_count,
                                     std::string_view rounding) {
  return divide_trailing_rows(source,
                              tables.at("moddown_inverses"),
                              tables.at("parameters"),
                              drop_count,
                              tables.get("half"),
                              tables.get("row_order"),
                              rounding);
}

at::Tensor moddown_qp_to_q(const at::Tensor& source,
                           const TensorTable& tables,
                           int64_t p_count,
                           std::string_view rounding) {
  if (tables.get("moddown_crt_matrix").has_value()) {
    const auto q_count = source.size(-2) - p_count;
    const auto& q_parameters = tables.at("q_parameters");
    auto result = centered_quotient_correction(
        source.narrow(-2, q_count, p_count), tables, "moddown_crt_");
    add_standard_(result,
                  montgomery_mul_row_scalars_standard(
                      source.narrow(-2, 0, q_count),
                      tables.at("p_inverse"),
                      q_parameters),
                  q_parameters);
    return result;
  }
  return divide_trailing_rows(source,
                              tables.at("moddown_inverses"),
                              tables.at("qp_parameters"),
                              p_count,
                              tables.get("moddown_half"),
                              std::nullopt,
                              rounding);
}

namespace {

// A zero residue Tensor with the shape of `like`, broadcast from one element
// so that readers do not require a materialized fill.
at::Tensor zero_residues(const at::Tensor& like) {
  return at::zeros({1}, like.options()).expand(like.sizes());
}

}  // namespace

}  // namespace fhelium::rns

namespace fhelium::detail_operations {

at::Tensor moddown_ntt_rows(const at::Tensor& q_rows,
                            at::Tensor p_rows,
                            const std::optional<at::Tensor>& coefficient_c0,
                            const TensorTable& tables,
                            std::string_view ntt_backend,
                            std::string_view rounding) {
  ntt::transform_(p_rows, tables, ntt_backend, "p", "inverse_to_standard");
  auto correction =
      tables.get("moddown_crt_matrix").has_value()
          ? rns::centered_quotient_correction(p_rows, tables, "moddown_crt_")
          : ckks::keyswitch_moddown_qp_to_q(rns::zero_residues(q_rows),
                                            p_rows,
                                            tables.at("moddown_inverses"),
                                            tables.at("qp_parameters"),
                                            tables.get("moddown_half"),
                                            rounding);
  const auto& q_parameters = tables.at("q_parameters");
  if (coefficient_c0.has_value()) {
    auto component = correction.select(0, 0);
    rns::add_standard_(component, *coefficient_c0, q_parameters);
  }
  return forward_add_scaled(correction,
                            q_rows,
                            tables.at("p_inverse"),
                            q_parameters,
                            tables,
                            ntt_backend,
                            "q",
                            true);
}

}  // namespace fhelium::detail_operations

namespace fhelium::rns {

at::Tensor moddown_ntt_qp_to_q(const at::Tensor& source,
                               const std::optional<at::Tensor>& coefficient_c0,
                               const TensorTable& tables,
                               std::string_view ntt_backend,
                               int64_t p_count,
                               std::string_view rounding) {
  const auto q_count = source.size(-2) - p_count;
  return detail_operations::moddown_ntt_rows(source.narrow(-2, 0, q_count),
                          source.narrow(-2, q_count, p_count).clone(),
                          coefficient_c0,
                          tables,
                          ntt_backend,
                          rounding);
}

at::Tensor moddown_add_rescale(const at::Tensor& products,
                               const at::Tensor& base,
                               const TensorTable& tables,
                               std::string_view ntt_backend,
                               int64_t q_count,
                               int64_t joint_drop_count,
                               bool joint_drop_is_prefix,
                               std::string_view output_domain) {
  const auto rows = [&](const at::Tensor& value, bool dropped) {
    if (joint_drop_is_prefix) {
      return dropped
                 ? value.narrow(-2, 0, joint_drop_count)
                 : value.narrow(
                       -2, joint_drop_count, value.size(-2) - joint_drop_count);
    }
    return value.index_select(-2,
                              tables.at(dropped ? "joint_drop_positions"
                                                : "joint_remaining_positions"));
  };
  const auto& q_parameters = tables.at("q_parameters");
  const auto restore = tables.get("joint_restore");
  const auto scaled_products =
      restore.has_value()
          ? montgomery_mul_row_scalars_standard(
                products, *restore, tables.at("joint_product_parameters"))
          : products;
  auto numerator_q =
      montgomery_mul_row_scalars_standard(base.narrow(0, 0, products.size(0)),
                                          tables.at("joint_p_times_base"),
                                          q_parameters);
  add_standard_(
      numerator_q, scaled_products.narrow(-2, 0, q_count), q_parameters);
  auto dropped = at::cat(
      {rows(numerator_q, true),
       scaled_products.narrow(-2, q_count, products.size(-2) - q_count)},
      -2);
  ntt::transform_(
      dropped, tables, ntt_backend, "joint_dropped", "inverse_to_standard");
  auto remaining = rows(numerator_q, false);
  const auto& remaining_parameters = tables.at("joint_remaining_parameters");
  if (const auto kept = tables.get("joint_target_keep_positions")) {
    remaining = place_rows(remaining, *kept, remaining_parameters.size(-1));
  }
  if (tables.get("joint_crt_t_matrix").has_value()) {
    // Centered quotient by P on the D rows, y = (x - L_P(x)) / P, then the
    // exact quotient of y by D; the target rows combine both corrections.
    const auto p_rows =
        dropped.narrow(-2, joint_drop_count, dropped.size(-2) - joint_drop_count);
    const auto& d_parameters = tables.at("joint_crt_d_target_parameters");
    auto quotient = centered_quotient_correction(p_rows, tables, "joint_crt_d_");
    add_standard_(quotient,
                  montgomery_mul_row_scalars_standard(
                      dropped.narrow(-2, 0, joint_drop_count),
                      tables.at("joint_crt_p_inverse"),
                      d_parameters),
                  d_parameters);
    auto correction =
        centered_quotient_correction(p_rows, tables, "joint_crt_t_");
    const auto half = tables.get("joint_crt_half");
    add_standard_(correction,
                  ckks::keyswitch_moddown_qp_to_q(
                      zero_residues(correction),
                      quotient,
                      tables.at("joint_crt_inverses"),
                      tables.at("joint_crt_ordered_parameters"),
                      half,
                      half.has_value() ? "nearest" : "floor"),
                  remaining_parameters);
    if (output_domain == "coefficient") {
      remaining = remaining.clone();
      ntt::transform_(remaining,
                      tables,
                      ntt_backend,
                      "joint_remaining",
                      "inverse_to_standard");
      add_standard_(correction,
                    montgomery_mul_row_scalars_standard(
                        remaining,
                        tables.at("joint_product_inverse"),
                        remaining_parameters),
                    remaining_parameters);
      return correction;
    }
    return detail_operations::forward_add_scaled(
        correction,
        remaining,
        tables.at("joint_product_inverse"),
        remaining_parameters,
        tables,
        ntt_backend,
        "joint_remaining",
        false);
  }
  add_standard_(dropped,
                tables.at("joint_dropped_offset"),
                tables.at("joint_dropped_parameters"));
  if (output_domain == "coefficient") {
    remaining = remaining.clone();
    ntt::transform_(remaining,
                    tables,
                    ntt_backend,
                    "joint_remaining",
                    "inverse_to_standard");
    add_standard_(
        remaining, tables.at("joint_remaining_offset"), remaining_parameters);
    return ckks::keyswitch_moddown_qp_to_q(
        remaining,
        dropped,
        tables.at("joint_inverses"),
        tables.at("joint_ordered_parameters"),
        std::nullopt,
        "floor");
  }
  std::vector<int64_t> offset_shape(dropped.sizes().begin(),
                                    dropped.sizes().end() - 2);
  offset_shape.push_back(-1);
  offset_shape.push_back(-1);
  const auto offset = tables.at("joint_remaining_offset").expand(offset_shape);
  auto correction =
      ckks::keyswitch_moddown_qp_to_q(offset,
                                      dropped,
                                      tables.at("joint_inverses"),
                                      tables.at("joint_ordered_parameters"),
                                      std::nullopt,
                                      "floor");
  return detail_operations::forward_add_scaled(
      correction,
      remaining,
      tables.at("joint_product_inverse"),
      remaining_parameters,
      tables,
      ntt_backend,
      "joint_remaining",
      false);
}

}  // namespace fhelium::rns
