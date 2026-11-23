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
    const std::optional<at::Tensor>& source_normalizers) {
  const auto destinations = matrix.size(1);
  auto values = source.clone();
  rns::reduce_to_standard_(values, source_parameters);
  if (source_normalizers.has_value()) {
    auto normalizers = source_normalizers->unsqueeze(-1).clone();
    rns::to_montgomery_(normalizers, source_parameters);
    values = rns::montgomery_mul_row_scalars_standard(
        values, normalizers.squeeze(-1), source_parameters);
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
  auto result = at::zeros(shape, source.options());
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

at::Tensor drop_coefficient_rows(at::Tensor source,
                                 const TensorTable& tables,
                                 c10::IntArrayRef half_primes,
                                 int64_t drop_count,
                                 bool nearest,
                                 bool in_place) {
  for (int64_t step = 0; step < drop_count; ++step) {
    const auto prefix = "step" + std::to_string(step) + "_";
    const auto& parameters = tables.at(prefix + "parameters");
    const auto& inverse = tables.at(prefix + "inverse");
    auto remaining = source.narrow(-2, 1, source.size(-2) - 1);
    const auto dropped = source.select(-2, 0);
    if (in_place || step != 0) {
      if (nearest) {
        ckks::rescale_drop_leading_prime_nearest_(
            remaining, inverse, dropped, parameters, half_primes[step]);
      } else {
        ckks::rescale_drop_leading_prime_truncate_(
            remaining, inverse, dropped, parameters);
      }
      source = remaining;
    } else if (nearest) {
      source = ckks::rescale_drop_leading_prime_nearest(
          remaining, inverse, dropped, parameters, half_primes[step]);
    } else {
      source = ckks::rescale_drop_leading_prime_truncate(
          remaining, inverse, dropped, parameters);
    }
  }
  return source;
}

at::Tensor rescale(at::Tensor source,
                   const TensorTable& tables,
                   int64_t drop_count,
                   c10::IntArrayRef half_primes,
                   std::string_view rounding,
                   std::string_view input_domain,
                   std::string_view ntt_backend,
                   bool in_place) {
  TORCH_CHECK_VALUE(static_cast<int64_t>(half_primes.size()) >= drop_count,
                    "half_primes requires one entry per dropped prime");
  const bool nearest = rounding == "nearest";
  if (input_domain != "ntt") {
    return drop_coefficient_rows(
        source, tables, half_primes, drop_count, nearest, in_place);
  }
  auto remaining = source.narrow(-2, drop_count, source.size(-2) - drop_count);
  auto dropped = source.narrow(-2, 0, drop_count).clone();
  ntt::transform_(
      dropped, tables, ntt_backend, "dropped", "inverse_to_standard");
  auto correction = at::zeros_like(source);
  correction.narrow(-2, 0, drop_count).copy_(dropped);
  correction = drop_coefficient_rows(
      correction, tables, half_primes, drop_count, nearest, true);
  correction =
      detail_operations::forward_add_scaled(correction,
                                            remaining,
                                            tables.at("prefix_inverse"),
                                            tables.at("remaining_forward_0"),
                                            tables,
                                            ntt_backend,
                                            "remaining",
                                            false);
  if (in_place) {
    remaining.copy_(correction);
    return remaining;
  }
  return correction;
}

}  // namespace

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
  return divide_trailing_rows(source,
                              tables.at("moddown_inverses"),
                              tables.at("qp_parameters"),
                              p_count,
                              tables.get("moddown_half"),
                              std::nullopt,
                              rounding);
}

at::Tensor moddown_ntt_qp_to_q(const at::Tensor& source,
                               const std::optional<at::Tensor>& coefficient_c0,
                               const TensorTable& tables,
                               std::string_view ntt_backend,
                               int64_t p_count,
                               std::string_view rounding) {
  const auto rows = source.size(-2);
  auto p_rows = source.narrow(-2, rows - p_count, p_count).clone();
  ntt::transform_(p_rows, tables, ntt_backend, "p", "inverse_to_standard");
  const auto q_rows = source.narrow(-2, 0, rows - p_count);
  auto correction =
      ckks::keyswitch_moddown_qp_to_q(at::zeros_like(q_rows),
                                      p_rows,
                                      tables.at("moddown_inverses"),
                                      tables.at("qp_parameters"),
                                      tables.get("moddown_half"),
                                      rounding);
  const auto& q_parameters = tables.at("q_parameters");
  if (coefficient_c0.has_value()) {
    auto component = correction.select(0, 0);
    add_standard_(component, *coefficient_c0, q_parameters);
  }
  return detail_operations::forward_add_scaled(correction,
                                               q_rows,
                                               tables.at("p_inverse"),
                                               q_parameters,
                                               tables,
                                               ntt_backend,
                                               "q",
                                               true);
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
  auto numerator_q =
      montgomery_mul_row_scalars_standard(base.narrow(0, 0, products.size(0)),
                                          tables.at("joint_p_times_base"),
                                          q_parameters);
  add_standard_(numerator_q, products.narrow(-2, 0, q_count), q_parameters);
  auto dropped =
      at::cat({rows(numerator_q, true),
               products.narrow(-2, q_count, products.size(-2) - q_count)},
              -2);
  ntt::transform_(
      dropped, tables, ntt_backend, "joint_dropped", "inverse_to_standard");
  add_standard_(dropped,
                tables.at("joint_dropped_offset"),
                tables.at("joint_dropped_parameters"));
  auto remaining = rows(numerator_q, false);
  const auto& remaining_parameters = tables.at("joint_remaining_parameters");
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

at::Tensor rescale_drop_leading_primes(const at::Tensor& source,
                                       const TensorTable& tables,
                                       int64_t drop_count,
                                       c10::IntArrayRef half_primes,
                                       std::string_view rounding,
                                       std::string_view input_domain,
                                       std::string_view ntt_backend) {
  return rescale(source,
                 tables,
                 drop_count,
                 half_primes,
                 rounding,
                 input_domain,
                 ntt_backend,
                 false);
}

at::Tensor rescale_drop_leading_primes_(at::Tensor& source,
                                        const TensorTable& tables,
                                        int64_t drop_count,
                                        c10::IntArrayRef half_primes,
                                        std::string_view rounding,
                                        std::string_view input_domain,
                                        std::string_view ntt_backend) {
  return rescale(source,
                 tables,
                 drop_count,
                 half_primes,
                 rounding,
                 input_domain,
                 ntt_backend,
                 true);
}

}  // namespace fhelium::rns
