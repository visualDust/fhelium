#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_operations.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include <string>
#include <vector>

#include "composition.h"

namespace fhelium::ckks {

namespace {

using detail_operations::DigitSpan;
using detail_operations::NttFamily;

int64_t log2_exact(int64_t value) {
  int64_t bits = 0;
  while ((int64_t{1} << bits) < value) {
    ++bits;
  }
  return bits;
}

int64_t galois_element(int64_t n, int64_t step, int64_t generator) {
  return rotation_galois_element(log2_exact(n), step, generator);
}

const at::Tensor& ntt_indices(int64_t n,
                              int64_t step,
                              const KeySwitchOptions& options,
                              const at::Device& device) {
  return detail_operations::ntt_galois_indices(
      n, galois_element(n, step, options.galois_generator), device);
}

at::Tensor rotate_coefficients(const at::Tensor& source,
                               int64_t element,
                               const TensorTable& tables) {
  const auto& [indices, signs] = detail_operations::coefficient_galois_indices(
      source.size(-1), element, source.device());
  return apply_coefficient_galois_automorphism(
      source, indices, signs, tables.at("q_parameters").select(0, 0));
}

at::Tensor moddown_coefficient(at::Tensor accumulator,
                               const TensorTable& tables,
                               const KeySwitchOptions& options) {
  ntt::transform_(
      accumulator, tables, options.ntt_backend, "qp", "inverse_to_standard");
  return rns::moddown_qp_to_q(
      accumulator, tables, options.p_count, options.moddown_rounding);
}

at::Tensor corrections(const at::Tensor& component,
                       const at::Tensor& key,
                       const TensorTable& tables,
                       const KeySwitchOptions& options,
                       bool output_ntt,
                       const std::optional<at::Tensor>& coefficient_c0) {
  auto accumulator = key_switch_products(component, key, tables, options);
  if (output_ntt) {
    return rns::moddown_ntt_qp_to_q(accumulator,
                                    coefficient_c0,
                                    tables,
                                    options.ntt_backend,
                                    options.p_count,
                                    options.moddown_rounding);
  }
  return moddown_coefficient(accumulator, tables, options);
}

at::Tensor forward_copy(const at::Tensor& source,
                        const TensorTable& tables,
                        const KeySwitchOptions& options) {
  return ntt::transform(
      source, tables, options.ntt_backend, "q", "forward_to_montgomery");
}

// Lifts every digit of `component` once to QP NTT Montgomery residues,
// shape [digit, *batch, limb, N].
at::Tensor accumulate_rotation(const at::Tensor& prepared,
                               const at::Tensor& key,
                               int64_t step,
                               const TensorTable& tables,
                               const KeySwitchOptions& options) {
  const auto prototype = prepared.select(0, 0);
  auto shape = prototype.sizes().vec();
  shape.insert(shape.begin(), 2);
  auto accumulator = at::zeros(shape, prototype.options());
  auto accumulator0 = accumulator.select(0, 0);
  auto accumulator1 = accumulator.select(0, 1);
  const auto spans = detail_operations::digit_spans(options.digits);
  const auto& indices =
      ntt_indices(prototype.size(-1), step, options, prototype.device());
  const auto& parameters = tables.at("qp_parameters");
  bool ordered = true;
  for (size_t i = 0; i < spans.size(); ++i) {
    ordered = ordered && spans[i].key_digit == static_cast<int64_t>(i);
  }
  if (prototype.is_cuda() && prepared.size(0) <= 5 && ordered) {
    keyswitch_accumulate_products_(accumulator0,
                                   accumulator1,
                                   prepared.unbind(0),
                                   key,
                                   parameters,
                                   options.key_row_start,
                                   indices);
  } else {
    for (size_t i = 0; i < spans.size(); ++i) {
      keyswitch_accumulate_digit_products_(
          accumulator0,
          accumulator1,
          prepared.select(0, static_cast<int64_t>(i)),
          key.select(0, spans[i].key_digit),
          parameters,
          options.key_row_start,
          indices);
    }
  }
  return accumulator;
}

}  // namespace

at::Tensor rotation_digits(const at::Tensor& component,
                           const TensorTable& tables,
                           const KeySwitchOptions& options) {
  const auto spans = detail_operations::digit_spans(options.digits);
  TORCH_CHECK_VALUE(!spans.empty(),
                    "Rotation hoisting requires an active RNS digit");
  auto shape = component.sizes().vec();
  shape[shape.size() - 2] = tables.at("qp_parameters").size(1);
  shape.insert(shape.begin(), static_cast<int64_t>(spans.size()));
  auto digits = at::empty(shape, component.options());
  int64_t width = 0;
  for (const auto& span : spans) {
    width = std::max(width, span.stop - span.start);
  }
  at::Tensor scratch;
  if (width > 1 && options.modup_method == "exact") {
    auto scratch_shape = component.sizes().vec();
    scratch_shape[scratch_shape.size() - 2] = width;
    scratch = at::empty(scratch_shape, component.options());
  }
  for (size_t index = 0; index < spans.size(); ++index) {
    auto lifted = detail_operations::lift_key_switch_digit(
        component,
        tables,
        options.modup_method,
        static_cast<int64_t>(index),
        spans[index],
        digits.select(0, static_cast<int64_t>(index)),
        scratch);
    ntt::transform_(
        lifted, tables, options.ntt_backend, "qp", "forward_montgomery");
  }
  return digits;
}

at::Tensor rotate_hoisted(const at::Tensor& c0,
                          const at::Tensor& hoisted_digits,
                          const at::Tensor& key,
                          int64_t step,
                          const TensorTable& tables,
                          const KeySwitchOptions& options) {
  const bool output_ntt = options.output_domain == "ntt";
  auto accumulator =
      accumulate_rotation(hoisted_digits, key, step, tables, options);
  const auto& q_parameters = tables.at("q_parameters");
  if (output_ntt) {
    const auto corrections = rns::moddown_ntt_qp_to_q(accumulator,
                                                      std::nullopt,
                                                      tables,
                                                      options.ntt_backend,
                                                      options.p_count,
                                                      options.moddown_rounding);
    const auto rotated = apply_ntt_galois_automorphism(
        c0, ntt_indices(c0.size(-1), step, options, c0.device()));
    return at::stack(
        {rns::add_lazy(rotated, corrections.select(0, 0), q_parameters),
         corrections.select(0, 1)});
  }
  std::vector<at::Tensor> corrections;
  for (auto component : accumulator.unbind(0)) {
    ntt::transform_(
        component, tables, options.ntt_backend, "qp", "inverse_to_standard");
    corrections.push_back(rns::moddown_qp_to_q(
        component, tables, options.p_count, options.moddown_rounding));
  }
  const auto rotated = rotate_coefficients(
      c0, galois_element(c0.size(-1), step, options.galois_generator), tables);
  return at::stack({rns::add_standard(rotated, corrections[0], q_parameters),
                    corrections[1]});
}

at::Tensor key_switch_products(const at::Tensor& component,
                               const at::Tensor& key,
                               const TensorTable& tables,
                               const KeySwitchOptions& options,
                               const std::optional<at::Tensor>& accumulator) {
  const auto policy = detail_operations::ntt_policy(options.ntt_backend);
  const auto spans = detail_operations::digit_spans(options.digits);
  TORCH_CHECK_VALUE(!spans.empty(),
                    "Key switching requires at least one RNS digit");
  const auto& parameters = tables.at("qp_parameters");
  at::Tensor result = accumulator.has_value() ? *accumulator : at::Tensor();
  for (size_t index = 0; index < spans.size(); ++index) {
    const auto& span = spans[index];
    auto lifted =
        detail_operations::lift_key_switch_digit(component,
                                                 tables,
                                                 options.modup_method,
                                                 static_cast<int64_t>(index),
                                                 span,
                                                 at::Tensor(),
                                                 at::Tensor());
    if (!result.defined()) {
      auto shape = lifted.sizes().vec();
      shape.insert(shape.begin(), 2);
      result = at::zeros(shape, lifted.options());
    }
    auto accumulator0 = result.select(0, 0);
    auto accumulator1 = result.select(0, 1);
    const auto key_digit = key.select(0, span.key_digit);
    if (policy.family == NttFamily::kCompact) {
      ntt::forward_ntt_montgomery_compact_keyswitch_accumulate_(
          lifted,
          tables.at("qp_forward_1"),
          parameters,
          key_digit,
          accumulator0,
          accumulator1,
          options.key_row_start,
          policy.grouped_stage_count);
    } else {
      ntt::transform_(
          lifted, tables, options.ntt_backend, "qp", "forward_montgomery");
      keyswitch_accumulate_digit_products_(accumulator0,
                                           accumulator1,
                                           lifted,
                                           key_digit,
                                           parameters,
                                           options.key_row_start);
    }
  }
  return result;
}

at::Tensor key_switch(const at::Tensor& source,
                      const at::Tensor& key,
                      const TensorTable& tables,
                      const KeySwitchOptions& options,
                      bool conjugate) {
  TORCH_CHECK_VALUE(source.size(0) == 2,
                    "Key switching requires ciphertext components=2");
  const auto input =
      conjugate ? rotate_coefficients(source, 2 * source.size(-1) - 1, tables)
                : source;
  const bool output_ntt = options.output_domain == "ntt";
  const auto result = corrections(
      input.select(0, 1), key, tables, options, output_ntt, input.select(0, 0));
  if (output_ntt) {
    return result;
  }
  return at::stack(
      {rns::add_standard(
           input.select(0, 0), result.select(0, 0), tables.at("q_parameters")),
       result.select(0, 1)});
}

at::Tensor relinearize(const at::Tensor& source,
                       const at::Tensor& key,
                       const TensorTable& tables,
                       const KeySwitchOptions& options,
                       const std::optional<JointRescaleOptions>& rescale) {
  TORCH_CHECK_VALUE(source.size(0) == 3,
                    "Relinearization requires ciphertext components=3");
  const bool output_ntt = options.output_domain == "ntt";
  auto component = source.select(0, 2).clone();
  ntt::transform_(
      component, tables, options.ntt_backend, "q", "inverse_to_standard");
  if (rescale.has_value()) {
    const auto products = key_switch_products(component, key, tables, options);
    return rns::moddown_add_rescale(products,
                                    source,
                                    tables,
                                    options.ntt_backend,
                                    rescale->q_count,
                                    rescale->joint_drop_count,
                                    rescale->joint_drop_is_prefix,
                                    options.output_domain);
  }
  const auto result =
      corrections(component, key, tables, options, output_ntt, std::nullopt);
  const auto& q_parameters = tables.at("q_parameters");
  if (output_ntt) {
    return rns::add_lazy(source.narrow(0, 0, 2), result, q_parameters);
  }
  auto coefficient = source.narrow(0, 0, 2).clone();
  ntt::transform_(
      coefficient, tables, options.ntt_backend, "q", "inverse_to_standard");
  return rns::add_standard(coefficient, result, q_parameters);
}

at::Tensor rotate(const at::Tensor& source,
                  const at::Tensor& key,
                  int64_t step,
                  const TensorTable& tables,
                  const KeySwitchOptions& options) {
  TORCH_CHECK_VALUE(source.size(0) == 2,
                    "Rotation requires ciphertext components=2");
  const auto n = source.size(-1);
  const auto element = galois_element(n, step, options.galois_generator);
  const bool input_ntt = options.input_domain == "ntt";
  const bool output_ntt = options.output_domain == "ntt";
  at::Tensor rotated;
  at::Tensor component;
  if (input_ntt) {
    rotated = apply_ntt_galois_automorphism(
        source,
        detail_operations::ntt_galois_indices(n, element, source.device()));
    component = rotated.select(0, 1).clone();
    ntt::transform_(
        component, tables, options.ntt_backend, "q", "inverse_to_standard");
  } else {
    rotated = rotate_coefficients(source, element, tables);
    component = rotated.select(0, 1);
  }
  const auto result = corrections(
      component,
      key,
      tables,
      options,
      output_ntt,
      output_ntt && !input_ntt ? std::optional<at::Tensor>(rotated.select(0, 0))
                               : std::nullopt);
  const auto& q_parameters = tables.at("q_parameters");
  if (output_ntt) {
    if (!input_ntt) {
      return result;
    }
    return at::stack(
        {rns::add_lazy(rotated.select(0, 0), result.select(0, 0), q_parameters),
         result.select(0, 1)});
  }
  auto component0 = rotated.select(0, 0);
  if (input_ntt) {
    component0 = component0.clone();
    ntt::transform_(
        component0, tables, options.ntt_backend, "q", "inverse_to_standard");
  }
  return at::stack(
      {rns::add_standard(component0, result.select(0, 0), q_parameters),
       result.select(0, 1)});
}

std::vector<at::Tensor> rotate_many(const at::Tensor& source,
                                    at::TensorList keys,
                                    c10::IntArrayRef steps,
                                    const TensorTable& tables,
                                    const KeySwitchOptions& options) {
  TORCH_CHECK_VALUE(keys.size() == steps.size(),
                    "rotate_many requires one key per step");
  std::vector<at::Tensor> results;
  if (steps.empty()) {
    return results;
  }
  const auto digits = rotation_digits(source.select(0, 1), tables, options);
  const auto c0 = options.output_domain == "ntt"
                      ? forward_copy(source.select(0, 0), tables, options)
                      : source.select(0, 0);
  for (size_t i = 0; i < steps.size(); ++i) {
    results.push_back(
        rotate_hoisted(c0, digits, keys[i], steps[i], tables, options));
  }
  return results;
}

void rotation_sum_add_(at::Tensor& ordinary,
                       at::Tensor& products,
                       const at::Tensor& source,
                       const std::optional<at::Tensor>& key,
                       int64_t step,
                       const TensorTable& tables,
                       const KeySwitchOptions& options) {
  TORCH_CHECK_VALUE(
      options.input_domain == "ntt" || options.input_domain == "coefficient",
      "Rotation-sum input domain must be ntt or coefficient");
  const bool input_ntt = options.input_domain == "ntt";
  const auto& q_parameters = tables.at("q_parameters");
  const auto accumulate = [&](at::Tensor target, const at::Tensor& value) {
    target.copy_(input_ntt ? rns::add_lazy(target, value, q_parameters)
                           : rns::add_standard(target, value, q_parameters));
  };
  if (step == 0) {
    accumulate(ordinary, source);
    return;
  }
  TORCH_CHECK_VALUE(key.has_value(),
                    "A nonzero rotation requires its evaluation key");
  const auto n = source.size(-1);
  const auto element = galois_element(n, step, options.galois_generator);
  at::Tensor rotated;
  at::Tensor component;
  if (input_ntt) {
    const auto supplied = tables.get("sum_galois_" + std::to_string(step));
    rotated = apply_ntt_galois_automorphism(
        source,
        supplied.has_value() ? *supplied
                             : detail_operations::ntt_galois_indices(
                                   n, element, source.device()));
    component = rotated.select(0, 1).clone();
    ntt::transform_(
        component, tables, options.ntt_backend, "q", "inverse_to_standard");
  } else {
    rotated = rotate_coefficients(source, element, tables);
    component = rotated.select(0, 1);
  }
  key_switch_products(component, *key, tables, options, products);
  accumulate(ordinary.select(0, 0), rotated.select(0, 0));
}

at::Tensor rotation_sum_finish_(at::Tensor& ordinary,
                                at::Tensor& products,
                                const TensorTable& tables,
                                const KeySwitchOptions& options,
                                int64_t remaining_q_count) {
  const auto& q_parameters = tables.at("q_parameters");
  if (options.input_domain != "ntt") {
    ntt::transform_(
        ordinary, tables, options.ntt_backend, "q", "forward_to_montgomery");
  }
  rns::montgomery_mul_row_scalars_(
      ordinary, tables.at("sum_p_montgomery"), q_parameters);
  auto q_rows = products.narrow(-2, 0, ordinary.size(-2));
  q_rows.copy_(rns::add_lazy(q_rows, ordinary, q_parameters));
  ntt::transform_(
      products, tables, options.ntt_backend, "qp", "inverse_to_standard");
  return detail_operations::divide_trailing_rows(
      products,
      tables.at("sum_quotient_inverses"),
      tables.at("sum_quotient_parameters"),
      products.size(-2) - remaining_q_count,
      tables.get("sum_quotient_half"),
      tables.get("sum_quotient_order"),
      options.moddown_rounding);
}

at::Tensor rotation_sum(at::TensorList sources,
                        at::TensorList keys,
                        c10::IntArrayRef steps,
                        const TensorTable& tables,
                        const KeySwitchOptions& options,
                        int64_t remaining_q_count) {
  TORCH_CHECK_VALUE(sources.size() == steps.size(),
                    "rotation_sum requires one step per source");
  TORCH_CHECK_VALUE(!sources.empty(),
                    "Rotation sum requires at least one contribution");
  auto ordinary = at::zeros_like(sources[0]);
  auto shape = ordinary.sizes().vec();
  shape[shape.size() - 2] = tables.at("qp_parameters").size(1);
  auto products = at::zeros(shape, ordinary.options());
  size_t key_index = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    std::optional<at::Tensor> key;
    if (steps[i] != 0) {
      TORCH_CHECK_VALUE(key_index < keys.size(),
                        "A nonzero rotation requires its evaluation key");
      key = keys[key_index++];
    }
    rotation_sum_add_(
        ordinary, products, sources[i], key, steps[i], tables, options);
  }
  return rotation_sum_finish_(
      ordinary, products, tables, options, remaining_q_count);
}

at::Tensor grouped_rotation_weighted_sum(const at::Tensor& source,
                                         at::TensorList keys,
                                         c10::IntArrayRef baby_steps,
                                         at::TensorList plaintexts,
                                         int64_t group_count,
                                         const TensorTable& tables,
                                         const KeySwitchOptions& options) {
  const auto digits = rotation_digits(source.select(0, 1), tables, options);
  const auto source_ntt = forward_copy(source, tables, options);
  auto ntt_options = options;
  ntt_options.output_domain = "ntt";
  std::vector<at::Tensor> babies;
  size_t key_index = 0;
  for (const auto step : baby_steps) {
    if (step == 0) {
      babies.push_back(source_ntt);
    } else {
      TORCH_CHECK_VALUE(key_index < keys.size(),
                        "A nonzero rotation requires its evaluation key");
      babies.push_back(rotate_hoisted(source_ntt.select(0, 0),
                                      digits,
                                      keys[key_index++],
                                      step,
                                      tables,
                                      ntt_options));
    }
  }
  return rns::montgomery_weighted_sums(
      babies, plaintexts, group_count, tables.at("q_parameters"));
}

}  // namespace fhelium::ckks
