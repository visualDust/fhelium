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

// Seed row of digit `digit` of key `key_index`, or none for dense keys.
std::optional<at::Tensor> key_seed(const KeySwitchOptions& options,
                                   size_t key_index,
                                   int64_t digit) {
  if (options.key_seeds.empty()) return std::nullopt;
  return options.key_seeds[key_index].select(0, digit);
}

// Options of a single-key call that uses key `key_index` of `options`.
KeySwitchOptions for_key(const KeySwitchOptions& options, size_t key_index) {
  auto selected = options;
  if (!options.key_seeds.empty()) {
    selected.key_seeds = {options.key_seeds[key_index]};
  }
  return selected;
}

// Accumulates one digit's key products. With `key_q_rows`, active Q rows
// read the listed key rows: each maximal run of consecutive key rows, and the
// trailing auxiliary rows at `key_p_row_start`, is one contiguous product.
void accumulate_digit_products(at::Tensor& accumulator0,
                               at::Tensor& accumulator1,
                               const at::Tensor& digit,
                               const at::Tensor& key_digit,
                               const at::Tensor& parameters,
                               const KeySwitchOptions& options,
                               const std::optional<at::Tensor>& indices,
                               const std::optional<at::Tensor>& seed) {
  const auto& rows = options.key_q_rows;
  if (rows.empty()) {
    keyswitch_accumulate_digit_products_(accumulator0,
                                         accumulator1,
                                         digit,
                                         key_digit,
                                         parameters,
                                         options.key_row_start,
                                         options.key_p_row_start,
                                         indices,
                                         seed,
                                         options.key_rounds);
    return;
  }
  const auto q_rows = static_cast<int64_t>(rows.size());
  const auto operand_rows = digit.size(-2);
  TORCH_CHECK_VALUE(options.key_p_row_start >= 0 && q_rows < operand_rows,
                    "key_q_rows requires trailing auxiliary key rows");
  auto product = [&](int64_t start, int64_t count, int64_t key_start) {
    auto out0 = accumulator0.narrow(-2, start, count);
    auto out1 = accumulator1.narrow(-2, start, count);
    keyswitch_accumulate_digit_products_(out0,
                                         out1,
                                         digit.narrow(-2, start, count),
                                         key_digit,
                                         parameters.narrow(1, start, count),
                                         key_start,
                                         -1,
                                         indices,
                                         seed,
                                         options.key_rounds);
  };
  int64_t start = 0;
  for (int64_t row = 1; row <= q_rows; ++row) {
    if (row == q_rows || rows[row] != rows[row - 1] + 1) {
      product(start, row - start, rows[start]);
      start = row;
    }
  }
  product(q_rows, operand_rows - q_rows, options.key_p_row_start);
}

at::Tensor moddown_coefficient(at::Tensor accumulator,
                               const TensorTable& tables,
                               const KeySwitchOptions& options) {
  ntt::transform_(
      accumulator, tables, options.ntt_backend, "qp", "inverse_to_standard");
  return rns::moddown_qp_to_q(
      accumulator, tables, options.p_count, options.moddown_rounding);
}

// NTT ModDown of a QP accumulator owned by the caller; its P rows are
// overwritten.
at::Tensor moddown_ntt_owned(const at::Tensor& accumulator,
                             const std::optional<at::Tensor>& coefficient_c0,
                             const TensorTable& tables,
                             const KeySwitchOptions& options) {
  const auto q_count = accumulator.size(-2) - options.p_count;
  return detail_operations::moddown_ntt_rows(
      accumulator.narrow(-2, 0, q_count),
      accumulator.narrow(-2, q_count, options.p_count),
      coefficient_c0,
      tables,
      options.ntt_backend,
      options.moddown_rounding);
}

at::Tensor corrections(const at::Tensor& component,
                       const at::Tensor& key,
                       const TensorTable& tables,
                       const KeySwitchOptions& options,
                       bool output_ntt,
                       const std::optional<at::Tensor>& coefficient_c0) {
  auto accumulator = key_switch_products(component, key, tables, options);
  if (output_ntt) {
    return moddown_ntt_owned(accumulator, coefficient_c0, tables, options);
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
// Key-switching products of the hoisted digits for every rotation, shape
// [rotation, 2, *batch, limb, N].
at::Tensor accumulate_rotations(const at::Tensor& prepared,
                                at::TensorList keys,
                                at::TensorList seeds,
                                c10::IntArrayRef steps,
                                const TensorTable& tables,
                                const KeySwitchOptions& options) {
  const auto prototype = prepared.select(0, 0);
  auto shape = prototype.sizes().vec();
  shape.insert(shape.begin(), {static_cast<int64_t>(steps.size()), 2});
  const auto spans = detail_operations::digit_spans(options.digits);
  const auto& parameters = tables.at("qp_parameters");
  bool ordered = true;
  for (size_t i = 0; i < spans.size(); ++i) {
    ordered = ordered && spans[i].key_digit == static_cast<int64_t>(i);
  }
  // The grouped product kernel overwrites every accumulator and reads the
  // digits once for all rotations; the per-digit path accumulates into zeros.
  const bool grouped =
      prototype.is_cuda() && ordered && options.key_q_rows.empty();
  auto accumulators = grouped ? at::empty(shape, prototype.options())
                              : at::zeros(shape, prototype.options());
  std::vector<at::Tensor> accumulators0, accumulators1, indices, narrowed,
      narrowed_seeds;
  for (size_t r = 0; r < steps.size(); ++r) {
    accumulators0.push_back(accumulators[r][0]);
    accumulators1.push_back(accumulators[r][1]);
    indices.push_back(
        ntt_indices(prototype.size(-1), steps[r], options, prototype.device()));
  }
  if (grouped) {
    for (const auto& key : keys) narrowed.push_back(key.narrow(0, 0, prepared.size(0)));
    for (const auto& seed : seeds)
      narrowed_seeds.push_back(seed.narrow(0, 0, prepared.size(0)));
    keyswitch_accumulate_products_(accumulators0,
                                   accumulators1,
                                   prepared.unbind(0),
                                   narrowed,
                                   parameters,
                                   options.key_row_start,
                                   options.key_p_row_start,
                                   indices,
                                   narrowed_seeds,
                                   options.key_rounds);
    return accumulators;
  }
  for (size_t r = 0; r < steps.size(); ++r) {
    for (size_t i = 0; i < spans.size(); ++i) {
      accumulate_digit_products(accumulators0[r],
                                accumulators1[r],
                                prepared.select(0, static_cast<int64_t>(i)),
                                keys[r].select(0, spans[i].key_digit),
                                parameters,
                                options,
                                indices[r],
                                seeds.empty() ? std::nullopt
                                              : std::optional<at::Tensor>(
                                                    seeds[r].select(0, spans[i].key_digit)));
    }
  }
  return accumulators;
}

// Completes one hoisted rotation from its QP products: ModDown and the
// rotated c0 term.
at::Tensor finish_rotation(const at::Tensor& c0,
                           at::Tensor accumulator,
                           int64_t step,
                           const TensorTable& tables,
                           const KeySwitchOptions& options) {
  const auto& q_parameters = tables.at("q_parameters");
  if (options.output_domain == "ntt") {
    auto result =
        moddown_ntt_owned(accumulator, std::nullopt, tables, options);
    auto component0 = result.select(0, 0);
    add_ntt_galois_automorphism_(
        component0,
        c0,
        ntt_indices(c0.size(-1), step, options, c0.device()),
        q_parameters);
    return result;
  }
  auto result = moddown_coefficient(accumulator, tables, options);
  auto component0 = result.select(0, 0);
  rns::add_standard_(
      component0,
      rotate_coefficients(
          c0, galois_element(c0.size(-1), step, options.galois_generator), tables),
      q_parameters);
  return result;
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
  if (width > 1 && options.basis_conversion == "exact") {
    auto scratch_shape = component.sizes().vec();
    scratch_shape[scratch_shape.size() - 2] = width;
    scratch = at::empty(scratch_shape, component.options());
  }
  for (size_t index = 0; index < spans.size(); ++index) {
    auto lifted = detail_operations::lift_key_switch_digit(
        component,
        tables,
        options.basis_conversion,
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
  const auto accumulators = accumulate_rotations(
      hoisted_digits, {key}, options.key_seeds, {step}, tables, options);
  return finish_rotation(c0, accumulators[0], step, tables, options);
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
                                                 options.basis_conversion,
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
    const auto seed = key_seed(options, 0, span.key_digit);
    if (policy.family == NttFamily::kCompact && options.key_q_rows.empty() &&
        !seed.has_value()) {
      ntt::forward_ntt_montgomery_compact_keyswitch_accumulate_(
          lifted,
          tables.at("qp_forward_1"),
          parameters,
          key_digit,
          accumulator0,
          accumulator1,
          options.key_row_start,
          options.key_p_row_start);
    } else {
      ntt::transform_(
          lifted, tables, options.ntt_backend, "qp", "forward_montgomery");
      accumulate_digit_products(accumulator0,
                                accumulator1,
                                lifted,
                                key_digit,
                                parameters,
                                options,
                                std::nullopt,
                                seed);
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
  if (!output_ntt) {
    auto component0 = result.select(0, 0);
    rns::add_standard_(
        component0, input.select(0, 0), tables.at("q_parameters"));
  }
  return result;
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
  auto rotated0 = rotated.select(0, 0);
  if (input_ntt) {
    rotated0 = rotated0.clone();
    ntt::transform_(
        rotated0, tables, options.ntt_backend, "q", "inverse_to_standard");
  }
  auto component0 = result.select(0, 0);
  rns::add_standard_(component0, rotated0, q_parameters);
  return result;
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
  // With several digits, one product launch for all rotations reads the
  // digits once. With one digit the saving is small, and producing every QP
  // accumulator before its ModDown evicts it from L2, so each rotation runs
  // its product and ModDown in turn.
  const size_t group = digits.size(0) > 1 ? steps.size() : 1;
  for (size_t first = 0; first < steps.size(); first += group) {
    const size_t count = std::min(group, steps.size() - first);
    const auto accumulators = accumulate_rotations(
        digits,
        keys.slice(first, count),
        options.key_seeds.empty()
            ? at::TensorList()
            : at::TensorList(options.key_seeds).slice(first, count),
        steps.slice(first, count),
        tables,
        options);
    for (size_t i = 0; i < count; ++i) {
      results.push_back(finish_rotation(
          c0, accumulators[i], steps[first + i], tables, options));
    }
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
  if (input_ntt) {
    const auto supplied = tables.get("sum_galois_" + std::to_string(step));
    const auto& indices =
        supplied.has_value()
            ? *supplied
            : detail_operations::ntt_galois_indices(n, element, source.device());
    auto component = apply_ntt_galois_automorphism(source.select(0, 1), indices);
    ntt::transform_(
        component, tables, options.ntt_backend, "q", "inverse_to_standard");
    key_switch_products(component, *key, tables, options, products);
    auto ordinary0 = ordinary.select(0, 0);
    add_ntt_galois_automorphism_(
        ordinary0, source.select(0, 0), indices, q_parameters);
    return;
  }
  const auto rotated = rotate_coefficients(source, element, tables);
  key_switch_products(rotated.select(0, 1), *key, tables, options, products);
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
      key = keys[key_index];
    }
    rotation_sum_add_(ordinary,
                      products,
                      sources[i],
                      key,
                      steps[i],
                      tables,
                      key ? for_key(options, key_index++) : options);
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
                                      keys[key_index],
                                      step,
                                      tables,
                                      for_key(ntt_options, key_index)));
      ++key_index;
    }
  }
  return rns::montgomery_weighted_sums(
      babies, plaintexts, group_count, tables.at("q_parameters"));
}

namespace {

// Key data of `keys` and `options` with the seeds of compact keys.
std::pair<std::vector<at::Tensor>, KeySwitchOptions> key_operands(
    const std::vector<KeySwitchKey>& keys, const KeySwitchOptions& options) {
  std::vector<at::Tensor> data;
  auto result = options;
  for (const auto& key : keys) {
    TORCH_CHECK_VALUE(key.expanded() == keys.front().expanded(),
                      "The keys of one key switch must share one form");
    data.push_back(key.data);
    if (!key.expanded()) {
      result.key_seeds.push_back(*key.seed_rows);
      result.key_rounds = key.spec.rounds;
    }
  }
  return {std::move(data), std::move(result)};
}

}  // namespace

at::Tensor key_switch_products(const at::Tensor& component,
                               const KeySwitchKey& key,
                               const TensorTable& tables,
                               const KeySwitchOptions& options,
                               const std::optional<at::Tensor>& accumulator) {
  const auto [data, keyed] = key_operands({key}, options);
  return key_switch_products(component, data[0], tables, keyed, accumulator);
}

at::Tensor key_switch(const at::Tensor& source,
                      const KeySwitchKey& key,
                      const TensorTable& tables,
                      const KeySwitchOptions& options,
                      bool conjugate) {
  const auto [data, keyed] = key_operands({key}, options);
  return key_switch(source, data[0], tables, keyed, conjugate);
}

at::Tensor relinearize(const at::Tensor& source,
                       const KeySwitchKey& key,
                       const TensorTable& tables,
                       const KeySwitchOptions& options,
                       const std::optional<JointRescaleOptions>& rescale_options) {
  const auto [data, keyed] = key_operands({key}, options);
  return relinearize(source, data[0], tables, keyed, rescale_options);
}

at::Tensor rotate(const at::Tensor& source,
                  const KeySwitchKey& key,
                  int64_t step,
                  const TensorTable& tables,
                  const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands({key}, options);
  return rotate(source, data[0], step, tables, keyed);
}

at::Tensor rotate_hoisted(const at::Tensor& c0,
                          const at::Tensor& hoisted_digits,
                          const KeySwitchKey& key,
                          int64_t step,
                          const TensorTable& tables,
                          const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands({key}, options);
  return rotate_hoisted(c0, hoisted_digits, data[0], step, tables, keyed);
}

std::vector<at::Tensor> rotate_many(const at::Tensor& source,
                                    const std::vector<KeySwitchKey>& keys,
                                    c10::IntArrayRef steps,
                                    const TensorTable& tables,
                                    const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands(keys, options);
  return rotate_many(source, data, steps, tables, keyed);
}

at::Tensor rotation_sum(at::TensorList sources,
                        const std::vector<KeySwitchKey>& keys,
                        c10::IntArrayRef steps,
                        const TensorTable& tables,
                        const KeySwitchOptions& options,
                        int64_t remaining_q_count) {
  const auto [data, keyed] = key_operands(keys, options);
  return rotation_sum(sources, data, steps, tables, keyed, remaining_q_count);
}

at::Tensor grouped_rotation_weighted_sum(const at::Tensor& source,
                                         const std::vector<KeySwitchKey>& keys,
                                         c10::IntArrayRef baby_steps,
                                         at::TensorList plaintexts,
                                         int64_t group_count,
                                         const TensorTable& tables,
                                         const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands(keys, options);
  return grouped_rotation_weighted_sum(
      source, data, baby_steps, plaintexts, group_count, tables, keyed);
}

at::Tensor encapsulated_modulus_raise(const at::Tensor& source,
                                      const KeySwitchKey& low_key,
                                      const KeySwitchKey& return_key,
                                      const TensorTable& tables,
                                      const KeySwitchOptions& low_options,
                                      int64_t return_drop_count,
                                      int64_t target_row_start,
                                      int64_t q_count) {
  const auto [data, keyed] = key_operands({low_key, return_key}, low_options);
  return encapsulated_modulus_raise(source, data[0], data[1], tables, keyed,
                                    return_drop_count, target_row_start,
                                    q_count);
}

}  // namespace fhelium::ckks
