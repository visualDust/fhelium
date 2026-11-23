#include "composition.h"

#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/ntt_tables.h>
#include <fhelium/rns.h>

#include <map>
#include <mutex>
#include <string>
#include <tuple>

namespace fhelium::detail_operations {

namespace {

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

int64_t log2_exact(int64_t value) {
  int64_t bits = 0;
  while ((int64_t{1} << bits) < value) {
    ++bits;
  }
  return bits;
}

std::string operand(std::string_view basis,
                    std::string_view direction,
                    int index) {
  std::string name(basis);
  if (!name.empty()) {
    name.append("_");
  }
  name.append(direction).append("_").append(std::to_string(index));
  return name;
}

using CacheKey = std::tuple<int64_t, int64_t, int, int>;

CacheKey cache_key(int64_t n, int64_t g, const at::Device& device) {
  return {
      n, g, static_cast<int>(device.type()), static_cast<int>(device.index())};
}

}  // namespace

NttPolicy ntt_policy(std::string_view name) {
  if (name == "radix2_indexed") {
    return {NttFamily::kIndexed, 0};
  }
  constexpr std::string_view compact = "radix2_compact_group";
  if (starts_with(name, compact)) {
    const auto rest = name.substr(compact.size());
    const auto width = rest.substr(0, rest.find('_'));
    if (width == "4" || width == "8" || width == "16") {
      return {NttFamily::kCompact, log2_exact(std::stoll(std::string(width)))};
    }
  }
  if (name == "radix4_compact" || name == "radix8_compact" ||
      name == "radix16_compact") {
    return {NttFamily::kFixedRadix, 0};
  }
  TORCH_CHECK_VALUE(false, "Unsupported NTT backend '", name, "'");
}

std::vector<DigitSpan> digit_spans(c10::IntArrayRef flat) {
  TORCH_CHECK_VALUE(flat.size() % 3 == 0,
                    "digits must list (start, stop, key_digit) triples");
  std::vector<DigitSpan> spans;
  for (size_t i = 0; i < flat.size(); i += 3) {
    spans.push_back({flat[i], flat[i + 1], flat[i + 2]});
  }
  return spans;
}

const at::Tensor& ntt_galois_indices(int64_t ring_dimension,
                                     int64_t galois_element,
                                     const at::Device& device) {
  static std::mutex mutex;
  static std::map<CacheKey, at::Tensor> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto key = cache_key(ring_dimension, galois_element, device);
  auto entry = cache.find(key);
  if (entry == cache.end()) {
    entry = cache
                .emplace(key,
                         ntt::galois_indices(log2_exact(ring_dimension),
                                             galois_element)
                             .to(device))
                .first;
  }
  return entry->second;
}

const std::pair<at::Tensor, at::Tensor>& coefficient_galois_indices(
    int64_t ring_dimension, int64_t galois_element, const at::Device& device) {
  static std::mutex mutex;
  static std::map<CacheKey, std::pair<at::Tensor, at::Tensor>> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto key = cache_key(ring_dimension, galois_element, device);
  auto entry = cache.find(key);
  if (entry == cache.end()) {
    auto [indices, signs] = ckks::coefficient_galois_indices(
        log2_exact(ring_dimension), galois_element);
    entry =
        cache.emplace(key, std::make_pair(indices.to(device), signs.to(device)))
            .first;
  }
  return entry->second;
}

at::Tensor forward_add_scaled(at::Tensor correction,
                              const at::Tensor& addend,
                              const at::Tensor& multiplier,
                              const at::Tensor& parameters,
                              const TensorTable& tables,
                              std::string_view ntt_backend,
                              std::string_view basis,
                              bool lazy_scaling) {
  const auto policy = ntt_policy(ntt_backend);
  if (policy.family == NttFamily::kCompact) {
    ntt::forward_ntt_to_montgomery_compact_add_scaled_(
        correction,
        addend,
        multiplier,
        tables.at(operand(basis, "forward", 1)),
        parameters,
        policy.grouped_stage_count);
    return correction;
  }
  ntt::transform_(
      correction, tables, ntt_backend, basis, "forward_to_montgomery");
  at::Tensor scaled;
  if (lazy_scaling) {
    scaled = addend.clone();
    rns::montgomery_mul_row_scalars_(scaled, multiplier, parameters);
  } else {
    scaled = rns::montgomery_mul_row_scalars_standard(
        addend, multiplier, parameters);
  }
  return rns::add_lazy(scaled, correction, parameters);
}

at::Tensor divide_trailing_rows(const at::Tensor& source,
                                const at::Tensor& inverses,
                                const at::Tensor& parameters,
                                int64_t drop_count,
                                const std::optional<at::Tensor>& half,
                                const std::optional<at::Tensor>& row_order,
                                std::string_view rounding) {
  const auto ordered =
      row_order.has_value() ? source.index_select(-2, *row_order) : source;
  const auto rows = ordered.size(-2);
  return ckks::keyswitch_moddown_qp_to_q(
      ordered.narrow(-2, 0, rows - drop_count),
      ordered.narrow(-2, rows - drop_count, drop_count),
      inverses,
      parameters,
      half,
      rounding);
}

at::Tensor hybrid_modup_digit(const at::Tensor& source,
                              const TensorTable& tables,
                              int64_t digit_index,
                              const DigitSpan& span,
                              const at::Tensor& out,
                              const at::Tensor& scratch) {
  const auto prefix = "digit" + std::to_string(digit_index) + "_";
  const auto width = span.stop - span.start;
  auto mixed = source.narrow(-2, span.start, width);
  const auto& parameters = tables.at("qp_parameters");
  const auto destinations = parameters.size(1);
  const int64_t columns = source.scalar_type() == at::kInt ? 128 : 64;
  const auto tile_bytes = (width * columns + width + width * width + 4 * width +
                           (width + 8) * destinations) *
                          static_cast<int64_t>(source.element_size());
  const auto& extension = tables.at(prefix + "extension");
#if defined(FHELIUM_WITH_CUDA)
  if (source.is_cuda() && width > 1 && !scratch.defined() &&
      tile_bytes <= 48 * 1024) {
    const std::vector<at::Tensor> decomposition = {
        tables.at(prefix + "normalizers"),
        tables.at(prefix + "propagation"),
        tables.at(prefix + "reduction0"),
        tables.at(prefix + "reduction1"),
        tables.at(prefix + "reduction2"),
        tables.at(prefix + "reduction3")};
    if (!out.defined()) {
      return rns::mixed_radix_modup_tiled(
          mixed, decomposition, extension, parameters);
    }
    auto output = out;
    rns::mixed_radix_modup_tiled_out_(
        mixed, decomposition, extension, parameters, output);
    return output;
  }
#else
  (void)tile_bytes;
#endif
  if (width > 1) {
    const auto& normalizers = tables.at(prefix + "normalizers");
    const auto& propagation = tables.at(prefix + "propagation");
    const auto& lo = tables.at(prefix + "reduction0");
    const auto& hi = tables.at(prefix + "reduction1");
    const auto& neg_lo = tables.at(prefix + "reduction2");
    const auto& neg_hi = tables.at(prefix + "reduction3");
    if (!scratch.defined()) {
      mixed = rns::mixed_radix_decompose(
          mixed, normalizers, propagation, lo, hi, neg_lo, neg_hi);
    } else {
      auto decomposed = scratch.narrow(-2, 0, width);
      rns::mixed_radix_decompose_out_(
          mixed, normalizers, propagation, lo, hi, neg_lo, neg_hi, decomposed);
      mixed = decomposed;
    }
  }
  if (!out.defined()) {
    return rns::mixed_radix_basis_extend_to_montgomery(
        mixed, extension, parameters, destinations);
  }
  auto output = out;
  rns::mixed_radix_basis_extend_to_montgomery_out_(
      mixed, extension, parameters, destinations, output);
  return output;
}

at::Tensor lift_key_switch_digit(const at::Tensor& source,
                                 const TensorTable& tables,
                                 std::string_view modup_method,
                                 int64_t digit_index,
                                 const DigitSpan& span,
                                 const at::Tensor& out,
                                 const at::Tensor& scratch) {
  if (modup_method == "exact") {
    return hybrid_modup_digit(source, tables, digit_index, span, out, scratch);
  }
  TORCH_CHECK_VALUE(modup_method == "crt" || modup_method == "normalized_crt",
                    "modup_method must be exact, crt, or normalized_crt");
  const auto prefix = "digit" + std::to_string(digit_index) + "_crt_";
  const auto digit = source.narrow(-2, span.start, span.stop - span.start);
  const auto& normalizers = tables.at(prefix + "source_normalizers");
  const bool normalized = normalizers.numel() != 0;
  const std::optional<at::Tensor> source_normalizers =
      normalized ? std::optional<at::Tensor>(normalizers) : std::nullopt;
  const std::optional<at::Tensor> source_reciprocals =
      normalized ? tables.get(prefix + "source_reciprocals") : std::nullopt;
  if (!out.defined()) {
    return rns::crt_basis_extend(digit,
                                 tables.at(prefix + "source_parameters"),
                                 tables.at(prefix + "target_parameters"),
                                 tables.at(prefix + "matrix"),
                                 tables.at(prefix + "target_reciprocals"),
                                 source_normalizers,
                                 source_reciprocals);
  }
  auto output = out;
  rns::crt_basis_extend_out_(digit,
                             tables.at(prefix + "source_parameters"),
                             tables.at(prefix + "target_parameters"),
                             tables.at(prefix + "matrix"),
                             tables.at(prefix + "target_reciprocals"),
                             source_normalizers,
                             source_reciprocals,
                             output);
  return output;
}

}  // namespace fhelium::detail_operations

namespace fhelium::ntt {

namespace {

using detail_operations::NttFamily;

enum class Transition {
  kForwardMontgomery,
  kForwardToMontgomery,
  kInverseMontgomery,
  kInverseToStandardLazy,
  kInverseToStandard,
  kInverseToCentered,
};

Transition parse_transition(std::string_view name) {
  if (name == "forward_montgomery") {
    return Transition::kForwardMontgomery;
  }
  if (name == "forward_to_montgomery") {
    return Transition::kForwardToMontgomery;
  }
  if (name == "inverse_montgomery") {
    return Transition::kInverseMontgomery;
  }
  if (name == "inverse_to_standard_lazy") {
    return Transition::kInverseToStandardLazy;
  }
  if (name == "inverse_to_standard") {
    return Transition::kInverseToStandard;
  }
  if (name == "inverse_to_centered") {
    return Transition::kInverseToCentered;
  }
  TORCH_CHECK_VALUE(false, "Unsupported NTT transition '", name, "'");
}

bool is_forward(Transition transition) {
  return transition == Transition::kForwardMontgomery ||
         transition == Transition::kForwardToMontgomery;
}

std::string operand(std::string_view basis, bool forward, int index) {
  std::string name(basis);
  if (!name.empty()) {
    name.append("_");
  }
  name.append(forward ? "forward_" : "inverse_").append(std::to_string(index));
  return name;
}

}  // namespace

void transform_(at::Tensor& residues,
                const TensorTable& tables,
                std::string_view ntt_backend,
                std::string_view basis,
                std::string_view transition_name) {
  const auto transition = parse_transition(transition_name);
  const bool forward = is_forward(transition);
  const auto& parameters = tables.at(operand(basis, forward, 0));
  const auto& twiddles = tables.at(operand(basis, forward, 1));
  const auto policy = detail_operations::ntt_policy(ntt_backend);
  switch (policy.family) {
    case NttFamily::kIndexed: {
      const auto& even = tables.at(operand(basis, forward, 2));
      const auto& odd = tables.at(operand(basis, forward, 3));
      switch (transition) {
        case Transition::kForwardMontgomery:
          return forward_ntt_montgomery_indexed_(
              residues, even, odd, twiddles, parameters);
        case Transition::kForwardToMontgomery:
          return forward_ntt_to_montgomery_indexed_(
              residues, even, odd, twiddles, parameters);
        case Transition::kInverseMontgomery:
          return inverse_ntt_montgomery_indexed_(
              residues, even, odd, twiddles, parameters);
        case Transition::kInverseToStandardLazy:
          return inverse_ntt_to_standard_lazy_indexed_(
              residues, even, odd, twiddles, parameters);
        case Transition::kInverseToStandard:
          return inverse_ntt_to_standard_indexed_(
              residues, even, odd, twiddles, parameters);
        case Transition::kInverseToCentered:
          return inverse_ntt_to_centered_indexed_(
              residues, even, odd, twiddles, parameters);
      }
      break;
    }
    case NttFamily::kCompact: {
      const auto group = policy.grouped_stage_count;
      switch (transition) {
        case Transition::kForwardMontgomery:
          return forward_ntt_montgomery_compact_grouped_smem_(
              residues, twiddles, parameters, group);
        case Transition::kForwardToMontgomery:
          return forward_ntt_to_montgomery_compact_grouped_smem_(
              residues, twiddles, parameters, group);
        case Transition::kInverseMontgomery:
          return inverse_ntt_montgomery_compact_grouped_smem_(
              residues, twiddles, parameters, group);
        case Transition::kInverseToStandardLazy:
          return inverse_ntt_to_standard_lazy_compact_grouped_smem_(
              residues, twiddles, parameters, group);
        case Transition::kInverseToStandard:
          return inverse_ntt_to_standard_compact_grouped_smem_(
              residues, twiddles, parameters, group);
        case Transition::kInverseToCentered:
          return inverse_ntt_to_centered_compact_grouped_smem_(
              residues, twiddles, parameters, group);
      }
      break;
    }
    case NttFamily::kFixedRadix: {
      const auto& roots = tables.at(operand(basis, forward, 2));
      switch (transition) {
        case Transition::kForwardMontgomery:
          return forward_ntt_montgomery_power_of_two_radix_compact_(
              residues, twiddles, roots, parameters);
        case Transition::kForwardToMontgomery:
          return forward_ntt_to_montgomery_power_of_two_radix_compact_(
              residues, twiddles, roots, parameters);
        case Transition::kInverseMontgomery:
          return inverse_ntt_montgomery_power_of_two_radix_compact_(
              residues, twiddles, roots, parameters);
        case Transition::kInverseToStandardLazy:
          return inverse_ntt_to_standard_lazy_power_of_two_radix_compact_(
              residues, twiddles, roots, parameters);
        case Transition::kInverseToStandard:
          return inverse_ntt_to_standard_power_of_two_radix_compact_(
              residues, twiddles, roots, parameters);
        case Transition::kInverseToCentered:
          return inverse_ntt_to_centered_power_of_two_radix_compact_(
              residues, twiddles, roots, parameters);
      }
      break;
    }
  }
}

at::Tensor transform(const at::Tensor& residues,
                     const TensorTable& tables,
                     std::string_view ntt_backend,
                     std::string_view basis,
                     std::string_view transition_name) {
  if (parse_transition(transition_name) == Transition::kForwardToMontgomery) {
    const auto& parameters = tables.at(operand(basis, true, 0));
    const auto& twiddles = tables.at(operand(basis, true, 1));
    const auto policy = detail_operations::ntt_policy(ntt_backend);
    switch (policy.family) {
      case NttFamily::kIndexed:
        return forward_ntt_to_montgomery_indexed(
            residues,
            tables.at(operand(basis, true, 2)),
            tables.at(operand(basis, true, 3)),
            twiddles,
            parameters);
      case NttFamily::kCompact:
        return forward_ntt_to_montgomery_compact_grouped_smem(
            residues, twiddles, parameters, policy.grouped_stage_count);
      case NttFamily::kFixedRadix:
        return forward_ntt_to_montgomery_power_of_two_radix_compact(
            residues, twiddles, tables.at(operand(basis, true, 2)), parameters);
    }
  }
  auto output = residues.clone();
  transform_(output, tables, ntt_backend, basis, transition_name);
  return output;
}

}  // namespace fhelium::ntt
