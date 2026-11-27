#include "composition.h"

#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/ntt_tables.h>
#include <fhelium/rns.h>

#include <string>

namespace fhelium::detail_operations {

NttPolicy ntt_policy(std::string_view name) {
  if (name == "radix2_indexed") {
    return {NttFamily::kIndexed};
  }
  if (name == "radix2_compact" || name == "radix2_compact_shoup32") {
    return {NttFamily::kCompact};
  }
  if (name == "radix4_compact" || name == "radix8_compact" ||
      name == "radix16_compact") {
    return {NttFamily::kFixedRadix};
  }
  TORCH_CHECK_VALUE(false, "Unsupported NTT backend '", name, "'");
}

int64_t ntt_schedule_count(std::string_view name) {
  switch (ntt_policy(name).family) {
    case NttFamily::kIndexed:
      return 3;
    case NttFamily::kCompact:
      return 1;
    case NttFamily::kFixedRadix:
      return 2;
  }
  return 0;
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

at::Tensor forward_add_scaled(at::Tensor correction,
                              const at::Tensor& addend,
                              const at::Tensor& multiplier,
                              const at::Tensor& parameters,
                              const NttTables& forward,
                              std::string_view ntt_backend,
                              bool lazy_scaling,
                              const at::Tensor& out,
                              const at::Tensor& scaled_buffer) {
  const auto policy = ntt_policy(ntt_backend);
  if (policy.family == NttFamily::kCompact) {
    if (out.defined() && !out.is_same(correction)) {
      at::Tensor target = out;
      target.copy_(correction);
      correction = target;
    }
    ntt::forward_ntt_to_montgomery_compact_add_scaled_(
        correction, addend, multiplier, forward.schedule[0], parameters);
    return correction;
  }
  ntt::transform_(correction, forward, ntt_backend, "forward_to_montgomery");
  at::Tensor scaled;
  if (lazy_scaling) {
    if (scaled_buffer.defined()) {
      scaled = scaled_buffer;
      scaled.copy_(addend);
    } else {
      scaled = addend.clone();
    }
    rns::montgomery_mul_row_scalars_(scaled, multiplier, parameters);
  } else if (scaled_buffer.defined()) {
    scaled = scaled_buffer;
    rns::montgomery_mul_row_scalars_standard_out(
        addend, multiplier, parameters, scaled);
  } else {
    scaled = rns::montgomery_mul_row_scalars_standard(
        addend, multiplier, parameters);
  }
  if (out.defined()) {
    auto target = out;
    rns::add_lazy_out(scaled, correction, parameters, target);
    return target;
  }
  return rns::add_lazy(scaled, correction, parameters);
}

at::Tensor moddown_rows(const at::Tensor& q_rows,
                        const at::Tensor& p_rows,
                        const at::Tensor& inverses,
                        const at::Tensor& parameters,
                        const std::optional<at::Tensor>& half,
                        std::string_view rounding,
                        const at::Tensor& out,
                        const at::Tensor& p_scratch) {
  if (!out.defined() && !p_scratch.defined()) {
    return ckks::keyswitch_moddown_qp_to_q(
        q_rows, p_rows, inverses, parameters, half, rounding);
  }
  auto target = out.defined() ? out : at::empty_like(q_rows);
  ckks::keyswitch_moddown_qp_to_q_out(q_rows,
                                      p_rows,
                                      inverses,
                                      parameters,
                                      half,
                                      rounding,
                                      target,
                                      p_scratch.defined()
                                          ? std::optional<at::Tensor>(p_scratch)
                                          : std::nullopt);
  return target;
}

at::Tensor divide_trailing_rows(const at::Tensor& source,
                                const at::Tensor& inverses,
                                const at::Tensor& parameters,
                                int64_t drop_count,
                                const std::optional<at::Tensor>& half,
                                const std::optional<at::Tensor>& row_order,
                                std::string_view rounding,
                                const at::Tensor& out,
                                const at::Tensor& ordered_buffer,
                                const at::Tensor& p_scratch) {
  auto ordered = source;
  if (row_order.has_value()) {
    if (ordered_buffer.defined()) {
      ordered = ordered_buffer;
      at::index_select_out(ordered, source, -2, *row_order);
    } else {
      ordered = source.index_select(-2, *row_order);
    }
  }
  const auto rows = ordered.size(-2);
  return moddown_rows(ordered.narrow(-2, 0, rows - drop_count),
                      ordered.narrow(-2, rows - drop_count, drop_count),
                      inverses,
                      parameters,
                      half,
                      rounding,
                      out,
                      p_scratch);
}

bool uses_tiled_modup(const at::Tensor& source,
                      const at::Tensor& qp_parameters,
                      int64_t width) {
#if defined(FHELIUM_WITH_CUDA)
  const int64_t columns = source.scalar_type() == at::kInt ? 128 : 64;
  const auto destinations = qp_parameters.size(1);
  const auto tile_bytes = (width * columns + width + width * width + 4 * width +
                           (width + 8) * destinations) *
                          static_cast<int64_t>(source.element_size());
  return source.is_cuda() && width > 1 && tile_bytes <= 48 * 1024;
#else
  (void)source;
  (void)qp_parameters;
  (void)width;
  return false;
#endif
}

at::Tensor hybrid_modup_digit(const at::Tensor& source,
                              const at::Tensor& qp_parameters,
                              const DigitTables& digit,
                              const DigitSpan& span,
                              const at::Tensor& out,
                              const at::Tensor& scratch) {
  const auto width = span.stop - span.start;
  auto mixed = source.narrow(-2, span.start, width);
  const auto& parameters = qp_parameters;
  const auto destinations = parameters.size(1);
  const auto& extension = digit.extension;
#if defined(FHELIUM_WITH_CUDA)
  if (!scratch.defined() && uses_tiled_modup(source, parameters, width)) {
    const std::vector<at::Tensor> decomposition = {digit.normalizers,
                                                   digit.propagation,
                                                   digit.reduction[0],
                                                   digit.reduction[1],
                                                   digit.reduction[2],
                                                   digit.reduction[3]};
    if (!out.defined()) {
      return rns::mixed_radix_modup_tiled(
          mixed, decomposition, extension, parameters);
    }
    auto output = out;
    rns::mixed_radix_modup_tiled_out(
        mixed, decomposition, extension, parameters, output);
    return output;
  }
#endif
  if (width > 1) {
    const auto& normalizers = digit.normalizers;
    const auto& propagation = digit.propagation;
    const auto& lo = digit.reduction[0];
    const auto& hi = digit.reduction[1];
    const auto& neg_lo = digit.reduction[2];
    const auto& neg_hi = digit.reduction[3];
    if (!scratch.defined()) {
      mixed = rns::mixed_radix_decompose(
          mixed, normalizers, propagation, lo, hi, neg_lo, neg_hi);
    } else {
      auto decomposed = scratch.narrow(-2, 0, width);
      rns::mixed_radix_decompose_out(
          mixed, normalizers, propagation, lo, hi, neg_lo, neg_hi, decomposed);
      mixed = decomposed;
    }
  }
  if (!out.defined()) {
    return rns::mixed_radix_basis_extend_to_montgomery(
        mixed, extension, parameters, destinations);
  }
  auto output = out;
  rns::mixed_radix_basis_extend_to_montgomery_out(
      mixed, extension, parameters, destinations, output);
  return output;
}

at::Tensor lift_key_switch_digit(const at::Tensor& source,
                                 const KeySwitchTables& tables,
                                 std::string_view basis_conversion,
                                 int64_t digit_index,
                                 const DigitSpan& span,
                                 const at::Tensor& out,
                                 const at::Tensor& scratch) {
  TORCH_CHECK_VALUE(digit_index >= 0 && digit_index < static_cast<int64_t>(
                                                          tables.digits.size()),
                    "Key-switch tables hold ",
                    tables.digits.size(),
                    " digits; digit ",
                    digit_index,
                    " is missing");
  const auto& digit_tables = tables.digits[digit_index];
  if (basis_conversion == "exact") {
    return hybrid_modup_digit(
        source, tables.qp_parameters, digit_tables, span, out, scratch);
  }
  TORCH_CHECK_VALUE(
      basis_conversion == "crt" || basis_conversion == "normalized_crt",
      "basis_conversion must be exact, crt, or normalized_crt");
  const auto& crt = digit_tables.crt;
  const auto digit = source.narrow(-2, span.start, span.stop - span.start);
  const auto optional = [](const at::Tensor& tensor) {
    return tensor.defined() && tensor.numel() != 0
               ? std::optional<at::Tensor>(tensor)
               : std::nullopt;
  };
  const auto source_normalizers = optional(crt.source_normalizers);
  const bool normalized = source_normalizers.has_value();
  const auto source_reciprocals =
      normalized ? optional(crt.source_reciprocals) : std::nullopt;
  const auto target_offsets =
      normalized ? optional(crt.target_offsets) : std::nullopt;
  if (!out.defined()) {
    return rns::crt_basis_extend(digit,
                                 crt.source_parameters,
                                 crt.target_parameters,
                                 crt.matrix,
                                 crt.target_reciprocals,
                                 source_normalizers,
                                 source_reciprocals,
                                 target_offsets);
  }
  auto output = out;
  rns::crt_basis_extend_out(digit,
                            crt.source_parameters,
                            crt.target_parameters,
                            crt.matrix,
                            crt.target_reciprocals,
                            source_normalizers,
                            source_reciprocals,
                            target_offsets,
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

}  // namespace

void transform_(at::Tensor& residues,
                const NttTables& tables,
                std::string_view ntt_backend,
                std::string_view transition_name) {
  const auto transition = parse_transition(transition_name);
  const auto& parameters = tables.parameters;
  const auto& twiddles = tables.schedule[0];
  const auto policy = detail_operations::ntt_policy(ntt_backend);
  switch (policy.family) {
    case NttFamily::kIndexed: {
      const auto& even = tables.schedule[1];
      const auto& odd = tables.schedule[2];
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
      switch (transition) {
        case Transition::kForwardMontgomery:
          return forward_ntt_montgomery_compact_(
              residues, twiddles, parameters);
        case Transition::kForwardToMontgomery:
          return forward_ntt_to_montgomery_compact_(
              residues, twiddles, parameters);
        case Transition::kInverseMontgomery:
          return inverse_ntt_montgomery_compact_(
              residues, twiddles, parameters);
        case Transition::kInverseToStandardLazy:
          return inverse_ntt_to_standard_lazy_compact_(
              residues, twiddles, parameters);
        case Transition::kInverseToStandard:
          return inverse_ntt_to_standard_compact_(
              residues, twiddles, parameters);
        case Transition::kInverseToCentered:
          return inverse_ntt_to_centered_compact_(
              residues, twiddles, parameters);
      }
      break;
    }
    case NttFamily::kFixedRadix: {
      const auto& roots = tables.schedule[1];
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

void transform_out(const at::Tensor& residues,
                   const NttTables& tables,
                   std::string_view ntt_backend,
                   std::string_view transition_name,
                   at::Tensor& out) {
  if (parse_transition(transition_name) == Transition::kForwardToMontgomery) {
    const auto& parameters = tables.parameters;
    const auto& twiddles = tables.schedule[0];
    const auto policy = detail_operations::ntt_policy(ntt_backend);
    switch (policy.family) {
      case NttFamily::kIndexed:
        forward_ntt_to_montgomery_indexed_out(residues,
                                              tables.schedule[1],
                                              tables.schedule[2],
                                              twiddles,
                                              parameters,
                                              out);
        return;
      case NttFamily::kCompact:
        forward_ntt_to_montgomery_compact_out(
            residues, twiddles, parameters, out);
        return;
      case NttFamily::kFixedRadix:
        forward_ntt_to_montgomery_power_of_two_radix_compact_out(
            residues, twiddles, tables.schedule[1], parameters, out);
        return;
    }
  }
  out.copy_(residues);
  transform_(out, tables, ntt_backend, transition_name);
}

at::Tensor transform(const at::Tensor& residues,
                     const NttTables& tables,
                     std::string_view ntt_backend,
                     std::string_view transition_name) {
  auto output = at::empty_like(residues);
  transform_out(residues, tables, ntt_backend, transition_name, output);
  return output;
}

}  // namespace fhelium::ntt
