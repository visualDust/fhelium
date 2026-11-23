#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_operations.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/random.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include <vector>

#include "composition.h"

namespace fhelium::ckks {

namespace {

// Advances the nonce words that follow the four key words of a seed.
void advance(at::Tensor& seed) { seed.narrow(0, 4, seed.size(0) - 4).add_(1); }

at::Tensor stochastic_round(const at::Tensor& values,
                            const at::Tensor& seed,
                            int64_t word_stride = 1) {
  return random::stochastic_round(seed, values.to(at::kDouble), word_stride);
}

std::vector<int64_t> modulus_view(int64_t leading) {
  std::vector<int64_t> shape(leading, 1);
  shape.push_back(-1);
  shape.push_back(1);
  return shape;
}

at::Tensor inverse_embed_slots(const at::Tensor& message,
                               const at::Tensor& pre,
                               const at::Tensor& twister) {
  const auto ring_dimension = message.size(-1) * 2;
  auto shape = message.sizes().vec();
  shape.back() = ring_dimension;
  auto permuted = at::zeros(shape, message.options());
  permuted.index_put_({at::indexing::Ellipsis, pre}, message);
  const auto symmetric = permuted + permuted.conj().flip({-1});
  const auto transformed = at::fft_fft(symmetric, std::nullopt, -1, "forward");
  return at::real(transformed * twister);
}

// Short indexed NTT on the RNS primitives for transforms below the native
// minimum of 256 coefficients.
at::Tensor forward_small_to_montgomery(const at::Tensor& operand,
                                       const at::Tensor& even_indices,
                                       const at::Tensor& odd_indices,
                                       const at::Tensor& twiddles,
                                       const at::Tensor& parameters) {
  auto result = operand.clone();
  rns::to_montgomery_(result, parameters);
  for (int64_t stage = 0; stage < even_indices.size(0); ++stage) {
    const auto even = even_indices.select(0, stage).to(at::kLong);
    const auto odd = odd_indices.select(0, stage).to(at::kLong);
    const auto a = result.index_select(-1, even);
    const auto b = rns::montgomery_mul(result.index_select(-1, odd),
                                       twiddles.select(1, stage).contiguous(),
                                       parameters);
    result.index_copy_(-1, even, rns::add_standard(a, b, parameters));
    result.index_copy_(-1, odd, rns::sub_standard(a, b, parameters));
  }
  return result;
}

}  // namespace

at::Tensor encapsulated_modulus_raise(const at::Tensor& source,
                                      const at::Tensor& low_key,
                                      const at::Tensor& return_key,
                                      const TensorTable& tables,
                                      const KeySwitchOptions& low_options,
                                      int64_t return_drop_count,
                                      int64_t target_row_start,
                                      int64_t q_count) {
  const auto low_tables = tables.with_prefix("low_");
  auto low = low_options;
  low.output_domain = "coefficient";
  const auto switched = key_switch(source, low_key, low_tables, low, false);
  const auto raised =
      rns::centered_basis_extend(switched, tables.with_prefix("raise_"));
  auto component = raised.select(0, 1);
  ntt::transform_(component,
                  tables,
                  low_options.ntt_backend,
                  "return",
                  "forward_to_montgomery");
  auto shape = component.sizes().vec();
  shape.insert(shape.begin(), 2);
  auto products = at::zeros(shape, component.options());
  auto products0 = products.select(0, 0);
  auto products1 = products.select(0, 1);
  keyswitch_accumulate_digit_products_(products0,
                                       products1,
                                       component,
                                       return_key.select(0, 0),
                                       tables.at("return_parameters"),
                                       0);
  ntt::transform_(products,
                  tables,
                  low_options.ntt_backend,
                  "return",
                  "inverse_to_standard");
  const auto quotient =
      detail_operations::divide_trailing_rows(products,
                                              tables.at("quotient_inverses"),
                                              tables.at("quotient_parameters"),
                                              return_drop_count,
                                              tables.get("quotient_half"),
                                              tables.at("quotient_order"),
                                              low_options.moddown_rounding);
  auto output =
      quotient
          .narrow(-2, target_row_start, quotient.size(-2) - target_row_start)
          .contiguous();
  auto output0 = output.select(0, 0);
  rns::add_standard_(output0,
                     raised.select(0, 0).narrow(
                         -2, target_row_start + 1, q_count - target_row_start),
                     tables.at("target_parameters"));
  return output;
}

namespace {

at::Tensor add_compressed(const at::Tensor& ciphertext,
                          const at::Tensor& compressed,
                          const std::optional<at::Tensor>& implicit,
                          const at::Tensor& parameters,
                          std::string_view layout,
                          std::string_view polynomial_domain,
                          bool in_place) {
  auto plaintext = compressed;
  auto implicit_rows = implicit;
  if (polynomial_domain == "ntt") {
    plaintext = compressed.clone();
    rns::to_montgomery_(plaintext, parameters);
    if (implicit_rows.has_value()) {
      implicit_rows = implicit_rows->clone();
      rns::to_montgomery_(*implicit_rows, parameters);
    }
  }
  auto output = in_place ? ciphertext : ciphertext.clone();
  auto component = output.select(0, 0);
  if (layout == "cyclic") {
    add_cyclic_compressed_plaintext_component_(
        component, plaintext, parameters);
  } else if (layout == "contiguous") {
    add_contiguous_compressed_plaintext_component_(
        component, plaintext, parameters);
  } else if (layout == "strided_sparse" && implicit_rows.has_value()) {
    auto rows = implicit_rows->squeeze(-1);
    if (rows.device().is_cpu()) {
      rows = rows.reshape({-1, rows.size(-1)}).contiguous();
    }
    add_strided_plaintext_component_(component, plaintext, rows, parameters);
  } else {
    TORCH_CHECK_VALUE(false,
                      "Compressed addition requires a supported layout and "
                      "its implicit rows");
  }
  return output;
}

at::Tensor multiply_compressed(const at::Tensor& ciphertext,
                               const at::Tensor& compressed,
                               const std::optional<at::Tensor>& implicit,
                               const at::Tensor& parameters,
                               std::string_view layout) {
  const auto multiply = [&](const at::Tensor& component) {
    if (layout == "cyclic") {
      return rns::montgomery_mul_cyclic_compressed(
          component, compressed, parameters);
    }
    if (layout == "contiguous") {
      return rns::montgomery_mul_contiguous_compressed(
          component, compressed, parameters);
    }
    TORCH_CHECK_VALUE(layout == "strided_sparse" && implicit.has_value(),
                      "Compressed multiplication requires a supported layout "
                      "and its implicit rows");
    auto result =
        rns::montgomery_mul_cyclic_compressed(component, *implicit, parameters);
    const auto repeat = component.size(-1) / compressed.size(-1);
    using at::indexing::Ellipsis;
    using at::indexing::Slice;
    result
        .index(
            {Ellipsis, Slice(at::indexing::None, at::indexing::None, repeat)})
        .copy_(rns::montgomery_mul(
            component.index(
                {Ellipsis,
                 Slice(at::indexing::None, at::indexing::None, repeat)}),
            compressed,
            parameters));
    return result;
  };
  int64_t leading = 1;
  for (int64_t i = 0; i + 2 < compressed.dim(); ++i) {
    leading *= compressed.size(i);
  }
  at::Tensor product;
  if (leading == 1) {
    product = multiply(ciphertext);
  } else {
    std::vector<at::Tensor> components;
    for (const auto& component : ciphertext.unbind(0)) {
      components.push_back(multiply(component));
    }
    product = at::stack(components);
  }
  return product;
}

}  // namespace

at::Tensor add_compressed_plaintext(const at::Tensor& ciphertext,
                                    const at::Tensor& compressed,
                                    const std::optional<at::Tensor>& implicit,
                                    const at::Tensor& parameters,
                                    std::string_view layout,
                                    std::string_view polynomial_domain) {
  return add_compressed(ciphertext,
                        compressed,
                        implicit,
                        parameters,
                        layout,
                        polynomial_domain,
                        false);
}

at::Tensor& add_compressed_plaintext_(at::Tensor& ciphertext,
                                      const at::Tensor& compressed,
                                      const std::optional<at::Tensor>& implicit,
                                      const at::Tensor& parameters,
                                      std::string_view layout,
                                      std::string_view polynomial_domain) {
  add_compressed(ciphertext,
                 compressed,
                 implicit,
                 parameters,
                 layout,
                 polynomial_domain,
                 true);
  return ciphertext;
}

at::Tensor multiply_compressed_plaintext(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout) {
  return multiply_compressed(
      ciphertext, compressed, implicit, parameters, layout);
}

at::Tensor& multiply_compressed_plaintext_(
    at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout) {
  ciphertext.copy_(multiply_compressed(
      ciphertext, compressed, implicit, parameters, layout));
  return ciphertext;
}

at::Tensor scalar_arithmetic(const at::Tensor& ciphertext,
                             const at::Tensor& parameters,
                             const at::Tensor& seed,
                             double scalar,
                             double scalar_scale,
                             bool add) {
  const auto scaled = scalar * scalar_scale;
  const auto coefficient = stochastic_round(
      at::full({1}, scaled, ciphertext.options().dtype(at::kDouble)), seed);
  auto residues =
      rns::lift_integer_coefficients(coefficient, parameters.select(0, 0));
  if (add) {
    auto output = ciphertext.clone();
    auto constant = output.select(0, 0).narrow(-1, 0, 1);
    rns::add_standard_(constant, residues, parameters);
    return output;
  }
  rns::to_montgomery_(residues, parameters);
  return rns::montgomery_mul_row_scalars_standard(
      ciphertext, residues.squeeze(-1), parameters);
}

at::Tensor encode(const at::Tensor& message,
                  const at::Tensor& pre,
                  const at::Tensor& twister,
                  const at::Tensor& seed,
                  double scale) {
  const auto slots = pre.numel();
  const auto full =
      message.dim() == 0 ? message.expand({slots}).contiguous() : message;
  TORCH_CHECK_VALUE(full.size(-1) <= slots,
                    "Message slot extent ",
                    full.size(-1),
                    " exceeds ",
                    slots);
  const auto padded =
      full.size(-1) < slots
          ? at::constant_pad_nd(full, {0, slots - full.size(-1)}, 0)
          : full;
  const auto coefficients = inverse_embed_slots(padded, pre, twister);
  return stochastic_round(coefficients * scale, seed);
}

at::Tensor decode(const at::Tensor& coefficients,
                  const at::Tensor& post,
                  const at::Tensor& skewer,
                  double scale,
                  bool is_real) {
  const auto recovered =
      at::fft_ifft(coefficients * skewer, std::nullopt, -1, "forward");
  auto result = at::zeros_like(recovered);
  result.index_put_({at::indexing::Ellipsis, post}, recovered);
  const auto decoded =
      (result / scale).narrow(-1, 0, coefficients.size(-1) / 2);
  return is_real ? at::real(decoded) : decoded;
}

std::vector<at::Tensor> prepare_periodic_plaintext(
    const at::Tensor& message,
    const at::Tensor& pre,
    const at::Tensor& twister,
    const at::Tensor& seed,
    const at::Tensor& parameters,
    const at::Tensor& twice_moduli,
    const at::Tensor& even,
    const at::Tensor& odd,
    const at::Tensor& twiddles,
    int64_t ring_dimension,
    double scale,
    std::string_view polynomial_domain) {
  const auto slots = message.size(-1) == pre.numel()
                         ? message
                         : message.repeat_interleave(pre.numel(), -1);
  const auto coefficients = inverse_embed_slots(slots, pre, twister);
  const auto rounded = stochastic_round(
      coefficients * scale, seed, ring_dimension / coefficients.size(-1));
  auto residues = rns::lift_integer_coefficients(rounded, twice_moduli);
  if (polynomial_domain == "coefficient") {
    rns::to_montgomery_(residues, parameters);
    auto shape = residues.sizes().vec();
    shape.back() = 1;
    return {residues, at::zeros(shape, residues.options())};
  }
  if (residues.size(-1) < 256) {
    return {
        forward_small_to_montgomery(residues, even, odd, twiddles, parameters)};
  }
  return {ntt::forward_ntt_to_montgomery_indexed(
      residues, even, odd, twiddles, parameters)};
}

at::Tensor encrypt(const at::Tensor& coefficients,
                   const at::Tensor& public_key,
                   const TensorTable& tables,
                   const at::Tensor& seed,
                   const at::Tensor& cdt_table,
                   std::string_view ntt_backend,
                   int64_t key_row_start,
                   std::string_view output_domain) {
  const auto& parameters = tables.at("parameters");
  const auto shape = coefficients.sizes().vec();
  auto stream = seed.clone();
  const auto sample = [&](const std::optional<at::Tensor>& rows) {
    auto out = random::sample_discrete_gaussian(stream, shape, cdt_table, rows);
    advance(stream);
    return out;
  };
  const auto plaintext = rns::lift_integer_coefficients(
      coefficients + sample(std::nullopt), parameters.select(0, 0));
  const auto keys = public_key.narrow(1, key_row_start, parameters.size(1));
  auto v = random::sample_ternary(stream, shape, 0.5, parameters);
  advance(stream);
  ntt::transform_(v, tables, ntt_backend, "", "forward_to_montgomery");
  auto products =
      at::stack({rns::montgomery_mul(v, keys.select(0, 0), parameters),
                 rns::montgomery_mul(v, keys.select(0, 1), parameters)});
  if (output_domain == "ntt") {
    auto added = at::stack({plaintext, sample(parameters)});
    ntt::transform_(added, tables, ntt_backend, "", "forward_to_montgomery");
    return rns::add_lazy(products, added, parameters);
  }
  TORCH_CHECK_VALUE(output_domain == "coefficient",
                    "Encryption output_domain is unsupported");
  ntt::transform_(
      products, tables, ntt_backend, "", "inverse_to_standard_lazy");
  auto component0 = products.select(0, 0);
  rns::add_standard_(component0, plaintext, parameters);
  auto component1 = products.select(0, 1);
  random::add_discrete_gaussian_(component1, stream, cdt_table, parameters);
  return products;
}

at::Tensor decrypt_phase(const at::Tensor& ciphertext,
                         const at::Tensor& secret_key,
                         const TensorTable& tables,
                         std::string_view ntt_backend,
                         int64_t key_row_start,
                         std::string_view input_domain) {
  const auto& parameters = tables.at("parameters");
  const auto secret = secret_key.narrow(0, key_row_start, parameters.size(1));
  const auto components = ciphertext.size(0);
  TORCH_CHECK_VALUE(components == 2 || components == 3,
                    "Decryption requires a CT2 or CT3 payload");
  TORCH_CHECK_VALUE(input_domain == "coefficient" || input_domain == "ntt",
                    "Decryption requires a coefficient or NTT input");
  const bool coefficient = input_domain == "coefficient";
  auto phase = ciphertext.select(0, 0).clone();
  auto power = secret;
  for (int64_t k = 1; k < components; ++k) {
    auto value = ciphertext.select(0, k);
    if (coefficient) {
      value = ntt::transform(
          value, tables, ntt_backend, "", "forward_to_montgomery");
    }
    auto product = rns::montgomery_mul(value, power, parameters);
    if (coefficient) {
      ntt::transform_(product, tables, ntt_backend, "", "inverse_to_standard");
      phase = rns::add_standard(phase, product, parameters);
    } else {
      phase = rns::add_lazy(phase, product, parameters);
    }
    if (k + 1 < components) {
      power = rns::montgomery_mul(power, secret, parameters);
    }
  }
  if (!coefficient) {
    ntt::transform_(phase, tables, ntt_backend, "", "inverse_to_standard");
  }
  return phase;
}

at::Tensor reconstruct_q_coefficients(const at::Tensor& residues,
                                      const at::Tensor& source_parameters,
                                      const at::Tensor& normalizers,
                                      const at::Tensor& propagation,
                                      const at::Tensor& half_digits) {
  const auto count = source_parameters.size(1);
  auto source = residues.narrow(-2, 0, count).clone();
  rns::reduce_to_standard_(source, source_parameters);
  const auto moduli = at::floor_divide(source_parameters.select(0, 0), 2);
  if (count == 1) {
    const auto modulus = moduli[0];
    const auto centered = source.squeeze(-2);
    return at::where(centered > at::floor_divide(modulus, 2),
                     centered - modulus,
                     centered)
        .to(at::kDouble);
  }
  const auto decompose = [&](const at::Tensor& values) {
    auto digits = rns::mixed_radix_decompose(values,
                                             normalizers,
                                             propagation,
                                             source_parameters[1],
                                             source_parameters[2],
                                             source_parameters[3],
                                             source_parameters[4]);
    rns::reduce_to_standard_(digits, source_parameters);
    return digits;
  };
  const auto positive_digits = decompose(source);
  auto equal = at::ones_like(positive_digits.select(-2, 0),
                             positive_digits.options().dtype(at::kBool));
  auto negative = at::zeros_like(equal);
  for (int64_t row = half_digits.numel() - 1; row >= 0; --row) {
    const auto digit = positive_digits.select(-2, row);
    negative.bitwise_or_(equal.bitwise_and(digit > half_digits[row]));
    equal.bitwise_and_(digit == half_digits[row]);
  }
  const auto moduli_view = moduli.view(modulus_view(source.dim() - 2));
  const auto negated = at::where(source == 0, source, moduli_view - source);
  const auto negative_digits = decompose(negated);
  const auto reconstruct = [&](const at::Tensor& digits) {
    auto value = digits.select(-2, digits.size(-2) - 1).to(at::kDouble);
    for (int64_t row = moduli.size(0) - 2; row >= 0; --row) {
      value = value * moduli[row] + digits.select(-2, row);
    }
    return value;
  };
  return at::where(
      negative, -reconstruct(negative_digits), reconstruct(positive_digits));
}

at::Tensor decrypt(const at::Tensor& ciphertext,
                   const at::Tensor& secret_key,
                   const TensorTable& tables,
                   std::string_view ntt_backend,
                   int64_t key_row_start,
                   std::string_view input_domain) {
  const auto phase = decrypt_phase(
      ciphertext, secret_key, tables, ntt_backend, key_row_start, input_domain);
  return reconstruct_q_coefficients(phase,
                                    tables.at("reconstruction_parameters"),
                                    tables.at("normalizers"),
                                    tables.at("propagation"),
                                    tables.at("half_digits"));
}

}  // namespace fhelium::ckks
