#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_operations.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/random.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include <vector>

#include "../common/overlap.h"
#include "composition.h"

namespace fhelium::ckks {

namespace {

// Advances the nonce words that follow the four key words of a seed.
void advance(at::Tensor& seed) { seed.narrow(0, 4, seed.size(0) - 4).add_(1); }

// Rounds `values` with the stream of `seed` and advances its nonce, so the
// next call reads a fresh stream.
at::Tensor stochastic_round(const at::Tensor& values,
                            at::Tensor& seed,
                            int64_t word_stride = 1) {
  auto rounded =
      random::stochastic_round(seed, values.to(at::kDouble), word_stride);
  advance(seed);
  return rounded;
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
                                      const EncapsulationTables& tables,
                                      const KeySwitchOptions& low_options,
                                      int64_t return_drop_count,
                                      int64_t target_row_start,
                                      int64_t q_count) {
  auto low = low_options;
  low.output_domain = "coefficient";
  const bool seeded = !low_options.key_seeds.empty();
  TORCH_CHECK_VALUE(!seeded || low_options.key_seeds.size() == 2,
                    "Seeded encapsulated modulus raising requires the low and "
                    "return key seeds");
  if (seeded) low.key_seeds = {low_options.key_seeds[0]};
  const auto switched =
      key_switch(source, low_key, tables.low, low, std::nullopt);
  const auto raised = rns::centered_basis_extend(switched, tables.raise);
  auto component = raised.select(0, 1);
  ntt::transform_(component,
                  tables.return_forward,
                  low_options.ntt_backend,
                  "forward_to_montgomery");
  auto shape = component.sizes().vec();
  shape.insert(shape.begin(), 2);
  auto products = at::zeros(shape, component.options());
  auto products0 = products.select(0, 0);
  auto products1 = products.select(0, 1);
  keyswitch_accumulate_digit_products_(
      products0,
      products1,
      component,
      return_key.select(0, 0),
      tables.return_forward.parameters,
      0,
      -1,
      std::nullopt,
      seeded ? std::optional<at::Tensor>(low_options.key_seeds[1].select(0, 0))
             : std::nullopt,
      low_options.key_rounds);
  ntt::transform_(products,
                  tables.return_inverse,
                  low_options.ntt_backend,
                  "inverse_to_standard");
  const auto quotient = detail_operations::divide_trailing_rows(
      products,
      tables.quotient_inverses,
      tables.quotient_parameters,
      return_drop_count,
      detail_operations::optional_table(tables.quotient_half),
      tables.quotient_order,
      low_options.moddown_rounding);
  auto output =
      quotient.narrow(-2, target_row_start, q_count - target_row_start)
          .contiguous();
  auto output0 = output.select(0, 0);
  rns::add_standard_(output0,
                     raised.select(0, 0).narrow(
                         -2, target_row_start + 1, q_count - target_row_start),
                     tables.target_parameters);
  return output;
}

namespace {

// `out` has the shape, dtype and device of `ciphertext` and is `ciphertext` or
// disjoint from it.
void check_ciphertext_out(const at::Tensor& out,
                          const at::Tensor& ciphertext,
                          const char* operation) {
  TORCH_CHECK_VALUE(out.sizes() == ciphertext.sizes() &&
                        out.scalar_type() == ciphertext.scalar_type() &&
                        out.device() == ciphertext.device() &&
                        (fhelium::detail::same_elements(out, ciphertext) ||
                         fhelium::detail::storage_disjoint(out, ciphertext)),
                    operation,
                    " requires an output with the ciphertext shape, dtype and "
                    "device that is the ciphertext or disjoint from it");
}

// Copies the components after component zero when `out` is a separate
// Tensor.
void copy_trailing_components(const at::Tensor& ciphertext, at::Tensor& out) {
  if (!fhelium::detail::same_elements(out, ciphertext) &&
      ciphertext.size(0) > 1) {
    out.narrow(0, 1, ciphertext.size(0) - 1)
        .copy_(ciphertext.narrow(0, 1, ciphertext.size(0) - 1));
  }
}

void multiply_compressed_component(const at::Tensor& component,
                                   const at::Tensor& compressed,
                                   const std::optional<at::Tensor>& implicit,
                                   const at::Tensor& parameters,
                                   std::string_view layout,
                                   at::Tensor target) {
  if (layout == "cyclic") {
    rns::montgomery_mul_cyclic_compressed_out(
        component, compressed, parameters, target);
    return;
  }
  if (layout == "contiguous") {
    rns::montgomery_mul_contiguous_compressed_out(
        component, compressed, parameters, target);
    return;
  }
  TORCH_CHECK_VALUE(layout == "strided_sparse" && implicit.has_value(),
                    "Compressed multiplication requires a supported layout "
                    "and its implicit rows");
  using at::indexing::Ellipsis;
  using at::indexing::Slice;
  const auto repeat = component.size(-1) / compressed.size(-1);
  const auto support = Slice(at::indexing::None, at::indexing::None, repeat);
  // The support product is formed before `target`, which may be `component`,
  // is overwritten.
  const auto supported = rns::montgomery_mul(
      component.index({Ellipsis, support}), compressed, parameters);
  rns::montgomery_mul_cyclic_compressed_out(
      component, *implicit, parameters, target);
  target.index({Ellipsis, support}).copy_(supported);
}

}  // namespace

void add_compressed_plaintext_out(const at::Tensor& ciphertext,
                                  const at::Tensor& compressed,
                                  const std::optional<at::Tensor>& implicit,
                                  const at::Tensor& parameters,
                                  std::string_view layout,
                                  std::string_view polynomial_domain,
                                  at::Tensor& out) {
  check_ciphertext_out(out, ciphertext, "add_compressed_plaintext_out");
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
  copy_trailing_components(ciphertext, out);
  const auto component = ciphertext.select(0, 0);
  auto target = out.select(0, 0);
  if (layout == "cyclic") {
    add_cyclic_compressed_plaintext_component_out(
        component, plaintext, parameters, target);
  } else if (layout == "contiguous") {
    add_contiguous_compressed_plaintext_component_out(
        component, plaintext, parameters, target);
  } else if (layout == "strided_sparse" && implicit_rows.has_value()) {
    auto rows = implicit_rows->squeeze(-1);
    if (rows.device().is_cpu()) {
      rows = rows.reshape({-1, rows.size(-1)}).contiguous();
    }
    add_strided_plaintext_component_out(
        component, plaintext, rows, parameters, target);
  } else {
    TORCH_CHECK_VALUE(false,
                      "Compressed addition requires a supported layout and "
                      "its implicit rows");
  }
}

at::Tensor add_compressed_plaintext(const at::Tensor& ciphertext,
                                    const at::Tensor& compressed,
                                    const std::optional<at::Tensor>& implicit,
                                    const at::Tensor& parameters,
                                    std::string_view layout,
                                    std::string_view polynomial_domain) {
  auto out = at::empty_like(ciphertext);
  add_compressed_plaintext_out(ciphertext,
                               compressed,
                               implicit,
                               parameters,
                               layout,
                               polynomial_domain,
                               out);
  return out;
}

at::Tensor& add_compressed_plaintext_(at::Tensor& ciphertext,
                                      const at::Tensor& compressed,
                                      const std::optional<at::Tensor>& implicit,
                                      const at::Tensor& parameters,
                                      std::string_view layout,
                                      std::string_view polynomial_domain) {
  add_compressed_plaintext_out(ciphertext,
                               compressed,
                               implicit,
                               parameters,
                               layout,
                               polynomial_domain,
                               ciphertext);
  return ciphertext;
}

void multiply_compressed_plaintext_out(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout,
    at::Tensor& out) {
  check_ciphertext_out(out, ciphertext, "multiply_compressed_plaintext_out");
  int64_t leading = 1;
  for (int64_t i = 0; i + 2 < compressed.dim(); ++i) {
    leading *= compressed.size(i);
  }
  if (leading == 1) {
    multiply_compressed_component(
        ciphertext, compressed, implicit, parameters, layout, out);
    return;
  }
  for (int64_t component = 0; component < ciphertext.size(0); ++component) {
    multiply_compressed_component(ciphertext.select(0, component),
                                  compressed,
                                  implicit,
                                  parameters,
                                  layout,
                                  out.select(0, component));
  }
}

at::Tensor multiply_compressed_plaintext(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout) {
  auto out = at::empty_like(ciphertext);
  multiply_compressed_plaintext_out(
      ciphertext, compressed, implicit, parameters, layout, out);
  return out;
}

at::Tensor& multiply_compressed_plaintext_(
    at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout) {
  multiply_compressed_plaintext_out(
      ciphertext, compressed, implicit, parameters, layout, ciphertext);
  return ciphertext;
}

void scalar_arithmetic_out(const at::Tensor& ciphertext,
                           const at::Tensor& parameters,
                           at::Tensor& seed,
                           double scalar,
                           double scalar_scale,
                           bool add,
                           at::Tensor& out) {
  check_ciphertext_out(out, ciphertext, "scalar_arithmetic_out");
  const auto scaled = scalar * scalar_scale;
  const auto coefficient = stochastic_round(
      at::full({1}, scaled, ciphertext.options().dtype(at::kDouble)), seed);
  auto residues =
      rns::lift_integer_coefficients(coefficient, parameters.select(0, 0));
  if (add) {
    if (!fhelium::detail::same_elements(out, ciphertext)) {
      out.copy_(ciphertext);
    }
    auto constant = out.select(0, 0).narrow(-1, 0, 1);
    rns::add_standard_(constant, residues, parameters);
    return;
  }
  rns::to_montgomery_(residues, parameters);
  rns::montgomery_mul_row_scalars_standard_out(
      ciphertext, residues.squeeze(-1), parameters, out);
}

at::Tensor scalar_arithmetic(const at::Tensor& ciphertext,
                             const at::Tensor& parameters,
                             at::Tensor& seed,
                             double scalar,
                             double scalar_scale,
                             bool add) {
  auto out = at::empty_like(ciphertext);
  scalar_arithmetic_out(
      ciphertext, parameters, seed, scalar, scalar_scale, add, out);
  return out;
}

at::Tensor encode(const at::Tensor& message,
                  const at::Tensor& pre,
                  const at::Tensor& twister,
                  at::Tensor& seed,
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
  const auto scaled = coefficients * scale;
  // Rounded coefficients are stored as int64.
  const auto magnitude = at::abs(scaled).max().item<double>();
  TORCH_CHECK_VALUE(magnitude < 0x1p63,
                    "Encoded coefficient magnitude ",
                    magnitude,
                    " at scale ",
                    scale,
                    " reaches the signed 64-bit integer range 2^63");
  return stochastic_round(scaled, seed);
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
    at::Tensor& seed,
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
                   const RingTables& tables,
                   at::Tensor& seed,
                   const at::Tensor& cdt_table,
                   std::string_view ntt_backend,
                   int64_t key_row_start,
                   std::string_view output_domain) {
  const auto& parameters = tables.forward.parameters;
  const auto shape = coefficients.sizes().vec();
  // Each sampler reads its own nonce: the plaintext error, the ternary mask,
  // and the mask error. The seed advances by three.
  auto& stream = seed;
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
  ntt::transform_(v, tables.forward, ntt_backend, "forward_to_montgomery");
  auto products =
      at::stack({rns::montgomery_mul(v, keys.select(0, 0), parameters),
                 rns::montgomery_mul(v, keys.select(0, 1), parameters)});
  if (output_domain == "ntt") {
    auto added = at::stack({plaintext, sample(parameters)});
    ntt::transform_(
        added, tables.forward, ntt_backend, "forward_to_montgomery");
    return rns::add_lazy(products, added, parameters);
  }
  TORCH_CHECK_VALUE(output_domain == "coefficient",
                    "Encryption output_domain is unsupported");
  ntt::transform_(
      products, tables.inverse, ntt_backend, "inverse_to_standard_lazy");
  auto component0 = products.select(0, 0);
  rns::add_standard_(component0, plaintext, parameters);
  auto component1 = products.select(0, 1);
  random::add_discrete_gaussian_(component1, stream, cdt_table, parameters);
  advance(stream);
  return products;
}

at::Tensor decrypt_phase(const at::Tensor& ciphertext,
                         const at::Tensor& secret_key,
                         const RingTables& tables,
                         std::string_view ntt_backend,
                         int64_t key_row_start,
                         std::string_view input_domain) {
  const auto& parameters = tables.forward.parameters;
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
          value, tables.forward, ntt_backend, "forward_to_montgomery");
    }
    auto product = rns::montgomery_mul(value, power, parameters);
    if (coefficient) {
      ntt::transform_(
          product, tables.inverse, ntt_backend, "inverse_to_standard");
      phase = rns::add_standard(phase, product, parameters);
    } else {
      phase = rns::add_lazy(phase, product, parameters);
    }
    if (k + 1 < components) {
      power = rns::montgomery_mul(power, secret, parameters);
    }
  }
  if (!coefficient) {
    ntt::transform_(phase, tables.inverse, ntt_backend, "inverse_to_standard");
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
                   const DecryptionTables& tables,
                   std::string_view ntt_backend,
                   int64_t key_row_start,
                   std::string_view input_domain) {
  const auto phase = decrypt_phase(ciphertext,
                                   secret_key,
                                   tables.ring,
                                   ntt_backend,
                                   key_row_start,
                                   input_domain);
  return reconstruct_q_coefficients(phase,
                                    tables.reconstruction_parameters,
                                    tables.normalizers,
                                    tables.propagation,
                                    tables.half_digits);
}

}  // namespace fhelium::ckks
