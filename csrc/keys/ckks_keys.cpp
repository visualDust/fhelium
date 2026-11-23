#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_keys.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/random.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include <vector>

namespace fhelium::ckks {

namespace {

// Advances the nonce words that follow the four key words of a seed.
void advance(at::Tensor& seed) { seed.narrow(0, 4, seed.size(0) - 4).add_(1); }

int64_t log2_exact(int64_t value) {
  int64_t bits = 0;
  while ((int64_t{1} << bits) < value) {
    ++bits;
  }
  TORCH_CHECK_VALUE((int64_t{1} << bits) == value,
                    "The ring dimension must be a power of two");
  return bits;
}

// Samples or lifts the public key of `secret` and advances `stream` once per
// sampled component.
at::Tensor public_key(const at::Tensor& secret,
                      const TensorTable& tables,
                      at::Tensor& stream,
                      const at::Tensor& cdt_table,
                      std::string_view ntt_backend,
                      const std::optional<at::Tensor>& uniform,
                      const std::optional<at::Tensor>& error_coefficients) {
  const auto& parameters = tables.at("parameters");
  const int64_t n = secret.size(-1);
  at::Tensor error;
  if (error_coefficients.has_value()) {
    error = rns::lift_integer_coefficients(*error_coefficients,
                                           parameters.select(0, 0));
  } else {
    error =
        random::sample_discrete_gaussian(stream, {n}, cdt_table, parameters);
    advance(stream);
  }
  ntt::transform_(error, tables, ntt_backend, "", "forward_to_montgomery");
  at::Tensor a;
  if (uniform.has_value()) {
    a = *uniform;
  } else {
    a = random::sample_uniform_residues(stream, {n}, parameters);
    advance(stream);
  }
  const auto product = rns::montgomery_mul(
      a, secret.narrow(0, 0, parameters.size(1)), parameters);
  return at::stack({rns::sub_lazy(error, product, parameters), a});
}

}  // namespace

at::Tensor create_secret_key(int64_t ring_dimension,
                             const TensorTable& tables,
                             const at::Tensor& seed,
                             std::string_view ntt_backend,
                             std::optional<int64_t> hamming_weight) {
  const auto& parameters = tables.at("parameters");
  auto secret = hamming_weight.has_value()
                    ? random::sample_fixed_weight_ternary(
                          seed, {ring_dimension}, *hamming_weight, parameters)
                    : random::sample_ternary(
                          seed, {ring_dimension}, 2.0 / 3.0, parameters);
  ntt::transform_(secret, tables, ntt_backend, "", "forward_to_montgomery");
  return secret;
}

at::Tensor prepare_secret_key(const at::Tensor& coefficients,
                              const TensorTable& tables,
                              std::string_view ntt_backend) {
  auto secret = rns::lift_integer_coefficients(
      coefficients, tables.at("parameters").select(0, 0));
  ntt::transform_(secret, tables, ntt_backend, "", "forward_to_montgomery");
  return secret;
}

at::Tensor create_public_key(
    const at::Tensor& secret_key,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    std::string_view ntt_backend,
    const std::optional<at::Tensor>& uniform,
    const std::optional<at::Tensor>& error_coefficients) {
  auto stream = seed.clone();
  return public_key(secret_key,
                    tables,
                    stream,
                    cdt_table,
                    ntt_backend,
                    uniform,
                    error_coefficients);
}

at::Tensor create_key_switch_key(
    const at::Tensor& source_secret_key,
    const at::Tensor& destination_secret_key,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    const KeySwitchKeyLayout& layout,
    const std::optional<at::Tensor>& uniform_by_digit,
    const std::optional<at::Tensor>& error_by_digit) {
  const auto& parameters = tables.at("parameters");
  const auto& product = tables.at("source_product");
  const int64_t q_count = product.size(0);
  auto scaled = source_secret_key.narrow(0, 0, q_count).clone();
  rns::montgomery_mul_row_scalars_(
      scaled, product, parameters.narrow(1, 0, q_count));
  const auto twice_modulus = parameters.select(0, 0);

  // Contiguous runs of the key rows: active Q, then the auxiliary rows.
  std::vector<std::pair<int64_t, int64_t>> runs{
      {layout.q_row_start, q_count - layout.q_row_start}};
  for (const int64_t row : layout.p_rows) {
    auto& last = runs.back();
    if (last.first + last.second == row) {
      ++last.second;
    } else {
      runs.emplace_back(row, 1);
    }
  }

  auto stream = seed.clone();
  std::vector<at::Tensor> digits;
  int64_t row = layout.q_row_start;
  for (size_t j = 0; j < layout.digit_sizes.size(); ++j) {
    const auto index = static_cast<int64_t>(j);
    auto key = public_key(
        destination_secret_key,
        tables,
        stream,
        cdt_table,
        layout.ntt_backend,
        uniform_by_digit.has_value()
            ? std::optional<at::Tensor>(uniform_by_digit->select(0, index))
            : std::nullopt,
        error_by_digit.has_value()
            ? std::optional<at::Tensor>(error_by_digit->select(0, index))
            : std::nullopt);
    const int64_t size = layout.digit_sizes[j];
    auto c0 = key.select(0, 0).narrow(0, row, size);
    c0.copy_(rns::add_lazy_with_twice_modulus(
        c0, scaled.narrow(0, row, size), twice_modulus.narrow(0, row, size)));
    std::vector<at::Tensor> rows;
    for (const auto& [start, length] : runs) {
      rows.push_back(key.narrow(1, start, length));
    }
    digits.push_back(at::cat(rows, 1));
    row += size;
  }
  TORCH_CHECK_VALUE(row == q_count,
                    "Key digits must cover the Q rows from q_row_start");
  return at::stack(digits);
}

at::Tensor create_relinearization_key(
    const at::Tensor& secret_key,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    const KeySwitchKeyLayout& layout,
    const std::optional<at::Tensor>& uniform_by_digit,
    const std::optional<at::Tensor>& error_by_digit) {
  const auto squared =
      rns::montgomery_mul(secret_key, secret_key, tables.at("parameters"));
  return create_key_switch_key(squared,
                               secret_key,
                               tables,
                               seed,
                               cdt_table,
                               layout,
                               uniform_by_digit,
                               error_by_digit);
}

at::Tensor create_galois_key(const at::Tensor& secret_key,
                             int64_t galois_element,
                             const TensorTable& tables,
                             const at::Tensor& seed,
                             const at::Tensor& cdt_table,
                             const KeySwitchKeyLayout& layout,
                             const std::optional<at::Tensor>& uniform_by_digit,
                             const std::optional<at::Tensor>& error_by_digit) {
  auto coefficients = ntt::transform(
      secret_key, tables, layout.ntt_backend, "", "inverse_montgomery");
  const auto [indices, signs] = coefficient_galois_indices(
      log2_exact(secret_key.size(-1)), galois_element);
  auto mapped = apply_coefficient_galois_automorphism(
      coefficients,
      indices.to(secret_key.device()),
      signs.to(secret_key.device()),
      tables.at("parameters").select(0, 0));
  ntt::transform_(mapped, tables, layout.ntt_backend, "", "forward_montgomery");
  return create_key_switch_key(mapped,
                               secret_key,
                               tables,
                               seed,
                               cdt_table,
                               layout,
                               uniform_by_digit,
                               error_by_digit);
}

at::Tensor convert_secret_key(const at::Tensor& secret_key,
                              const TensorTable& tables,
                              std::string_view ntt_backend) {
  const auto source = ntt::transform(
      secret_key, tables, ntt_backend, "source", "inverse_to_standard");
  auto converted = rns::centered_basis_extend(
      source.to(tables.at("source_parameters").scalar_type()), tables);
  converted = converted.to(tables.at("target_forward_0").scalar_type());
  ntt::transform_(
      converted, tables, ntt_backend, "target", "forward_to_montgomery");
  return converted;
}

}  // namespace fhelium::ckks
