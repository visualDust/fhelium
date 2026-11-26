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

// Seed rows [digit, 6] of a key: the first ten words of the stream block at
// the current nonce as five int64 values, followed by the digit index.
at::Tensor seed_rows(at::Tensor& stream, int64_t digits) {
  auto words = random::chacha20_keystream(stream, 0, 1)
                   .select(0, 0)
                   .narrow(0, 0, 10)
                   .to(at::kLong)
                   .bitwise_and(0xffffffffLL);
  auto packed =
      words.slice(0, 0, 10, 2)
          .bitwise_or(words.slice(0, 1, 10, 2).bitwise_left_shift(32));
  advance(stream);
  auto index = at::arange(digits, packed.options()).unsqueeze(1);
  return at::cat({packed.unsqueeze(0).expand({digits, 5}), index}, 1);
}

at::Tensor key_moduli(const SeedSpec& spec, const at::Tensor& like) {
  return at::tensor(spec.moduli, at::TensorOptions().dtype(at::kLong))
      .to(like.device(), like.scalar_type());
}

}  // namespace

KeySwitchKey expand_key_switch_key(const KeySwitchKey& key) {
  if (key.expanded()) {
    return key;
  }
  TORCH_CHECK_VALUE(key.seed_rows.has_value(),
                    "A compact key requires seed rows");
  TORCH_CHECK_VALUE(key.spec.prf == "chacha",
                    "Unsupported key seed generator ",
                    key.spec.prf);
  auto uniform = random::expand_seeded_key(*key.seed_rows,
                                           key_moduli(key.spec, key.data),
                                           key.data.size(-1),
                                           key.spec.rounds);
  return {
      at::stack({key.data.select(1, 0), uniform}, 1), key.seed_rows, key.spec};
}

KeySwitchKey compact_key_switch_key(const KeySwitchKey& key) {
  TORCH_CHECK_VALUE(key.seed_rows.has_value(),
                    "A key without seed rows cannot be compacted");
  if (!key.expanded()) {
    return key;
  }
  return {key.data.narrow(1, 0, 1).clone(), key.seed_rows, key.spec};
}

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

KeySwitchKey create_key_switch_key(
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

  int64_t q_row_count = 0;
  for (const int64_t size : layout.digit_sizes) {
    q_row_count += size;
  }
  TORCH_CHECK_VALUE(q_row_count > 0 && q_row_count == static_cast<int64_t>(
                                                          layout.q_rows.size()),
                    "Key digits must partition the active Q rows");
  for (const int64_t row : layout.q_rows) {
    TORCH_CHECK_VALUE(row >= 0 && row < q_count,
                      "Active key rows must be Q rows");
  }
  // Maximal runs of consecutive parameter rows in `rows[first, last)`.
  const auto runs_of = [](const std::vector<int64_t>& rows,
                          size_t first,
                          size_t last) {
    std::vector<std::pair<int64_t, int64_t>> runs;
    for (size_t i = first; i < last; ++i) {
      if (!runs.empty() && runs.back().first + runs.back().second == rows[i]) {
        ++runs.back().second;
      } else {
        runs.emplace_back(rows[i], 1);
      }
    }
    return runs;
  };
  std::vector<int64_t> key_rows(layout.q_rows);
  key_rows.insert(key_rows.end(), layout.p_rows.begin(), layout.p_rows.end());
  const auto runs = runs_of(key_rows, 0, key_rows.size());

  auto stream = seed.clone();
  const auto digit_count = static_cast<int64_t>(layout.digit_sizes.size());
  std::optional<at::Tensor> seeds;
  SeedSpec spec;
  auto uniform = uniform_by_digit;
  if (!uniform.has_value()) {
    // Seeded key: k_1 is the seed expansion on the key rows, placed at their
    // parameter rows for the public-key construction.
    seeds = seed_rows(stream, digit_count);
    const auto rows =
        at::tensor(key_rows, at::TensorOptions().dtype(at::kLong));
    const auto moduli =
        twice_modulus.to(at::kCPU).to(at::kLong).index_select(0, rows).div(
            2, "floor");
    spec.moduli.assign(moduli.data_ptr<int64_t>(),
                       moduli.data_ptr<int64_t>() + moduli.numel());
    const auto expanded =
        random::expand_seeded_key(*seeds,
                                  key_moduli(spec, parameters),
                                  destination_secret_key.size(-1),
                                  spec.rounds);
    auto full = at::zeros({digit_count, parameters.size(1), expanded.size(-1)},
                          expanded.options());
    full.index_copy_(1, rows.to(full.device()), expanded);
    uniform = full;
  }
  std::vector<at::Tensor> digits;
  size_t offset = 0;
  for (size_t j = 0; j < layout.digit_sizes.size(); ++j) {
    const auto index = static_cast<int64_t>(j);
    auto key = public_key(
        destination_secret_key,
        tables,
        stream,
        cdt_table,
        layout.ntt_backend,
        std::optional<at::Tensor>(uniform->select(0, index)),
        error_by_digit.has_value()
            ? std::optional<at::Tensor>(error_by_digit->select(0, index))
            : std::nullopt);
    const auto size = static_cast<size_t>(layout.digit_sizes[j]);
    for (const auto& [row, length] :
         runs_of(layout.q_rows, offset, offset + size)) {
      auto c0 = key.select(0, 0).narrow(0, row, length);
      c0.copy_(rns::add_lazy_with_twice_modulus(
          c0,
          scaled.narrow(0, row, length),
          twice_modulus.narrow(0, row, length)));
    }
    std::vector<at::Tensor> rows;
    for (const auto& [start, length] : runs) {
      rows.push_back(key.narrow(1, start, length));
    }
    digits.push_back(at::cat(rows, 1));
    offset += size;
  }
  return {at::stack(digits), seeds, spec};
}

KeySwitchKey create_relinearization_key(
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

KeySwitchKey create_galois_key(
    const at::Tensor& secret_key,
    int64_t galois_element,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    const KeySwitchKeyLayout& layout,
    const std::optional<at::Tensor>& uniform_by_digit,
    const std::optional<at::Tensor>& error_by_digit) {
  // The coefficient automorphism reads residues in [0, q_j).
  auto coefficients = ntt::transform(
      secret_key, tables, layout.ntt_backend, "", "inverse_to_standard");
  const auto [indices, signs] = coefficient_galois_indices(
      log2_exact(secret_key.size(-1)), galois_element);
  auto mapped = apply_coefficient_galois_automorphism(
      coefficients,
      indices.to(secret_key.device()),
      signs.to(secret_key.device()),
      tables.at("parameters").select(0, 0));
  ntt::transform_(
      mapped, tables, layout.ntt_backend, "", "forward_to_montgomery");
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
