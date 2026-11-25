#include "backends.h"

#include <ATen/ATen.h>

#include "../common/backend_dispatch.h"

#undef FHELIUM_DISPATCH_NAMESPACE
#define FHELIUM_DISPATCH_NAMESPACE "fhelium::random::"

namespace fhelium::random {

at::Tensor chacha20_keystream(const at::Tensor& seed,
                              int64_t first_block,
                              int64_t block_count) {
  FHELIUM_DISPATCH_CPU_GPU(chacha20_keystream, seed, first_block, block_count);
}

at::Tensor sample_uniform_residues(const at::Tensor& seed,
                                   at::IntArrayRef shape,
                                   const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(sample_uniform_residues, seed, shape, rns_params);
}

at::Tensor sample_discrete_gaussian(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    const at::Tensor& cdt_table,
    const std::optional<at::Tensor>& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      sample_discrete_gaussian, seed, shape, cdt_table, rns_params);
}

at::Tensor sample_centered_binomial(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    int64_t k,
    const std::optional<at::Tensor>& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      sample_centered_binomial, seed, shape, k, rns_params);
}

at::Tensor sample_ternary(const at::Tensor& seed,
                          at::IntArrayRef shape,
                          double nonzero_probability,
                          const std::optional<at::Tensor>& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      sample_ternary, seed, shape, nonzero_probability, rns_params);
}

void add_discrete_gaussian_(at::Tensor& residues,
                            const at::Tensor& seed,
                            const at::Tensor& cdt_table,
                            const at::Tensor& rns_params) {
  FHELIUM_DISPATCH_CPU_GPU(
      add_discrete_gaussian_, residues, seed, cdt_table, rns_params);
}

at::Tensor stochastic_round(const at::Tensor& seed,
                            const at::Tensor& values,
                            int64_t word_stride) {
  FHELIUM_DISPATCH_CPU_GPU(stochastic_round, seed, values, word_stride);
}

at::Tensor expand_seeded_key(const at::Tensor& seed_rows,
                             const at::Tensor& moduli,
                             int64_t ring_dimension,
                             int64_t rounds) {
  FHELIUM_DISPATCH_CPU_GPU(
      expand_seeded_key, seed_rows, moduli, ring_dimension, rounds);
}

at::Tensor sample_fixed_weight_ternary(
    const at::Tensor& seed,
    at::IntArrayRef shape,
    int64_t weight,
    const std::optional<at::Tensor>& rns_params) {
  constexpr const char* operation = "sample_fixed_weight_ternary";
  TORCH_CHECK(!shape.empty(),
              operation,
              " requires a shape ending in the coefficient axis");
  const int64_t coefficients = shape.back();
  TORCH_CHECK(weight >= 0 && weight <= coefficients,
              operation,
              " requires 0 <= weight <= shape[-1]");
  int64_t count = 1;
  for (const int64_t extent : shape) {
    TORCH_CHECK(extent >= 0, operation, " requires nonnegative extents");
    count *= extent;
  }
  const int64_t rows = coefficients == 0 ? 0 : count / coefficients;
  // Coefficient i reads stream words [4i, 4i + 4): a 64-bit label key, a
  // 63-bit label key, and a sign bit.
  const auto words = chacha20_keystream(seed, 0, (count + 3) / 4)
                         .view(-1)
                         .narrow(0, 0, 4 * count)
                         .to(at::kLong)
                         .view({rows, coefficients, 4});
  const auto high = words.select(-1, 0).bitwise_or(
      words.select(-1, 1).bitwise_left_shift(32));
  const auto low = words.select(-1, 2).bitwise_or(
      words.select(-1, 3).bitwise_right_shift(1).bitwise_left_shift(32));
  const auto sign = words.select(-1, 3).bitwise_and(1);
  // Lexicographic (high, low) order from two stable sorts.
  const auto by_low =
      std::get<1>(low.sort(std::optional<bool>(true), -1, false));
  const auto by_high = std::get<1>(
      high.gather(-1, by_low).sort(std::optional<bool>(true), -1, false));
  const auto support = by_low.gather(-1, by_high).narrow(-1, 0, weight);
  const auto values = sign.gather(-1, support).mul(-2).add_(1);
  auto centered = at::zeros({rows, coefficients}, seed.options())
                      .scatter_(-1, support, values);
  if (!rns_params.has_value()) {
    return centered.view(shape);
  }
  TORCH_CHECK(rns_params->device() == seed.device(),
              operation,
              " requires rns_params on the device of seed");
  const auto modulus =
      rns_params->select(0, 0).bitwise_right_shift(1).unsqueeze(-1);
  const auto lifted = centered.to(rns_params->scalar_type()).unsqueeze(-2);
  auto residues = at::where(lifted.lt(0), lifted.add(modulus), lifted);
  std::vector<int64_t> output_shape(shape.begin(), shape.end() - 1);
  output_shape.push_back(modulus.size(0));
  output_shape.push_back(coefficients);
  return residues.view(output_shape);
}

}  // namespace fhelium::random
