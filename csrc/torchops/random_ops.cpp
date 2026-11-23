#include <torch/library.h>

#include <fhelium/random.h>

// Torch operator schemas and backend registrations for the public functions
// of the same names. Each registered kernel is the public function itself.

TORCH_LIBRARY_FRAGMENT(fhelium_random_ops, m) {
  m.def("generate_seed() -> Tensor");
  m.def(
      "chacha20_keystream(Tensor seed, int first_block, int block_count) -> "
      "Tensor");
  m.def(
      "sample_uniform_residues(Tensor seed, int[] shape, Tensor rns_params) "
      "-> Tensor");
  m.def("gaussian_cdt_table(float sigma) -> Tensor");
  m.def(
      "sample_discrete_gaussian(Tensor seed, int[] shape, Tensor cdt_table, "
      "Tensor? rns_params=None) -> Tensor");
  m.def(
      "sample_centered_binomial(Tensor seed, int[] shape, int k, "
      "Tensor? rns_params=None) -> Tensor");
  m.def(
      "sample_ternary(Tensor seed, int[] shape, float nonzero_probability, "
      "Tensor? rns_params=None) -> Tensor");
  m.def(
      "sample_fixed_weight_ternary(Tensor seed, int[] shape, int weight, "
      "Tensor? rns_params=None) -> Tensor");
  m.def(
      "add_discrete_gaussian_(Tensor(a!) residues, Tensor seed, "
      "Tensor cdt_table, Tensor rns_params) -> ()");
  m.def(
      "stochastic_round(Tensor seed, Tensor values, int word_stride=1) -> "
      "Tensor");
}

// Host functions without Tensor arguments run for every dispatch key.
TORCH_LIBRARY_IMPL(fhelium_random_ops, CompositeImplicitAutograd, m) {
  m.impl("generate_seed", &fhelium::random::generate_seed);
  m.impl("gaussian_cdt_table", &fhelium::random::gaussian_cdt_table);
}

TORCH_LIBRARY_IMPL(fhelium_random_ops, CompositeExplicitAutograd, m) {
  m.impl("sample_fixed_weight_ternary",
         &fhelium::random::sample_fixed_weight_ternary);
}

#if defined(FHELIUM_WITH_CPU)
TORCH_LIBRARY_IMPL(fhelium_random_ops, CPU, m) {
  m.impl("chacha20_keystream", &fhelium::random::chacha20_keystream);
  m.impl("sample_uniform_residues", &fhelium::random::sample_uniform_residues);
  m.impl("sample_discrete_gaussian",
         &fhelium::random::sample_discrete_gaussian);
  m.impl("sample_centered_binomial",
         &fhelium::random::sample_centered_binomial);
  m.impl("sample_ternary", &fhelium::random::sample_ternary);
  m.impl("add_discrete_gaussian_", &fhelium::random::add_discrete_gaussian_);
  m.impl("stochastic_round", &fhelium::random::stochastic_round);
}
#endif

#if defined(FHELIUM_WITH_CUDA) || defined(FHELIUM_WITH_HIP)
TORCH_LIBRARY_IMPL(fhelium_random_ops, CUDA, m) {
  m.impl("chacha20_keystream", &fhelium::random::chacha20_keystream);
  m.impl("sample_uniform_residues", &fhelium::random::sample_uniform_residues);
  m.impl("sample_discrete_gaussian",
         &fhelium::random::sample_discrete_gaussian);
  m.impl("sample_centered_binomial",
         &fhelium::random::sample_centered_binomial);
  m.impl("sample_ternary", &fhelium::random::sample_ternary);
  m.impl("add_discrete_gaussian_", &fhelium::random::add_discrete_gaussian_);
  m.impl("stochastic_round", &fhelium::random::stochastic_round);
}
#endif
