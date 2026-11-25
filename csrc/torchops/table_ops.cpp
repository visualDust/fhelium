#include <torch/library.h>

#include <fhelium/ckks_parameters.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt_tables.h>
#include <fhelium/rns_tables.h>

#include <string_view>
#include <tuple>
#include <vector>

// Torch operator schemas for the host table and parameter functions of the
// same names. These functions take no Tensor arguments, so one kernel serves
// every dispatch key. Each registered kernel is the public function itself,
// except preset_parameters, whose CkksParameters result is returned as
// (log_n, q_moduli, flattened basis rows, basis sizes, scales, p_moduli).

namespace {

std::tuple<int64_t,
           std::vector<int64_t>,
           std::vector<int64_t>,
           std::vector<int64_t>,
           std::vector<double>,
           std::vector<int64_t>>
preset_parameters_op(std::string_view name) {
  auto parameters = fhelium::ckks::preset_parameters(name);
  std::vector<int64_t> rows;
  std::vector<int64_t> sizes;
  for (const auto& basis : parameters.q_bases) {
    rows.insert(rows.end(), basis.begin(), basis.end());
    sizes.push_back(static_cast<int64_t>(basis.size()));
  }
  return {parameters.log_n,
          std::move(parameters.q_moduli),
          std::move(rows),
          std::move(sizes),
          std::move(parameters.scales),
          std::move(parameters.p_moduli)};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fhelium_rns_ops, m) {
  m.def(
      "parameter_table(int[] moduli, int log_n, float default_scale, "
      "ScalarType dtype) -> Tensor");
  m.def(
      "product_residues(int[] factors, int[] targets, int exponent, bool "
      "montgomery, ScalarType dtype) -> Tensor");
  m.def(
      "half_product_residues(int[] factors, int[] targets, ScalarType dtype) "
      "-> Tensor");
  m.def("half_product_digits(int[] factors, ScalarType dtype) -> Tensor");
  m.def(
      "quotient_offset_residues(int[] p_moduli, int[] d_moduli, bool "
      "p_nearest, bool d_nearest, int[] targets, ScalarType dtype) -> Tensor");
  m.def(
      "mixed_radix_tables(int[] source, int[] destination, ScalarType dtype) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "crt_basis_extension_tables(int[] source, int[] target, bool "
      "normalized, ScalarType dtype) -> (Tensor, Tensor, Tensor, Tensor, "
      "Tensor)");
  m.def(
      "crt_quotient_tables(int[] source, int[] target, int[] divisor, "
      "ScalarType dtype) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "trailing_quotient_inverses(int[] moduli, int drop_count, ScalarType "
      "dtype) -> Tensor");
  m.def("rescale_inverse_matrix(int[] moduli, ScalarType dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_rns_ops, CompositeImplicitAutograd, m) {
  m.impl("parameter_table", &fhelium::rns::parameter_table);
  m.impl("product_residues", &fhelium::rns::product_residues);
  m.impl("half_product_residues", &fhelium::rns::half_product_residues);
  m.impl("half_product_digits", &fhelium::rns::half_product_digits);
  m.impl("quotient_offset_residues", &fhelium::rns::quotient_offset_residues);
  m.impl("mixed_radix_tables", &fhelium::rns::mixed_radix_tables);
  m.impl("crt_basis_extension_tables",
         &fhelium::rns::crt_basis_extension_tables);
  m.impl("crt_quotient_tables", &fhelium::rns::crt_quotient_tables);
  m.impl("trailing_quotient_inverses",
         &fhelium::rns::trailing_quotient_inverses);
  m.impl("rescale_inverse_matrix", &fhelium::rns::rescale_inverse_matrix);
}

TORCH_LIBRARY_FRAGMENT(fhelium_ntt_ops, m) {
  m.def("negacyclic_root(int modulus, int log_n) -> int");
  m.def(
      "compact_twiddles(int[] moduli, int log_n, ScalarType dtype, bool "
      "shoup32=False) -> (Tensor, Tensor)");
  m.def(
      "power_of_two_radix_twiddles(int[] moduli, int log_n, int radix, "
      "ScalarType dtype) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "indexed_radix2_plan(int[] moduli, int log_n, ScalarType dtype) -> "
      "(Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "subring_indexed_radix2_plan(int[] moduli, int log_n, int log_extent, "
      "ScalarType dtype) -> (Tensor, Tensor, Tensor)");
  m.def("galois_indices(int log_n, int galois_element) -> Tensor");
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CompositeImplicitAutograd, m) {
  m.impl("negacyclic_root", &fhelium::ntt::negacyclic_root);
  m.impl("compact_twiddles", &fhelium::ntt::compact_twiddles);
  m.impl("power_of_two_radix_twiddles",
         &fhelium::ntt::power_of_two_radix_twiddles);
  m.impl("indexed_radix2_plan", &fhelium::ntt::indexed_radix2_plan);
  m.impl("subring_indexed_radix2_plan",
         &fhelium::ntt::subring_indexed_radix2_plan);
  m.impl("galois_indices", &fhelium::ntt::galois_indices);
}

TORCH_LIBRARY_FRAGMENT(fhelium_ckks_ops, m) {
  m.def("rotation_galois_element(int log_n, int step, int generator) -> int");
  m.def(
      "coefficient_galois_indices(int log_n, int galois_element) -> (Tensor, "
      "Tensor)");
  m.def("slot_permutations(int log_n, int generator) -> (Tensor, Tensor)");
  m.def("periodic_extent(int period, int log_n, int generator) -> int");
  m.def("periodic_slot_positions(int extent, int generator) -> Tensor");
  m.def("slot_twist(int length, bool inverse, Device device) -> Tensor");
  m.def("scaling_primes(int prime_bits, int ring_dimension) -> int[]");
  m.def("special_primes(int prime_bits, int ring_dimension) -> int[]");
  m.def("preset_names() -> str[]");
  m.def(
      "preset_parameters(str name) -> (int, int[], int[], int[], float[], "
      "int[])");
  m.def("validate_moduli(int log_n, int[] moduli) -> ()");
  m.def("modulus_bits(int[] moduli) -> int");
  m.def("key_digit_sizes(int[] q_moduli, int[] p_moduli) -> int[]");
  m.def(
      "maximum_modulus_bits(int ring_dimension, int target_bits, str "
      "secret_distribution, float error_stddev) -> int?");
}

TORCH_LIBRARY_IMPL(fhelium_ckks_ops, CompositeImplicitAutograd, m) {
  m.impl("rotation_galois_element", &fhelium::ckks::rotation_galois_element);
  m.impl("coefficient_galois_indices",
         &fhelium::ckks::coefficient_galois_indices);
  m.impl("slot_permutations", &fhelium::ckks::slot_permutations);
  m.impl("periodic_extent", &fhelium::ckks::periodic_extent);
  m.impl("periodic_slot_positions", &fhelium::ckks::periodic_slot_positions);
  m.impl("slot_twist", &fhelium::ckks::slot_twist);
  m.impl("scaling_primes", &fhelium::ckks::scaling_primes);
  m.impl("special_primes", &fhelium::ckks::special_primes);
  m.impl("preset_names", &fhelium::ckks::preset_names);
  m.impl("preset_parameters", &preset_parameters_op);
  m.impl("key_digit_sizes", &fhelium::ckks::key_digit_sizes);
  m.impl("validate_moduli", &fhelium::ckks::validate_moduli);
  m.impl("modulus_bits", &fhelium::ckks::modulus_bits);
  m.impl("maximum_modulus_bits", &fhelium::ckks::maximum_modulus_bits);
}
