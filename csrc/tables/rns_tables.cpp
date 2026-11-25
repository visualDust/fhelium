#include <fhelium/rns_tables.h>

#include <c10/util/Exception.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <fhelium/rns_parameters.h>

#include "host_modular.h"
#include "table_support.h"

namespace fhelium::rns {
namespace {

namespace host = ::fhelium::detail_host;
using ::fhelium::detail_tables::check_moduli;
using ::fhelium::detail_tables::montgomery_radix_bits;
using ::fhelium::detail_tables::to_tensor;
using ::fhelium::detail_tables::to_u64;

// Exact residue of a nonnegative integral double modulo m.
uint64_t double_mod(double value, uint64_t m) {
  int exponent = 0;
  const double fraction = std::frexp(value, &exponent);
  if (exponent <= 53) {
    return static_cast<uint64_t>(value) % m;
  }
  const auto mantissa = static_cast<uint64_t>(std::ldexp(fraction, 53));
  return host::mul_mod(mantissa % m, host::power_of_two_mod(exponent - 53, m),
                       m);
}

}  // namespace

at::Tensor parameter_table(at::IntArrayRef moduli,
                           int64_t log_n,
                           double default_scale,
                           at::ScalarType dtype) {
  const auto q = to_u64(moduli);
  const int64_t w = montgomery_radix_bits(dtype);
  check_moduli(q, w, "parameter_table");
  TORCH_CHECK_VALUE(log_n >= 1 && log_n < 63,
                    "parameter_table requires 1 <= log_n < 63");
  TORCH_CHECK_VALUE(std::isfinite(default_scale) && default_scale > 0.0,
                    "parameter_table requires a positive finite scale");
  const double scale = std::nearbyint(default_scale);
  const int64_t h = w / 2;
  const uint64_t half_mask = (uint64_t{1} << h) - 1;
  const uint64_t radix_mask =
      w == 64 ? ~uint64_t{0} : (uint64_t{1} << w) - 1;
  const uint64_t radix_modulus = uint64_t{1} << w;
  std::vector<std::vector<uint64_t>> rows(RNS_PARAMETER_COUNT,
                                          std::vector<uint64_t>(q.size()));
  for (size_t j = 0; j < q.size(); ++j) {
    const uint64_t modulus = q[j];
    const uint64_t r = host::power_of_two_mod(w, modulus);
    const uint64_t r2 = host::mul_mod(r, r, modulus);
    const uint64_t neg_inverse =
        (radix_modulus - host::inv_mod(modulus, radix_modulus)) & radix_mask;
    rows[RNS_PARAM_TWICE_MODULUS][j] = modulus << 1;
    rows[RNS_PARAM_MODULUS_LO][j] = modulus & half_mask;
    rows[RNS_PARAM_MODULUS_HI][j] = modulus >> h;
    rows[RNS_PARAM_NEG_INV_MODULUS_LO][j] = neg_inverse & half_mask;
    rows[RNS_PARAM_NEG_INV_MODULUS_HI][j] = neg_inverse >> h;
    rows[RNS_PARAM_R2][j] = r2;
    rows[RNS_PARAM_SCALED_R2][j] =
        host::mul_mod(r2, double_mod(scale, modulus), modulus);
    rows[RNS_PARAM_N_INV_MONTGOMERY][j] = host::mul_mod(
        host::inv_mod(host::power_of_two_mod(log_n, modulus), modulus), r,
        modulus);
  }
  return to_tensor(rows, dtype);
}

at::Tensor product_residues(at::IntArrayRef factors,
                            at::IntArrayRef targets,
                            int64_t exponent,
                            bool montgomery,
                            at::ScalarType dtype) {
  TORCH_CHECK_VALUE(exponent == 1 || exponent == -1,
                    "product_residues requires exponent 1 or -1");
  const auto f = to_u64(factors);
  const auto q = to_u64(targets);
  const int64_t w = montgomery_radix_bits(dtype);
  check_moduli(q, w, "product_residues");
  std::vector<uint64_t> values(q.size());
  for (size_t i = 0; i < q.size(); ++i) {
    uint64_t value = host::product_mod(f, q[i]);
    if (exponent < 0) {
      value = host::inv_mod(value, q[i]);
    }
    if (montgomery) {
      value = host::mul_mod(value, host::power_of_two_mod(w, q[i]), q[i]);
    }
    values[i] = value;
  }
  return to_tensor(values, dtype);
}

at::Tensor half_product_residues(at::IntArrayRef factors,
                                 at::IntArrayRef targets,
                                 at::ScalarType dtype) {
  const auto f = to_u64(factors);
  const auto q = to_u64(targets);
  check_moduli(q, montgomery_radix_bits(dtype), "half_product_residues");
  std::vector<uint64_t> values(q.size());
  for (size_t i = 0; i < q.size(); ++i) {
    values[i] = host::half_product_mod(f, q[i]);
  }
  return to_tensor(values, dtype);
}

at::Tensor half_product_digits(at::IntArrayRef factors, at::ScalarType dtype) {
  const auto f = to_u64(factors);
  check_moduli(f, montgomery_radix_bits(dtype), "half_product_digits");
  std::vector<uint64_t> residues(f.size());
  for (size_t i = 0; i < f.size(); ++i) {
    residues[i] = host::half_product_mod(f, f[i]);
  }
  return to_tensor(host::mixed_radix_digits(residues, f), dtype);
}

at::Tensor quotient_offset_residues(at::IntArrayRef p_moduli,
                                    at::IntArrayRef d_moduli,
                                    bool p_nearest,
                                    bool d_nearest,
                                    at::IntArrayRef targets,
                                    at::ScalarType dtype) {
  const auto p = to_u64(p_moduli);
  const auto d = to_u64(d_moduli);
  const auto q = to_u64(targets);
  check_moduli(q, montgomery_radix_bits(dtype), "quotient_offset_residues");
  std::vector<uint64_t> values(q.size());
  for (size_t i = 0; i < q.size(); ++i) {
    uint64_t value = 0;
    if (p_nearest) {
      value = host::half_product_mod(p, q[i]);
    }
    if (d_nearest) {
      value = host::add_mod(value,
                            host::mul_mod(host::product_mod(p, q[i]),
                                          host::half_product_mod(d, q[i]),
                                          q[i]),
                            q[i]);
    }
    values[i] = value;
  }
  return to_tensor(values, dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mixed_radix_tables(
    at::IntArrayRef source,
    at::IntArrayRef destination,
    at::ScalarType dtype) {
  const auto s = to_u64(source);
  const auto p = to_u64(destination);
  const int64_t w = montgomery_radix_bits(dtype);
  check_moduli(s, w, "mixed_radix_tables");
  check_moduli(p, w, "mixed_radix_tables");
  const size_t width = s.size();
  TORCH_CHECK_VALUE(width >= 1, "mixed_radix_tables requires a source basis");
  const size_t rows = width - 1;
  std::vector<uint64_t> normalizers(rows);
  std::vector<std::vector<uint64_t>> propagation(rows,
                                                 std::vector<uint64_t>(width));
  std::vector<std::vector<uint64_t>> extension(rows,
                                               std::vector<uint64_t>(p.size()));
  std::vector<uint64_t> prefix;
  for (size_t k = 0; k < rows; ++k) {
    prefix.push_back(s[k]);
    const uint64_t next = s[k + 1];
    normalizers[k] = host::mul_mod(
        host::inv_mod(host::product_mod(prefix, next), next),
        host::power_of_two_mod(w, next), next);
    for (size_t t = k + 2; t < width; ++t) {
      propagation[k][t] = host::mul_mod(host::product_mod(prefix, s[t]),
                                        host::power_of_two_mod(w, s[t]), s[t]);
    }
    for (size_t t = 0; t < p.size(); ++t) {
      const uint64_t r = host::power_of_two_mod(w, p[t]);
      extension[k][t] = host::mul_mod(host::product_mod(prefix, p[t]),
                                      host::mul_mod(r, r, p[t]), p[t]);
    }
  }
  return {to_tensor(normalizers, dtype),
          to_tensor(propagation, dtype, static_cast<int64_t>(width)),
          to_tensor(extension, dtype, static_cast<int64_t>(p.size()))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
crt_basis_extension_tables(at::IntArrayRef source,
                           at::IntArrayRef target,
                           bool normalized,
                           at::ScalarType dtype) {
  const auto s = to_u64(source);
  const auto p = to_u64(target);
  const int64_t w = montgomery_radix_bits(dtype);
  check_moduli(s, w, "crt_basis_extension_tables");
  check_moduli(p, w, "crt_basis_extension_tables");
  std::vector<std::vector<uint64_t>> matrix(s.size(),
                                            std::vector<uint64_t>(p.size()));
  std::vector<uint64_t> inverses(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    std::vector<uint64_t> others;
    for (size_t j = 0; j < s.size(); ++j) {
      if (j != i) {
        others.push_back(s[j]);
      }
    }
    inverses[i] = host::inv_mod(host::product_mod(others, s[i]), s[i]);
    for (size_t t = 0; t < p.size(); ++t) {
      uint64_t value = host::product_mod(others, p[t]);
      if (!normalized) {
        value = host::mul_mod(value, inverses[i], p[t]);
      }
      matrix[i][t] =
          host::mul_mod(value, host::power_of_two_mod(w, p[t]), p[t]);
    }
  }
  auto reciprocals = [](const std::vector<uint64_t>& moduli) {
    std::vector<uint64_t> values(moduli.size());
    for (size_t i = 0; i < moduli.size(); ++i) {
      values[i] = std::numeric_limits<uint64_t>::max() / moduli[i];
    }
    return to_tensor(values, at::kLong);
  };
  // Centered lift: -sum_i ((q_i - 1)/2) D_i R mod p_t.
  std::vector<uint64_t> offsets;
  if (normalized) {
    for (size_t t = 0; t < p.size(); ++t) {
      uint64_t sum = 0;
      for (size_t i = 0; i < s.size(); ++i) {
        sum = (sum + host::mul_mod((s[i] - 1) / 2 % p[t], matrix[i][t], p[t])) %
              p[t];
      }
      offsets.push_back((p[t] - sum) % p[t]);
    }
  }
  const std::vector<uint64_t> none;
  return {to_tensor(matrix, dtype, static_cast<int64_t>(p.size())),
          reciprocals(p), to_tensor(normalized ? inverses : none, dtype),
          reciprocals(normalized ? s : none), to_tensor(offsets, dtype)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
crt_quotient_tables(at::IntArrayRef source,
                    at::IntArrayRef target,
                    at::IntArrayRef divisor,
                    at::ScalarType dtype) {
  auto [unused_matrix, target_reciprocals, source_normalizers,
        source_reciprocals, unused_offsets] =
      crt_basis_extension_tables(source, target, true, dtype);
  const auto s = to_u64(source);
  const auto p = to_u64(target);
  const auto e = to_u64(divisor);
  check_moduli(e, montgomery_radix_bits(dtype), "crt_quotient_tables");
  std::vector<std::vector<uint64_t>> matrix(s.size(),
                                            std::vector<uint64_t>(p.size()));
  std::vector<uint64_t> offsets(p.size());
  for (size_t t = 0; t < p.size(); ++t) {
    const uint64_t divisor_mod = host::product_mod(e, p[t]);
    uint64_t sum = 0;
    for (size_t i = 0; i < s.size(); ++i) {
      const uint64_t inverse = host::inv_mod(
          host::mul_mod(s[i] % p[t], divisor_mod, p[t]), p[t]);
      matrix[i][t] = (p[t] - inverse) % p[t];
      sum = (sum + host::mul_mod((s[i] - 1) / 2 % p[t], matrix[i][t], p[t])) %
            p[t];
    }
    offsets[t] = (p[t] - sum) % p[t];
  }
  return {to_tensor(matrix, dtype, static_cast<int64_t>(p.size())),
          target_reciprocals, source_normalizers, source_reciprocals,
          to_tensor(offsets, dtype)};
}

at::Tensor trailing_quotient_inverses(at::IntArrayRef moduli,
                                      int64_t drop_count,
                                      at::ScalarType dtype) {
  const auto q = to_u64(moduli);
  const int64_t w = montgomery_radix_bits(dtype);
  check_moduli(q, w, "trailing_quotient_inverses");
  TORCH_CHECK_VALUE(
      drop_count >= 0 && drop_count < static_cast<int64_t>(q.size()),
      "trailing_quotient_inverses requires 0 <= drop_count < moduli");
  const size_t columns = q.size() - 1;
  std::vector<std::vector<uint64_t>> rows(static_cast<size_t>(drop_count),
                                          std::vector<uint64_t>(columns));
  for (size_t j = 0; j < rows.size(); ++j) {
    const uint64_t prime = q[q.size() - 1 - j];
    for (size_t i = 0; i + j + 1 < q.size(); ++i) {
      rows[j][i] =
          host::mul_mod(host::inv_mod(prime % q[i], q[i]),
                        host::power_of_two_mod(w, q[i]), q[i]);
    }
  }
  return to_tensor(rows, dtype, static_cast<int64_t>(columns));
}

at::Tensor rescale_inverse_matrix(at::IntArrayRef moduli,
                                  at::ScalarType dtype) {
  const auto q = to_u64(moduli);
  const int64_t w = montgomery_radix_bits(dtype);
  check_moduli(q, w, "rescale_inverse_matrix");
  std::vector<std::vector<uint64_t>> rows(q.size(),
                                          std::vector<uint64_t>(q.size()));
  for (size_t d = 0; d < q.size(); ++d) {
    for (size_t i = 0; i < q.size(); ++i) {
      if (i != d) {
        rows[d][i] =
            host::mul_mod(host::inv_mod(q[d] % q[i], q[i]),
                          host::power_of_two_mod(w, q[i]), q[i]);
      }
    }
  }
  return to_tensor(rows, dtype, static_cast<int64_t>(q.size()));
}

}  // namespace fhelium::rns
