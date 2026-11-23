#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <fhelium/random.h>

#if defined(_WIN32)
#include <windows.h>
// clang-format off
#include <bcrypt.h>
// clang-format on
#pragma comment(lib, "bcrypt.lib")
#elif defined(__APPLE__)
#include <sys/random.h>
#else
#include <unistd.h>
#endif

namespace fhelium::random {

namespace {

void system_random_bytes(void* out, size_t size) {
#if defined(_WIN32)
  const NTSTATUS status = BCryptGenRandom(nullptr,
                                          static_cast<PUCHAR>(out),
                                          static_cast<ULONG>(size),
                                          BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  TORCH_CHECK(status >= 0, "BCryptGenRandom failed with status ", status);
#else
  TORCH_CHECK(size <= 256 && getentropy(out, size) == 0,
              "getentropy failed to supply ",
              size,
              " bytes");
#endif
}

// Double-double arithmetic: value = hi + lo with |lo| <= ulp(hi) / 2.
struct DoubleDouble {
  double hi;
  double lo;
};

DoubleDouble quick_two_sum(double a, double b) {
  const double s = a + b;
  return {s, b - (s - a)};
}

DoubleDouble two_sum(double a, double b) {
  const double s = a + b;
  const double v = s - a;
  return {s, (a - (s - v)) + (b - v)};
}

DoubleDouble operator+(DoubleDouble a, DoubleDouble b) {
  DoubleDouble s = two_sum(a.hi, b.hi);
  const DoubleDouble t = two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return quick_two_sum(s.hi, s.lo);
}

DoubleDouble operator-(DoubleDouble a) { return {-a.hi, -a.lo}; }

DoubleDouble operator-(DoubleDouble a, DoubleDouble b) { return a + (-b); }

DoubleDouble operator*(DoubleDouble a, DoubleDouble b) {
  const double p = a.hi * b.hi;
  const double e = std::fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
  return quick_two_sum(p, e);
}

DoubleDouble operator/(DoubleDouble a, DoubleDouble b) {
  const double q1 = a.hi / b.hi;
  DoubleDouble r = a - b * DoubleDouble{q1, 0.0};
  const double q2 = r.hi / b.hi;
  r = r - b * DoubleDouble{q2, 0.0};
  const double q3 = r.hi / b.hi;
  return DoubleDouble{q1, 0.0} + DoubleDouble{q2, 0.0} + DoubleDouble{q3, 0.0};
}

DoubleDouble ldexp(DoubleDouble a, int exponent) {
  return {std::ldexp(a.hi, exponent), std::ldexp(a.lo, exponent)};
}

// exp(-t) for t >= 0 by reduction t = n ln 2 - r with |r| <= ln(2) / 2.
DoubleDouble exp_negative(DoubleDouble t) {
  constexpr DoubleDouble kLn2{6.931471805599452862e-01,
                              2.319046813846299558e-17};
  const double n = std::nearbyint(t.hi / kLn2.hi);
  const DoubleDouble r = DoubleDouble{n, 0.0} * kLn2 - t;
  DoubleDouble term{1.0, 0.0};
  DoubleDouble sum{1.0, 0.0};
  for (int i = 1; i <= 32; ++i) {
    term = term * r / DoubleDouble{static_cast<double>(i), 0.0};
    sum = sum + term;
  }
  return ldexp(sum, -static_cast<int>(n));
}

// Splits a fraction in [0, 1] into the high and low words of round-down
// 2^127 times it.
std::pair<uint64_t, uint64_t> fixed_point127(DoubleDouble fraction) {
  const DoubleDouble scaled = ldexp(fraction, 63);
  // floor(scaled) = floor(hi) + floor(remainder), where the remainder holds
  // the fractional part of hi plus lo and can exceed one in magnitude.
  const double high_part = std::floor(scaled.hi);
  DoubleDouble remainder = scaled - DoubleDouble{high_part, 0.0};
  double carry = std::floor(remainder.hi);
  remainder = remainder - DoubleDouble{carry, 0.0};
  if (remainder.hi + remainder.lo < 0.0) {
    carry -= 1.0;
    remainder = remainder + DoubleDouble{1.0, 0.0};
  }
  const double low_fraction = remainder.hi + remainder.lo;
  const uint64_t high = static_cast<uint64_t>(high_part) +
                        static_cast<uint64_t>(static_cast<int64_t>(carry));
  const double low_scaled = std::ldexp(low_fraction, 64);
  const uint64_t low = low_scaled >= 18446744073709551615.0
                           ? UINT64_MAX
                           : static_cast<uint64_t>(low_scaled);
  return {high, low};
}

}  // namespace

at::Tensor generate_seed() {
  int64_t words[5] = {};
  system_random_bytes(words, 4 * sizeof(int64_t));
  auto seed = at::empty({5}, at::TensorOptions().dtype(at::kLong));
  std::copy(words, words + 5, seed.mutable_data_ptr<int64_t>());
  return seed;
}

at::Tensor gaussian_cdt_table(double sigma) {
  TORCH_CHECK(std::isfinite(sigma) && sigma > 0.0,
              "gaussian_cdt_table requires a positive finite sigma");
  const auto bound = static_cast<int64_t>(
                         std::ceil(sigma * std::sqrt(256.0 * std::log(2.0)))) +
                     1;
  TORCH_CHECK(bound <= (int64_t{1} << 20),
              "gaussian_cdt_table supports sigma up to about 7.9e4");
  // rho(k) = exp(-k^2 / (2 sigma^2)); the normalizer sums through a bound
  // whose tail lies below 2^-140.
  const DoubleDouble variance2 = DoubleDouble{2.0, 0.0} *
                                 DoubleDouble{sigma, 0.0} *
                                 DoubleDouble{sigma, 0.0};
  const auto sum_bound = static_cast<int64_t>(std::ceil(
                             sigma * std::sqrt(280.0 * std::log(2.0)))) +
                         2;
  std::vector<DoubleDouble> weights(static_cast<size_t>(sum_bound) + 1);
  DoubleDouble normalizer{0.0, 0.0};
  for (int64_t k = 0; k <= sum_bound; ++k) {
    const double square = static_cast<double>(k) * static_cast<double>(k);
    const DoubleDouble rho =
        exp_negative(DoubleDouble{square, 0.0} / variance2);
    weights[static_cast<size_t>(k)] =
        k == 0 ? rho : DoubleDouble{2.0, 0.0} * rho;
    normalizer = normalizer + weights[static_cast<size_t>(k)];
  }
  auto table = at::empty({bound, 2}, at::TensorOptions().dtype(at::kLong));
  auto* rows = table.mutable_data_ptr<int64_t>();
  DoubleDouble cumulative{0.0, 0.0};
  for (int64_t k = 0; k < bound; ++k) {
    cumulative = cumulative + weights[static_cast<size_t>(k)];
    const auto [high, low] = fixed_point127(cumulative / normalizer);
    rows[2 * k] = static_cast<int64_t>(high);
    rows[2 * k + 1] = static_cast<int64_t>(low);
  }
  return table;
}

}  // namespace fhelium::random
