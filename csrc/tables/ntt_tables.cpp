#include <fhelium/ntt_tables.h>

#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

#include "host_modular.h"
#include "table_support.h"

namespace fhelium::ntt {
namespace {

namespace host = ::fhelium::detail_host;
using ::fhelium::detail_tables::check_moduli;
using ::fhelium::detail_tables::montgomery_radix_bits;
using ::fhelium::detail_tables::to_tensor;
using ::fhelium::detail_tables::to_u64;

void check_log_n(int64_t log_n, const char* operation) {
  TORCH_CHECK_VALUE(log_n >= 1 && log_n <= 30, operation,
                    " requires 1 <= log_n <= 30, got ", log_n);
}

uint64_t bit_reverse(uint64_t value, int64_t bits) {
  uint64_t result = 0;
  for (int64_t i = 0; i < bits; ++i) {
    result = (result << 1) | ((value >> i) & 1);
  }
  return result;
}

uint64_t root_of(uint64_t modulus, int64_t log_n) {
  const uint64_t n = uint64_t{1} << log_n;
  TORCH_CHECK_VALUE((modulus - 1) % (2 * n) == 0, "modulus ", modulus,
                    " does not satisfy q = 1 mod 2N for N = ", n);
  const uint64_t exponent = (modulus - 1) / (2 * n);
  for (uint64_t candidate = 2; candidate < n; ++candidate) {
    const uint64_t root = host::pow_mod(candidate, exponent, modulus);
    if (host::pow_mod(root, n, modulus) != 1) {
      return root;
    }
  }
  TORCH_CHECK(false, "no primitive root of order ", 2 * n, " modulo ",
              modulus);
  return 0;
}

std::vector<uint64_t> power_series(uint64_t root,
                                   uint64_t length,
                                   uint64_t modulus) {
  std::vector<uint64_t> powers(length);
  uint64_t value = 1;
  for (uint64_t i = 0; i < length; ++i) {
    powers[i] = value;
    value = host::mul_mod(value, root, modulus);
  }
  return powers;
}

// Rows k -> root^br(k) and root^-br(k) for a transform of 2^log_size points.
void bit_reversed_rows(uint64_t root,
                       uint64_t modulus,
                       int64_t log_size,
                       std::vector<uint64_t>& forward,
                       std::vector<uint64_t>& inverse) {
  const uint64_t size = uint64_t{1} << log_size;
  const auto forward_powers = power_series(root, size, modulus);
  const auto inverse_powers =
      power_series(host::inv_mod(root, modulus), size, modulus);
  forward.resize(size);
  inverse.resize(size);
  for (uint64_t k = 0; k < size; ++k) {
    const uint64_t e = bit_reverse(k, log_size);
    forward[k] = forward_powers[e];
    inverse[k] = inverse_powers[e];
  }
}

struct Schedule {
  std::vector<int32_t> even;
  std::vector<int32_t> odd;
  std::vector<int64_t> twiddle;
};

Schedule forward_schedule(int64_t log_size) {
  const int64_t size = int64_t{1} << log_size;
  Schedule schedule;
  int64_t half_span = size;
  for (int64_t stage = 0; stage < log_size; ++stage) {
    const int64_t groups = int64_t{1} << stage;
    half_span /= 2;
    for (int64_t group = 0; group < groups; ++group) {
      const int64_t first = 2 * group * half_span;
      for (int64_t e = first; e < first + half_span; ++e) {
        schedule.even.push_back(static_cast<int32_t>(e));
        schedule.odd.push_back(static_cast<int32_t>(e + half_span));
        schedule.twiddle.push_back(groups + group);
      }
    }
  }
  return schedule;
}

Schedule inverse_schedule(int64_t log_size) {
  Schedule schedule;
  int64_t half_span = 1;
  for (int64_t remaining = log_size; remaining > 0; --remaining) {
    const int64_t groups = int64_t{1} << (remaining - 1);
    int64_t first = 0;
    for (int64_t group = 0; group < groups; ++group) {
      for (int64_t e = first; e < first + half_span; ++e) {
        schedule.even.push_back(static_cast<int32_t>(e));
        schedule.odd.push_back(static_cast<int32_t>(e + half_span));
        schedule.twiddle.push_back(groups + group);
      }
      first += 2 * half_span;
    }
    half_span *= 2;
  }
  return schedule;
}

at::Tensor index_tensor(const Schedule& schedule,
                        int64_t stages,
                        int64_t butterflies,
                        bool stacked) {
  auto out = at::empty(stacked ? std::vector<int64_t>{2, stages, butterflies}
                               : std::vector<int64_t>{stages, butterflies},
                       at::TensorOptions().dtype(at::kInt));
  auto* data = static_cast<int32_t*>(out.data_ptr());
  const size_t count = schedule.even.size();
  for (size_t i = 0; i < count; ++i) {
    data[i] = schedule.even[i];
    if (stacked) {
      data[count + i] = schedule.odd[i];
    }
  }
  return out;
}

at::Tensor gather_twiddles(const std::vector<std::vector<uint64_t>>& compact,
                           const Schedule& schedule,
                           int64_t stages,
                           int64_t butterflies,
                           at::ScalarType dtype) {
  std::vector<std::vector<uint64_t>> rows(compact.size());
  for (size_t r = 0; r < compact.size(); ++r) {
    rows[r].reserve(schedule.twiddle.size());
    for (const int64_t column : schedule.twiddle) {
      rows[r].push_back(compact[r][column]);
    }
  }
  return to_tensor(rows, dtype, stages * butterflies)
      .view({static_cast<int64_t>(compact.size()), stages, butterflies});
}

}  // namespace

int64_t negacyclic_root(int64_t modulus, int64_t log_n) {
  check_log_n(log_n, "negacyclic_root");
  TORCH_CHECK_VALUE(modulus > 2 && (modulus & 1) != 0,
                    "negacyclic_root requires an odd modulus above two");
  return static_cast<int64_t>(root_of(static_cast<uint64_t>(modulus), log_n));
}

std::tuple<at::Tensor, at::Tensor> compact_twiddles(at::IntArrayRef moduli,
                                                    int64_t log_n,
                                                    at::ScalarType dtype,
                                                    bool shoup32) {
  check_log_n(log_n, "compact_twiddles");
  const auto q = to_u64(moduli);
  check_moduli(q, montgomery_radix_bits(dtype), "compact_twiddles");
  TORCH_CHECK_VALUE(!shoup32 || dtype == at::kInt,
                    "Shoup32 twiddles require int32 storage");
  std::vector<std::vector<uint64_t>> forward(q.size());
  std::vector<std::vector<uint64_t>> inverse(q.size());
  for (size_t r = 0; r < q.size(); ++r) {
    bit_reversed_rows(root_of(q[r], log_n), q[r], log_n, forward[r],
                      inverse[r]);
  }
  const int64_t n = int64_t{1} << log_n;
  if (!shoup32) {
    return {to_tensor(forward, dtype, n), to_tensor(inverse, dtype, n)};
  }
  auto pairs = [&](const std::vector<std::vector<uint64_t>>& rows) {
    auto out = at::empty({static_cast<int64_t>(q.size()), n, 2},
                         at::TensorOptions().dtype(at::kInt));
    auto* data = static_cast<int32_t*>(out.data_ptr());
    for (size_t r = 0; r < q.size(); ++r) {
      for (int64_t k = 0; k < n; ++k) {
        const uint64_t w = rows[r][k];
        const size_t offset = (r * n + k) * 2;
        data[offset] = static_cast<int32_t>(w);
        data[offset + 1] =
            static_cast<int32_t>(static_cast<uint32_t>((w << 32) / q[r]));
      }
    }
    return out;
  };
  return {pairs(forward), pairs(inverse)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
power_of_two_radix_twiddles(at::IntArrayRef moduli,
                            int64_t log_n,
                            int64_t radix,
                            at::ScalarType dtype) {
  check_log_n(log_n, "power_of_two_radix_twiddles");
  TORCH_CHECK_VALUE(radix == 4 || radix == 8 || radix == 16,
                    "radix must be 4, 8, or 16");
  const int64_t radix_bits = radix == 4 ? 2 : (radix == 8 ? 3 : 4);
  TORCH_CHECK_VALUE(log_n % radix_bits == 0,
                    "the radix width must divide log_n");
  const auto q = to_u64(moduli);
  check_moduli(q, montgomery_radix_bits(dtype), "power_of_two_radix_twiddles");
  const uint64_t n = uint64_t{1} << log_n;
  const uint64_t order = 2 * n;
  auto outer_exponents = [&](bool inverse_execution) {
    std::vector<uint64_t> exponents;
    exponents.reserve(n - 1);
    for (int64_t start = 0; start < log_n; start += radix_bits) {
      const int64_t forward_start =
          inverse_execution ? log_n - start - radix_bits : start;
      const int64_t groups = int64_t{1} << forward_start;
      const int64_t leaf = forward_start + radix_bits - 1;
      for (int64_t group = 0; group < groups; ++group) {
        const uint64_t compact = (uint64_t{1} << leaf) +
                                 static_cast<uint64_t>(group * (radix / 2));
        const uint64_t beta = bit_reverse(compact, log_n);
        for (int64_t lane = 1; lane < radix; ++lane) {
          exponents.push_back(beta * static_cast<uint64_t>(lane) % order);
        }
      }
    }
    return exponents;
  };
  const auto forward_exponents = outer_exponents(false);
  const auto inverse_exponents = outer_exponents(true);
  std::vector<std::vector<uint64_t>> forward_outer(q.size());
  std::vector<std::vector<uint64_t>> inverse_outer(q.size());
  std::vector<std::vector<uint64_t>> forward_roots(q.size());
  std::vector<std::vector<uint64_t>> inverse_roots(q.size());
  for (size_t r = 0; r < q.size(); ++r) {
    const uint64_t root = root_of(q[r], log_n);
    const auto forward_powers = power_series(root, order, q[r]);
    const auto inverse_powers =
        power_series(host::inv_mod(root, q[r]), order, q[r]);
    for (const uint64_t e : forward_exponents) {
      forward_outer[r].push_back(forward_powers[e]);
    }
    for (const uint64_t e : inverse_exponents) {
      inverse_outer[r].push_back(inverse_powers[e]);
    }
    for (int64_t k = 0; k < radix; ++k) {
      const uint64_t e = static_cast<uint64_t>(k) * (order / radix);
      forward_roots[r].push_back(forward_powers[e]);
      inverse_roots[r].push_back(inverse_powers[e]);
    }
  }
  const auto outer = static_cast<int64_t>(n - 1);
  return {to_tensor(forward_outer, dtype, outer),
          to_tensor(inverse_outer, dtype, outer),
          to_tensor(forward_roots, dtype, radix),
          to_tensor(inverse_roots, dtype, radix)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
indexed_radix2_plan(at::IntArrayRef moduli,
                    int64_t log_n,
                    at::ScalarType dtype) {
  check_log_n(log_n, "indexed_radix2_plan");
  const auto q = to_u64(moduli);
  check_moduli(q, montgomery_radix_bits(dtype), "indexed_radix2_plan");
  std::vector<std::vector<uint64_t>> forward(q.size());
  std::vector<std::vector<uint64_t>> inverse(q.size());
  for (size_t r = 0; r < q.size(); ++r) {
    bit_reversed_rows(root_of(q[r], log_n), q[r], log_n, forward[r],
                      inverse[r]);
  }
  const int64_t butterflies = (int64_t{1} << log_n) / 2;
  const auto forward_plan = forward_schedule(log_n);
  const auto inverse_plan = inverse_schedule(log_n);
  return {
      index_tensor(forward_plan, log_n, butterflies, true),
      gather_twiddles(forward, forward_plan, log_n, butterflies, dtype),
      index_tensor(inverse_plan, log_n, butterflies, true),
      gather_twiddles(inverse, inverse_plan, log_n, butterflies, dtype)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> subring_indexed_radix2_plan(
    at::IntArrayRef moduli,
    int64_t log_n,
    int64_t log_extent,
    at::ScalarType dtype) {
  check_log_n(log_n, "subring_indexed_radix2_plan");
  TORCH_CHECK_VALUE(log_extent >= 1 && log_extent < log_n,
                    "subring_indexed_radix2_plan requires 1 <= log_extent < "
                    "log_n");
  const auto q = to_u64(moduli);
  check_moduli(q, montgomery_radix_bits(dtype), "subring_indexed_radix2_plan");
  std::vector<std::vector<uint64_t>> compact(q.size());
  std::vector<uint64_t> unused;
  for (size_t r = 0; r < q.size(); ++r) {
    const uint64_t root = host::pow_mod(
        root_of(q[r], log_n), uint64_t{1} << (log_n - log_extent), q[r]);
    bit_reversed_rows(root, q[r], log_extent, compact[r], unused);
  }
  const int64_t butterflies = (int64_t{1} << log_extent) / 2;
  const auto plan = forward_schedule(log_extent);
  Schedule odd{plan.odd, {}, {}};
  return {index_tensor(plan, log_extent, butterflies, false),
          index_tensor(odd, log_extent, butterflies, false),
          gather_twiddles(compact, plan, log_extent, butterflies, dtype)};
}

at::Tensor galois_indices(int64_t log_n, int64_t galois_element) {
  check_log_n(log_n, "galois_indices");
  const int64_t n = int64_t{1} << log_n;
  const int64_t order = 2 * n;
  const int64_t g = ((galois_element % order) + order) % order;
  TORCH_CHECK_VALUE((g & 1) != 0, "galois_element must be odd");
  auto out = at::empty({n}, at::TensorOptions().dtype(at::kInt));
  auto* data = static_cast<int32_t*>(out.data_ptr());
  for (int64_t i = 0; i < n; ++i) {
    const auto r = static_cast<int64_t>(bit_reverse(i, log_n));
    const int64_t position = ((2 * r + 1) * g % order - 1) / 2;
    data[i] =
        static_cast<int32_t>(bit_reverse(static_cast<uint64_t>(position), log_n));
  }
  return out;
}

}  // namespace fhelium::ntt
