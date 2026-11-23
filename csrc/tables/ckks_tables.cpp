#include <fhelium/ckks_tables.h>

#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "host_modular.h"

namespace fhelium::ckks {
namespace {

namespace host = ::fhelium::detail_host;

constexpr double kPi = 3.141592653589793;

void check_generator(int64_t generator) {
  TORCH_CHECK_VALUE(generator == 3 || generator == 5,
                    "generator must be 3 or 5, got ", generator);
}

void check_log_n(int64_t log_n, const char* operation) {
  TORCH_CHECK_VALUE(log_n >= 1 && log_n <= 30, operation,
                    " requires 1 <= log_n <= 30, got ", log_n);
}

at::Tensor long_tensor(const std::vector<int64_t>& values) {
  auto out = at::empty({static_cast<int64_t>(values.size())},
                       at::TensorOptions().dtype(at::kLong));
  std::copy(values.begin(), values.end(),
            static_cast<int64_t*>(out.data_ptr()));
  return out;
}

// Cycles of a permutation, each starting at the image of its smallest
// remaining element.
std::vector<std::vector<int64_t>> cycles(const std::vector<int64_t>& perm) {
  std::vector<bool> present(perm.size(), true);
  std::vector<std::vector<int64_t>> result;
  size_t first = 0;
  while (true) {
    while (first < perm.size() && !present[first]) {
      ++first;
    }
    if (first == perm.size()) {
      return result;
    }
    std::vector<int64_t> cycle;
    int64_t current = perm[first];
    int64_t following = perm[current];
    while (true) {
      cycle.push_back(current);
      present[current] = false;
      current = following;
      if (!present[following]) {
        break;
      }
      following = perm[following];
    }
    result.push_back(std::move(cycle));
  }
}

}  // namespace

int64_t rotation_galois_element(int64_t log_n,
                                int64_t step,
                                int64_t generator) {
  check_log_n(log_n, "rotation_galois_element");
  check_generator(generator);
  const int64_t n = int64_t{1} << log_n;
  const int64_t exponent = generator == 5 ? -step : step;
  const int64_t reduced = ((exponent % n) + n) % n;
  return static_cast<int64_t>(
      host::pow_mod(static_cast<uint64_t>(generator),
                    static_cast<uint64_t>(reduced), static_cast<uint64_t>(2 * n)));
}

std::tuple<at::Tensor, at::Tensor> coefficient_galois_indices(
    int64_t log_n,
    int64_t galois_element) {
  check_log_n(log_n, "coefficient_galois_indices");
  const int64_t n = int64_t{1} << log_n;
  const int64_t order = 2 * n;
  const int64_t g = ((galois_element % order) + order) % order;
  TORCH_CHECK_VALUE((g & 1) != 0, "galois_element must be odd");
  auto indices = at::empty({n}, at::TensorOptions().dtype(at::kInt));
  auto signs = at::empty({n}, at::TensorOptions().dtype(at::kChar));
  auto* index_data = static_cast<int32_t*>(indices.data_ptr());
  auto* sign_data = static_cast<int8_t*>(signs.data_ptr());
  for (int64_t source = 0; source < n; ++source) {
    const int64_t mapped = g * source % order;
    const int64_t destination = mapped % n;
    index_data[destination] = static_cast<int32_t>(source);
    sign_data[destination] = ((mapped / n) & 1) != 0 ? -1 : 1;
  }
  return {indices, signs};
}

std::tuple<at::Tensor, at::Tensor> slot_permutations(int64_t log_n,
                                                     int64_t generator) {
  check_log_n(log_n, "slot_permutations");
  check_generator(generator);
  const int64_t n = int64_t{1} << log_n;
  const int64_t half = n / 2;
  std::vector<int64_t> pre(half);
  std::vector<int64_t> post(n);
  if (generator == 5) {
    uint64_t value = 1;
    for (int64_t i = 0; i < half; ++i) {
      pre[i] = static_cast<int64_t>((value - 1) / 2);
      value = value * 5 % static_cast<uint64_t>(2 * n);
    }
    for (int64_t i = 0; i < half; ++i) {
      post[pre[i]] = i;
      post[n - 1 - pre[i]] = n - 1 - i;
    }
    return {long_tensor(pre), long_tensor(post)};
  }
  std::vector<int64_t> circular(n);
  std::vector<int64_t> folded(n);
  for (int64_t i = 0; i < half; ++i) {
    circular[i] = (i - 1 + half) % half;
    circular[half + i] = (i + 1) % half + half;
  }
  for (int64_t k = 0; k < n; ++k) {
    folded[k] = (3 * (2 * k + 1) % (2 * n) - 1) / 2;
  }
  const auto first_cycles = cycles(circular);
  const auto second_cycles = cycles(folded);
  TORCH_CHECK(first_cycles.size() == second_cycles.size(),
              "slot permutation cycle structures differ");
  for (size_t c = 0; c < first_cycles.size(); ++c) {
    TORCH_CHECK(first_cycles[c].size() == second_cycles[c].size(),
                "slot permutation cycle structures differ");
    for (size_t i = 0; i < first_cycles[c].size(); ++i) {
      post[second_cycles[c][i]] = first_cycles[c][i];
    }
  }
  std::vector<int64_t> inverse(n);
  for (int64_t k = 0; k < n; ++k) {
    inverse[post[k]] = k;
  }
  std::copy(inverse.begin(), inverse.begin() + half, pre.begin());
  return {long_tensor(pre), long_tensor(post)};
}

int64_t periodic_extent(int64_t period, int64_t log_n, int64_t generator) {
  check_log_n(log_n, "periodic_extent");
  check_generator(generator);
  const int64_t n = int64_t{1} << log_n;
  TORCH_CHECK_VALUE(period >= 1 && (period & (period - 1)) == 0 &&
                        (n / 2) % period == 0,
                    "Periodic slot count must be a power of two dividing N/2");
  const int64_t extent = std::max<int64_t>(generator == 3 ? 4 : 2, 2 * period);
  TORCH_CHECK_VALUE(extent < n,
                    "The periodic message must admit a shorter encoded axis");
  return extent;
}

at::Tensor periodic_slot_positions(int64_t extent, int64_t generator) {
  check_generator(generator);
  TORCH_CHECK_VALUE(extent >= 2 && (extent & (extent - 1)) == 0,
                    "extent must be a power of two of at least two");
  const auto order = static_cast<uint64_t>(2 * extent);
  const uint64_t step = generator == 5 ? 5 % order : host::inv_mod(3, order);
  std::vector<int64_t> positions(extent / 2);
  uint64_t value = 1 % order;
  for (int64_t i = 0; i < extent / 2; ++i) {
    positions[i] = (static_cast<int64_t>(value) - 1) / 2;
    value = host::mul_mod(value, step, order);
  }
  return long_tensor(positions);
}

at::Tensor slot_twist(int64_t length, bool inverse, c10::Device device) {
  TORCH_CHECK_VALUE(length >= 1, "slot_twist requires a positive length");
  const auto indices = at::arange(
      length, at::TensorOptions().dtype(at::kDouble).device(device));
  const c10::complex<double> unit(inverse ? 0.0 : -0.0,
                                  inverse ? kPi : -kPi);
  return indices.mul(unit).div(length).exp();
}

}  // namespace fhelium::ckks
