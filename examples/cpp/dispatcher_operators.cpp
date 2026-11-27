// FHElium Torch operators called through c10::Dispatcher from C++.
//
// Linking FHElium::torch loads libfhelium_torch, whose static registrations
// define the fhelium_* operator schemas and their kernels. The program looks
// up operators by schema name, as a deployment runtime that executes a
// recorded operator sequence does, and compares each dispatched result with
// the direct libfhelium call: parameter-table construction, Montgomery
// conversion and multiplication, standard addition, and the in-place
// standard addition that mutates its first argument. It runs on CPU and, when
// available, on the first CUDA device, and returns a nonzero status when a
// check fails.

#include <ATen/core/dispatch/Dispatcher.h>
#include <fhelium/fhelium.h>
#include <fhelium/torch_library.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Binary = at::Tensor(const at::Tensor&,
                          const at::Tensor&,
                          const at::Tensor&);
using BinaryInPlace = void(at::Tensor&, const at::Tensor&, const at::Tensor&);
using ParameterTable = at::Tensor(c10::ArrayRef<int64_t>,
                                  int64_t,
                                  double,
                                  c10::ScalarType);

template <typename Signature>
c10::TypedOperatorHandle<Signature> find(const char* name) {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(name, "")
      .typed<Signature>();
}

bool check(bool ok, const char* label, const torch::Device& device) {
  std::cout << (ok ? "PASS " : "FAIL ") << label << " on " << device << "\n";
  return ok;
}

bool check_device(const torch::Device& device) {
  const std::vector<int64_t> moduli = {1152921504606584833LL,
                                       1152921504598720513LL};
  const int64_t n = 16;
  const auto limbs = static_cast<int64_t>(moduli.size());
  const auto parameter_table =
      find<ParameterTable>("fhelium_rns_ops::parameter_table");
  const auto params =
      parameter_table.call(moduli, 4, 1.0, at::kLong).to(device);
  bool ok = check(params.cpu().equal(fhelium::rns::parameter_table(
                      moduli, 4, 1.0, torch::kInt64)),
                  "parameter_table",
                  device);

  std::mt19937_64 rng(7);
  auto values = torch::empty({2, 2, limbs, n}, torch::kInt64);
  auto accessor = values.accessor<int64_t, 4>();
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t c = 0; c < 2; ++c) {
      for (int64_t j = 0; j < limbs; ++j) {
        for (int64_t k = 0; k < n; ++k) {
          accessor[i][c][j][k] = static_cast<int64_t>(rng() % moduli[j]);
        }
      }
    }
  }
  values = values.to(device);
  const auto lhs = values[0].clone();
  const auto rhs = values[1].clone();

  const auto add = find<Binary>("fhelium_rns_ops::add_standard");
  const auto sum = add.call(lhs, rhs, params);
  ok &= check(sum.equal(fhelium::rns::add_standard(lhs, rhs, params)),
              "add_standard",
              device);

  const auto add_ = find<BinaryInPlace>("fhelium_rns_ops::add_standard_");
  auto accumulated = lhs.clone();
  add_.call(accumulated, rhs, params);
  ok &= check(accumulated.equal(sum), "add_standard_", device);

  auto lhs_montgomery = lhs.clone();
  auto rhs_montgomery = rhs.clone();
  fhelium::rns::to_montgomery_(lhs_montgomery, params);
  fhelium::rns::to_montgomery_(rhs_montgomery, params);
  const auto multiply = find<Binary>("fhelium_rns_ops::montgomery_mul");
  ok &= check(multiply.call(lhs_montgomery, rhs_montgomery, params)
                  .equal(fhelium::rns::montgomery_mul(
                      lhs_montgomery, rhs_montgomery, params)),
              "montgomery_mul",
              device);
  return ok;
}

}  // namespace

int main() {
  fhelium::load_torch_operators();
  bool ok = check_device(torch::Device(torch::kCPU));
  if (torch::cuda::is_available()) {
    ok &= check_device(torch::Device(torch::kCUDA, 0));
  }
  return ok ? 0 : 1;
}
