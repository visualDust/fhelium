// An exec schedule built and run with the FHElium Runner from C++.
//
// The schedule takes RNS residues x over two primes and an integer count k,
// and computes y = sum over i in range(k) of (x if i is even, -x if i is odd)
// with the dispatcher operators fhelium_rns_ops::add_standard_ and
// fhelium_rns_ops::negate_standard: the loop and the parity test run on host
// registers, the parameter table is a bound material buffer, and the
// accumulator is a static arena buffer that the schedule allocates and zeroes.
// The program compares y with a direct libfhelium evaluation for several k on
// CPU and, when available, on the first CUDA device. On CUDA, a second
// schedule computes 3x = x + x - x' with x' = -x inside one device-graph
// segment into a placed arena buffer: the first run executes the segment
// directly, the second captures it, and later runs replay the graph with the
// new input staged into the captured storage. The program returns a nonzero
// status when a check fails.

#include <fhelium/fhelium.h>
#include <fhelium/runner.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using fhelium::exec::ArgumentKind;
using fhelium::exec::Runner;

int64_t kind(ArgumentKind value) { return static_cast<int64_t>(value); }

c10::intrusive_ptr<Runner> build(const at::Tensor& parameters,
                                 const std::vector<int64_t>& shape,
                                 const torch::Device& device) {
  auto runner = c10::make_intrusive<Runner>();
  const auto x = runner->buffer("input", {}, {}, {}, {});
  const auto table = runner->buffer("material", {}, {}, {}, {});
  const auto strides = std::vector<int64_t>{shape[1], 1};
  const auto sum = runner->buffer("arena", shape, strides, at::kLong, device);
  const auto negated = runner->buffer("internal", {}, {}, {}, {});
  const auto count = runner->int_register();
  const auto zero = runner->int_register();
  const auto one = runner->int_register();
  const auto two = runner->int_register();
  const auto i = runner->int_register();
  const auto parity = runner->int_register();
  runner->bind(table, parameters);
  runner->inputs({x}, {count}, {});
  runner->outputs({sum});

  runner->int_constant(zero, 0);
  runner->int_constant(one, 1);
  runner->int_constant(two, 2);
  runner->alloc(sum);
  runner->zero(sum);
  const auto buffer = kind(ArgumentKind::buffer);
  runner->begin_for(i, zero, count, one);
  runner->int_binary("rem", parity, i, two);
  runner->begin_if(parity);
  runner->call_op("fhelium_rns_ops::negate_standard.default",
                  {buffer, buffer},
                  {x, table},
                  {negated});
  runner->call_op("fhelium_rns_ops::add_standard_.default",
                  {buffer, buffer, buffer},
                  {sum, negated, table},
                  {});
  runner->begin_else();
  runner->call_op("fhelium_rns_ops::add_standard_.default",
                  {buffer, buffer, buffer},
                  {sum, x, table},
                  {});
  runner->end_if();
  runner->end_for();
  runner->finalize();
  return runner;
}

bool check_device(const torch::Device& device) {
  const std::vector<int64_t> moduli = {1152921504606584833LL,
                                       1152921504598720513LL};
  const std::vector<int64_t> shape = {2, 16};
  const auto parameters =
      fhelium::rns::parameter_table(moduli, 4, 1.0, torch::kInt64).to(device);
  const auto x = (torch::randint(0, 1LL << 60, shape, torch::kInt64) %
                  torch::tensor(moduli).view({2, 1}))
                     .to(device);
  const auto runner = build(parameters, shape, device);
  bool ok = true;
  for (const int64_t k : {0, 1, 2, 5}) {
    auto expected = torch::zeros_like(x);
    for (int64_t i = 0; i < k; ++i) {
      fhelium::rns::add_standard_(
          expected,
          i % 2 ? fhelium::rns::negate_standard(x, parameters) : x,
          parameters);
    }
    const auto result = runner->run({x}, {k}, {})[0];
    const bool match = result.equal(expected);
    std::cout << (match ? "PASS " : "FAIL ") << "k=" << k << " on " << device
              << "\n";
    ok &= match;
  }
  return ok;
}

// Builds a graph-enabled schedule whose single segment writes
// x + x - (-x) = 3x into a placed arena buffer.
c10::intrusive_ptr<Runner> build_segment(const at::Tensor& parameters,
                                         const std::vector<int64_t>& shape,
                                         const torch::Device& device) {
  auto runner = c10::make_intrusive<Runner>();
  runner->graphs("on");
  const auto x = runner->buffer("input", {}, {}, {}, {});
  const auto table = runner->buffer("material", {}, {}, {}, {});
  const auto strides = std::vector<int64_t>{shape[1], 1};
  const auto sum = runner->buffer("arena", shape, strides, at::kLong, device);
  const auto negated = runner->buffer("internal", {}, {}, {}, {});
  runner->place(sum, 0);
  runner->bind(table, parameters);
  runner->inputs({x}, {}, {});
  runner->outputs({sum});
  const auto buffer = kind(ArgumentKind::buffer);
  runner->begin_segment(device.index());
  runner->copy(sum, x);
  runner->call_op("fhelium_rns_ops::add_standard_.default",
                  {buffer, buffer, buffer},
                  {sum, x, table},
                  {});
  runner->call_op("fhelium_rns_ops::negate_standard.default",
                  {buffer, buffer},
                  {x, table},
                  {negated});
  runner->call_op("fhelium_rns_ops::sub_standard_.default",
                  {buffer, buffer, buffer},
                  {sum, negated, table},
                  {});
  runner->end_segment();
  runner->finalize();
  return runner;
}

bool check_segment(const torch::Device& device) {
  const std::vector<int64_t> moduli = {1152921504606584833LL,
                                       1152921504598720513LL};
  const std::vector<int64_t> shape = {2, 16};
  const auto parameters =
      fhelium::rns::parameter_table(moduli, 4, 1.0, torch::kInt64).to(device);
  const auto runner = build_segment(parameters, shape, device);
  bool ok = true;
  for (int run = 0; run < 4; ++run) {
    const auto x = (torch::randint(0, 1LL << 60, shape, torch::kInt64) %
                    torch::tensor(moduli).view({2, 1}))
                       .to(device);
    auto expected = x.clone();
    fhelium::rns::add_standard_(expected, x, parameters);
    fhelium::rns::add_standard_(expected, x, parameters);
    const bool match = runner->run({x}, {}, {})[0].equal(expected);
    std::cout << (match ? "PASS " : "FAIL ") << "segment run " << run << " on "
              << device << "\n";
    ok &= match;
  }
  // [segments, captures, replays, staged copies, output copies]
  const auto stats = runner->graph_stats();
  const bool replayed = stats[1] == 1 && stats[2] == 3;
  std::cout << (replayed ? "PASS " : "FAIL ") << "captures=" << stats[1]
            << " replays=" << stats[2] << "\n";
  return ok && replayed;
}

}  // namespace

int main() {
  bool ok = check_device(torch::Device(torch::kCPU));
  if (torch::cuda::is_available()) {
    ok &= check_device(torch::Device(torch::kCUDA, 0));
    ok &= check_segment(torch::Device(torch::kCUDA, 0));
  }
  return ok ? 0 : 1;
}
