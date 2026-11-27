// A compiled artifact loaded and called from C++.
//
// Usage: load_artifact [--seed SEED] ARTIFACT OUTPUT INPUT KEY...
//
// ARTIFACT is a file written by fhelium.compile.export, INPUT the value file
// of the argument of its first signature, and each KEY a value file of an
// evaluation key the artifact declares. The program loads the artifact on the
// first CUDA device (or the CPU without one), binds each key to the materials
// with its role and layout, expanding compact keys and skipping keys the
// artifact does not use, builds the tables and prepared plaintexts the
// artifact records, and runs the
// signature's entry with automatic CUDA Graph selection. It writes the result
// to OUTPUT as a value file that fhelium.load_value reads, and prints the
// latency of the first call and the median of the following calls. Plaintext
// encoding during preparation rounds with a random stream seeded from the
// operating system, or derived from SEED as fhelium.Engine(rng_seed=SEED)
// derives it.

#include <fhelium/artifact.h>
#include <fhelium/values.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

void synchronize(const torch::Device& device) {
  if (device.is_cuda()) torch::cuda::synchronize(device.index());
}

}  // namespace

int main(int argc, char** argv) {
  std::optional<int64_t> seed;
  if (argc > 2 && std::string(argv[1]) == "--seed") {
    seed = std::stoll(argv[2]);
    argv[2] = argv[0];
    argc -= 2;
    argv += 2;
  }
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " [--seed SEED] ARTIFACT OUTPUT INPUT KEY...\n";
    return 2;
  }
  const torch::Device device = torch::cuda::is_available()
                                   ? torch::Device(torch::kCUDA, 0)
                                   : torch::Device(torch::kCPU);
  using Clock = std::chrono::steady_clock;
  const auto seconds = [](Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
  };

  auto begin = Clock::now();
  auto artifact = c10::make_intrusive<fhelium::Artifact>(argv[1], device);
  size_t bound = 0;
  for (int i = 4; i < argc; ++i)
    bound +=
        artifact->bind_key(fhelium::io::load_value(argv[i], device)).size();
  std::cout << "bound " << bound << " key materials from " << argc - 4
            << " key files\n";
  if (seed) artifact->seed_rounding(seed);
  artifact->prepare();
  const auto signature = artifact->signatures().at(0);
  auto runner = artifact->runner(signature.entry, std::nullopt, "auto", false);
  synchronize(device);
  std::cout << "load, key binding and preparation: " << seconds(begin)
            << " s\n";

  const auto input = fhelium::io::load_value(argv[3], device);
  std::vector<at::Tensor> outputs;
  std::vector<double> latencies;
  for (int call = 0; call < 20; ++call) {
    begin = Clock::now();
    outputs = runner->run({input.tensors.at("data")}, {}, {});
    synchronize(device);
    latencies.push_back(seconds(begin));
  }
  std::cout << "first call: " << latencies[0] * 1e3 << " ms\n";
  std::vector<double> steady(latencies.begin() + 1, latencies.end());
  std::nth_element(
      steady.begin(), steady.begin() + steady.size() / 2, steady.end());
  std::cout << "median of " << steady.size()
            << " further calls: " << steady[steady.size() / 2] * 1e3 << " ms\n";

  auto result = signature.results.at(0);
  result.tensors["data"] = outputs.at(0);
  fhelium::io::save_value(argv[2], result, false, true);
  std::cout << "wrote " << result.type << " to " << argv[2] << "\n";
  return 0;
}
