#pragma once

// Loading and launching compiled device kernels for the Runner. The CUDA
// implementation uses the CUDA runtime library API and the HIP implementation
// the HIP module API, so libfhelium does not link the CUDA driver library.

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "fhelium/runner.h"

namespace fhelium::exec::detail {

class LoadedKernel {
 public:
  virtual ~LoadedKernel() = default;
  // Loads the kernel into the context of `device` and applies its launch
  // attributes there. Runner::finalize prepares every launched (kernel,
  // device) pair, so no load happens during a run or a graph capture.
  virtual void prepare(int64_t device) = 0;
  // Launches on a prepared `device` on `device` and `stream` with one pointer
  // per kernel parameter, the two scratch pointers included. The grid counts
  // programs; each program is a cluster of num_ctas blocks.
  virtual void launch(int64_t device,
                      const std::array<int64_t, 3>& grid,
                      void** parameters,
                      void* stream) = 0;
};

std::unique_ptr<LoadedKernel> load_kernel(const KernelRecord& record);

// Kernel image architecture of `device`: "sm_<major><minor>" on CUDA, the
// gfx target name on HIP.
std::string device_architecture(int64_t device);

// GPU runtime of the build: "cuda-<major>.<minor>", "hip-<version>" or "none".
std::string runtime_version();

}  // namespace fhelium::exec::detail
