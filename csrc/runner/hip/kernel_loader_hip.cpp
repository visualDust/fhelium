#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>

#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../kernel_loader.h"

namespace fhelium::exec::detail {
namespace {

void check(hipError_t status, const char* what, const std::string& name) {
  TORCH_CHECK(status == hipSuccess,
              what,
              " failed for kernel ",
              name,
              ": ",
              hipGetErrorString(status));
}

// A HIP module belongs to the context of the device that loads it, so the
// kernel loads once per device on its first launch there.
class HipKernel final : public LoadedKernel {
 public:
  explicit HipKernel(const KernelRecord& record)
      : name_(record.name),
        image_(record.image),
        block_(static_cast<unsigned>(record.warp_size * record.num_warps)),
        shared_(static_cast<unsigned>(record.shared)) {
    TORCH_CHECK_VALUE(record.image_kind == "hsaco",
                      "HIP kernel ",
                      name_,
                      " has image kind ",
                      record.image_kind);
    TORCH_CHECK_VALUE(record.num_ctas == 1,
                      "HIP kernel ",
                      name_,
                      " requests ",
                      record.num_ctas,
                      " blocks per program");
  }

  ~HipKernel() override {
    for (auto& [device, loaded] : loaded_) hipModuleUnload(loaded.first);
  }

  void launch(int64_t device,
              const std::array<int64_t, 3>& grid,
              void** parameters,
              void* stream) override {
    if (grid[0] * grid[1] * grid[2] == 0) return;
    check(hipModuleLaunchKernel(function(device),
                                static_cast<unsigned>(grid[0]),
                                static_cast<unsigned>(grid[1]),
                                static_cast<unsigned>(grid[2]),
                                block_,
                                1,
                                1,
                                shared_,
                                static_cast<hipStream_t>(stream),
                                parameters,
                                nullptr),
          "hipModuleLaunchKernel",
          name_);
  }

  void prepare(int64_t device) override {
    int current = 0;
    check(hipGetDevice(&current), "hipGetDevice", name_);
    check(hipSetDevice(static_cast<int>(device)), "hipSetDevice", name_);
    try {
      function(device);
    } catch (...) {
      hipSetDevice(current);
      throw;
    }
    hipSetDevice(current);
  }

 private:
  // Loads the module into the context of the current device `device` on the
  // first request.
  hipFunction_t function(int64_t device) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [loaded_device, loaded] : loaded_)
      if (loaded_device == device) return loaded.second;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    check(
        hipModuleLoadData(&module, image_.data()), "hipModuleLoadData", name_);
    check(hipModuleGetFunction(&function, module, name_.c_str()),
          "hipModuleGetFunction",
          name_);
    loaded_.emplace_back(device, std::make_pair(module, function));
    return function;
  }

  std::string name_;
  std::vector<uint8_t> image_;
  unsigned block_;
  unsigned shared_;
  std::mutex mutex_;
  std::vector<std::pair<int64_t, std::pair<hipModule_t, hipFunction_t>>>
      loaded_;
};

}  // namespace

std::unique_ptr<LoadedKernel> load_kernel(const KernelRecord& record) {
  return std::make_unique<HipKernel>(record);
}

std::string device_architecture(int64_t device) {
  hipDeviceProp_t properties;
  check(hipGetDeviceProperties(&properties, static_cast<int>(device)),
        "hipGetDeviceProperties",
        "architecture query");
  const std::string name = properties.gcnArchName;
  return name.substr(0, name.find(':'));
}

std::string runtime_version() { return "hip-" + std::to_string(HIP_VERSION); }

}  // namespace fhelium::exec::detail
