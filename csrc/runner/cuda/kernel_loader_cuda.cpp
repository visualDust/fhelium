#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <string>
#include <vector>

#include "../kernel_loader.h"

namespace fhelium::exec::detail {
namespace {

void check(cudaError_t status, const char* what, const std::string& name) {
  TORCH_CHECK(status == cudaSuccess,
              what,
              " failed for kernel ",
              name,
              ": ",
              cudaGetErrorString(status));
}

// A CUDA library is context-independent: the runtime loads it into the
// context of each device on first use, so one library serves every device.
class CudaKernel final : public LoadedKernel {
 public:
  explicit CudaKernel(const KernelRecord& record)
      : name_(record.name),
        block_(static_cast<unsigned>(record.warp_size * record.num_warps)),
        num_ctas_(static_cast<unsigned>(record.num_ctas)),
        shared_(static_cast<size_t>(record.shared)) {
    TORCH_CHECK_VALUE(
        record.image_kind == "cubin" || record.image_kind == "ptx",
        "CUDA kernel ",
        name_,
        " has image kind ",
        record.image_kind);
    std::vector<uint8_t> image = record.image;
    if (record.image_kind == "ptx") image.push_back(0);
    check(
        cudaLibraryLoadData(
            &library_, image.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
        "cudaLibraryLoadData",
        name_);
    check(cudaLibraryGetKernel(&kernel_, library_, name_.c_str()),
          "cudaLibraryGetKernel",
          name_);
  }

  ~CudaKernel() override { cudaLibraryUnload(library_); }

  void launch(int64_t device,
              const std::array<int64_t, 3>& grid,
              void** parameters,
              void* stream) override {
    if (grid[0] * grid[1] * grid[2] == 0) return;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<unsigned>(grid[0]) * num_ctas_,
                          static_cast<unsigned>(grid[1]),
                          static_cast<unsigned>(grid[2]));
    config.blockDim = dim3(block_, 1, 1);
    config.dynamicSmemBytes = shared_;
    config.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attributes[2];
    if (num_ctas_ != 1) {
      attributes[0].id = cudaLaunchAttributeClusterDimension;
      attributes[0].val.clusterDim.x = num_ctas_;
      attributes[0].val.clusterDim.y = 1;
      attributes[0].val.clusterDim.z = 1;
      attributes[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
      attributes[1].val.clusterSchedulingPolicyPreference =
          cudaClusterSchedulingPolicySpread;
      config.attrs = attributes;
      config.numAttrs = 2;
    }
    check(cudaLaunchKernelExC(
              &config, reinterpret_cast<const void*>(kernel_), parameters),
          "cudaLaunchKernelEx",
          name_);
  }

  // Querying the function attributes loads the library into the device
  // context; dynamic shared memory above the 48 KiB default needs an opt-in
  // per device.
  void prepare(int64_t device) override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto prepared : prepared_)
      if (prepared == device) return;
    int current = 0;
    check(cudaGetDevice(&current), "cudaGetDevice", name_);
    check(cudaSetDevice(static_cast<int>(device)), "cudaSetDevice", name_);
    cudaFuncAttributes attributes = {};
    const auto status = cudaFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(kernel_));
    cudaSetDevice(current);
    check(status, "cudaFuncGetAttributes", name_);
    if (shared_ > 48 * 1024)
      check(cudaKernelSetAttributeForDevice(
                kernel_,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_),
                static_cast<int>(device)),
            "cudaKernelSetAttributeForDevice",
            name_);
    prepared_.push_back(device);
  }

 private:
  std::string name_;
  unsigned block_;
  unsigned num_ctas_;
  size_t shared_;
  cudaLibrary_t library_ = nullptr;
  cudaKernel_t kernel_ = nullptr;
  std::mutex mutex_;
  std::vector<int64_t> prepared_;
};

}  // namespace

std::unique_ptr<LoadedKernel> load_kernel(const KernelRecord& record) {
  return std::make_unique<CudaKernel>(record);
}

std::string device_architecture(int64_t device) {
  int major = 0;
  int minor = 0;
  check(
      cudaDeviceGetAttribute(
          &major, cudaDevAttrComputeCapabilityMajor, static_cast<int>(device)),
      "cudaDeviceGetAttribute",
      "architecture query");
  check(
      cudaDeviceGetAttribute(
          &minor, cudaDevAttrComputeCapabilityMinor, static_cast<int>(device)),
      "cudaDeviceGetAttribute",
      "architecture query");
  return "sm_" + std::to_string(major * 10 + minor);
}

std::string runtime_version() {
  return "cuda-" + std::to_string(CUDART_VERSION / 1000) + "." +
         std::to_string(CUDART_VERSION % 1000 / 10);
}

}  // namespace fhelium::exec::detail
