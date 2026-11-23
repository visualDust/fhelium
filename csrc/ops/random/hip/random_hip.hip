#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "../../common/hip/kernel_support.hpp"
#include "../backends.h"
#include "../random_core.h"

#define FHELIUM_RANDOM_GPU_STREAM(device) \
  c10::hip::getCurrentHIPStream(device).stream()
namespace fhelium::random::gpu::detail_hip {
inline int multiprocessor_count(int device) {
  int count = 0;
  TORCH_CHECK(
      hipDeviceGetAttribute(
          &count, hipDeviceAttributeMultiprocessorCount, device) == hipSuccess,
      "hipDeviceGetAttribute failed");
  return count;
}
}  // namespace fhelium::random::gpu::detail_hip

#define FHELIUM_RANDOM_GPU_SM_COUNT(device) \
  ::fhelium::random::gpu::detail_hip::multiprocessor_count(device)
#define FHELIUM_RANDOM_GPU_CHECK_LAUNCH() check_hip_launch()

namespace fhelium::random::gpu {
#include "../random_gpu.inl"
}  // namespace fhelium::random::gpu
