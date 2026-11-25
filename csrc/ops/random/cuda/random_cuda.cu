#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "../backends.h"
#include "../random_core.h"

#define FHELIUM_RANDOM_GPU_STREAM(device) \
  c10::cuda::getCurrentCUDAStream(device).stream()
namespace fhelium::random::gpu::detail_cuda {
inline int multiprocessor_count(int device) {
  int count = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, device));
  return count;
}
}  // namespace fhelium::random::gpu::detail_cuda

#define FHELIUM_RANDOM_GPU_SM_COUNT(device) \
  ::fhelium::random::gpu::detail_cuda::multiprocessor_count(device)
#define FHELIUM_RANDOM_GPU_CHECK_LAUNCH() C10_CUDA_KERNEL_LAUNCH_CHECK()

namespace fhelium::random::gpu {
#include "../random_gpu.inl"
}  // namespace fhelium::random::gpu
