#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "../backends.h"
#include "../random_core.h"

#define FHELIUM_RANDOM_GPU_STREAM(device) \
  at::cuda::getCurrentCUDAStream(device).stream()
#define FHELIUM_RANDOM_GPU_SM_COUNT(device) \
  at::cuda::getDeviceProperties(device)->multiProcessorCount
#define FHELIUM_RANDOM_GPU_CHECK_LAUNCH() C10_CUDA_KERNEL_LAUNCH_CHECK()

namespace fhelium::random::gpu {
#include "../random_gpu.inl"
}  // namespace fhelium::random::gpu
