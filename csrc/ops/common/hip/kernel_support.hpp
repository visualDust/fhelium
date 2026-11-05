#pragma once

#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

// HIP launch and tensor-access helpers for native operator kernels.
constexpr int kHipBlockSize = 256;

template <typename scalar_t, int dimension>
using HipTensorAccessor32 = torch::
    PackedTensorAccessor32<scalar_t, dimension, torch::RestrictPtrTraits>;

#define FHELIUM_HIP_ACCESSOR32(tensor, scalar_t, dimension) \
  tensor.packed_accessor32<scalar_t, dimension, torch::RestrictPtrTraits>()

inline void check_hip_launch() {
  const auto status = hipGetLastError();
  TORCH_CHECK(status == hipSuccess, "HIP kernel launch failed: ", hipGetErrorString(status));
}
