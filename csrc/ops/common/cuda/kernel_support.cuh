#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

// Shared CUDA launch and tensor-access helpers for native operator kernels.
constexpr int kCudaBlockSize = 256;

template <typename scalar_t, int dimension>
using CudaTensorAccessor32 = torch::
    PackedTensorAccessor32<scalar_t, dimension, torch::RestrictPtrTraits>;

#define FHELIUM_CUDA_ACCESSOR32(tensor, scalar_t, dimension) \
  tensor.packed_accessor32<scalar_t, dimension, torch::RestrictPtrTraits>()
