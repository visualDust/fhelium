#pragma once

#include <ATen/native/CanUse32BitIndexMath.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <optional>
#include <type_traits>
#include "../../common/cuda/kernel_support.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

// mu=floor(2^64/q). For x<2^64 the quotient estimate is low by at
// most one; the subtraction returns the canonical residue.
__device__ __forceinline__ uint32_t crt_reduce_u64(
    uint64_t x, uint32_t q, uint64_t mu) {
  const uint64_t quotient = __umul64hi(x, mu);
  const uint64_t remainder = x - quotient * q;
  return static_cast<uint32_t>(remainder >= q ? remainder - q : remainder);
}

template <int Columns, bool Normalize>
__global__ void crt_basis_extend_kernel(
    CudaTensorAccessor32<int32_t, 3> output,
    const CudaTensorAccessor32<int32_t, 3> source,
    const CudaTensorAccessor32<int32_t, 2> source_params,
    const CudaTensorAccessor32<int32_t, 2> target_params,
    const CudaTensorAccessor32<int32_t, 2> matrix,
    const CudaTensorAccessor32<int64_t, 1> target_reciprocals,
    const int32_t* source_normalizers, int normalizer_stride,
    const int64_t* source_reciprocals, int reciprocal_stride) {
  extern __shared__ int32_t shared[];
  const int width = source.size(1);
  const int destinations = matrix.size(1);
  int32_t* values = shared;
  int32_t* coefficients = values + width * Columns;
  for (int index = threadIdx.x; index < width * destinations;
       index += blockDim.x)
    coefficients[index] = matrix[index / destinations][index % destinations];
  for (int index = threadIdx.x; index < width * Columns;
       index += blockDim.x) {
    const int row = index / Columns;
    const int column = index % Columns;
    const int coefficient = blockIdx.x * Columns + column;
    uint32_t value = coefficient < source.size(2)
        ? static_cast<uint32_t>(source[blockIdx.z][row][coefficient]) : 0;
    const uint32_t q = static_cast<uint32_t>(
        source_params[RNS_PARAM_TWICE_MODULUS][row]) >> 1;
    if (value >= q) value -= q;
    if constexpr (Normalize) {
      value = crt_reduce_u64(
          static_cast<uint64_t>(value) *
              static_cast<uint32_t>(source_normalizers[row * normalizer_stride]),
          q, static_cast<uint64_t>(source_reciprocals[row * reciprocal_stride]));
    }
    values[index] = static_cast<int32_t>(value);
  }
  __syncthreads();
  const int column = threadIdx.x % Columns;
  const int coefficient = blockIdx.x * Columns + column;
  if (coefficient >= source.size(2)) return;
  for (int target = threadIdx.x / Columns; target < destinations;
       target += blockDim.x / Columns) {
    const uint32_t q = static_cast<uint32_t>(
        target_params[RNS_PARAM_TWICE_MODULUS][target]) >> 1;
    const uint64_t mu = static_cast<uint64_t>(target_reciprocals[target]);
    uint64_t accumulator = 0;
    for (int row = 0; row < width; ++row) {
      accumulator += static_cast<uint64_t>(
          static_cast<uint32_t>(values[row * Columns + column])) *
          static_cast<uint32_t>(coefficients[row * destinations + target]);
      // Each factor is below 2^30. Sixteen products plus a previously
      // reduced remainder fit uint64, including primes near 2^30.
      if ((row & 15) == 15)
        accumulator = crt_reduce_u64(accumulator, q, mu);
    }
    output[blockIdx.z][target][coefficient] = static_cast<int32_t>(
        crt_reduce_u64(accumulator, q, mu));
  }
}

static torch::Tensor crt_basis_extend_cuda_impl(
    const torch::Tensor source_residues,
    const torch::Tensor source_parameters,
    const torch::Tensor target_parameters,
    const torch::Tensor matrix,
    const torch::Tensor target_reciprocals,
    const std::optional<torch::Tensor> source_normalizers,
    const std::optional<torch::Tensor> source_reciprocals,
    const std::optional<torch::Tensor>& output_buffer) {
  const auto source = view_rns_batch_3d(source_residues, "source_residues");
  TORCH_CHECK(source.scalar_type() == torch::kInt32,
              "CUDA tiled CRT extension requires int32 standard residues");
  const int width = source.size(1);
  TORCH_CHECK(matrix.dim() == 2 && matrix.size(0) == width && matrix.size(1) > 0,
              "CRT matrix must have shape [source rows, destination rows]");
  const int destinations = matrix.size(1);
  auto check = [&](const torch::Tensor& table, c10::ScalarType dtype) {
    TORCH_CHECK(table.device() == source.device() && table.scalar_type() == dtype,
                "CRT numerical operands have incompatible dtype or device");
    TORCH_CHECK(at::native::canUse32BitIndexMath(table),
                "CRT numerical operand exceeds 32-bit indexing");
  };
  check(matrix, torch::kInt32);
  check(source_parameters, torch::kInt32);
  check(target_parameters, torch::kInt32);
  check(target_reciprocals, torch::kInt64);
  TORCH_CHECK(source_parameters.dim() == 2 &&
                  source_parameters.size(0) >= RNS_PARAMETER_COUNT &&
                  source_parameters.size(1) == width &&
                  target_parameters.dim() == 2 &&
                  target_parameters.size(0) >= RNS_PARAMETER_COUNT &&
                  target_parameters.size(1) == destinations,
              "CRT parameter rows differ from the matrix extents");
  check_rns_row_vector(target_reciprocals, destinations, "CRT extension", "target reciprocals");
  const bool normalize = source_normalizers.has_value();
  TORCH_CHECK(normalize == source_reciprocals.has_value(),
              "CRT source normalization needs multipliers and reciprocals");
  if (normalize) {
    check(*source_normalizers, torch::kInt32);
    check(*source_reciprocals, torch::kInt64);
    check_rns_row_vector(*source_normalizers, width, "CRT extension", "source normalizers");
    check_rns_row_vector(*source_reciprocals, width, "CRT extension", "source reciprocals");
  }
  TORCH_CHECK(at::native::canUse32BitIndexMath(source),
              "CRT source exceeds 32-bit indexing");
  constexpr int columns = 128;
  const size_t shared_bytes = static_cast<size_t>(width) *
      (columns + destinations) * sizeof(int32_t);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "CRT coefficient tile exceeds 48 KiB shared storage");
  auto shape = source_residues.sizes().vec();
  shape[shape.size() - 2] = destinations;
  auto result = output_buffer ? *output_buffer : torch::empty(shape, source_residues.options());
  std::vector<torch::Tensor> reads{source_residues, source_parameters,
      target_parameters, matrix, target_reciprocals};
  if (normalize) { reads.push_back(*source_normalizers); reads.push_back(*source_reciprocals); }
  auto output = output_buffer
      ? view_rns_output_3d(result, source_residues, destinations, reads, "crt_basis_extend")
      : view_rns_batch_3d(result, "result");
  TORCH_CHECK(at::native::canUse32BitIndexMath(output),
              "CRT output exceeds 32-bit indexing");
  const c10::cuda::CUDAGuard guard(source.device());
  auto stream = at::cuda::getCurrentCUDAStream(source.device().index());
  dim3 grid((source.size(2) + columns - 1) / columns, 1, source.size(0));
  auto launch = [&](auto tag) {
    constexpr bool Normalize = decltype(tag)::value;
    crt_basis_extend_kernel<columns, Normalize><<<grid, 256, shared_bytes, stream>>>(
        FHELIUM_CUDA_ACCESSOR32(output, int32_t, 3),
        FHELIUM_CUDA_ACCESSOR32(source, int32_t, 3),
        FHELIUM_CUDA_ACCESSOR32(source_parameters, int32_t, 2),
        FHELIUM_CUDA_ACCESSOR32(target_parameters, int32_t, 2),
        FHELIUM_CUDA_ACCESSOR32(matrix, int32_t, 2),
        FHELIUM_CUDA_ACCESSOR32(target_reciprocals, int64_t, 1),
        normalize ? source_normalizers->data_ptr<int32_t>() : nullptr,
        normalize ? source_normalizers->stride(0) : 0,
        normalize ? source_reciprocals->data_ptr<int64_t>() : nullptr,
        normalize ? source_reciprocals->stride(0) : 0);
  };
  if (normalize) launch(std::true_type{});
  else launch(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

torch::Tensor crt_basis_extend_cuda(
    torch::Tensor source, torch::Tensor source_parameters,
    torch::Tensor target_parameters, torch::Tensor matrix,
    torch::Tensor target_reciprocals,
    std::optional<torch::Tensor> source_normalizers,
    std::optional<torch::Tensor> source_reciprocals) {
  return crt_basis_extend_cuda_impl(source, source_parameters, target_parameters,
      matrix, target_reciprocals, source_normalizers, source_reciprocals, std::nullopt);
}

void crt_basis_extend_out_cuda(
    torch::Tensor source, torch::Tensor source_parameters,
    torch::Tensor target_parameters, torch::Tensor matrix,
    torch::Tensor target_reciprocals,
    std::optional<torch::Tensor> source_normalizers,
    std::optional<torch::Tensor> source_reciprocals, torch::Tensor output) {
  crt_basis_extend_cuda_impl(source, source_parameters, target_parameters,
      matrix, target_reciprocals, source_normalizers, source_reciprocals, output);
}
