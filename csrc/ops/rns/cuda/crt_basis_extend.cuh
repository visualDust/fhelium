#pragma once

#include <ATen/native/CanUse32BitIndexMath.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
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

// Coefficients per thread and per block of the CRT extension.
constexpr int kCrtColumnsPerThread = 2;
constexpr int kCrtBlockColumns = 256;

// blockIdx.x selects a chunk of Targets destination rows, so that the chunks of
// one coefficient tile run adjacently and share its source digits in L2;
// blockIdx.y selects the tile. The chunk's matrix columns
// are staged in shared memory as [source row][Targets]. Each thread reads two
// adjacent source coefficients per row from global memory and accumulates
// them against all Targets coefficients, which one broadcast shared load
// per four targets supplies.
template <int Targets, bool Normalize>
__global__ void __launch_bounds__(kCrtBlockColumns / kCrtColumnsPerThread)
crt_basis_extend_kernel(
    CudaTensorAccessor32<int32_t, 3> output,
    const CudaTensorAccessor32<int32_t, 3> source,
    const CudaTensorAccessor32<int32_t, 2> source_params,
    const CudaTensorAccessor32<int32_t, 2> target_params,
    const CudaTensorAccessor32<int32_t, 2> matrix,
    const CudaTensorAccessor32<int64_t, 1> target_reciprocals,
    const int32_t* source_normalizers, int normalizer_stride,
    const int64_t* source_reciprocals, int reciprocal_stride,
    const int32_t* target_offsets, int target_offset_stride) {
  extern __shared__ __align__(16) int32_t coefficients[];  // [width][Targets]
  // Target moduli and reciprocals stay in shared memory so that the register
  // file holds only the accumulators.
  __shared__ uint64_t reciprocals[Targets];
  __shared__ uint32_t moduli[Targets];
  const int width = source.size(1);
  const int destinations = matrix.size(1);
  const int chunk = blockIdx.x * Targets;
  for (int index = threadIdx.x; index < width * Targets; index += blockDim.x) {
    const int target = chunk + index % Targets;
    coefficients[index] = target < destinations ? matrix[index / Targets][target] : 0;
  }
  if (threadIdx.x < Targets) {
    const int target = min(chunk + static_cast<int>(threadIdx.x), destinations - 1);
    moduli[threadIdx.x] =
        static_cast<uint32_t>(target_params[RNS_PARAM_TWICE_MODULUS][target]) >> 1;
    reciprocals[threadIdx.x] = static_cast<uint64_t>(target_reciprocals[target]);
  }
  uint64_t sums[Targets][kCrtColumnsPerThread];
#pragma unroll
  for (int t = 0; t < Targets; ++t) {
    const int target = min(chunk + t, destinations - 1);
    const uint64_t offset = Normalize
        ? static_cast<uint32_t>(target_offsets[target * target_offset_stride]) : 0;
#pragma unroll
    for (int c = 0; c < kCrtColumnsPerThread; ++c) sums[t][c] = offset;
  }
  __syncthreads();
  const int coefficient =
      (blockIdx.y * blockDim.x + threadIdx.x) * kCrtColumnsPerThread;
  if (coefficient >= source.size(2)) return;
  const int32_t* digits = &source[blockIdx.z][0][coefficient];
  const int row_stride = source.stride(1);
#pragma unroll 4
  for (int row = 0; row < width; ++row) {
    const int2 v = *reinterpret_cast<const int2*>(digits + row * row_stride);
    const uint32_t q = static_cast<uint32_t>(
        __ldg(&source_params[RNS_PARAM_TWICE_MODULUS][row])) >> 1;
    uint32_t value[2] = {static_cast<uint32_t>(v.x), static_cast<uint32_t>(v.y)};
#pragma unroll
    for (int c = 0; c < kCrtColumnsPerThread; ++c) {
      if (value[c] >= q) value[c] -= q;
      if constexpr (Normalize) {
        // t = [a u + (q - 1)/2]_q; the centered digit is t - (q - 1)/2, whose
        // matrix image the target offset subtracts.
        value[c] = crt_reduce_u64(
            static_cast<uint64_t>(value[c]) *
                    static_cast<uint32_t>(__ldg(source_normalizers + row * normalizer_stride)) +
                (q >> 1),
            q, static_cast<uint64_t>(__ldg(source_reciprocals + row * reciprocal_stride)));
      }
    }
#pragma unroll
    for (int t0 = 0; t0 < Targets; t0 += 4) {
      const int4 w = *reinterpret_cast<const int4*>(coefficients + row * Targets + t0);
      const uint32_t weight[4] = {static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y),
                                  static_cast<uint32_t>(w.z), static_cast<uint32_t>(w.w)};
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int c = 0; c < kCrtColumnsPerThread; ++c)
          sums[t0 + t][c] += static_cast<uint64_t>(value[c]) * weight[t];
    }
    // Each factor is below 2^30. Sixteen products plus an offset or a
    // previously reduced remainder fit uint64, including primes near 2^30.
    if ((row & 15) == 15) {
#pragma unroll
      for (int t = 0; t < Targets; ++t)
#pragma unroll
        for (int c = 0; c < kCrtColumnsPerThread; ++c)
          sums[t][c] = crt_reduce_u64(sums[t][c], moduli[t], reciprocals[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < Targets; ++t) {
    if (chunk + t >= destinations) break;
    int2 result;
    result.x = static_cast<int32_t>(crt_reduce_u64(sums[t][0], moduli[t], reciprocals[t]));
    result.y = static_cast<int32_t>(crt_reduce_u64(sums[t][1], moduli[t], reciprocals[t]));
    *reinterpret_cast<int2*>(&output[blockIdx.z][chunk + t][coefficient]) = result;
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
    const std::optional<torch::Tensor> target_offsets,
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
  TORCH_CHECK(normalize == source_reciprocals.has_value() &&
                  normalize == target_offsets.has_value(),
              "CRT source normalization needs multipliers, reciprocals and "
              "target offsets");
  if (normalize) {
    check(*source_normalizers, torch::kInt32);
    check(*source_reciprocals, torch::kInt64);
    check(*target_offsets, torch::kInt32);
    check_rns_row_vector(*target_offsets, destinations, "CRT extension", "target offsets");
    check_rns_row_vector(*source_normalizers, width, "CRT extension", "source normalizers");
    check_rns_row_vector(*source_reciprocals, width, "CRT extension", "source reciprocals");
  }
  TORCH_CHECK(at::native::canUse32BitIndexMath(source),
              "CRT source exceeds 32-bit indexing");
  const int targets = destinations <= 4 ? 4 : 16;
  const size_t shared_bytes = static_cast<size_t>(width) * targets * sizeof(int32_t);
  TORCH_CHECK(shared_bytes <= 48 * 1024,
              "CRT source rows exceed the shared coefficient chunk");
  TORCH_CHECK(source.size(2) % kCrtColumnsPerThread == 0 && source.stride(2) == 1 &&
                  source.stride(1) % 2 == 0 && source.stride(0) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(source.data_ptr()) % 8 == 0,
              "CRT extension requires contiguous, 8-byte aligned coefficient pairs");
  auto shape = source_residues.sizes().vec();
  shape[shape.size() - 2] = destinations;
  auto result = output_buffer ? *output_buffer : torch::empty(shape, source_residues.options());
  std::vector<torch::Tensor> reads{source_residues, source_parameters,
      target_parameters, matrix, target_reciprocals};
  if (normalize) {
    reads.push_back(*source_normalizers);
    reads.push_back(*source_reciprocals);
    reads.push_back(*target_offsets);
  }
  auto output = output_buffer
      ? view_rns_output_3d(result, source_residues, destinations, reads, "crt_basis_extend")
      : view_rns_batch_3d(result, "result");
  TORCH_CHECK(at::native::canUse32BitIndexMath(output),
              "CRT output exceeds 32-bit indexing");
  TORCH_CHECK(output.stride(2) == 1 && output.stride(1) % 2 == 0 &&
                  output.stride(0) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0,
              "CRT output requires contiguous, 8-byte aligned coefficient pairs");
  const c10::cuda::CUDAGuard guard(source.device());
  auto stream = at::cuda::getCurrentCUDAStream(source.device().index());
  dim3 grid((destinations + targets - 1) / targets,
            (source.size(2) + kCrtBlockColumns - 1) / kCrtBlockColumns,
            source.size(0));
  auto launch = [&](auto normalize_tag, auto targets_tag) {
    constexpr bool Normalize = decltype(normalize_tag)::value;
    constexpr int Targets = decltype(targets_tag)::value;
    crt_basis_extend_kernel<Targets, Normalize>
        <<<grid, kCrtBlockColumns / kCrtColumnsPerThread, shared_bytes, stream>>>(
        FHELIUM_CUDA_ACCESSOR32(output, int32_t, 3),
        FHELIUM_CUDA_ACCESSOR32(source, int32_t, 3),
        FHELIUM_CUDA_ACCESSOR32(source_parameters, int32_t, 2),
        FHELIUM_CUDA_ACCESSOR32(target_parameters, int32_t, 2),
        FHELIUM_CUDA_ACCESSOR32(matrix, int32_t, 2),
        FHELIUM_CUDA_ACCESSOR32(target_reciprocals, int64_t, 1),
        normalize ? source_normalizers->data_ptr<int32_t>() : nullptr,
        normalize ? source_normalizers->stride(0) : 0,
        normalize ? source_reciprocals->data_ptr<int64_t>() : nullptr,
        normalize ? source_reciprocals->stride(0) : 0,
        normalize ? target_offsets->data_ptr<int32_t>() : nullptr,
        normalize ? target_offsets->stride(0) : 0);
  };
  auto with_targets = [&](auto normalize_tag) {
    if (targets == 4) launch(normalize_tag, std::integral_constant<int, 4>{});
    else launch(normalize_tag, std::integral_constant<int, 16>{});
  };
  if (normalize) with_targets(std::true_type{});
  else with_targets(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

torch::Tensor crt_basis_extend_cuda(
    torch::Tensor source, torch::Tensor source_parameters,
    torch::Tensor target_parameters, torch::Tensor matrix,
    torch::Tensor target_reciprocals,
    std::optional<torch::Tensor> source_normalizers,
    std::optional<torch::Tensor> source_reciprocals,
    std::optional<torch::Tensor> target_offsets) {
  return crt_basis_extend_cuda_impl(source, source_parameters, target_parameters,
      matrix, target_reciprocals, source_normalizers, source_reciprocals,
      target_offsets, std::nullopt);
}

void crt_basis_extend_out_cuda(
    torch::Tensor source, torch::Tensor source_parameters,
    torch::Tensor target_parameters, torch::Tensor matrix,
    torch::Tensor target_reciprocals,
    std::optional<torch::Tensor> source_normalizers,
    std::optional<torch::Tensor> source_reciprocals,
    std::optional<torch::Tensor> target_offsets, torch::Tensor output) {
  crt_basis_extend_cuda_impl(source, source_parameters, target_parameters,
      matrix, target_reciprocals, source_normalizers, source_reciprocals,
      target_offsets, output);
}
