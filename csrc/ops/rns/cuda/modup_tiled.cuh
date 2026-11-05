#pragma once

#include <ATen/native/CanUse32BitIndexMath.h>
#include <optional>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../../common/cuda/kernel_support.cuh"
#include "../../common/montgomery.hpp"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

// A coefficient tile owns its Garner digits until every destination row has
// consumed them. The ordered modular arithmetic is the same as the separate
// decomposition and extension operations; only their storage and scheduling
// are combined.
template <typename scalar_t, int Columns>
__global__ void modup_tiled_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> source,
    const scalar_t* normalizers, int normalizer_stride,
    const scalar_t* propagation, int propagation_row_stride, int propagation_col_stride,
    const scalar_t* modulus_lo, int modulus_lo_stride,
    const scalar_t* modulus_hi, int modulus_hi_stride,
    const scalar_t* neg_lo, int neg_lo_stride,
    const scalar_t* neg_hi, int neg_hi_stride,
    const CudaTensorAccessor32<scalar_t, 2> extension,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  extern __shared__ unsigned char storage[];
  const int width = source.size(1), destinations = out.size(1);
  scalar_t* digits = reinterpret_cast<scalar_t*>(storage);
  scalar_t* normal = digits + width * Columns;
  scalar_t* propagate = normal + width;
  scalar_t* reduction = propagate + width * width;
  scalar_t* factors = reduction + 4 * width;
  scalar_t* destination_params = factors + width * destinations;
  const int tid = threadIdx.x;
  if (width > 1) {
    for (int i = tid; i < width - 1; i += blockDim.x)
      normal[i] = normalizers[i * normalizer_stride];
    for (int i = tid; i < (width - 1) * width; i += blockDim.x)
      propagate[i] = propagation[(i / width) * propagation_row_stride +
                                  (i % width) * propagation_col_stride];
    for (int i = tid; i < width; i += blockDim.x) {
      reduction[i] = modulus_lo[i * modulus_lo_stride];
      reduction[width + i] = modulus_hi[i * modulus_hi_stride];
      reduction[2 * width + i] = neg_lo[i * neg_lo_stride];
      reduction[3 * width + i] = neg_hi[i * neg_hi_stride];
    }
    for (int i = tid; i < (width - 1) * destinations; i += blockDim.x)
      factors[i] = extension[i / destinations][i % destinations];
  }
  for (int i = tid; i < RNS_PARAMETER_COUNT * destinations; i += blockDim.x)
    destination_params[i] = params[i / destinations][i % destinations];
  __syncthreads();

  const int coefficient = blockIdx.x * Columns + tid;
  if (tid < Columns && coefficient < source.size(2)) {
    if (width == 1) {
      digits[tid] = source[blockIdx.z][0][coefficient];
    } else {
      const scalar_t first = reduce_lazy_montgomery_operand(
          source[blockIdx.z][0][coefficient], reduction[0], reduction[width]);
      for (int row = 0; row < width; ++row)
        digits[row * Columns + tid] = reduce_montgomery_operand(
            first, reduction[row], reduction[width + row]);
      for (int step = 0; step < width - 1; ++step) {
        const int row = step + 1;
        const scalar_t difference = source[blockIdx.z][row][coefficient] -
                                    digits[row * Columns + tid];
        const scalar_t twice_modulus =
            (reduction[row] + (reduction[width + row] <<
             (kMontgomeryRadixBits<scalar_t> / 2))) << 1;
        const scalar_t digit = reduce_lazy_residue(
            montgomery_mul_split(difference, normal[step], reduction[row],
              reduction[width + row], reduction[2 * width + row],
              reduction[3 * width + row]), twice_modulus);
        digits[row * Columns + tid] = digit;
        for (int target = row + 1; target < width; ++target) {
          const scalar_t product = reduce_lazy_montgomery_operand(
              montgomery_mul_split(digit, propagate[step * width + target],
                reduction[target], reduction[width + target],
                reduction[2 * width + target], reduction[3 * width + target]),
              reduction[target], reduction[width + target]);
          digits[target * Columns + tid] = reduce_lazy_montgomery_operand(
              static_cast<scalar_t>(digits[target * Columns + tid] + product),
              reduction[target], reduction[width + target]);
        }
      }
    }
  }
  __syncthreads();
  const int column = tid % Columns;
  const int output_coefficient = blockIdx.x * Columns + column;
  if (output_coefficient >= source.size(2)) return;
  for (int target = tid / Columns; target < destinations;
       target += blockDim.x / Columns) {
    const scalar_t twice_modulus = destination_params[RNS_PARAM_TWICE_MODULUS * destinations + target];
    const scalar_t lo = destination_params[RNS_PARAM_MODULUS_LO * destinations + target];
    const scalar_t hi = destination_params[RNS_PARAM_MODULUS_HI * destinations + target];
    const scalar_t nlo = destination_params[RNS_PARAM_NEG_INV_MODULUS_LO * destinations + target];
    const scalar_t nhi = destination_params[RNS_PARAM_NEG_INV_MODULUS_HI * destinations + target];
    scalar_t value = montgomery_mul(digits[column],
        destination_params[RNS_PARAM_R2 * destinations + target], lo, hi, nlo, nhi);
    for (int row = 1; row < width; ++row)
      value = add_lazy_residues(value,
          montgomery_mul(digits[row * Columns + column],
            factors[(row - 1) * destinations + target], lo, hi, nlo, nhi),
          twice_modulus);
    out[blockIdx.z][target][output_coefficient] = value;
  }
}

static torch::Tensor mixed_radix_modup_tiled_cuda_impl(
    const torch::Tensor source_residues,
    at::TensorList decomposition_tables,
    const torch::Tensor extension,
    const torch::Tensor parameters,
    const std::optional<torch::Tensor>& output_buffer) {
  const auto source = view_rns_batch_3d(source_residues, "source_residues");
  const int width = source.size(1);
  TORCH_CHECK(source.scalar_type() == torch::kInt32 || source.scalar_type() == torch::kInt64,
              "Tiled ModUp requires int32 or int64 residues");
  TORCH_CHECK(parameters.dim() == 2 && parameters.size(0) >= RNS_PARAMETER_COUNT && parameters.size(1) > 0,
              "Tiled ModUp parameter extent mismatch");
  const int destinations = parameters.size(1);
  TORCH_CHECK(decomposition_tables.size() == (width > 1 ? 6 : 0),
              "Tiled ModUp requires six decomposition tables for a multi-row digit");
  TORCH_CHECK(extension.dim() == 2 && extension.size(0) == width - 1 &&
                  (width == 1 || extension.size(1) == destinations),
              "Tiled ModUp extension table extent mismatch");
  auto check_table = [&](const torch::Tensor& table) {
    TORCH_CHECK(table.device() == source.device() && table.scalar_type() == source.scalar_type(),
                "Tiled ModUp tables must share source dtype and device");
    TORCH_CHECK(at::native::canUse32BitIndexMath(table),
                "Tiled ModUp table strides exceed 32-bit indexing");
  };
  check_table(parameters); check_table(extension);
  for (const auto& table : decomposition_tables) check_table(table);
  if (width > 1) {
    check_rns_row_vector(decomposition_tables[0], width - 1, "Tiled ModUp", "normalizers");
    TORCH_CHECK(decomposition_tables[1].dim() == 2 && decomposition_tables[1].size(0) == width - 1 &&
                    decomposition_tables[1].size(1) == width, "Tiled ModUp propagation extent mismatch");
    for (int i = 2; i < 6; ++i)
      check_rns_row_vector(decomposition_tables[i], width, "Tiled ModUp", "reduction vector");
  }
  TORCH_CHECK(at::native::canUse32BitIndexMath(source), "Tiled ModUp source strides exceed 32-bit indexing");
  const int columns = source.scalar_type() == torch::kInt32 ? 128 : 64;
  const size_t shared_bytes = (static_cast<size_t>(width) * columns + width +
      static_cast<size_t>(width) * width + 4 * width +
      static_cast<size_t>(width + RNS_PARAMETER_COUNT) * destinations) * source.element_size();
  TORCH_CHECK(shared_bytes <= 48 * 1024, "Tiled ModUp tile exceeds its 48 KiB shared-memory extent");
  auto shape = source_residues.sizes().vec(); shape[shape.size() - 2] = destinations;
  auto out = output_buffer ? *output_buffer : torch::empty(shape, source_residues.options());
  std::vector<torch::Tensor> reads{source_residues, extension, parameters};
  reads.insert(reads.end(), decomposition_tables.begin(), decomposition_tables.end());
  auto output = output_buffer
      ? view_rns_output_3d(out, source_residues, destinations, reads,
                           "mixed_radix_modup_tiled")
      : view_rns_batch_3d(out, "out");
  TORCH_CHECK(at::native::canUse32BitIndexMath(output), "Tiled ModUp output extent exceeds 32-bit indexing");
  const c10::cuda::CUDAGuard guard(source.device());
  auto stream = at::cuda::getCurrentCUDAStream(source.device().index());
  dim3 grid((source.size(2) + columns - 1) / columns, 1, source.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(source.scalar_type(), "mixed_radix_modup_tiled", [&] {
    auto pointer = [&](int i) -> const scalar_t* { return width > 1 ? decomposition_tables[i].data_ptr<scalar_t>() : nullptr; };
    auto stride = [&](int i, int axis) -> int { return width > 1 ? decomposition_tables[i].stride(axis) : 0; };
    if constexpr (std::is_same_v<scalar_t, int32_t> || std::is_same_v<scalar_t, int64_t>) {
      constexpr int Cols = std::is_same_v<scalar_t, int32_t> ? 128 : 64;
      modup_tiled_kernel<scalar_t, Cols><<<grid, 256, shared_bytes, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(output, scalar_t, 3), FHELIUM_CUDA_ACCESSOR32(source, scalar_t, 3),
          pointer(0), stride(0, 0), pointer(1), stride(1, 0), stride(1, 1),
          pointer(2), stride(2, 0), pointer(3), stride(3, 0), pointer(4), stride(4, 0), pointer(5), stride(5, 0),
          FHELIUM_CUDA_ACCESSOR32(extension, scalar_t, 2), FHELIUM_CUDA_ACCESSOR32(parameters, scalar_t, 2));
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Both entry points share arithmetic, table reads, and output validation.
torch::Tensor mixed_radix_modup_tiled_cuda(
    const torch::Tensor source_residues, at::TensorList decomposition_tables,
    const torch::Tensor extension, const torch::Tensor parameters) {
  return mixed_radix_modup_tiled_cuda_impl(source_residues, decomposition_tables,
                                          extension, parameters, std::nullopt);
}

void mixed_radix_modup_tiled_out_cuda(
    const torch::Tensor source_residues, at::TensorList decomposition_tables,
    const torch::Tensor extension, const torch::Tensor parameters,
    torch::Tensor out) {
  mixed_radix_modup_tiled_cuda_impl(source_residues, decomposition_tables,
                                   extension, parameters, out);
}
