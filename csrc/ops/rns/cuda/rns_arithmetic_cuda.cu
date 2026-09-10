#include "rns_arithmetic_cuda.h"

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <vector>

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/cuda/repetition.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

namespace {

enum class BinaryRnsOperation : int { kMontgomeryMul, kAddLazy, kSubLazy };
enum class UnaryRnsOperation : int {
  kToMontgomery,
  kFromMontgomery,
  kCanonicalize,
  kCenter,
  kShiftPositive,
};

template <typename scalar_t, BinaryRnsOperation operation>
__global__ void rns_binary_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> rhs,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;

  const int rhs_batch = rhs.size(0) == lhs.size(0) ? batch : 0;
  const scalar_t a = lhs[batch][row][coefficient];
  const scalar_t b = rhs[rhs_batch][row][coefficient];

  if constexpr (operation == BinaryRnsOperation::kMontgomeryMul) {
    out[batch][row][coefficient] =
        montgomery_mul(a,
                       b,
                       params[RNS_PARAM_MODULUS_LO][row],
                       params[RNS_PARAM_MODULUS_HI][row],
                       params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                       params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  } else if constexpr (operation == BinaryRnsOperation::kAddLazy) {
    out[batch][row][coefficient] =
        add_lazy_residues(a, b, params[RNS_PARAM_TWICE_MODULUS][row]);
  } else {
    out[batch][row][coefficient] =
        sub_lazy_residues(a, b, params[RNS_PARAM_TWICE_MODULUS][row]);
  }
}

constexpr int kMaximumWeightedTermsPerLaunch = 32;

template <typename scalar_t>
struct WeightedSumPointers {
  const scalar_t* ciphertexts[kMaximumWeightedTermsPerLaunch];
  const scalar_t* plaintexts[kMaximumWeightedTermsPerLaunch];
};

template <typename scalar_t>
__global__ void rns_montgomery_weighted_sum_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    WeightedSumPointers<scalar_t> pointers,
    const CudaTensorAccessor32<scalar_t, 2> params,
    int term_count,
    int ciphertext_stride0,
    int ciphertext_stride1,
    int ciphertext_stride2,
    int plaintext_batches,
    int plaintext_stride0,
    int plaintext_stride1,
    int plaintext_stride2,
    bool accumulate) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= out.size(2)) return;

  scalar_t value = accumulate ? out[batch][row][coefficient] : scalar_t{0};
  const int plaintext_batch = batch % plaintext_batches;
  const int64_t ciphertext_index =
      static_cast<int64_t>(batch) * ciphertext_stride0 +
      static_cast<int64_t>(row) * ciphertext_stride1 +
      static_cast<int64_t>(coefficient) * ciphertext_stride2;
  const int64_t plaintext_index =
      static_cast<int64_t>(plaintext_batch) * plaintext_stride0 +
      static_cast<int64_t>(row) * plaintext_stride1 +
      static_cast<int64_t>(coefficient) * plaintext_stride2;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  for (int term = 0; term < term_count; ++term) {
    value = add_lazy_residues(
        value,
        montgomery_mul(pointers.ciphertexts[term][ciphertext_index],
                       pointers.plaintexts[term][plaintext_index],
                       params[RNS_PARAM_MODULUS_LO][row],
                       params[RNS_PARAM_MODULUS_HI][row],
                       params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                       params[RNS_PARAM_NEG_INV_MODULUS_HI][row]),
        twice_modulus);
  }
  out[batch][row][coefficient] =
      reduce_lazy_residue(value, twice_modulus);
}

template <typename scalar_t>
void launch_montgomery_weighted_sum(
    torch::Tensor out,
    const std::vector<torch::Tensor>& ciphertexts,
    const std::vector<torch::Tensor>& plaintexts,
    const torch::Tensor params) {
  const int device = out.device().index();
  c10::cuda::CUDAGuard guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(out.size(1),
            (out.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            out.size(0));
  for (int64_t begin = 0; begin < ciphertexts.size();
       begin += kMaximumWeightedTermsPerLaunch) {
    const int64_t count = std::min<int64_t>(
        kMaximumWeightedTermsPerLaunch, ciphertexts.size() - begin);
    WeightedSumPointers<scalar_t> pointers{};
    for (int64_t term = 0; term < count; ++term) {
      pointers.ciphertexts[term] =
          ciphertexts[begin + term].data_ptr<scalar_t>();
      pointers.plaintexts[term] =
          plaintexts[begin + term].data_ptr<scalar_t>();
    }
    const auto& ciphertext = ciphertexts[0];
    const auto& plaintext = plaintexts[0];
    rns_montgomery_weighted_sum_kernel<scalar_t>
        <<<grid, kCudaBlockSize, 0, stream>>>(
            FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
            pointers,
            FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2),
            count,
            ciphertext.stride(0),
            ciphertext.stride(1),
            ciphertext.stride(2),
            plaintext.size(0),
            plaintext.stride(0),
            plaintext.stride(1),
            plaintext.stride(2),
            begin != 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

constexpr int kMaximumWeightedGroups = 16;
constexpr int kMaximumSharedWeightedTerms = 16;

template <typename scalar_t>
struct WeightedSumsPointers {
  const scalar_t* ciphertexts[kMaximumSharedWeightedTerms];
  const scalar_t*
      plaintexts[kMaximumWeightedGroups * kMaximumSharedWeightedTerms];
};

template <typename scalar_t, int TermCount, int GroupCount>
__global__ void rns_montgomery_weighted_sums_kernel(
    scalar_t* out,
    WeightedSumsPointers<scalar_t> pointers,
    const CudaTensorAccessor32<scalar_t, 2> params,
    int batch_count,
    int component_count,
    int limb_count,
    int coefficient_count,
    int ciphertext_stride0,
    int ciphertext_stride1,
    int ciphertext_stride2,
    int plaintext_batches,
    int plaintext_stride0,
    int plaintext_stride1,
    int plaintext_stride2) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= coefficient_count) return;

  const int plaintext_batch = batch % plaintext_batches;
  const int64_t ciphertext_index =
      static_cast<int64_t>(batch) * ciphertext_stride0 +
      static_cast<int64_t>(row) * ciphertext_stride1 +
      static_cast<int64_t>(coefficient) * ciphertext_stride2;
  const int64_t plaintext_index =
      static_cast<int64_t>(plaintext_batch) * plaintext_stride0 +
      static_cast<int64_t>(row) * plaintext_stride1 +
      static_cast<int64_t>(coefficient) * plaintext_stride2;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  scalar_t values[GroupCount];
#pragma unroll
  for (int group = 0; group < GroupCount; ++group) values[group] = 0;
#pragma unroll
  for (int term = 0; term < TermCount; ++term) {
    const scalar_t ciphertext = pointers.ciphertexts[term][ciphertext_index];
#pragma unroll
    for (int group = 0; group < GroupCount; ++group) {
      values[group] = add_lazy_residues(
          values[group],
          montgomery_mul(
              ciphertext,
              pointers.plaintexts[group * TermCount + term][plaintext_index],
              params[RNS_PARAM_MODULUS_LO][row],
              params[RNS_PARAM_MODULUS_HI][row],
              params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
              params[RNS_PARAM_NEG_INV_MODULUS_HI][row]),
          twice_modulus);
    }
  }
#pragma unroll
  for (int group = 0; group < GroupCount; ++group) {
    const int operand_batch_count = batch_count / component_count;
    const int component = batch / operand_batch_count;
    const int operand_batch = batch % operand_batch_count;
    const int grouped_batch =
        (component * GroupCount + group) * operand_batch_count + operand_batch;
    const int64_t output_index =
        (static_cast<int64_t>(grouped_batch) * limb_count + row) *
            coefficient_count +
        coefficient;
    out[output_index] = reduce_lazy_residue(values[group], twice_modulus);
  }
}

template <typename scalar_t>
void launch_montgomery_weighted_sums(
    torch::Tensor out,
    const std::vector<torch::Tensor>& ciphertexts,
    const std::vector<torch::Tensor>& plaintexts,
    const torch::Tensor params,
    int64_t group_count) {
  const auto ciphertext = view_rns_batch_3d(ciphertexts[0], "ciphertext");
  const auto plaintext = view_rns_batch_3d(plaintexts[0], "plaintext");
  const int term_count = ciphertexts.size();
  const int device = out.device().index();
  c10::cuda::CUDAGuard guard(out.device());
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(ciphertext.size(1),
            (ciphertext.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            ciphertext.size(0));
  WeightedSumsPointers<scalar_t> pointers{};
  for (int term = 0; term < term_count; ++term) {
    pointers.ciphertexts[term] = ciphertexts[term].data_ptr<scalar_t>();
  }
  for (int term = 0; term < plaintexts.size(); ++term) {
    pointers.plaintexts[term] = plaintexts[term].data_ptr<scalar_t>();
  }
#define FHELIUM_LAUNCH_WEIGHTED_SUMS(TERMS, GROUPS)                         \
  rns_montgomery_weighted_sums_kernel<scalar_t, TERMS, GROUPS>             \
      <<<grid, kCudaBlockSize, 0, stream>>>(                                \
          out.data_ptr<scalar_t>(),                                         \
          pointers,                                                         \
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2),                      \
          ciphertext.size(0),                                               \
          ciphertexts[0].size(0),                                           \
          ciphertext.size(1),                                               \
          ciphertext.size(2),                                               \
          ciphertext.stride(0),                                             \
          ciphertext.stride(1),                                             \
          ciphertext.stride(2),                                             \
          plaintext.size(0),                                                \
          plaintext.stride(0),                                              \
          plaintext.stride(1),                                              \
          plaintext.stride(2))
  if (term_count == 16 && group_count == 16) {
    FHELIUM_LAUNCH_WEIGHTED_SUMS(16, 16);
  } else if (term_count == 8 && group_count == 8) {
    FHELIUM_LAUNCH_WEIGHTED_SUMS(8, 8);
  } else if (term_count == 4 && group_count == 8) {
    FHELIUM_LAUNCH_WEIGHTED_SUMS(4, 8);
  }
#undef FHELIUM_LAUNCH_WEIGHTED_SUMS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, RepetitionLayout layout>
__global__ void rns_montgomery_mul_compressed_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> compressed_rhs,
    const CudaTensorAccessor32<scalar_t, 2> params,
    int unique_mask,
    int repeat_shift) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;

  const int rhs_batch = compressed_rhs.size(0) == lhs.size(0) ? batch : 0;
  const int rhs_index =
      repeated_rhs_index<layout>(coefficient, unique_mask, repeat_shift);
  out[batch][row][coefficient] =
      montgomery_mul(lhs[batch][row][coefficient],
                     compressed_rhs[rhs_batch][row][rhs_index],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t, BinaryRnsOperation operation>
void launch_binary(torch::Tensor out,
                   const torch::Tensor lhs,
                   const torch::Tensor rhs,
                   const torch::Tensor params) {
  const int device = lhs.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(lhs.size(1),
            (lhs.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            lhs.size(0));
  rns_binary_kernel<scalar_t, operation><<<grid, kCudaBlockSize, 0, stream>>>(
      FHELIUM_CUDA_ACCESSOR32(out, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(lhs, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(rhs, scalar_t, 3),
      FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

template <BinaryRnsOperation operation>
torch::Tensor binary_rns(const torch::Tensor& lhs,
                         const torch::Tensor& rhs,
                         const torch::Tensor& params,
                         const char* operation_name) {
  auto out = torch::empty_like(lhs);
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(rhs, "rhs");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_rns_binary_3d(lhs_rows, rhs_rows, operation_name, true);
  check_rns_parameter_rows(lhs_rows, params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(lhs.scalar_type(), "rns_binary", [&] {
    launch_binary<scalar_t, operation>(out_rows, lhs_rows, rhs_rows, params);
  });
  return out;
}

template <typename scalar_t, UnaryRnsOperation operation>
__global__ void rns_unary_inplace_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;

  scalar_t value = residues[batch][row][coefficient];
  if constexpr (operation == UnaryRnsOperation::kToMontgomery) {
    value = montgomery_mul(value,
                           params[RNS_PARAM_R2][row],
                           params[RNS_PARAM_MODULUS_LO][row],
                           params[RNS_PARAM_MODULUS_HI][row],
                           params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                           params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  } else if constexpr (operation == UnaryRnsOperation::kFromMontgomery) {
    value = montgomery_reduce(value,
                              params[RNS_PARAM_MODULUS_LO][row],
                              params[RNS_PARAM_MODULUS_HI][row],
                              params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                              params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
  } else if constexpr (operation == UnaryRnsOperation::kCanonicalize) {
    value = reduce_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  } else if constexpr (operation == UnaryRnsOperation::kCenter) {
    value = center_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  } else {
    value = shift_residue_positive(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  }
  residues[batch][row][coefficient] = value;
}

template <typename scalar_t, UnaryRnsOperation operation>
void launch_unary(torch::Tensor residues, const torch::Tensor params) {
  const int device = residues.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(residues.size(1),
            (residues.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            residues.size(0));
  rns_unary_inplace_kernel<scalar_t, operation>
      <<<grid, kCudaBlockSize, 0, stream>>>(
          FHELIUM_CUDA_ACCESSOR32(residues, scalar_t, 3),
          FHELIUM_CUDA_ACCESSOR32(params, scalar_t, 2));
}

template <UnaryRnsOperation operation>
void unary_rns_inplace(torch::Tensor residues,
                       const torch::Tensor& params,
                       const char* operand_name,
                       const char* operation_name) {
  auto rows = view_rns_batch_3d(residues, operand_name);
  check_rns_parameter_rows(rows, params, operation_name);
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(), "rns_unary", [&] {
    launch_unary<scalar_t, operation>(rows, params);
  });
}

template <typename scalar_t>
__global__ void rns_montgomery_mul_row_scalars_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 1> scalars,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  residues[batch][row][coefficient] =
      montgomery_mul(residues[batch][row][coefficient],
                     scalars[row],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t>
__global__ void rns_add_lazy_twice_modulus_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> lhs,
    const CudaTensorAccessor32<scalar_t, 3> rhs,
    const CudaTensorAccessor32<scalar_t, 1> twice_modulus) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= lhs.size(2)) return;
  const int rhs_batch = rhs.size(0) == lhs.size(0) ? batch : 0;
  out[batch][row][coefficient] =
      add_lazy_residues(lhs[batch][row][coefficient],
                        rhs[rhs_batch][row][coefficient],
                        twice_modulus[row]);
}

template <typename scalar_t>
__global__ void rns_lift_centered_coefficients_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 2> coefficients,
    const CudaTensorAccessor32<scalar_t, 1> twice_modulus) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= coefficients.size(1)) return;
  out[batch][row][coefficient] = shift_residue_positive(
      coefficients[batch][coefficient], twice_modulus[row]);
}

}  // namespace

torch::Tensor rns_montgomery_mul_cuda(const torch::Tensor lhs,
                                      const torch::Tensor rhs,
                                      const torch::Tensor rns_params) {
  return binary_rns<BinaryRnsOperation::kMontgomeryMul>(
      lhs, rhs, rns_params, "rns_montgomery_mul");
}

torch::Tensor rns_montgomery_weighted_sum_cuda(
    const at::TensorList ciphertexts,
    const at::TensorList plaintexts,
    const torch::Tensor rns_params) {
  constexpr const char* operation = "rns_montgomery_weighted_sum";
  TORCH_CHECK(!ciphertexts.empty(), operation, " requires at least one term");
  TORCH_CHECK(ciphertexts.size() == plaintexts.size(),
              operation,
              " ciphertext and plaintext term counts differ");

  std::vector<torch::Tensor> ciphertext_rows;
  std::vector<torch::Tensor> plaintext_rows;
  ciphertext_rows.reserve(ciphertexts.size());
  plaintext_rows.reserve(plaintexts.size());
  const auto reference = view_rns_batch_3d(ciphertexts[0], "ciphertexts[0]");
  TORCH_CHECK(reference.is_cuda(), operation, " requires CUDA operands");
  for (int64_t term = 0; term < ciphertexts.size(); ++term) {
    TORCH_CHECK(ciphertexts[term].device() == reference.device() &&
                    plaintexts[term].device() == reference.device(),
                operation,
                " requires every term on the input device");
    TORCH_CHECK(ciphertexts[term].scalar_type() == reference.scalar_type() &&
                    plaintexts[term].scalar_type() == reference.scalar_type(),
                operation,
                " requires one integral dtype");
    auto ciphertext =
        view_rns_batch_3d(ciphertexts[term], "ciphertext term");
    auto plaintext = view_rns_batch_3d(plaintexts[term], "plaintext term");
    TORCH_CHECK(ciphertext.sizes() == reference.sizes() &&
                    ciphertext.strides() == reference.strides(),
                operation,
                " ciphertext term layouts differ");
    TORCH_CHECK(plaintext.size(1) == reference.size(1) &&
                    plaintext.size(2) == reference.size(2) &&
                    reference.size(0) % plaintext.size(0) == 0,
                operation,
                " plaintext term shape does not broadcast over components");
    if (!plaintext_rows.empty()) {
      TORCH_CHECK(plaintext.sizes() == plaintext_rows[0].sizes() &&
                      plaintext.strides() == plaintext_rows[0].strides(),
                  operation,
                  " plaintext term layouts differ");
    }
    ciphertext_rows.push_back(std::move(ciphertext));
    plaintext_rows.push_back(std::move(plaintext));
  }
  TORCH_CHECK(rns_params.device() == reference.device() &&
                  rns_params.scalar_type() == reference.scalar_type(),
              operation,
              " RNS parameters must match the input device and dtype");
  check_rns_parameter_rows(reference, rns_params, operation);

  auto out = torch::empty_like(ciphertexts[0]);
  auto output = view_rns_batch_3d(out, "out");
  at::assert_no_internal_overlap(output);
  AT_DISPATCH_INTEGRAL_TYPES(
      reference.scalar_type(), "rns_montgomery_weighted_sum_cuda", [&] {
        launch_montgomery_weighted_sum<scalar_t>(
            output, ciphertext_rows, plaintext_rows, rns_params);
      });
  return out;
}

torch::Tensor rns_montgomery_weighted_sums_cuda(
    const at::TensorList ciphertexts,
    const at::TensorList plaintexts,
    const int64_t group_count,
    const torch::Tensor rns_params) {
  constexpr const char* operation = "rns_montgomery_weighted_sums";
  TORCH_CHECK(group_count > 0, operation, " requires at least one group");
  TORCH_CHECK(!ciphertexts.empty(), operation, " requires at least one term");
  TORCH_CHECK(plaintexts.size() == group_count * ciphertexts.size(),
              operation,
              " plaintext matrix must have group_count * term_count entries");

  std::vector<torch::Tensor> ciphertext_rows;
  std::vector<torch::Tensor> plaintext_rows;
  ciphertext_rows.reserve(ciphertexts.size());
  plaintext_rows.reserve(plaintexts.size());
  const auto reference = view_rns_batch_3d(ciphertexts[0], "ciphertexts[0]");
  TORCH_CHECK(reference.is_cuda(), operation, " requires CUDA operands");
  for (int64_t term = 0; term < ciphertexts.size(); ++term) {
    TORCH_CHECK(ciphertexts[term].device() == reference.device() &&
                    ciphertexts[term].scalar_type() == reference.scalar_type(),
                operation,
                " requires every ciphertext on one device and dtype");
    auto ciphertext =
        view_rns_batch_3d(ciphertexts[term], "ciphertext term");
    TORCH_CHECK(ciphertext.sizes() == reference.sizes() &&
                    ciphertext.strides() == reference.strides(),
                operation,
                " ciphertext term layouts differ");
    ciphertext_rows.push_back(std::move(ciphertext));
  }
  for (int64_t term = 0; term < plaintexts.size(); ++term) {
    TORCH_CHECK(plaintexts[term].device() == reference.device() &&
                    plaintexts[term].scalar_type() == reference.scalar_type(),
                operation,
                " requires every plaintext on the ciphertext device and dtype");
    auto plaintext = view_rns_batch_3d(plaintexts[term], "plaintext term");
    TORCH_CHECK(plaintext.size(1) == reference.size(1) &&
                    plaintext.size(2) == reference.size(2) &&
                    reference.size(0) % plaintext.size(0) == 0,
                operation,
                " plaintext term shape does not broadcast over components");
    if (!plaintext_rows.empty()) {
      TORCH_CHECK(plaintext.sizes() == plaintext_rows[0].sizes() &&
                      plaintext.strides() == plaintext_rows[0].strides(),
                  operation,
                  " plaintext term layouts differ");
    }
    plaintext_rows.push_back(std::move(plaintext));
  }
  TORCH_CHECK(rns_params.device() == reference.device() &&
                  rns_params.scalar_type() == reference.scalar_type(),
              operation,
              " RNS parameters must match the input device and dtype");
  check_rns_parameter_rows(reference, rns_params, operation);

  std::vector<int64_t> output_shape;
  output_shape.reserve(ciphertexts[0].dim() + 1);
  output_shape.push_back(ciphertexts[0].size(0));
  output_shape.push_back(group_count);
  output_shape.insert(output_shape.end(),
                      ciphertexts[0].sizes().begin() + 1,
                      ciphertexts[0].sizes().end());
  auto out = torch::empty(output_shape, ciphertexts[0].options());
  const bool has_shared_kernel =
      (ciphertexts.size() == 16 && group_count == 16) ||
      (ciphertexts.size() == 8 && group_count == 8) ||
      (ciphertexts.size() == 4 && group_count == 8);
  if (has_shared_kernel) {
    AT_DISPATCH_INTEGRAL_TYPES(
        reference.scalar_type(), "rns_montgomery_weighted_sums_cuda", [&] {
          launch_montgomery_weighted_sums<scalar_t>(
              out, ciphertext_rows, plaintext_rows, rns_params, group_count);
        });
    return out;
  }

  std::vector<torch::Tensor> group_plaintexts;
  group_plaintexts.reserve(ciphertexts.size());
  for (int64_t group = 0; group < group_count; ++group) {
    group_plaintexts.clear();
    const int64_t first = group * ciphertexts.size();
    for (int64_t term = 0; term < ciphertexts.size(); ++term) {
      group_plaintexts.push_back(plaintexts[first + term]);
    }
    out.select(1, group).copy_(rns_montgomery_weighted_sum_cuda(
        ciphertexts, group_plaintexts, rns_params));
  }
  return out;
}

namespace {

template <RepetitionLayout layout>
torch::Tensor rns_montgomery_mul_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params,
    const char* operation_name) {
  auto out = torch::empty_like(lhs);
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(compressed_rhs, "compressed_rhs");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_compressed_rns_binary_3d(lhs_rows, rhs_rows, operation_name);
  check_rns_parameter_rows(lhs_rows, rns_params, operation_name);
  const int unique_mask = rhs_rows.size(2) - 1;
  const int repeat_shift =
      compressed_repeat_shift(lhs_rows.size(2), rhs_rows.size(2));
  AT_DISPATCH_INTEGRAL_TYPES(
      lhs.scalar_type(), "rns_montgomery_mul_compressed", [&] {
        const int device = lhs_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(lhs_rows.size(1),
                  (lhs_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  lhs_rows.size(0));
        rns_montgomery_mul_compressed_kernel<scalar_t, layout>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(lhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2),
                unique_mask,
                repeat_shift);
      });
  return out;
}

}  // namespace

torch::Tensor rns_montgomery_mul_cyclic_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_compressed_cuda<RepetitionLayout::kCyclic>(
      lhs, compressed_rhs, rns_params, "rns_montgomery_mul_cyclic_compressed");
}

torch::Tensor rns_montgomery_mul_contiguous_compressed_cuda(
    const torch::Tensor lhs,
    const torch::Tensor compressed_rhs,
    const torch::Tensor rns_params) {
  return rns_montgomery_mul_compressed_cuda<RepetitionLayout::kContiguous>(
      lhs,
      compressed_rhs,
      rns_params,
      "rns_montgomery_mul_contiguous_compressed");
}

void rns_montgomery_mul_row_scalars_inplace_cuda(
    torch::Tensor residues,
    const torch::Tensor row_scalars,
    const torch::Tensor rns_params) {
  auto rows = view_rns_batch_3d(residues, "residues");
  check_rns_parameter_rows(rows, rns_params, "rns_montgomery_mul_row_scalars");
  check_rns_row_vector(row_scalars,
                       rows.size(1),
                       "rns_montgomery_mul_row_scalars",
                       "row_scalars");
  AT_DISPATCH_INTEGRAL_TYPES(
      residues.scalar_type(), "rns_montgomery_mul_row_scalars", [&] {
        const int device = rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(rows.size(1),
                  (rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  rows.size(0));
        rns_montgomery_mul_row_scalars_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(row_scalars, scalar_t, 1),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2));
      });
}

void rns_to_montgomery_inplace_cuda(torch::Tensor standard_residues,
                                    const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kToMontgomery>(
      standard_residues, rns_params, "standard_residues", "rns_to_montgomery");
}

void rns_from_montgomery_inplace_cuda(torch::Tensor montgomery_residues,
                                      const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kFromMontgomery>(montgomery_residues,
                                                        rns_params,
                                                        "montgomery_residues",
                                                        "rns_from_montgomery");
}

void rns_reduce_to_standard_inplace_cuda(torch::Tensor lazy_residues,
                                         const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kCanonicalize>(
      lazy_residues, rns_params, "lazy_residues", "rns_reduce_to_standard");
}

torch::Tensor rns_add_lazy_cuda(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params) {
  return binary_rns<BinaryRnsOperation::kAddLazy>(
      lhs, rhs, rns_params, "rns_add_lazy");
}

torch::Tensor rns_add_lazy_with_twice_modulus_cuda(
    const torch::Tensor lhs,
    const torch::Tensor rhs,
    const torch::Tensor twice_modulus) {
  auto out = torch::empty_like(lhs);
  const auto lhs_rows = view_rns_batch_3d(lhs, "lhs");
  const auto rhs_rows = view_rns_batch_3d(rhs, "rhs");
  auto out_rows = view_rns_batch_3d(out, "out");
  check_rns_binary_3d(
      lhs_rows, rhs_rows, "rns_add_lazy_with_twice_modulus", true);
  check_rns_row_vector(twice_modulus,
                       lhs_rows.size(1),
                       "rns_add_lazy_with_twice_modulus",
                       "twice_modulus");
  AT_DISPATCH_INTEGRAL_TYPES(
      lhs.scalar_type(), "rns_add_lazy_with_twice_modulus", [&] {
        const int device = lhs_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(lhs_rows.size(1),
                  (lhs_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  lhs_rows.size(0));
        rns_add_lazy_twice_modulus_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(lhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rhs_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(twice_modulus, scalar_t, 1));
      });
  return out;
}

torch::Tensor rns_sub_lazy_cuda(const torch::Tensor lhs,
                                const torch::Tensor rhs,
                                const torch::Tensor rns_params) {
  return binary_rns<BinaryRnsOperation::kSubLazy>(
      lhs, rhs, rns_params, "rns_sub_lazy");
}

void rns_center_residues_inplace_cuda(torch::Tensor standard_residues,
                                      const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kCenter>(standard_residues,
                                                rns_params,
                                                "standard_residues",
                                                "rns_center_residues");
}

void rns_shift_residues_positive_inplace_cuda(torch::Tensor centered_residues,
                                              const torch::Tensor rns_params) {
  unary_rns_inplace<UnaryRnsOperation::kShiftPositive>(
      centered_residues,
      rns_params,
      "centered_residues",
      "rns_shift_residues_positive");
}

torch::Tensor rns_lift_centered_coefficients_cuda(
    const torch::Tensor centered_coefficients,
    const torch::Tensor twice_modulus) {
  const auto coefficient_rows =
      view_coefficient_batch_2d(centered_coefficients, "centered_coefficients");
  check_rns_row_vector(twice_modulus,
                       twice_modulus.size(0),
                       "rns_lift_centered_coefficients",
                       "twice_modulus");
  std::vector<int64_t> output_shape(centered_coefficients.sizes().begin(),
                                    centered_coefficients.sizes().end() - 1);
  output_shape.push_back(twice_modulus.size(0));
  output_shape.push_back(centered_coefficients.size(-1));
  auto out = centered_coefficients.new_empty(output_shape);
  auto out_rows = view_rns_batch_3d(out, "out");
  AT_DISPATCH_INTEGRAL_TYPES(
      centered_coefficients.scalar_type(),
      "rns_lift_centered_coefficients",
      [&] {
        const int device = coefficient_rows.device().index();
        cudaSetDevice(device);
        auto stream = at::cuda::getCurrentCUDAStream(device);
        dim3 grid(out_rows.size(1),
                  (out_rows.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
                  out_rows.size(0));
        rns_lift_centered_coefficients_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(out_rows, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(coefficient_rows, scalar_t, 2),
                FHELIUM_CUDA_ACCESSOR32(twice_modulus, scalar_t, 1));
      });
  return out;
}
