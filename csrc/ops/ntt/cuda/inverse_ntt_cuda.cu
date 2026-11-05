#include "inverse_ntt_cuda.h"
#include "stage_range_cuda.h"
#include "../../common/cuda/kernel_support.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"
#include "../ntt_execution_constants.h"
#include "ntt_representation_kernels.cuh"

#include <algorithm>
#include <array>

//------------------------------------------------------------------
// Inverse NTT implementation assembly
//------------------------------------------------------------------

// Device kernels and launch helpers must be declared before entry points.
#include "inverse_ntt_compact_detail.cuh"
#include "inverse_ntt_indexed_detail.cuh"
#include "inverse_ntt_power_of_two_radix_detail.cuh"

// Public CUDA entry points.
#include "inverse_ntt_compact_cuda.inc.cuh"
#include "inverse_ntt_montgomery_indexed_cuda.inc.cuh"
#include "inverse_ntt_power_of_two_radix_cuda.inc.cuh"
#include "inverse_ntt_to_centered_indexed_cuda.inc.cuh"
#include "inverse_ntt_to_standard_indexed_cuda.inc.cuh"
#include "inverse_ntt_to_standard_lazy_indexed_cuda.inc.cuh"

// Unnormalized stage ranges compose with generated producer/consumer kernels.
void inverse_ntt_compact_stage_range_cuda(
    torch::Tensor residues, const torch::Tensor twiddles,
    const torch::Tensor parameters, int64_t start_stage, int64_t end_stage,
    int64_t grouped_stage_count) {
  auto operand = view_rns_batch_3d(residues, "residues");
  const auto device_id = residues.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(),
                            "inverse_ntt_compact_stage_range_cuda", ([&] {
    launch_inverse_ntt_compact_grouped_stage_range_cuda<scalar_t>(
        operand, twiddles, parameters, static_cast<int>(grouped_stage_count),
        static_cast<int>(operand.size(1)), static_cast<int>(start_stage),
        static_cast<int>(end_stage), stream);
  }));
}

template <typename scalar_t, int BITS, int RADIX>
void inverse_ntt_fixed_radix_range(
    torch::Tensor operand, const torch::Tensor outer,
    const torch::Tensor roots, const torch::Tensor parameters,
    int start, int end, cudaStream_t stream) {
  const int n = static_cast<int>(operand.size(-1));
  const auto a = FHELIUM_CUDA_ACCESSOR32(operand, scalar_t, 3);
  const auto t = FHELIUM_CUDA_ACCESSOR32(outer, scalar_t, 2);
  const auto r = FHELIUM_CUDA_ACCESSOR32(roots, scalar_t, 2);
  const auto p = FHELIUM_CUDA_ACCESSOR32(parameters, scalar_t, 2);
  for (int stage = start; stage < end; stage += BITS) {
    const int offset = n - (n >> stage);
    launch_inverse_ntt_fixed_radix_digit_cuda<scalar_t, BITS, RADIX>(
        a, t, r, p, stage, offset, stream);
  }
}

void inverse_ntt_fixed_radix_stage_range_cuda(
    torch::Tensor residues, const torch::Tensor outer,
    const torch::Tensor roots, const torch::Tensor parameters,
    int64_t start_stage, int64_t end_stage, int64_t radix) {
  auto operand = view_rns_batch_3d(residues, "residues");
  const auto device_id = residues.device().index();
  cudaSetDevice(device_id);
  auto stream = at::cuda::getCurrentCUDAStream(device_id);
  AT_DISPATCH_INTEGRAL_TYPES(residues.scalar_type(),
                            "inverse_ntt_fixed_radix_stage_range_cuda", ([&] {
    if (radix == 4) {
      inverse_ntt_fixed_radix_range<scalar_t, 2, 4>(operand, outer, roots, parameters, start_stage, end_stage, stream);
    } else if (radix == 8) {
      inverse_ntt_fixed_radix_range<scalar_t, 3, 8>(operand, outer, roots, parameters, start_stage, end_stage, stream);
    } else {
      inverse_ntt_fixed_radix_range<scalar_t, 4, 16>(operand, outer, roots, parameters, start_stage, end_stage, stream);
    }
  }));
}
