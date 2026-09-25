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
