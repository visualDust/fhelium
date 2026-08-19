#include "inverse_ntt_cuda.h"
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
