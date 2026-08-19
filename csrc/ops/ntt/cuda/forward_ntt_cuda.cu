#include "forward_ntt_cuda.h"
#include "../../common/cuda/kernel_support.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"
#include "../ntt_execution_constants.h"
#include "ntt_representation_kernels.cuh"

#include <algorithm>
#include <array>

//------------------------------------------------------------------
// Forward NTT implementation assembly
//------------------------------------------------------------------

// Device kernels and launch helpers must be declared before entry points.
#include "forward_ntt_compact_detail.cuh"
#include "forward_ntt_compact_keyswitch_fused.inc.cuh"
#include "forward_ntt_indexed_detail.cuh"
#include "forward_ntt_power_of_two_radix_detail.cuh"

// Public CUDA entry points.
#include "forward_ntt_compact_cuda.inc.cuh"
#include "forward_ntt_indexed_cuda.inc.cuh"
#include "forward_ntt_power_of_two_radix_cuda.inc.cuh"
#include "forward_ntt_to_montgomery_indexed_cuda.inc.cuh"
