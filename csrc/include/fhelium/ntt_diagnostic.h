#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/string_view.h>
#include <fhelium/export.h>
#include <fhelium/rns_parameters.h>

#include <cstdint>
#include <optional>

//! NTT functions with a selectable shared-memory transform size, for
//! differential testing and performance profiling.
//!
//! Each function computes the result of the function of the same name in
//! `fhelium::ntt` without the `_override` suffix. `shared_memory_log_n`,
//! between 0 and 8, replaces the default $\log_2$ size of the transform tail
//! executed in shared memory.

namespace fhelium::ntt::diagnostic {

/// `forward_ntt_montgomery_compact_grouped_smem_` with a selected shared-memory
/// transform size.
FHELIUM_API void forward_ntt_montgomery_compact_grouped_smem_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    int64_t grouped_stage_count,
    int64_t shared_memory_log_n);

/// `forward_ntt_montgomery_power_of_two_radix_compact_` with a selected
/// shared-memory transform size.
FHELIUM_API void forward_ntt_montgomery_power_of_two_radix_compact_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    int64_t shared_memory_log_n);

/// `inverse_ntt_montgomery_power_of_two_radix_compact_` with a selected
/// shared-memory transform size.
FHELIUM_API void inverse_ntt_montgomery_power_of_two_radix_compact_override_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params,
    int64_t shared_memory_log_n);

}  // namespace fhelium::ntt::diagnostic
