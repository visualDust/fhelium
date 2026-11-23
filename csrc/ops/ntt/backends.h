#pragma once

#include <fhelium/ntt.h>
#include <fhelium/ntt_diagnostic.h>

// Backend entry points behind the public functions of the same name.
// The cpu namespace is compiled from cpu/ sources; the gpu namespace is
// compiled from either cuda/ or hip/ sources.

namespace fhelium::ntt::cpu {
decltype(::fhelium::ntt::forward_ntt_montgomery_indexed_)
    forward_ntt_montgomery_indexed_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_indexed_)
    forward_ntt_to_montgomery_indexed_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_indexed)
    forward_ntt_to_montgomery_indexed;
decltype(::fhelium::ntt::inverse_ntt_montgomery_indexed_)
    inverse_ntt_montgomery_indexed_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_lazy_indexed_)
    inverse_ntt_to_standard_lazy_indexed_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_indexed_)
    inverse_ntt_to_standard_indexed_;
decltype(::fhelium::ntt::inverse_ntt_to_centered_indexed_)
    inverse_ntt_to_centered_indexed_;
}  // namespace fhelium::ntt::cpu

namespace fhelium::ntt::gpu {
decltype(::fhelium::ntt::forward_ntt_montgomery_indexed_)
    forward_ntt_montgomery_indexed_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_indexed_)
    forward_ntt_to_montgomery_indexed_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_indexed)
    forward_ntt_to_montgomery_indexed;
decltype(::fhelium::ntt::forward_ntt_montgomery_compact_grouped_smem_)
    forward_ntt_montgomery_compact_grouped_smem_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_compact_grouped_smem_)
    forward_ntt_to_montgomery_compact_grouped_smem_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_compact_grouped_smem)
    forward_ntt_to_montgomery_compact_grouped_smem;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_compact_add_scaled_)
    forward_ntt_to_montgomery_compact_add_scaled_;
decltype(::fhelium::ntt::forward_ntt_montgomery_compact_keyswitch_accumulate_)
    forward_ntt_montgomery_compact_keyswitch_accumulate_;
decltype(::fhelium::ntt::forward_ntt_montgomery_power_of_two_radix_compact_)
    forward_ntt_montgomery_power_of_two_radix_compact_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_power_of_two_radix_compact_)
    forward_ntt_to_montgomery_power_of_two_radix_compact_;
decltype(::fhelium::ntt::forward_ntt_to_montgomery_power_of_two_radix_compact)
    forward_ntt_to_montgomery_power_of_two_radix_compact;
decltype(::fhelium::ntt::inverse_ntt_montgomery_indexed_)
    inverse_ntt_montgomery_indexed_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_lazy_indexed_)
    inverse_ntt_to_standard_lazy_indexed_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_indexed_)
    inverse_ntt_to_standard_indexed_;
decltype(::fhelium::ntt::inverse_ntt_to_centered_indexed_)
    inverse_ntt_to_centered_indexed_;
decltype(::fhelium::ntt::inverse_ntt_montgomery_compact_grouped_smem_)
    inverse_ntt_montgomery_compact_grouped_smem_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_lazy_compact_grouped_smem_)
    inverse_ntt_to_standard_lazy_compact_grouped_smem_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_compact_grouped_smem_)
    inverse_ntt_to_standard_compact_grouped_smem_;
decltype(::fhelium::ntt::inverse_ntt_to_centered_compact_grouped_smem_)
    inverse_ntt_to_centered_compact_grouped_smem_;
decltype(::fhelium::ntt::inverse_ntt_montgomery_power_of_two_radix_compact_)
    inverse_ntt_montgomery_power_of_two_radix_compact_;
decltype(::fhelium::ntt::
             inverse_ntt_to_standard_lazy_power_of_two_radix_compact_)
    inverse_ntt_to_standard_lazy_power_of_two_radix_compact_;
decltype(::fhelium::ntt::inverse_ntt_to_standard_power_of_two_radix_compact_)
    inverse_ntt_to_standard_power_of_two_radix_compact_;
decltype(::fhelium::ntt::inverse_ntt_to_centered_power_of_two_radix_compact_)
    inverse_ntt_to_centered_power_of_two_radix_compact_;
decltype(::fhelium::ntt::compact_ntt_stage_range_) compact_ntt_stage_range_;
decltype(::fhelium::ntt::fixed_radix_ntt_stage_range_)
    fixed_radix_ntt_stage_range_;
}  // namespace fhelium::ntt::gpu

namespace fhelium::ntt::diagnostic::gpu {
decltype(::fhelium::ntt::diagnostic::
             forward_ntt_montgomery_compact_grouped_smem_override_)
    forward_ntt_montgomery_compact_grouped_smem_override_;
decltype(::fhelium::ntt::diagnostic::
             forward_ntt_montgomery_power_of_two_radix_compact_override_)
    forward_ntt_montgomery_power_of_two_radix_compact_override_;
decltype(::fhelium::ntt::diagnostic::
             inverse_ntt_montgomery_power_of_two_radix_compact_override_)
    inverse_ntt_montgomery_power_of_two_radix_compact_override_;
}  // namespace fhelium::ntt::diagnostic::gpu
