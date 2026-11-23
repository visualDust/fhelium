#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/export.h>
#include <fhelium/tensor_table.h>

#include <cstdint>
#include <string_view>

//! NTT transitions selected by NTT policy name over named schedule tables.
//!
//! A policy name is one of the FHElium NTT backend names: `radix2_indexed`,
//! `radix2_compact_group{4,8,16}_smem8`,
//! `radix2_compact_group{4,8,16}_shoup32_smem8`, and
//! `radix{4,8,16}_compact`. For a basis prefix `b` and direction `d`
//! (`forward` or `inverse`), the table supplies `b_d_0`, the `rns_params` of
//! the transformed rows, and the schedule of the policy family: indexed
//! radix-2 reads twiddles `b_d_1`, even indices `b_d_2`, and odd indices
//! `b_d_3`; compact radix-2 reads twiddles `b_d_1`; fixed radix reads outer
//! twiddles `b_d_1` and radix root powers `b_d_2`. An empty basis prefix
//! names the operands `d_0`, `d_1`, and so on.
//!
//! A transition is one of `forward_montgomery` (coefficient Montgomery to NTT
//! Montgomery), `forward_to_montgomery` (coefficient standard to NTT
//! Montgomery), `inverse_montgomery` (NTT Montgomery to coefficient
//! Montgomery), `inverse_to_standard` (NTT Montgomery to coefficient
//! standard in $[0, q_j)$), `inverse_to_standard_lazy` (the same in
//! $[0, 2q_j)$), and `inverse_to_centered` (NTT Montgomery to centered
//! coefficients).

namespace fhelium::ntt {

/// Applies `transition` to `residues` in place with the `basis` schedule of
/// policy `ntt_backend`.
FHELIUM_API void transform_(at::Tensor& residues,
                            const TensorTable& tables,
                            std::string_view ntt_backend,
                            std::string_view basis,
                            std::string_view transition);

/// Returns `transition` of `residues` with the `basis` schedule of policy
/// `ntt_backend`; `residues` is unchanged.
FHELIUM_API at::Tensor transform(const at::Tensor& residues,
                                 const TensorTable& tables,
                                 std::string_view ntt_backend,
                                 std::string_view basis,
                                 std::string_view transition);

}  // namespace fhelium::ntt
