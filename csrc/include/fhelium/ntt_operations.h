#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/export.h>
#include <fhelium/operation_tables.h>

#include <cstdint>
#include <string_view>

//! NTT transitions selected by NTT policy name over the tables of one
//! direction.
//!
//! A policy name is one of the FHElium NTT backend names: `radix2_indexed`,
//! `radix2_compact`, `radix2_compact_shoup32`, and `radix{4,8,16}_compact`.
//! `NttTables::parameters` holds the `rns_params` of the transformed rows and
//! `NttTables::schedule` the schedule of the policy family: indexed radix-2
//! reads twiddles, even indices, and odd indices; compact radix-2 reads
//! twiddles; fixed radix reads outer twiddles and radix root powers. Forward
//! transitions read forward tables and inverse transitions inverse tables.
//!
//! A transition is one of `forward_montgomery` (coefficient Montgomery to NTT
//! Montgomery), `forward_to_montgomery` (coefficient standard to NTT
//! Montgomery), `inverse_montgomery` (NTT Montgomery to coefficient
//! Montgomery in $[0, 2q_j)$), `inverse_to_standard` (NTT Montgomery to
//! coefficient standard in $[0, q_j)$), `inverse_to_standard_lazy` (the same in
//! $[0, 2q_j)$), and `inverse_to_centered` (NTT Montgomery to centered
//! coefficients).

namespace fhelium::ntt {

/// Applies `transition` to `residues` in place with the schedule `tables`
/// of policy `ntt_backend`.
FHELIUM_API void transform_(at::Tensor& residues,
                            const NttTables& tables,
                            std::string_view ntt_backend,
                            std::string_view transition);

/// Returns `transition` of `residues` with the schedule `tables` of policy
/// `ntt_backend`; `residues` is unchanged.
FHELIUM_API at::Tensor transform(const at::Tensor& residues,
                                 const NttTables& tables,
                                 std::string_view ntt_backend,
                                 std::string_view transition);

/// Out form of `transform`: writes `transition` of `residues` to `out`, a
/// Tensor of the same shape, dtype and device disjoint from `residues`.
FHELIUM_API void transform_out(const at::Tensor& residues,
                               const NttTables& tables,
                               std::string_view ntt_backend,
                               std::string_view transition,
                               at::Tensor& out);

}  // namespace fhelium::ntt
