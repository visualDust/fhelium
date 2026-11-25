#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/string_view.h>
#include <fhelium/export.h>
#include <fhelium/rns_parameters.h>

#include <cstdint>
#include <optional>

//! Forward and inverse negacyclic number-theoretic transforms on RNS residue
//! Tensors.
//!
//! A residue operand has shape `[*batch, limb, N]`, where limb $j$ holds
//! residues modulo the prime $q_j$ described by column $j$ of `rns_params`
//! (`fhelium/rns_parameters.h`). Row $j$ of every twiddle table belongs to the
//! same prime. For a primitive $2N$-th root of unity $\psi_j$ and bit reversal
//! $\operatorname{br}$, the forward transform computes
//!
//! $$A_j[k] = \sum_{n=0}^{N-1} a_j[n]\, \psi_j^{(2\operatorname{br}(k)+1)n}
//! \bmod q_j,$$
//!
//! and the inverse transform computes its inverse including the factor $N^{-1}
//! \bmod q_j$. Twiddle tables hold Montgomery residues. Implementations view
//! `*batch` as one axis; they neither broadcast nor copy.
//!
//! Three table layouts select the implementation:
//!
//! - Indexed: int32 `even_indices` and `odd_indices` of shape `[stage, N/2]`
//!   that pair coefficients in each radix-2 stage, with twiddles of shape
//!   `[limb, stage, N/2]`. A schedule must be accepted by
//!   `validate_indexed_schedule` before use and must not change while execution
//!   that uses it is prepared.
//! - Compact: radix-2 twiddles of shape `[limb, N]` in bit-reversed stage
//!   order. For $N = 2^L$ the transform makes two passes over the operand: one
//!   pass applies $\lfloor L/2 \rfloor$ stages to coefficients spaced
//!   $2^{L - \lfloor L/2 \rfloor}$ apart, and the other applies the remaining
//!   stages to contiguous tiles. CUDA int32 kernels also accept
//!   `[limb, N, 2]` Shoup pairs.
//! - Power-of-two radix: `outer_twiddles` of shape `[limb, N-1]` and
//!   `radix_root_powers` of shape `[limb, radix]` for radix 4, 8, or 16;
//!   $\log_2 N$ is a multiple of $\log_2$ radix.
//!
//! Forward `montgomery` functions consume coefficient-domain Montgomery
//! residues, and forward `to_montgomery` functions consume coefficient-domain
//! standard residues; both produce NTT-domain Montgomery residues in the lazy
//! range $[0, 2q_j)$. Inverse functions consume NTT-domain Montgomery residues
//! in $[0, 2q_j)$ and produce coefficient-domain residues: `montgomery` keeps
//! Montgomery form in $[0, 2q_j)$, `to_standard_lazy` gives standard residues
//! in $[0, 2q_j)$, `to_standard` gives standard residues in $[0, q_j)$, and
//! `to_centered` gives standard residues in the centered range. A trailing
//! underscore marks a function that transforms its first argument in place; the
//! other functions allocate their result.

namespace fhelium::ntt {

/// Forward NTT of coefficient-domain Montgomery residues with an indexed
/// schedule, in place.
FHELIUM_API void forward_ntt_montgomery_indexed_(
    at::Tensor& montgomery_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain standard residues with an indexed
/// schedule, in place.
FHELIUM_API void forward_ntt_to_montgomery_indexed_(
    at::Tensor& standard_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain standard residues with an indexed
/// schedule; returns a new Tensor.
FHELIUM_API at::Tensor forward_ntt_to_montgomery_indexed(
    const at::Tensor& standard_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain Montgomery residues with compact twiddles,
/// in place.
FHELIUM_API void forward_ntt_montgomery_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain standard residues with compact twiddles,
/// in place.
FHELIUM_API void forward_ntt_to_montgomery_compact_(
    at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain standard residues with compact twiddles;
/// returns a new Tensor.
FHELIUM_API at::Tensor forward_ntt_to_montgomery_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Transforms a correction to the NTT domain and adds a scaled NTT addend in
/// the same pass.
///
/// `source` holds coefficient-domain standard residues $e$ and is overwritten
/// with $\operatorname{NTT}(e) + a \cdot m_j R^{-1} \bmod q_j$ in NTT-domain
/// Montgomery form and the lazy range. `addend` holds NTT-domain Montgomery
/// residues $a$ with the shape of `source`, and `multiplier` holds one
/// Montgomery scalar $m_j$ per limb. The transform uses compact twiddles as in
/// `forward_ntt_to_montgomery_compact_`.
FHELIUM_API void forward_ntt_to_montgomery_compact_add_scaled_(
    at::Tensor& source,
    const at::Tensor& addend,
    const at::Tensor& multiplier,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params);

/// Transforms one coefficient-domain key-switching digit and accumulates its
/// key products.
///
/// `coefficient_digit_qp` holds coefficient-domain Montgomery residues of one
/// extended digit. The function computes its compact forward NTT $D$ and then
/// $\mathrm{acc}_c \mathrel{+}= D \cdot k_c R^{-1} \bmod q_r$ for $c \in \{0,
/// 1\}$, with `key_digit_qp` of shape `[2, key_rows, N]` and digit row $r$
/// using key row `key_row_start + r`. `coefficient_digit_qp` serves as scratch
/// storage; its contents after the call are unspecified.
FHELIUM_API void forward_ntt_montgomery_compact_keyswitch_accumulate_(
    at::Tensor& coefficient_digit_qp,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    const at::Tensor& key_digit_qp,
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    int64_t key_row_start,
    int64_t key_p_row_start = -1);

/// Forward NTT of coefficient-domain Montgomery residues with power-of-two
/// radix stages, in place.
FHELIUM_API void forward_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain standard residues with power-of-two radix
/// stages, in place.
FHELIUM_API void forward_ntt_to_montgomery_power_of_two_radix_compact_(
    at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Forward NTT of coefficient-domain standard residues with power-of-two radix
/// stages; returns a new Tensor.
FHELIUM_API at::Tensor forward_ntt_to_montgomery_power_of_two_radix_compact(
    const at::Tensor& standard_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Inverse NTT with an indexed schedule, in place, producing coefficient-domain
/// Montgomery residues in $[0, 2q_j)$.
FHELIUM_API void inverse_ntt_montgomery_indexed_(
    at::Tensor& montgomery_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with an indexed schedule, in place, producing coefficient-domain
/// standard residues in $[0, 2q_j)$.
FHELIUM_API void inverse_ntt_to_standard_lazy_indexed_(
    at::Tensor& montgomery_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with an indexed schedule, in place, producing coefficient-domain
/// standard residues in $[0, q_j)$.
FHELIUM_API void inverse_ntt_to_standard_indexed_(
    at::Tensor& montgomery_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with an indexed schedule, in place, producing coefficient-domain
/// standard residues in the centered range.
FHELIUM_API void inverse_ntt_to_centered_indexed_(
    at::Tensor& montgomery_residues,
    const at::Tensor& even_indices,
    const at::Tensor& odd_indices,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with compact twiddles, in place, producing
/// coefficient-domain Montgomery residues in $[0, 2q_j)$.
FHELIUM_API void inverse_ntt_montgomery_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with compact twiddles, in place, producing
/// coefficient-domain standard residues in $[0, 2q_j)$.
FHELIUM_API void inverse_ntt_to_standard_lazy_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with compact twiddles, in place, producing
/// coefficient-domain standard residues in $[0, q_j)$.
FHELIUM_API void inverse_ntt_to_standard_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with compact twiddles, in place, producing
/// coefficient-domain standard residues in the centered range.
FHELIUM_API void inverse_ntt_to_centered_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& inverse_twiddles,
    const at::Tensor& rns_params);

/// Inverse NTT with power-of-two radix stages, in place, producing
/// coefficient-domain Montgomery residues in $[0, 2q_j)$.
FHELIUM_API void inverse_ntt_montgomery_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Inverse NTT with power-of-two radix stages, in place, producing
/// coefficient-domain standard residues in $[0, 2q_j)$.
FHELIUM_API void inverse_ntt_to_standard_lazy_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Inverse NTT with power-of-two radix stages, in place, producing
/// coefficient-domain standard residues in $[0, q_j)$.
FHELIUM_API void inverse_ntt_to_standard_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Inverse NTT with power-of-two radix stages, in place, producing
/// coefficient-domain standard residues in the centered range.
FHELIUM_API void inverse_ntt_to_centered_power_of_two_radix_compact_(
    at::Tensor& montgomery_residues,
    const at::Tensor& outer_twiddles,
    const at::Tensor& radix_root_powers,
    const at::Tensor& rns_params);

/// Checks that an indexed NTT schedule is a valid radix-2 pairing for ring
/// dimension `ring_dimension`.
///
/// `even_indices` and `odd_indices` are int32 Tensors of shape `[stage, N/2]`
/// with $N$ = `ring_dimension`. In every stage, the two Tensors together list
/// each coefficient index in $[0, N)$ exactly once. Schedules on an accelerator
/// are copied to the CPU for the check; the Tensors are neither modified nor
/// retained. Raises an error when the schedule is invalid.
FHELIUM_API void validate_indexed_schedule(const at::Tensor& even_indices,
                                           const at::Tensor& odd_indices,
                                           int64_t ring_dimension);

/// Executes a half-open interval of unnormalized power-of-two radix stages in
/// place.
///
/// The interval `[start_stage, end_stage)` with $0 \le$ `start_stage` $\le$
/// `end_stage` $\le \log_2 N$ selects forward decimation-in-frequency stages
/// or, when `inverse` is true, inverse decimation-in-time stages, on Montgomery
/// residues with `outer_twiddles` and `root_powers` in the power-of-two radix
/// layout. `start_stage` and `end_stage` are multiples of $\log_2$ radix.
/// Representation conversion and the inverse factor $N^{-1}$ are left to the
/// caller. Successive calls compose a full transform.
FHELIUM_API void fixed_radix_ntt_stage_range_(at::Tensor& residues,
                                              const at::Tensor& outer_twiddles,
                                              const at::Tensor& root_powers,
                                              const at::Tensor& rns_params,
                                              bool inverse,
                                              int64_t start_stage,
                                              int64_t end_stage);

}  // namespace fhelium::ntt
