#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/export.h>
#include <fhelium/tensor_table.h>

#include <cstdint>
#include <optional>
#include <string_view>

//! Whole RNS operations over named numerical operands: hybrid-digit ModUp,
//! centered basis extension, ModDown by the auxiliary product, joint ModDown
//! and rescale, rounded trailing quotients, and leading-prime rescale.
//!
//! Each function reads prepared tables from a `TensorTable`
//! (`fhelium/tensor_table.h`) and lists the names it reads. Residue operands
//! have shape `[*batch, limb, N]`. Parameter tables follow the layout of
//! `fhelium/rns_parameters.h`. NTT steps use the policy names and schedule
//! names of `fhelium/ntt_operations.h`. `digits` lists hybrid digits as
//! flattened `(start, stop, key_digit)` triples, where `[start, stop)` are
//! source rows. `rounding` is `nearest` or `floor`. Inputs and tables are
//! read-only, and results own their storage unless a function states
//! otherwise.

namespace fhelium::rns {

/// Lifts hybrid digit `digit_index` of coefficient standard residues over the
/// active Q rows to QP Montgomery residues.
///
/// For digit rows $[s, t)$ with product $D$, the Garner decomposition of the
/// least nonnegative representative $a \in [0, D)$ is extended to every row
/// of `qp_parameters` and multiplied by $R$. Reads `qp_parameters` and
/// `digit{i}_normalizers`, `digit{i}_propagation`, `digit{i}_reduction{0..3}`,
/// and `digit{i}_extension` for $i$ = `digit_index`. The result has the
/// `qp_parameters` row count.
FHELIUM_API at::Tensor hybrid_modup_digit(const at::Tensor& source,
                                          const TensorTable& tables,
                                          int64_t digit_index,
                                          c10::IntArrayRef digits);

/// Returns the residues of the centered CRT representative of `source` in the
/// target basis.
///
/// With $D$ the product of the source primes and $a \in [0, D)$ the CRT value
/// of the rows of `source`, the result holds $a - D$ when $a > \lfloor D/2
/// \rfloor$ and $a$ otherwise, reduced to standard residues in $[0, p_t)$ of
/// each target prime. Reads `source_parameters`, `target_parameters`,
/// `source_lo`, `source_hi`, `source_neg_lo`, `source_neg_hi`, `normalizers`,
/// `propagation`, `extension`, `half_digits`, `source_product`, and
/// `target_moduli`.
FHELIUM_API at::Tensor centered_basis_extend(const at::Tensor& source,
                                             const TensorTable& tables);

/// Returns $\lfloor (x + h) / D \rfloor$ on the leading rows, where the
/// trailing `drop_count` rows of `source`, after the optional `row_order`
/// gather, are the divisor rows with product $D$.
///
/// $h = (D-1)/2$ (table `half`) for nearest rounding and $0$ for floor. Input
/// and output are coefficient standard residues. Reads `moddown_inverses`,
/// `parameters`, and the optional `half` and `row_order`.
FHELIUM_API at::Tensor rounded_trailing_quotient(const at::Tensor& source,
                                                 const TensorTable& tables,
                                                 int64_t drop_count,
                                                 std::string_view rounding);

/// Returns $\lfloor (x + h) / P \rfloor \bmod Q$ for coefficient standard QP
/// residues whose trailing `p_count` rows are the auxiliary primes.
///
/// Reads `moddown_inverses`, `qp_parameters`, and `moddown_half` for nearest
/// rounding.
FHELIUM_API at::Tensor moddown_qp_to_q(const at::Tensor& source,
                                       const TensorTable& tables,
                                       int64_t p_count,
                                       std::string_view rounding);

/// Returns $\widehat{x}_Q P^{-1} + \operatorname{NTT}_Q(-r/P)$ for NTT
/// Montgomery QP residues, where $r$ is the remainder of the coefficient
/// representative modulo $P$ ($r \in [0, P)$ for floor, centered for
/// nearest).
///
/// When `coefficient_c0` is supplied, it is added to the coefficient
/// correction of component zero before the forward transform. The result
/// holds NTT Montgomery Q residues in the lazy range. Reads `qp_parameters`,
/// `q_parameters`, `moddown_inverses`, `p_inverse`, the optional
/// `moddown_half`, and the `p` inverse and `q` forward schedules.
FHELIUM_API at::Tensor moddown_ntt_qp_to_q(
    const at::Tensor& source,
    const std::optional<at::Tensor>& coefficient_c0,
    const TensorTable& tables,
    std::string_view ntt_backend,
    int64_t p_count,
    std::string_view rounding);

/// Computes $\lfloor (v + P a + H_P + P H_D) / (P D) \rfloor$ on the
/// surviving Q rows, combining ModDown by $P$ with rescale by the dropped Q
/// product $D$.
///
/// `products` are NTT Montgomery QP key products $v$ and `base` the NTT
/// Montgomery Q components $a$; the leading component extent of `products`
/// selects the matching prefix of `base`. `q_count` is the active Q row count,
/// `joint_drop_count` the dropped row count, and `joint_drop_is_prefix`
/// whether the dropped rows lead the Q rows; otherwise
/// `joint_drop_positions` and `joint_remaining_positions` select them. The
/// result is coefficient standard or NTT Montgomery residues according to
/// `output_domain` (`coefficient` or `ntt`). Reads `q_parameters`,
/// `joint_p_times_base`, `joint_dropped_offset`, `joint_dropped_parameters`,
/// `joint_remaining_offset`, `joint_remaining_parameters`, `joint_inverses`,
/// `joint_ordered_parameters`, `joint_product_inverse`, and the
/// `joint_dropped` inverse and `joint_remaining` schedules.
FHELIUM_API at::Tensor moddown_add_rescale(const at::Tensor& products,
                                           const at::Tensor& base,
                                           const TensorTable& tables,
                                           std::string_view ntt_backend,
                                           int64_t q_count,
                                           int64_t joint_drop_count,
                                           bool joint_drop_is_prefix,
                                           std::string_view output_domain);

/// Divides by the product of the leading `drop_count` rows, one prime at a
/// time: each step replaces $x$ by $\lfloor (x + h_k) / q_k \rceil$ on the
/// remaining rows, with $h_k$ = `half_primes[k]` $= (q_k - 1)/2$ for nearest
/// rounding and $0$ for floor (`truncate`).
///
/// `input_domain` is `coefficient` (standard residues) or `ntt` (Montgomery
/// residues); the result has the same domain over the remaining rows. Reads
/// `step{k}_parameters` and `step{k}_inverse` for each step, and for NTT input
/// `prefix_inverse` with the `dropped` inverse and `remaining` forward
/// schedules.
FHELIUM_API at::Tensor rescale_drop_leading_primes(
    const at::Tensor& source,
    const TensorTable& tables,
    int64_t drop_count,
    c10::IntArrayRef half_primes,
    std::string_view rounding,
    std::string_view input_domain,
    std::string_view ntt_backend);

/// In-place form of `rescale_drop_leading_primes`. The quotient replaces the
/// remaining rows of `source`, and the returned Tensor is the view of `source`
/// that holds them.
FHELIUM_API at::Tensor rescale_drop_leading_primes_(
    at::Tensor& source,
    const TensorTable& tables,
    int64_t drop_count,
    c10::IntArrayRef half_primes,
    std::string_view rounding,
    std::string_view input_domain,
    std::string_view ntt_backend);

}  // namespace fhelium::rns
