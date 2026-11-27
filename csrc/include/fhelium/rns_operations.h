#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/export.h>
#include <fhelium/operation_tables.h>

#include <cstdint>
#include <optional>
#include <string_view>

//! Whole RNS operations over prepared tables: hybrid-digit ModUp,
//! centered basis extension, ModDown by the auxiliary product, joint ModDown
//! and rescale, rounded trailing quotients, and rational rescale.
//!
//! Each function reads its prepared tables from one struct of
//! `fhelium/operation_tables.h` and lists the fields it reads. Residue operands
//! have shape `[*batch, limb, N]`. Parameter tables follow the layout of
//! `fhelium/rns_parameters.h`. NTT steps use the policy names and schedules
//! of `fhelium/ntt_operations.h`. `digits` lists hybrid digits as
//! flattened `(start, stop, key_digit)` triples, where `[start, stop)` are
//! source rows. `rounding` is `nearest` or `floor`. Inputs and tables are
//! read-only, and results own their storage unless a function states
//! otherwise. Each operation has an `_out` form that writes its result into a
//! caller-supplied Tensor with the result's shape, dtype, and device that does
//! not overlap the operands.

namespace fhelium::rns {

/// Lifts hybrid digit `digit_index` of coefficient standard residues over the
/// active Q rows to QP Montgomery residues.
///
/// For digit rows $[s, t)$ with product $D$, the Garner decomposition of the
/// least nonnegative representative $a \in [0, D)$ is extended to every row
/// of `qp_parameters` and multiplied by $R$. Reads `qp_parameters` and the
/// `normalizers`, `propagation`, `reduction`, and `extension` fields of
/// `digits[digit_index]`. The result has the `qp_parameters` row count.
FHELIUM_API at::Tensor hybrid_modup_digit(const at::Tensor& source,
                                          const KeySwitchTables& tables,
                                          int64_t digit_index,
                                          c10::IntArrayRef digits);

/// Out form of `hybrid_modup_digit`.
FHELIUM_API void hybrid_modup_digit_out(const at::Tensor& source,
                                        const KeySwitchTables& tables,
                                        int64_t digit_index,
                                        c10::IntArrayRef digits,
                                        at::Tensor& out);

/// Returns the residues of the centered CRT representative of `source` in the
/// target basis.
///
/// With $D$ the product of the source primes and $a \in [0, D)$ the CRT value
/// of the rows of `source`, the result holds $a - D$ when $a > \lfloor D/2
/// \rfloor$ and $a$ otherwise, reduced to standard residues in $[0, p_t)$ of
/// each target prime. Reads every field of `tables`.
FHELIUM_API at::Tensor centered_basis_extend(
    const at::Tensor& source, const BasisExtensionTables& tables);

/// Out form of `centered_basis_extend`. A defined `workspace` holds the
/// temporaries: a contiguous uint8 Tensor of at least
/// `centered_basis_extend_workspace_bytes` bytes on the operand device with a
/// 256-byte aligned base. With an undefined `workspace`, the temporaries are
/// allocated per call.
FHELIUM_API void centered_basis_extend_out(
    const at::Tensor& source,
    const BasisExtensionTables& tables,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `centered_basis_extend_out` for these operands; reads
/// only their sizes, dtypes and devices.
FHELIUM_API int64_t centered_basis_extend_workspace_bytes(
    const at::Tensor& source, const BasisExtensionTables& tables);

/// Returns $\lfloor (x + h) / D \rfloor$ on the leading rows, where the
/// trailing `drop_count` rows of `source`, after the optional `row_order`
/// gather, are the divisor rows with product $D$.
///
/// $h = (D-1)/2$ (field `half`) for nearest rounding and $0$ for floor.
/// Input and output are coefficient standard residues. Reads `inverses`,
/// `parameters`, and the optional `half` and `row_order`.
FHELIUM_API at::Tensor rounded_trailing_quotient(
    const at::Tensor& source,
    const TrailingQuotientTables& tables,
    int64_t drop_count,
    std::string_view rounding);

/// Out form of `rounded_trailing_quotient`. A defined `workspace` holds the
/// temporaries: a contiguous uint8 Tensor of at least
/// `rounded_trailing_quotient_workspace_bytes` bytes on the operand device with
/// a 256-byte aligned base. With an undefined `workspace`, the temporaries are
/// allocated per call.
FHELIUM_API void rounded_trailing_quotient_out(
    const at::Tensor& source,
    const TrailingQuotientTables& tables,
    int64_t drop_count,
    std::string_view rounding,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `rounded_trailing_quotient_out` for these operands; reads
/// only their sizes, dtypes and devices.
FHELIUM_API int64_t
rounded_trailing_quotient_workspace_bytes(const at::Tensor& source,
                                          const TrailingQuotientTables& tables,
                                          int64_t drop_count,
                                          std::string_view rounding);

/// Returns $\lfloor (x + h) / P \rfloor \bmod Q$ for coefficient standard QP
/// residues whose trailing `p_count` rows are the auxiliary primes.
///
/// Reads `moddown_inverses`, `qp_parameters`, and `moddown_half` for nearest
/// rounding. When `moddown_crt` holds the centered quotient tables, the
/// operands of `fhelium::rns::crt_basis_extend` with source P, target Q, and
/// standard-form matrix entries
/// $-p_j^{-1} \bmod q_i$, the result is $(x - r - kP)/P \bmod Q$, where
/// $r + kP$ is the centered CRT value of the P rows and $|k|$ is below half
/// the P row count. It also reads `p_inverse` ($P^{-1} R \bmod q_i$) and
/// `q_parameters`, and `rounding` does not apply.
FHELIUM_API at::Tensor moddown_qp_to_q(const at::Tensor& source,
                                       const KeySwitchTables& tables,
                                       int64_t p_count,
                                       std::string_view rounding);

/// Out form of `moddown_qp_to_q`. A defined `workspace` holds the temporaries:
/// a contiguous uint8 Tensor of at least `moddown_qp_to_q_workspace_bytes`
/// bytes on the operand device with a 256-byte aligned base. With an undefined
/// `workspace`, the temporaries are allocated per call.
FHELIUM_API void moddown_qp_to_q_out(
    const at::Tensor& source,
    const KeySwitchTables& tables,
    int64_t p_count,
    std::string_view rounding,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `moddown_qp_to_q_out` for these operands; reads only
/// their sizes, dtypes and devices.
FHELIUM_API int64_t
moddown_qp_to_q_workspace_bytes(const at::Tensor& source,
                                const KeySwitchTables& tables,
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
/// `moddown_half`, `p_inverse_ntt`, and `q_forward`. The centered quotient
/// tables `moddown_crt` select the remainder of `moddown_qp_to_q`.
FHELIUM_API at::Tensor moddown_ntt_qp_to_q(
    const at::Tensor& source,
    const std::optional<at::Tensor>& coefficient_c0,
    const KeySwitchTables& tables,
    std::string_view ntt_backend,
    int64_t p_count,
    std::string_view rounding);

/// Out form of `moddown_ntt_qp_to_q`.
FHELIUM_API void moddown_ntt_qp_to_q_out(
    const at::Tensor& source,
    const std::optional<at::Tensor>& coefficient_c0,
    const KeySwitchTables& tables,
    std::string_view ntt_backend,
    int64_t p_count,
    std::string_view rounding,
    at::Tensor& out);

/// Computes $\lfloor (v + P a + H_P + P H_D) / (P D) \rfloor$ on the
/// surviving Q rows, combining ModDown by $P$ with rescale by the dropped Q
/// product $D$.
///
/// `products` are NTT Montgomery QP key products $v$ and `base` the NTT
/// Montgomery Q components $a$; the leading component extent of `products`
/// selects the matching prefix of `base`. `q_count` is the active Q row count,
/// `joint_drop_count` the dropped row count, and `joint_drop_is_prefix`
/// whether the dropped rows lead the Q rows; otherwise `joint.drop_positions`
/// and `joint.remaining_positions` select them. The result is coefficient
/// standard or NTT Montgomery residues according to `output_domain`
/// (`coefficient` or `ntt`). Reads `q_parameters` and the `joint` fields
/// `p_times_base`, `dropped_offset`, `dropped_parameters`, `remaining_offset`,
/// `remaining_parameters`, `inverses`, `ordered_parameters`,
/// `product_inverse`, `dropped_inverse`, and `remaining_forward` or
/// `remaining_inverse`. When a restored prime set $R$ is present, the result
/// is $\lfloor (R(v + Pa) + H)/(PD) \rfloor$ over $B'$: `joint.restore` holds
/// the Montgomery multipliers of $R$ for the product rows described by
/// `joint.product_parameters`, `joint.p_times_base` holds $PR$, the
/// `joint.remaining_*` tables describe $B'$, and `joint.target_keep_positions`
/// places the surviving Q rows in $B'$. When `joint.crt_t` is defined, the
/// quotient by $P$ uses the centered CRT remainder of the P rows
/// (`joint.crt_d` into the D rows, `joint.crt_t` into $B'$,
/// `joint.crt_p_inverse`), and the following quotient by $D$ is exact
/// (`joint.crt_inverses`, `joint.crt_ordered_parameters`, nearest when
/// `joint.crt_half` is defined).
FHELIUM_API at::Tensor moddown_add_rescale(const at::Tensor& products,
                                           const at::Tensor& base,
                                           const KeySwitchTables& tables,
                                           std::string_view ntt_backend,
                                           int64_t q_count,
                                           int64_t joint_drop_count,
                                           bool joint_drop_is_prefix,
                                           std::string_view output_domain);

/// Out form of `moddown_add_rescale`. A defined `workspace` holds the
/// temporaries: a contiguous uint8 Tensor of at least
/// `moddown_add_rescale_workspace_bytes` bytes on the operand device with a
/// 256-byte aligned base. With an undefined `workspace`, the temporaries are
/// allocated per call.
FHELIUM_API void moddown_add_rescale_out(
    const at::Tensor& products,
    const at::Tensor& base,
    const KeySwitchTables& tables,
    std::string_view ntt_backend,
    int64_t q_count,
    int64_t joint_drop_count,
    bool joint_drop_is_prefix,
    std::string_view output_domain,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `moddown_add_rescale_out` for these operands; reads only
/// their sizes, dtypes and devices.
FHELIUM_API int64_t
moddown_add_rescale_workspace_bytes(const at::Tensor& products,
                                    const at::Tensor& base,
                                    const KeySwitchTables& tables,
                                    std::string_view ntt_backend,
                                    int64_t q_count,
                                    int64_t joint_drop_count,
                                    bool joint_drop_is_prefix,
                                    std::string_view output_domain);

/// Rescales a value from basis $B$ to basis $B'$ by the ratio $D/R$, with
/// $D = B \setminus B'$ and $R = B' \setminus B$.
///
/// `source` holds coefficient standard residues or, for `input_domain =
/// "ntt"`, NTT Montgomery residues of $x$ over $B$ in the row order of
/// `source_parameters`. The result over the rows of `target_parameters` is
/// $\lfloor (xR + H)/D \rfloor$ in the input domain, with $H = (D - 1)/2$
/// for nearest rounding and $0$ for floor rounding, computed with one exact
/// mixed-radix division by $D$. Rows of $R$ hold $\lfloor (xR + H)/D
/// \rfloor \bmod r$ although $xR \equiv 0 \pmod r$.
///
/// Reads `source_parameters`, `target_parameters`, `dropped_parameters`,
/// `ordered_parameters` (target rows followed by D rows), `drop_positions`
/// and `keep_positions` (rows of $B$), `inverses` (the trailing-quotient
/// inverses of the ordered rows), `dropped_offset` and `target_offset`
/// (residues of $H$), and, when $R$ is nonempty, `restore` (Montgomery
/// multipliers of $R$ for the rows of $B$) and `target_keep_positions` (rows
/// of $B'$ holding $B \cap B'$). Without `target_keep_positions`, the kept
/// rows are the rows of $B'$ in order. NTT input also reads `product_inverse`
/// ($D^{-1}$ for the target rows), `dropped_inverse`, and `target_forward`.
FHELIUM_API at::Tensor rational_rescale(const at::Tensor& source,
                                        const RescaleTables& tables,
                                        std::string_view input_domain,
                                        std::string_view ntt_backend);

/// Out form of `rational_rescale`. A defined `workspace` holds the temporaries:
/// a contiguous uint8 Tensor of at least `rational_rescale_workspace_bytes`
/// bytes on the operand device with a 256-byte aligned base. With an undefined
/// `workspace`, the temporaries are allocated per call.
FHELIUM_API void rational_rescale_out(
    const at::Tensor& source,
    const RescaleTables& tables,
    std::string_view input_domain,
    std::string_view ntt_backend,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `rational_rescale_out` for these operands; reads only
/// their sizes, dtypes and devices.
FHELIUM_API int64_t
rational_rescale_workspace_bytes(const at::Tensor& source,
                                 const RescaleTables& tables,
                                 std::string_view input_domain,
                                 std::string_view ntt_backend);

}  // namespace fhelium::rns
