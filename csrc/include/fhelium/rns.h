#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/string_view.h>
#include <fhelium/export.h>
#include <fhelium/rns_parameters.h>

#include <cstdint>
#include <optional>

//! Modular arithmetic, representation conversion, and basis conversion on RNS
//! residue Tensors.
//!
//! A residue operand has shape `[*batch, limb, index]`. Limb $j$ holds residues
//! modulo the prime $q_j$ described by column $j$ of `rns_params` (row layout
//! in `fhelium/rns_parameters.h`), and the final axis holds polynomial
//! coefficients or NTT evaluations. Arithmetic functions preserve that domain.
//! Implementations view `*batch` as one axis without copying, and they
//! broadcast neither the limb axis nor the final axis. All Tensors of one call
//! have one integral dtype (int32 or int64 residues) and one device.
//!
//! A standard residue represents $x$ directly; a Montgomery residue represents
//! $x$ by $xR \bmod q_j$ with the Montgomery radix $R$ of `rns_params`.
//! Residues are stored in one of three ranges: reduced $[0, q_j)$, lazy $[0,
//! 2q_j)$, or centered $[-\lfloor q_j/2 \rfloor, \lfloor q_j/2 \rfloor]$.
//!
//! Functions without a trailing underscore return newly allocated Tensors that
//! do not alias an input. A trailing underscore marks a function that mutates
//! its first Tensor argument in place and keeps its storage. An `_out_`
//! function writes into the caller-owned `out` or `output` Tensor without
//! resizing it; that Tensor needs the shape of the functional result, the input
//! dtype and device, disjoint element addresses, and no overlap with any input
//! or table. All other Tensor arguments are read-only.

namespace fhelium::rns {

/// Multiplies two residue Tensors elementwise in Montgomery form.
///
/// Computes $c_j = a_j b_j R^{-1} \bmod q_j$ in the lazy range. With both
/// operands in Montgomery form, the result is the Montgomery form of the
/// product. `rhs` has the shape of `lhs` or a batch extent of one, which is
/// broadcast over the batch of `lhs`. The result has the shape of `lhs`.
FHELIUM_API at::Tensor montgomery_mul(const at::Tensor& lhs,
                                      const at::Tensor& rhs,
                                      const at::Tensor& rns_params);

/// Evaluates the weighted sum $\sum_t c_t p_t$ of ciphertext and plaintext
/// terms.
///
/// `ciphertexts` and `plaintexts` are non-empty lists of equal length. Every
/// ciphertext has shape `[component, *batch, limb, index]`, and every plaintext
/// has shape `[*batch, limb, index]` and is broadcast over the component axis.
/// For every component, batch element, limb, and index, the function computes
/// $\sum_t c_t p_t R^{-1} \bmod q_j$ and returns it in the reduced range with
/// the shape of `ciphertexts[0]`. Implementations may fuse the term loop.
FHELIUM_API at::Tensor montgomery_weighted_sum(at::TensorList ciphertexts,
                                               at::TensorList plaintexts,
                                               const at::Tensor& rns_params);

/// Evaluates `group_count` weighted sums that share one list of ciphertext
/// terms.
///
/// `plaintexts` is a row-major matrix of `group_count` groups with one
/// plaintext per ciphertext: group $g$ uses `plaintexts[g * T + t]` for term
/// $t$, where $T$ is the number of ciphertexts. The result has shape
/// `[component, group, *batch, limb, index]` with `out[:, g] =
/// montgomery_weighted_sum(ciphertexts, plaintexts[g*T : (g+1)*T])`. Groups are
/// evaluated in matrix order.
FHELIUM_API at::Tensor montgomery_weighted_sums(at::TensorList ciphertexts,
                                                at::TensorList plaintexts,
                                                int64_t group_count,
                                                const at::Tensor& rns_params);

/// Multiplies by a compressed right operand whose values repeat cyclically
/// along the final axis.
///
/// `compressed_rhs` has shape `[*compressed_batch, limb, U]`, where $U$ is a
/// power of two smaller than the final extent $N$ of `lhs`. Index $k$ of `lhs`
/// is multiplied by index $k \bmod U$ of `compressed_rhs`, as `montgomery_mul`
/// would with the expanded operand. The compressed batch extent, after `*batch`
/// is collapsed, is one or equal to that of `lhs`.
FHELIUM_API at::Tensor montgomery_mul_cyclic_compressed(
    const at::Tensor& lhs,
    const at::Tensor& compressed_rhs,
    const at::Tensor& rns_params);

/// Multiplies by a compressed right operand whose values repeat in contiguous
/// runs along the final axis.
///
/// `compressed_rhs` has shape `[*compressed_batch, limb, U]`, where $U$ is a
/// power of two smaller than the final extent $N$ of `lhs`. Index $k$ of `lhs`
/// is multiplied by index $\lfloor kU/N \rfloor$ of `compressed_rhs`, as
/// `montgomery_mul` would with the expanded operand. The compressed batch
/// extent, after `*batch` is collapsed, is one or equal to that of `lhs`.
FHELIUM_API at::Tensor montgomery_mul_contiguous_compressed(
    const at::Tensor& lhs,
    const at::Tensor& compressed_rhs,
    const at::Tensor& rns_params);

/// Multiplies every limb row by one Montgomery scalar in place.
///
/// `row_scalars` is a vector with one entry per limb. Each residue of limb $j$
/// becomes $x \cdot s_j R^{-1} \bmod q_j$ in the lazy range.
FHELIUM_API void montgomery_mul_row_scalars_(at::Tensor& residues,
                                             const at::Tensor& row_scalars,
                                             const at::Tensor& rns_params);

/// Converts standard residues to Montgomery form in place.
///
/// Each residue $x$ of limb $j$ becomes $xR \bmod q_j$, computed as a
/// Montgomery product with the `RNS_PARAM_R2` row. The result is in the lazy
/// range.
FHELIUM_API void to_montgomery_(at::Tensor& standard_residues,
                                const at::Tensor& rns_params);

/// Converts Montgomery residues to standard residues in place.
///
/// Each residue $y$ of limb $j$ becomes the Montgomery reduction $yR^{-1} \bmod
/// q_j$, in the lazy range.
FHELIUM_API void from_montgomery_(at::Tensor& montgomery_residues,
                                  const at::Tensor& rns_params);

/// Adds two residue Tensors modulo $2q_j$.
///
/// Computes $a + b$ reduced to the lazy range for lazy inputs. `rhs` has the
/// shape of `lhs` or a batch extent of one.
FHELIUM_API at::Tensor add_lazy(const at::Tensor& lhs,
                                const at::Tensor& rhs,
                                const at::Tensor& rns_params);

/// Adds two residue Tensors modulo $2q_j$, taking the moduli from a vector.
///
/// Equivalent to `add_lazy` with `twice_modulus[j]` $= 2q_j$ supplied as a
/// one-dimensional Tensor with one entry per limb in place of `rns_params`.
FHELIUM_API at::Tensor add_lazy_with_twice_modulus(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& twice_modulus);

/// Subtracts two residue Tensors modulo $2q_j$.
///
/// Computes $a - b$ reduced to the lazy range for lazy inputs. `rhs` has the
/// shape of `lhs` or a batch extent of one.
FHELIUM_API at::Tensor sub_lazy(const at::Tensor& lhs,
                                const at::Tensor& rhs,
                                const at::Tensor& rns_params);

/// Adds two standard residue Tensors and returns reduced residues.
///
/// Computes $(a + b) \bmod q_j$ in $[0, q_j)$ for inputs in $[0, q_j)$. `rhs`
/// has the shape of `lhs` or a batch extent of one; limb and final axes are
/// never broadcast.
FHELIUM_API at::Tensor add_standard(const at::Tensor& lhs,
                                    const at::Tensor& rhs,
                                    const at::Tensor& rns_params);

/// In-place form of `add_standard`; the sum replaces `lhs`.
FHELIUM_API void add_standard_(at::Tensor& lhs,
                               const at::Tensor& rhs,
                               const at::Tensor& rns_params);

/// Sums standard residues over one batch axis.
///
/// `dim` selects a batch axis of `source`, and the result keeps the remaining
/// axes. Each addition is reduced to $[0, q_j)$, so the result equals $\sum x
/// \bmod q_j$ for any association order.
FHELIUM_API at::Tensor sum_standard_batch(const at::Tensor& source,
                                          int64_t dim,
                                          const at::Tensor& rns_params);

/// Subtracts two standard residue Tensors and returns reduced residues.
///
/// Computes $(a - b) \bmod q_j$ in $[0, q_j)$ for inputs in $[0, q_j)$. `rhs`
/// has the shape of `lhs` or a batch extent of one.
FHELIUM_API at::Tensor sub_standard(const at::Tensor& lhs,
                                    const at::Tensor& rhs,
                                    const at::Tensor& rns_params);

/// In-place form of `sub_standard`; the difference replaces `lhs`.
FHELIUM_API void sub_standard_(at::Tensor& lhs,
                               const at::Tensor& rhs,
                               const at::Tensor& rns_params);

/// Multiplies every limb row by one Montgomery scalar and returns reduced
/// residues.
///
/// Computes $x \cdot s_j R^{-1} \bmod q_j$ in $[0, q_j)$ for the per-limb
/// vector `row_scalars`. With a Montgomery scalar $s_j = cR \bmod q_j$, a
/// standard input stays standard and is multiplied by $c$.
FHELIUM_API at::Tensor montgomery_mul_row_scalars_standard(
    const at::Tensor& residues,
    const at::Tensor& row_scalars,
    const at::Tensor& rns_params);

/// Reduces lazy residues to the reduced range in place.
///
/// Maps each residue in $[0, 2q_j)$ to $[0, q_j)$ by one conditional
/// subtraction of $q_j$.
FHELIUM_API void reduce_to_standard_(at::Tensor& lazy_residues,
                                     const at::Tensor& rns_params);

/// Maps reduced residues to centered representatives in place.
///
/// A residue $x \in [0, q_j)$ becomes $x$ when $x \le \lfloor q_j/2 \rfloor$
/// and $x - q_j$ otherwise. The result is signed and lies in $[-\lfloor q_j/2
/// \rfloor, \lfloor q_j/2 \rfloor]$.
FHELIUM_API void center_residues_(at::Tensor& standard_residues,
                                  const at::Tensor& rns_params);

/// Adds $q_j$ to centered residues in place.
///
/// The result of $x + q_j$ is a nonnegative standard residue in the lazy range
/// for centered input $x$.
FHELIUM_API void shift_residues_positive_(at::Tensor& centered_residues,
                                          const at::Tensor& rns_params);

/// Lifts centered integer coefficients to standard residues for every limb.
///
/// `centered_coefficients` has shape `[*batch, coefficient]`, and
/// `twice_modulus` holds $2q_j$ for each output limb. The result has shape
/// `[*batch, limb, coefficient]` with entry $c + q_j$, a standard residue in
/// the lazy range for $|c| < q_j$. The caller orders `twice_modulus` to match
/// the output limbs.
FHELIUM_API at::Tensor lift_centered_coefficients(
    const at::Tensor& centered_coefficients, const at::Tensor& twice_modulus);

/// Output form of `crt_basis_extend`; the result is written to `output`.
FHELIUM_API void crt_basis_extend_out_(
    const at::Tensor& source_residues,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const at::Tensor& target_reciprocals,
    const std::optional<at::Tensor>& source_normalizers,
    const std::optional<at::Tensor>& source_reciprocals,
    at::Tensor& output);

/// Extends standard residues to a target basis by a supplied CRT lift matrix.
///
/// `source_residues` has shape `[*batch, source_limb, coefficient]` with
/// standard residues $a_i$ in $[0, 2q_i)$ for the source primes described by
/// `source_parameters`. Let $D = \prod_i q_i$, $D_i = D/q_i$, and $u_i =
/// D_i^{-1} \bmod q_i$. Without normalizers, `matrix[i][t]` $= D_i u_i R \bmod
/// p_t$ and the result is $\sum_i a_i D_i u_i R \bmod p_t$. With
/// `source_normalizers` $u_i$ and `source_reciprocals`, each source row is
/// first replaced by $a_i u_i \bmod q_i$, and `matrix[i][t]` $= D_i R \bmod
/// p_t$. The output has shape `[*batch, target_limb, coefficient]` with reduced
/// Montgomery residues for the target primes described by `target_parameters`.
/// Neither lift selects the least nonnegative representative modulo $D$.
///
/// `target_reciprocals` and `source_reciprocals` are int64 vectors of $\lfloor
/// 2^{64}/q \rfloor$ for their primes. CUDA int32 execution requires primes
/// below $2^{30}$.
FHELIUM_API at::Tensor crt_basis_extend(
    const at::Tensor& source_residues,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const at::Tensor& target_reciprocals,
    const std::optional<at::Tensor>& source_normalizers = std::nullopt,
    const std::optional<at::Tensor>& source_reciprocals = std::nullopt);

/// Extends source residues to a target basis through mixed-radix digits in one
/// fused pass.
///
/// Computes the result of `mixed_radix_decompose` followed by
/// `mixed_radix_basis_extend_to_montgomery` without writing the digits to
/// global memory. `decomposition_tables` holds, in order, the normalizers, the
/// propagation coefficients, and the source rows' `modulus_lo`, `modulus_hi`,
/// `neg_inv_modulus_lo`, and `neg_inv_modulus_hi`. `extension` holds the
/// basis-extension coefficients and `parameters` the target `rns_params`. The
/// output has shape `[*batch, target_limb, coefficient]`.
FHELIUM_API at::Tensor mixed_radix_modup_tiled(
    const at::Tensor& source_residues,
    at::TensorList decomposition_tables,
    const at::Tensor& extension,
    const at::Tensor& parameters);

/// Output form of `mixed_radix_modup_tiled`; the result is written to `out`.
FHELIUM_API void mixed_radix_modup_tiled_out_(
    const at::Tensor& source_residues,
    at::TensorList decomposition_tables,
    const at::Tensor& extension,
    const at::Tensor& parameters,
    at::Tensor& out);

/// Decomposes residues into mixed-radix digits by Garner's algorithm.
///
/// For source primes $q_0, \ldots, q_{L-1}$ and $x$ the CRT value of the
/// residues of one coefficient, the result holds digits $d_0, \ldots, d_{L-1}$
/// with $x = d_0 + d_1 q_0 + d_2 q_0 q_1 + \cdots$, $d_0 = x \bmod q_0$, and
/// $d_r$ stored in limb row $r$. `mixed_radix_normalizers[r]` is $(q_0 \cdots
/// q_r)^{-1} R \bmod q_{r+1}$, and `mixed_radix_propagation_coefficients[r][c]`
/// is $(q_0 \cdots q_r) R \bmod q_c$ for $c \ge r+2$ and zero otherwise. The
/// four `modulus_*` vectors are the corresponding `rns_params` rows of the
/// source primes. The output has the shape of `source_residues`.
FHELIUM_API at::Tensor mixed_radix_decompose(
    const at::Tensor& source_residues,
    const at::Tensor& mixed_radix_normalizers,
    const at::Tensor& mixed_radix_propagation_coefficients,
    const at::Tensor& modulus_lo,
    const at::Tensor& modulus_hi,
    const at::Tensor& neg_inv_modulus_lo,
    const at::Tensor& neg_inv_modulus_hi);

/// Evaluates mixed-radix digits modulo target primes in Montgomery form.
///
/// `mixed_radix_components` holds digits $d_r$ in limb rows, as returned by
/// `mixed_radix_decompose`. `basis_extension_coefficients[r-1][t]` is $(q_0
/// \cdots q_{r-1}) R^2 \bmod p_t$ for digit $r \ge 1$. For each of the
/// `destination_row_count` target primes $p_t$ described by `rns_params`, the
/// function computes $\left(d_0 + \sum_{r \ge 1} d_r \prod_{s<r} q_s\right) R
/// \bmod p_t$ . The output has shape `[*batch, destination_row_count,
/// coefficient]`.
FHELIUM_API at::Tensor mixed_radix_basis_extend_to_montgomery(
    const at::Tensor& mixed_radix_components,
    const at::Tensor& basis_extension_coefficients,
    const at::Tensor& rns_params,
    int64_t destination_row_count);

/// Output form of `mixed_radix_decompose`; the digits are written to `out`.
FHELIUM_API void mixed_radix_decompose_out_(
    const at::Tensor& source_residues,
    const at::Tensor& mixed_radix_normalizers,
    const at::Tensor& mixed_radix_propagation_coefficients,
    const at::Tensor& modulus_lo,
    const at::Tensor& modulus_hi,
    const at::Tensor& neg_inv_modulus_lo,
    const at::Tensor& neg_inv_modulus_hi,
    at::Tensor& out);

/// Output form of `mixed_radix_basis_extend_to_montgomery`; the result is
/// written to `out`.
FHELIUM_API void mixed_radix_basis_extend_to_montgomery_out_(
    const at::Tensor& mixed_radix_components,
    const at::Tensor& basis_extension_coefficients,
    const at::Tensor& rns_params,
    int64_t destination_row_count,
    at::Tensor& out);

}  // namespace fhelium::rns
