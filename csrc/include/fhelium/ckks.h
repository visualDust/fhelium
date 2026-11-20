#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/string_view.h>
#include <fhelium/export.h>
#include <fhelium/rns_parameters.h>

#include <cstdint>
#include <optional>

//! CKKS ciphertext arithmetic on RNS residue Tensors: ciphertext products,
//! plaintext addition, rescaling, Galois automorphisms, and key-switching
//! steps.
//!
//! A residue operand has shape `[*batch, limb, index]`, where limb $j$ holds
//! residues modulo the prime $q_j$ described by column $j$ of `rns_params`
//! (`fhelium/rns_parameters.h`) and the final axis holds $N$ polynomial
//! coefficients or NTT evaluations. Ciphertext component Tensors place the
//! component axis first. Implementations view `*batch` as one axis without
//! copying; the only broadcast is a plaintext batch extent of one. All Tensors
//! of one call share one integral dtype and one device. Each function states
//! the domain (coefficient or NTT) and representation (standard or Montgomery)
//! it consumes and produces; $R$ denotes the Montgomery radix.
//!
//! Functions without a trailing underscore allocate their result. A trailing
//! underscore marks a function that mutates its first Tensor argument, or both
//! accumulators, in place. Tables and parameters are read-only.

namespace fhelium::ckks {

/// Multiplies two two-component ciphertexts without relinearization.
///
/// `lhs_components` and `rhs_components` have shape `[2, *batch, limb, N]` with
/// NTT-domain Montgomery residues. The result has shape `[3, *batch, limb, N]`
/// with $d_0 = a_0 b_0$, $d_1 = a_0 b_1 + a_1 b_0$, and $d_2 = a_1 b_1$, each
/// product computed as a Montgomery product modulo $q_j$. The result stays in
/// the NTT domain and Montgomery form, in the lazy range $[0, 2q_j)$.
FHELIUM_API at::Tensor multiply_two_component_ntt_montgomery(
    const at::Tensor& lhs_components,
    const at::Tensor& rhs_components,
    const at::Tensor& rns_params);

/// Adds a prepared plaintext to one ciphertext component.
///
/// `ciphertext_component` holds coefficient-domain standard residues $c$, and
/// `prepared_plaintext` holds coefficient-domain Montgomery residues $p$ of the
/// same shape or with a batch extent of one. The result is $c + pR^{-1} \bmod
/// q_j$, a coefficient-domain standard residue in $[0, q_j)$.
FHELIUM_API at::Tensor add_prepared_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& prepared_plaintext,
    const at::Tensor& rns_params);

/// In-place form of `add_prepared_plaintext_component`; the sum replaces
/// `ciphertext_component`.
FHELIUM_API void add_prepared_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& prepared_plaintext,
    const at::Tensor& rns_params);

/// Adds a plaintext whose coefficients repeat cyclically.
///
/// `compressed_plaintext` has final extent $U$, a power of two smaller than
/// $N$. Coefficient $k$ of the ciphertext component receives entry $k \bmod U$
/// of the compressed plaintext; otherwise the function computes
/// `add_prepared_plaintext_component`.
FHELIUM_API at::Tensor add_cyclic_compressed_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params);

/// In-place form of `add_cyclic_compressed_plaintext_component`.
FHELIUM_API void add_cyclic_compressed_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params);

/// Adds a plaintext whose coefficients repeat in contiguous runs.
///
/// `compressed_plaintext` has final extent $U$, a power of two smaller than
/// $N$. Coefficient $k$ of the ciphertext component receives entry $\lfloor
/// kU/N \rfloor$ of the compressed plaintext; otherwise the function computes
/// `add_prepared_plaintext_component`.
FHELIUM_API at::Tensor add_contiguous_compressed_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params);

/// In-place form of `add_contiguous_compressed_plaintext_component`.
FHELIUM_API void add_contiguous_compressed_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params);

/// Adds a plaintext given on a coefficient stride, with one value per limb at
/// the remaining coefficients.
///
/// `strided_plaintext` has final extent $U$, a power of two smaller than $N$,
/// and $r = N/U$. `implicit_plaintext` holds one Montgomery residue per
/// plaintext batch element and limb, as a contiguous Tensor of shape
/// `[plaintext_batch, limb]` or its flattened vector. Coefficient $k$ receives
/// entry $k/r$ of `strided_plaintext` when $r$ divides $k$ and the implicit
/// value of its batch element and limb otherwise; the addition is that of
/// `add_prepared_plaintext_component`.
FHELIUM_API at::Tensor add_strided_plaintext_component(
    const at::Tensor& ciphertext_component,
    const at::Tensor& strided_plaintext,
    const at::Tensor& implicit_plaintext,
    const at::Tensor& rns_params);

/// In-place form of `add_strided_plaintext_component`.
FHELIUM_API void add_strided_plaintext_component_(
    at::Tensor& ciphertext_component,
    const at::Tensor& strided_plaintext,
    const at::Tensor& implicit_plaintext,
    const at::Tensor& rns_params);

/// Divides by a dropped prime and rounds to the nearest integer.
///
/// `remaining_residues` holds coefficient-domain standard residues $x_j$ of an
/// integer $X$ for the remaining primes $q_j$, and `dropped_residue`, of shape
/// `[*batch, N]`, holds $d = X \bmod q_\ell$ for the dropped prime $q_\ell$.
/// `drop_prime_inverse_mont` is the per-limb vector $q_\ell^{-1} R \bmod q_j$,
/// and `half_drop_prime` is $\lfloor q_\ell / 2 \rfloor$. The result is $(x_j -
/// d)\, q_\ell^{-1} + [d > \lfloor q_\ell/2 \rfloor] \bmod q_j$, the residue of
/// $X / q_\ell$ rounded to the nearest integer, as a standard residue in $[0,
/// q_j)$.
FHELIUM_API at::Tensor rescale_drop_leading_prime_nearest(
    const at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params,
    int64_t half_drop_prime);

/// In-place form of `rescale_drop_leading_prime_nearest`; the quotient replaces
/// `remaining_residues`.
FHELIUM_API void rescale_drop_leading_prime_nearest_(
    at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params,
    int64_t half_drop_prime);

/// Divides by a dropped prime and rounds down.
///
/// Computes $(x_j - d)\, q_\ell^{-1} \bmod q_j$, the residue of $\lfloor X /
/// q_\ell \rfloor$, with the operands of `rescale_drop_leading_prime_nearest`.
FHELIUM_API at::Tensor rescale_drop_leading_prime_truncate(
    const at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params);

/// In-place form of `rescale_drop_leading_prime_truncate`; the quotient
/// replaces `remaining_residues`.
FHELIUM_API void rescale_drop_leading_prime_truncate_(
    at::Tensor& remaining_residues,
    const at::Tensor& drop_prime_inverse_mont,
    const at::Tensor& dropped_residue,
    const at::Tensor& rns_params);

/// Applies a Galois automorphism $\sigma_g: X \mapsto X^g$ to
/// coefficient-domain residues.
///
/// `source_indices` (int32) and `source_sign` (int8, $\pm 1$) are vectors of
/// length $N$ that encode $\sigma_g$ on the negacyclic ring: output coefficient
/// $k$ is $s_k \cdot x_{i_k} \bmod q_j$ with $i_k$ = `source_indices[k]` and
/// $s_k$ = `source_sign[k]`. `twice_modulus` holds $2q_j$ per limb. Inputs in
/// $[0, q_j)$ produce standard residues in $[0, q_j)$. The result is newly
/// allocated with the shape of `residues`.
FHELIUM_API at::Tensor apply_coefficient_galois_automorphism(
    const at::Tensor& residues,
    const at::Tensor& source_indices,
    const at::Tensor& source_sign,
    const at::Tensor& twice_modulus);

/// Applies a Galois automorphism to NTT-domain residues by a permutation.
///
/// Output evaluation $k$ is input evaluation `source_indices[k]` for the int32
/// permutation vector `source_indices` of length $N$. Representation and range
/// are preserved.
FHELIUM_API at::Tensor apply_ntt_galois_automorphism(
    const at::Tensor& residues_ntt, const at::Tensor& source_indices);

/// Divides a QP-basis value by $P$ and returns its Q-basis residues.
///
/// `q_residues` (`[*batch, Q, N]`) and `p_residues` (`[*batch, P, N]`) hold
/// coefficient-domain standard residues in $[0, q_i)$ and $[0, p_j)$ of one
/// integer $X$, and the columns of `rns_params` describe the Q primes followed
/// by the P primes. With $P = \prod_j p_j$, the result is $\lfloor X / P \rceil
/// \bmod q_i$ for `rounding = "nearest"` and $\lfloor X / P \rfloor \bmod q_i$
/// for `rounding = "floor"`, as newly allocated coefficient-domain standard
/// residues in $[0, q_i)$. The division drops the P primes one at a time from
/// the last. Row $j$ of `moddown_p_drop_inverses_montgomery` holds $p^{-1} R
/// \bmod q$ for the $j$-th prime dropped and every row before it, with at least
/// $Q + P - 1$ columns. Nearest rounding requires `moddown_half`, the residues
/// of $(P-1)/2$ for every QP row.
FHELIUM_API at::Tensor keyswitch_moddown_qp_to_q(
    const at::Tensor& q_residues,
    const at::Tensor& p_residues,
    const at::Tensor& moddown_p_drop_inverses_montgomery,
    const at::Tensor& rns_params,
    const std::optional<at::Tensor>& moddown_half = std::nullopt,
    c10::string_view rounding = "nearest");

/// Accumulates the product of one extended digit with one key-switching-key
/// digit.
///
/// `extended_digit_ntt_qp` and both accumulators have shape `[*batch, rows, N]`
/// with NTT-domain Montgomery residues. `key_switch_key_digit` has shape `[2,
/// key_rows, N]`, and digit row $r$ uses key row `key_digit_row_start + r`. For
/// each component $c \in \{0, 1\}$, the function computes $\mathrm{acc}_c
/// \mathrel{+}= d_{\pi(k)} \cdot k_c\, R^{-1} \bmod q_r$ in the lazy range,
/// where $\pi$ is the int32 permutation `source_indices` when supplied and the
/// identity otherwise. Supplying `source_indices` applies an NTT-domain Galois
/// automorphism to the digit within the same pass.
FHELIUM_API void keyswitch_accumulate_digit_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    const at::Tensor& extended_digit_ntt_qp,
    const at::Tensor& key_switch_key_digit,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    const std::optional<at::Tensor>& source_indices = std::nullopt);

/// Accumulates the products of several extended digits with their
/// key-switching-key digits.
///
/// `extended_digits_ntt_qp` lists between 1 and 16 digits, and `key_switch_key`
/// has shape `[digit, 2, key_rows, N]`. The result equals calling
/// `keyswitch_accumulate_digit_products_` for every digit $t$ with key digit
/// `key_switch_key[t]`, accumulated in one pass.
FHELIUM_API void keyswitch_accumulate_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    at::TensorList extended_digits_ntt_qp,
    const at::Tensor& key_switch_key,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    const std::optional<at::Tensor>& source_indices = std::nullopt);

}  // namespace fhelium::ckks
