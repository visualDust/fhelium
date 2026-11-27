#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Tensor.h>
#include <c10/util/string_view.h>
#include <fhelium/export.h>
#include <fhelium/rns_parameters.h>

#include <cstdint>
#include <optional>

//! CKKS ciphertext arithmetic on RNS residue Tensors: ciphertext products,
//! plaintext addition, Galois automorphisms, and key-switching steps.
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

/// Out form of `multiply_two_component_ntt_montgomery`. `out` has shape
/// `[3, *batch, limb, N]` with the operand dtype and device and is disjoint
/// from both operands.
FHELIUM_API void multiply_two_component_ntt_montgomery_out(
    const at::Tensor& lhs_components,
    const at::Tensor& rhs_components,
    const at::Tensor& rns_params,
    at::Tensor& out);

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

/// Out form of `add_prepared_plaintext_component`. `out` has the shape, dtype
/// and device of `ciphertext_component`, is that component or disjoint from
/// it, and is disjoint from the plaintext operands; the in-place form passes
/// the component as `out`. The other plaintext additions follow the same
/// output requirements.
FHELIUM_API void add_prepared_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& prepared_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out);

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

/// Out form of `add_cyclic_compressed_plaintext_component`.
FHELIUM_API void add_cyclic_compressed_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out);

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

/// Out form of `add_contiguous_compressed_plaintext_component`.
FHELIUM_API void add_contiguous_compressed_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& compressed_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out);

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

/// Out form of `add_strided_plaintext_component`.
FHELIUM_API void add_strided_plaintext_component_out(
    const at::Tensor& ciphertext_component,
    const at::Tensor& strided_plaintext,
    const at::Tensor& implicit_plaintext,
    const at::Tensor& rns_params,
    at::Tensor& out);

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

/// Output form of `apply_coefficient_galois_automorphism`; `out` has the
/// shape, dtype and device of `residues` and does not overlap it.
FHELIUM_API void apply_coefficient_galois_automorphism_out(
    const at::Tensor& residues,
    const at::Tensor& source_indices,
    const at::Tensor& source_sign,
    const at::Tensor& twice_modulus,
    at::Tensor& out);

/// Applies a Galois automorphism to NTT-domain residues by a permutation.
///
/// Output evaluation $k$ is input evaluation `source_indices[k]` for the int32
/// permutation vector `source_indices` of length $N$. Representation and range
/// are preserved.
FHELIUM_API at::Tensor apply_ntt_galois_automorphism(
    const at::Tensor& residues_ntt, const at::Tensor& source_indices);

/// Output form of `apply_ntt_galois_automorphism`; `out` has the shape, dtype
/// and device of `residues_ntt` and does not overlap it.
FHELIUM_API void apply_ntt_galois_automorphism_out(
    const at::Tensor& residues_ntt,
    const at::Tensor& source_indices,
    at::Tensor& out);

/// Adds a Galois automorphism of NTT-domain residues to `target`, in place.
///
/// With $\sigma$ the permutation of `apply_ntt_galois_automorphism`, each
/// target evaluation $t_k$ becomes $t_k + r_{\mathtt{source\_indices}[k]}$
/// reduced to $[0, 2q_j)$ for lazy inputs in $[0, 2q_j)$. `target` and
/// `residues_ntt` share shape, dtype and device and do not alias;
/// `rns_params` describes their rows.
FHELIUM_API void add_ntt_galois_automorphism_(at::Tensor& target,
                                              const at::Tensor& residues_ntt,
                                              const at::Tensor& source_indices,
                                              const at::Tensor& rns_params);

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

/// Output form of `keyswitch_moddown_qp_to_q`; `out` has the shape, dtype and
/// device of `q_residues` and overlaps neither residue input. GPU backends
/// write the mixed-radix digits of the P rows to `p_scratch`, which has the
/// shape, dtype and device of `p_residues` and may be `p_residues` itself;
/// without it they allocate the digits per call.
FHELIUM_API void keyswitch_moddown_qp_to_q_out(
    const at::Tensor& q_residues,
    const at::Tensor& p_residues,
    const at::Tensor& moddown_p_drop_inverses_montgomery,
    const at::Tensor& rns_params,
    const std::optional<at::Tensor>& moddown_half,
    c10::string_view rounding,
    at::Tensor& out,
    const std::optional<at::Tensor>& p_scratch = std::nullopt);

/// Accumulates the product of one extended digit with one key-switching-key
/// digit.
///
/// `extended_digit_ntt_qp` and both accumulators have shape `[*batch, rows, N]`
/// with NTT-domain Montgomery residues. `key_switch_key_digit` has shape `[2,
/// key_rows, N]`. With $q$ the active Q row count, digit row $r < q$ uses key
/// row `key_digit_row_start + r` and auxiliary row $r \ge q$ uses key row
/// `key_p_row_start + r - q`, where the auxiliary rows are the trailing
/// `key_rows - key_p_row_start` key rows. A negative `key_p_row_start` reads
/// every digit row from `key_digit_row_start + r`. For
/// each component $c \in \{0, 1\}$, the function computes $\mathrm{acc}_c
/// \mathrel{+}= d_{\pi(k)} \cdot k_c\, R^{-1} \bmod q_r$ in the lazy range,
/// where $\pi$ is the int32 permutation `source_indices` when supplied and the
/// identity otherwise. Supplying `source_indices` applies an NTT-domain Galois
/// automorphism to the digit within the same pass. A seeded key digit has
/// shape `[1, key_rows, N]` holding $k_0$, and `key_seed` is its int64 seed
/// row `[6]`; $k_1$ at key row $j$ is then the value of
/// `fhelium::random::expand_seeded_key` at stored row $j$ for ChaCha
/// `key_rounds`, generated within the product.
FHELIUM_API void keyswitch_accumulate_digit_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    const at::Tensor& extended_digit_ntt_qp,
    const at::Tensor& key_switch_key_digit,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    int64_t key_p_row_start = -1,
    const std::optional<at::Tensor>& source_indices = std::nullopt,
    const std::optional<at::Tensor>& key_seed = std::nullopt,
    int64_t key_rounds = 0);

/// Writes the key-switching products of one set of extended digits for each
/// of several keys.
///
/// `extended_digits_ntt_qp` lists between 1 and 16 digits $d_t$. Entry $r$ of
/// `key_switch_keys` has shape `[digit, 2, key_rows, N]`; all entries share one
/// layout. For each rotation $r$ the kernel overwrites
/// `accumulators0_qp[r]` and `accumulators1_qp[r]` with
/// $\sum_t d_t[\pi_r(i)] \cdot k_{r,t,c}[i]$ for $c = 0, 1$ (Montgomery
/// products, lazy sums in $[0, 2q)$), which equals zero-initialized
/// `keyswitch_accumulate_digit_products_` calls for every digit $t$. An empty
/// `source_indices` uses $\pi_r(i) = i$; otherwise entry $r$ supplies the NTT
/// automorphism gather $\pi_r$. Each launch reads the digits once for up to
/// 16 rotations. Seeded keys have shape `[digit, 1, key_rows, N]`, and entry
/// $r$ of `key_seeds` holds the int64 seed rows `[digit, 6]` of key $r$; their
/// $k_1$ is generated as in `keyswitch_accumulate_digit_products_`. An empty
/// `key_seeds` reads dense keys.
FHELIUM_API void keyswitch_accumulate_products_(
    at::TensorList accumulators0_qp,
    at::TensorList accumulators1_qp,
    at::TensorList extended_digits_ntt_qp,
    at::TensorList key_switch_keys,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    int64_t key_p_row_start = -1,
    at::TensorList source_indices = {},
    at::TensorList key_seeds = {},
    int64_t key_rounds = 0);

}  // namespace fhelium::ckks
