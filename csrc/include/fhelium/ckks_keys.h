#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/export.h>
#include <fhelium/tensor_table.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

//! CKKS key generation: secret, public, key-switching, relinearization, and
//! Galois keys, and the conversion of a secret key to another RNS basis.
//!
//! Every key is returned in the NTT domain with Montgomery residues over the
//! rows of the `parameters` table and the empty-basis NTT schedules
//! (`forward_{i}`, `inverse_{i}`) of `fhelium/ntt_operations.h`. A secret key
//! has shape `[limb, N]`, a public key `[2, limb, N]`, and a key-switching key
//! `[key_digit, 2, key_row, N]`. Random functions read the stream `seed` of
//! `fhelium/random.h` at its current nonce $n$ and sample their components at
//! consecutive nonces $n, n+1, \dots$; the caller then advances the nonce by
//! the documented count. Discrete-Gaussian errors use the table `cdt_table`.

namespace fhelium::ckks {

/// Rows of a hybrid key-switching key.
struct KeySwitchKeyLayout {
  /// NTT policy name (`fhelium/ntt_operations.h`).
  std::string ntt_backend;
  /// First active Q row. The digits cover the Q rows from this row onward.
  int64_t q_row_start = 0;
  /// Row counts of the consecutive digits (`key_digit_sizes`).
  std::vector<int64_t> digit_sizes;
  /// Auxiliary rows of the key, in key-row order.
  std::vector<int64_t> p_rows;
};

/// Samples a ternary secret $s$ of degree `ring_dimension`.
///
/// Without `hamming_weight`, every coefficient is $\pm 1$ with probability
/// $1/3$ each and 0 otherwise; with weight $h$, exactly $h$ coefficients are
/// $\pm 1$ with independent uniform signs. Uses one nonce.
FHELIUM_API at::Tensor create_secret_key(int64_t ring_dimension,
                                         const TensorTable& tables,
                                         const at::Tensor& seed,
                                         std::string_view ntt_backend,
                                         std::optional<int64_t> hamming_weight);

/// Returns the secret key of the integer coefficients `[N]` (int32 or int64).
FHELIUM_API at::Tensor prepare_secret_key(const at::Tensor& coefficients,
                                          const TensorTable& tables,
                                          std::string_view ntt_backend);

/// Returns the public key $(e - a s, a)$ for the leading rows of
/// `secret_key` that match `parameters`.
///
/// The error $e$ is the integer coefficients `error_coefficients` `[N]` or a
/// discrete-Gaussian sample, and $a$ is `uniform` (NTT Montgomery `[limb, N]`)
/// or a uniform sample. Sampled components use one nonce each, in the order
/// $e, a$.
FHELIUM_API at::Tensor create_public_key(
    const at::Tensor& secret_key,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    std::string_view ntt_backend,
    const std::optional<at::Tensor>& uniform,
    const std::optional<at::Tensor>& error_coefficients);

/// Returns the hybrid key that switches $s_{src}$ = `source_secret_key` to
/// $s_{dst}$ = `destination_secret_key`.
///
/// Both secrets cover the rows of `parameters`. Table `source_product` holds
/// $(m \prod_{p \in P} p) R \bmod q_i$ for the Q rows, where $P$ is the
/// auxiliary basis of the key and $m$ a source multiplier. For digit $j$ with
/// Q rows $D_j$, a public key $(k_0, k_1)$ of $s_{dst}$ over all rows receives
/// $m P s_{src}$ on the rows $D_j$ of $k_0$, so that
/// $k_0 + k_1 s_{dst} = m P s_{src} + e_j$ there; the key keeps the active Q
/// rows and `layout.p_rows`. `uniform_by_digit` `[key_digit, limb, N]` and
/// `error_by_digit` `[key_digit, N]` replace the samples of each digit's public
/// key. Uses one nonce per sampled public-key component.
FHELIUM_API at::Tensor create_key_switch_key(
    const at::Tensor& source_secret_key,
    const at::Tensor& destination_secret_key,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    const KeySwitchKeyLayout& layout,
    const std::optional<at::Tensor>& uniform_by_digit,
    const std::optional<at::Tensor>& error_by_digit);

/// Returns the key that switches $s^2$ to $s$ (`create_key_switch_key`).
FHELIUM_API at::Tensor create_relinearization_key(
    const at::Tensor& secret_key,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    const KeySwitchKeyLayout& layout,
    const std::optional<at::Tensor>& uniform_by_digit,
    const std::optional<at::Tensor>& error_by_digit);

/// Returns the key that switches $\sigma_g(s)$ to $s$ for the automorphism
/// $\sigma_g: X \mapsto X^g$ with $g$ = `galois_element`
/// (`create_key_switch_key`). Rotation keys use `rotation_galois_element`;
/// the conjugation key uses $g = 2N - 1$.
FHELIUM_API at::Tensor create_galois_key(
    const at::Tensor& secret_key,
    int64_t galois_element,
    const TensorTable& tables,
    const at::Tensor& seed,
    const at::Tensor& cdt_table,
    const KeySwitchKeyLayout& layout,
    const std::optional<at::Tensor>& uniform_by_digit,
    const std::optional<at::Tensor>& error_by_digit);

/// Re-expresses `secret_key` in another RNS basis.
///
/// The `source` NTT schedules return the secret to coefficients, whose centered
/// representative modulo the source product is extended to the target rows
/// with the tables of `fhelium::rns::centered_basis_extend`, and the `target`
/// NTT schedules transform the result. The extension tables fix the working
/// dtype; the result takes the dtype of the target schedules.
FHELIUM_API at::Tensor convert_secret_key(const at::Tensor& secret_key,
                                          const TensorTable& tables,
                                          std::string_view ntt_backend);

}  // namespace fhelium::ckks
