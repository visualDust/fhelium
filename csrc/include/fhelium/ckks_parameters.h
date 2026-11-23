#pragma once

//! CKKS parameter presets, the packaged prime catalog, and the security
//! budget table.
//!
//! A parameter set defines CKKS over $\mathbb{Z}[X]/(X^N + 1)$ with
//! $N = 2^{\mathtt{log\_n}}$, ordered Q depth groups, and the special primes
//! P of hybrid key switching. At depth $d$ the active Q basis is the
//! concatenation of groups $d, d + 1, \dots$; a rescale removes group $d$.
//! Every Q and P prime is odd and satisfies $q \equiv 1 \pmod{2N}$.

#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fhelium/export.h>

namespace fhelium::ckks {

/// A resolved CKKS parameter set.
struct CkksParameters {
  /// $\log_2 N$.
  int64_t log_n = 0;
  /// Scale of values created without an explicit scale.
  double default_scale = 0.0;
  /// Ordered Q depth groups; the last group is the terminal basis.
  std::vector<std::vector<int64_t>> q_depth_groups;
  /// Special primes of the key-switch modulus P.
  std::vector<int64_t> p_moduli;
};

/// Returns the catalog's ordered scaling-prime candidates of width
/// `prime_bits` for ring dimension `ring_dimension`.
///
/// Every candidate is an NTT prime of the given bit width. The catalog
/// covers widths 20 to 50 in steps of 5 and $N = 2^{12}, \dots, 2^{17}$,
/// except width 20 at $N = 2^{17}$.
FHELIUM_API std::vector<int64_t> scaling_primes(int64_t prime_bits,
                                                int64_t ring_dimension);

/// Returns the catalog's ordered special-prime candidates of width
/// `prime_bits` (28 or 60) for ring dimension `ring_dimension`.
FHELIUM_API std::vector<int64_t> special_primes(int64_t prime_bits,
                                                int64_t ring_dimension);

/// Returns the names of the built-in presets.
FHELIUM_API std::vector<std::string> preset_names();

/// Resolves the built-in preset `name` to its primes.
///
/// A preset with scaling width $b$, maximum depth $D$, and $k$ special
/// primes takes the first $D$ scaling candidates of width $b$ as singleton
/// Q groups, the next candidate as the terminal group, and the first $k$
/// special candidates not already in Q as P. The default scale is $2^b$.
/// Special primes are 60 bits for int64 presets and 28 bits for int32
/// presets.
FHELIUM_API CkksParameters preset_parameters(std::string_view name);

/// Checks that `moduli` are distinct odd integers above two with
/// $q \equiv 1 \pmod{2N}$.
FHELIUM_API void validate_moduli(int64_t log_n, c10::ArrayRef<int64_t> moduli);

/// Returns $\lceil \log_2 \prod q \rceil$ for the product of `moduli`.
FHELIUM_API int64_t modulus_bits(c10::ArrayRef<int64_t> moduli);

/// Returns the sizes of the hybrid key-switching digits of `q_moduli`.
///
/// Digits are consecutive groups of `q_moduli`, taken in order, each the
/// longest group whose product is below $P = \prod$ `p_moduli`. A modulus at
/// or above $P$ forms a digit by itself.
FHELIUM_API std::vector<int64_t> key_digit_sizes(
    c10::ArrayRef<int64_t> q_moduli, c10::ArrayRef<int64_t> p_moduli);

/// Returns the maximum bit width of the complete modulus in the built-in
/// security table, or no value when no row matches.
///
/// The table reproduces the limits of *Security Guidelines for Implementing
/// Homomorphic Encryption* (IACR Communications in Cryptology 2025),
/// computed with lattice-estimator under the classical RC.MATZOV cost model.
/// Rows exist for `secret_distribution` `"ternary"` and `"gaussian"`,
/// `target_bits` 128, 192, and 256, $N = 2^{10}, \dots, 2^{17}$ (from
/// $2^{11}$ above 128 bits), and `error_stddev` exactly 3.19.
FHELIUM_API std::optional<int64_t> maximum_modulus_bits(
    int64_t ring_dimension,
    int64_t target_bits,
    std::string_view secret_distribution,
    double error_stddev);

}  // namespace fhelium::ckks
