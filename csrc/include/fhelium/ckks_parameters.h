#pragma once

//! CKKS parameter presets, the packaged prime catalog, and the security
//! budget table.
//!
//! A parameter set defines CKKS over $\mathbb{Z}[X]/(X^N + 1)$ with
//! $N = 2^{\mathtt{log\_n}}$, an ordered Q prime list, the active Q basis and
//! default scale of every depth, and the catalog of primes reserved for
//! key-switch auxiliary moduli. The basis $B_d$ of depth $d$ is an increasing
//! list of Q prime ids. A rescale
//! from depth $d$ to $d + 1$ divides by the dropped primes $B_d \setminus
//! B_{d+1}$ and restores the primes $B_{d+1} \setminus B_d$. Every Q and P
//! prime is odd and satisfies $q \equiv 1 \pmod{2N}$.

#include <c10/util/ArrayRef.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fhelium/export.h>

namespace fhelium::ckks {

/// A resolved CKKS parameter set.
struct CkksParameters {
  /// $\log_2 N$.
  int64_t log_n = 0;
  /// Ordered Q primes.
  std::vector<int64_t> q_moduli;
  /// Increasing Q prime ids of each depth's basis.
  std::vector<std::vector<int64_t>> q_bases;
  /// Default scale of each depth.
  std::vector<double> scales;
  /// Primes reserved for key-switch auxiliary moduli.
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
/// Every preset uses NTT primes below $2^{30}$: main primes near $2^{30}$
/// and terminal primes near $2^{25}$. Adjacent bases differ by a rescale
/// ratio $D/R$ near the scale, and the maximum-depth basis holds about
/// $s + 10$ bits for scale $2^s$. `scripts/generate_ckks_presets.py`
/// generates the table.
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

/// Prime rows of a hybrid key-switch layout: active Q rows, auxiliary P
/// rows, and the Q rows of each digit.
struct KeyLayout {
  std::vector<int64_t> q_prime_ids;
  std::vector<int64_t> p_prime_ids;
  std::vector<std::vector<int64_t>> digit_prime_ids;
};

/// Hybrid key-switch layout of the active Q rows `q_prime_ids` of a chain whose
/// configured primes are `moduli` (Q and P, indexed by prime id).
///
/// The auxiliary primes P are the largest configured primes outside the
/// active rows, ties to the lower id. With q active rows, p auxiliary primes,
/// and k digits, one key switch lifts and multiplies k(q + p) rows and divides
/// q + p rows by P; the layout minimizes the row work (k + 1)(q + p), then the
/// key rows k(q + p), then p, where k is the number of greedy runs below P.
/// The digits are k consecutive runs of the active rows that minimize the
/// largest run product.
FHELIUM_API KeyLayout plan_key_layout(c10::ArrayRef<int64_t> moduli,
                                      c10::ArrayRef<int64_t> q_prime_ids);

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
