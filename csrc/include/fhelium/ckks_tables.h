#pragma once

//! Construction of the CKKS slot-embedding and Galois index tables.
//!
//! The ring is $\mathbb{Z}[X]/(X^N + 1)$ with $N = 2^{\mathtt{log\_n}}$ and
//! $N/2$ complex slots. A rotation by `step` follows
//! `torch.roll(slots, shifts=step)`. The slot generator $\gamma \in \{3, 5\}$
//! orders the slots by the powers of $\gamma$ modulo $2N$. Index tables are
//! computed on the host and returned as new CPU Tensors.

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <tuple>

#include <c10/util/ArrayRef.h>
#include <fhelium/export.h>
#include <string>
#include <string_view>
#include <vector>

namespace fhelium::ckks {

/// Returns the Galois element $g = \gamma^{e} \bmod 2N$ of the automorphism
/// $X \mapsto X^g$ that rotates the slots by `step`, where $e$ is $-$`step`
/// for $\gamma = 5$ and `step` for $\gamma = 3$, reduced modulo $N$.
FHELIUM_API int64_t rotation_galois_element(int64_t log_n,
                                            int64_t step,
                                            int64_t generator);

/// Returns the gather that applies $X \mapsto X^g$ to coefficient-domain
/// residues: destination coefficient $j$ reads source coefficient
/// `indices[j]` and multiplies it by `signs[j]` $\in \{-1, 1\}$. The results
/// are int32 and int8 Tensors of shape `[N]`.
FHELIUM_API std::tuple<at::Tensor, at::Tensor> coefficient_galois_indices(
    int64_t log_n, int64_t galois_element);

/// Returns the int64 slot permutations of the canonical embedding.
///
/// Encoding scatters slot $i$ to FFT position `pre[i]` (shape `[N/2]`), and
/// decoding gathers FFT position `post[k]` into slot position $k$ of the
/// conjugate-extended vector (shape `[N]`).
FHELIUM_API std::tuple<at::Tensor, at::Tensor> slot_permutations(
    int64_t log_n, int64_t generator);

/// Returns the encoded length $U$ of a message that repeats with period
/// `period` slots: $U = \max(2\,\mathtt{period}, 4)$ for $\gamma = 3$ and
/// $\max(2\,\mathtt{period}, 2)$ for $\gamma = 5$. `period` is a power of
/// two dividing $N/2$, and $U < N$.
FHELIUM_API int64_t periodic_extent(int64_t period,
                                    int64_t log_n,
                                    int64_t generator);

/// Returns the int64 slot scatter of shape `[U/2]` for a periodic message of
/// encoded length $U$: position $i$ is $(\sigma^i \bmod 2U - 1)/2$ for
/// $\sigma = 5$ when $\gamma = 5$ and $\sigma = 3^{-1} \bmod 2U$ when
/// $\gamma = 3$.
FHELIUM_API at::Tensor periodic_slot_positions(int64_t extent,
                                               int64_t generator);

/// Returns the complex128 twist $e^{-i\pi k/N}$ (`inverse` false) or
/// $e^{i\pi k/N}$ (`inverse` true) for $0 \le k < N$ on `device`.
///
/// Encoding multiplies the FFT of the conjugate-extended slots by the
/// forward twist; decoding multiplies the coefficients by the inverse twist
/// before the inverse FFT.
FHELIUM_API at::Tensor slot_twist(int64_t length,
                                  bool inverse,
                                  c10::Device device);

/// Named `KeySwitchTables` of one key-switch configuration, without NTT
/// schedules.
///
/// `digits` lists flattened (start, stop, key_digit) triples over the Q rows.
/// With `rescale`, the joint ModDown/rescale tables for the dropped rows and
/// target rows are added. The result pairs the slot names of the defined
/// fields with their Tensors.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
key_switch_table_set(const at::Tensor& parameters,
                     c10::IntArrayRef moduli,
                     int64_t log_n,
                     c10::IntArrayRef q_prime_ids,
                     c10::IntArrayRef p_prime_ids,
                     c10::IntArrayRef digits,
                     std::string basis_conversion,
                     std::string moddown_rounding,
                     bool rescale,
                     c10::IntArrayRef drop_prime_ids,
                     c10::IntArrayRef target_prime_ids,
                     std::string rescale_rounding);

/// Named rotation-sum quotient tables (`sum_*` slots of `KeySwitchTables`);
/// `next_prime_ids` lists the Q rows that remain after the quotient.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
rotation_sum_table_set(const at::Tensor& parameters,
                       c10::IntArrayRef moduli,
                       int64_t log_n,
                       c10::IntArrayRef q_prime_ids,
                       c10::IntArrayRef p_prime_ids,
                       c10::IntArrayRef next_prime_ids,
                       std::string_view rounding);

/// Named `EncapsulationTables` of an encapsulated modulus raise: the low
/// key switch over `source_q_moduli` and `low_p_moduli`, the centered
/// extension to the return basis [guard, Q, P], and the return quotient onto
/// `target_prime_ids`.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
encapsulation_table_set(const at::Tensor& parameters,
                        c10::IntArrayRef moduli,
                        int64_t log_n,
                        std::string_view ntt_backend,
                        double scale,
                        c10::IntArrayRef source_q_moduli,
                        c10::IntArrayRef low_p_moduli,
                        int64_t return_prime,
                        c10::IntArrayRef return_p_moduli,
                        c10::IntArrayRef q_moduli,
                        c10::IntArrayRef target_prime_ids,
                        std::string_view rounding);

/// Named tables of the canonical embedding on `device`: the encoding scatter
/// `pre` and the decoding gather `post` of `slot_permutations`, the forward
/// twist `twister` and the inverse twist `skewer` of `slot_twist` of length
/// $N$.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
embedding_table_set(int64_t log_n, int64_t generator, c10::Device device);

/// Named tables of a periodic plaintext with period `period` slots over the
/// basis rows of `parameters` (moduli `moduli`), on the parameters' device:
/// the slot scatter `pre` and twist `twister` of length $U$ =
/// `periodic_extent`, `twice_moduli` $2q_i$, and the indexed subring NTT
/// schedule `even_indices`, `odd_indices` and Montgomery `twiddles` whose
/// roots are $\psi_N^{N/U}$.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
periodic_table_set(const at::Tensor& parameters,
                   c10::IntArrayRef moduli,
                   int64_t log_n,
                   int64_t period,
                   int64_t generator);

/// Named mixed-radix reconstruction tables of decryption over `prime_ids`.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
reconstruction_table_set(const at::Tensor& parameters,
                         c10::IntArrayRef moduli,
                         int64_t log_n,
                         c10::IntArrayRef prime_ids);

/// `GaloisTables` of $X \mapsto X^g$ for $g$ = `galois_element` on `device`:
/// NTT-domain source indices, coefficient source indices, and coefficient
/// signs.
FHELIUM_API std::vector<at::Tensor> galois_table_set(int64_t log_n,
                                                     int64_t galois_element,
                                                     c10::Device device);

/// Slot names of one table family in positional order.
///
/// `family` is `ntt`, `key_switch`, `rescale`, `basis_extension`,
/// `trailing_quotient`, `encapsulation`, `ring`, `decryption`,
/// `secret_conversion`, or `galois`. The `key_switch` and `encapsulation`
/// families append `digit_count` groups of `digit<i>_` slots.
FHELIUM_API std::vector<std::string> table_slot_names(std::string_view family,
                                                      int64_t digit_count);

}  // namespace fhelium::ckks
