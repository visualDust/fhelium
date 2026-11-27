#pragma once

//! Construction of the RNS constant tables consumed by `fhelium::rns`,
//! `fhelium::ntt`, and `fhelium::ckks` functions.
//!
//! Every function computes its entries on the host from a list of odd
//! moduli and returns new CPU Tensors; move them to the execution device
//! before use. `dtype` is the residue dtype, int32 or int64, and selects the
//! Montgomery radix $R = 2^w$ with $w = 32$ for int32 and $w = 62$ for int64.
//! An entry described as "in Montgomery form" carries one factor $R$. For a
//! list of factors $F$, $\Pi_F$ denotes their product and
//! $H_F = \lfloor \Pi_F / 2 \rfloor$; products are evaluated modulo each
//! target, so no entry depends on multi-precision arithmetic.

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#include <tuple>

#include <fhelium/export.h>
#include <string>
#include <string_view>
#include <vector>

namespace fhelium::rns {

/// Returns the `rns_params` Tensor of shape `[RNS_PARAMETER_COUNT, limb]`
/// for `moduli`, whose rows follow `fhelium/rns_parameters.h`.
///
/// The ring dimension is $N = 2^{\mathtt{log\_n}}$, and the base scale
/// $\Delta_0$ is `default_scale` rounded to the nearest integer, ties to
/// even.
FHELIUM_API at::Tensor parameter_table(at::IntArrayRef moduli,
                                       int64_t log_n,
                                       double default_scale,
                                       at::ScalarType dtype);

/// Returns $kR \bmod q$ in $[0, q)$ for the integer $k$ written in decimal by
/// `scalar` (an optional leading minus sign and digits) and every
/// modulus $q$ of `moduli`, the row residues of an integer-scalar product in
/// Montgomery form, as a `dtype` Tensor of shape `[moduli]` on `device`.
FHELIUM_API at::Tensor integer_scalar_table(at::IntArrayRef moduli,
                                            std::string_view scalar,
                                            at::ScalarType dtype,
                                            c10::Device device);

/// Returns $\Pi_F^{e} R^{m} \bmod q$ for every target modulus $q$.
///
/// $e$ is `exponent`, either 1 or -1, and $m$ is 1 when `montgomery` is
/// true and 0 otherwise. The result has shape `[target]`.
FHELIUM_API at::Tensor product_residues(at::IntArrayRef factors,
                                        at::IntArrayRef targets,
                                        int64_t exponent,
                                        bool montgomery,
                                        at::ScalarType dtype);

/// Returns $H_F \bmod q$ for every target modulus $q$; the product of
/// `factors` is odd. The result has shape `[target]`.
FHELIUM_API at::Tensor half_product_residues(at::IntArrayRef factors,
                                             at::IntArrayRef targets,
                                             at::ScalarType dtype);

/// Returns the mixed-radix digits $d_i$ of $H_F$ in the radix of `factors`,
/// so that $H_F = \sum_i d_i \prod_{j<i} f_j$ with $0 \le d_i < f_i$.
///
/// The result has shape `[factor]`.
FHELIUM_API at::Tensor half_product_digits(at::IntArrayRef factors,
                                           at::ScalarType dtype);

/// Returns the residues of the rounding offset
/// $H = a H_P + b\,\Pi_P H_D$ for the joint quotient by $\Pi_P \Pi_D$,
/// where $a$ is 1 when `p_nearest` is true and $b$ is 1 when `d_nearest`
/// is true, and 0 otherwise.
///
/// The result has shape `[target]` and holds $H \bmod q$ for every target.
FHELIUM_API at::Tensor quotient_offset_residues(at::IntArrayRef p_moduli,
                                                at::IntArrayRef d_moduli,
                                                bool p_nearest,
                                                bool d_nearest,
                                                at::IntArrayRef targets,
                                                at::ScalarType dtype);

/// Returns the Garner tables for a source basis $q_0, \dots, q_{w-1}$ and
/// an extension into `destination`.
///
/// With prefix products $M_k = \prod_{i \le k} q_i$, the results are, in
/// Montgomery form:
/// - `normalizers` of shape `[w - 1]`: $M_k^{-1} R \bmod q_{k+1}$;
/// - `propagation` of shape `[w - 1, w]`: $M_k R \bmod q_t$ for $t \ge k + 2$
///   and zero otherwise;
/// - `extension` of shape `[w - 1, destination]`:
///   $M_k R^2 \bmod p$ for every destination modulus $p$.
FHELIUM_API std::tuple<at::Tensor, at::Tensor, at::Tensor> mixed_radix_tables(
    at::IntArrayRef source, at::IntArrayRef destination, at::ScalarType dtype);

/// Returns the CRT basis-extension tables from `source` into `target`.
///
/// For $D = \Pi_{\mathtt{source}}$, $D_i = D / q_i$ and
/// $u_i = D_i^{-1} \bmod q_i$, `matrix` of shape `[source, target]` holds
/// $D_i u_i R \bmod p$ for the raw lift and $D_i R \bmod p$ when
/// `normalized` is true. `target_reciprocals` holds $\lfloor 2^{64}/p
/// \rfloor$ as int64. For the normalized lift, `source_normalizers` holds
/// $u_i$, `source_reciprocals` holds $\lfloor 2^{64}/q_i \rfloor$, and
/// `target_offsets` holds $-\sum_i \frac{q_i-1}{2} D_i R \bmod p$, which makes
/// `crt_basis_extend` lift the centered residues $[a_i u_i]_{q_i}$; all three
/// are empty for the raw lift.
FHELIUM_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
crt_basis_extension_tables(at::IntArrayRef source,
                           at::IntArrayRef target,
                           bool normalized,
                           at::ScalarType dtype);

/// Returns the centered CRT tables whose lift from `source` into `target`
/// yields the negated quotient correction of an approximate division.
///
/// With $D = \Pi_{\mathtt{source}}$, $E = \Pi_{\mathtt{divisor}}$ (1 when
/// `divisor` is empty) and $u_i = (D/q_i)^{-1} \bmod q_i$, `matrix` holds
/// $-(q_i E)^{-1} \bmod p$ in standard form and `target_offsets` holds
/// $-\sum_i \frac{q_i-1}{2}\,\mathtt{matrix}_{i,p} \bmod p$; the
/// remaining outputs equal those of the normalized
/// `crt_basis_extension_tables`. For source residues $a_i$ of an integer $x$,
/// `crt_basis_extend` with these tables returns $-(r + kD)/(DE) \bmod p$, where
/// $r$ is the centered remainder of $x$ modulo $D$ and $|k|$ is below half the
/// source count. Adding $x(DE)^{-1} \bmod p$ gives $(x-r-kD)/(DE)$, which
/// approximate ModDown (`moddown_crt_*`, $E=1$) and joint ModDown with rescale
/// (`joint_crt_*`) consume.
FHELIUM_API
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
crt_quotient_tables(at::IntArrayRef source,
                    at::IntArrayRef target,
                    at::IntArrayRef divisor,
                    at::ScalarType dtype);

/// Returns the inverses for removing the last `drop_count` moduli one at a
/// time.
///
/// Row $j$ removes the $j$-th modulus from the end, $p_j$, and holds
/// $p_j^{-1} R \bmod q_i$ for the moduli before it; the remaining columns of
/// the shape `[drop_count, moduli - 1]` result are zero.
FHELIUM_API at::Tensor trailing_quotient_inverses(at::IntArrayRef moduli,
                                                  int64_t drop_count,
                                                  at::ScalarType dtype);

/// Returns the shape `[moduli, moduli]` matrix whose entry $(d, i)$ is
/// $q_d^{-1} R \bmod q_i$ for $i \ne d$ and zero on the diagonal.
FHELIUM_API at::Tensor rescale_inverse_matrix(at::IntArrayRef moduli,
                                              at::ScalarType dtype);

/// Named tables of the ModDown quotient fused with rescaling
/// (`JointRescaleTables`, slots prefixed `joint_`).
///
/// `parameters` is the parameter table of the context primes `moduli`. Q
/// and P list the ordered input rows, D the Q rows that rescaling drops, and
/// `target_prime_ids` the output rows (empty selects Q without D). The
/// result pairs the slot names of the defined fields with their Tensors.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
joint_rescale_table_set(const at::Tensor& parameters,
                        c10::IntArrayRef moduli,
                        int64_t log_n,
                        c10::IntArrayRef q_prime_ids,
                        c10::IntArrayRef p_prime_ids,
                        c10::IntArrayRef drop_prime_ids,
                        c10::IntArrayRef target_prime_ids,
                        std::string_view moddown_rounding,
                        std::string_view rescale_rounding,
                        std::string_view basis_conversion);

/// Named `RescaleTables` of the rational rescale from the source rows to the
/// target rows; `ntt_input` adds $D^{-1} \bmod b'$ for NTT-domain input.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
rescale_table_set(const at::Tensor& parameters,
                  c10::IntArrayRef moduli,
                  int64_t log_n,
                  c10::IntArrayRef source_prime_ids,
                  c10::IntArrayRef target_prime_ids,
                  std::string_view rounding,
                  bool ntt_input);

/// Named `BasisExtensionTables` from the source rows of one parameter table
/// to the target rows of another.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
basis_extension_table_set(const at::Tensor& source_parameters,
                          c10::IntArrayRef source_moduli,
                          const at::Tensor& target_parameters,
                          c10::IntArrayRef target_moduli,
                          int64_t log_n,
                          c10::IntArrayRef source_prime_ids,
                          c10::IntArrayRef target_prime_ids);

/// Named `CrtTables` of the approximate basis conversion from the source rows
/// to the target rows; `normalized` selects the centered quotient form.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
crt_table_set(const at::Tensor& parameters,
              c10::IntArrayRef moduli,
              int64_t log_n,
              c10::IntArrayRef source_prime_ids,
              c10::IntArrayRef target_prime_ids,
              bool normalized);

/// Named `CrtTables` of the centered quotient by the source rows evaluated
/// on the target rows, scaled by the inverse of the divisor rows' product.
FHELIUM_API std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
centered_quotient_table_set(const at::Tensor& parameters,
                            c10::IntArrayRef moduli,
                            int64_t log_n,
                            c10::IntArrayRef source_prime_ids,
                            c10::IntArrayRef target_prime_ids,
                            c10::IntArrayRef divisor_prime_ids);

}  // namespace fhelium::rns
