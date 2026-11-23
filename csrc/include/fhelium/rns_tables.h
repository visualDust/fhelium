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
FHELIUM_API std::tuple<at::Tensor, at::Tensor, at::Tensor>
mixed_radix_tables(at::IntArrayRef source,
                   at::IntArrayRef destination,
                   at::ScalarType dtype);

/// Returns the CRT basis-extension tables from `source` into `target`.
///
/// For $D = \Pi_{\mathtt{source}}$, $D_i = D / q_i$ and
/// $u_i = D_i^{-1} \bmod q_i$, `matrix` of shape `[source, target]` holds
/// $D_i u_i R \bmod p$ for the raw lift and $D_i R \bmod p$ when
/// `normalized` is true. `target_reciprocals` holds $\lfloor 2^{64}/p
/// \rfloor$ as int64. For the normalized lift, `source_normalizers` holds
/// $u_i$ and `source_reciprocals` holds $\lfloor 2^{64}/q_i \rfloor$; both
/// are empty for the raw lift.
FHELIUM_API std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
crt_basis_extension_tables(at::IntArrayRef source,
                           at::IntArrayRef target,
                           bool normalized,
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

}  // namespace fhelium::rns
