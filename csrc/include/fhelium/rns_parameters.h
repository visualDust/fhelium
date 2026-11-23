#pragma once

//! Row layout of the `rns_params` Tensor accepted by `fhelium::ckks`,
//! `fhelium::rns`, and `fhelium::ntt` functions.
//!
//! `rns_params` is an integral Tensor of shape `[RNS_PARAMETER_COUNT, limb]`
//! with the residue dtype of its operands, on their device. Column $j$
//! describes the prime $q_j$ of operand limb $j$. Every $q_j$ is odd and
//! satisfies $4q_j < R$ for the Montgomery radix $R = 2^w$, where $w = 32$ for
//! int32 residues and $w = 62$ for int64 residues. Rows that split a word use
//! $h = w/2$.

namespace fhelium::rns {

enum RnsParameter : int {
  /// $2q_j$.
  RNS_PARAM_TWICE_MODULUS = 0,

  /// $q_j \bmod 2^h$.
  RNS_PARAM_MODULUS_LO = 1,

  /// $\lfloor q_j / 2^h \rfloor$.
  RNS_PARAM_MODULUS_HI = 2,

  /// $(-q_j^{-1} \bmod R) \bmod 2^h$.
  RNS_PARAM_NEG_INV_MODULUS_LO = 3,

  /// $\lfloor (-q_j^{-1} \bmod R) / 2^h \rfloor$.
  RNS_PARAM_NEG_INV_MODULUS_HI = 4,

  /// $R^2 \bmod q_j$.
  RNS_PARAM_R2 = 5,

  /// $\Delta_0 R^2 \bmod q_j$ for the base scale $\Delta_0$.
  RNS_PARAM_SCALED_R2 = 6,

  /// $N^{-1} R \bmod q_j$ for ring dimension $N$.
  RNS_PARAM_N_INV_MONTGOMERY = 7,

  /// Number of rows.
  RNS_PARAMETER_COUNT = 8,
};

}  // namespace fhelium::rns
