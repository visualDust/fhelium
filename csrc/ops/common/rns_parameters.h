#pragma once

// Row-major engine-owned native parameter tensor layout:
// [parameter, full local QP row]. Passing this tensor keeps
// independent CKKS contexts safe on the same execution device.
enum RnsParameter : int {
  RNS_PARAM_TWICE_MODULUS = 0,
  RNS_PARAM_MODULUS_LO = 1,
  RNS_PARAM_MODULUS_HI = 2,
  RNS_PARAM_NEG_INV_MODULUS_LO = 3,
  RNS_PARAM_NEG_INV_MODULUS_HI = 4,
  RNS_PARAM_R2 = 5,
  RNS_PARAM_SCALED_R2 = 6,
  RNS_PARAM_N_INV_MONTGOMERY = 7,
  RNS_PARAMETER_COUNT = 8,
};
