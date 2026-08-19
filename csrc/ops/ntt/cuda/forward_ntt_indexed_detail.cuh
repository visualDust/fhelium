#pragma once

// Private indexed radix-2 NTT kernels and launch helpers.

template <typename scalar_t>
__global__ void forward_ntt_indexed_stage_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<int, 2> even_acc,
    const CudaTensorAccessor32<int, 2> odd_acc,
    const CudaTensorAccessor32<scalar_t, 3> forward_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int stage) {
  // Thread Indexing
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int j = blockIdx.y * kCudaBlockSize + threadIdx.x;

  // Montgomery constants.
  const scalar_t twice_modulus = params_acc[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params_acc[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params_acc[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi =
      params_acc[RNS_PARAM_NEG_INV_MODULUS_HI][row];

  // Butterfly.
  const int even_j = even_acc[stage][j];
  const int odd_j = odd_acc[stage][j];

  const scalar_t U = a_acc[batch][row][even_j];
  const scalar_t S = forward_twiddles_acc[row][stage][j];
  const scalar_t O = a_acc[batch][row][odd_j];
  const scalar_t V = montgomery_mul(
      S, O, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);

  // Store back.
  const scalar_t UplusV = U + V;
  const scalar_t UminusV = U + twice_modulus - V;

  a_acc[batch][row][even_j] =
      (UplusV < twice_modulus) ? UplusV : UplusV - twice_modulus;
  a_acc[batch][row][odd_j] =
      (UminusV < twice_modulus) ? UminusV : UminusV - twice_modulus;
}
