#pragma once

// Private indexed radix-2 INTT kernels and launch helpers.

template <typename scalar_t>
__global__ void inverse_ntt_indexed_stage_kernel(
    CudaTensorAccessor32<scalar_t, 3> a_acc,
    const CudaTensorAccessor32<int, 2> even_acc,
    const CudaTensorAccessor32<int, 2> odd_acc,
    const CudaTensorAccessor32<scalar_t, 3> inverse_twiddles_acc,
    const CudaTensorAccessor32<scalar_t, 2> params_acc,
    const int level) {
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
  const int even_j = even_acc[level][j];
  const int odd_j = odd_acc[level][j];

  const scalar_t U = a_acc[batch][row][even_j];
  const scalar_t S = inverse_twiddles_acc[row][level][j];
  const scalar_t V = a_acc[batch][row][odd_j];

  const scalar_t UminusV = U + twice_modulus - V;
  const scalar_t O =
      (UminusV < twice_modulus) ? UminusV : UminusV - twice_modulus;

  const scalar_t W = montgomery_mul(
      S, O, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
  a_acc[batch][row][odd_j] = W;

  const scalar_t UplusV = U + V;
  a_acc[batch][row][even_j] =
      (UplusV < twice_modulus) ? UplusV : UplusV - twice_modulus;
}
