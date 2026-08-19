#pragma once

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_parameters.h"

// Representation-changing NTT kernels use the canonical
// [batch, prime, coefficient] operand ABI. Grid x selects the prime row,
// grid y selects the coefficient tile, and grid z selects the batch member.

template <typename scalar_t>
__global__ void ntt_to_montgomery_inplace_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  residues[batch][row][coefficient] =
      montgomery_mul(residues[batch][row][coefficient],
                     params[RNS_PARAM_R2][row],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t>
__global__ void ntt_to_montgomery_out_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  out[batch][row][coefficient] =
      montgomery_mul(residues[batch][row][coefficient],
                     params[RNS_PARAM_R2][row],
                     params[RNS_PARAM_MODULUS_LO][row],
                     params[RNS_PARAM_MODULUS_HI][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                     params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t inverse_ntt_normalize(
    scalar_t value, int row, const CudaTensorAccessor32<scalar_t, 2>& params) {
  return montgomery_mul(value,
                        params[RNS_PARAM_N_INV_MONTGOMERY][row],
                        params[RNS_PARAM_MODULUS_LO][row],
                        params[RNS_PARAM_MODULUS_HI][row],
                        params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                        params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t>
__global__ void inverse_ntt_normalize_montgomery_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  residues[batch][row][coefficient] =
      inverse_ntt_normalize(residues[batch][row][coefficient], row, params);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t inverse_ntt_to_standard_lazy(
    scalar_t value, int row, const CudaTensorAccessor32<scalar_t, 2>& params) {
  value = inverse_ntt_normalize(value, row, params);
  return montgomery_reduce(value,
                           params[RNS_PARAM_MODULUS_LO][row],
                           params[RNS_PARAM_MODULUS_HI][row],
                           params[RNS_PARAM_NEG_INV_MODULUS_LO][row],
                           params[RNS_PARAM_NEG_INV_MODULUS_HI][row]);
}

template <typename scalar_t>
__global__ void inverse_ntt_normalize_to_standard_lazy_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  residues[batch][row][coefficient] = inverse_ntt_to_standard_lazy(
      residues[batch][row][coefficient], row, params);
}

template <typename scalar_t>
__global__ void inverse_ntt_normalize_to_standard_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  scalar_t value = inverse_ntt_to_standard_lazy(
      residues[batch][row][coefficient], row, params);
  residues[batch][row][coefficient] =
      canonicalize_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
}

template <typename scalar_t>
__global__ void inverse_ntt_normalize_to_centered_kernel(
    CudaTensorAccessor32<scalar_t, 3> residues,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= residues.size(2)) return;
  scalar_t value = inverse_ntt_to_standard_lazy(
      residues[batch][row][coefficient], row, params);
  value =
      canonicalize_lazy_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
  residues[batch][row][coefficient] =
      center_residue(value, params[RNS_PARAM_TWICE_MODULUS][row]);
}
