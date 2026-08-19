#include "ckks_cuda.h"

#include "../../common/cuda/kernel_support.cuh"
#include "../../common/cuda/montgomery.cuh"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"

// Key-switch tensor requirements. ModDown takes integral coefficient/standard
// canonical q_residues [*batch, Q_limb, coefficient] in [0, q_i) and
// p_residues [*batch, P_limb, coefficient] in [0, p_j). rns_params columns are
// exact concatenated QP prime_ids. moddown_p_drop_inverses_montgomery has shape
// [P_limb, at least Q_limb+P_limb-1] and stores each sequentially dropped P
// prime inverse in Montgomery form. It computes a rounded divide by
// $P=\prod_jp_j$ and returns newly allocated Q-only standard canonical
// [*batch, Q_limb, coefficient]. Public inputs are read-only: P preparation
// mutates only an operator-owned clone.
//
// Accumulation takes NTT/Montgomery extended_digit
// [*batch, active_QP_limb, ntt_index], read-only key_digit
// [key_component=2, level_zero_QP_limb, ntt_index], and two same-shaped active
// accumulators. key_digit_row_start maps local limb j to stable key row
// key_digit_row_start+j. It mutates only the two accumulators by
// $a_{k,i}\leftarrow a_{k,i}+d_i k_{k,i}\bmod q_i$ in lazy [0, 2q_i).

inline void check_keyswitch_cuda_peer(const torch::Tensor& reference,
                                      const torch::Tensor& peer,
                                      const char* operation,
                                      const char* peer_name) {
  TORCH_CHECK(reference.is_cuda(), operation, " requires CUDA tensors");
  TORCH_CHECK(peer.is_cuda(),
              operation,
              " requires ",
              peer_name,
              " to be a CUDA tensor");
  TORCH_CHECK(reference.device() == peer.device(),
              operation,
              " requires all tensors on the same CUDA device; ",
              peer_name,
              " is on ",
              peer.device(),
              " while the primary operand is on ",
              reference.device());
  TORCH_CHECK(reference.scalar_type() == peer.scalar_type(),
              operation,
              " requires all tensors to have the same integral dtype; ",
              peer_name,
              " has ",
              peer.scalar_type(),
              " while the primary operand has ",
              reference.scalar_type());
}

// Prepare an operator-owned P-basis scratch tensor for sequential
// divide-and-round. Each batch item is independent.
template <typename scalar_t>
__global__ void keyswitch_moddown_prepare_p_basis_kernel(
    CudaTensorAccessor32<scalar_t, 3> p_scratch,
    const CudaTensorAccessor32<scalar_t, 2> inverse,
    const CudaTensorAccessor32<scalar_t, 2> params,
    const int q_row_count,
    const int p_row_count) {
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= p_scratch.size(2)) return;
  for (int row = p_row_count - 2; row >= 0; --row) {
    scalar_t value = p_scratch[batch][row][coefficient];
    const int param_row = params.size(1) - p_row_count + row;
    const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][param_row];
    const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][param_row];
    const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][param_row];
    const scalar_t neg_inv_modulus_lo =
        params[RNS_PARAM_NEG_INV_MODULUS_LO][param_row];
    const scalar_t neg_inv_modulus_hi =
        params[RNS_PARAM_NEG_INV_MODULUS_HI][param_row];
    for (int lower = p_row_count - 1; lower > row; --lower) {
      const scalar_t difference = sub_lazy_residues(
          value, p_scratch[batch][lower][coefficient], twice_modulus);
      // Match the CPU multiply_split recurrence by canonicalizing this lazy
      // difference before multiplication. Canonicalize the result as well
      // because later P steps reinterpret the stored residue as an ordinary
      // integer under a different prime.
      value = canonicalize_lazy_residue(
          montgomery_mul(canonicalize_lazy_residue(difference, twice_modulus),
                         inverse[p_row_count - lower - 1][row + q_row_count],
                         modulus_lo,
                         modulus_hi,
                         neg_inv_modulus_lo,
                         neg_inv_modulus_hi),
          twice_modulus);
    }
    p_scratch[batch][row][coefficient] = value;
  }
}

template <typename scalar_t>
__global__ void keyswitch_moddown_qp_to_q_kernel(
    CudaTensorAccessor32<scalar_t, 3> out,
    const CudaTensorAccessor32<scalar_t, 3> q_residues,
    const CudaTensorAccessor32<scalar_t, 3> p_scratch,
    const CudaTensorAccessor32<scalar_t, 2> inverse,
    const CudaTensorAccessor32<scalar_t, 2> params) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= q_residues.size(2)) return;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  scalar_t value = montgomery_mul(q_residues[batch][row][coefficient],
                                  params[RNS_PARAM_R2][row],
                                  modulus_lo,
                                  modulus_hi,
                                  neg_inv_modulus_lo,
                                  neg_inv_modulus_hi);
  const int p_count = p_scratch.size(1);
  for (int p_row = p_count - 1; p_row >= 0; --p_row) {
    const scalar_t p_value_mont =
        montgomery_mul(p_scratch[batch][p_row][coefficient],
                       params[RNS_PARAM_R2][row],
                       modulus_lo,
                       modulus_hi,
                       neg_inv_modulus_lo,
                       neg_inv_modulus_hi);
    value = sub_lazy_residues(value, p_value_mont, twice_modulus);
    // Match CPU multiply_split: a lazy subtraction result must be canonical
    // before entering the split-word Montgomery multiplication.
    value = montgomery_mul(canonicalize_lazy_residue(value, twice_modulus),
                           inverse[p_count - p_row - 1][row],
                           modulus_lo,
                           modulus_hi,
                           neg_inv_modulus_lo,
                           neg_inv_modulus_hi);
  }
  value = montgomery_reduce(
      value, modulus_lo, modulus_hi, neg_inv_modulus_lo, neg_inv_modulus_hi);
  out[batch][row][coefficient] =
      canonicalize_lazy_residue(value, twice_modulus);
}

torch::Tensor keyswitch_moddown_qp_to_q_cuda(
    const torch::Tensor q_residues,
    const torch::Tensor p_residues,
    const torch::Tensor moddown_p_drop_inverses_montgomery,
    const torch::Tensor rns_params) {
  constexpr const char* operation = "keyswitch_moddown_qp_to_q";
  check_keyswitch_cuda_peer(q_residues, p_residues, operation, "p_residues");
  check_keyswitch_cuda_peer(q_residues,
                            moddown_p_drop_inverses_montgomery,
                            operation,
                            "moddown_p_drop_inverses_montgomery");
  check_keyswitch_cuda_peer(q_residues, rns_params, operation, "rns_params");
  auto out = torch::empty_like(q_residues);
  auto p_scratch_public = p_residues.clone();
  const auto q = view_rns_batch_3d(q_residues, "q_residues");
  auto p = view_rns_batch_3d(p_scratch_public, "p_residues");
  auto output = view_rns_batch_3d(out, "out");
  TORCH_CHECK(q.size(0) == p.size(0) && q.size(2) == p.size(2),
              "keyswitch_moddown Q/P batch and coefficient shapes differ");
  TORCH_CHECK(
      rns_params.dim() == 2 && rns_params.size(1) == q.size(1) + p.size(1),
      "keyswitch_moddown RNS parameter row count mismatch");
  TORCH_CHECK(moddown_p_drop_inverses_montgomery.dim() == 2 &&
                  moddown_p_drop_inverses_montgomery.size(0) == p.size(1) &&
                  moddown_p_drop_inverses_montgomery.size(1) >=
                      q.size(1) + p.size(1) - 1,
              "keyswitch_moddown inverse table shape mismatch");
  const int device = q_residues.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 prepare_grid(
      1, (q.size(2) + kCudaBlockSize - 1) / kCudaBlockSize, q.size(0));
  dim3 moddown_grid(
      q.size(1), (q.size(2) + kCudaBlockSize - 1) / kCudaBlockSize, q.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      q_residues.scalar_type(), "keyswitch_moddown_qp_to_q", [&] {
        keyswitch_moddown_prepare_p_basis_kernel<scalar_t>
            <<<prepare_grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(p, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(
                    moddown_p_drop_inverses_montgomery, scalar_t, 2),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2),
                q.size(1),
                p.size(1));
        keyswitch_moddown_qp_to_q_kernel<scalar_t>
            <<<moddown_grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(output, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(q, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(p, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(
                    moddown_p_drop_inverses_montgomery, scalar_t, 2),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2));
      });
  return out;
}

// Multiply one ModUp digit by one unbatched key-switch-key pair and accumulate
// independently for each homogeneous batch item.
template <typename scalar_t>
__global__ void keyswitch_accumulate_digit_products_kernel(
    CudaTensorAccessor32<scalar_t, 3> accumulator0,
    CudaTensorAccessor32<scalar_t, 3> accumulator1,
    const CudaTensorAccessor32<scalar_t, 3> extended_digit,
    const CudaTensorAccessor32<scalar_t, 3> key_digit,
    const CudaTensorAccessor32<scalar_t, 2> params,
    const int key_digit_row_start) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= extended_digit.size(2)) return;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const scalar_t digit = extended_digit[batch][row][coefficient];
  const int key_row = key_digit_row_start + row;
  const scalar_t product0 = montgomery_mul(digit,
                                           key_digit[0][key_row][coefficient],
                                           modulus_lo,
                                           modulus_hi,
                                           neg_inv_modulus_lo,
                                           neg_inv_modulus_hi);
  const scalar_t product1 = montgomery_mul(digit,
                                           key_digit[1][key_row][coefficient],
                                           modulus_lo,
                                           modulus_hi,
                                           neg_inv_modulus_lo,
                                           neg_inv_modulus_hi);
  accumulator0[batch][row][coefficient] = add_lazy_residues(
      accumulator0[batch][row][coefficient], product0, twice_modulus);
  accumulator1[batch][row][coefficient] = add_lazy_residues(
      accumulator1[batch][row][coefficient], product1, twice_modulus);
}

void keyswitch_accumulate_digit_products_inplace_cuda(
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    const torch::Tensor extended_digit_ntt_qp,
    const torch::Tensor key_switch_key_digit,
    const torch::Tensor rns_params,
    const int64_t key_digit_row_start) {
  constexpr const char* operation = "keyswitch_accumulate_digit_products";
  check_keyswitch_cuda_peer(
      extended_digit_ntt_qp, accumulator0_qp, operation, "accumulator0_qp");
  check_keyswitch_cuda_peer(
      extended_digit_ntt_qp, accumulator1_qp, operation, "accumulator1_qp");
  check_keyswitch_cuda_peer(extended_digit_ntt_qp,
                            key_switch_key_digit,
                            operation,
                            "key_switch_key_digit");
  check_keyswitch_cuda_peer(
      extended_digit_ntt_qp, rns_params, operation, "rns_params");
  auto accumulator0 = view_rns_batch_3d(accumulator0_qp, "accumulator0_qp");
  auto accumulator1 = view_rns_batch_3d(accumulator1_qp, "accumulator1_qp");
  const auto digit =
      view_rns_batch_3d(extended_digit_ntt_qp, "extended_digit_ntt_qp");
  check_rns_binary_3d(
      accumulator0, digit, "keyswitch_accumulate_digit_products", false);
  check_rns_binary_3d(
      accumulator1, digit, "keyswitch_accumulate_digit_products", false);
  check_rns_parameter_rows(
      digit, rns_params, "keyswitch_accumulate_digit_products");
  TORCH_CHECK(
      key_switch_key_digit.dim() == 3 && key_switch_key_digit.size(0) == 2,
      "key_switch_key_digit must have shape [2, QP rows, N]");
  TORCH_CHECK(key_digit_row_start >= 0 && key_digit_row_start + digit.size(1) <=
                                              key_switch_key_digit.size(1),
              "active rows exceed key_switch_key_digit row extent");
  TORCH_CHECK(key_switch_key_digit.size(2) == digit.size(2),
              "key digit coefficient extent mismatch");
  const int device = extended_digit_ntt_qp.device().index();
  cudaSetDevice(device);
  auto stream = at::cuda::getCurrentCUDAStream(device);
  dim3 grid(digit.size(1),
            (digit.size(2) + kCudaBlockSize - 1) / kCudaBlockSize,
            digit.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      extended_digit_ntt_qp.scalar_type(),
      "keyswitch_accumulate_digit_products",
      [&] {
        keyswitch_accumulate_digit_products_kernel<scalar_t>
            <<<grid, kCudaBlockSize, 0, stream>>>(
                FHELIUM_CUDA_ACCESSOR32(accumulator0, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(accumulator1, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(digit, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(key_switch_key_digit, scalar_t, 3),
                FHELIUM_CUDA_ACCESSOR32(rns_params, scalar_t, 2),
                static_cast<int>(key_digit_row_start));
      });
}
