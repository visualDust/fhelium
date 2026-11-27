#include <torch/torch.h>
#include <algorithm>
#include <type_traits>

#include "../../common/hip/kernel_support.hpp"
#include "../../common/key_rows.h"
#include "../../common/montgomery.hpp"
#include "../../common/rns_batch.h"
#include "../../common/rns_parameters.h"
#include "../../random/random_core.h"
#include "../backends.h"

namespace {

namespace seeded = ::fhelium::random::detail;

// Key-switch tensor requirements. ModDown takes integral coefficient/standard
// standard q_residues [*batch, Q_limb, coefficient] in [0, q_i) and
// p_residues [*batch, P_limb, coefficient] in [0, p_j). rns_params columns are
// concatenated QP prime_ids order. moddown_p_drop_inverses_montgomery has shape
// [P_limb, at least Q_limb+P_limb-1] and stores each sequentially dropped P
// prime inverse in Montgomery form. Nearest adds H=(P-1)/2 on input loads;
// floor uses the original residues. Both compute division by
// $P=\prod_jp_j$ and returns newly allocated Q-only standard
// [*batch, Q_limb, coefficient]. Public inputs are read-only: P preparation
// writes only an operator-owned scratch tensor.
//
// Accumulation takes NTT/Montgomery extended_digit
// [*batch, active_QP_limb, ntt_index], read-only key_digit
// [key_component=2, level_zero_QP_limb, ntt_index], and two same-shaped active
// accumulators. key_digit_row_start maps local limb j to stable key row
// key_digit_row_start+j. It mutates only the two accumulators by
// $a_{k,i}\leftarrow a_{k,i}+d_i k_{k,i}\bmod q_i$ in lazy [0, 2q_i).
// Optional int32 source_indices[N] gather the digit's NTT indices while
// reading; key rows and accumulator indices stay in destination order.

inline void check_keyswitch_hip_peer(const torch::Tensor& reference,
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
template <typename scalar_t, bool Nearest>
__global__ void keyswitch_moddown_prepare_p_basis_kernel(
    HipTensorAccessor32<scalar_t, 3> p_scratch,
    const HipTensorAccessor32<scalar_t, 3> p_residues,
    const HipTensorAccessor32<scalar_t, 2> inverse,
    const HipTensorAccessor32<scalar_t, 2> params,
    const int q_row_count,
    const int p_row_count,
    const scalar_t* half) {
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= p_scratch.size(2)) return;
  for (int row = p_row_count - 1; row >= 0; --row) {
    scalar_t value = p_residues[batch][row][coefficient];
    const int param_row = params.size(1) - p_row_count + row;
    const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][param_row];
    const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][param_row];
    const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][param_row];
    const scalar_t neg_inv_modulus_lo =
        params[RNS_PARAM_NEG_INV_MODULUS_LO][param_row];
    const scalar_t neg_inv_modulus_hi =
        params[RNS_PARAM_NEG_INV_MODULUS_HI][param_row];
    if constexpr (Nearest) {
      value = reduce_lazy_residue(
          add_lazy_residues(value, half[q_row_count + row], twice_modulus),
          twice_modulus);
    }
    const int half_nbits = kMontgomeryRadixBits<scalar_t> / 2;
    const scalar_t modulus = modulus_lo + (modulus_hi << half_nbits);
    for (int lower = p_row_count - 1; lower > row; --lower) {
      // A reverse mixed-radix digit is bounded by its own source prime, not
      // by this later target prime. Reduce it before the lazy subtraction so
      // unequal-width dropped bases retain the same recurrence as CPU.
      const scalar_t lower_digit =
          p_scratch[batch][lower][coefficient] % modulus;
      const scalar_t difference =
          sub_lazy_residues(value, lower_digit, twice_modulus);
      // Match the CPU multiply_split recurrence by reducing this lazy
      // difference before multiplication. Canonicalize the result as well
      // because later P steps reinterpret the stored residue as an ordinary
      // integer under a different prime.
      value = reduce_lazy_residue(
          montgomery_mul(reduce_lazy_residue(difference, twice_modulus),
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

// Each thread produces one coefficient of rows_per_block <= Rows
// consecutive Q rows, so one load of each prepared P digit serves all of them.
// Fewer rows per block keep short Q bases spread over at least four row blocks.
constexpr int kModDownRows = 4;

template <typename scalar_t, bool Nearest, int Rows>
__global__ void keyswitch_moddown_qp_to_q_kernel(
    HipTensorAccessor32<scalar_t, 3> out,
    const HipTensorAccessor32<scalar_t, 3> q_residues,
    const HipTensorAccessor32<scalar_t, 3> p_scratch,
    const HipTensorAccessor32<scalar_t, 2> inverse,
    const HipTensorAccessor32<scalar_t, 2> params,
    const scalar_t* half,
    const int rows_per_block) {
  const int first_row = blockIdx.x * rows_per_block;
  const int row_count = min(rows_per_block, q_residues.size(1) - first_row);
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  const int p_count = p_scratch.size(1);
  extern __shared__ int64_t inverse_storage[];
  scalar_t* prefix_inverse = reinterpret_cast<scalar_t*>(inverse_storage);
  // Prepared P rows are reverse mixed-radix digits. Expanding the sequential
  // recurrence v <- (v - d_j) / p_j gives
  //   out = q / P - sum_j d_j / (p_0 ... p_j) mod q_i.
  // Build the Montgomery prefix inverses of each row once per block instead of
  // converting each digit to Montgomery and dividing at every step.
  if (threadIdx.x < row_count) {
    const int row = first_row + threadIdx.x;
    const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
    const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
    const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
    const scalar_t neg_inv_modulus_lo =
        params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
    const scalar_t neg_inv_modulus_hi =
        params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
    scalar_t* prefix = prefix_inverse + threadIdx.x * p_count;
    scalar_t product = inverse[p_count - 1][row];
    prefix[0] = product;
    for (int j = 1; j < p_count; ++j) {
      product =
          reduce_lazy_residue(montgomery_mul(product,
                                             inverse[p_count - j - 1][row],
                                             modulus_lo,
                                             modulus_hi,
                                             neg_inv_modulus_lo,
                                             neg_inv_modulus_hi),
                              twice_modulus);
      prefix[j] = product;
    }
  }
  __syncthreads();
  if (coefficient >= q_residues.size(2)) return;
  scalar_t value[Rows], twice_modulus[Rows], modulus_lo[Rows], modulus_hi[Rows],
      neg_inv_modulus_lo[Rows], neg_inv_modulus_hi[Rows];
#pragma unroll
  for (int k = 0; k < Rows; ++k) {
    if (k >= row_count) continue;
    const int row = first_row + k;
    twice_modulus[k] = params[RNS_PARAM_TWICE_MODULUS][row];
    modulus_lo[k] = params[RNS_PARAM_MODULUS_LO][row];
    modulus_hi[k] = params[RNS_PARAM_MODULUS_HI][row];
    neg_inv_modulus_lo[k] = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
    neg_inv_modulus_hi[k] = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
    scalar_t source = q_residues[batch][row][coefficient];
    if constexpr (Nearest) {
      source = reduce_lazy_residue(
          add_lazy_residues(source, half[row], twice_modulus[k]),
          twice_modulus[k]);
    }
    value[k] = montgomery_mul(source,
                              prefix_inverse[k * p_count + p_count - 1],
                              modulus_lo[k],
                              modulus_hi[k],
                              neg_inv_modulus_lo[k],
                              neg_inv_modulus_hi[k]);
  }
  for (int p_row = 0; p_row < p_count; ++p_row) {
    // d_j < p_j < R/4, coefficient < q_i: REDC(d_j * coefficient)
    // is in [0, 2q_i), even when p_j is much larger than q_i.
    const scalar_t digit = p_scratch[batch][p_row][coefficient];
#pragma unroll
    for (int k = 0; k < Rows; ++k) {
      if (k >= row_count) continue;
      const scalar_t contribution =
          montgomery_mul(digit,
                         prefix_inverse[k * p_count + p_row],
                         modulus_lo[k],
                         modulus_hi[k],
                         neg_inv_modulus_lo[k],
                         neg_inv_modulus_hi[k]);
      value[k] = sub_lazy_residues(value[k], contribution, twice_modulus[k]);
    }
  }
#pragma unroll
  for (int k = 0; k < Rows; ++k) {
    if (k < row_count) {
      out[batch][first_row + k][coefficient] =
          reduce_lazy_residue(value[k], twice_modulus[k]);
    }
  }
}

template <bool Nearest>
void keyswitch_moddown_qp_to_q_hip_impl(
    const torch::Tensor q_residues,
    const torch::Tensor p_residues,
    const torch::Tensor moddown_p_drop_inverses_montgomery,
    const torch::Tensor rns_params,
    const std::optional<torch::Tensor> half,
    torch::Tensor& out,
    const std::optional<torch::Tensor>& p_scratch) {
  constexpr const char* operation = "keyswitch_moddown_qp_to_q";
  check_keyswitch_hip_peer(q_residues, p_residues, operation, "p_residues");
  check_keyswitch_hip_peer(q_residues,
                           moddown_p_drop_inverses_montgomery,
                           operation,
                           "moddown_p_drop_inverses_montgomery");
  check_keyswitch_hip_peer(q_residues, rns_params, operation, "rns_params");
  if constexpr (Nearest) {
    TORCH_CHECK(half.has_value(), operation, " nearest requires moddown_half");
    check_keyswitch_hip_peer(q_residues, *half, operation, "moddown_half");
    TORCH_CHECK(half->dim() == 1 && half->is_contiguous() &&
                    half->numel() == q_residues.size(-2) + p_residues.size(-2),
                operation,
                " moddown_half must contain one value per QP row");
  }
  auto p_scratch_public =
      p_scratch ? *p_scratch : torch::empty_like(p_residues);
  const auto q = view_rns_batch_3d(q_residues, "q_residues");
  const auto p_source = view_rns_batch_3d(p_residues, "p_residues");
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
  hipSetDevice(device);
  auto stream = c10::hip::getCurrentHIPStream(device);
  dim3 prepare_grid(
      1, (q.size(2) + kHipBlockSize - 1) / kHipBlockSize, q.size(0));
  const int rows_per_block =
      std::max<int>(1, std::min<int>(kModDownRows, q.size(1) / 4));
  dim3 moddown_grid((q.size(1) + rows_per_block - 1) / rows_per_block,
                    (q.size(2) + kHipBlockSize - 1) / kHipBlockSize,
                    q.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      q_residues.scalar_type(), "keyswitch_moddown_qp_to_q", [&] {
        const scalar_t* half_values = nullptr;
        if constexpr (Nearest) half_values = half->data_ptr<scalar_t>();
        keyswitch_moddown_prepare_p_basis_kernel<scalar_t, Nearest>
            <<<prepare_grid, kHipBlockSize, 0, stream>>>(
                FHELIUM_HIP_ACCESSOR32(p, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(p_source, scalar_t, 3),
                FHELIUM_HIP_ACCESSOR32(
                    moddown_p_drop_inverses_montgomery, scalar_t, 2),
                FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2),
                q.size(1),
                p.size(1),
                half_values);
        const auto moddown = [&](auto rows) {
          keyswitch_moddown_qp_to_q_kernel<scalar_t,
                                           Nearest,
                                           decltype(rows)::value>
              <<<moddown_grid,
                 kHipBlockSize,
                 rows_per_block * p.size(1) * sizeof(scalar_t),
                 stream>>>(FHELIUM_HIP_ACCESSOR32(output, scalar_t, 3),
                           FHELIUM_HIP_ACCESSOR32(q, scalar_t, 3),
                           FHELIUM_HIP_ACCESSOR32(p, scalar_t, 3),
                           FHELIUM_HIP_ACCESSOR32(
                               moddown_p_drop_inverses_montgomery, scalar_t, 2),
                           FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2),
                           half_values,
                           rows_per_block);
        };
        if (rows_per_block == 1) {
          moddown(std::integral_constant<int, 1>{});
        } else {
          moddown(std::integral_constant<int, kModDownRows>{});
        }
      });
}

void keyswitch_moddown_qp_to_q_hip(
    const torch::Tensor q,
    const torch::Tensor p,
    const torch::Tensor inverse,
    const torch::Tensor parameters,
    const std::optional<torch::Tensor> half,
    const std::string& rounding,
    torch::Tensor& out,
    const std::optional<torch::Tensor>& p_scratch) {
  TORCH_CHECK(rounding == "nearest" || rounding == "floor",
              "ModDown rounding must be 'nearest' or 'floor'");
  if (rounding == "nearest") {
    keyswitch_moddown_qp_to_q_hip_impl<true>(
        q, p, inverse, parameters, half, out, p_scratch);
    return;
  }
  keyswitch_moddown_qp_to_q_hip_impl<false>(
      q, p, inverse, parameters, half, out, p_scratch);
}

// Multiply one ModUp digit by one unbatched key-switch-key pair and accumulate
// independently for each homogeneous batch item.
template <typename scalar_t, bool Permute>
__global__ void keyswitch_accumulate_digit_products_kernel(
    HipTensorAccessor32<scalar_t, 3> accumulator0,
    HipTensorAccessor32<scalar_t, 3> accumulator1,
    const HipTensorAccessor32<scalar_t, 3> extended_digit,
    const HipTensorAccessor32<scalar_t, 3> key_digit,
    const HipTensorAccessor32<scalar_t, 2> params,
    const int key_digit_row_start,
    const int key_q_rows,
    const int key_p_row_start,
    const int32_t* source_indices,
    const int64_t source_index_stride) {
  const int row = blockIdx.x;
  const int coefficient = blockIdx.y * blockDim.x + threadIdx.x;
  const int batch = blockIdx.z;
  if (coefficient >= extended_digit.size(2)) return;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  int source_coefficient = coefficient;
  if constexpr (Permute) {
    source_coefficient = source_indices[coefficient * source_index_stride];
    CUDA_KERNEL_ASSERT(source_coefficient >= 0 &&
                       source_coefficient < extended_digit.size(2));
  }
  const scalar_t digit = extended_digit[batch][row][source_coefficient];
  const int key_row = row < key_q_rows ? key_digit_row_start + row
                                       : key_p_row_start + (row - key_q_rows);
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

constexpr int kMaximumGroupedKeyDigits = 16;

template <typename scalar_t>
struct KeyswitchDigitPointers {
  const scalar_t* values[kMaximumGroupedKeyDigits];
  int64_t batch_stride[kMaximumGroupedKeyDigits];
  int64_t row_stride[kMaximumGroupedKeyDigits];
  int64_t coefficient_stride[kMaximumGroupedKeyDigits];
};

// Rotations of one hoisted decomposition that share a product launch.
constexpr int kMaximumGroupedRotations = 16;

template <typename scalar_t>
struct KeyswitchRotationPointers {
  scalar_t* accumulator0[kMaximumGroupedRotations];
  scalar_t* accumulator1[kMaximumGroupedRotations];
  const scalar_t* key[kMaximumGroupedRotations];
  const int32_t* source_indices[kMaximumGroupedRotations];
};

// Strides shared by every rotation of a launch, in elements.
struct KeyswitchRotationStrides {
  int64_t accumulator_batch, accumulator_row;
  int64_t key_digit, key_component, key_row, key_coefficient;
  int64_t source_index;
};

// blockIdx.y selects the QP row and blockIdx.x a coefficient tile, so the
// resident blocks cover few rows and the digit rows they gather stay in L2 for
// every rotation of the launch while the rotation keys stream once.
template <typename scalar_t, bool Permute>
__global__ void keyswitch_accumulate_products_kernel(
    const KeyswitchRotationPointers<scalar_t> rotations,
    const int rotation_count,
    const KeyswitchRotationStrides strides,
    const KeyswitchDigitPointers<scalar_t> digits,
    const int digit_count,
    const int coefficients,
    const HipTensorAccessor32<scalar_t, 2> params,
    const int key_digit_row_start,
    const int key_q_rows,
    const int key_p_row_start) {
  const int coefficient = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int batch = blockIdx.z;
  if (coefficient >= coefficients) return;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const int key_row = row < key_q_rows ? key_digit_row_start + row
                                       : key_p_row_start + (row - key_q_rows);
  const int64_t key_offset =
      key_row * strides.key_row + coefficient * strides.key_coefficient;
  const int64_t output = batch * strides.accumulator_batch +
                         row * strides.accumulator_row + coefficient;
  for (int rotation = 0; rotation < rotation_count; ++rotation) {
    int source_coefficient = coefficient;
    if constexpr (Permute) {
      source_coefficient =
          rotations
              .source_indices[rotation][coefficient * strides.source_index];
      CUDA_KERNEL_ASSERT(source_coefficient >= 0 &&
                         source_coefficient < coefficients);
    }
    const scalar_t* key = rotations.key[rotation] + key_offset;
    scalar_t sum0 = 0;
    scalar_t sum1 = 0;
#pragma unroll
    for (int digit_index = 0; digit_index < kMaximumGroupedKeyDigits;
         ++digit_index) {
      if (digit_index >= digit_count) break;
      const scalar_t digit =
          digits
              .values[digit_index][batch * digits.batch_stride[digit_index] +
                                   row * digits.row_stride[digit_index] +
                                   source_coefficient *
                                       digits.coefficient_stride[digit_index]];
      const scalar_t* key_digit = key + digit_index * strides.key_digit;
      sum0 = add_lazy_residues(sum0,
                               montgomery_mul(digit,
                                              key_digit[0],
                                              modulus_lo,
                                              modulus_hi,
                                              neg_inv_modulus_lo,
                                              neg_inv_modulus_hi),
                               twice_modulus);
      sum1 = add_lazy_residues(sum1,
                               montgomery_mul(digit,
                                              key_digit[strides.key_component],
                                              modulus_lo,
                                              modulus_hi,
                                              neg_inv_modulus_lo,
                                              neg_inv_modulus_hi),
                               twice_modulus);
    }
    rotations.accumulator0[rotation][output] = sum0;
    rotations.accumulator1[rotation][output] = sum1;
  }
}

// Threads per block of seeded-key product kernels; their register use leaves
// more resident warps at this granularity.
constexpr int kSeededBlockSize = 128;
// Coefficients of one warp tile: each lane samples one eight-coefficient call
// and the warp then reads the tile with one coefficient per lane per pass.
constexpr int kSeededTile = 32 * seeded::kSeededKeyCoefficientsPerCall;

// Writes the uniform component of coefficients tile .. tile + kSeededTile of
// stored row `key_row` to the warp's shared buffer, coefficient-major.
template <typename scalar_t, int Rounds>
__device__ __forceinline__ void seeded_uniform_tile(
    const int64_t* seed_row,
    int key_row,
    int tile,
    int coefficients,
    uint64_t q,
    int lane,
    std::make_unsigned_t<scalar_t>* buffer) {
  constexpr int kWidth = seeded::kSeededKeyCoefficientsPerCall;
  const int first = tile + lane * kWidth;
  if (first < coefficients) {
    std::make_unsigned_t<scalar_t> uniform[kWidth];
    seeded::seeded_key_residues<Rounds>(seed_row, key_row, first, q, uniform);
#pragma unroll
    for (int j = 0; j < kWidth; ++j) buffer[lane * kWidth + j] = uniform[j];
  }
  __builtin_amdgcn_wave_barrier();
}

// Seeded keys store b only; the uniform component a is generated where it is
// read (`fhelium::random::expand_seeded_key`). Each warp covers one tile.
template <typename scalar_t, bool Permute, int Rounds>
__global__ void __launch_bounds__(kSeededBlockSize)
    keyswitch_accumulate_seeded_digit_products_kernel(
        HipTensorAccessor32<scalar_t, 3> accumulator0,
        HipTensorAccessor32<scalar_t, 3> accumulator1,
        const HipTensorAccessor32<scalar_t, 3> extended_digit,
        const HipTensorAccessor32<scalar_t, 3> key_digit,
        const int64_t* __restrict__ seed_row,
        const HipTensorAccessor32<scalar_t, 2> params,
        const int key_digit_row_start,
        const int key_q_rows,
        const int key_p_row_start,
        const int32_t* source_indices,
        const int64_t source_index_stride) {
  __shared__ std::make_unsigned_t<scalar_t> buffers[kSeededBlockSize / 32]
                                                   [kSeededTile];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int coefficients = extended_digit.size(2);
  const int tile = (blockIdx.y * (kSeededBlockSize / 32) + warp) * kSeededTile;
  if (tile >= coefficients) return;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const int key_row = row < key_q_rows ? key_digit_row_start + row
                                       : key_p_row_start + (row - key_q_rows);
  auto* buffer = buffers[warp];
  seeded_uniform_tile<scalar_t, Rounds>(
      seed_row,
      key_row,
      tile,
      coefficients,
      static_cast<uint64_t>(twice_modulus) >> 1,
      lane,
      buffer);
#pragma unroll
  for (int pass = 0; pass < seeded::kSeededKeyCoefficientsPerCall; ++pass) {
    const int coefficient = tile + pass * 32 + lane;
    if (coefficient >= coefficients) continue;
    int source = coefficient;
    if constexpr (Permute) {
      source = source_indices[coefficient * source_index_stride];
    }
    const scalar_t digit = extended_digit[batch][row][source];
    const scalar_t product0 = montgomery_mul(digit,
                                             key_digit[0][key_row][coefficient],
                                             modulus_lo,
                                             modulus_hi,
                                             neg_inv_modulus_lo,
                                             neg_inv_modulus_hi);
    const scalar_t product1 =
        montgomery_mul(digit,
                       static_cast<scalar_t>(buffer[pass * 32 + lane]),
                       modulus_lo,
                       modulus_hi,
                       neg_inv_modulus_lo,
                       neg_inv_modulus_hi);
    accumulator0[batch][row][coefficient] = add_lazy_residues(
        accumulator0[batch][row][coefficient], product0, twice_modulus);
    accumulator1[batch][row][coefficient] = add_lazy_residues(
        accumulator1[batch][row][coefficient], product1, twice_modulus);
  }
}

// Seed rows [digit, 6] of each rotation key of a grouped seeded launch.
struct KeyswitchRotationSeeds {
  const int64_t* rows[kMaximumGroupedRotations];
};

// Grouped products of seeded rotation keys: each warp covers one tile and
// generates a once per rotation and digit.
template <typename scalar_t, bool Permute, int Rounds>
__global__ void __launch_bounds__(kSeededBlockSize)
    keyswitch_accumulate_seeded_products_kernel(
        const KeyswitchRotationPointers<scalar_t> rotations,
        const KeyswitchRotationSeeds seeds,
        const int rotation_count,
        const KeyswitchRotationStrides strides,
        const KeyswitchDigitPointers<scalar_t> digits,
        const int digit_count,
        const int coefficients,
        const HipTensorAccessor32<scalar_t, 2> params,
        const int key_digit_row_start,
        const int key_q_rows,
        const int key_p_row_start) {
  constexpr int kPasses = seeded::kSeededKeyCoefficientsPerCall;
  __shared__ std::make_unsigned_t<scalar_t> buffers[kSeededBlockSize / 32]
                                                   [kSeededTile];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int row = blockIdx.x;
  const int batch = blockIdx.z;
  const int tile = (blockIdx.y * (kSeededBlockSize / 32) + warp) * kSeededTile;
  if (tile >= coefficients) return;
  const scalar_t twice_modulus = params[RNS_PARAM_TWICE_MODULUS][row];
  const scalar_t modulus_lo = params[RNS_PARAM_MODULUS_LO][row];
  const scalar_t modulus_hi = params[RNS_PARAM_MODULUS_HI][row];
  const scalar_t neg_inv_modulus_lo = params[RNS_PARAM_NEG_INV_MODULUS_LO][row];
  const scalar_t neg_inv_modulus_hi = params[RNS_PARAM_NEG_INV_MODULUS_HI][row];
  const auto q = static_cast<uint64_t>(twice_modulus) >> 1;
  const int key_row = row < key_q_rows ? key_digit_row_start + row
                                       : key_p_row_start + (row - key_q_rows);
  const int key_offset = key_row * strides.key_row;
  const int output =
      batch * strides.accumulator_batch + row * strides.accumulator_row;
  auto* buffer = buffers[warp];
  // Generation dominates each (rotation, digit) step, so the loops stay
  // rolled; the pointer arrays are read in place from parameter space.
#pragma unroll 1
  for (int rotation = 0; rotation < rotation_count; ++rotation) {
    int source[kPasses];
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int coefficient = tile + pass * 32 + lane;
      source[pass] = coefficient;
      if constexpr (Permute) {
        if (coefficient < coefficients) {
          source[pass] =
              rotations
                  .source_indices[rotation][coefficient * strides.source_index];
        }
      }
    }
    const scalar_t* key = rotations.key[rotation] + key_offset;
    scalar_t sum0[kPasses] = {};
    scalar_t sum1[kPasses] = {};
#pragma unroll 1
    for (int digit_index = 0; digit_index < digit_count; ++digit_index) {
      seeded_uniform_tile<scalar_t, Rounds>(
          seeds.rows[rotation] + digit_index * seeded::kSeededKeyRowWords,
          key_row,
          tile,
          coefficients,
          q,
          lane,
          buffer);
      const scalar_t* digit_values = digits.values[digit_index] +
                                     batch * digits.batch_stride[digit_index] +
                                     row * digits.row_stride[digit_index];
      const scalar_t* key_digit = key + digit_index * strides.key_digit;
#pragma unroll
      for (int pass = 0; pass < kPasses; ++pass) {
        const int coefficient = tile + pass * 32 + lane;
        if (coefficient >= coefficients) continue;
        const scalar_t digit =
            digit_values[source[pass] * digits.coefficient_stride[digit_index]];
        const auto a = static_cast<scalar_t>(buffer[pass * 32 + lane]);
        sum0[pass] = add_lazy_residues(
            sum0[pass],
            montgomery_mul(digit,
                           key_digit[coefficient * strides.key_coefficient],
                           modulus_lo,
                           modulus_hi,
                           neg_inv_modulus_lo,
                           neg_inv_modulus_hi),
            twice_modulus);
        sum1[pass] = add_lazy_residues(sum1[pass],
                                       montgomery_mul(digit,
                                                      a,
                                                      modulus_lo,
                                                      modulus_hi,
                                                      neg_inv_modulus_lo,
                                                      neg_inv_modulus_hi),
                                       twice_modulus);
      }
      __builtin_amdgcn_wave_barrier();
    }
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
      const int coefficient = tile + pass * 32 + lane;
      if (coefficient >= coefficients) continue;
      rotations.accumulator0[rotation][output + coefficient] = sum0[pass];
      rotations.accumulator1[rotation][output + coefficient] = sum1[pass];
    }
  }
}

// Checks a seeded key's seed rows [digit, 6] against its digit count.
inline void check_key_seed(const torch::Tensor& reference,
                           const torch::Tensor& seed,
                           int64_t digits,
                           const char* operation) {
  TORCH_CHECK(
      seed.device() == reference.device() &&
          seed.scalar_type() == torch::kLong && seed.is_contiguous() &&
          seed.numel() == digits * seeded::kSeededKeyRowWords,
      operation,
      " requires contiguous int64 seed rows [key_digit, 6] on the key device");
}

void keyswitch_accumulate_digit_products_inplace_hip(
    torch::Tensor accumulator0_qp,
    torch::Tensor accumulator1_qp,
    const torch::Tensor extended_digit_ntt_qp,
    const torch::Tensor key_switch_key_digit,
    const torch::Tensor rns_params,
    const int64_t key_digit_row_start,
    const int64_t key_p_row_start,
    const std::optional<torch::Tensor> source_indices,
    const std::optional<torch::Tensor> key_seed,
    const int64_t key_rounds) {
  constexpr const char* operation = "keyswitch_accumulate_digit_products";
  check_keyswitch_hip_peer(
      extended_digit_ntt_qp, accumulator0_qp, operation, "accumulator0_qp");
  check_keyswitch_hip_peer(
      extended_digit_ntt_qp, accumulator1_qp, operation, "accumulator1_qp");
  check_keyswitch_hip_peer(extended_digit_ntt_qp,
                           key_switch_key_digit,
                           operation,
                           "key_switch_key_digit");
  check_keyswitch_hip_peer(
      extended_digit_ntt_qp, rns_params, operation, "rns_params");
  auto accumulator0 = view_rns_batch_3d(accumulator0_qp, "accumulator0_qp");
  auto accumulator1 = view_rns_batch_3d(accumulator1_qp, "accumulator1_qp");
  const auto digit =
      view_rns_batch_3d(extended_digit_ntt_qp, "extended_digit_ntt_qp");
  const int32_t* permutation = nullptr;
  int64_t permutation_stride = 0;
  if (source_indices) {
    const auto& indices = *source_indices;
    TORCH_CHECK(indices.device() == extended_digit_ntt_qp.device() &&
                    indices.scalar_type() == torch::kInt32 &&
                    indices.dim() == 1 && indices.size(0) == digit.size(2),
                operation,
                " requires int32 source indices matching N");
    permutation = indices.data_ptr<int32_t>();
    permutation_stride = indices.stride(0);
  }
  check_rns_binary_3d(
      accumulator0, digit, "keyswitch_accumulate_digit_products", false);
  check_rns_binary_3d(
      accumulator1, digit, "keyswitch_accumulate_digit_products", false);
  check_rns_parameter_rows(
      digit, rns_params, "keyswitch_accumulate_digit_products");
  TORCH_CHECK(key_switch_key_digit.dim() == 3 &&
                  key_switch_key_digit.size(0) == (key_seed ? 1 : 2),
              "key_switch_key_digit must have shape [2, QP rows, N], or "
              "[1, QP rows, N] with a seed row");
  if (key_seed) {
    check_key_seed(key_switch_key_digit, *key_seed, 1, operation);
    TORCH_CHECK(digit.size(2) % seeded::kSeededKeyCoefficientsPerCall == 0,
                operation,
                " requires a ring dimension divisible by 8");
  }
  const auto key_rows =
      fhelium::detail::key_row_map(digit.size(1),
                                   key_switch_key_digit.size(1),
                                   key_digit_row_start,
                                   key_p_row_start,
                                   operation);
  TORCH_CHECK(key_switch_key_digit.size(2) == digit.size(2),
              "key digit coefficient extent mismatch");
  at::assert_no_internal_overlap(accumulator0);
  at::assert_no_internal_overlap(accumulator1);
  at::assert_no_overlap(accumulator0, accumulator1);
  for (const auto& read_only : {digit, key_switch_key_digit, rns_params}) {
    at::assert_no_overlap(accumulator0, read_only);
    at::assert_no_overlap(accumulator1, read_only);
  }
  if (source_indices) {
    at::assert_no_overlap(accumulator0, *source_indices);
    at::assert_no_overlap(accumulator1, *source_indices);
  }
  if (key_seed) {
    const int device = extended_digit_ntt_qp.device().index();
    hipSetDevice(device);
    auto stream = c10::hip::getCurrentHIPStream(device);
    constexpr int width = seeded::kSeededKeyCoefficientsPerCall;
    dim3 grid(digit.size(1),
              (digit.size(2) / width + kSeededBlockSize - 1) / kSeededBlockSize,
              digit.size(0));
    seeded::dispatch_seeded_key_rounds(key_rounds, [&](auto rounds_constant) {
      constexpr int kRounds = decltype(rounds_constant)::value;
      AT_DISPATCH_INTEGRAL_TYPES(
          extended_digit_ntt_qp.scalar_type(), operation, [&] {
            // Key residues are int32 or int64.
            if constexpr (sizeof(scalar_t) < 4) {
              TORCH_CHECK(
                  false, operation, " requires int32 or int64 seeded keys");
            } else {
              const auto kernel =
                  permutation
                      ? keyswitch_accumulate_seeded_digit_products_kernel<
                            scalar_t,
                            true,
                            kRounds>
                      : keyswitch_accumulate_seeded_digit_products_kernel<
                            scalar_t,
                            false,
                            kRounds>;
              kernel<<<grid, kSeededBlockSize, 0, stream>>>(
                  FHELIUM_HIP_ACCESSOR32(accumulator0, scalar_t, 3),
                  FHELIUM_HIP_ACCESSOR32(accumulator1, scalar_t, 3),
                  FHELIUM_HIP_ACCESSOR32(digit, scalar_t, 3),
                  FHELIUM_HIP_ACCESSOR32(key_switch_key_digit, scalar_t, 3),
                  key_seed->const_data_ptr<int64_t>(),
                  FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2),
                  static_cast<int>(key_digit_row_start),
                  static_cast<int>(key_rows.q_rows),
                  static_cast<int>(key_rows.p_row_start),
                  permutation,
                  permutation_stride);
            }
          });
    });
    return;
  }
  const int device = extended_digit_ntt_qp.device().index();
  hipSetDevice(device);
  auto stream = c10::hip::getCurrentHIPStream(device);
  dim3 grid(digit.size(1),
            (digit.size(2) + kHipBlockSize - 1) / kHipBlockSize,
            digit.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(
      extended_digit_ntt_qp.scalar_type(),
      "keyswitch_accumulate_digit_products",
      [&] {
        const auto kernel =
            permutation
                ? keyswitch_accumulate_digit_products_kernel<scalar_t, true>
                : keyswitch_accumulate_digit_products_kernel<scalar_t, false>;
        kernel<<<grid, kHipBlockSize, 0, stream>>>(
            FHELIUM_HIP_ACCESSOR32(accumulator0, scalar_t, 3),
            FHELIUM_HIP_ACCESSOR32(accumulator1, scalar_t, 3),
            FHELIUM_HIP_ACCESSOR32(digit, scalar_t, 3),
            FHELIUM_HIP_ACCESSOR32(key_switch_key_digit, scalar_t, 3),
            FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2),
            static_cast<int>(key_digit_row_start),
            static_cast<int>(key_rows.q_rows),
            static_cast<int>(key_rows.p_row_start),
            permutation,
            permutation_stride);
      });
}

void keyswitch_accumulate_products_inplace_hip(
    const at::TensorList accumulators0_qp,
    const at::TensorList accumulators1_qp,
    const at::TensorList extended_digits_ntt_qp,
    const at::TensorList key_switch_keys,
    const torch::Tensor rns_params,
    const int64_t key_digit_row_start,
    const int64_t key_p_row_start,
    const at::TensorList source_indices,
    const at::TensorList key_seeds,
    const int64_t key_rounds) {
  constexpr const char* operation = "keyswitch_accumulate_products";
  const bool seeded_keys = !key_seeds.empty();
  TORCH_CHECK(!extended_digits_ntt_qp.empty() &&
                  extended_digits_ntt_qp.size() <= kMaximumGroupedKeyDigits,
              operation,
              " requires between 1 and ",
              kMaximumGroupedKeyDigits,
              " digits");
  const size_t rotation_count = key_switch_keys.size();
  TORCH_CHECK(
      rotation_count > 0 && accumulators0_qp.size() == rotation_count &&
          accumulators1_qp.size() == rotation_count &&
          (source_indices.empty() || source_indices.size() == rotation_count),
      operation,
      " requires one accumulator pair, key and optional index vector per "
      "rotation");
  const auto& prototype_tensor = extended_digits_ntt_qp.front();
  const auto prototype =
      view_rns_batch_3d(prototype_tensor, "extended_digits_ntt_qp[0]");
  check_keyswitch_hip_peer(
      prototype_tensor, rns_params, operation, "rns_params");
  check_rns_parameter_rows(prototype, rns_params, operation);
  for (const auto& digit_tensor : extended_digits_ntt_qp) {
    check_keyswitch_hip_peer(
        prototype_tensor, digit_tensor, operation, "digit");
    check_rns_binary_3d(
        prototype, view_rns_batch_3d(digit_tensor, "digit"), operation, false);
  }
  const auto& key0 = key_switch_keys.front();
  TORCH_CHECK(
      key0.dim() == 4 &&
          key0.size(0) == static_cast<int64_t>(extended_digits_ntt_qp.size()) &&
          key0.size(1) == (seeded_keys ? 1 : 2) &&
          key0.size(3) == prototype.size(2),
      operation,
      " key shape mismatch");
  TORCH_CHECK(!seeded_keys || key_seeds.size() == key_switch_keys.size(),
              operation,
              " requires one seed per seeded key");
  for (const auto& seed : key_seeds)
    check_key_seed(key0, seed, key0.size(0), operation);
  TORCH_CHECK(
      !seeded_keys ||
          prototype.size(2) % seeded::kSeededKeyCoefficientsPerCall == 0,
      operation,
      " requires a ring dimension divisible by 8");
  const auto key_rows = fhelium::detail::key_row_map(prototype.size(1),
                                                     key0.size(2),
                                                     key_digit_row_start,
                                                     key_p_row_start,
                                                     operation);
  const auto accumulator_prototype =
      view_rns_batch_3d(accumulators0_qp.front(), "accumulators0_qp[0]");
  TORCH_CHECK(accumulator_prototype.stride(2) == 1,
              operation,
              " requires contiguous accumulator coefficients");
  std::vector<torch::Tensor> outputs;
  for (size_t rotation = 0; rotation < rotation_count; ++rotation) {
    const auto& key = key_switch_keys[rotation];
    check_keyswitch_hip_peer(
        prototype_tensor, key, operation, "key_switch_key");
    TORCH_CHECK(key.sizes() == key0.sizes() && key.strides() == key0.strides(),
                operation,
                " requires rotation keys with one layout");
    for (const auto& accumulator_tensor :
         {accumulators0_qp[rotation], accumulators1_qp[rotation]}) {
      check_keyswitch_hip_peer(
          prototype_tensor, accumulator_tensor, operation, "accumulator");
      const auto accumulator =
          view_rns_batch_3d(accumulator_tensor, "accumulator");
      check_rns_binary_3d(accumulator, prototype, operation, false);
      TORCH_CHECK(accumulator.strides() == accumulator_prototype.strides(),
                  operation,
                  " requires accumulators with one layout");
      at::assert_no_internal_overlap(accumulator);
      outputs.push_back(accumulator);
    }
    if (!source_indices.empty()) {
      const auto& indices = source_indices[rotation];
      TORCH_CHECK(indices.device() == prototype_tensor.device() &&
                      indices.scalar_type() == torch::kInt32 &&
                      indices.dim() == 1 &&
                      indices.size(0) == prototype.size(2) &&
                      indices.stride(0) == source_indices.front().stride(0),
                  operation,
                  " requires int32 source indices matching N");
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = i + 1; j < outputs.size(); ++j)
      at::assert_no_overlap(outputs[i], outputs[j]);
    for (const auto& digit : extended_digits_ntt_qp)
      at::assert_no_overlap(outputs[i], digit);
    for (const auto& key : key_switch_keys)
      at::assert_no_overlap(outputs[i], key);
    for (const auto& indices : source_indices)
      at::assert_no_overlap(outputs[i], indices);
  }
  KeyswitchRotationStrides strides{};
  strides.accumulator_batch = accumulator_prototype.stride(0);
  strides.accumulator_row = accumulator_prototype.stride(1);
  strides.key_digit = key0.stride(0);
  strides.key_component = key0.stride(1);
  strides.key_row = key0.stride(2);
  strides.key_coefficient = key0.stride(3);
  strides.source_index =
      source_indices.empty() ? 0 : source_indices.front().stride(0);
  const int device = prototype_tensor.device().index();
  hipSetDevice(device);
  auto stream = c10::hip::getCurrentHIPStream(device);
  dim3 grid((prototype.size(2) + kHipBlockSize - 1) / kHipBlockSize,
            prototype.size(1),
            prototype.size(0));
  AT_DISPATCH_INTEGRAL_TYPES(prototype_tensor.scalar_type(), operation, [&] {
    KeyswitchDigitPointers<scalar_t> pointers{};
    for (int digit_index = 0;
         digit_index < static_cast<int>(extended_digits_ntt_qp.size());
         ++digit_index) {
      const auto digit =
          view_rns_batch_3d(extended_digits_ntt_qp[digit_index], "digit");
      pointers.values[digit_index] = digit.data_ptr<scalar_t>();
      pointers.batch_stride[digit_index] = digit.stride(0);
      pointers.row_stride[digit_index] = digit.stride(1);
      pointers.coefficient_stride[digit_index] = digit.stride(2);
    }
    if (seeded_keys) {
      if constexpr (sizeof(scalar_t) < 4) {
        TORCH_CHECK(false, operation, " requires int32 or int64 seeded keys");
      } else {
        constexpr int width = seeded::kSeededKeyCoefficientsPerCall;
        dim3 seeded_grid(prototype.size(1),
                         (prototype.size(2) / width + kSeededBlockSize - 1) /
                             kSeededBlockSize,
                         prototype.size(0));
        seeded::dispatch_seeded_key_rounds(
            key_rounds, [&](auto rounds_constant) {
              constexpr int kRounds = decltype(rounds_constant)::value;
              const auto kernel =
                  source_indices.empty()
                      ? keyswitch_accumulate_seeded_products_kernel<scalar_t,
                                                                    false,
                                                                    kRounds>
                      : keyswitch_accumulate_seeded_products_kernel<scalar_t,
                                                                    true,
                                                                    kRounds>;
              for (size_t first = 0; first < rotation_count;
                   first += kMaximumGroupedRotations) {
                const int count = static_cast<int>(std::min<size_t>(
                    kMaximumGroupedRotations, rotation_count - first));
                KeyswitchRotationPointers<scalar_t> rotations{};
                KeyswitchRotationSeeds seeds{};
                for (int r = 0; r < count; ++r) {
                  rotations.accumulator0[r] =
                      outputs[2 * (first + r)].data_ptr<scalar_t>();
                  rotations.accumulator1[r] =
                      outputs[2 * (first + r) + 1].data_ptr<scalar_t>();
                  rotations.key[r] =
                      key_switch_keys[first + r].data_ptr<scalar_t>();
                  rotations.source_indices[r] =
                      source_indices.empty()
                          ? nullptr
                          : source_indices[first + r].data_ptr<int32_t>();
                  seeds.rows[r] =
                      key_seeds[first + r].const_data_ptr<int64_t>();
                }
                kernel<<<seeded_grid, kSeededBlockSize, 0, stream>>>(
                    rotations,
                    seeds,
                    count,
                    strides,
                    pointers,
                    static_cast<int>(extended_digits_ntt_qp.size()),
                    static_cast<int>(prototype.size(2)),
                    FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2),
                    static_cast<int>(key_digit_row_start),
                    static_cast<int>(key_rows.q_rows),
                    static_cast<int>(key_rows.p_row_start));
              }
            });
      }
      return;
    }
    const auto kernel =
        source_indices.empty()
            ? keyswitch_accumulate_products_kernel<scalar_t, false>
            : keyswitch_accumulate_products_kernel<scalar_t, true>;
    for (size_t first = 0; first < rotation_count;
         first += kMaximumGroupedRotations) {
      const int count = static_cast<int>(
          std::min<size_t>(kMaximumGroupedRotations, rotation_count - first));
      KeyswitchRotationPointers<scalar_t> rotations{};
      for (int r = 0; r < count; ++r) {
        rotations.accumulator0[r] =
            outputs[2 * (first + r)].data_ptr<scalar_t>();
        rotations.accumulator1[r] =
            outputs[2 * (first + r) + 1].data_ptr<scalar_t>();
        rotations.key[r] = key_switch_keys[first + r].data_ptr<scalar_t>();
        rotations.source_indices[r] =
            source_indices.empty()
                ? nullptr
                : source_indices[first + r].data_ptr<int32_t>();
      }
      kernel<<<grid, kHipBlockSize, 0, stream>>>(
          rotations,
          count,
          strides,
          pointers,
          static_cast<int>(extended_digits_ntt_qp.size()),
          static_cast<int>(prototype.size(2)),
          FHELIUM_HIP_ACCESSOR32(rns_params, scalar_t, 2),
          static_cast<int>(key_digit_row_start),
          static_cast<int>(key_rows.q_rows),
          static_cast<int>(key_rows.p_row_start));
    }
  });
}

}  // namespace

namespace fhelium::ckks::gpu {

void keyswitch_moddown_qp_to_q_out(
    const at::Tensor& q_residues,
    const at::Tensor& p_residues,
    const at::Tensor& moddown_p_drop_inverses_montgomery,
    const at::Tensor& rns_params,
    const std::optional<at::Tensor>& moddown_half,
    c10::string_view rounding,
    at::Tensor& out,
    const std::optional<at::Tensor>& p_scratch) {
  ::keyswitch_moddown_qp_to_q_hip(q_residues,
                                  p_residues,
                                  moddown_p_drop_inverses_montgomery,
                                  rns_params,
                                  moddown_half,
                                  std::string(rounding),
                                  out,
                                  p_scratch);
}

void keyswitch_accumulate_digit_products_(
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    const at::Tensor& extended_digit_ntt_qp,
    const at::Tensor& key_switch_key_digit,
    const at::Tensor& rns_params,
    int64_t key_digit_row_start,
    int64_t key_p_row_start,
    const std::optional<at::Tensor>& source_indices,
    const std::optional<at::Tensor>& key_seed,
    int64_t key_rounds) {
  ::keyswitch_accumulate_digit_products_inplace_hip(accumulator0_qp,
                                                    accumulator1_qp,
                                                    extended_digit_ntt_qp,
                                                    key_switch_key_digit,
                                                    rns_params,
                                                    key_digit_row_start,
                                                    key_p_row_start,
                                                    source_indices,
                                                    key_seed,
                                                    key_rounds);
}

void keyswitch_accumulate_products_(at::TensorList accumulators0_qp,
                                    at::TensorList accumulators1_qp,
                                    at::TensorList extended_digits_ntt_qp,
                                    at::TensorList key_switch_keys,
                                    const at::Tensor& rns_params,
                                    int64_t key_digit_row_start,
                                    int64_t key_p_row_start,
                                    at::TensorList source_indices,
                                    at::TensorList key_seeds,
                                    int64_t key_rounds) {
  ::keyswitch_accumulate_products_inplace_hip(accumulators0_qp,
                                              accumulators1_qp,
                                              extended_digits_ntt_qp,
                                              key_switch_keys,
                                              rns_params,
                                              key_digit_row_start,
                                              key_p_row_start,
                                              source_indices,
                                              key_seeds,
                                              key_rounds);
}

}  // namespace fhelium::ckks::gpu
