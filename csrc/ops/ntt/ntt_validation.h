#pragma once

#include "ntt_execution_constants.h"

#include <ATen/MemoryOverlap.h>
#include <torch/torch.h>

#include <algorithm>

namespace fhelium::ntt {

inline void validate_no_residue_overlap(const torch::Tensor& residues,
                                        const torch::Tensor& read_only,
                                        const char* read_only_name) {
  // RestrictPtrTraits requires independent storage. Rejecting any storage
  // alias is intentionally stricter than proving element-range overlap and
  // remains reliable for non-contiguous active parameter views.
  TORCH_CHECK(!residues.is_alias_of(read_only),
              "NTT residues must not overlap ",
              read_only_name);
}

inline void validate_residues_and_parameters(const torch::Tensor& residues,
                                             const torch::Tensor& twiddles,
                                             const torch::Tensor& rns_params) {
  TORCH_CHECK(residues.device().is_cpu() || residues.is_cuda(),
              "NTT residues must be a CPU or CUDA tensor");
  at::assert_no_internal_overlap(residues);
  TORCH_CHECK(residues.dim() >= 2,
              "NTT residues must have shape [..., prime, coefficient]");
  TORCH_CHECK(
      residues.scalar_type() == at::kInt || residues.scalar_type() == at::kLong,
      "NTT residues must use torch.int32 or torch.int64");
  TORCH_CHECK(residues.size(-2) > 0, "NTT residues require at least one row");
  for (int64_t axis = 0; axis < residues.dim() - 2; ++axis) {
    TORCH_CHECK(residues.size(axis) > 0,
                "NTT residue batch dimensions must be nonzero");
  }

  const int64_t ring_dimension = residues.size(-1);
  TORCH_CHECK(
      ring_dimension >= 256 && (ring_dimension & (ring_dimension - 1)) == 0,
      "NTT coefficient count must be a power of two >= 256");
  if (residues.is_cuda()) {
    TORCH_CHECK(
        ring_dimension % 256 == 0,
        "CUDA NTT coefficient count must be divisible by the block size");
  }

  TORCH_CHECK(twiddles.device() == residues.device() &&
                  rns_params.device() == residues.device(),
              "NTT residues, twiddles, and RNS parameters must share a device");
  TORCH_CHECK(twiddles.scalar_type() == residues.scalar_type() &&
                  rns_params.scalar_type() == residues.scalar_type(),
              "NTT residues, twiddles, and RNS parameters must share a dtype");
  TORCH_CHECK(twiddles.dim() >= 1,
              "NTT twiddles must have a leading prime-row dimension");
  TORCH_CHECK(twiddles.size(0) == residues.size(-2),
              "NTT twiddle rows must equal operand rows");
  TORCH_CHECK(rns_params.dim() == 2 && rns_params.size(0) >= 8 &&
                  rns_params.size(1) == residues.size(-2),
              "RNS parameters must have shape [at least 8, operand rows]");
  validate_no_residue_overlap(residues, twiddles, "NTT twiddles");
  validate_no_residue_overlap(residues, rns_params, "RNS parameters");
}

inline void validate_indexed_tables(const torch::Tensor& residues,
                                    const torch::Tensor& even_indices,
                                    const torch::Tensor& odd_indices,
                                    const torch::Tensor& twiddles,
                                    const torch::Tensor& rns_params) {
  validate_residues_and_parameters(residues, twiddles, rns_params);
  TORCH_CHECK(even_indices.device() == residues.device() &&
                  odd_indices.device() == residues.device(),
              "Indexed NTT schedules must share the operand device");
  TORCH_CHECK(even_indices.scalar_type() == at::kInt &&
                  odd_indices.scalar_type() == at::kInt,
              "Indexed NTT schedules must use torch.int32");
  TORCH_CHECK(
      even_indices.dim() == 2 && even_indices.sizes() == odd_indices.sizes(),
      "Indexed NTT schedules must have matching [stage, butterfly] shapes");
  TORCH_CHECK(
      even_indices.size(0) > 0 && even_indices.size(1) == residues.size(-1) / 2,
      "Indexed NTT schedule shape does not match the operand");
  int64_t expected_stage_count = 0;
  for (int64_t n = residues.size(-1); n > 1; n >>= 1) {
    ++expected_stage_count;
  }
  TORCH_CHECK(even_indices.size(0) == expected_stage_count,
              "Indexed NTT schedules must contain log2(N) stages; got ",
              even_indices.size(0),
              " for N=",
              residues.size(-1));
  TORCH_CHECK(twiddles.dim() == 3 && twiddles.size(1) == even_indices.size(0) &&
                  twiddles.size(2) == even_indices.size(1),
              "Indexed twiddles must have shape [prime, stage, butterfly]");
  validate_no_residue_overlap(residues, even_indices, "indexed even schedule");
  validate_no_residue_overlap(residues, odd_indices, "indexed odd schedule");
}

inline void validate_compact_tables(const torch::Tensor& residues,
                                    const torch::Tensor& twiddles,
                                    const torch::Tensor& rns_params,
                                    const int64_t grouped_stage_count) {
  validate_residues_and_parameters(residues, twiddles, rns_params);
  TORCH_CHECK(twiddles.dim() == 2 && twiddles.size(1) == residues.size(-1),
              "Compact twiddles must have shape [prime, coefficient]");
  TORCH_CHECK(grouped_stage_count >= 2 && grouped_stage_count <= 4,
              "grouped_stage_count must be 2, 3, or 4");
}

inline void validate_shared_memory_log_n(const int64_t shared_memory_log_n) {
  TORCH_CHECK(shared_memory_log_n >= 0 &&
                  shared_memory_log_n <= kNttMaxSharedMemoryLogN,
              "NTT shared_memory_log_n must be between 0 and ",
              kNttMaxSharedMemoryLogN);
}

struct ValidatedFixedRadixPlan {
  int radix;
};

inline ValidatedFixedRadixPlan validate_fixed_radix_tables(
    const torch::Tensor& residues,
    const torch::Tensor& outer_twiddles,
    const torch::Tensor& radix_root_powers,
    const torch::Tensor& rns_params) {
  validate_residues_and_parameters(residues, outer_twiddles, rns_params);
  TORCH_CHECK(outer_twiddles.dim() == 2 &&
                  outer_twiddles.size(1) == residues.size(-1) - 1,
              "Power-of-two radix outer twiddles must have shape [prime, "
              "coefficient - 1]");
  const int64_t radix =
      radix_root_powers.dim() == 2 ? radix_root_powers.size(1) : -1;
  TORCH_CHECK(radix == 4 || radix == 8 || radix == 16,
              "Power-of-two radix root width must select radix 4, 8, or 16");
  TORCH_CHECK(radix_root_powers.device() == residues.device(),
              "Power-of-two radix root powers must share the operand device");
  TORCH_CHECK(radix_root_powers.scalar_type() == residues.scalar_type(),
              "Power-of-two radix root powers must share the operand dtype");
  TORCH_CHECK(radix_root_powers.dim() == 2 &&
                  radix_root_powers.size(0) == residues.size(-2) &&
                  radix_root_powers.size(1) == radix,
              "Power-of-two radix root powers must have shape [prime, "
              "radix]");
  validate_no_residue_overlap(
      residues, radix_root_powers, "power-of-two radix root powers");

  int logN = 0;
  for (int64_t n = residues.size(-1); n > 1; n >>= 1) ++logN;
  const int radix_bits = radix == 16 ? 4 : (radix == 8 ? 3 : 2);
  TORCH_CHECK(logN % radix_bits == 0,
              "Strict fixed-radix execution requires logN divisible by ",
              radix_bits,
              "; got logN=",
              logN);
  return ValidatedFixedRadixPlan{static_cast<int>(radix)};
}

}  // namespace fhelium::ntt
