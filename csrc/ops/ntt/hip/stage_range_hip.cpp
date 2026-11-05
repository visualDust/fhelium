#include <torch/library.h>
#include <torch/torch.h>

#include "../ntt_validation.h"
#include "stage_range_hip.h"

void compact_ntt_stage_range_(
    torch::Tensor residues, const torch::Tensor compact_twiddles,
    const torch::Tensor rns_params, bool inverse, int64_t start_stage,
    int64_t end_stage, int64_t grouped_stage_count) {
  fhelium::ntt::validate_compact_tables(
      residues, compact_twiddles, rns_params, grouped_stage_count);
  int64_t log_n = 0;
  for (int64_t n = residues.size(-1); n > 1; n >>= 1) ++log_n;
  TORCH_CHECK(0 <= start_stage && start_stage <= end_stage && end_stage <= log_n,
              "NTT stage interval must satisfy 0 <= start <= end <= log2(N)");
  if (start_stage == end_stage) return;
  if (inverse) {
    inverse_ntt_compact_stage_range_hip(
        residues, compact_twiddles, rns_params, start_stage, end_stage,
        grouped_stage_count);
  } else {
    forward_ntt_compact_stage_range_hip(
        residues, compact_twiddles, rns_params, start_stage, end_stage,
        grouped_stage_count);
  }
}

void fixed_radix_ntt_stage_range_(
    torch::Tensor residues, const torch::Tensor outer_twiddles,
    const torch::Tensor root_powers, const torch::Tensor rns_params,
    bool inverse, int64_t start_stage, int64_t end_stage) {
  const auto plan = fhelium::ntt::validate_fixed_radix_tables(
      residues, outer_twiddles, root_powers, rns_params);
  int64_t log_n = 0;
  for (int64_t n = residues.size(-1); n > 1; n >>= 1) ++log_n;
  const int bits = plan.radix == 4 ? 2 : plan.radix == 8 ? 3 : 4;
  TORCH_CHECK(0 <= start_stage && start_stage <= end_stage && end_stage <= log_n &&
                  start_stage % bits == 0 && end_stage % bits == 0,
              "Fixed-radix stage interval must select complete digits within log2(N)");
  if (start_stage == end_stage) return;
  if (inverse) {
    inverse_ntt_fixed_radix_stage_range_hip(
        residues, outer_twiddles, root_powers, rns_params,
        start_stage, end_stage, plan.radix);
  } else {
    forward_ntt_fixed_radix_stage_range_hip(
        residues, outer_twiddles, root_powers, rns_params,
        start_stage, end_stage, plan.radix);
  }
}

TORCH_LIBRARY_IMPL(fhelium_ntt_ops, CUDA, m) {
  m.impl("compact_ntt_stage_range_", TORCH_FN(compact_ntt_stage_range_));
  m.impl("fixed_radix_ntt_stage_range_", TORCH_FN(fixed_radix_ntt_stage_range_));
}
