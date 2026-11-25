#include <torch/torch.h>

#include "../backends.h"
#include "../ntt_validation.h"
#include "compact_ntt_hip.h"

namespace fhelium::ntt::gpu {

namespace {

void validate(const at::Tensor& residues,
              const at::Tensor& twiddles,
              const at::Tensor& rns_params) {
  fhelium::ntt::validate_compact_tables(residues, twiddles, rns_params);
}

void inverse(at::Tensor& residues,
             const at::Tensor& inverse_twiddles,
             const at::Tensor& rns_params,
             CompactInverseResult result) {
  validate(residues, inverse_twiddles, rns_params);
  compact_inverse_ntt_hip(residues, inverse_twiddles, rns_params, result);
}

}  // namespace

void forward_ntt_montgomery_compact_(at::Tensor& montgomery_residues,
                                     const at::Tensor& forward_twiddles,
                                     const at::Tensor& rns_params) {
  validate(montgomery_residues, forward_twiddles, rns_params);
  compact_forward_ntt_hip(montgomery_residues, montgomery_residues,
                           forward_twiddles, rns_params,
                           CompactForwardInput::Montgomery);
}

void forward_ntt_to_montgomery_compact_(at::Tensor& standard_residues,
                                        const at::Tensor& forward_twiddles,
                                        const at::Tensor& rns_params) {
  validate(standard_residues, forward_twiddles, rns_params);
  compact_forward_ntt_hip(standard_residues, standard_residues,
                           forward_twiddles, rns_params,
                           CompactForwardInput::Standard);
}

at::Tensor forward_ntt_to_montgomery_compact(const at::Tensor& standard_residues,
                                             const at::Tensor& forward_twiddles,
                                             const at::Tensor& rns_params) {
  validate(standard_residues, forward_twiddles, rns_params);
  auto out = at::empty(standard_residues.sizes(), standard_residues.options());
  compact_forward_ntt_hip(standard_residues, out, forward_twiddles, rns_params,
                           CompactForwardInput::Standard);
  return out;
}

void forward_ntt_to_montgomery_compact_add_scaled_(at::Tensor& source,
                                                   const at::Tensor& addend,
                                                   const at::Tensor& multiplier,
                                                   const at::Tensor& forward_twiddles,
                                                   const at::Tensor& rns_params) {
  validate(source, forward_twiddles, rns_params);
  compact_forward_ntt_add_scaled_hip(source, addend, multiplier,
                                      forward_twiddles, rns_params);
}

void forward_ntt_montgomery_compact_keyswitch_accumulate_(
    at::Tensor& coefficient_digit_qp,
    const at::Tensor& forward_twiddles,
    const at::Tensor& rns_params,
    const at::Tensor& key_digit_qp,
    at::Tensor& accumulator0_qp,
    at::Tensor& accumulator1_qp,
    int64_t key_row_start,
    int64_t key_p_row_start) {
  // coefficient_digit_qp is disposable scratch with coefficient/Montgomery
  // input [*batch, QP limb, N]; its NTT values are consumed by the final pass.
  validate(coefficient_digit_qp, forward_twiddles, rns_params);
  compact_forward_ntt_keyswitch_accumulate_hip(
      coefficient_digit_qp, forward_twiddles, rns_params, key_digit_qp,
      accumulator0_qp, accumulator1_qp, key_row_start, key_p_row_start);
}

void inverse_ntt_montgomery_compact_(at::Tensor& montgomery_residues,
                                     const at::Tensor& inverse_twiddles,
                                     const at::Tensor& rns_params) {
  inverse(montgomery_residues, inverse_twiddles, rns_params,
          CompactInverseResult::Montgomery);
}

void inverse_ntt_to_standard_lazy_compact_(at::Tensor& montgomery_residues,
                                           const at::Tensor& inverse_twiddles,
                                           const at::Tensor& rns_params) {
  inverse(montgomery_residues, inverse_twiddles, rns_params,
          CompactInverseResult::StandardLazy);
}

void inverse_ntt_to_standard_compact_(at::Tensor& montgomery_residues,
                                      const at::Tensor& inverse_twiddles,
                                      const at::Tensor& rns_params) {
  inverse(montgomery_residues, inverse_twiddles, rns_params,
          CompactInverseResult::Standard);
}

void inverse_ntt_to_centered_compact_(at::Tensor& montgomery_residues,
                                      const at::Tensor& inverse_twiddles,
                                      const at::Tensor& rns_params) {
  inverse(montgomery_residues, inverse_twiddles, rns_params,
          CompactInverseResult::Centered);
}

}  // namespace fhelium::ntt::gpu
