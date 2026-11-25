#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/tensor_table.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

// Shared steps of the whole RNS and CKKS operations. Each helper performs the
// same sequence of libfhelium primitives as the corresponding operation
// definition, so results are identical bit for bit across call sites.

namespace fhelium::detail_operations {

enum class NttFamily { kIndexed, kCompact, kFixedRadix };

struct NttPolicy {
  NttFamily family;
};

// Resolves an FHElium NTT backend name.
NttPolicy ntt_policy(std::string_view name);

// One hybrid digit (start, stop, key digit) of a flattened digit list.
struct DigitSpan {
  int64_t start;
  int64_t stop;
  int64_t key_digit;
};

std::vector<DigitSpan> digit_spans(c10::IntArrayRef flat);

// Cached device gathers of the Galois automorphism X -> X^g.
const at::Tensor& ntt_galois_indices(int64_t ring_dimension,
                                     int64_t galois_element,
                                     const at::Device& device);
const std::pair<at::Tensor, at::Tensor>& coefficient_galois_indices(
    int64_t ring_dimension, int64_t galois_element, const at::Device& device);

// Returns NTT_fwd(correction) + addend * multiplier on the `basis` forward
// schedule. Compact radix-2 policies fuse the sum into the transform and
// update `correction`. Other policies scale `addend` with
// montgomery_mul_row_scalars_ on a copy when `lazy_scaling`, otherwise with
// montgomery_mul_row_scalars_standard, and return a new lazy sum.
at::Tensor forward_add_scaled(at::Tensor correction,
                              const at::Tensor& addend,
                              const at::Tensor& multiplier,
                              const at::Tensor& parameters,
                              const TensorTable& tables,
                              std::string_view ntt_backend,
                              std::string_view basis,
                              bool lazy_scaling);

// rns::moddown_ntt_qp_to_q of NTT QP residues split into `q_rows` and
// `p_rows`; the inverse P transform overwrites `p_rows`, so callers that own
// the QP accumulator pass its P rows directly.
at::Tensor moddown_ntt_rows(const at::Tensor& q_rows,
                            at::Tensor p_rows,
                            const std::optional<at::Tensor>& coefficient_c0,
                            const TensorTable& tables,
                            std::string_view ntt_backend,
                            std::string_view rounding);

// floor((x + h) / D) with the trailing drop_count rows of `source` (after
// the optional row_order gather) as divisor rows.
at::Tensor divide_trailing_rows(const at::Tensor& source,
                                const at::Tensor& inverses,
                                const at::Tensor& parameters,
                                int64_t drop_count,
                                const std::optional<at::Tensor>& half,
                                const std::optional<at::Tensor>& row_order,
                                std::string_view rounding);

// Lifts one digit to QP Montgomery residues, writing `out` when defined; the
// exact method uses `scratch` for the Garner decomposition when defined.
at::Tensor hybrid_modup_digit(const at::Tensor& source,
                              const TensorTable& tables,
                              int64_t digit_index,
                              const DigitSpan& span,
                              const at::Tensor& out,
                              const at::Tensor& scratch);

// Evaluates a CRT basis-extension matrix with Montgomery products on any
// device; see fhelium::rns::crt_basis_extend.
at::Tensor portable_crt_basis_extend(
    const at::Tensor& source,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const std::optional<at::Tensor>& source_normalizers,
    const std::optional<at::Tensor>& target_offsets);

// Lifts one digit by the selected modup method (exact, crt, normalized_crt).
at::Tensor lift_key_switch_digit(const at::Tensor& source,
                                 const TensorTable& tables,
                                 std::string_view basis_conversion,
                                 int64_t digit_index,
                                 const DigitSpan& span,
                                 const at::Tensor& out,
                                 const at::Tensor& scratch);

}  // namespace fhelium::detail_operations
