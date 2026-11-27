#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/operation_tables.h>

#include <cstdint>
#include <optional>
#include <string_view>
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

// Number of schedule tables that follow the parameters of one NTT direction.
int64_t ntt_schedule_count(std::string_view name);

// One hybrid digit (start, stop, key digit) of a flattened digit list.
struct DigitSpan {
  int64_t start;
  int64_t stop;
  int64_t key_digit;
};

std::vector<DigitSpan> digit_spans(c10::IntArrayRef flat);

// A table field as an optional operand: no value when the field is undefined.
inline std::optional<at::Tensor> optional_table(const at::Tensor& table) {
  return table.defined() ? std::optional<at::Tensor>(table) : std::nullopt;
}

// Returns NTT_fwd(correction) + addend * multiplier on the `forward`
// schedule. Compact radix-2 policies fuse the sum into the transform and
// update `correction`. Other policies scale `addend` with
// montgomery_mul_row_scalars_ on a copy when `lazy_scaling`, otherwise with
// montgomery_mul_row_scalars_standard, and return a new lazy sum. A defined
// `scaled`, shaped like `addend`, holds the scaled addend.
at::Tensor forward_add_scaled(at::Tensor correction,
                              const at::Tensor& addend,
                              const at::Tensor& multiplier,
                              const at::Tensor& parameters,
                              const NttTables& forward,
                              std::string_view ntt_backend,
                              bool lazy_scaling,
                              const at::Tensor& out = at::Tensor(),
                              const at::Tensor& scaled = at::Tensor());

// rns::moddown_ntt_qp_to_q of NTT QP residues split into `q_rows` and
// `p_rows`; the inverse P transform overwrites `p_rows`, so callers that own
// the QP accumulator pass its P rows directly. They also hold the ModDown
// digits. A defined one-element `zero` provides the zero Q residues of the
// non-CRT ModDown.
at::Tensor moddown_ntt_rows(const at::Tensor& q_rows,
                            at::Tensor p_rows,
                            const std::optional<at::Tensor>& coefficient_c0,
                            const KeySwitchTables& tables,
                            std::string_view ntt_backend,
                            std::string_view rounding,
                            const at::Tensor& out = at::Tensor(),
                            const at::Tensor& scaled = at::Tensor(),
                            const at::Tensor& zero = at::Tensor());

// floor((x + h) / D) with the trailing drop_count rows of `source` (after
// the optional row_order gather into `ordered`, allocated when undefined) as
// divisor rows. A defined `p_scratch` holds the P digits.
at::Tensor divide_trailing_rows(const at::Tensor& source,
                                const at::Tensor& inverses,
                                const at::Tensor& parameters,
                                int64_t drop_count,
                                const std::optional<at::Tensor>& half,
                                const std::optional<at::Tensor>& row_order,
                                std::string_view rounding,
                                const at::Tensor& out = at::Tensor(),
                                const at::Tensor& ordered = at::Tensor(),
                                const at::Tensor& p_scratch = at::Tensor());

// keyswitch_moddown_qp_to_q into `out` when it is defined; a defined
// `p_scratch`, which may be `p_rows` itself, holds the P digits.
at::Tensor moddown_rows(const at::Tensor& q_rows,
                        const at::Tensor& p_rows,
                        const at::Tensor& inverses,
                        const at::Tensor& parameters,
                        const std::optional<at::Tensor>& half,
                        std::string_view rounding,
                        const at::Tensor& out,
                        const at::Tensor& p_scratch = at::Tensor());

// Copies `result` into a defined `out` that the computation did not write.
inline void finish_out(const at::Tensor& result, at::Tensor& out) {
  TORCH_CHECK_VALUE(out.sizes() == result.sizes() &&
                        out.scalar_type() == result.scalar_type() &&
                        out.device() == result.device(),
                    "out requires shape ",
                    result.sizes(),
                    " with the result dtype and device");
  if (!result.is_same(out)) out.copy_(result);
}

// Whether hybrid_modup_digit of a digit of `width` rows of `source` runs the
// fused CUDA tile kernel, which needs no scratch; it runs when no scratch is
// given.
bool uses_tiled_modup(const at::Tensor& source,
                      const at::Tensor& qp_parameters,
                      int64_t width);

// Lifts one digit to the QP rows of `qp_parameters` as Montgomery residues,
// writing `out` when defined; the exact method uses `scratch` for the Garner
// decomposition when defined.
at::Tensor hybrid_modup_digit(const at::Tensor& source,
                              const at::Tensor& qp_parameters,
                              const DigitTables& digit,
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
                                 const KeySwitchTables& tables,
                                 std::string_view basis_conversion,
                                 int64_t digit_index,
                                 const DigitSpan& span,
                                 const at::Tensor& out,
                                 const at::Tensor& scratch);

}  // namespace fhelium::detail_operations
