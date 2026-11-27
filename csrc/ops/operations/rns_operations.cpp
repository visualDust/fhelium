#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include "../common/overlap.h"
#include "../common/workspace.h"

#include <vector>

#include "composition.h"

namespace fhelium::detail_operations {

at::Tensor portable_crt_basis_extend(
    const at::Tensor& source,
    const at::Tensor& source_parameters,
    const at::Tensor& target_parameters,
    const at::Tensor& matrix,
    const std::optional<at::Tensor>& source_normalizers,
    const std::optional<at::Tensor>& target_offsets) {
  const auto destinations = matrix.size(1);
  auto values = source.clone();
  rns::reduce_to_standard_(values, source_parameters);
  if (source_normalizers.has_value()) {
    auto normalizers = source_normalizers->unsqueeze(-1).clone();
    rns::to_montgomery_(normalizers, source_parameters);
    values = rns::montgomery_mul_row_scalars_standard(
        values, normalizers.squeeze(-1), source_parameters);
    rns::reduce_to_standard_(values, source_parameters);
    // Shift by (q - 1)/2; target_offsets removes its matrix image, so the
    // lifted digit is the centered residue.
    auto halves = at::floor_divide(source_parameters.select(0, 0), 4)
                      .unsqueeze(-1)
                      .expand_as(values)
                      .contiguous();
    values = rns::add_lazy(values, halves, source_parameters);
    rns::reduce_to_standard_(values, source_parameters);
  }
  // A second factor R lets ordinary Montgomery multiplication map an
  // unencoded source residue directly to the encoded matrix result.
  auto coefficients =
      matrix.transpose(0, 1).clone(at::MemoryFormat::Contiguous);
  rns::to_montgomery_(coefficients, target_parameters);
  std::vector<int64_t> modulus_shape(source.dim() - 2, 1);
  modulus_shape.push_back(-1);
  modulus_shape.push_back(1);
  const auto moduli =
      at::floor_divide(target_parameters.select(0, 0), 2).view(modulus_shape);
  auto shape = source.sizes().vec();
  shape[shape.size() - 2] = destinations;
  auto result =
      target_offsets.has_value()
          ? target_offsets->view(modulus_shape).expand(shape).contiguous()
          : at::zeros(shape, source.options());
  for (int64_t row = 0; row < matrix.size(0); ++row) {
    const auto residues = at::remainder(values.narrow(-2, row, 1), moduli);
    const auto term = rns::montgomery_mul_row_scalars_standard(
        residues, coefficients.select(1, row), target_parameters);
    result = rns::add_lazy(result, term, target_parameters);
  }
  rns::reduce_to_standard_(result, target_parameters);
  return result;
}

}  // namespace fhelium::detail_operations

namespace fhelium::rns {

namespace {

using detail_operations::divide_trailing_rows;
using detail_operations::finish_out;
using detail_operations::forward_add_scaled;
using detail_operations::moddown_rows;
using detail_operations::NttFamily;
using detail_operations::optional_table;
using fhelium::detail::WorkspaceLayout;
using fhelium::detail::WorkspaceTensor;

// Centered CRT correction -r/D of the centered quotient tables `crt`; see
// `build_centered_quotient_tables`.
at::Tensor centered_quotient_correction(const at::Tensor& source,
                                        const CrtTables& crt,
                                        const at::Tensor& out = at::Tensor()) {
  if (out.defined()) {
    auto target = out;
    crt_basis_extend_out(source,
                         crt.source_parameters,
                         crt.target_parameters,
                         crt.matrix,
                         crt.target_reciprocals,
                         optional_table(crt.source_normalizers),
                         optional_table(crt.source_reciprocals),
                         optional_table(crt.target_offsets),
                         target);
    return target;
  }
  return crt_basis_extend(source,
                          crt.source_parameters,
                          crt.target_parameters,
                          crt.matrix,
                          crt.target_reciprocals,
                          optional_table(crt.source_normalizers),
                          optional_table(crt.source_reciprocals),
                          optional_table(crt.target_offsets));
}

// Returns `rows` placed at `positions` of a zero tensor with `row_count`
// rows; the other rows hold residue zero.
at::Tensor place_rows(const at::Tensor& rows,
                      const at::Tensor& positions,
                      int64_t row_count) {
  auto shape = rows.sizes().vec();
  shape[shape.size() - 2] = row_count;
  auto placed = at::zeros(shape, rows.options());
  placed.index_copy_(-2, positions, rows);
  return placed;
}

// Each `*_into` function below computes the operation of the same name; a
// defined `out` receives the final producer's result.

// Temporaries of rational_rescale; see `plan_rational_rescale`.
struct RationalRescalePlan {
  WorkspaceTensor dropped;   // the D rows, consumed by the ModDown
  WorkspaceTensor kept;      // the B ∩ B' rows
  WorkspaceTensor placed;    // the kept rows placed in B'
  WorkspaceTensor restored;  // R x
  WorkspaceTensor scaled;    // kept rows times product_inverse
  int64_t bytes = 0;
};

RationalRescalePlan plan_rational_rescale(const at::Tensor& source,
                                          const RescaleTables& tables,
                                          std::string_view input_domain,
                                          std::string_view ntt_backend) {
  RationalRescalePlan plan;
  const auto dtype = source.scalar_type();
  const auto with_rows = [&](int64_t rows) {
    auto shape = source.sizes().vec();
    shape[shape.size() - 2] = rows;
    return shape;
  };
  WorkspaceLayout layout;
  plan.dropped = layout.take(with_rows(tables.drop_positions.size(0)), dtype);
  auto kept = with_rows(tables.keep_positions.size(0));
  plan.kept = layout.take(kept, dtype);
  if (tables.target_keep_positions.defined()) {
    kept = with_rows(tables.target_parameters.size(-1));
    plan.placed = layout.take(kept, dtype);
  }
  const auto start = layout.mark();
  if (tables.restore.defined())
    plan.restored = layout.take(source.sizes().vec(), dtype);
  layout.release(start);
  if (input_domain == "ntt" &&
      detail_operations::ntt_policy(ntt_backend).family != NttFamily::kCompact)
    plan.scaled = layout.take(kept, dtype);
  plan.bytes = layout.bytes();
  return plan;
}

// Temporaries of `plan` are views of a defined `workspace` and are allocated
// otherwise.
at::Tensor rational_rescale_into(const at::Tensor& source,
                                 const RescaleTables& tables,
                                 std::string_view input_domain,
                                 std::string_view ntt_backend,
                                 const at::Tensor& out,
                                 const RationalRescalePlan& plan,
                                 const at::Tensor& workspace) {
  const auto temporary = [&](const WorkspaceTensor& tensor) {
    return workspace.defined()
               ? tensor.in(workspace)
               : at::empty(tensor.sizes, source.options().dtype(tensor.dtype));
  };
  const auto& source_parameters = tables.source_parameters;
  const auto& target_parameters = tables.target_parameters;
  const auto restore = optional_table(tables.restore);
  auto scaled = source;
  if (restore.has_value()) {
    scaled = temporary(plan.restored);
    montgomery_mul_row_scalars_standard_out(
        source, *restore, source_parameters, scaled);
  }
  auto dropped = temporary(plan.dropped);
  at::index_select_out(dropped, scaled, -2, tables.drop_positions);
  auto kept = temporary(plan.kept);
  at::index_select_out(kept, scaled, -2, tables.keep_positions);
  if (const auto positions = optional_table(tables.target_keep_positions)) {
    auto placed = temporary(plan.placed);
    placed.zero_();
    placed.index_copy_(-2, *positions, kept);
    kept = placed;
  }
  if (input_domain == "ntt") {
    ntt::transform_(
        dropped, tables.dropped_inverse, ntt_backend, "inverse_to_standard");
  }
  add_standard_(dropped, tables.dropped_offset, tables.dropped_parameters);
  if (input_domain != "ntt") {
    add_standard_(kept, tables.target_offset, target_parameters);
    return moddown_rows(kept,
                        dropped,
                        tables.inverses,
                        tables.ordered_parameters,
                        std::nullopt,
                        "floor",
                        out,
                        dropped);
  }
  std::vector<int64_t> offset_shape(dropped.sizes().begin(),
                                    dropped.sizes().end() - 2);
  offset_shape.push_back(-1);
  offset_shape.push_back(-1);
  auto correction = moddown_rows(tables.target_offset.expand(offset_shape),
                                 dropped,
                                 tables.inverses,
                                 tables.ordered_parameters,
                                 std::nullopt,
                                 "floor",
                                 out,
                                 dropped);
  return forward_add_scaled(
      correction,
      kept,
      tables.product_inverse,
      target_parameters,
      tables.target_forward,
      ntt_backend,
      false,
      out,
      plan.scaled.defined() ? temporary(plan.scaled) : at::Tensor());
}

at::Tensor hybrid_modup_digit_into(const at::Tensor& source,
                                   const KeySwitchTables& tables,
                                   int64_t digit_index,
                                   c10::IntArrayRef digits,
                                   const at::Tensor& out) {
  const auto spans = detail_operations::digit_spans(digits);
  TORCH_CHECK_VALUE(
      digit_index >= 0 && digit_index < static_cast<int64_t>(spans.size()) &&
          digit_index < static_cast<int64_t>(tables.digits.size()),
      "digit_index is outside the digit list");
  return detail_operations::hybrid_modup_digit(source,
                                               tables.qp_parameters,
                                               tables.digits[digit_index],
                                               spans[digit_index],
                                               out,
                                               at::Tensor());
}

// Temporaries of centered_basis_extend; see `plan_centered_basis_extend`.
struct CenteredBasisExtendPlan {
  WorkspaceTensor digits;      // reduced source residues
  WorkspaceTensor decomposed;  // mixed-radix digits
  WorkspaceTensor extended;    // the unsigned lift in the target basis
  WorkspaceTensor equal;       // leading digits equal to the half digits
  WorkspaceTensor negative;    // the centered value is negative
  WorkspaceTensor compare;
  WorkspaceTensor sign;  // `negative` in the residue dtype
  int64_t bytes = 0;
};

CenteredBasisExtendPlan plan_centered_basis_extend(
    const at::Tensor& source, const BasisExtensionTables& tables) {
  CenteredBasisExtendPlan plan;
  const auto dtype = source.scalar_type();
  WorkspaceLayout layout;
  plan.digits = layout.like(source);
  if (source.size(-2) > 1) plan.decomposed = layout.like(source);
  auto extended = source.sizes().vec();
  extended[extended.size() - 2] = tables.target_parameters.size(-1);
  plan.extended = layout.take(extended, dtype);
  auto mask = source.sizes().vec();
  mask.erase(mask.end() - 2);
  plan.equal = layout.take(mask, at::kBool);
  plan.negative = layout.take(mask, at::kBool);
  plan.compare = layout.take(mask, at::kBool);
  plan.sign = layout.take(mask, dtype);
  plan.bytes = layout.bytes();
  return plan;
}

// Temporaries of `plan` are views of a defined `workspace` and are allocated
// otherwise.
at::Tensor centered_basis_extend_into(const at::Tensor& source,
                                      const BasisExtensionTables& tables,
                                      const at::Tensor& out,
                                      const CenteredBasisExtendPlan& plan,
                                      const at::Tensor& workspace) {
  const auto temporary = [&](const WorkspaceTensor& tensor) {
    return workspace.defined()
               ? tensor.in(workspace)
               : at::empty(tensor.sizes, source.options().dtype(tensor.dtype));
  };
  const auto& parameters = tables.source_parameters;
  auto digits = temporary(plan.digits);
  digits.copy_(source);
  reduce_to_standard_(digits, parameters);
  const auto count = source.size(-2);
  if (count > 1) {
    auto decomposed = temporary(plan.decomposed);
    mixed_radix_decompose_out(digits,
                              tables.normalizers,
                              tables.propagation,
                              tables.source_lo,
                              tables.source_hi,
                              tables.source_neg_lo,
                              tables.source_neg_hi,
                              decomposed);
    digits = decomposed;
    reduce_to_standard_(digits, parameters);
  }
  const auto& target_parameters = tables.target_parameters;
  auto output = temporary(plan.extended);
  mixed_radix_basis_extend_to_montgomery_out(digits,
                                             tables.extension,
                                             target_parameters,
                                             target_parameters.size(-1),
                                             output);
  from_montgomery_(output, target_parameters);
  reduce_to_standard_(output, target_parameters);
  // The value is negative when its leading differing digit exceeds the half
  // digit.
  auto equal = temporary(plan.equal).fill_(true);
  auto negative = temporary(plan.negative).zero_();
  auto compare = temporary(plan.compare);
  const auto& half = tables.half_digits;
  for (int64_t row = count - 1; row >= 0; --row) {
    const auto digit = digits.select(-2, row);
    at::gt_out(compare, digit, half[row]);
    compare.bitwise_and_(equal);
    negative.bitwise_or_(compare);
    at::eq_out(compare, digit, half[row]);
    equal.bitwise_and_(compare);
  }
  std::vector<int64_t> shape(output.dim() - 2, 1);
  shape.push_back(-1);
  shape.push_back(1);
  auto sign = temporary(plan.sign);
  sign.copy_(negative);
  output.addcmul_(sign.unsqueeze(-2), tables.source_product.view(shape), -1);
  if (out.defined()) {
    auto target = out;
    at::remainder_out(target, output, tables.target_moduli.view(shape));
    return target;
  }
  return at::remainder(output, tables.target_moduli.view(shape));
}

// Temporaries of rounded_trailing_quotient: the gathered rows and the P
// digits.
struct TrailingQuotientPlan {
  WorkspaceTensor ordered;
  WorkspaceTensor scratch;
  int64_t bytes = 0;
};

TrailingQuotientPlan plan_rounded_trailing_quotient(
    const at::Tensor& source,
    const TrailingQuotientTables& tables,
    int64_t drop_count) {
  TrailingQuotientPlan plan;
  WorkspaceLayout layout;
  if (tables.row_order.defined()) plan.ordered = layout.like(source);
  auto shape = source.sizes().vec();
  shape[shape.size() - 2] = drop_count;
  plan.scratch = layout.take(shape, source.scalar_type());
  plan.bytes = layout.bytes();
  return plan;
}

at::Tensor rounded_trailing_quotient_into(const at::Tensor& source,
                                          const TrailingQuotientTables& tables,
                                          int64_t drop_count,
                                          std::string_view rounding,
                                          const at::Tensor& out,
                                          const TrailingQuotientPlan& plan,
                                          const at::Tensor& workspace) {
  const auto view = [&](const WorkspaceTensor& tensor) {
    return workspace.defined() && tensor.defined() ? tensor.in(workspace)
                                                   : at::Tensor();
  };
  return divide_trailing_rows(source,
                              tables.inverses,
                              tables.parameters,
                              drop_count,
                              optional_table(tables.half),
                              optional_table(tables.row_order),
                              rounding,
                              out,
                              view(plan.ordered),
                              view(plan.scratch));
}

// Temporaries of moddown_qp_to_q: the P^{-1} term of the CRT path, or the P
// digits otherwise.
struct ModdownQpToQPlan {
  WorkspaceTensor term;
  WorkspaceTensor scratch;
  int64_t bytes = 0;
};

ModdownQpToQPlan plan_moddown_qp_to_q(const at::Tensor& source,
                                      const KeySwitchTables& tables,
                                      int64_t p_count) {
  ModdownQpToQPlan plan;
  WorkspaceLayout layout;
  auto shape = source.sizes().vec();
  shape[shape.size() - 2] =
      tables.moddown_crt.matrix.defined() ? source.size(-2) - p_count : p_count;
  (tables.moddown_crt.matrix.defined() ? plan.term : plan.scratch) =
      layout.take(shape, source.scalar_type());
  plan.bytes = layout.bytes();
  return plan;
}

at::Tensor moddown_qp_to_q_into(const at::Tensor& source,
                                const KeySwitchTables& tables,
                                int64_t p_count,
                                std::string_view rounding,
                                const at::Tensor& out,
                                const ModdownQpToQPlan& plan,
                                const at::Tensor& workspace) {
  if (tables.moddown_crt.matrix.defined()) {
    const auto q_count = source.size(-2) - p_count;
    const auto& q_parameters = tables.q_parameters;
    auto result = centered_quotient_correction(
        source.narrow(-2, q_count, p_count), tables.moddown_crt, out);
    auto term = workspace.defined()
                    ? plan.term.in(workspace)
                    : at::empty(plan.term.sizes, source.options());
    montgomery_mul_row_scalars_standard_out(
        source.narrow(-2, 0, q_count), tables.p_inverse, q_parameters, term);
    add_standard_(result, term, q_parameters);
    return result;
  }
  return divide_trailing_rows(
      source,
      tables.moddown_inverses,
      tables.qp_parameters,
      p_count,
      optional_table(tables.moddown_half),
      std::nullopt,
      rounding,
      out,
      at::Tensor(),
      workspace.defined() ? plan.scratch.in(workspace) : at::Tensor());
}

// A zero residue Tensor with the shape of `like`, broadcast from one element
// so that readers do not require a materialized fill.
at::Tensor zero_residues(const at::Tensor& like) {
  return at::zeros({1}, like.options()).expand(like.sizes());
}

}  // namespace

}  // namespace fhelium::rns

namespace fhelium::detail_operations {

at::Tensor moddown_ntt_rows(const at::Tensor& q_rows,
                            at::Tensor p_rows,
                            const std::optional<at::Tensor>& coefficient_c0,
                            const KeySwitchTables& tables,
                            std::string_view ntt_backend,
                            std::string_view rounding,
                            const at::Tensor& out,
                            const at::Tensor& scaled,
                            const at::Tensor& zero) {
  ntt::transform_(
      p_rows, tables.p_inverse_ntt, ntt_backend, "inverse_to_standard");
  auto correction =
      tables.moddown_crt.matrix.defined()
          ? rns::centered_quotient_correction(p_rows, tables.moddown_crt, out)
          : moddown_rows(zero.defined() ? zero.zero_().expand(q_rows.sizes())
                                        : rns::zero_residues(q_rows),
                         p_rows,
                         tables.moddown_inverses,
                         tables.qp_parameters,
                         optional_table(tables.moddown_half),
                         rounding,
                         out,
                         p_rows);
  const auto& q_parameters = tables.q_parameters;
  if (coefficient_c0.has_value()) {
    auto component = correction.select(0, 0);
    rns::add_standard_(component, *coefficient_c0, q_parameters);
  }
  return forward_add_scaled(correction,
                            q_rows,
                            tables.p_inverse,
                            q_parameters,
                            tables.q_forward,
                            ntt_backend,
                            true,
                            out,
                            scaled);
}

}  // namespace fhelium::detail_operations

namespace fhelium::rns {

namespace {

at::Tensor moddown_ntt_qp_to_q_into(
    const at::Tensor& source,
    const std::optional<at::Tensor>& coefficient_c0,
    const KeySwitchTables& tables,
    std::string_view ntt_backend,
    int64_t p_count,
    std::string_view rounding,
    const at::Tensor& out) {
  const auto q_count = source.size(-2) - p_count;
  return detail_operations::moddown_ntt_rows(
      source.narrow(-2, 0, q_count),
      source.narrow(-2, q_count, p_count).clone(),
      coefficient_c0,
      tables,
      ntt_backend,
      rounding,
      out);
}

// Temporaries of moddown_add_rescale; see `plan_moddown_add_rescale`.
struct ModdownAddRescalePlan {
  WorkspaceTensor numerator;  // P a + v over the Q rows
  WorkspaceTensor dropped;    // the D rows and the P rows
  WorkspaceTensor selected;   // remaining rows gathered by position
  WorkspaceTensor placed;     // remaining rows placed in B'
  WorkspaceTensor restored;   // R v
  WorkspaceTensor quotient;   // centered quotient by P on the D rows
  WorkspaceTensor quotient_term;
  WorkspaceTensor zero;
  WorkspaceTensor moddown;      // exact quotient by D over B'
  WorkspaceTensor scaled;       // remaining rows times product_inverse
  WorkspaceTensor coefficient;  // remaining rows in the coefficient domain
  WorkspaceTensor coefficient_term;
  int64_t bytes = 0;
};

ModdownAddRescalePlan plan_moddown_add_rescale(const at::Tensor& products,
                                               const at::Tensor& base,
                                               const KeySwitchTables& tables,
                                               std::string_view ntt_backend,
                                               int64_t q_count,
                                               int64_t joint_drop_count,
                                               bool joint_drop_is_prefix,
                                               std::string_view output_domain) {
  ModdownAddRescalePlan plan;
  const auto dtype = products.scalar_type();
  const auto with_rows = [](std::vector<int64_t> shape, int64_t rows) {
    shape[shape.size() - 2] = rows;
    return shape;
  };
  auto numerator = base.sizes().vec();
  numerator[0] = products.size(0);
  const auto q_rows = numerator[numerator.size() - 2];
  const auto p_rows = products.size(-2) - q_count;
  const auto& joint = tables.joint;
  const auto target_rows = joint.remaining_parameters.size(-1);
  WorkspaceLayout layout;
  plan.numerator = layout.take(numerator, dtype);
  plan.dropped = layout.take(
      with_rows(products.sizes().vec(), joint_drop_count + p_rows), dtype);
  auto remaining = with_rows(numerator, q_rows - joint_drop_count);
  if (!joint_drop_is_prefix) plan.selected = layout.take(remaining, dtype);
  if (joint.target_keep_positions.defined()) {
    remaining = with_rows(numerator, target_rows);
    plan.placed = layout.take(remaining, dtype);
  }
  const auto start = layout.mark();
  if (joint.restore.defined())
    plan.restored = layout.take(products.sizes().vec(), dtype);
  layout.release(start);
  const auto target = with_rows(numerator, target_rows);
  if (joint.crt_t.matrix.defined()) {
    plan.quotient =
        layout.take(with_rows(products.sizes().vec(),
                              joint.crt_d.target_parameters.size(-1)),
                    dtype);
    const auto term = layout.mark();
    plan.quotient_term = layout.take(plan.quotient.sizes, dtype);
    layout.release(term);
    plan.zero = layout.take({1}, dtype);
    plan.moddown = layout.take(target, dtype);
    layout.release(start);
  }
  if (output_domain == "coefficient") {
    plan.coefficient = layout.take(remaining, dtype);
    if (joint.crt_t.matrix.defined())
      plan.coefficient_term = layout.take(remaining, dtype);
  } else if (detail_operations::ntt_policy(ntt_backend).family !=
             NttFamily::kCompact) {
    plan.scaled = layout.take(remaining, dtype);
  }
  plan.bytes = layout.bytes();
  return plan;
}

// Temporaries of `plan` are views of a defined `workspace` and are allocated
// otherwise.
at::Tensor moddown_add_rescale_into(const at::Tensor& products,
                                    const at::Tensor& base,
                                    const KeySwitchTables& tables,
                                    std::string_view ntt_backend,
                                    int64_t q_count,
                                    int64_t joint_drop_count,
                                    bool joint_drop_is_prefix,
                                    std::string_view output_domain,
                                    const at::Tensor& out,
                                    const ModdownAddRescalePlan& plan,
                                    const at::Tensor& workspace) {
  const auto temporary = [&](const WorkspaceTensor& tensor) {
    return workspace.defined()
               ? tensor.in(workspace)
               : at::empty(tensor.sizes,
                           products.options().dtype(tensor.dtype));
  };
  const auto& joint = tables.joint;
  const auto& q_parameters = tables.q_parameters;
  const auto restore = optional_table(joint.restore);
  auto scaled_products = products;
  if (restore.has_value()) {
    scaled_products = temporary(plan.restored);
    montgomery_mul_row_scalars_standard_out(
        products, *restore, joint.product_parameters, scaled_products);
  }
  auto numerator_q = temporary(plan.numerator);
  montgomery_mul_row_scalars_standard_out(base.narrow(0, 0, products.size(0)),
                                          joint.p_times_base,
                                          q_parameters,
                                          numerator_q);
  add_standard_(
      numerator_q, scaled_products.narrow(-2, 0, q_count), q_parameters);
  auto dropped = temporary(plan.dropped);
  auto dropped_q = dropped.narrow(-2, 0, joint_drop_count);
  auto remaining = numerator_q.narrow(
      -2, joint_drop_count, numerator_q.size(-2) - joint_drop_count);
  if (joint_drop_is_prefix) {
    dropped_q.copy_(numerator_q.narrow(-2, 0, joint_drop_count));
  } else {
    at::index_select_out(dropped_q, numerator_q, -2, joint.drop_positions);
    remaining = temporary(plan.selected);
    at::index_select_out(remaining, numerator_q, -2, joint.remaining_positions);
  }
  dropped.narrow(-2, joint_drop_count, products.size(-2) - q_count)
      .copy_(scaled_products.narrow(-2, q_count, products.size(-2) - q_count));
  ntt::transform_(
      dropped, joint.dropped_inverse, ntt_backend, "inverse_to_standard");
  const auto& remaining_parameters = joint.remaining_parameters;
  if (const auto kept = optional_table(joint.target_keep_positions)) {
    auto placed = temporary(plan.placed);
    placed.zero_();
    placed.index_copy_(-2, *kept, remaining);
    remaining = placed;
  }
  const auto coefficient_remaining = [&] {
    auto coefficient = temporary(plan.coefficient);
    coefficient.copy_(remaining);
    ntt::transform_(coefficient,
                    joint.remaining_inverse,
                    ntt_backend,
                    "inverse_to_standard");
    return coefficient;
  };
  if (joint.crt_t.matrix.defined()) {
    // Centered quotient by P on the D rows, y = (x - L_P(x)) / P, then the
    // exact quotient of y by D; the target rows combine both corrections.
    const auto p_rows = dropped.narrow(
        -2, joint_drop_count, dropped.size(-2) - joint_drop_count);
    const auto& d_parameters = joint.crt_d.target_parameters;
    auto quotient = centered_quotient_correction(
        p_rows, joint.crt_d, temporary(plan.quotient));
    auto quotient_term = temporary(plan.quotient_term);
    montgomery_mul_row_scalars_standard_out(
        dropped_q, joint.crt_p_inverse, d_parameters, quotient_term);
    add_standard_(quotient, quotient_term, d_parameters);
    auto correction = centered_quotient_correction(p_rows, joint.crt_t, out);
    const auto half = optional_table(joint.crt_half);
    // The quotient is consumed as the mixed-radix digit scratch.
    auto moddown = temporary(plan.moddown);
    ckks::keyswitch_moddown_qp_to_q_out(
        temporary(plan.zero).zero_().expand(correction.sizes()),
        quotient,
        joint.crt_inverses,
        joint.crt_ordered_parameters,
        half,
        half.has_value() ? "nearest" : "floor",
        moddown,
        quotient);
    add_standard_(correction, moddown, remaining_parameters);
    if (output_domain == "coefficient") {
      auto term = temporary(plan.coefficient_term);
      montgomery_mul_row_scalars_standard_out(coefficient_remaining(),
                                              joint.product_inverse,
                                              remaining_parameters,
                                              term);
      add_standard_(correction, term, remaining_parameters);
      return correction;
    }
    return forward_add_scaled(
        correction,
        remaining,
        joint.product_inverse,
        remaining_parameters,
        joint.remaining_forward,
        ntt_backend,
        false,
        out,
        plan.scaled.defined() ? temporary(plan.scaled) : at::Tensor());
  }
  add_standard_(dropped, joint.dropped_offset, joint.dropped_parameters);
  if (output_domain == "coefficient") {
    auto coefficient = coefficient_remaining();
    add_standard_(coefficient, joint.remaining_offset, remaining_parameters);
    return moddown_rows(coefficient,
                        dropped,
                        joint.inverses,
                        joint.ordered_parameters,
                        std::nullopt,
                        "floor",
                        out);
  }
  std::vector<int64_t> offset_shape(dropped.sizes().begin(),
                                    dropped.sizes().end() - 2);
  offset_shape.push_back(-1);
  offset_shape.push_back(-1);
  const auto offset = joint.remaining_offset.expand(offset_shape);
  auto correction = moddown_rows(offset,
                                 dropped,
                                 joint.inverses,
                                 joint.ordered_parameters,
                                 std::nullopt,
                                 "floor",
                                 out);
  return forward_add_scaled(
      correction,
      remaining,
      joint.product_inverse,
      remaining_parameters,
      joint.remaining_forward,
      ntt_backend,
      false,
      out,
      plan.scaled.defined() ? temporary(plan.scaled) : at::Tensor());
}

}  // namespace

at::Tensor hybrid_modup_digit(const at::Tensor& source,
                              const KeySwitchTables& tables,
                              int64_t digit_index,
                              c10::IntArrayRef digits) {
  return hybrid_modup_digit_into(
      source, tables, digit_index, digits, at::Tensor());
}

void hybrid_modup_digit_out(const at::Tensor& source,
                            const KeySwitchTables& tables,
                            int64_t digit_index,
                            c10::IntArrayRef digits,
                            at::Tensor& out) {
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, source),
      "hybrid_modup_digit_out requires out disjoint from its operands");
  finish_out(hybrid_modup_digit_into(source, tables, digit_index, digits, out),
             out);
}

at::Tensor centered_basis_extend(const at::Tensor& source,
                                 const BasisExtensionTables& tables) {
  return centered_basis_extend_into(source,
                                    tables,
                                    at::Tensor(),
                                    plan_centered_basis_extend(source, tables),
                                    at::Tensor());
}

void centered_basis_extend_out(const at::Tensor& source,
                               const BasisExtensionTables& tables,
                               at::Tensor& out,
                               const at::Tensor& workspace) {
  const auto plan = plan_centered_basis_extend(source, tables);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    fhelium::detail::check_workspace(
        workspace, plan.bytes, source, "centered_basis_extend_out");
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, source) &&
          (!with_workspace ||
           (fhelium::detail::storage_disjoint(out, workspace) &&
            fhelium::detail::storage_disjoint(workspace, source))),
      "centered_basis_extend_out requires out, workspace and operands to be "
      "disjoint");
  finish_out(
      centered_basis_extend_into(
          source, tables, out, plan, with_workspace ? workspace : at::Tensor()),
      out);
}

int64_t centered_basis_extend_workspace_bytes(
    const at::Tensor& source, const BasisExtensionTables& tables) {
  return plan_centered_basis_extend(source, tables).bytes;
}

at::Tensor rounded_trailing_quotient(const at::Tensor& source,
                                     const TrailingQuotientTables& tables,
                                     int64_t drop_count,
                                     std::string_view rounding) {
  return rounded_trailing_quotient_into(
      source,
      tables,
      drop_count,
      rounding,
      at::Tensor(),
      plan_rounded_trailing_quotient(source, tables, drop_count),
      at::Tensor());
}

void rounded_trailing_quotient_out(const at::Tensor& source,
                                   const TrailingQuotientTables& tables,
                                   int64_t drop_count,
                                   std::string_view rounding,
                                   at::Tensor& out,
                                   const at::Tensor& workspace) {
  const auto plan = plan_rounded_trailing_quotient(source, tables, drop_count);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    fhelium::detail::check_workspace(
        workspace, plan.bytes, source, "rounded_trailing_quotient_out");
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, source) &&
          (!with_workspace ||
           (fhelium::detail::storage_disjoint(out, workspace) &&
            fhelium::detail::storage_disjoint(workspace, source))),
      "rounded_trailing_quotient_out requires out, workspace and operands to "
      "be disjoint");
  finish_out(
      rounded_trailing_quotient_into(source,
                                     tables,
                                     drop_count,
                                     rounding,
                                     out,
                                     plan,
                                     with_workspace ? workspace : at::Tensor()),
      out);
}

int64_t rounded_trailing_quotient_workspace_bytes(
    const at::Tensor& source,
    const TrailingQuotientTables& tables,
    int64_t drop_count,
    std::string_view rounding) {
  return plan_rounded_trailing_quotient(source, tables, drop_count).bytes;
}

at::Tensor moddown_qp_to_q(const at::Tensor& source,
                           const KeySwitchTables& tables,
                           int64_t p_count,
                           std::string_view rounding) {
  return moddown_qp_to_q_into(source,
                              tables,
                              p_count,
                              rounding,
                              at::Tensor(),
                              plan_moddown_qp_to_q(source, tables, p_count),
                              at::Tensor());
}

void moddown_qp_to_q_out(const at::Tensor& source,
                         const KeySwitchTables& tables,
                         int64_t p_count,
                         std::string_view rounding,
                         at::Tensor& out,
                         const at::Tensor& workspace) {
  const auto plan = plan_moddown_qp_to_q(source, tables, p_count);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    fhelium::detail::check_workspace(
        workspace, plan.bytes, source, "moddown_qp_to_q_out");
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, source) &&
          (!with_workspace ||
           (fhelium::detail::storage_disjoint(out, workspace) &&
            fhelium::detail::storage_disjoint(workspace, source))),
      "moddown_qp_to_q_out requires out, workspace and operands to be "
      "disjoint");
  finish_out(moddown_qp_to_q_into(source,
                                  tables,
                                  p_count,
                                  rounding,
                                  out,
                                  plan,
                                  with_workspace ? workspace : at::Tensor()),
             out);
}

int64_t moddown_qp_to_q_workspace_bytes(const at::Tensor& source,
                                        const KeySwitchTables& tables,
                                        int64_t p_count,
                                        std::string_view rounding) {
  return plan_moddown_qp_to_q(source, tables, p_count).bytes;
}

at::Tensor moddown_ntt_qp_to_q(const at::Tensor& source,
                               const std::optional<at::Tensor>& coefficient_c0,
                               const KeySwitchTables& tables,
                               std::string_view ntt_backend,
                               int64_t p_count,
                               std::string_view rounding) {
  return moddown_ntt_qp_to_q_into(source,
                                  coefficient_c0,
                                  tables,
                                  ntt_backend,
                                  p_count,
                                  rounding,
                                  at::Tensor());
}

void moddown_ntt_qp_to_q_out(const at::Tensor& source,
                             const std::optional<at::Tensor>& coefficient_c0,
                             const KeySwitchTables& tables,
                             std::string_view ntt_backend,
                             int64_t p_count,
                             std::string_view rounding,
                             at::Tensor& out) {
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, source) &&
          (!coefficient_c0.has_value() ||
           fhelium::detail::storage_disjoint(out, *coefficient_c0)),
      "moddown_ntt_qp_to_q_out requires out disjoint from its operands");
  finish_out(
      moddown_ntt_qp_to_q_into(
          source, coefficient_c0, tables, ntt_backend, p_count, rounding, out),
      out);
}

at::Tensor moddown_add_rescale(const at::Tensor& products,
                               const at::Tensor& base,
                               const KeySwitchTables& tables,
                               std::string_view ntt_backend,
                               int64_t q_count,
                               int64_t joint_drop_count,
                               bool joint_drop_is_prefix,
                               std::string_view output_domain) {
  return moddown_add_rescale_into(products,
                                  base,
                                  tables,
                                  ntt_backend,
                                  q_count,
                                  joint_drop_count,
                                  joint_drop_is_prefix,
                                  output_domain,
                                  at::Tensor(),
                                  plan_moddown_add_rescale(products,
                                                           base,
                                                           tables,
                                                           ntt_backend,
                                                           q_count,
                                                           joint_drop_count,
                                                           joint_drop_is_prefix,
                                                           output_domain),
                                  at::Tensor());
}

int64_t moddown_add_rescale_workspace_bytes(const at::Tensor& products,
                                            const at::Tensor& base,
                                            const KeySwitchTables& tables,
                                            std::string_view ntt_backend,
                                            int64_t q_count,
                                            int64_t joint_drop_count,
                                            bool joint_drop_is_prefix,
                                            std::string_view output_domain) {
  return plan_moddown_add_rescale(products,
                                  base,
                                  tables,
                                  ntt_backend,
                                  q_count,
                                  joint_drop_count,
                                  joint_drop_is_prefix,
                                  output_domain)
      .bytes;
}

void moddown_add_rescale_out(const at::Tensor& products,
                             const at::Tensor& base,
                             const KeySwitchTables& tables,
                             std::string_view ntt_backend,
                             int64_t q_count,
                             int64_t joint_drop_count,
                             bool joint_drop_is_prefix,
                             std::string_view output_domain,
                             at::Tensor& out,
                             const at::Tensor& workspace) {
  const auto plan = plan_moddown_add_rescale(products,
                                             base,
                                             tables,
                                             ntt_backend,
                                             q_count,
                                             joint_drop_count,
                                             joint_drop_is_prefix,
                                             output_domain);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    detail::check_workspace(
        workspace, plan.bytes, products, "moddown_add_rescale_out");
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, products) &&
          fhelium::detail::storage_disjoint(out, base) &&
          (!with_workspace ||
           (fhelium::detail::storage_disjoint(out, workspace) &&
            fhelium::detail::storage_disjoint(workspace, products) &&
            fhelium::detail::storage_disjoint(workspace, base))),
      "moddown_add_rescale_out requires out, workspace and operands to be "
      "disjoint");
  finish_out(
      moddown_add_rescale_into(products,
                               base,
                               tables,
                               ntt_backend,
                               q_count,
                               joint_drop_count,
                               joint_drop_is_prefix,
                               output_domain,
                               out,
                               plan,
                               with_workspace ? workspace : at::Tensor()),
      out);
}

at::Tensor rational_rescale(const at::Tensor& source,
                            const RescaleTables& tables,
                            std::string_view input_domain,
                            std::string_view ntt_backend) {
  return rational_rescale_into(
      source,
      tables,
      input_domain,
      ntt_backend,
      at::Tensor(),
      plan_rational_rescale(source, tables, input_domain, ntt_backend),
      at::Tensor());
}

void rational_rescale_out(const at::Tensor& source,
                          const RescaleTables& tables,
                          std::string_view input_domain,
                          std::string_view ntt_backend,
                          at::Tensor& out,
                          const at::Tensor& workspace) {
  const auto plan =
      plan_rational_rescale(source, tables, input_domain, ntt_backend);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    fhelium::detail::check_workspace(
        workspace, plan.bytes, source, "rational_rescale_out");
  TORCH_CHECK_VALUE(
      fhelium::detail::storage_disjoint(out, source) &&
          (!with_workspace ||
           (fhelium::detail::storage_disjoint(out, workspace) &&
            fhelium::detail::storage_disjoint(workspace, source))),
      "rational_rescale_out requires out, workspace and operands to be "
      "disjoint");
  finish_out(rational_rescale_into(source,
                                   tables,
                                   input_domain,
                                   ntt_backend,
                                   out,
                                   plan,
                                   with_workspace ? workspace : at::Tensor()),
             out);
}

int64_t rational_rescale_workspace_bytes(const at::Tensor& source,
                                         const RescaleTables& tables,
                                         std::string_view input_domain,
                                         std::string_view ntt_backend) {
  return plan_rational_rescale(source, tables, input_domain, ntt_backend).bytes;
}

}  // namespace fhelium::rns
