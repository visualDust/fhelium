#include <ATen/ATen.h>
#include <fhelium/ckks.h>
#include <fhelium/ckks_operations.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_operations.h>
#include <fhelium/rns.h>
#include <fhelium/rns_operations.h>

#include <string>
#include <vector>

#include "../common/overlap.h"
#include "../common/workspace.h"
#include "composition.h"

namespace fhelium::ckks {

using detail::storage_disjoint;
using detail::WorkspaceLayout;
using detail::WorkspaceTensor;
using detail_operations::finish_out;

namespace {

using detail_operations::DigitSpan;
using detail_operations::NttFamily;

using detail_operations::optional_table;

// NTT-domain gather of one rotation; see `GaloisTables`.
const at::Tensor& ntt_indices(const GaloisTables& galois) {
  TORCH_CHECK_VALUE(galois.ntt_indices.defined(),
                    "An NTT-domain rotation requires Galois ntt_indices");
  return galois.ntt_indices;
}

at::Tensor rotate_coefficients(const at::Tensor& source,
                               const GaloisTables& galois,
                               const KeySwitchTables& tables,
                               const at::Tensor& out = at::Tensor()) {
  TORCH_CHECK_VALUE(galois.coefficient_indices.defined() &&
                        galois.coefficient_signs.defined(),
                    "A coefficient-domain rotation requires Galois "
                    "coefficient_indices and coefficient_signs");
  if (out.defined()) {
    auto target = out;
    apply_coefficient_galois_automorphism_out(source,
                                              galois.coefficient_indices,
                                              galois.coefficient_signs,
                                              tables.q_parameters.select(0, 0),
                                              target);
    return target;
  }
  return apply_coefficient_galois_automorphism(
      source,
      galois.coefficient_indices,
      galois.coefficient_signs,
      tables.q_parameters.select(0, 0));
}

// Galois tables of steps[i], one per step.
void check_galois_count(c10::ArrayRef<GaloisTables> galois, size_t steps) {
  TORCH_CHECK_VALUE(galois.size() == steps,
                    "Rotations require one Galois table set per step; got ",
                    galois.size(),
                    " for ",
                    steps,
                    " steps");
}

// Seed row of digit `digit` of key `key_index`, or none for dense keys.
std::optional<at::Tensor> key_seed(const KeySwitchOptions& options,
                                   size_t key_index,
                                   int64_t digit) {
  if (options.key_seeds.empty()) return std::nullopt;
  return options.key_seeds[key_index].select(0, digit);
}

// Options of a single-key call that uses key `key_index` of `options`.
KeySwitchOptions for_key(const KeySwitchOptions& options, size_t key_index) {
  auto selected = options;
  if (!options.key_seeds.empty()) {
    selected.key_seeds = {options.key_seeds[key_index]};
  }
  return selected;
}

// Accumulates one digit's key products. With `key_q_rows`, active Q rows
// read the listed key rows: each maximal run of consecutive key rows, and the
// trailing auxiliary rows at `key_p_row_start`, is one contiguous product.
void accumulate_digit_products(at::Tensor& accumulator0,
                               at::Tensor& accumulator1,
                               const at::Tensor& digit,
                               const at::Tensor& key_digit,
                               const at::Tensor& parameters,
                               const KeySwitchOptions& options,
                               const std::optional<at::Tensor>& indices,
                               const std::optional<at::Tensor>& seed) {
  const auto& rows = options.key_q_rows;
  if (rows.empty()) {
    keyswitch_accumulate_digit_products_(accumulator0,
                                         accumulator1,
                                         digit,
                                         key_digit,
                                         parameters,
                                         options.key_row_start,
                                         options.key_p_row_start,
                                         indices,
                                         seed,
                                         options.key_rounds);
    return;
  }
  const auto q_rows = static_cast<int64_t>(rows.size());
  const auto operand_rows = digit.size(-2);
  TORCH_CHECK_VALUE(options.key_p_row_start >= 0 && q_rows < operand_rows,
                    "key_q_rows requires trailing auxiliary key rows");
  auto product = [&](int64_t start, int64_t count, int64_t key_start) {
    auto out0 = accumulator0.narrow(-2, start, count);
    auto out1 = accumulator1.narrow(-2, start, count);
    keyswitch_accumulate_digit_products_(out0,
                                         out1,
                                         digit.narrow(-2, start, count),
                                         key_digit,
                                         parameters.narrow(1, start, count),
                                         key_start,
                                         -1,
                                         indices,
                                         seed,
                                         options.key_rounds);
  };
  int64_t start = 0;
  for (int64_t row = 1; row <= q_rows; ++row) {
    if (row == q_rows || rows[row] != rows[row - 1] + 1) {
      product(start, row - start, rows[start]);
      start = row;
    }
  }
  product(q_rows, operand_rows - q_rows, options.key_p_row_start);
}

at::Tensor moddown_coefficient(at::Tensor accumulator,
                               const KeySwitchTables& tables,
                               const KeySwitchOptions& options,
                               const at::Tensor& out,
                               const at::Tensor& workspace = at::Tensor()) {
  ntt::transform_(accumulator,
                  tables.qp_inverse,
                  options.ntt_backend,
                  "inverse_to_standard");
  if (out.defined() || workspace.defined()) {
    auto target =
        out.defined()
            ? out
            : at::empty(
                  accumulator
                      .narrow(-2, 0, accumulator.size(-2) - options.p_count)
                      .sizes(),
                  accumulator.options());
    rns::moddown_qp_to_q_out(accumulator,
                             tables,
                             options.p_count,
                             options.moddown_rounding,
                             target,
                             workspace);
    return target;
  }
  return rns::moddown_qp_to_q(
      accumulator, tables, options.p_count, options.moddown_rounding);
}

// NTT ModDown of a QP accumulator owned by the caller; its P rows are
// overwritten.
at::Tensor moddown_ntt_owned(const at::Tensor& accumulator,
                             const std::optional<at::Tensor>& coefficient_c0,
                             const KeySwitchTables& tables,
                             const KeySwitchOptions& options,
                             const at::Tensor& out,
                             const at::Tensor& scaled = at::Tensor(),
                             const at::Tensor& zero = at::Tensor()) {
  const auto q_count = accumulator.size(-2) - options.p_count;
  return detail_operations::moddown_ntt_rows(
      accumulator.narrow(-2, 0, q_count),
      accumulator.narrow(-2, q_count, options.p_count),
      coefficient_c0,
      tables,
      options.ntt_backend,
      options.moddown_rounding,
      out,
      scaled,
      zero);
}

at::Tensor corrections(const at::Tensor& component,
                       const at::Tensor& key,
                       const KeySwitchTables& tables,
                       const KeySwitchOptions& options,
                       bool output_ntt,
                       const std::optional<at::Tensor>& coefficient_c0,
                       const at::Tensor& out) {
  auto accumulator = key_switch_products(component, key, tables, options);
  if (output_ntt) {
    return moddown_ntt_owned(accumulator, coefficient_c0, tables, options, out);
  }
  return moddown_coefficient(accumulator, tables, options, out);
}

at::Tensor forward_copy(const at::Tensor& source,
                        const KeySwitchTables& tables,
                        const KeySwitchOptions& options,
                        const at::Tensor& out = at::Tensor()) {
  if (out.defined()) {
    auto target = out;
    ntt::transform_out(source,
                       tables.q_forward,
                       options.ntt_backend,
                       "forward_to_montgomery",
                       target);
    return target;
  }
  return ntt::transform(
      source, tables.q_forward, options.ntt_backend, "forward_to_montgomery");
}

// Lifts every digit of `component` once to QP NTT Montgomery residues,
// shape [digit, *batch, limb, N].
// Key-switching products of the hoisted digits for every rotation, shape
// [rotation, 2, *batch, limb, N], in `buffer` when it is defined.
at::Tensor accumulate_rotations(const at::Tensor& prepared,
                                at::TensorList keys,
                                at::TensorList seeds,
                                c10::ArrayRef<GaloisTables> galois,
                                const KeySwitchTables& tables,
                                const KeySwitchOptions& options,
                                const at::Tensor& buffer = at::Tensor()) {
  const auto prototype = prepared.select(0, 0);
  auto shape = prototype.sizes().vec();
  shape.insert(shape.begin(), {static_cast<int64_t>(galois.size()), 2});
  const auto spans = detail_operations::digit_spans(options.digits);
  const auto& parameters = tables.qp_parameters;
  bool ordered = true;
  for (size_t i = 0; i < spans.size(); ++i) {
    ordered = ordered && spans[i].key_digit == static_cast<int64_t>(i);
  }
  // The grouped product kernel overwrites every accumulator and reads the
  // digits once for all rotations; the per-digit path accumulates into zeros.
  const bool grouped =
      prototype.is_cuda() && ordered && options.key_q_rows.empty();
  at::Tensor accumulators;
  if (buffer.defined()) {
    accumulators = buffer;
    if (!grouped) accumulators.zero_();
  } else {
    accumulators = grouped ? at::empty(shape, prototype.options())
                           : at::zeros(shape, prototype.options());
  }
  std::vector<at::Tensor> accumulators0, accumulators1, indices, narrowed,
      narrowed_seeds;
  for (size_t r = 0; r < galois.size(); ++r) {
    accumulators0.push_back(accumulators[r][0]);
    accumulators1.push_back(accumulators[r][1]);
    indices.push_back(ntt_indices(galois[r]));
  }
  if (grouped) {
    for (const auto& key : keys)
      narrowed.push_back(key.narrow(0, 0, prepared.size(0)));
    for (const auto& seed : seeds)
      narrowed_seeds.push_back(seed.narrow(0, 0, prepared.size(0)));
    keyswitch_accumulate_products_(accumulators0,
                                   accumulators1,
                                   prepared.unbind(0),
                                   narrowed,
                                   parameters,
                                   options.key_row_start,
                                   options.key_p_row_start,
                                   indices,
                                   narrowed_seeds,
                                   options.key_rounds);
    return accumulators;
  }
  for (size_t r = 0; r < galois.size(); ++r) {
    for (size_t i = 0; i < spans.size(); ++i) {
      accumulate_digit_products(accumulators0[r],
                                accumulators1[r],
                                prepared.select(0, static_cast<int64_t>(i)),
                                keys[r].select(0, spans[i].key_digit),
                                parameters,
                                options,
                                indices[r],
                                seeds.empty()
                                    ? std::nullopt
                                    : std::optional<at::Tensor>(seeds[r].select(
                                          0, spans[i].key_digit)));
    }
  }
  return accumulators;
}

// Completes one hoisted rotation from its QP products: ModDown and the
// rotated c0 term. Defined `scaled` and `rotated` hold the temporaries of
// the NTT and coefficient outputs.
// Temporaries of `finish_rotation`: the lazy-scaling copy and the zero Q
// residues of the NTT ModDown, or the rotated c0 and the ModDown workspace
// of the coefficient output.
struct FinishRotationPlan {
  WorkspaceTensor scaled;
  WorkspaceTensor zero;
  WorkspaceTensor rotated;
  WorkspaceTensor moddown;
};

struct FinishRotationBuffers {
  at::Tensor scaled;
  at::Tensor zero;
  at::Tensor rotated;
  at::Tensor moddown;
};

// Takes the temporaries of one rotation of Q components shaped `component`.
FinishRotationPlan plan_finish_rotation(WorkspaceLayout& layout,
                                        const std::vector<int64_t>& component,
                                        const at::Tensor& like,
                                        const KeySwitchTables& tables,
                                        const KeySwitchOptions& options) {
  FinishRotationPlan plan;
  const auto dtype = like.scalar_type();
  auto result = component;
  result.insert(result.begin(), 2);
  if (options.output_domain == "ntt") {
    if (detail_operations::ntt_policy(options.ntt_backend).family !=
        NttFamily::kCompact)
      plan.scaled = layout.take(result, dtype);
    if (!tables.moddown_crt.matrix.defined())
      plan.zero = layout.take({1}, dtype);
    return plan;
  }
  plan.rotated = layout.take(component, dtype);
  auto qp = result;
  qp[qp.size() - 2] = tables.qp_parameters.size(1);
  const auto bytes = rns::moddown_qp_to_q_workspace_bytes(
      at::empty(qp, like.options().device(at::kMeta)),
      tables,
      options.p_count,
      options.moddown_rounding);
  if (bytes > 0) plan.moddown = layout.take({bytes}, at::kByte);
  return plan;
}

FinishRotationBuffers finish_rotation_buffers(const FinishRotationPlan& plan,
                                              const at::Tensor& workspace) {
  if (!workspace.defined()) return {};
  const auto view = [&](const WorkspaceTensor& tensor) {
    return tensor.defined() ? tensor.in(workspace) : at::Tensor();
  };
  return {view(plan.scaled),
          view(plan.zero),
          view(plan.rotated),
          view(plan.moddown)};
}

// The exact-lift scratch `scratch`, or undefined when the fused CUDA lift
// applies to digits of its width.
at::Tensor lift_scratch(const at::Tensor& scratch,
                        const at::Tensor& like,
                        const KeySwitchTables& tables) {
  if (scratch.defined() && detail_operations::uses_tiled_modup(
                               like, tables.qp_parameters, scratch.size(-2)))
    return at::Tensor();
  return scratch;
}

at::Tensor finish_rotation(const at::Tensor& c0,
                           at::Tensor accumulator,
                           const GaloisTables& galois,
                           const KeySwitchTables& tables,
                           const KeySwitchOptions& options,
                           const at::Tensor& out,
                           const FinishRotationBuffers& buffers = {}) {
  const auto& q_parameters = tables.q_parameters;
  if (options.output_domain == "ntt") {
    auto result = moddown_ntt_owned(accumulator,
                                    std::nullopt,
                                    tables,
                                    options,
                                    out,
                                    buffers.scaled,
                                    buffers.zero);
    auto component0 = result.select(0, 0);
    add_ntt_galois_automorphism_(
        component0, c0, ntt_indices(galois), q_parameters);
    return result;
  }
  auto result =
      moddown_coefficient(accumulator, tables, options, out, buffers.moddown);
  auto component0 = result.select(0, 0);
  rns::add_standard_(component0,
                     rotate_coefficients(c0, galois, tables, buffers.rotated),
                     q_parameters);
  return result;
}

// Each `*_into` function below computes the operation of the same name; a
// defined `out` receives the final producer's result.

at::Tensor rotation_digits_into(
    const at::Tensor& component,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const at::Tensor& out,
    const at::Tensor& scratch_buffer = at::Tensor()) {
  const auto spans = detail_operations::digit_spans(options.digits);
  TORCH_CHECK_VALUE(!spans.empty(),
                    "Rotation hoisting requires an active RNS digit");
  auto shape = component.sizes().vec();
  shape[shape.size() - 2] = tables.qp_parameters.size(1);
  shape.insert(shape.begin(), static_cast<int64_t>(spans.size()));
  auto digits = out.defined() ? out : at::empty(shape, component.options());
  int64_t width = 0;
  for (const auto& span : spans) {
    width = std::max(width, span.stop - span.start);
  }
  // The fused CUDA lift needs no scratch; the portable exact lift uses one.
  at::Tensor scratch = scratch_buffer;
  if (!scratch.defined() && width > 1 && options.basis_conversion == "exact" &&
      !detail_operations::uses_tiled_modup(
          component, tables.qp_parameters, width)) {
    auto scratch_shape = component.sizes().vec();
    scratch_shape[scratch_shape.size() - 2] = width;
    scratch = at::empty(scratch_shape, component.options());
  }
  for (size_t index = 0; index < spans.size(); ++index) {
    auto lifted = detail_operations::lift_key_switch_digit(
        component,
        tables,
        options.basis_conversion,
        static_cast<int64_t>(index),
        spans[index],
        digits.select(0, static_cast<int64_t>(index)),
        scratch);
    ntt::transform_(
        lifted, tables.qp_forward, options.ntt_backend, "forward_montgomery");
  }
  return digits;
}

// Workspace plan of `rotate_hoisted`: the QP accumulators and the
// temporaries of `finish_rotation`.
struct RotateHoistedPlan {
  WorkspaceTensor accumulator;
  FinishRotationPlan finish;
  int64_t bytes = 0;
};

RotateHoistedPlan plan_rotate_hoisted(const at::Tensor& c0,
                                      const at::Tensor& hoisted_digits,
                                      const KeySwitchTables& tables,
                                      const KeySwitchOptions& options) {
  RotateHoistedPlan plan;
  WorkspaceLayout layout;
  auto accumulator = hoisted_digits.sizes().vec();
  accumulator[0] = 2;
  accumulator.insert(accumulator.begin(), 1);
  plan.accumulator = layout.take(accumulator, hoisted_digits.scalar_type());
  plan.finish =
      plan_finish_rotation(layout, c0.sizes().vec(), c0, tables, options);
  plan.bytes = layout.bytes();
  return plan;
}

at::Tensor rotate_hoisted_into(const at::Tensor& c0,
                               const at::Tensor& hoisted_digits,
                               const at::Tensor& key,
                               const GaloisTables& galois,
                               const KeySwitchTables& tables,
                               const KeySwitchOptions& options,
                               const at::Tensor& out,
                               const RotateHoistedPlan& plan = {},
                               const at::Tensor& workspace = at::Tensor()) {
  const auto accumulators = accumulate_rotations(
      hoisted_digits,
      {key},
      options.key_seeds,
      {galois},
      tables,
      options,
      workspace.defined() ? plan.accumulator.in(workspace) : at::Tensor());
  return finish_rotation(c0,
                         accumulators[0],
                         galois,
                         tables,
                         options,
                         out,
                         finish_rotation_buffers(plan.finish, workspace));
}

}  // namespace

namespace {

// key_switch_products; defined `digit` and `scratch` hold each lifted digit
// and its exact-lift scratch.
at::Tensor key_switch_products_into(const at::Tensor& component,
                                    const at::Tensor& key,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options,
                                    at::Tensor result,
                                    const at::Tensor& digit = at::Tensor(),
                                    const at::Tensor& scratch = at::Tensor()) {
  const auto policy = detail_operations::ntt_policy(options.ntt_backend);
  const auto spans = detail_operations::digit_spans(options.digits);
  TORCH_CHECK_VALUE(!spans.empty(),
                    "Key switching requires at least one RNS digit");
  const auto& parameters = tables.qp_parameters;
  for (size_t index = 0; index < spans.size(); ++index) {
    const auto& span = spans[index];
    auto lifted =
        detail_operations::lift_key_switch_digit(component,
                                                 tables,
                                                 options.basis_conversion,
                                                 static_cast<int64_t>(index),
                                                 span,
                                                 digit,
                                                 scratch);
    if (!result.defined()) {
      auto shape = lifted.sizes().vec();
      shape.insert(shape.begin(), 2);
      result = at::zeros(shape, lifted.options());
    }
    auto accumulator0 = result.select(0, 0);
    auto accumulator1 = result.select(0, 1);
    const auto key_digit = key.select(0, span.key_digit);
    const auto seed = key_seed(options, 0, span.key_digit);
    if (policy.family == NttFamily::kCompact && options.key_q_rows.empty() &&
        !seed.has_value()) {
      ntt::forward_ntt_montgomery_compact_keyswitch_accumulate_(
          lifted,
          tables.qp_forward.schedule[0],
          parameters,
          key_digit,
          accumulator0,
          accumulator1,
          options.key_row_start,
          options.key_p_row_start);
    } else {
      ntt::transform_(
          lifted, tables.qp_forward, options.ntt_backend, "forward_montgomery");
      accumulate_digit_products(accumulator0,
                                accumulator1,
                                lifted,
                                key_digit,
                                parameters,
                                options,
                                std::nullopt,
                                seed);
    }
  }
  return result;
}

}  // namespace

at::Tensor key_switch_products(const at::Tensor& component,
                               const at::Tensor& key,
                               const KeySwitchTables& tables,
                               const KeySwitchOptions& options,
                               const std::optional<at::Tensor>& accumulator) {
  return key_switch_products_into(
      component, key, tables, options, accumulator.value_or(at::Tensor()));
}

namespace {

at::Tensor key_switch_into(const at::Tensor& source,
                           const at::Tensor& key,
                           const KeySwitchTables& tables,
                           const KeySwitchOptions& options,
                           const std::optional<GaloisTables>& conjugation,
                           const at::Tensor& out) {
  TORCH_CHECK_VALUE(source.size(0) == 2,
                    "Key switching requires ciphertext components=2");
  const auto input = conjugation.has_value()
                         ? rotate_coefficients(source, *conjugation, tables)
                         : source;
  const bool output_ntt = options.output_domain == "ntt";
  const auto result = corrections(input.select(0, 1),
                                  key,
                                  tables,
                                  options,
                                  output_ntt,
                                  input.select(0, 0),
                                  out);
  if (!output_ntt) {
    auto component0 = result.select(0, 0);
    rns::add_standard_(component0, input.select(0, 0), tables.q_parameters);
  }
  return result;
}

// Workspace plan of `relinearize`: the QP accumulators, then, released after
// the key products, the coefficient copy of c2, one lifted digit and the
// exact-lift scratch, and afterwards the ModDown temporaries: the workspace
// of the joint ModDown and rescale or of the coefficient ModDown, the lazy
// scaling copy of the NTT ModDown, and the coefficient copy of (c0, c1).
struct RelinearizePlan {
  WorkspaceTensor accumulator;
  WorkspaceTensor component;
  WorkspaceTensor digit;
  WorkspaceTensor scratch;
  WorkspaceTensor moddown;  // uint8 workspace of the ModDown operation
  WorkspaceTensor scaled;
  WorkspaceTensor zero;  // zero Q residues of the non-CRT NTT ModDown
  WorkspaceTensor coefficient;
  int64_t bytes = 0;
};

RelinearizePlan plan_relinearize(
    const at::Tensor& source,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale) {
  RelinearizePlan plan;
  const auto dtype = source.scalar_type();
  const auto component = source.sizes().slice(1).vec();
  const auto rows = component.size() - 2;
  auto qp = component;
  qp[rows] = tables.qp_parameters.size(1);
  auto accumulator = qp;
  accumulator.insert(accumulator.begin(), 2);
  WorkspaceLayout layout;
  plan.accumulator = layout.take(accumulator, dtype);
  const auto start = layout.mark();
  plan.component = layout.take(component, dtype);
  plan.digit = layout.take(qp, dtype);
  if (options.basis_conversion == "exact") {
    int64_t width = 0;
    for (const auto& span : detail_operations::digit_spans(options.digits))
      width = std::max(width, span.stop - span.start);
    // Reserved on every device so that Meta operands give the same size; the
    // fused CUDA lift does not use it.
    if (width > 1) {
      auto scratch = component;
      scratch[rows] = width;
      plan.scratch = layout.take(scratch, dtype);
    }
  }
  layout.release(start);
  const auto meta = at::empty(accumulator, source.options().device(at::kMeta));
  int64_t moddown = 0;
  if (rescale.has_value()) {
    moddown =
        rns::moddown_add_rescale_workspace_bytes(meta,
                                                 source,
                                                 tables,
                                                 options.ntt_backend,
                                                 rescale->q_count,
                                                 rescale->joint_drop_count,
                                                 rescale->joint_drop_is_prefix,
                                                 options.output_domain);
  } else if (options.output_domain != "ntt") {
    moddown = rns::moddown_qp_to_q_workspace_bytes(
        meta, tables, options.p_count, options.moddown_rounding);
  }
  if (moddown > 0) plan.moddown = layout.take({moddown}, at::kByte);
  if (!rescale.has_value()) {
    auto q = accumulator;
    q[rows + 1] = qp[rows] - options.p_count;
    if (options.output_domain == "ntt") {
      if (detail_operations::ntt_policy(options.ntt_backend).family !=
          NttFamily::kCompact)
        plan.scaled = layout.take(q, dtype);
      if (!tables.moddown_crt.matrix.defined())
        plan.zero = layout.take({1}, dtype);
    } else {
      plan.coefficient = layout.take(q, dtype);
    }
  }
  plan.bytes = layout.bytes();
  return plan;
}

// With `workspace`, the temporaries of `plan` are views of it; otherwise they
// are allocated.
at::Tensor relinearize_into(const at::Tensor& source,
                            const at::Tensor& key,
                            const KeySwitchTables& tables,
                            const KeySwitchOptions& options,
                            const std::optional<JointRescaleOptions>& rescale,
                            const at::Tensor& out,
                            const RelinearizePlan& plan,
                            const at::Tensor& workspace) {
  TORCH_CHECK_VALUE(source.size(0) == 3,
                    "Relinearization requires ciphertext components=3");
  const auto temporary = [&](const WorkspaceTensor& tensor) {
    return workspace.defined()
               ? tensor.in(workspace)
               : at::empty(tensor.sizes, source.options().dtype(tensor.dtype));
  };
  const auto view = [&](const WorkspaceTensor& tensor) {
    return workspace.defined() && tensor.defined() ? tensor.in(workspace)
                                                   : at::Tensor();
  };
  const bool output_ntt = options.output_domain == "ntt";
  auto component = temporary(plan.component);
  component.copy_(source.select(0, 2));
  ntt::transform_(
      component, tables.q_inverse, options.ntt_backend, "inverse_to_standard");
  auto products = temporary(plan.accumulator).zero_();

  key_switch_products_into(component,
                           key,
                           tables,
                           options,
                           products,
                           temporary(plan.digit),
                           lift_scratch(view(plan.scratch), source, tables));
  const auto moddown = view(plan.moddown);
  if (rescale.has_value()) {
    if (out.defined()) {
      auto target = out;
      rns::moddown_add_rescale_out(products,
                                   source,
                                   tables,
                                   options.ntt_backend,
                                   rescale->q_count,
                                   rescale->joint_drop_count,
                                   rescale->joint_drop_is_prefix,
                                   options.output_domain,
                                   target,
                                   moddown);
      return target;
    }
    return rns::moddown_add_rescale(products,
                                    source,
                                    tables,
                                    options.ntt_backend,
                                    rescale->q_count,
                                    rescale->joint_drop_count,
                                    rescale->joint_drop_is_prefix,
                                    options.output_domain);
  }
  const auto& q_parameters = tables.q_parameters;
  if (output_ntt) {
    auto result = moddown_ntt_owned(products,
                                    std::nullopt,
                                    tables,
                                    options,
                                    out,
                                    view(plan.scaled),
                                    view(plan.zero));
    rns::add_lazy_out(source.narrow(0, 0, 2), result, q_parameters, result);
    return result;
  }
  auto result = moddown_coefficient(products, tables, options, out, moddown);
  auto coefficient = temporary(plan.coefficient);
  coefficient.copy_(source.narrow(0, 0, 2));
  ntt::transform_(coefficient,
                  tables.q_inverse,
                  options.ntt_backend,
                  "inverse_to_standard");
  rns::add_standard_(result, coefficient, q_parameters);
  return result;
}

at::Tensor rotate_into(const at::Tensor& source,
                       const at::Tensor& key,
                       const GaloisTables& galois,
                       const KeySwitchTables& tables,
                       const KeySwitchOptions& options,
                       const at::Tensor& out) {
  TORCH_CHECK_VALUE(source.size(0) == 2,
                    "Rotation requires ciphertext components=2");
  const bool input_ntt = options.input_domain == "ntt";
  const bool output_ntt = options.output_domain == "ntt";
  at::Tensor rotated;
  at::Tensor component;
  if (input_ntt) {
    rotated = apply_ntt_galois_automorphism(source, ntt_indices(galois));
    component = rotated.select(0, 1).clone();
    ntt::transform_(component,
                    tables.q_inverse,
                    options.ntt_backend,
                    "inverse_to_standard");
  } else {
    rotated = rotate_coefficients(source, galois, tables);
    component = rotated.select(0, 1);
  }
  const auto result = corrections(
      component,
      key,
      tables,
      options,
      output_ntt,
      output_ntt && !input_ntt ? std::optional<at::Tensor>(rotated.select(0, 0))
                               : std::nullopt,
      out);
  const auto& q_parameters = tables.q_parameters;
  if (output_ntt) {
    if (input_ntt) {
      auto component0 = result.select(0, 0);
      rns::add_lazy_out(
          rotated.select(0, 0), component0, q_parameters, component0);
    }
    return result;
  }
  auto rotated0 = rotated.select(0, 0);
  if (input_ntt) {
    rotated0 = rotated0.clone();
    ntt::transform_(
        rotated0, tables.q_inverse, options.ntt_backend, "inverse_to_standard");
  }
  auto component0 = result.select(0, 0);
  rns::add_standard_(component0, rotated0, q_parameters);
  return result;
}

// Workspace plan of `rotate_many`: the hoisted digits, the scratch of exact
// digit lifting, the NTT copy of c0, the QP accumulators of one rotation
// group, and the temporaries of `finish_rotation` for one rotation.
struct RotateManyPlan {
  WorkspaceTensor digits;
  WorkspaceTensor scratch;
  WorkspaceTensor c0;
  WorkspaceTensor accumulators;
  FinishRotationPlan finish;
  // Rotations whose products are accumulated together.
  size_t group = 1;
  int64_t bytes = 0;
};

RotateManyPlan plan_rotate_many(const at::Tensor& source,
                                size_t steps,
                                const KeySwitchTables& tables,
                                const KeySwitchOptions& options) {
  RotateManyPlan plan;
  if (steps == 0) return plan;
  const auto spans = detail_operations::digit_spans(options.digits);
  TORCH_CHECK_VALUE(!spans.empty(),
                    "Rotation hoisting requires an active RNS digit");
  const auto dtype = source.scalar_type();
  const auto component = source.sizes().slice(1).vec();
  const auto rows = component.size() - 2;
  auto qp = component;
  qp[rows] = tables.qp_parameters.size(1);
  WorkspaceLayout layout;
  auto digits = qp;
  digits.insert(digits.begin(), static_cast<int64_t>(spans.size()));
  plan.digits = layout.take(digits, dtype);
  int64_t width = 0;
  for (const auto& span : spans)
    width = std::max(width, span.stop - span.start);
  if (width > 1 && options.basis_conversion == "exact" &&
      !detail_operations::uses_tiled_modup(
          source, tables.qp_parameters, width)) {
    auto scratch = component;
    scratch[rows] = width;
    plan.scratch = layout.take(scratch, dtype);
  }
  const bool output_ntt = options.output_domain == "ntt";
  if (output_ntt) plan.c0 = layout.take(component, dtype);
  // With several digits, one product launch for all rotations reads the
  // digits once. With one digit the saving is small, and producing every QP
  // accumulator before its ModDown evicts it from L2, so each rotation runs
  // its product and ModDown in turn.
  plan.group = spans.size() > 1 ? steps : 1;
  auto accumulators = qp;
  accumulators.insert(accumulators.begin(),
                      {static_cast<int64_t>(plan.group), 2});
  plan.accumulators = layout.take(accumulators, dtype);
  plan.finish =
      plan_finish_rotation(layout, component, source, tables, options);
  plan.bytes = layout.bytes();
  return plan;
}

// With `workspace`, the temporaries of `plan` are views of it.
std::vector<at::Tensor> rotate_many_into(const at::Tensor& source,
                                         at::TensorList keys,
                                         c10::ArrayRef<GaloisTables> galois,
                                         const KeySwitchTables& tables,
                                         const KeySwitchOptions& options,
                                         at::TensorList out,
                                         const RotateManyPlan& plan,
                                         const at::Tensor& workspace) {
  TORCH_CHECK_VALUE(keys.size() == galois.size(),
                    "rotate_many requires one key per step");
  TORCH_CHECK_VALUE(out.empty() || out.size() == galois.size(),
                    "rotate_many_out requires one output per step");
  std::vector<at::Tensor> results;
  if (galois.empty()) {
    return results;
  }
  const auto view = [&](const WorkspaceTensor& tensor) {
    return workspace.defined() && tensor.defined() ? tensor.in(workspace)
                                                   : at::Tensor();
  };
  const auto digits = rotation_digits_into(source.select(0, 1),
                                           tables,
                                           options,
                                           view(plan.digits),
                                           view(plan.scratch));
  const auto c0 =
      options.output_domain == "ntt"
          ? forward_copy(source.select(0, 0), tables, options, view(plan.c0))
          : source.select(0, 0);
  const auto buffer = view(plan.accumulators);
  const auto finish = finish_rotation_buffers(plan.finish, workspace);
  const size_t group = plan.group;
  for (size_t first = 0; first < galois.size(); first += group) {
    const size_t count = std::min(group, galois.size() - first);
    const auto accumulators = accumulate_rotations(
        digits,
        keys.slice(first, count),
        options.key_seeds.empty()
            ? at::TensorList()
            : at::TensorList(options.key_seeds).slice(first, count),
        galois.slice(first, count),
        tables,
        options,
        buffer.defined() ? buffer.narrow(0, 0, static_cast<int64_t>(count))
                         : at::Tensor());
    for (size_t i = 0; i < count; ++i) {
      results.push_back(
          finish_rotation(c0,
                          accumulators[i],
                          galois[first + i],
                          tables,
                          options,
                          out.empty() ? at::Tensor() : out[first + i],
                          finish));
    }
  }
  return results;
}

}  // namespace

namespace {

// Workspace plan of `rotation_sum_add_` for a nonzero step: the rotated
// component one (NTT input) or rotated source (coefficient input), one lifted
// digit and the exact-lift scratch.
struct RotationSumAddPlan {
  WorkspaceTensor component;
  WorkspaceTensor rotated;
  WorkspaceTensor digit;
  WorkspaceTensor scratch;
  int64_t bytes = 0;
};

RotationSumAddPlan plan_rotation_sum_add(const at::Tensor& source,
                                         bool rotation,
                                         const KeySwitchTables& tables,
                                         const KeySwitchOptions& options) {
  RotationSumAddPlan plan;
  if (!rotation) return plan;
  const auto dtype = source.scalar_type();
  const auto component = source.sizes().slice(1).vec();
  const auto rows = component.size() - 2;
  WorkspaceLayout layout;
  if (options.input_domain == "ntt")
    plan.component = layout.take(component, dtype);
  else
    plan.rotated = layout.like(source);
  auto qp = component;
  qp[rows] = tables.qp_parameters.size(1);
  plan.digit = layout.take(qp, dtype);
  if (options.basis_conversion == "exact") {
    int64_t width = 0;
    for (const auto& span : detail_operations::digit_spans(options.digits))
      width = std::max(width, span.stop - span.start);
    // Reserved on every device so that Meta operands give the same size.
    if (width > 1) {
      auto scratch = component;
      scratch[rows] = width;
      plan.scratch = layout.take(scratch, dtype);
    }
  }
  plan.bytes = layout.bytes();
  return plan;
}

void rotation_sum_add_into(at::Tensor& ordinary,
                           at::Tensor& products,
                           const at::Tensor& source,
                           const std::optional<at::Tensor>& key,
                           const std::optional<GaloisTables>& galois,
                           const KeySwitchTables& tables,
                           const KeySwitchOptions& options,
                           const RotationSumAddPlan& plan,
                           const at::Tensor& workspace) {
  TORCH_CHECK_VALUE(
      options.input_domain == "ntt" || options.input_domain == "coefficient",
      "Rotation-sum input domain must be ntt or coefficient");
  const bool input_ntt = options.input_domain == "ntt";
  const auto& q_parameters = tables.q_parameters;
  const auto accumulate = [&](at::Tensor target, const at::Tensor& value) {
    if (input_ntt)
      rns::add_lazy_out(target, value, q_parameters, target);
    else
      rns::add_standard_(target, value, q_parameters);
  };
  if (!galois.has_value()) {
    accumulate(ordinary, source);
    return;
  }
  TORCH_CHECK_VALUE(key.has_value(),
                    "A nonzero rotation requires its evaluation key");
  const auto temporary = [&](const WorkspaceTensor& tensor) {
    return workspace.defined()
               ? tensor.in(workspace)
               : at::empty(tensor.sizes, source.options().dtype(tensor.dtype));
  };
  const auto digit = temporary(plan.digit);
  const auto scratch = lift_scratch(
      workspace.defined() && plan.scratch.defined() ? plan.scratch.in(workspace)
                                                    : at::Tensor(),
      source,
      tables);
  if (input_ntt) {
    const auto& indices = ntt_indices(*galois);
    auto component = temporary(plan.component);
    apply_ntt_galois_automorphism_out(source.select(0, 1), indices, component);
    ntt::transform_(component,
                    tables.q_inverse,
                    options.ntt_backend,
                    "inverse_to_standard");
    key_switch_products_into(
        component, *key, tables, options, products, digit, scratch);
    auto ordinary0 = ordinary.select(0, 0);
    add_ntt_galois_automorphism_(
        ordinary0, source.select(0, 0), indices, q_parameters);
    return;
  }
  const auto rotated =
      rotate_coefficients(source, *galois, tables, temporary(plan.rotated));
  key_switch_products_into(
      rotated.select(0, 1), *key, tables, options, products, digit, scratch);
  accumulate(ordinary.select(0, 0), rotated.select(0, 0));
}

}  // namespace

void rotation_sum_add_(at::Tensor& ordinary,
                       at::Tensor& products,
                       const at::Tensor& source,
                       const std::optional<at::Tensor>& key,
                       const std::optional<GaloisTables>& galois,
                       const KeySwitchTables& tables,
                       const KeySwitchOptions& options,
                       const at::Tensor& workspace) {
  const auto plan =
      plan_rotation_sum_add(source, galois.has_value(), tables, options);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace) {
    detail::check_workspace(workspace, plan.bytes, source, "rotation_sum_add_");
    bool disjoint = storage_disjoint(workspace, ordinary) &&
                    storage_disjoint(workspace, products) &&
                    storage_disjoint(workspace, source);
    if (key.has_value())
      disjoint = disjoint && storage_disjoint(workspace, *key);
    TORCH_CHECK_VALUE(disjoint,
                      "rotation_sum_add_ requires a workspace disjoint from "
                      "its operands");
  }
  rotation_sum_add_into(ordinary,
                        products,
                        source,
                        key,
                        galois,
                        tables,
                        options,
                        plan,
                        with_workspace ? workspace : at::Tensor());
}

int64_t rotation_sum_add_workspace_bytes(
    const at::Tensor& source,
    const std::optional<GaloisTables>& galois,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options) {
  return plan_rotation_sum_add(source, galois.has_value(), tables, options)
      .bytes;
}

namespace {

// Workspace plan of `rotation_sum_finish_`: the row-ordered products and
// the dropped rows of the trailing quotient.
struct RotationSumFinishPlan {
  WorkspaceTensor ordered;
  WorkspaceTensor scratch;
  int64_t bytes = 0;
};

RotationSumFinishPlan plan_rotation_sum_finish(const at::Tensor& products,
                                               const KeySwitchTables& tables,
                                               int64_t remaining_q_count) {
  RotationSumFinishPlan plan;
  WorkspaceLayout layout;
  if (tables.sum_quotient_order.defined()) plan.ordered = layout.like(products);
  auto shape = products.sizes().vec();
  shape[shape.size() - 2] = products.size(-2) - remaining_q_count;
  plan.scratch = layout.take(shape, products.scalar_type());
  plan.bytes = layout.bytes();
  return plan;
}

at::Tensor rotation_sum_finish_into(
    at::Tensor& ordinary,
    at::Tensor& products,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    int64_t remaining_q_count,
    const at::Tensor& out,
    const RotationSumFinishPlan& plan = {},
    const at::Tensor& workspace = at::Tensor()) {
  const auto view = [&](const WorkspaceTensor& tensor) {
    return workspace.defined() && tensor.defined() ? tensor.in(workspace)
                                                   : at::Tensor();
  };
  const auto& q_parameters = tables.q_parameters;
  if (options.input_domain != "ntt") {
    ntt::transform_(ordinary,
                    tables.q_forward,
                    options.ntt_backend,
                    "forward_to_montgomery");
  }
  rns::montgomery_mul_row_scalars_(
      ordinary, tables.sum_p_montgomery, q_parameters);
  auto q_rows = products.narrow(-2, 0, ordinary.size(-2));
  rns::add_lazy_out(q_rows, ordinary, q_parameters, q_rows);
  ntt::transform_(
      products, tables.qp_inverse, options.ntt_backend, "inverse_to_standard");
  return detail_operations::divide_trailing_rows(
      products,
      tables.sum_quotient_inverses,
      tables.sum_quotient_parameters,
      products.size(-2) - remaining_q_count,
      optional_table(tables.sum_quotient_half),
      optional_table(tables.sum_quotient_order),
      options.moddown_rounding,
      out,
      view(plan.ordered),
      view(plan.scratch));
}

}  // namespace

at::Tensor rotation_sum_finish_(at::Tensor& ordinary,
                                at::Tensor& products,
                                const KeySwitchTables& tables,
                                const KeySwitchOptions& options,
                                int64_t remaining_q_count) {
  return rotation_sum_finish_into(
      ordinary, products, tables, options, remaining_q_count, at::Tensor());
}

void rotation_sum_finish_out_(at::Tensor& ordinary,
                              at::Tensor& products,
                              const KeySwitchTables& tables,
                              const KeySwitchOptions& options,
                              int64_t remaining_q_count,
                              at::Tensor& out,
                              const at::Tensor& workspace) {
  const auto plan =
      plan_rotation_sum_finish(products, tables, remaining_q_count);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    detail::check_workspace(
        workspace, plan.bytes, products, "rotation_sum_finish_out_");
  bool disjoint =
      storage_disjoint(out, ordinary) && storage_disjoint(out, products);
  if (with_workspace)
    disjoint = disjoint && storage_disjoint(out, workspace) &&
               storage_disjoint(workspace, ordinary) &&
               storage_disjoint(workspace, products);
  TORCH_CHECK_VALUE(disjoint,
                    "rotation_sum_finish_out_ requires out, workspace and "
                    "accumulators to be disjoint");
  finish_out(
      rotation_sum_finish_into(ordinary,
                               products,
                               tables,
                               options,
                               remaining_q_count,
                               out,
                               plan,
                               with_workspace ? workspace : at::Tensor()),
      out);
}

int64_t rotation_sum_finish_workspace_bytes(const at::Tensor& products,
                                            const KeySwitchTables& tables,
                                            int64_t remaining_q_count) {
  return plan_rotation_sum_finish(products, tables, remaining_q_count).bytes;
}

namespace {

at::Tensor rotation_sum_into(at::TensorList sources,
                             at::TensorList keys,
                             c10::IntArrayRef steps,
                             c10::ArrayRef<GaloisTables> galois,
                             const KeySwitchTables& tables,
                             const KeySwitchOptions& options,
                             int64_t remaining_q_count,
                             const at::Tensor& out) {
  TORCH_CHECK_VALUE(sources.size() == steps.size(),
                    "rotation_sum requires one step per source");
  check_galois_count(galois, steps.size());
  TORCH_CHECK_VALUE(!sources.empty(),
                    "Rotation sum requires at least one contribution");
  auto ordinary = at::zeros_like(sources[0]);
  auto shape = ordinary.sizes().vec();
  shape[shape.size() - 2] = tables.qp_parameters.size(1);
  auto products = at::zeros(shape, ordinary.options());
  size_t key_index = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    std::optional<at::Tensor> key;
    if (steps[i] != 0) {
      TORCH_CHECK_VALUE(key_index < keys.size(),
                        "A nonzero rotation requires its evaluation key");
      key = keys[key_index];
    }
    rotation_sum_add_(
        ordinary,
        products,
        sources[i],
        key,
        steps[i] == 0 ? std::nullopt : std::optional<GaloisTables>(galois[i]),
        tables,
        key ? for_key(options, key_index++) : options);
  }
  return rotation_sum_finish_into(
      ordinary, products, tables, options, remaining_q_count, out);
}

at::Tensor grouped_rotation_weighted_sum_into(
    const at::Tensor& source,
    at::TensorList keys,
    c10::IntArrayRef baby_steps,
    c10::ArrayRef<GaloisTables> galois,
    at::TensorList plaintexts,
    int64_t group_count,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const at::Tensor& out) {
  check_galois_count(galois, baby_steps.size());
  const auto digits = rotation_digits(source.select(0, 1), tables, options);
  const auto source_ntt = forward_copy(source, tables, options);
  auto ntt_options = options;
  ntt_options.output_domain = "ntt";
  std::vector<at::Tensor> babies;
  size_t key_index = 0;
  for (size_t index = 0; index < baby_steps.size(); ++index) {
    if (baby_steps[index] == 0) {
      babies.push_back(source_ntt);
    } else {
      TORCH_CHECK_VALUE(key_index < keys.size(),
                        "A nonzero rotation requires its evaluation key");
      babies.push_back(rotate_hoisted(source_ntt.select(0, 0),
                                      digits,
                                      keys[key_index],
                                      galois[index],
                                      tables,
                                      for_key(ntt_options, key_index)));
      ++key_index;
    }
  }
  if (out.defined()) {
    auto target = out;
    rns::montgomery_weighted_sums_out(
        babies, plaintexts, group_count, tables.q_parameters, target);
    return target;
  }
  return rns::montgomery_weighted_sums(
      babies, plaintexts, group_count, tables.q_parameters);
}

}  // namespace

at::Tensor key_switch(const at::Tensor& source,
                      const at::Tensor& key,
                      const KeySwitchTables& tables,
                      const KeySwitchOptions& options,
                      const std::optional<GaloisTables>& conjugation) {
  return key_switch_into(
      source, key, tables, options, conjugation, at::Tensor());
}

void key_switch_out(const at::Tensor& source,
                    const at::Tensor& key,
                    const KeySwitchTables& tables,
                    const KeySwitchOptions& options,
                    const std::optional<GaloisTables>& conjugation,
                    at::Tensor& out) {
  bool disjoint = storage_disjoint(out, source) && storage_disjoint(out, key);
  TORCH_CHECK_VALUE(disjoint,
                    "key_switch_out requires out disjoint from its operands");
  finish_out(key_switch_into(source, key, tables, options, conjugation, out),
             out);
}

at::Tensor relinearize(
    const at::Tensor& source,
    const at::Tensor& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options) {
  return relinearize_into(
      source,
      key,
      tables,
      options,
      rescale_options,
      at::Tensor(),
      plan_relinearize(source, tables, options, rescale_options),
      at::Tensor());
}

void relinearize_out(const at::Tensor& source,
                     const at::Tensor& key,
                     const KeySwitchTables& tables,
                     const KeySwitchOptions& options,
                     const std::optional<JointRescaleOptions>& rescale_options,
                     at::Tensor& out,
                     const at::Tensor& workspace) {
  const auto plan = plan_relinearize(source, tables, options, rescale_options);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    detail::check_workspace(workspace, plan.bytes, source, "relinearize_out");
  bool disjoint = storage_disjoint(out, source) && storage_disjoint(out, key);
  if (with_workspace)
    disjoint = disjoint && storage_disjoint(out, workspace) &&
               storage_disjoint(workspace, source) &&
               storage_disjoint(workspace, key);
  TORCH_CHECK_VALUE(disjoint,
                    "relinearize_out requires out, workspace and operands to "
                    "be disjoint");
  finish_out(relinearize_into(source,
                              key,
                              tables,
                              options,
                              rescale_options,
                              out,
                              plan,
                              with_workspace ? workspace : at::Tensor()),
             out);
}

int64_t relinearize_workspace_bytes(
    const at::Tensor& source,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options) {
  return plan_relinearize(source, tables, options, rescale_options).bytes;
}

at::Tensor rotate(const at::Tensor& source,
                  const at::Tensor& key,
                  const GaloisTables& galois,
                  const KeySwitchTables& tables,
                  const KeySwitchOptions& options) {
  return rotate_into(source, key, galois, tables, options, at::Tensor());
}

void rotate_out(const at::Tensor& source,
                const at::Tensor& key,
                const GaloisTables& galois,
                const KeySwitchTables& tables,
                const KeySwitchOptions& options,
                at::Tensor& out) {
  bool disjoint = storage_disjoint(out, source) && storage_disjoint(out, key);
  TORCH_CHECK_VALUE(disjoint,
                    "rotate_out requires out disjoint from its operands");
  finish_out(rotate_into(source, key, galois, tables, options, out), out);
}

at::Tensor rotation_digits(const at::Tensor& component,
                           const KeySwitchTables& tables,
                           const KeySwitchOptions& options) {
  return rotation_digits_into(component, tables, options, at::Tensor());
}

void rotation_digits_out(const at::Tensor& component,
                         const KeySwitchTables& tables,
                         const KeySwitchOptions& options,
                         at::Tensor& out) {
  bool disjoint = storage_disjoint(out, component);
  TORCH_CHECK_VALUE(
      disjoint, "rotation_digits_out requires out disjoint from its operands");
  finish_out(rotation_digits_into(component, tables, options, out), out);
}

at::Tensor rotate_hoisted(const at::Tensor& c0,
                          const at::Tensor& hoisted_digits,
                          const at::Tensor& key,
                          const GaloisTables& galois,
                          const KeySwitchTables& tables,
                          const KeySwitchOptions& options) {
  return rotate_hoisted_into(
      c0, hoisted_digits, key, galois, tables, options, at::Tensor());
}

void rotate_hoisted_out(const at::Tensor& c0,
                        const at::Tensor& hoisted_digits,
                        const at::Tensor& key,
                        const GaloisTables& galois,
                        const KeySwitchTables& tables,
                        const KeySwitchOptions& options,
                        at::Tensor& out,
                        const at::Tensor& workspace) {
  const auto plan = plan_rotate_hoisted(c0, hoisted_digits, tables, options);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    detail::check_workspace(workspace, plan.bytes, c0, "rotate_hoisted_out");
  bool disjoint = storage_disjoint(out, c0) &&
                  storage_disjoint(out, hoisted_digits) &&
                  storage_disjoint(out, key);
  if (with_workspace)
    disjoint = disjoint && storage_disjoint(out, workspace) &&
               storage_disjoint(workspace, c0) &&
               storage_disjoint(workspace, hoisted_digits) &&
               storage_disjoint(workspace, key);
  TORCH_CHECK_VALUE(disjoint,
                    "rotate_hoisted_out requires out, workspace and operands "
                    "to be disjoint");
  finish_out(rotate_hoisted_into(c0,
                                 hoisted_digits,
                                 key,
                                 galois,
                                 tables,
                                 options,
                                 out,
                                 plan,
                                 with_workspace ? workspace : at::Tensor()),
             out);
}

int64_t rotate_hoisted_workspace_bytes(const at::Tensor& c0,
                                       const at::Tensor& hoisted_digits,
                                       const KeySwitchTables& tables,
                                       const KeySwitchOptions& options) {
  return plan_rotate_hoisted(c0, hoisted_digits, tables, options).bytes;
}

std::vector<at::Tensor> rotate_many(const at::Tensor& source,
                                    at::TensorList keys,
                                    c10::ArrayRef<GaloisTables> galois,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options) {
  return rotate_many_into(
      source,
      keys,
      galois,
      tables,
      options,
      {},
      plan_rotate_many(source, galois.size(), tables, options),
      {});
}

int64_t rotate_many_workspace_bytes(const at::Tensor& source,
                                    at::TensorList keys,
                                    c10::ArrayRef<GaloisTables> galois,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options) {
  (void)keys;
  return plan_rotate_many(source, galois.size(), tables, options).bytes;
}

void rotate_many_out(const at::Tensor& source,
                     at::TensorList keys,
                     c10::ArrayRef<GaloisTables> galois,
                     const KeySwitchTables& tables,
                     const KeySwitchOptions& options,
                     at::TensorList out,
                     at::Tensor& workspace) {
  const auto plan = plan_rotate_many(source, galois.size(), tables, options);
  const bool with_workspace = workspace.defined() && plan.bytes > 0;
  if (with_workspace)
    detail::check_workspace(workspace, plan.bytes, source, "rotate_many_out");
  for (const auto& target : out) {
    bool disjoint = storage_disjoint(target, source);
    for (const auto& operand : keys)
      disjoint = disjoint && storage_disjoint(target, operand);
    if (with_workspace)
      disjoint = disjoint && storage_disjoint(target, workspace);
    TORCH_CHECK_VALUE(disjoint,
                      "rotate_many_out requires out disjoint from its operands "
                      "and workspace");
  }
  TORCH_CHECK_VALUE(
      !with_workspace || storage_disjoint(workspace, source),
      "rotate_many_out requires a workspace disjoint from source");
  const auto results =
      rotate_many_into(source,
                       keys,
                       galois,
                       tables,
                       options,
                       out,
                       plan,
                       with_workspace ? workspace : at::Tensor());
  for (size_t i = 0; i < results.size(); ++i) {
    auto target = out[i];
    finish_out(results[i], target);
  }
}

at::Tensor rotation_sum(at::TensorList sources,
                        at::TensorList keys,
                        c10::IntArrayRef steps,
                        c10::ArrayRef<GaloisTables> galois,
                        const KeySwitchTables& tables,
                        const KeySwitchOptions& options,
                        int64_t remaining_q_count) {
  return rotation_sum_into(sources,
                           keys,
                           steps,
                           galois,
                           tables,
                           options,
                           remaining_q_count,
                           at::Tensor());
}

void rotation_sum_out(at::TensorList sources,
                      at::TensorList keys,
                      c10::IntArrayRef steps,
                      c10::ArrayRef<GaloisTables> galois,
                      const KeySwitchTables& tables,
                      const KeySwitchOptions& options,
                      int64_t remaining_q_count,
                      at::Tensor& out) {
  bool disjoint = true;
  for (const auto& operand : sources)
    disjoint = disjoint && storage_disjoint(out, operand);
  for (const auto& operand : keys)
    disjoint = disjoint && storage_disjoint(out, operand);
  TORCH_CHECK_VALUE(disjoint,
                    "rotation_sum_out requires out disjoint from its operands");
  finish_out(rotation_sum_into(sources,
                               keys,
                               steps,
                               galois,
                               tables,
                               options,
                               remaining_q_count,
                               out),
             out);
}

at::Tensor grouped_rotation_weighted_sum(const at::Tensor& source,
                                         at::TensorList keys,
                                         c10::IntArrayRef baby_steps,
                                         c10::ArrayRef<GaloisTables> galois,
                                         at::TensorList plaintexts,
                                         int64_t group_count,
                                         const KeySwitchTables& tables,
                                         const KeySwitchOptions& options) {
  return grouped_rotation_weighted_sum_into(source,
                                            keys,
                                            baby_steps,
                                            galois,
                                            plaintexts,
                                            group_count,
                                            tables,
                                            options,
                                            at::Tensor());
}

void grouped_rotation_weighted_sum_out(const at::Tensor& source,
                                       at::TensorList keys,
                                       c10::IntArrayRef baby_steps,
                                       c10::ArrayRef<GaloisTables> galois,
                                       at::TensorList plaintexts,
                                       int64_t group_count,
                                       const KeySwitchTables& tables,
                                       const KeySwitchOptions& options,
                                       at::Tensor& out) {
  bool disjoint = storage_disjoint(out, source);
  for (const auto& operand : keys)
    disjoint = disjoint && storage_disjoint(out, operand);
  for (const auto& operand : plaintexts)
    disjoint = disjoint && storage_disjoint(out, operand);
  TORCH_CHECK_VALUE(disjoint,
                    "grouped_rotation_weighted_sum_out requires out disjoint "
                    "from its operands");
  finish_out(grouped_rotation_weighted_sum_into(source,
                                                keys,
                                                baby_steps,
                                                galois,
                                                plaintexts,
                                                group_count,
                                                tables,
                                                options,
                                                out),
             out);
}

namespace {

// Key data of `keys` and `options` with the seeds of compact keys.
std::pair<std::vector<at::Tensor>, KeySwitchOptions> key_operands(
    const std::vector<KeySwitchKey>& keys, const KeySwitchOptions& options) {
  std::vector<at::Tensor> data;
  auto result = options;
  for (const auto& key : keys) {
    TORCH_CHECK_VALUE(key.expanded() == keys.front().expanded(),
                      "The keys of one key switch must share one form");
    data.push_back(key.data);
    if (!key.expanded()) {
      result.key_seeds.push_back(*key.seed_rows);
      result.key_rounds = key.spec.rounds;
    }
  }
  return {std::move(data), std::move(result)};
}

}  // namespace

at::Tensor key_switch_products(const at::Tensor& component,
                               const KeySwitchKey& key,
                               const KeySwitchTables& tables,
                               const KeySwitchOptions& options,
                               const std::optional<at::Tensor>& accumulator) {
  const auto [data, keyed] = key_operands({key}, options);
  return key_switch_products(component, data[0], tables, keyed, accumulator);
}

at::Tensor key_switch(const at::Tensor& source,
                      const KeySwitchKey& key,
                      const KeySwitchTables& tables,
                      const KeySwitchOptions& options,
                      const std::optional<GaloisTables>& conjugation) {
  const auto [data, keyed] = key_operands({key}, options);
  return key_switch(source, data[0], tables, keyed, conjugation);
}

at::Tensor relinearize(
    const at::Tensor& source,
    const KeySwitchKey& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options) {
  const auto [data, keyed] = key_operands({key}, options);
  return relinearize(source, data[0], tables, keyed, rescale_options);
}

at::Tensor rotate(const at::Tensor& source,
                  const KeySwitchKey& key,
                  const GaloisTables& galois,
                  const KeySwitchTables& tables,
                  const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands({key}, options);
  return rotate(source, data[0], galois, tables, keyed);
}

at::Tensor rotate_hoisted(const at::Tensor& c0,
                          const at::Tensor& hoisted_digits,
                          const KeySwitchKey& key,
                          const GaloisTables& galois,
                          const KeySwitchTables& tables,
                          const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands({key}, options);
  return rotate_hoisted(c0, hoisted_digits, data[0], galois, tables, keyed);
}

std::vector<at::Tensor> rotate_many(const at::Tensor& source,
                                    const std::vector<KeySwitchKey>& keys,
                                    c10::ArrayRef<GaloisTables> galois,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands(keys, options);
  return rotate_many(source, data, galois, tables, keyed);
}

at::Tensor rotation_sum(at::TensorList sources,
                        const std::vector<KeySwitchKey>& keys,
                        c10::IntArrayRef steps,
                        c10::ArrayRef<GaloisTables> galois,
                        const KeySwitchTables& tables,
                        const KeySwitchOptions& options,
                        int64_t remaining_q_count) {
  const auto [data, keyed] = key_operands(keys, options);
  return rotation_sum(
      sources, data, steps, galois, tables, keyed, remaining_q_count);
}

at::Tensor grouped_rotation_weighted_sum(const at::Tensor& source,
                                         const std::vector<KeySwitchKey>& keys,
                                         c10::IntArrayRef baby_steps,
                                         c10::ArrayRef<GaloisTables> galois,
                                         at::TensorList plaintexts,
                                         int64_t group_count,
                                         const KeySwitchTables& tables,
                                         const KeySwitchOptions& options) {
  const auto [data, keyed] = key_operands(keys, options);
  return grouped_rotation_weighted_sum(
      source, data, baby_steps, galois, plaintexts, group_count, tables, keyed);
}

at::Tensor encapsulated_modulus_raise(const at::Tensor& source,
                                      const KeySwitchKey& low_key,
                                      const KeySwitchKey& return_key,
                                      const EncapsulationTables& tables,
                                      const KeySwitchOptions& low_options,
                                      int64_t return_drop_count,
                                      int64_t target_row_start,
                                      int64_t q_count) {
  const auto [data, keyed] = key_operands({low_key, return_key}, low_options);
  return encapsulated_modulus_raise(source,
                                    data[0],
                                    data[1],
                                    tables,
                                    keyed,
                                    return_drop_count,
                                    target_row_start,
                                    q_count);
}

}  // namespace fhelium::ckks
