#include <ATen/ATen.h>
#include <fhelium/ckks_parameters.h>
#include <fhelium/ckks_tables.h>
#include <fhelium/ntt.h>
#include <fhelium/ntt_tables.h>
#include <fhelium/operation_tables.h>
#include <fhelium/rns.h>
#include <fhelium/rns_tables.h>

#include <algorithm>
#include <string>
#include <vector>

#include "composition.h"

namespace fhelium {

namespace {

using detail_operations::NttFamily;

std::vector<int64_t> moduli_of(const RnsBasis& basis, c10::IntArrayRef ids) {
  std::vector<int64_t> result;
  result.reserve(ids.size());
  for (const auto id : ids) {
    TORCH_CHECK_VALUE(id >= 0 && id < static_cast<int64_t>(basis.moduli.size()),
                      "Prime id ",
                      id,
                      " is outside the RNS basis");
    result.push_back(basis.moduli[id]);
  }
  return result;
}

bool is_interval(c10::IntArrayRef ids) {
  for (size_t i = 1; i < ids.size(); ++i) {
    if (ids[i] != ids[0] + static_cast<int64_t>(i)) {
      return false;
    }
  }
  return !ids.empty();
}

at::Tensor index_tensor(c10::IntArrayRef values, const at::Device& device) {
  return at::tensor(values, at::TensorOptions().dtype(at::kLong)).to(device);
}

// Rows `ids` of a table whose leading dimension follows the basis primes.
at::Tensor select_rows(const at::Tensor& table, c10::IntArrayRef ids) {
  if (is_interval(ids)) {
    return table.narrow(0, ids[0], static_cast<int64_t>(ids.size()));
  }
  return table.index_select(0, index_tensor(ids, table.device()));
}

std::vector<int64_t> concat(std::initializer_list<c10::IntArrayRef> parts) {
  std::vector<int64_t> result;
  for (const auto& part : parts) {
    result.insert(result.end(), part.begin(), part.end());
  }
  return result;
}

int64_t position(c10::IntArrayRef ids, int64_t id) {
  const auto found = std::find(ids.begin(), ids.end(), id);
  TORCH_CHECK_VALUE(
      found != ids.end(), "Prime id ", id, " is not in the basis");
  return static_cast<int64_t>(found - ids.begin());
}

std::vector<int64_t> positions(c10::IntArrayRef ids, c10::IntArrayRef basis) {
  std::vector<int64_t> result;
  for (const auto id : ids) {
    result.push_back(position(basis, id));
  }
  return result;
}

bool contains(c10::IntArrayRef ids, int64_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

struct Context {
  const RnsBasis& basis;
  at::ScalarType dtype;
  at::Device device;

  explicit Context(const RnsBasis& rns)
      : basis(rns),
        dtype(rns.parameters.scalar_type()),
        device(rns.parameters.device()) {}

  std::vector<int64_t> moduli(c10::IntArrayRef ids) const {
    return moduli_of(basis, ids);
  }
  at::Tensor parameters(c10::IntArrayRef ids) const {
    return select_rns_parameters(basis, ids);
  }
  at::Tensor gathered_parameters(c10::IntArrayRef ids) const {
    return basis.parameters.index_select(1, index_tensor(ids, device));
  }
  at::Tensor on_device(const at::Tensor& tensor) const {
    return tensor.to(device);
  }
  at::Tensor positions_tensor(c10::IntArrayRef ids,
                              c10::IntArrayRef basis_ids) const {
    return index_tensor(positions(ids, basis_ids), device);
  }
  at::Tensor trailing_inverses(c10::IntArrayRef ids, int64_t drop) const {
    return on_device(rns::trailing_quotient_inverses(moduli(ids), drop, dtype));
  }
  // floor or nearest offsets of the quotient by P·D on rows `ids`, expanded
  // over the ring dimension.
  at::Tensor offset(c10::IntArrayRef p,
                    c10::IntArrayRef d,
                    bool p_nearest,
                    bool d_nearest,
                    c10::IntArrayRef ids) const {
    return on_device(rns::quotient_offset_residues(
                         p, d, p_nearest, d_nearest, moduli(ids), dtype))
        .view({-1, 1})
        .expand({-1, basis.ring_dimension});
  }
};

void check_rounding(std::string_view rounding) {
  TORCH_CHECK_VALUE(rounding == "nearest" || rounding == "floor",
                    "Quotient rounding must be nearest or floor");
}

void check_conversion(std::string_view conversion) {
  TORCH_CHECK_VALUE(conversion == "exact" || conversion == "crt" ||
                        conversion == "normalized_crt",
                    "basis_conversion must be exact, crt, or normalized_crt");
}

CrtTables crt_tables(
    const Context& context,
    c10::IntArrayRef source,
    c10::IntArrayRef target,
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
        values) {
  CrtTables tables;
  tables.source_parameters = context.gathered_parameters(source);
  tables.target_parameters = context.gathered_parameters(target);
  tables.matrix = context.on_device(std::get<0>(values));
  tables.target_reciprocals = context.on_device(std::get<1>(values));
  tables.source_normalizers = context.on_device(std::get<2>(values));
  tables.source_reciprocals = context.on_device(std::get<3>(values));
  tables.target_offsets = context.on_device(std::get<4>(values));
  return tables;
}

}  // namespace

JointRescaleTables build_joint_rescale_tables(const RnsBasis& basis,
                                              c10::IntArrayRef q,
                                              c10::IntArrayRef p,
                                              c10::IntArrayRef d,
                                              c10::IntArrayRef target_ids,
                                              std::string_view moddown_rounding,
                                              std::string_view rescale_rounding,
                                              std::string_view conversion) {
  const Context context(basis);
  check_rounding(moddown_rounding);
  check_rounding(rescale_rounding);
  TORCH_CHECK_VALUE(!d.empty(),
                    "The rescale group must be a nonempty subset of Q rows");
  for (const auto id : d) {
    TORCH_CHECK_VALUE(contains(q, id),
                      "The rescale group must be a nonempty subset of Q rows");
  }
  std::vector<int64_t> remaining;
  for (const auto id : q) {
    if (!contains(d, id)) {
      remaining.push_back(id);
    }
  }
  const std::vector<int64_t> target =
      target_ids.empty() ? remaining : target_ids.vec();
  std::vector<int64_t> restored;
  for (const auto id : target) {
    if (!contains(q, id)) {
      restored.push_back(id);
    }
  }
  for (const auto id : remaining) {
    TORCH_CHECK_VALUE(contains(target, id),
                      "The rescale target must contain the remaining Q rows");
  }
  TORCH_CHECK_VALUE(
      !restored.empty() || target == remaining,
      "A rescale target without restored primes keeps the Q order");
  const auto dropped = concat({d, p});
  const auto ordered = concat({target, d, p});
  const auto P = context.moduli(p);
  const auto D = context.moduli(d);
  const auto R = context.moduli(restored);
  const bool p_nearest = moddown_rounding == "nearest";
  const bool d_nearest = rescale_rounding == "nearest";
  JointRescaleTables tables;
  tables.remaining_parameters = context.parameters(target);
  tables.dropped_parameters = context.parameters(dropped);
  tables.ordered_parameters = context.parameters(ordered);
  tables.p_times_base = context.on_device(rns::product_residues(
      concat({P, R}), context.moduli(q), 1, true, context.dtype));
  tables.inverses =
      context.trailing_inverses(ordered, static_cast<int64_t>(dropped.size()));
  tables.product_inverse = context.on_device(rns::product_residues(
      concat({P, D}), context.moduli(target), -1, true, context.dtype));
  tables.dropped_offset = context.offset(P, D, p_nearest, d_nearest, dropped);
  tables.remaining_offset = context.offset(P, D, p_nearest, d_nearest, target);
  tables.drop_positions = context.positions_tensor(d, q);
  tables.remaining_positions = context.positions_tensor(remaining, q);
  if (conversion == "normalized_crt") {
    const auto target_d = concat({target, d});
    tables.crt_d = build_centered_quotient_tables(context.basis, p, d, {});
    tables.crt_t = build_centered_quotient_tables(context.basis, p, target, d);
    tables.crt_p_inverse =
        context.on_device(rns::product_residues(P, D, -1, true, context.dtype));
    tables.crt_inverses =
        context.trailing_inverses(target_d, static_cast<int64_t>(d.size()));
    tables.crt_ordered_parameters = context.parameters(target_d);
    if (d_nearest) {
      tables.crt_half = context.on_device(rns::half_product_residues(
          D, context.moduli(target_d), context.dtype));
    }
  }
  if (!restored.empty()) {
    tables.restore = context.on_device(rns::product_residues(
        R, context.moduli(concat({q, p})), 1, true, context.dtype));
    tables.product_parameters = context.parameters(concat({q, p}));
    tables.target_keep_positions = context.positions_tensor(remaining, target);
  }
  return tables;
}

namespace {

DigitTables digit_tables(const Context& context,
                         c10::IntArrayRef rows,
                         c10::IntArrayRef qp,
                         std::string_view conversion) {
  DigitTables tables;
  if (conversion != "exact") {
    tables.crt = build_crt_tables(
        context.basis, rows, qp, conversion == "normalized_crt");
    return tables;
  }
  if (rows.size() == 1) {
    tables.extension = at::empty(
        {0, 0},
        at::TensorOptions().dtype(context.dtype).device(context.device));
    return tables;
  }
  auto [normalizers, propagation, extension] = rns::mixed_radix_tables(
      context.moduli(rows), context.moduli(qp), context.dtype);
  tables.normalizers = context.on_device(normalizers).contiguous();
  tables.propagation = context.on_device(propagation).contiguous();
  const auto parameters = context.parameters(rows);
  for (int64_t part = 0; part < 4; ++part) {
    tables.reduction[part] = parameters.select(0, part + 1).contiguous();
  }
  tables.extension = context.on_device(extension);
  return tables;
}

}  // namespace

std::vector<GaloisTables> unflatten_galois_tables(const TableSlots& slots) {
  constexpr size_t kSlots = 3;
  TORCH_CHECK_VALUE(slots.size() % kSlots == 0,
                    "Galois tables require three slots per step; got ",
                    slots.size());
  std::vector<GaloisTables> result(slots.size() / kSlots);
  size_t index = 0;
  for (auto& tables : result) {
    auto assign = [&](const TableSlotName&, at::Tensor& tensor) {
      const std::optional<at::Tensor> value = slots.get(index++);
      if (value.has_value()) {
        tensor = *value;
      }
    };
    visit_tables(tables, nullptr, assign);
  }
  return result;
}

at::Tensor select_rns_parameters(const RnsBasis& basis,
                                 c10::IntArrayRef prime_ids) {
  TORCH_CHECK_VALUE(!prime_ids.empty(), "RNS prime_ids must be nonempty");
  if (is_interval(prime_ids)) {
    return basis.parameters.narrow(
        1, prime_ids[0], static_cast<int64_t>(prime_ids.size()));
  }
  return basis.parameters.index_select(
      1, index_tensor(prime_ids, basis.parameters.device()));
}

NttContextTables prepare_ntt_tables(const RnsBasis& basis,
                                    std::string_view ntt_backend) {
  const auto policy = detail_operations::ntt_policy(ntt_backend);
  const bool shoup32 = ntt_backend == "radix2_compact_shoup32";
#if defined(FHELIUM_WITH_HIP)
  TORCH_CHECK_VALUE(!shoup32,
                    "Shoup32 NTT tables are not supported by the HIP backend");
#endif
  const auto& parameters = basis.parameters;
  const auto device = parameters.device();
  const auto dtype = parameters.scalar_type();
  int64_t log_n = 0;
  while ((int64_t{1} << log_n) < basis.ring_dimension) {
    ++log_n;
  }
  const auto materialize = [&](const at::Tensor& table) {
    auto result = table.to(device);
    if (!shoup32) {
      auto rows = result.view({result.size(0), -1});
      rns::to_montgomery_(rows, parameters);
    }
    return result;
  };
  NttContextTables tables;
  tables.ntt_backend = std::string(ntt_backend);
  tables.forward.parameters = parameters;
  tables.inverse.parameters = parameters;
  switch (policy.family) {
    case NttFamily::kCompact: {
      auto [forward, inverse] =
          ntt::compact_twiddles(basis.moduli, log_n, dtype, shoup32);
      tables.forward.schedule[0] = materialize(forward);
      tables.inverse.schedule[0] = materialize(inverse);
      break;
    }
    case NttFamily::kIndexed: {
      auto [forward_indices,
            forward_twiddles,
            inverse_indices,
            inverse_twiddles] =
          ntt::indexed_radix2_plan(basis.moduli, log_n, dtype);
      for (const auto& indices : {forward_indices, inverse_indices}) {
        ntt::validate_indexed_schedule(
            indices[0], indices[1], basis.ring_dimension);
      }
      forward_indices = forward_indices.to(device);
      inverse_indices = inverse_indices.to(device);
      tables.forward.schedule[0] = materialize(forward_twiddles);
      tables.forward.schedule[1] = forward_indices[0];
      tables.forward.schedule[2] = forward_indices[1];
      tables.inverse.schedule[0] = materialize(inverse_twiddles);
      tables.inverse.schedule[1] = inverse_indices[0];
      tables.inverse.schedule[2] = inverse_indices[1];
      break;
    }
    case NttFamily::kFixedRadix: {
      const auto radix = std::stoll(
          std::string(ntt_backend.substr(5, ntt_backend.find('_') - 5)));
      auto [forward_outer, inverse_outer, forward_roots, inverse_roots] =
          ntt::power_of_two_radix_twiddles(basis.moduli, log_n, radix, dtype);
      tables.forward.schedule[0] = materialize(forward_outer);
      tables.forward.schedule[1] = materialize(forward_roots);
      tables.inverse.schedule[0] = materialize(inverse_outer);
      tables.inverse.schedule[1] = materialize(inverse_roots);
      break;
    }
  }
  return tables;
}

NttTables select_ntt_tables(const RnsBasis& basis,
                            const NttContextTables& tables,
                            c10::IntArrayRef prime_ids,
                            bool inverse) {
  const auto& full = inverse ? tables.inverse : tables.forward;
  NttTables result;
  result.parameters = select_rns_parameters(basis, prime_ids);
  result.schedule[0] = select_rows(full.schedule[0], prime_ids).contiguous();
  switch (detail_operations::ntt_policy(tables.ntt_backend).family) {
    case NttFamily::kIndexed:
      result.schedule[1] = full.schedule[1];
      result.schedule[2] = full.schedule[2];
      break;
    case NttFamily::kCompact:
      break;
    case NttFamily::kFixedRadix:
      result.schedule[1] = select_rows(full.schedule[1], prime_ids);
      break;
  }
  return result;
}

CrtTables build_crt_tables(const RnsBasis& basis,
                           c10::IntArrayRef source_prime_ids,
                           c10::IntArrayRef target_prime_ids,
                           bool normalized) {
  const Context context(basis);
  return crt_tables(
      context,
      source_prime_ids,
      target_prime_ids,
      rns::crt_basis_extension_tables(context.moduli(source_prime_ids),
                                      context.moduli(target_prime_ids),
                                      normalized,
                                      context.dtype));
}

CrtTables build_centered_quotient_tables(const RnsBasis& basis,
                                         c10::IntArrayRef source_prime_ids,
                                         c10::IntArrayRef target_prime_ids,
                                         c10::IntArrayRef divisor_prime_ids) {
  const Context context(basis);
  return crt_tables(context,
                    source_prime_ids,
                    target_prime_ids,
                    rns::crt_quotient_tables(context.moduli(source_prime_ids),
                                             context.moduli(target_prime_ids),
                                             context.moduli(divisor_prime_ids),
                                             context.dtype));
}

KeySwitchTables build_key_switch_tables(const RnsBasis& basis,
                                        const KeySwitchTableSpec& spec) {
  check_conversion(spec.basis_conversion);
  check_rounding(spec.moddown_rounding);
  const Context context(basis);
  const auto& q = spec.q_prime_ids;
  const auto& p = spec.p_prime_ids;
  TORCH_CHECK_VALUE(!q.empty() && !p.empty(),
                    "Key-switch tables require Q and P rows");
  const auto qp = concat({q, p});
  KeySwitchTables tables;
  tables.q_parameters = context.parameters(q);
  tables.qp_parameters = context.parameters(qp);
  if (spec.rescale) {
    tables.joint = build_joint_rescale_tables(basis,
                                              q,
                                              p,
                                              spec.drop_prime_ids,
                                              spec.target_prime_ids,
                                              spec.moddown_rounding,
                                              spec.rescale_rounding,
                                              spec.basis_conversion);
  } else {
    const auto P = context.moduli(p);
    tables.moddown_inverses =
        context.trailing_inverses(qp, static_cast<int64_t>(p.size()));
    tables.p_inverse = context.on_device(
        rns::product_residues(P, context.moduli(q), -1, true, context.dtype));
    if (spec.moddown_rounding == "nearest") {
      tables.moddown_half = context.on_device(
          rns::half_product_residues(P, context.moduli(qp), context.dtype));
    }
    if (spec.basis_conversion == "normalized_crt") {
      tables.moddown_crt = build_centered_quotient_tables(basis, p, q, {});
    }
  }
  for (const auto& span : detail_operations::digit_spans(spec.digits)) {
    TORCH_CHECK_VALUE(span.start >= 0 && span.start < span.stop &&
                          span.stop <= static_cast<int64_t>(q.size()),
                      "Digit spans must lie within the Q rows");
    const c10::IntArrayRef rows(q.data() + span.start, span.stop - span.start);
    tables.digits.push_back(
        digit_tables(context, rows, qp, spec.basis_conversion));
  }
  return tables;
}

void fill_key_switch_ntt_tables(KeySwitchTables& tables,
                                const RnsBasis& basis,
                                const NttContextTables& ntt,
                                const KeySwitchTableSpec& spec) {
  const auto& q = spec.q_prime_ids;
  const auto& p = spec.p_prime_ids;
  const auto qp = concat({q, p});
  const auto select = [&](c10::IntArrayRef ids, bool inverse) {
    return select_ntt_tables(basis, ntt, ids, inverse);
  };
  if (!spec.rescale) {
    tables.q_forward = select(q, false);
    tables.q_inverse = select(q, true);
    tables.qp_forward = select(qp, false);
    tables.qp_inverse = select(qp, true);
    tables.p_forward = select(p, false);
    tables.p_inverse_ntt = select(p, true);
    return;
  }
  std::vector<int64_t> target = spec.target_prime_ids;
  if (target.empty()) {
    for (const auto id : q) {
      if (!contains(spec.drop_prime_ids, id)) {
        target.push_back(id);
      }
    }
  }
  tables.q_inverse = select(q, true);
  tables.qp_forward = select(qp, false);
  tables.joint.dropped_inverse = select(concat({spec.drop_prime_ids, p}), true);
  if (spec.output_domain == "ntt") {
    tables.joint.remaining_forward = select(target, false);
  } else {
    tables.joint.remaining_inverse = select(target, true);
  }
}

void add_rotation_sum_tables(KeySwitchTables& tables,
                             const RnsBasis& basis,
                             const KeySwitchTableSpec& spec,
                             c10::IntArrayRef next_prime_ids,
                             std::string_view rounding) {
  check_rounding(rounding);
  const Context context(basis);
  const auto& q = spec.q_prime_ids;
  const auto& p = spec.p_prime_ids;
  const std::vector<int64_t> next =
      next_prime_ids.empty() ? q : next_prime_ids.vec();
  std::vector<int64_t> dropped;
  for (const auto id : q) {
    if (!contains(next, id)) {
      dropped.push_back(id);
    }
  }
  for (const auto id : next) {
    TORCH_CHECK_VALUE(
        contains(q, id),
        "Rotation-sum rescale requires a transition that only drops Q rows");
  }
  TORCH_CHECK_VALUE(
      next_prime_ids.empty() || !dropped.empty(),
      "Rotation-sum rescale requires a transition that only drops Q rows");
  const auto P = context.moduli(p);
  const auto M = context.moduli(dropped);
  tables.sum_p_montgomery = context.on_device(
      rns::product_residues(P, context.moduli(q), 1, true, context.dtype));
  const auto quotient_ids = concat({next, dropped, p});
  tables.sum_quotient_inverses = context.trailing_inverses(
      quotient_ids, static_cast<int64_t>(p.size() + dropped.size()));
  if (rounding == "nearest") {
    tables.sum_quotient_half = context.on_device(rns::half_product_residues(
        concat({P, M}), context.moduli(quotient_ids), context.dtype));
  }
  const auto& qp_parameters = tables.qp_parameters;
  if (dropped.empty()) {
    tables.sum_quotient_parameters = qp_parameters;
    return;
  }
  auto order = positions(concat({next, dropped}), q);
  for (size_t i = 0; i < p.size(); ++i) {
    order.push_back(static_cast<int64_t>(q.size() + i));
  }
  tables.sum_quotient_order = index_tensor(order, context.device);
  tables.sum_quotient_parameters =
      qp_parameters.index_select(1, tables.sum_quotient_order);
}

RescaleTables build_rescale_tables(const RnsBasis& basis,
                                   c10::IntArrayRef source_prime_ids,
                                   c10::IntArrayRef target_prime_ids,
                                   std::string_view rounding,
                                   bool ntt_input,
                                   const NttContextTables* ntt) {
  check_rounding(rounding);
  const Context context(basis);
  const auto source = source_prime_ids;
  const auto target = target_prime_ids;
  std::vector<int64_t> dropped, restored, kept;
  for (const auto id : source) {
    (contains(target, id) ? kept : dropped).push_back(id);
  }
  for (const auto id : target) {
    if (!contains(source, id)) {
      restored.push_back(id);
    }
  }
  TORCH_CHECK_VALUE(!dropped.empty(), "Rescaling must drop at least one prime");
  const auto D = context.moduli(dropped);
  const auto R = context.moduli(restored);
  const bool nearest = rounding == "nearest";
  const auto ordered = concat({target, dropped});
  RescaleTables tables;
  tables.source_parameters = context.parameters(source);
  tables.target_parameters = context.parameters(target);
  tables.dropped_parameters = context.parameters(dropped);
  tables.ordered_parameters = context.parameters(ordered);
  tables.drop_positions = context.positions_tensor(dropped, source);
  tables.keep_positions = context.positions_tensor(kept, source);
  tables.inverses =
      context.trailing_inverses(ordered, static_cast<int64_t>(dropped.size()));
  tables.dropped_offset = context.offset({}, D, false, nearest, dropped);
  tables.target_offset = context.offset({}, D, false, nearest, target);
  if (!restored.empty()) {
    tables.target_keep_positions = context.positions_tensor(kept, target);
    tables.restore = context.on_device(rns::product_residues(
        R, context.moduli(source), 1, true, context.dtype));
  }
  if (ntt_input) {
    tables.product_inverse = context.on_device(rns::product_residues(
        D, context.moduli(target), -1, true, context.dtype));
    if (ntt != nullptr) {
      tables.dropped_inverse = select_ntt_tables(basis, *ntt, dropped, true);
      tables.target_forward = select_ntt_tables(basis, *ntt, target, false);
    }
  }
  return tables;
}

BasisExtensionTables build_basis_extension_tables(
    const RnsBasis& source,
    const RnsBasis& target,
    c10::IntArrayRef source_prime_ids,
    c10::IntArrayRef target_prime_ids) {
  const Context context(source);
  TORCH_CHECK_VALUE(target.parameters.scalar_type() == context.dtype &&
                        target.parameters.device() == context.device,
                    "Basis-extension contexts must share dtype and device");
  const auto source_moduli = context.moduli(source_prime_ids);
  const auto target_moduli = moduli_of(target, target_prime_ids);
  TORCH_CHECK_VALUE(!source_moduli.empty() && !target_moduli.empty(),
                    "Basis extension requires nonempty bases");
  auto [normalizers, propagation, extension] =
      rns::mixed_radix_tables(source_moduli, target_moduli, context.dtype);
  BasisExtensionTables tables;
  tables.source_parameters = context.gathered_parameters(source_prime_ids);
  tables.target_parameters = target.parameters.index_select(
      1, index_tensor(target_prime_ids, context.device));
  tables.source_lo = tables.source_parameters[1];
  tables.source_hi = tables.source_parameters[2];
  tables.source_neg_lo = tables.source_parameters[3];
  tables.source_neg_hi = tables.source_parameters[4];
  tables.normalizers = context.on_device(normalizers);
  tables.propagation = context.on_device(propagation);
  tables.extension = context.on_device(extension);
  tables.half_digits =
      context.on_device(rns::half_product_digits(source_moduli, context.dtype));
  tables.source_product = context.on_device(rns::product_residues(
      source_moduli, target_moduli, 1, false, context.dtype));
  tables.target_moduli =
      context.on_device(at::tensor(c10::IntArrayRef(target_moduli),
                                   at::TensorOptions().dtype(at::kLong))
                            .to(context.dtype));
  return tables;
}

EncapsulationTables build_encapsulation_tables(
    const RnsBasis& basis,
    std::string_view ntt_backend,
    double scale,
    c10::IntArrayRef source_q_moduli,
    c10::IntArrayRef low_p_moduli,
    int64_t return_prime,
    c10::IntArrayRef return_p_moduli,
    c10::IntArrayRef q_moduli,
    c10::IntArrayRef target_prime_ids,
    std::string_view rounding) {
  check_rounding(rounding);
  const Context context(basis);
  int64_t log_n = 0;
  while ((int64_t{1} << log_n) < basis.ring_dimension) {
    ++log_n;
  }
  const auto make_basis = [&](std::vector<int64_t> moduli) {
    RnsBasis result;
    result.parameters =
        rns::parameter_table(moduli, log_n, scale, context.dtype)
            .to(context.device);
    result.moduli = std::move(moduli);
    result.ring_dimension = basis.ring_dimension;
    return result;
  };
  const auto low = make_basis(concat({source_q_moduli, low_p_moduli}));
  const auto high = make_basis(
      concat({c10::IntArrayRef(return_prime), q_moduli, return_p_moduli}));
  const auto low_ntt = prepare_ntt_tables(low, ntt_backend);
  const auto high_ntt = prepare_ntt_tables(high, ntt_backend);

  const auto q_count = static_cast<int64_t>(source_q_moduli.size());
  KeySwitchTableSpec spec;
  for (int64_t i = 0; i < q_count; ++i) {
    spec.q_prime_ids.push_back(i);
  }
  for (size_t i = 0; i < low_p_moduli.size(); ++i) {
    spec.p_prime_ids.push_back(q_count + static_cast<int64_t>(i));
  }
  int64_t start = 0;
  int64_t key_digit = 0;
  for (const auto size : ckks::key_digit_sizes(source_q_moduli, low_p_moduli)) {
    spec.digits.insert(spec.digits.end(), {start, start + size, key_digit++});
    start += size;
  }
  spec.moddown_rounding = std::string(rounding);
  EncapsulationTables tables;
  tables.low = build_key_switch_tables(low, spec);
  fill_key_switch_ntt_tables(tables.low, low, low_ntt, spec);

  std::vector<int64_t> high_ids(high.moduli.size());
  for (size_t i = 0; i < high_ids.size(); ++i) {
    high_ids[i] = static_cast<int64_t>(i);
  }
  tables.raise =
      build_basis_extension_tables(low, high, spec.q_prime_ids, high_ids);
  tables.return_forward = select_ntt_tables(high, high_ntt, high_ids, false);
  tables.return_inverse = select_ntt_tables(high, high_ntt, high_ids, true);

  const auto full_q = static_cast<int64_t>(q_moduli.size());
  std::vector<int64_t> order;
  for (int64_t i = 1; i <= full_q; ++i) {
    order.push_back(i);
  }
  order.push_back(0);
  for (int64_t i = full_q + 1; i < static_cast<int64_t>(high_ids.size()); ++i) {
    order.push_back(i);
  }
  tables.quotient_order = index_tensor(order, context.device);
  tables.quotient_parameters =
      high.parameters.index_select(1, tables.quotient_order);
  const auto quotient_moduli =
      concat({q_moduli, c10::IntArrayRef(return_prime), return_p_moduli});
  const auto drop_count = static_cast<int64_t>(return_p_moduli.size()) + 1;
  tables.quotient_inverses = context.on_device(rns::trailing_quotient_inverses(
      quotient_moduli, drop_count, context.dtype));
  if (rounding == "nearest") {
    tables.quotient_half = context.on_device(rns::half_product_residues(
        concat({c10::IntArrayRef(return_prime), return_p_moduli}),
        quotient_moduli,
        context.dtype));
  }
  tables.target_parameters = context.parameters(target_prime_ids);
  return tables;
}

RingTables build_ring_tables(const RnsBasis& basis,
                             const NttContextTables& ntt,
                             c10::IntArrayRef prime_ids) {
  RingTables tables;
  tables.forward = select_ntt_tables(basis, ntt, prime_ids, false);
  tables.inverse = select_ntt_tables(basis, ntt, prime_ids, true);
  return tables;
}

DecryptionTables build_reconstruction_tables(const RnsBasis& basis,
                                             c10::IntArrayRef prime_ids) {
  const Context context(basis);
  const auto moduli = context.moduli(prime_ids);
  auto [normalizers, propagation, extension] =
      rns::mixed_radix_tables(moduli, {}, context.dtype);
  (void)extension;
  DecryptionTables tables;
  tables.reconstruction_parameters = context.parameters(prime_ids);
  tables.normalizers = context.on_device(normalizers);
  tables.propagation = context.on_device(propagation);
  tables.half_digits =
      context.on_device(rns::half_product_digits(moduli, context.dtype));
  return tables;
}

DecryptionTables build_decryption_tables(const RnsBasis& basis,
                                         const NttContextTables& ntt,
                                         c10::IntArrayRef prime_ids) {
  auto tables = build_reconstruction_tables(basis, prime_ids);
  tables.ring = build_ring_tables(basis, ntt, prime_ids);
  return tables;
}

GaloisTables build_galois_tables(int64_t log_n,
                                 int64_t galois_element,
                                 const at::Device& device) {
  GaloisTables tables;
  tables.ntt_indices = ntt::galois_indices(log_n, galois_element).to(device);
  auto [indices, signs] =
      ckks::coefficient_galois_indices(log_n, galois_element);
  tables.coefficient_indices = indices.to(device);
  tables.coefficient_signs = signs.to(device);
  return tables;
}

// Operator functions: table builders with Torch operator signatures.

namespace {

RnsBasis basis(const at::Tensor& parameters,
               c10::IntArrayRef moduli,
               int64_t log_n) {
  return {moduli.vec(), parameters, int64_t{1} << log_n};
}

}  // namespace

}  // namespace fhelium

namespace fhelium::ntt {

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> prepare_tables(
    const at::Tensor& parameters,
    c10::IntArrayRef moduli,
    int64_t log_n,
    std::string_view ntt_backend) {
  const auto tables = fhelium::prepare_ntt_tables(
      basis(parameters, moduli, log_n), ntt_backend);
  const auto schedule = [](const fhelium::NttTables& direction) {
    std::vector<at::Tensor> result;
    for (const auto& tensor : direction.schedule) {
      if (tensor.defined()) {
        result.push_back(tensor);
      }
    }
    return result;
  };
  return {schedule(tables.forward), schedule(tables.inverse)};
}

}  // namespace fhelium::ntt

namespace fhelium::rns {

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
joint_rescale_table_set(const at::Tensor& parameters,
                        c10::IntArrayRef moduli,
                        int64_t log_n,
                        c10::IntArrayRef q_prime_ids,
                        c10::IntArrayRef p_prime_ids,
                        c10::IntArrayRef drop_prime_ids,
                        c10::IntArrayRef target_prime_ids,
                        std::string_view moddown_rounding,
                        std::string_view rescale_rounding,
                        std::string_view basis_conversion) {
  fhelium::KeySwitchTables tables;
  tables.joint =
      fhelium::build_joint_rescale_tables(basis(parameters, moduli, log_n),
                                          q_prime_ids,
                                          p_prime_ids,
                                          drop_prime_ids,
                                          target_prime_ids,
                                          moddown_rounding,
                                          rescale_rounding,
                                          basis_conversion);
  return fhelium::defined_table_slots(std::move(tables));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>> rescale_table_set(
    const at::Tensor& parameters,
    c10::IntArrayRef moduli,
    int64_t log_n,
    c10::IntArrayRef source_prime_ids,
    c10::IntArrayRef target_prime_ids,
    std::string_view rounding,
    bool ntt_input) {
  return fhelium::defined_table_slots(
      fhelium::build_rescale_tables(basis(parameters, moduli, log_n),
                                    source_prime_ids,
                                    target_prime_ids,
                                    rounding,
                                    ntt_input,
                                    nullptr));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
basis_extension_table_set(const at::Tensor& source_parameters,
                          c10::IntArrayRef source_moduli,
                          const at::Tensor& target_parameters,
                          c10::IntArrayRef target_moduli,
                          int64_t log_n,
                          c10::IntArrayRef source_prime_ids,
                          c10::IntArrayRef target_prime_ids) {
  return fhelium::defined_table_slots(fhelium::build_basis_extension_tables(
      basis(source_parameters, source_moduli, log_n),
      basis(target_parameters, target_moduli, log_n),
      source_prime_ids,
      target_prime_ids));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>> crt_table_set(
    const at::Tensor& parameters,
    c10::IntArrayRef moduli,
    int64_t log_n,
    c10::IntArrayRef source_prime_ids,
    c10::IntArrayRef target_prime_ids,
    bool normalized) {
  return fhelium::defined_table_slots(
      fhelium::build_crt_tables(basis(parameters, moduli, log_n),
                                source_prime_ids,
                                target_prime_ids,
                                normalized));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
centered_quotient_table_set(const at::Tensor& parameters,
                            c10::IntArrayRef moduli,
                            int64_t log_n,
                            c10::IntArrayRef source_prime_ids,
                            c10::IntArrayRef target_prime_ids,
                            c10::IntArrayRef divisor_prime_ids) {
  return fhelium::defined_table_slots(
      fhelium::build_centered_quotient_tables(basis(parameters, moduli, log_n),
                                              source_prime_ids,
                                              target_prime_ids,
                                              divisor_prime_ids));
}

}  // namespace fhelium::rns

namespace fhelium::ckks {

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
key_switch_table_set(const at::Tensor& parameters,
                     c10::IntArrayRef moduli,
                     int64_t log_n,
                     c10::IntArrayRef q_prime_ids,
                     c10::IntArrayRef p_prime_ids,
                     c10::IntArrayRef digits,
                     std::string basis_conversion,
                     std::string moddown_rounding,
                     bool rescale,
                     c10::IntArrayRef drop_prime_ids,
                     c10::IntArrayRef target_prime_ids,
                     std::string rescale_rounding) {
  fhelium::KeySwitchTableSpec spec;
  spec.q_prime_ids = q_prime_ids.vec();
  spec.p_prime_ids = p_prime_ids.vec();
  spec.digits = digits.vec();
  spec.basis_conversion = std::move(basis_conversion);
  spec.moddown_rounding = std::move(moddown_rounding);
  spec.rescale = rescale;
  spec.drop_prime_ids = drop_prime_ids.vec();
  spec.target_prime_ids = target_prime_ids.vec();
  spec.rescale_rounding = std::move(rescale_rounding);
  return fhelium::defined_table_slots(
      fhelium::build_key_switch_tables(basis(parameters, moduli, log_n), spec));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
rotation_sum_table_set(const at::Tensor& parameters,
                       c10::IntArrayRef moduli,
                       int64_t log_n,
                       c10::IntArrayRef q_prime_ids,
                       c10::IntArrayRef p_prime_ids,
                       c10::IntArrayRef next_prime_ids,
                       std::string_view rounding) {
  const auto rns = basis(parameters, moduli, log_n);
  fhelium::KeySwitchTableSpec spec;
  spec.q_prime_ids = q_prime_ids.vec();
  spec.p_prime_ids = p_prime_ids.vec();
  std::vector<int64_t> qp = spec.q_prime_ids;
  qp.insert(qp.end(), p_prime_ids.begin(), p_prime_ids.end());
  fhelium::KeySwitchTables tables;
  tables.qp_parameters = fhelium::select_rns_parameters(rns, qp);
  fhelium::add_rotation_sum_tables(tables, rns, spec, next_prime_ids, rounding);
  tables.qp_parameters = at::Tensor();
  return fhelium::defined_table_slots(std::move(tables));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
encapsulation_table_set(const at::Tensor& parameters,
                        c10::IntArrayRef moduli,
                        int64_t log_n,
                        std::string_view ntt_backend,
                        double scale,
                        c10::IntArrayRef source_q_moduli,
                        c10::IntArrayRef low_p_moduli,
                        int64_t return_prime,
                        c10::IntArrayRef return_p_moduli,
                        c10::IntArrayRef q_moduli,
                        c10::IntArrayRef target_prime_ids,
                        std::string_view rounding) {
  return fhelium::defined_table_slots(
      fhelium::build_encapsulation_tables(basis(parameters, moduli, log_n),
                                          ntt_backend,
                                          scale,
                                          source_q_moduli,
                                          low_p_moduli,
                                          return_prime,
                                          return_p_moduli,
                                          q_moduli,
                                          target_prime_ids,
                                          rounding));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
reconstruction_table_set(const at::Tensor& parameters,
                         c10::IntArrayRef moduli,
                         int64_t log_n,
                         c10::IntArrayRef prime_ids) {
  return fhelium::defined_table_slots(fhelium::build_reconstruction_tables(
      basis(parameters, moduli, log_n), prime_ids));
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
embedding_table_set(int64_t log_n, int64_t generator, c10::Device device) {
  auto [pre, post] = slot_permutations(log_n, generator);
  const auto n = int64_t{1} << log_n;
  return {{"pre", "post", "twister", "skewer"},
          {pre.to(device),
           post.to(device),
           slot_twist(n, false, device),
           slot_twist(n, true, device)}};
}

std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
periodic_table_set(const at::Tensor& parameters,
                   c10::IntArrayRef moduli,
                   int64_t log_n,
                   int64_t period,
                   int64_t generator) {
  const auto device = parameters.device();
  const auto extent = periodic_extent(period, log_n, generator);
  int64_t log_extent = 0;
  while ((int64_t{1} << log_extent) < extent) ++log_extent;
  auto [even, odd, twiddles] = fhelium::ntt::subring_indexed_radix2_plan(
      moduli, log_n, log_extent, parameters.scalar_type());
  fhelium::ntt::validate_indexed_schedule(even, odd, extent);
  twiddles = twiddles.to(device);
  auto rows = twiddles.view({twiddles.size(0), -1});
  fhelium::rns::to_montgomery_(rows, parameters);
  std::vector<int64_t> twice;
  for (const auto modulus : moduli) twice.push_back(2 * modulus);
  return {{"pre",
           "twister",
           "twice_moduli",
           "even_indices",
           "odd_indices",
           "twiddles"},
          {periodic_slot_positions(extent, generator).to(device),
           slot_twist(extent, false, device),
           at::tensor(twice, at::TensorOptions().dtype(at::kLong))
               .to(device, parameters.scalar_type()),
           even.to(device),
           odd.to(device),
           twiddles}};
}

std::vector<at::Tensor> galois_table_set(int64_t log_n,
                                         int64_t galois_element,
                                         c10::Device device) {
  auto tables = fhelium::build_galois_tables(log_n, galois_element, device);
  return {
      tables.ntt_indices, tables.coefficient_indices, tables.coefficient_signs};
}

std::vector<std::string> table_slot_names(std::string_view family,
                                          int64_t digit_count) {
  TORCH_CHECK_VALUE(digit_count >= 0, "digit_count must be nonnegative");
  const auto digits = static_cast<size_t>(digit_count);
  if (family == "ntt") return fhelium::table_slot_names(fhelium::NttTables{});
  if (family == "key_switch") {
    fhelium::KeySwitchTables tables;
    tables.digits.resize(digits);
    return fhelium::table_slot_names(tables);
  }
  if (family == "encapsulation") {
    fhelium::EncapsulationTables tables;
    tables.low.digits.resize(digits);
    return fhelium::table_slot_names(tables);
  }
  if (family == "rescale") {
    return fhelium::table_slot_names(fhelium::RescaleTables{});
  }
  if (family == "basis_extension") {
    return fhelium::table_slot_names(fhelium::BasisExtensionTables{});
  }
  if (family == "trailing_quotient") {
    return fhelium::table_slot_names(fhelium::TrailingQuotientTables{});
  }
  if (family == "ring") return fhelium::table_slot_names(fhelium::RingTables{});
  if (family == "decryption") {
    return fhelium::table_slot_names(fhelium::DecryptionTables{});
  }
  if (family == "secret_conversion") {
    return fhelium::table_slot_names(fhelium::SecretConversionTables{});
  }
  if (family == "galois") {
    return fhelium::table_slot_names(fhelium::GaloisTables{});
  }
  TORCH_CHECK_VALUE(false, "Unknown table family '", family, "'");
}

}  // namespace fhelium::ckks
