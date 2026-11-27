#pragma once

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <fhelium/export.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

//! Prepared numerical tables of the whole NTT, RNS, and CKKS operations.
//!
//! Each operation family reads its tables from one struct of Tensor fields.
//! A field that the selected method does not read stays undefined. The
//! function `visit_tables` enumerates the fields of a struct in a fixed slot
//! order together with their slot names; a Torch operator receives the same
//! struct as one `Tensor?[] tables` argument in that order, with `None` for
//! undefined fields. Repeated groups, such as the digits of a key switch,
//! follow the fixed fields, so their count follows from the list length.
//!
//! `fhelium::ckks::table_slot_names` lists the slot names of a family. The
//! builders in this header produce the tables from the prime moduli and the RNS
//! parameter table; builders whose result contains NTT tables select those rows
//! from the complete tables of `prepare_ntt_tables`.

namespace fhelium {

/// Name of one table slot: the names of the enclosing groups followed by
/// `text`, an optional decimal `number`, and `suffix`. Visitors build names
/// on the stack; `str()` materializes one only when a caller needs it.
struct TableSlotName {
  const TableSlotName* group = nullptr;
  std::string_view text;
  int64_t number = -1;
  std::string_view suffix;

  std::string str() const {
    std::string name = group == nullptr ? std::string() : group->str();
    name.append(text);
    if (number >= 0) {
      name.append(std::to_string(number));
    }
    name.append(suffix);
    return name;
  }
};

/// Forward or inverse NTT tables of one ordered prime-row set.
///
/// `parameters` holds the RNS parameter columns of the rows. `schedule`
/// holds the policy operands: radix2_indexed uses twiddles and the even and
/// odd butterfly indices; radix2_compact and radix2_compact_shoup32 use
/// twiddles; radix4/8/16_compact use outer twiddles and root powers. The
/// slots are named `<basis>_<direction>_0` (parameters) through `_3`.
struct NttTables {
  at::Tensor parameters;
  at::Tensor schedule[3];
};

/// CRT basis-extension or centered-quotient matrix and its operands; see
/// `fhelium::rns::crt_basis_extend`. An empty `source_normalizers` selects the
/// raw lift.
struct CrtTables {
  at::Tensor source_parameters;
  at::Tensor target_parameters;
  at::Tensor matrix;
  at::Tensor target_reciprocals;
  at::Tensor source_normalizers;
  at::Tensor source_reciprocals;
  at::Tensor target_offsets;
};

/// Lift of one hybrid key-switch digit to the QP rows. The exact method
/// reads the Garner tables (`normalizers`, `propagation`, and the four
/// Montgomery reduction rows for digits of several primes) and `extension`;
/// the CRT methods read `crt`. Slots are prefixed `digit<i>_`.
struct DigitTables {
  at::Tensor normalizers;
  at::Tensor propagation;
  at::Tensor reduction[4];
  at::Tensor extension;
  CrtTables crt;
};

/// Quotient of QP rows by D·P that a relinearization with rescale fuses into
/// its ModDown. Slots are prefixed `joint_`; `remaining_forward` serves NTT
/// output and `remaining_inverse` coefficient output.
struct JointRescaleTables {
  at::Tensor remaining_parameters;
  at::Tensor dropped_parameters;
  at::Tensor ordered_parameters;
  at::Tensor p_times_base;
  at::Tensor inverses;
  at::Tensor product_inverse;
  at::Tensor dropped_offset;
  at::Tensor remaining_offset;
  at::Tensor drop_positions;
  at::Tensor remaining_positions;
  at::Tensor restore;
  at::Tensor product_parameters;
  at::Tensor target_keep_positions;
  CrtTables crt_d;
  CrtTables crt_t;
  at::Tensor crt_p_inverse;
  at::Tensor crt_inverses;
  at::Tensor crt_ordered_parameters;
  at::Tensor crt_half;
  NttTables dropped_inverse;
  NttTables remaining_forward;
  NttTables remaining_inverse;
};

/// Tables of the hybrid key-switching family: ModDown, joint rescale,
/// key switching, rotations, and rotation sums.
///
/// The Q rows are the active rows of the operand and the P rows the
/// auxiliary rows of the key. `moddown_inverses`, `p_inverse`, `moddown_half`
/// and `moddown_crt` divide QP rows by P; `joint` replaces them for a
/// relinearization with rescale. The `sum_` fields complete a rotation sum:
/// its quotient rows are ordered surviving Q, dropped Q, P.
struct KeySwitchTables {
  at::Tensor q_parameters;
  at::Tensor qp_parameters;
  at::Tensor moddown_inverses;
  at::Tensor p_inverse;
  at::Tensor moddown_half;
  CrtTables moddown_crt;
  JointRescaleTables joint;
  NttTables q_forward;
  NttTables q_inverse;
  NttTables qp_forward;
  NttTables qp_inverse;
  NttTables p_forward;
  NttTables p_inverse_ntt;
  at::Tensor sum_p_montgomery;
  at::Tensor sum_quotient_parameters;
  at::Tensor sum_quotient_inverses;
  at::Tensor sum_quotient_order;
  at::Tensor sum_quotient_half;
  std::vector<DigitTables> digits;
};

/// Rational rescale from basis B to B'; see `fhelium::rns::rational_rescale`.
struct RescaleTables {
  at::Tensor source_parameters;
  at::Tensor target_parameters;
  at::Tensor dropped_parameters;
  at::Tensor ordered_parameters;
  at::Tensor drop_positions;
  at::Tensor keep_positions;
  at::Tensor inverses;
  at::Tensor dropped_offset;
  at::Tensor target_offset;
  at::Tensor target_keep_positions;
  at::Tensor restore;
  at::Tensor product_inverse;
  NttTables dropped_inverse;
  NttTables target_forward;
};

/// Centered extension from a source basis to target rows; see
/// `fhelium::rns::centered_basis_extend`.
struct BasisExtensionTables {
  at::Tensor source_parameters;
  at::Tensor target_parameters;
  at::Tensor source_lo;
  at::Tensor source_hi;
  at::Tensor source_neg_lo;
  at::Tensor source_neg_hi;
  at::Tensor normalizers;
  at::Tensor propagation;
  at::Tensor extension;
  at::Tensor half_digits;
  at::Tensor source_product;
  at::Tensor target_moduli;
};

/// Rounded quotient by trailing rows; see
/// `fhelium::rns::rounded_trailing_quotient`.
struct TrailingQuotientTables {
  at::Tensor parameters;
  at::Tensor inverses;
  at::Tensor half;
  at::Tensor row_order;
};

/// Low key switch, centered extension, and guard-prime return quotient of
/// `fhelium::ckks::encapsulated_modulus_raise`. The return rows are
/// [guard, Q, P_return]; quotient rows are ordered [Q, guard, P_return].
struct EncapsulationTables {
  KeySwitchTables low;
  BasisExtensionTables raise;
  NttTables return_forward;
  NttTables return_inverse;
  at::Tensor quotient_order;
  at::Tensor quotient_parameters;
  at::Tensor quotient_inverses;
  at::Tensor quotient_half;
  at::Tensor target_parameters;
};

/// NTT tables of one basis for encryption, decryption, and key generation.
/// `source_product` holds the hybrid key factor of
/// `fhelium::ckks::create_key_switch_key`.
struct RingTables {
  NttTables forward;
  NttTables inverse;
  at::Tensor source_product;
};

/// Phase tables and the centered Q reconstruction of decryption.
struct DecryptionTables {
  RingTables ring;
  at::Tensor reconstruction_parameters;
  at::Tensor normalizers;
  at::Tensor propagation;
  at::Tensor half_digits;
};

/// Re-expression of a secret key in another basis.
struct SecretConversionTables {
  NttTables source_inverse;
  BasisExtensionTables extension;
  NttTables target_forward;
};

/// Gathers of the Galois automorphism X -> X^g for one rotation step:
/// NTT-domain source indices and coefficient-domain indices with signs.
/// Operations receive them as ordinary operands rather than caching device
/// copies, because a first-use host-to-device copy cannot run during CUDA
/// Graph capture.
struct GaloisTables {
  at::Tensor ntt_indices;
  at::Tensor coefficient_indices;
  at::Tensor coefficient_signs;
};

// ---------------------------------------------------------------------------
// Slot enumeration

namespace detail_tables {

template <typename F>
void slot(F& f,
          const TableSlotName* group,
          std::string_view name,
          at::Tensor& tensor) {
  f(TableSlotName{group, name}, tensor);
}

}  // namespace detail_tables

template <typename F>
void visit_tables(NttTables& tables, const TableSlotName* group, F& f) {
  for (int64_t index = 0; index < 4; ++index) {
    f(TableSlotName{group, "", index},
      index == 0 ? tables.parameters : tables.schedule[index - 1]);
  }
}

template <typename F>
void visit_tables(CrtTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "source_parameters", t.source_parameters);
  slot(f, group, "target_parameters", t.target_parameters);
  slot(f, group, "matrix", t.matrix);
  slot(f, group, "target_reciprocals", t.target_reciprocals);
  slot(f, group, "source_normalizers", t.source_normalizers);
  slot(f, group, "source_reciprocals", t.source_reciprocals);
  slot(f, group, "target_offsets", t.target_offsets);
}

template <typename F>
void visit_tables(DigitTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "normalizers", t.normalizers);
  slot(f, group, "propagation", t.propagation);
  for (int64_t part = 0; part < 4; ++part) {
    f(TableSlotName{group, "reduction", part}, t.reduction[part]);
  }
  slot(f, group, "extension", t.extension);
  const TableSlotName crt{group, "crt_"};
  visit_tables(t.crt, &crt, f);
}

template <typename F>
void visit_tables(JointRescaleTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  const TableSlotName joint{group, "joint_"};
  slot(f, &joint, "remaining_parameters", t.remaining_parameters);
  slot(f, &joint, "dropped_parameters", t.dropped_parameters);
  slot(f, &joint, "ordered_parameters", t.ordered_parameters);
  slot(f, &joint, "p_times_base", t.p_times_base);
  slot(f, &joint, "inverses", t.inverses);
  slot(f, &joint, "product_inverse", t.product_inverse);
  slot(f, &joint, "dropped_offset", t.dropped_offset);
  slot(f, &joint, "remaining_offset", t.remaining_offset);
  slot(f, &joint, "drop_positions", t.drop_positions);
  slot(f, &joint, "remaining_positions", t.remaining_positions);
  slot(f, &joint, "restore", t.restore);
  slot(f, &joint, "product_parameters", t.product_parameters);
  slot(f, &joint, "target_keep_positions", t.target_keep_positions);
  const TableSlotName crt_d{&joint, "crt_d_"};
  visit_tables(t.crt_d, &crt_d, f);
  const TableSlotName crt_t{&joint, "crt_t_"};
  visit_tables(t.crt_t, &crt_t, f);
  slot(f, &joint, "crt_p_inverse", t.crt_p_inverse);
  slot(f, &joint, "crt_inverses", t.crt_inverses);
  slot(f, &joint, "crt_ordered_parameters", t.crt_ordered_parameters);
  slot(f, &joint, "crt_half", t.crt_half);
  const TableSlotName dropped{&joint, "dropped_inverse_"};
  visit_tables(t.dropped_inverse, &dropped, f);
  const TableSlotName forward{&joint, "remaining_forward_"};
  visit_tables(t.remaining_forward, &forward, f);
  const TableSlotName inverse{&joint, "remaining_inverse_"};
  visit_tables(t.remaining_inverse, &inverse, f);
}

template <typename F>
void visit_tables(KeySwitchTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "q_parameters", t.q_parameters);
  slot(f, group, "qp_parameters", t.qp_parameters);
  slot(f, group, "moddown_inverses", t.moddown_inverses);
  slot(f, group, "p_inverse", t.p_inverse);
  slot(f, group, "moddown_half", t.moddown_half);
  const TableSlotName crt{group, "moddown_crt_"};
  visit_tables(t.moddown_crt, &crt, f);
  visit_tables(t.joint, group, f);
  const TableSlotName q_forward{group, "q_forward_"};
  visit_tables(t.q_forward, &q_forward, f);
  const TableSlotName q_inverse{group, "q_inverse_"};
  visit_tables(t.q_inverse, &q_inverse, f);
  const TableSlotName qp_forward{group, "qp_forward_"};
  visit_tables(t.qp_forward, &qp_forward, f);
  const TableSlotName qp_inverse{group, "qp_inverse_"};
  visit_tables(t.qp_inverse, &qp_inverse, f);
  const TableSlotName p_forward{group, "p_forward_"};
  visit_tables(t.p_forward, &p_forward, f);
  const TableSlotName p_inverse{group, "p_inverse_"};
  visit_tables(t.p_inverse_ntt, &p_inverse, f);
  slot(f, group, "sum_p_montgomery", t.sum_p_montgomery);
  slot(f, group, "sum_quotient_parameters", t.sum_quotient_parameters);
  slot(f, group, "sum_quotient_inverses", t.sum_quotient_inverses);
  slot(f, group, "sum_quotient_order", t.sum_quotient_order);
  slot(f, group, "sum_quotient_half", t.sum_quotient_half);
  for (size_t index = 0; index < t.digits.size(); ++index) {
    const TableSlotName digit{group, "digit", static_cast<int64_t>(index), "_"};
    visit_tables(t.digits[index], &digit, f);
  }
}

template <typename F>
void visit_tables(RescaleTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "source_parameters", t.source_parameters);
  slot(f, group, "target_parameters", t.target_parameters);
  slot(f, group, "dropped_parameters", t.dropped_parameters);
  slot(f, group, "ordered_parameters", t.ordered_parameters);
  slot(f, group, "drop_positions", t.drop_positions);
  slot(f, group, "keep_positions", t.keep_positions);
  slot(f, group, "inverses", t.inverses);
  slot(f, group, "dropped_offset", t.dropped_offset);
  slot(f, group, "target_offset", t.target_offset);
  slot(f, group, "target_keep_positions", t.target_keep_positions);
  slot(f, group, "restore", t.restore);
  slot(f, group, "product_inverse", t.product_inverse);
  const TableSlotName dropped{group, "dropped_inverse_"};
  visit_tables(t.dropped_inverse, &dropped, f);
  const TableSlotName target{group, "target_forward_"};
  visit_tables(t.target_forward, &target, f);
}

template <typename F>
void visit_tables(BasisExtensionTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "source_parameters", t.source_parameters);
  slot(f, group, "target_parameters", t.target_parameters);
  slot(f, group, "source_lo", t.source_lo);
  slot(f, group, "source_hi", t.source_hi);
  slot(f, group, "source_neg_lo", t.source_neg_lo);
  slot(f, group, "source_neg_hi", t.source_neg_hi);
  slot(f, group, "normalizers", t.normalizers);
  slot(f, group, "propagation", t.propagation);
  slot(f, group, "extension", t.extension);
  slot(f, group, "half_digits", t.half_digits);
  slot(f, group, "source_product", t.source_product);
  slot(f, group, "target_moduli", t.target_moduli);
}

template <typename F>
void visit_tables(TrailingQuotientTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "parameters", t.parameters);
  slot(f, group, "moddown_inverses", t.inverses);
  slot(f, group, "half", t.half);
  slot(f, group, "row_order", t.row_order);
}

template <typename F>
void visit_tables(EncapsulationTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  const TableSlotName raise{group, "raise_"};
  visit_tables(t.raise, &raise, f);
  const TableSlotName forward{group, "return_forward_"};
  visit_tables(t.return_forward, &forward, f);
  const TableSlotName inverse{group, "return_inverse_"};
  visit_tables(t.return_inverse, &inverse, f);
  slot(f, group, "quotient_order", t.quotient_order);
  slot(f, group, "quotient_parameters", t.quotient_parameters);
  slot(f, group, "quotient_inverses", t.quotient_inverses);
  slot(f, group, "quotient_half", t.quotient_half);
  slot(f, group, "target_parameters", t.target_parameters);
  // The low key switch carries the repeated digit group, so it comes last.
  const TableSlotName low{group, "low_"};
  visit_tables(t.low, &low, f);
}

template <typename F>
void visit_tables(RingTables& t, const TableSlotName* group, F& f) {
  const TableSlotName forward{group, "forward_"};
  visit_tables(t.forward, &forward, f);
  const TableSlotName inverse{group, "inverse_"};
  visit_tables(t.inverse, &inverse, f);
  detail_tables::slot(f, group, "source_product", t.source_product);
}

template <typename F>
void visit_tables(DecryptionTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  visit_tables(t.ring, group, f);
  slot(f, group, "reconstruction_parameters", t.reconstruction_parameters);
  slot(f, group, "normalizers", t.normalizers);
  slot(f, group, "propagation", t.propagation);
  slot(f, group, "half_digits", t.half_digits);
}

template <typename F>
void visit_tables(SecretConversionTables& t, const TableSlotName* group, F& f) {
  const TableSlotName source{group, "source_inverse_"};
  visit_tables(t.source_inverse, &source, f);
  visit_tables(t.extension, group, f);
  const TableSlotName target{group, "target_forward_"};
  visit_tables(t.target_forward, &target, f);
}

template <typename F>
void visit_tables(GaloisTables& t, const TableSlotName* group, F& f) {
  using detail_tables::slot;
  slot(f, group, "ntt_indices", t.ntt_indices);
  slot(f, group, "coefficient_indices", t.coefficient_indices);
  slot(f, group, "coefficient_signs", t.coefficient_signs);
}

// ---------------------------------------------------------------------------
// Positional form

// Positional slots of a table struct, as passed to a `Tensor?[]` operand.
using TableSlots = c10::List<std::optional<at::Tensor>>;

// Returns the number of slots of `tables`.
template <typename T>
size_t table_slot_count(T tables) {
  size_t count = 0;
  auto counter = [&](const TableSlotName&, at::Tensor&) { ++count; };
  visit_tables(tables, nullptr, counter);
  return count;
}

namespace detail_tables {

// Sizes the repeated group of `tables` for `slots` positional slots.
inline void resize_for_slots(KeySwitchTables& tables, size_t slots) {
  const auto fixed = table_slot_count(KeySwitchTables{});
  const auto digit = table_slot_count(DigitTables{});
  TORCH_CHECK_VALUE(slots >= fixed && (slots - fixed) % digit == 0,
                    "Key-switch tables require ",
                    fixed,
                    " slots and ",
                    digit,
                    " slots per digit; got ",
                    slots);
  tables.digits.resize((slots - fixed) / digit);
}

inline void resize_for_slots(EncapsulationTables& tables, size_t slots) {
  const auto fixed = table_slot_count(EncapsulationTables{}) -
                     table_slot_count(KeySwitchTables{});
  TORCH_CHECK_VALUE(slots >= fixed,
                    "Encapsulation tables require at least ",
                    fixed,
                    " slots; got ",
                    slots);
  resize_for_slots(tables.low, slots - fixed);
}

template <typename T>
void resize_for_slots(T&, size_t) {}

}  // namespace detail_tables

// Returns the positional slots of `tables`; undefined fields become `None`.
template <typename T>
TableSlots flatten_tables(T tables) {
  TableSlots slots;
  slots.reserve(table_slot_count(tables));
  auto append = [&](const TableSlotName&, at::Tensor& tensor) {
    slots.push_back(tensor.defined() ? std::optional<at::Tensor>(tensor)
                                     : std::nullopt);
  };
  visit_tables(tables, nullptr, append);
  return slots;
}

// Returns the table struct of positional `slots`.
template <typename T>
T unflatten_tables(const TableSlots& slots) {
  T tables;
  detail_tables::resize_for_slots(tables, slots.size());
  size_t index = 0;
  auto assign = [&](const TableSlotName&, at::Tensor& tensor) {
    TORCH_CHECK_VALUE(
        index < slots.size(), "Table list has too few slots: ", slots.size());
    const std::optional<at::Tensor> value = slots.get(index++);
    if (value.has_value()) {
      tensor = *value;
    }
  };
  visit_tables(tables, nullptr, assign);
  TORCH_CHECK_VALUE(index == slots.size(),
                    "Table list has ",
                    slots.size(),
                    " slots; expected ",
                    index);
  return tables;
}

// Returns the slot names of `tables` in positional order.
template <typename T>
std::vector<std::string> table_slot_names(T tables) {
  std::vector<std::string> names;
  auto append = [&](const TableSlotName& name, at::Tensor&) {
    names.push_back(name.str());
  };
  visit_tables(tables, nullptr, append);
  return names;
}

// Returns the names and Tensors of the defined fields of `tables`.
template <typename T>
std::tuple<std::vector<std::string>, std::vector<at::Tensor>>
defined_table_slots(T tables) {
  std::vector<std::string> names;
  std::vector<at::Tensor> tensors;
  auto append = [&](const TableSlotName& name, at::Tensor& tensor) {
    if (tensor.defined()) {
      names.push_back(name.str());
      tensors.push_back(tensor);
    }
  };
  visit_tables(tables, nullptr, append);
  return {std::move(names), std::move(tensors)};
}

/// Returns the Galois tables of each step from `slots`, three per step.
FHELIUM_API std::vector<GaloisTables> unflatten_galois_tables(
    const TableSlots& slots);

// ---------------------------------------------------------------------------
// Builders

/// Configured primes of one RNS context and their parameter table.
///
/// `moduli` lists every prime in context order; `parameters` is
/// `fhelium::rns::parameter_table(moduli, ...)` on the execution device. Its
/// dtype is the residue dtype of every built table.
struct RnsBasis {
  std::vector<int64_t> moduli;
  at::Tensor parameters;
  int64_t ring_dimension = 0;
};

/// Complete NTT tables of every prime of an `RnsBasis`, in the
/// `NttTables` operand order of `ntt_backend`. Twiddles are converted to
/// Montgomery form on the parameter device, except the Shoup32 pairs, which
/// keep their standard roots and quotient words.
struct NttContextTables {
  std::string ntt_backend;
  NttTables forward;
  NttTables inverse;
};

FHELIUM_API NttContextTables prepare_ntt_tables(const RnsBasis& basis,
                                                std::string_view ntt_backend);

/// Returns the parameter columns of `prime_ids`: a view for one increasing
/// interval and a gathered copy otherwise.
FHELIUM_API at::Tensor select_rns_parameters(const RnsBasis& basis,
                                             c10::IntArrayRef prime_ids);

/// Returns the NTT tables of `prime_ids` in one direction, selecting rows of
/// the complete tables. Twiddle rows are contiguous; index schedules are
/// shared by every row set.
FHELIUM_API NttTables select_ntt_tables(const RnsBasis& basis,
                                        const NttContextTables& tables,
                                        c10::IntArrayRef prime_ids,
                                        bool inverse);

/// Rows and method of one key-switch table set.
///
/// `digits` lists (start, stop, key_digit) triples of positions in
/// `q_prime_ids`. With `rescale`, `drop_prime_ids` (a subset of Q) is
/// divided out with P and `target_prime_ids` orders the result rows.
struct KeySwitchTableSpec {
  std::vector<int64_t> q_prime_ids;
  std::vector<int64_t> p_prime_ids;
  std::vector<int64_t> digits;
  std::string basis_conversion = "exact";
  std::string moddown_rounding = "nearest";
  bool rescale = false;
  std::vector<int64_t> drop_prime_ids;
  std::vector<int64_t> target_prime_ids;
  std::string rescale_rounding = "nearest";
  std::string output_domain = "coefficient";
};

/// Builds the RNS tables of one key-switch configuration: parameters,
/// ModDown or joint-rescale quotient tables, and digit lifts. NTT fields stay
/// undefined; `fill_key_switch_ntt_tables` selects them.
FHELIUM_API KeySwitchTables
build_key_switch_tables(const RnsBasis& basis, const KeySwitchTableSpec& spec);

/// Selects the NTT fields of `tables` that the configuration reads.
FHELIUM_API void fill_key_switch_ntt_tables(KeySwitchTables& tables,
                                            const RnsBasis& basis,
                                            const NttContextTables& ntt,
                                            const KeySwitchTableSpec& spec);

/// Adds the rotation-sum quotient: with `rescale`, the leading Q depth group
/// `next_prime_ids` survives and the other Q rows are divided out with P.
FHELIUM_API void add_rotation_sum_tables(KeySwitchTables& tables,
                                         const RnsBasis& basis,
                                         const KeySwitchTableSpec& spec,
                                         c10::IntArrayRef next_prime_ids,
                                         std::string_view rounding);

/// Builds the quotient of R(v + Pa) + H_P + P H_D by P·D for Q rows `q`,
/// auxiliary rows `p`, and the dropped Q subset `drop_prime_ids`.
/// `target_prime_ids` orders the result rows and defaults (when empty) to the
/// remaining Q rows; its rows outside Q form the restored set R. With
/// `normalized_crt` conversion the `crt_` fields replace the exact P
/// remainder. NTT fields stay undefined.
FHELIUM_API JointRescaleTables
build_joint_rescale_tables(const RnsBasis& basis,
                           c10::IntArrayRef q_prime_ids,
                           c10::IntArrayRef p_prime_ids,
                           c10::IntArrayRef drop_prime_ids,
                           c10::IntArrayRef target_prime_ids,
                           std::string_view moddown_rounding,
                           std::string_view rescale_rounding,
                           std::string_view basis_conversion);

/// Builds the rational rescale tables from `source_prime_ids` to
/// `target_prime_ids`. `ntt_input` adds the inverse of the dropped product
/// on the target rows; supplying `ntt` also selects the dropped inverse and
/// target forward NTT tables.
FHELIUM_API RescaleTables
build_rescale_tables(const RnsBasis& basis,
                     c10::IntArrayRef source_prime_ids,
                     c10::IntArrayRef target_prime_ids,
                     std::string_view rounding,
                     bool ntt_input,
                     const NttContextTables* ntt);

/// Builds the centered basis-extension tables between two contexts that share
/// dtype and device.
FHELIUM_API BasisExtensionTables
build_basis_extension_tables(const RnsBasis& source,
                             const RnsBasis& target,
                             c10::IntArrayRef source_prime_ids,
                             c10::IntArrayRef target_prime_ids);

/// Builds the CRT lift of `source_prime_ids` to `target_prime_ids`, raw or
/// normalized; see `fhelium::rns::crt_basis_extension_tables`.
FHELIUM_API CrtTables build_crt_tables(const RnsBasis& basis,
                                       c10::IntArrayRef source_prime_ids,
                                       c10::IntArrayRef target_prime_ids,
                                       bool normalized);

/// Builds the centered CRT quotient of `source_prime_ids` onto
/// `target_prime_ids`, further divided by the product of `divisor_prime_ids`.
FHELIUM_API CrtTables
build_centered_quotient_tables(const RnsBasis& basis,
                               c10::IntArrayRef source_prime_ids,
                               c10::IntArrayRef target_prime_ids,
                               c10::IntArrayRef divisor_prime_ids);

/// Builds the tables of `encapsulated_modulus_raise` from Q basis
/// `source_q_moduli`, low auxiliary primes, and the return basis [guard, Q, P].
/// `q_moduli` is the full Q chain; `target_prime_ids` selects the output rows
/// of `basis`.
FHELIUM_API EncapsulationTables
build_encapsulation_tables(const RnsBasis& basis,
                           std::string_view ntt_backend,
                           double scale,
                           c10::IntArrayRef source_q_moduli,
                           c10::IntArrayRef low_p_moduli,
                           int64_t return_prime,
                           c10::IntArrayRef return_p_moduli,
                           c10::IntArrayRef q_moduli,
                           c10::IntArrayRef target_prime_ids,
                           std::string_view rounding);

/// Builds the NTT tables of `prime_ids` in both directions; `source_product`
/// stays undefined.
FHELIUM_API RingTables build_ring_tables(const RnsBasis& basis,
                                         const NttContextTables& ntt,
                                         c10::IntArrayRef prime_ids);

/// Builds the centered Q reconstruction of decryption for `prime_ids`; the
/// ring fields stay undefined.
FHELIUM_API DecryptionTables
build_reconstruction_tables(const RnsBasis& basis, c10::IntArrayRef prime_ids);

/// Builds the ring and reconstruction tables of decryption for `prime_ids`.
FHELIUM_API DecryptionTables
build_decryption_tables(const RnsBasis& basis,
                        const NttContextTables& ntt,
                        c10::IntArrayRef prime_ids);

/// Builds the Galois gathers of X -> X^`galois_element` on `device`.
FHELIUM_API GaloisTables build_galois_tables(int64_t log_n,
                                             int64_t galois_element,
                                             const at::Device& device);

}  // namespace fhelium
