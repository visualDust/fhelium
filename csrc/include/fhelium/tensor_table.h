#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

//! Named numerical operands of whole CKKS and RNS operations.
//!
//! A whole operation reads its prepared tables, such as modulus parameters,
//! NTT twiddles and schedules, digit decomposition tables, and quotient
//! inverses, from a `TensorTable` by name. The names are those the FHElium
//! Compile requirement declarations assign to each operation; every function
//! that takes a table lists the names it reads. A Torch operator receives the
//! same table as parallel `str[] table_names` and `Tensor[] tables` arguments.

namespace fhelium {

/// Maps operand names to Tensors.
///
/// Lookup by a missing name raises `c10::ValueError` that names the operand.
/// A table holds Tensor references; it never copies or modifies Tensor data.
class TensorTable {
 public:
  TensorTable() = default;

  /// Pairs `names[i]` with `tensors[i]`; both sequences have one length.
  TensorTable(const std::vector<std::string>& names, at::TensorList tensors) {
    TORCH_CHECK_VALUE(names.size() == tensors.size(),
                      "TensorTable requires one Tensor per name; got ",
                      names.size(),
                      " names and ",
                      tensors.size(),
                      " Tensors");
    for (size_t i = 0; i < names.size(); ++i) {
      entries_.insert_or_assign(names[i], tensors[i]);
    }
  }

  /// Adds or replaces the Tensor named `name`.
  void insert(std::string name, at::Tensor tensor) {
    entries_.insert_or_assign(std::move(name), std::move(tensor));
  }

  /// Returns whether the table holds `name`.
  bool contains(std::string_view name) const {
    return entries_.find(name) != entries_.end();
  }

  /// Returns the Tensor named `name`.
  const at::Tensor& at(std::string_view name) const {
    const auto entry = entries_.find(name);
    TORCH_CHECK_VALUE(
        entry != entries_.end(), "Missing numerical operand '", name, "'");
    return entry->second;
  }

  /// Returns the Tensor named `name`, or no value when the table lacks it.
  std::optional<at::Tensor> get(std::string_view name) const {
    const auto entry = entries_.find(name);
    if (entry == entries_.end()) {
      return std::nullopt;
    }
    return entry->second;
  }

  /// Returns the entries whose names begin with `prefix`, with the prefix
  /// removed from each name.
  TensorTable with_prefix(std::string_view prefix) const {
    TensorTable result;
    for (const auto& [name, tensor] : entries_) {
      if (std::string_view(name).substr(0, prefix.size()) == prefix) {
        result.entries_.emplace(name.substr(prefix.size()), tensor);
      }
    }
    return result;
  }

  /// Returns the entries whose names do not begin with `prefix`.
  TensorTable without_prefix(std::string_view prefix) const {
    TensorTable result;
    for (const auto& [name, tensor] : entries_) {
      if (std::string_view(name).substr(0, prefix.size()) != prefix) {
        result.entries_.emplace(name, tensor);
      }
    }
    return result;
  }

 private:
  std::map<std::string, at::Tensor, std::less<>> entries_;
};

}  // namespace fhelium
