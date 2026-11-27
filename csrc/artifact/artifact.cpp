#include "fhelium/artifact.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "../runner/kernel_loader.h"
#include "container_detail.h"
#include "fhelium/container.h"

namespace fhelium {
namespace {

using exec::Arena;
using exec::KernelRecord;
using exec::Runner;
using nlohmann::json;

// Built tables of the artifacts in the process, by the key of the builder
// calls that produce them (see `Artifact::State::table_keys`), so artifacts
// that declare the same table bind one copy. Entries hold storage weakly: a
// table is freed once no artifact or other owner holds it.
class TableCache {
 public:
  at::Tensor find(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = tables_.find(key);
    if (found == tables_.end()) return {};
    auto storage = found->second.storage.lock();
    if (!storage) {
      tables_.erase(found);
      return {};
    }
    const auto& table = found->second;
    return at::empty({0}, table.options)
        .set_(c10::Storage(std::move(storage)),
              table.offset,
              table.sizes,
              table.strides);
  }

  void insert(const std::string& key, const at::Tensor& value) {
    if (!value.defined() || !value.has_storage()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (tables_.size() >= 2 * live_) {
      for (auto it = tables_.begin(); it != tables_.end();)
        it = it->second.storage.expired() ? tables_.erase(it) : std::next(it);
      live_ = std::max<size_t>(tables_.size(), 64);
    }
    tables_.insert_or_assign(key,
                             Table{value.storage().getWeakStorageImpl(),
                                   value.storage_offset(),
                                   value.sizes().vec(),
                                   value.strides().vec(),
                                   value.options()});
  }

 private:
  struct Table {
    c10::weak_intrusive_ptr<c10::StorageImpl> storage;
    int64_t offset;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    c10::TensorOptions options;
  };
  std::mutex mutex_;
  std::unordered_map<std::string, Table> tables_;
  size_t live_ = 64;
};

TableCache& table_cache() {
  static TableCache cache;
  return cache;
}

// Text of a constant builder argument for a table key. Doubles are written
// by their bits so distinct values never share a key.
void describe(std::ostringstream& out, const c10::IValue& value) {
  if (value.isNone()) {
    out << "None";
  } else if (value.isBool()) {
    out << (value.toBool() ? "True" : "False");
  } else if (value.isInt()) {
    out << value.toInt();
  } else if (value.isDouble()) {
    const double number = value.toDouble();
    uint64_t bits;
    std::memcpy(&bits, &number, sizeof bits);
    out << "f" << bits;
  } else if (value.isString()) {
    out << '"' << value.toStringRef() << '"';
  } else if (value.isDevice()) {
    out << "device " << value.toDevice();
  } else if (value.isList()) {
    out << "[";
    for (const auto& item : value.toListRef()) {
      describe(out, item);
      out << ",";
    }
    out << "]";
  } else if (value.isTuple()) {
    out << "(";
    for (const auto& item : value.toTupleRef().elements()) {
      describe(out, item);
      out << ",";
    }
    out << ")";
  } else {
    out << value.tagKind() << " " << value;
  }
}

constexpr const char* kFormat = "fhelium-artifact";
constexpr const char* kSchemaVersion = "1";

c10::ScalarType torch_dtype(const std::string& name) {
  static const std::map<std::string, c10::ScalarType> types = {
      {"torch.bool", c10::ScalarType::Bool},
      {"torch.uint8", c10::ScalarType::Byte},
      {"torch.int8", c10::ScalarType::Char},
      {"torch.int16", c10::ScalarType::Short},
      {"torch.uint16", c10::ScalarType::UInt16},
      {"torch.int32", c10::ScalarType::Int},
      {"torch.uint32", c10::ScalarType::UInt32},
      {"torch.int64", c10::ScalarType::Long},
      {"torch.uint64", c10::ScalarType::UInt64},
      {"torch.float16", c10::ScalarType::Half},
      {"torch.bfloat16", c10::ScalarType::BFloat16},
      {"torch.float32", c10::ScalarType::Float},
      {"torch.float64", c10::ScalarType::Double},
      {"torch.complex64", c10::ScalarType::ComplexFloat},
      {"torch.complex128", c10::ScalarType::ComplexDouble},
  };
  const auto found = types.find(name);
  TORCH_CHECK_VALUE(found != types.end(), "Unsupported artifact dtype ", name);
  return found->second;
}

c10::MemoryFormat memory_format(const std::string& name) {
  if (name == "contiguous_format") return c10::MemoryFormat::Contiguous;
  if (name == "preserve_format") return c10::MemoryFormat::Preserve;
  if (name == "channels_last") return c10::MemoryFormat::ChannelsLast;
  TORCH_CHECK_VALUE(
      name == "channels_last_3d", "Unsupported artifact memory format ", name);
  return c10::MemoryFormat::ChannelsLast3d;
}

c10::Layout layout(const std::string& name) {
  if (name == "strided") return c10::kStrided;
  if (name == "sparse_coo") return c10::kSparse;
  TORCH_CHECK_VALUE(name == "_mkldnn", "Unsupported artifact layout ", name);
  return c10::kMkldnn;
}

// Optional device of a buffer or argument, with CUDA devices moved to the
// artifact's device.
c10::Device placed(const std::string& name, const c10::Device& device) {
  c10::Device parsed(name);
  if (parsed.type() == device.type() && !parsed.is_cpu()) return device;
  return parsed;
}

// Device of a constant or material record: the host for "cpu", otherwise the
// artifact's device.
c10::Device record_device(const json& record, const c10::Device& device) {
  return record.value("device", std::string()) == "cpu" ? c10::Device(c10::kCPU)
                                                        : device;
}

// A copy of stored tensor `tensor` on `device` with its own storage; stored
// tensors are views of the container's one host buffer.
at::Tensor own(const at::Tensor& tensor, const c10::Device& device) {
  return device.is_cpu() ? tensor.clone() : tensor.to(device);
}

struct BufferSpec {
  std::string storage;
  std::optional<std::vector<int64_t>> sizes;
  std::optional<std::vector<int64_t>> strides;
  std::optional<c10::ScalarType> dtype;
  std::optional<c10::Device> device;
};

struct Call {
  std::string method;
  std::vector<c10::IValue> arguments;
};

// One entry, decoded once per artifact.
struct Entry {
  std::string name;
  std::vector<BufferSpec> buffers;
  std::vector<Call> calls;
  int64_t int_registers = 0;
  int64_t float_registers = 0;
  std::vector<int64_t> input_buffers;
  std::vector<int64_t> read_in_place;
  std::vector<int64_t> input_ints;
  std::vector<int64_t> input_floats;
  std::vector<int64_t> output_buffers;
  // (buffer, material symbol) and (buffer, constant index)
  std::vector<std::pair<int64_t, std::string>> materials;
  std::vector<std::pair<int64_t, int64_t>> constants;
  // (symbol, size) by group id
  std::vector<std::pair<std::string, int64_t>> groups;
  bool graphs = false;
};

class Reader {
 public:
  Reader(const at::Tensor& stream,
         const std::vector<std::string>& strings,
         const c10::Device& device,
         const std::string& entry)
      : data_(stream.const_data_ptr<int64_t>()),
        end_(data_ + stream.numel()),
        strings_(strings),
        device_(device),
        entry_(entry) {}

  int64_t next() {
    TORCH_CHECK_VALUE(data_ < end_, "Artifact entry ", entry_, " is truncated");
    return *data_++;
  }

  const std::string& string() {
    const auto index = next();
    TORCH_CHECK_VALUE(
        index >= 0 && index < static_cast<int64_t>(strings_.size()),
        "Artifact entry ",
        entry_,
        " refers to string ",
        index);
    return strings_[index];
  }

  std::optional<std::vector<int64_t>> optional_integers() {
    const auto count = next();
    if (count < 0) return std::nullopt;
    std::vector<int64_t> values(count);
    for (auto& value : values) value = next();
    return values;
  }

  c10::IValue value() {
    switch (next()) {
      case 0:
        return c10::IValue();
      case 1:
        return next();
      case 2: {
        const auto bits = next();
        double value;
        std::memcpy(&value, &bits, sizeof value);
        return value;
      }
      case 3:
        return next() != 0;
      case 4:
        return string();
      case 5:
        return list();
      case 6:
        return torch_dtype(string());
      case 7:
        return placed(string(), device_);
      case 8:
        return c10::IValue(next());
      case 9:
        return string();
      case 10:
        return memory_format(string());
      case 11:
        return layout(string());
      default:
        TORCH_CHECK_VALUE(false, "Artifact entry ", entry_, " has a bad tag");
    }
  }

  bool done() const { return data_ == end_; }

 private:
  // A list of ints, floats, bools or strings becomes a typed list; any other
  // list, such as operator or launch arguments of mixed kinds, a generic one.
  c10::IValue list() {
    const auto count = next();
    std::vector<c10::IValue> items;
    for (int64_t i = 0; i < count; ++i) items.push_back(value());
    const auto all = [&](auto predicate) {
      return std::all_of(items.begin(), items.end(), predicate);
    };
    if (all([](const c10::IValue& item) { return item.isInt(); })) {
      c10::List<int64_t> result;
      for (const auto& item : items) result.push_back(item.toInt());
      return result;
    }
    if (all([](const c10::IValue& item) { return item.isDouble(); })) {
      c10::List<double> result;
      for (const auto& item : items) result.push_back(item.toDouble());
      return result;
    }
    if (all([](const c10::IValue& item) { return item.isBool(); })) {
      c10::List<bool> result;
      for (const auto& item : items) result.push_back(item.toBool());
      return result;
    }
    if (all([](const c10::IValue& item) { return item.isString(); })) {
      c10::List<std::string> result;
      for (const auto& item : items) result.push_back(item.toStringRef());
      return result;
    }
    c10::impl::GenericList result(c10::AnyType::get());
    for (auto& item : items) result.push_back(std::move(item));
    return result;
  }

  const int64_t* data_;
  const int64_t* end_;
  const std::vector<std::string>& strings_;
  c10::Device device_;
  std::string entry_;
};

std::vector<int64_t> integers(const c10::IValue& value) {
  return value.toIntVector();
}

struct Material {
  json record;
  at::Tensor value;
};

}  // namespace

struct Artifact::State {
  c10::Device device = c10::Device(c10::kCPU);
  json manifest;
  std::map<std::string, at::Tensor> tensors;  // on the host
  std::vector<KernelRecord> kernels;
  std::vector<at::Tensor> constants;
  std::map<std::string, Material> materials;
  std::map<std::string, Entry> entries;
  std::map<std::string, c10::intrusive_ptr<Runner>> callees;
  std::map<std::string, c10::intrusive_ptr<c10d::ProcessGroup>> groups;
  // Runners built with groups, for release_groups.
  std::vector<c10::weak_intrusive_ptr<Runner>> grouped;
  // Runners of the entries that build and prepare materials.
  std::map<std::string, c10::intrusive_ptr<Runner>> preparers;
  bool prepared = false;
  std::map<std::string, std::string> digests;  // stored tensor digests

  const c10::intrusive_ptr<Runner>& preparer(const std::string& name) {
    auto found = preparers.find(name);
    if (found == preparers.end())
      found = preparers.emplace(name, build(name, std::nullopt, "off", false))
                  .first;
    return found->second;
  }

  const at::Tensor& tensor(const std::string& name) const {
    const auto found = tensors.find(name);
    TORCH_CHECK_VALUE(found != tensors.end(), "Artifact lacks tensor ", name);
    return found->second;
  }

  const Entry& entry(const std::string& name) {
    const auto found = entries.find(name);
    if (found != entries.end()) return found->second;
    const auto& item = manifest.at("entries").at(name);
    Entry entry;
    entry.name = name;
    const auto strings = item.at("strings").get<std::vector<std::string>>();
    Reader buffers(
        tensor(item.at("buffers").get<std::string>()), strings, device, name);
    const auto count = buffers.next();
    for (int64_t i = 0; i < count; ++i) {
      BufferSpec spec;
      spec.storage = buffers.string();
      spec.sizes = buffers.optional_integers();
      spec.strides = buffers.optional_integers();
      if (const auto dtype = buffers.next(); dtype >= 0)
        spec.dtype = torch_dtype(strings.at(dtype));
      if (const auto index = buffers.next(); index >= 0)
        spec.device = placed(strings.at(index), device);
      entry.buffers.push_back(std::move(spec));
    }
    Reader code(
        tensor(item.at("code").get<std::string>()), strings, device, name);
    const auto calls = code.next();
    for (int64_t i = 0; i < calls; ++i) {
      Call call;
      call.method = code.string();
      const auto arguments = code.next();
      for (int64_t a = 0; a < arguments; ++a)
        call.arguments.push_back(code.value());
      entry.calls.push_back(std::move(call));
    }
    TORCH_CHECK_VALUE(code.done() && buffers.done(),
                      "Artifact entry ",
                      name,
                      " has trailing code");
    entry.int_registers = item.at("int_registers").get<int64_t>();
    entry.float_registers = item.at("float_registers").get<int64_t>();
    for (const auto& input : item.at("inputs")) {
      const auto kind = input.at(0).get<std::string>();
      const auto id = input.at(1).get<int64_t>();
      if (kind == "buffer")
        entry.input_buffers.push_back(id);
      else if (kind == "int")
        entry.input_ints.push_back(id);
      else
        entry.input_floats.push_back(id);
    }
    entry.read_in_place = item.value("read_in_place", std::vector<int64_t>{});
    entry.output_buffers =
        item.at("output_buffers").get<std::vector<int64_t>>();
    for (const auto& binding : item.at("bindings")) {
      const auto buffer = binding.at(0).get<int64_t>();
      const auto& source = binding.at(1);
      if (source.contains("material"))
        entry.materials.emplace_back(buffer,
                                     source.at("material").get<std::string>());
      else
        entry.constants.emplace_back(buffer,
                                     source.at("constant").get<int64_t>());
    }
    for (const auto& group : item.value("groups", json::array()))
      entry.groups.emplace_back(group.at("symbol").get<std::string>(),
                                group.at("size").get<int64_t>());
    entry.graphs = item.value("graphs", false);
    return entries.emplace(name, std::move(entry)).first->second;
  }

  c10::intrusive_ptr<Runner> build(
      const std::string& name,
      const std::optional<c10::intrusive_ptr<Arena>>& arena,
      const std::string& graphs,
      bool profile) {
    const auto& entry = this->entry(name);
    std::vector<std::string> missing;
    for (const auto& [buffer, symbol] : entry.materials) {
      const auto found = materials.find(symbol);
      TORCH_CHECK_VALUE(found != materials.end(),
                        "Artifact entry ",
                        name,
                        " binds material ",
                        symbol,
                        ", which the artifact does not declare");
      if (!found->second.value.defined()) missing.push_back(symbol);
    }
    if (!missing.empty()) {
      std::string list;
      for (const auto& symbol : missing)
        list += (list.empty() ? "" : ", ") + symbol;
      TORCH_CHECK_VALUE(
          false, "Artifact entry ", name, " needs unbound materials: ", list);
    }
    auto runner = c10::make_intrusive<Runner>();
    if (entry.graphs && graphs != "off") {
      runner->graphs(graphs);
      if (profile) runner->profile_segments(true);
    }
    if (arena.has_value()) runner->share_arena(*arena);
    for (const auto& spec : entry.buffers)
      runner->buffer(
          spec.storage, spec.sizes, spec.strides, spec.dtype, spec.device);
    for (int64_t i = 0; i < entry.int_registers; ++i) runner->int_register();
    for (int64_t i = 0; i < entry.float_registers; ++i)
      runner->float_register();
    std::vector<std::string> unbound;
    for (const auto& [symbol, size] : entry.groups)
      if (groups.find(symbol) == groups.end()) unbound.push_back(symbol);
    if (!unbound.empty()) {
      std::string list;
      for (const auto& symbol : unbound)
        list += (list.empty() ? "" : ", ") + symbol;
      TORCH_CHECK_VALUE(
          false, "Artifact entry ", name, " needs unbound groups: ", list);
    }
    const auto index = device.is_cpu() ? int64_t{-1} : device.index();
    std::map<int64_t, int64_t> kernel_ids;
    size_t group_count = 0;
    for (const auto& call : entry.calls) {
      const auto& a = call.arguments;
      const auto& m = call.method;
      if (m == "call_op") {
        runner->call_op(a.at(0).toStringRef(),
                        integers(a.at(1)),
                        a.at(2).toListRef().vec(),
                        integers(a.at(3)));
      } else if (m == "launch") {
        const auto kernel = a.at(0).toInt();
        auto id = kernel_ids.find(kernel);
        if (id == kernel_ids.end())
          id = kernel_ids.emplace(kernel, runner->kernel(kernels.at(kernel)))
                   .first;
        runner->launch(id->second,
                       a.at(1).toInt() >= 0 ? index : a.at(1).toInt(),
                       integers(a.at(2)),
                       integers(a.at(3)),
                       a.at(4).toListRef().vec());
      } else if (m == "call") {
        const auto callee = a.at(0).toStringRef();
        auto found = callees.find(callee);
        if (found == callees.end())
          found =
              callees.emplace(callee, build(callee, std::nullopt, "off", false))
                  .first;
        runner->call(found->second,
                     integers(a.at(1)),
                     integers(a.at(2)),
                     integers(a.at(3)),
                     integers(a.at(4)));
      } else if (m == "view") {
        runner->view(a.at(0).toInt(),
                     a.at(1).toInt(),
                     integers(a.at(2)),
                     integers(a.at(3)),
                     a.at(4).toInt());
      } else if (m == "place") {
        runner->place(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "place_output") {
        runner->place_output(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "alloc") {
        runner->alloc(a.at(0).toInt());
      } else if (m == "copy") {
        runner->copy(
            a.at(0).toInt(), a.at(1).toInt(), a.size() > 2 && a.at(2).toBool());
      } else if (m == "group") {
        const auto& [symbol, size] = entry.groups.at(group_count++);
        const auto id = runner->group(size);
        runner->bind_group(id, groups.at(symbol));
      } else if (m == "group_query") {
        runner->group_query(
            a.at(0).toStringRef(), a.at(1).toInt(), a.at(2).toInt());
      } else if (m == "collective") {
        runner->collective(a.at(0).toStringRef(),
                           a.at(1).toInt(),
                           integers(a.at(2)),
                           integers(a.at(3)),
                           integers(a.at(4)));
      } else if (m == "stream") {
        TORCH_CHECK_VALUE(!device.is_cpu(),
                          "Artifact entry ",
                          name,
                          " uses streams; load it on a GPU device");
        runner->stream(index, a.at(1).toStringRef());
      } else if (m == "event") {
        runner->event();
      } else if (m == "use_stream") {
        runner->use_stream(a.at(0).toInt());
      } else if (m == "record") {
        runner->record(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "wait") {
        runner->wait(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "host_wait") {
        runner->host_wait(a.at(0).toInt());
      } else if (m == "record_stream") {
        runner->record_stream(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "task") {
        runner->task();
      } else if (m == "begin_task") {
        runner->begin_task(a.at(0).toInt(), integers(a.at(1)));
      } else if (m == "end_task") {
        runner->end_task();
      } else if (m == "join") {
        runner->join(a.at(0).toInt());
      } else if (m == "zero") {
        runner->zero(a.at(0).toInt());
      } else if (m == "assign") {
        runner->assign(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "release") {
        runner->release(a.at(0).toInt());
      } else if (m == "int_constant") {
        runner->int_constant(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "float_constant") {
        runner->float_constant(a.at(0).toInt(), a.at(1).toDouble());
      } else if (m == "int_binary") {
        runner->int_binary(a.at(0).toStringRef(),
                           a.at(1).toInt(),
                           a.at(2).toInt(),
                           a.at(3).toInt());
      } else if (m == "select") {
        runner->select(
            a.at(0).toInt(), a.at(1).toInt(), a.at(2).toInt(), a.at(3).toInt());
      } else if (m == "assign_register") {
        runner->assign_register(a.at(0).toInt(), a.at(1).toInt());
      } else if (m == "begin_for") {
        runner->begin_for(
            a.at(0).toInt(), a.at(1).toInt(), a.at(2).toInt(), a.at(3).toInt());
      } else if (m == "end_for") {
        runner->end_for();
      } else if (m == "begin_if") {
        runner->begin_if(a.at(0).toInt());
      } else if (m == "begin_else") {
        runner->begin_else();
      } else if (m == "end_if") {
        runner->end_if();
      } else if (m == "begin_segment") {
        runner->begin_segment(a.at(0).toInt() >= 0 ? index : -1);
      } else if (m == "end_segment") {
        runner->end_segment();
      } else {
        TORCH_CHECK_VALUE(false,
                          "Artifact entry ",
                          name,
                          " calls unknown builder method ",
                          m);
      }
    }
    runner->inputs(entry.input_buffers, entry.input_ints, entry.input_floats);
    if (!entry.read_in_place.empty())
      runner->read_in_place(entry.read_in_place);
    runner->outputs(entry.output_buffers);
    for (const auto& [buffer, symbol] : entry.materials)
      runner->bind(buffer, materials.at(symbol).value);
    for (const auto& [buffer, constant] : entry.constants)
      runner->bind(buffer, constants.at(constant));
    runner->finalize();
    if (!entry.groups.empty()) {
      std::erase_if(grouped, [](const auto& weak) { return weak.expired(); });
      grouped.emplace_back(runner);
    }
    return runner;
  }

  // Digest of a stored tensor with the view a record takes of it.
  std::string stored_key(const json& record, const json& view) {
    const auto name = record.at("tensor").get<std::string>();
    auto found = digests.find(name);
    if (found == digests.end()) {
      const auto& value = tensor(name);
      std::ostringstream text;
      text << value.scalar_type() << value.sizes() << " "
           << io::detail::sha256_hex(std::string_view(
                  static_cast<const char*>(value.data_ptr()), value.nbytes()));
      found = digests.emplace(name, text.str()).first;
    }
    return found->second + " " + record.value("device", "") + " " + view.dump();
  }

  // Key of each output of built-material entry `name`: a digest of the
  // builder calls and views that produce it from constants, stored materials
  // and constant arguments, on the load device. Two outputs with one key hold
  // equal Tensors. Entries with inputs, other bound materials, register
  // arguments, mutating operators or other instructions have no keys.
  std::optional<std::vector<std::string>> table_keys(const std::string& name) {
    const auto& entry = this->entry(name);
    if (!entry.input_buffers.empty() || !entry.input_ints.empty() ||
        !entry.input_floats.empty())
      return std::nullopt;
    std::map<int64_t, std::string> keys;
    for (const auto& [buffer, symbol] : entry.materials) {
      const auto& record = materials.at(symbol).record;
      if (record.value("binding", "") != "embedded") return std::nullopt;
      keys[buffer] =
          "material " + stored_key(record, record.value("view", json()));
    }
    const auto& constant_records = manifest.at("constants");
    for (const auto& [buffer, index] : entry.constants) {
      const auto& record = constant_records.at(index);
      keys[buffer] =
          "constant " + stored_key(record,
                                   json::array({record.at("sizes"),
                                                record.at("strides"),
                                                record.at("offset")}));
    }
    const auto key_of = [&](int64_t buffer) -> const std::string* {
      const auto found = keys.find(buffer);
      return found == keys.end() ? nullptr : &found->second;
    };
    for (const auto& call : entry.calls) {
      const auto& a = call.arguments;
      if (call.method == "release") continue;
      std::ostringstream text;
      if (call.method == "view") {
        const auto* source = key_of(a.at(1).toInt());
        if (source == nullptr) return std::nullopt;
        text << *source << " view ";
        describe(text, a.at(2));
        describe(text, a.at(3));
        describe(text, a.at(4));
        keys[a.at(0).toInt()] = io::detail::sha256_hex(text.str());
        continue;
      }
      if (call.method != "call_op") return std::nullopt;
      const auto& schema = a.at(0).toStringRef();
      const auto dot = schema.rfind('.');
      auto overload = schema.substr(dot + 1);
      if (overload == "default") overload.clear();
      const auto handle = c10::Dispatcher::singleton().findSchema(
          {schema.substr(0, dot), overload});
      if (!handle.has_value() || handle->schema().is_mutable())
        return std::nullopt;
      text << device << " " << schema << "(";
      const auto kinds = integers(a.at(1));
      const auto values = a.at(2).toListRef();
      for (size_t i = 0; i < kinds.size(); ++i) {
        switch (static_cast<exec::ArgumentKind>(kinds[i])) {
          case exec::ArgumentKind::constant:
            describe(text, values[i]);
            break;
          case exec::ArgumentKind::empty_list:
            text << "[]";
            break;
          case exec::ArgumentKind::buffer: {
            const auto* key = key_of(values[i].toInt());
            if (key == nullptr) return std::nullopt;
            text << "{" << *key << "}";
            break;
          }
          case exec::ArgumentKind::buffer_list:
            text << "[";
            for (const auto id : integers(values[i])) {
              const auto* key = id < 0 ? nullptr : key_of(id);
              if (id >= 0 && key == nullptr) return std::nullopt;
              text << (key == nullptr ? "None" : "{" + *key + "}") << ",";
            }
            text << "]";
            break;
          default:
            return std::nullopt;
        }
        text << ",";
      }
      text << ")";
      const auto digest = io::detail::sha256_hex(text.str());
      const auto results = integers(a.at(3));
      for (size_t i = 0; i < results.size(); ++i)
        keys[results[i]] = digest + "#" + std::to_string(i);
    }
    std::vector<std::string> result;
    for (const auto buffer : entry.output_buffers) {
      const auto* key = key_of(buffer);
      if (key == nullptr) return std::nullopt;
      result.push_back(*key);
    }
    return result;
  }

  void set(const std::string& symbol, at::Tensor value) {
    auto found = materials.find(symbol);
    TORCH_CHECK_VALUE(
        found != materials.end(), "Artifact declares no material ", symbol);
    found->second.value = std::move(value);
  }
};

Artifact::Artifact(const std::string& path, c10::Device device)
    : state_(std::make_unique<State>()) {
  auto& s = *state_;
  if (device.is_cuda() && !device.has_index()) device.set_index(0);
  s.device = device;
  auto container = io::load_container(path, c10::Device(c10::kCPU));
  const auto& metadata = container.metadata;
  const auto format = metadata.find("fhelium.format");
  TORCH_CHECK_VALUE(format != metadata.end() && format->second == kFormat,
                    path,
                    " is not an FHElium artifact");
  const auto version = metadata.find("fhelium.schema_version");
  TORCH_CHECK_VALUE(
      version != metadata.end() && version->second == kSchemaVersion,
      "Unsupported artifact schema version in ",
      path);
  s.manifest = json::parse(metadata.at("fhelium.manifest"));
  const auto runtime = json::parse(io::native_identity());
  const auto& identity = s.manifest.at("identity");
  for (const auto& [field, value] : runtime.items()) {
    const auto recorded = identity.value(field, json());
    TORCH_CHECK(recorded == value,
                "Artifact identity mismatch: ",
                field,
                " is ",
                recorded.dump(),
                " in ",
                path,
                " and ",
                value.dump(),
                " in the loaded library");
  }
  for (auto& [name, tensor] : container.tensors)
    s.tensors.emplace(name, std::move(tensor));

  const auto architecture =
      device.is_cpu() ? std::string()
                      : exec::detail::device_architecture(device.index());
  const auto kernels = s.manifest.value("kernels", json::array());
  for (const auto& item : kernels) {
    KernelRecord record;
    record.name = item.at("name").get<std::string>();
    const auto& images = item.at("images");
    std::string image;
    if (images.contains(architecture)) {
      image = images.at(architecture).get<std::string>();
      record.image_kind = architecture.rfind("sm_", 0) == 0 ? "cubin" : "hsaco";
    } else {
      TORCH_CHECK_VALUE(images.contains("ptx"),
                        "Kernel ",
                        record.name,
                        " has no image for ",
                        architecture,
                        " and no PTX");
      image = images.at("ptx").get<std::string>();
      record.image_kind = "ptx";
    }
    const auto& bytes = s.tensor(image);
    const auto* begin = bytes.const_data_ptr<uint8_t>();
    record.image.assign(begin, begin + bytes.numel());
    record.num_warps = item.at("num_warps").get<int64_t>();
    record.warp_size = item.at("warp_size").get<int64_t>();
    record.num_ctas = item.at("num_ctas").get<int64_t>();
    record.shared = item.at("shared").get<int64_t>();
    record.parameter_types =
        item.at("parameter_types").get<std::vector<std::string>>();
    const auto scratch = item.at("scratch").get<std::vector<int64_t>>();
    record.global_scratch_size = scratch.at(0);
    record.global_scratch_align = scratch.at(1);
    record.profile_scratch_size = scratch.at(2);
    record.profile_scratch_align = scratch.at(3);
    record.triton_version = item.value("triton_version", "");
    s.kernels.push_back(std::move(record));
  }
  // Records that name one stored tensor share its placed copy.
  std::map<std::pair<std::string, std::string>, at::Tensor> placed;
  const auto base = [&](const json& record) -> const at::Tensor& {
    const auto name = record.at("tensor").get<std::string>();
    const auto target = record_device(record, device);
    auto& value = placed[{name, target.str()}];
    if (!value.defined()) value = own(s.tensor(name), target);
    return value;
  };
  const auto constants = s.manifest.value("constants", json::array());
  for (const auto& item : constants)
    s.constants.push_back(
        base(item).as_strided(item.at("sizes").get<std::vector<int64_t>>(),
                              item.at("strides").get<std::vector<int64_t>>(),
                              item.at("offset").get<int64_t>()));
  const auto materials = s.manifest.value("materials", json::object());
  for (const auto& [kind, records] : materials.items())
    for (const auto& record : records) {
      Material material{record, at::Tensor()};
      if (record.value("binding", "") == "embedded") {
        material.value = base(record);
        if (record.contains("view")) {
          const auto& view = record.at("view");
          material.value = material.value.as_strided(
              view.at("sizes").get<std::vector<int64_t>>(),
              view.at("strides").get<std::vector<int64_t>>(),
              view.at("offset").get<int64_t>());
        }
      }
      s.materials.emplace(record.at("symbol").get<std::string>(),
                          std::move(material));
    }
}

Artifact::~Artifact() = default;

std::string Artifact::manifest() const { return state_->manifest.dump(); }

std::vector<std::string> Artifact::entries() const {
  std::vector<std::string> names;
  for (const auto& [name, item] : state_->manifest.at("entries").items())
    names.push_back(name);
  return names;
}

std::vector<ArtifactSignature> Artifact::signatures() const {
  std::vector<ArtifactSignature> result;
  for (const auto& item : state_->manifest.value("signatures", json::array())) {
    ArtifactSignature signature{item.at("name").get<std::string>(),
                                item.at("entry").get<std::string>(),
                                {}};
    for (const auto& value : item.value("results", json::array()))
      signature.results.push_back(
          io::Value{value.at("value_type").get<std::string>(),
                    value.at("value_metadata").dump(),
                    {}});
    result.push_back(std::move(signature));
  }
  return result;
}

std::vector<std::string> Artifact::bind_key(const io::Value& key) {
  const auto metadata = json::parse(key.metadata);
  TORCH_CHECK_VALUE(key.tensors.count("data") != 0,
                    "Key value ",
                    key.type,
                    " has no data tensor");
  const auto& data = key.tensors.at("data");
  at::Tensor expanded;
  std::vector<std::string> bound;
  for (const auto& [symbol, material] : state_->materials) {
    const auto& record = material.record;
    if (record.value("binding", "") != "caller" ||
        record.value("role", "") != key.type || !record.contains("layout"))
      continue;
    bool same = true;
    for (const auto& [field, value] : record.at("layout").items())
      if (field != "kind" && field != "seed_rounds" &&
          (!metadata.contains(field) || metadata.at(field) != value))
        same = false;
    if (!same) continue;
    auto value = data;
    const auto shape = record.at("shape").get<std::vector<int64_t>>();
    if (value.sizes() != c10::IntArrayRef(shape) && metadata.contains("seed") &&
        key.tensors.count("seed") != 0) {
      if (!expanded.defined()) {
        static const auto expand =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("fhelium_ckks_ops::expand_key_switch_key",
                                   "")
                .typed<at::Tensor(const at::Tensor&,
                                  const at::Tensor&,
                                  c10::IntArrayRef,
                                  std::string_view,
                                  int64_t)>();
        const auto& seed = metadata.at("seed");
        expanded = expand.call(data,
                               key.tensors.at("seed"),
                               seed.at("moduli").get<std::vector<int64_t>>(),
                               seed.at("prf").get<std::string>(),
                               seed.at("rounds").get<int64_t>());
      }
      value = expanded;
    }
    bind(symbol, value);
    bound.push_back(symbol);
  }
  return bound;
}

void Artifact::bind(const std::string& symbol, const at::Tensor& value) {
  auto& s = *state_;
  const auto found = s.materials.find(symbol);
  TORCH_CHECK_VALUE(
      found != s.materials.end(), "Artifact declares no material ", symbol);
  const auto& record = found->second.record;
  if (record.contains("shape"))
    TORCH_CHECK_VALUE(
        value.sizes() ==
            c10::IntArrayRef(record.at("shape").get<std::vector<int64_t>>()),
        "Material ",
        symbol,
        " has shape ",
        record.at("shape").dump(),
        ", not ",
        value.sizes());
  if (record.contains("dtype"))
    TORCH_CHECK_VALUE(value.scalar_type() ==
                          torch_dtype(record.at("dtype").get<std::string>()),
                      "Material ",
                      symbol,
                      " has dtype ",
                      record.at("dtype").get<std::string>(),
                      ", not ",
                      value.scalar_type());
  const auto device = record_device(record, s.device);
  TORCH_CHECK_VALUE(value.device() == device,
                    "Material ",
                    symbol,
                    " must be on ",
                    device,
                    ", not ",
                    value.device());
  found->second.value = value;
}

at::Tensor Artifact::material(const std::string& symbol) const {
  const auto found = state_->materials.find(symbol);
  TORCH_CHECK_VALUE(found != state_->materials.end(),
                    "Artifact declares no material ",
                    symbol);
  return found->second.value;
}

std::vector<std::string> Artifact::unbound() const {
  std::vector<std::string> names;
  for (const auto& [symbol, material] : state_->materials)
    if (!material.value.defined()) names.push_back(symbol);
  return names;
}

namespace {

// The seed words of the random stream of `device` derived from `seed`:
// SHA-256 of "fhelium.random\0", the seed as 32 little-endian signed bytes
// and the stream id (0 for the CPU, 1 + index for a GPU) as 8 little-endian
// bytes gives four little-endian key words; the nonce word is 0.
at::Tensor derived_seed(int64_t seed, const c10::Device& device) {
  std::string message("fhelium.random", 15);
  for (int i = 0; i < 32; ++i)
    message.push_back(static_cast<char>(
        i < 8 ? (static_cast<uint64_t>(seed) >> (8 * i)) & 0xff
              : (seed < 0 ? 0xff : 0)));
  const uint64_t stream = device.is_cpu() ? 0 : 1 + device.index();
  for (int i = 0; i < 8; ++i)
    message.push_back(static_cast<char>((stream >> (8 * i)) & 0xff));
  const auto digest = io::detail::sha256_hex(message);
  auto words = at::zeros({5}, at::kLong);
  auto* data = words.mutable_data_ptr<int64_t>();
  for (int word = 0; word < 4; ++word) {
    uint64_t value = 0;
    for (int byte = 7; byte >= 0; --byte)
      value = value << 8 |
              std::stoul(digest.substr(16 * word + 2 * byte, 2), nullptr, 16);
    data[word] = static_cast<int64_t>(value);
  }
  return words;
}

}  // namespace

void Artifact::seed_rounding(std::optional<int64_t> seed) {
  auto& s = *state_;
  // One stream per device, as one Engine has one stream per device.
  std::map<std::string, at::Tensor> streams;
  for (auto& [symbol, material] : s.materials) {
    if (material.record.value("kind", "") != "rounding_state") continue;
    const auto device = record_device(material.record, s.device);
    auto& stream = streams[device.str()];
    if (!stream.defined()) {
      static const auto generate =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("fhelium_random_ops::generate_seed", "")
              .typed<at::Tensor()>();
      stream =
          (seed ? derived_seed(*seed, device) : generate.call()).to(device);
    }
    bind(symbol, stream);
  }
}

void Artifact::prepare() {
  auto& s = *state_;
  if (s.prepared) return;
  for (const auto& [symbol, material] : s.materials)
    if (material.record.value("kind", "") == "rounding_state" &&
        !material.value.defined()) {
      seed_rounding(std::nullopt);
      break;
    }
  // Built materials: each entry runs once.
  std::map<std::string, std::vector<std::pair<int64_t, std::string>>> built;
  for (const auto& [symbol, material] : s.materials)
    if (material.record.value("binding", "") == "built")
      built[material.record.at("entry").get<std::string>()].emplace_back(
          material.record.at("output").get<int64_t>(), symbol);
  // Tables already built by an artifact of the process are bound from the
  // table cache; an entry runs when any of its outputs is missing.
  for (const auto& [entry, outputs] : built) {
    const auto keys = s.table_keys(entry);
    std::vector<std::pair<int64_t, std::string>> missing;
    for (const auto& [output, symbol] : outputs) {
      auto cached = keys ? table_cache().find(keys->at(output)) : at::Tensor();
      if (cached.defined())
        s.set(symbol, std::move(cached));
      else
        missing.emplace_back(output, symbol);
    }
    if (missing.empty()) continue;
    const auto results = s.preparer(entry)->run({}, {}, {});
    for (const auto& [output, symbol] : missing) {
      s.set(symbol, results.at(output));
      if (keys) table_cache().insert(keys->at(output), results.at(output));
    }
  }
  const auto prepare = s.manifest.value("prepare", json::object());
  for (const auto& [name, item] : prepare.items()) {
    if (item.value("per_call", false)) continue;
    std::vector<at::Tensor> inputs;
    for (const auto& symbol : item.at("inputs"))
      inputs.push_back(material(symbol.get<std::string>()));
    const auto results =
        s.preparer(item.at("entry").get<std::string>())->run(inputs, {}, {});
    const auto outputs = item.at("outputs").get<std::vector<std::string>>();
    for (size_t i = 0; i < outputs.size(); ++i)
      s.set(outputs[i], results.at(i));
  }
  s.prepared = true;
}

void Artifact::refresh(const std::string& name,
                       const std::vector<at::Tensor>& inputs) {
  auto& s = *state_;
  const auto& item = s.manifest.at("prepare").at(name);
  const auto results =
      s.preparer(item.at("entry").get<std::string>())->run(inputs, {}, {});
  const auto outputs = item.at("outputs").get<std::vector<std::string>>();
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& target = s.materials.at(outputs[i]).value;
    if (target.defined())
      target.copy_(results.at(i));
    else
      target = results.at(i);
  }
}

void Artifact::bind_group(const std::string& symbol, const std::string& name) {
  bind_group(symbol, c10d::resolve_process_group(name));
}

void Artifact::bind_group(const std::string& symbol,
                          c10::intrusive_ptr<c10d::ProcessGroup> group) {
  TORCH_CHECK_VALUE(group, "Group ", symbol, " is bound to no process group");
  state_->groups[symbol] = std::move(group);
  // Called entries are built again with the new group.
  state_->callees.clear();
}

void Artifact::release_groups() {
  for (auto& weak : state_->grouped)
    if (auto runner = weak.lock()) runner->release_groups();
  state_->grouped.clear();
  state_->groups.clear();
  state_->callees.clear();
}

c10::intrusive_ptr<exec::Runner> Artifact::runner(
    const std::string& entry,
    const std::optional<c10::intrusive_ptr<exec::Arena>>& arena,
    const std::string& graphs,
    bool profile_segments) {
  return state_->build(entry, arena, graphs, profile_segments);
}

}  // namespace fhelium
