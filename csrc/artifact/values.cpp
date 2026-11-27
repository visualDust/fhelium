#include "fhelium/values.h"

#include <ATen/ops/complex.h>
#include <ATen/ops/view_as_real.h>
#include <c10/util/Exception.h>

#include <array>
#include <nlohmann/json.hpp>
#include <utility>

#include "fhelium/container.h"

namespace fhelium::io {
namespace {

constexpr const char* kFormat = "fhelium-value";
constexpr int64_t kFileSchemaVersion = 2;
constexpr int64_t kValueSchemaVersion = 5;

constexpr std::array<std::pair<c10::ScalarType, const char*>, 15> kTorchNames =
    {{
        {c10::ScalarType::Bool, "torch.bool"},
        {c10::ScalarType::Byte, "torch.uint8"},
        {c10::ScalarType::Char, "torch.int8"},
        {c10::ScalarType::Short, "torch.int16"},
        {c10::ScalarType::UInt16, "torch.uint16"},
        {c10::ScalarType::Int, "torch.int32"},
        {c10::ScalarType::UInt32, "torch.uint32"},
        {c10::ScalarType::Long, "torch.int64"},
        {c10::ScalarType::UInt64, "torch.uint64"},
        {c10::ScalarType::Half, "torch.float16"},
        {c10::ScalarType::BFloat16, "torch.bfloat16"},
        {c10::ScalarType::Float, "torch.float32"},
        {c10::ScalarType::Double, "torch.float64"},
        {c10::ScalarType::ComplexFloat, "torch.complex64"},
        {c10::ScalarType::ComplexDouble, "torch.complex128"},
    }};

const char* torch_name(c10::ScalarType type) {
  for (const auto& [item, name] : kTorchNames)
    if (item == type) return name;
  TORCH_CHECK_VALUE(false, "Value files do not store dtype ", type);
}

c10::ScalarType torch_type(const std::string& name) {
  for (const auto& [item, text] : kTorchNames)
    if (name == text) return item;
  TORCH_CHECK_VALUE(false, "Unsupported value tensor dtype ", name);
}

nlohmann::json field(const Value& value, const std::string& name) {
  const auto metadata = nlohmann::json::parse(value.metadata);
  TORCH_CHECK_VALUE(
      metadata.contains(name), value.type, " has no metadata field ", name);
  return metadata.at(name);
}

}  // namespace

bool Value::has(const std::string& name) const {
  return nlohmann::json::parse(metadata).contains(name);
}

int64_t Value::integer(const std::string& name) const {
  return field(*this, name).get<int64_t>();
}

double Value::number(const std::string& name) const {
  return field(*this, name).get<double>();
}

std::string Value::string(const std::string& name) const {
  return field(*this, name).get<std::string>();
}

std::vector<int64_t> Value::integers(const std::string& name) const {
  return field(*this, name).get<std::vector<int64_t>>();
}

std::vector<std::vector<int64_t>> Value::integer_rows(
    const std::string& name) const {
  return field(*this, name).get<std::vector<std::vector<int64_t>>>();
}

Value load_value(const std::string& path, const c10::Device& device) {
  auto container = load_container(path, device);
  const auto& metadata = container.metadata;
  const auto format = metadata.find("fhelium.format");
  TORCH_CHECK_VALUE(format != metadata.end() && format->second == kFormat,
                    path,
                    " is not an FHElium value file");
  const auto version = metadata.find("fhelium.schema_version");
  TORCH_CHECK_VALUE(version != metadata.end() &&
                        version->second == std::to_string(kFileSchemaVersion),
                    "Unsupported value file schema version in ",
                    path);
  const auto manifest =
      nlohmann::json::parse(metadata.at("fhelium.manifest"), nullptr, false);
  TORCH_CHECK_VALUE(manifest.is_object() &&
                        manifest.value("value_schema_version", int64_t{-1}) ==
                            kValueSchemaVersion,
                    "Unsupported value envelope in ",
                    path);
  std::map<std::string, at::Tensor> payloads(container.tensors.begin(),
                                             container.tensors.end());
  Value value;
  value.type = manifest.at("value_type").get<std::string>();
  value.metadata = manifest.at("value_metadata").dump();
  int64_t nbytes = 0;
  size_t used = 0;
  for (const auto& [name, item] : manifest.at("tensor_metadata").items()) {
    const auto names = item.at("payloads").get<std::vector<std::string>>();
    std::vector<at::Tensor> parts;
    for (const auto& payload : names) {
      const auto found = payloads.find(payload);
      TORCH_CHECK_VALUE(
          found != payloads.end(), path, " lacks tensor payload ", payload);
      parts.push_back(found->second);
    }
    used += parts.size();
    const auto encoding = item.at("encoding").get<std::string>();
    at::Tensor tensor;
    if (encoding == "direct" && parts.size() == 1) {
      tensor = parts[0];
    } else {
      TORCH_CHECK_VALUE(encoding == "complex_split" && parts.size() == 2,
                        path,
                        " uses tensor encoding ",
                        encoding);
      tensor = at::complex(parts[0], parts[1]);
    }
    TORCH_CHECK_VALUE(
        tensor.sizes() == c10::IntArrayRef(
                              item.at("shape").get<std::vector<int64_t>>()) &&
            tensor.scalar_type() ==
                torch_type(item.at("dtype").get<std::string>()),
        path,
        " tensor ",
        name,
        " does not match its manifest");
    nbytes += static_cast<int64_t>(tensor.nbytes());
    value.tensors.emplace(name, std::move(tensor));
  }
  TORCH_CHECK_VALUE(used == payloads.size(),
                    path,
                    " holds payloads that its manifest does not list");
  TORCH_CHECK_VALUE(nbytes == manifest.at("nbytes").get<int64_t>(),
                    path,
                    " byte count does not match its manifest");
  return value;
}

void save_value(const std::string& path,
                const Value& value,
                bool allow_secret,
                bool overwrite) {
  TORCH_CHECK(value.type != "SecretKey" || allow_secret,
              "Saving a SecretKey requires allow_secret");
  TORCH_CHECK_VALUE(!value.tensors.empty(),
                    "A value file holds at least one tensor");
  nlohmann::json tensor_metadata = nlohmann::json::object();
  std::vector<std::pair<std::string, at::Tensor>> payloads;
  int64_t nbytes = 0;
  for (const auto& [name, tensor] : value.tensors) {
    auto snapshot = tensor.detach().cpu().contiguous();
    nbytes += static_cast<int64_t>(snapshot.nbytes());
    nlohmann::json item;
    item["shape"] = snapshot.sizes().vec();
    item["dtype"] = torch_name(snapshot.scalar_type());
    if (snapshot.is_complex()) {
      const auto parts = at::view_as_real(snapshot);
      item["encoding"] = "complex_split";
      item["payloads"] = {name + ".real", name + ".imag"};
      payloads.emplace_back(name + ".real", parts.select(-1, 0).contiguous());
      payloads.emplace_back(name + ".imag", parts.select(-1, 1).contiguous());
    } else {
      item["encoding"] = "direct";
      item["payloads"] = {name};
      payloads.emplace_back(name, snapshot);
    }
    tensor_metadata[name] = std::move(item);
  }
  nlohmann::json manifest;
  manifest["file_schema_version"] = kFileSchemaVersion;
  manifest["value_schema_version"] = kValueSchemaVersion;
  manifest["value_type"] = value.type;
  manifest["nbytes"] = nbytes;
  manifest["tensor_metadata"] = std::move(tensor_metadata);
  manifest["value_metadata"] = nlohmann::json::parse(value.metadata);
  // Sorted keys, compact separators and ASCII escapes, as Python's
  // json.dumps(sort_keys=True, separators=(",", ":")) writes them.
  save_container(
      path,
      payloads,
      {{"fhelium.format", kFormat},
       {"fhelium.schema_version", std::to_string(kFileSchemaVersion)},
       {"fhelium.manifest", manifest.dump(-1, ' ', true)}},
      overwrite);
}

}  // namespace fhelium::io
