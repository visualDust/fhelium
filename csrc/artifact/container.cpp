#include "fhelium/container.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>
#include <torch/headeronly/version.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>

#include "../runner/kernel_loader.h"
#include "container_detail.h"

namespace fhelium::io {
namespace {

namespace fs = std::filesystem;

// Safetensors dtype names; `rank` orders tensors in the file, larger first,
// which keeps every tensor aligned to its element size. C128 is an FHElium
// extension; the other names are those of the safetensors format.
struct DtypeName {
  c10::ScalarType type;
  const char* name;
  int rank;
};

constexpr std::array<DtypeName, 17> kDtypes = {{
    {c10::ScalarType::Bool, "BOOL", 0},
    {c10::ScalarType::Byte, "U8", 1},
    {c10::ScalarType::Char, "I8", 2},
    {c10::ScalarType::Float8_e5m2, "F8_E5M2", 3},
    {c10::ScalarType::Float8_e4m3fn, "F8_E4M3", 4},
    {c10::ScalarType::Short, "I16", 5},
    {c10::ScalarType::UInt16, "U16", 6},
    {c10::ScalarType::Half, "F16", 7},
    {c10::ScalarType::BFloat16, "BF16", 8},
    {c10::ScalarType::Int, "I32", 9},
    {c10::ScalarType::UInt32, "U32", 10},
    {c10::ScalarType::Float, "F32", 11},
    {c10::ScalarType::ComplexFloat, "C64", 12},
    {c10::ScalarType::Double, "F64", 13},
    {c10::ScalarType::Long, "I64", 14},
    {c10::ScalarType::UInt64, "U64", 15},
    {c10::ScalarType::ComplexDouble, "C128", 16},
}};

const DtypeName& dtype_of(c10::ScalarType type) {
  for (const auto& item : kDtypes)
    if (item.type == type) return item;
  TORCH_CHECK_VALUE(false, "Containers do not store dtype ", type);
}

c10::ScalarType scalar_type_of(const std::string& name) {
  for (const auto& item : kDtypes)
    if (name == item.name) return item.type;
  TORCH_CHECK_VALUE(false, "Unsupported container dtype ", name);
}

// JSON string escaping of the safetensors reference writer (serde_json).
void append_string(std::string& out, const std::string& text) {
  static constexpr char kHex[] = "0123456789abcdef";
  out.push_back('"');
  for (const unsigned char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out.push_back(kHex[c >> 4]);
          out.push_back(kHex[c & 15]);
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

std::string temporary_name(const fs::path& destination) {
  std::random_device device;
  std::uniform_int_distribution<uint64_t> distribution;
  std::ostringstream name;
  name << '.' << destination.filename().string() << '.' << std::hex
       << distribution(device) << ".tmp";
  return (destination.parent_path() / name.str()).string();
}

struct Header {
  std::map<std::string, std::string> metadata;
  nlohmann::json tensors = nlohmann::json::object();
  uint64_t data_offset = 0;  // file offset of the data section
};

Header read_header(std::ifstream& file, const std::string& path) {
  uint8_t length_bytes[8];
  file.read(reinterpret_cast<char*>(length_bytes), 8);
  TORCH_CHECK_VALUE(file.good(), "Container ", path, " has no header");
  uint64_t length = 0;
  for (int i = 7; i >= 0; --i) length = (length << 8) | length_bytes[i];
  file.seekg(0, std::ios::end);
  const auto size = static_cast<uint64_t>(file.tellg());
  TORCH_CHECK_VALUE(length <= size - 8,
                    "Container ",
                    path,
                    " declares a header longer than the file");
  file.seekg(8);
  std::string text(length, '\0');
  file.read(text.data(), static_cast<std::streamsize>(length));
  TORCH_CHECK_VALUE(file.good(), "Container ", path, " header is truncated");
  Header header;
  auto parsed = nlohmann::json::parse(text, nullptr, false);
  TORCH_CHECK_VALUE(
      parsed.is_object(), "Container ", path, " header is not a JSON object");
  header.data_offset = 8 + length;
  const auto data_size = size - header.data_offset;
  for (auto& [name, item] : parsed.items()) {
    if (name == "__metadata__") {
      TORCH_CHECK_VALUE(item.is_object(), "Container metadata is not a map");
      for (auto& [key, value] : item.items()) {
        TORCH_CHECK_VALUE(value.is_string(),
                          "Container metadata value ",
                          key,
                          " is not a string");
        header.metadata.emplace(key, value.get<std::string>());
      }
      continue;
    }
    const auto& offsets = item.at("data_offsets");
    const auto begin = offsets.at(0).get<uint64_t>();
    const auto end = offsets.at(1).get<uint64_t>();
    TORCH_CHECK_VALUE(begin <= end && end <= data_size,
                      "Container tensor ",
                      name,
                      " lies outside the data section");
    uint64_t elements = 1;
    for (const auto& extent : item.at("shape"))
      elements *= extent.get<uint64_t>();
    const auto type = scalar_type_of(item.at("dtype").get<std::string>());
    TORCH_CHECK_VALUE(elements * c10::elementSize(type) == end - begin,
                      "Container tensor ",
                      name,
                      " has ",
                      end - begin,
                      " bytes for its shape and dtype");
    header.tensors[name] = std::move(item);
  }
  return header;
}

// SHA-256 (FIPS 180-4).
class Sha256 {
 public:
  void update(std::string_view text) {
    for (const unsigned char c : text) {
      block_[used_++] = c;
      if (used_ == 64) compress();
    }
    length_ += text.size();
  }

  std::string hex() {
    const uint64_t bits = length_ * 8;
    block_[used_++] = 0x80;
    if (used_ > 56) {
      while (used_ < 64) block_[used_++] = 0;
      compress();
    }
    while (used_ < 56) block_[used_++] = 0;
    for (int i = 7; i >= 0; --i)
      block_[used_++] = static_cast<uint8_t>(bits >> (8 * i));
    compress();
    std::ostringstream out;
    out << std::hex;
    for (const auto word : state_)
      for (int i = 3; i >= 0; --i) {
        const auto byte = (word >> (8 * i)) & 0xff;
        out << (byte >> 4) << (byte & 15);
      }
    return out.str();
  }

 private:
  static uint32_t rotate(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void compress() {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(block_[4 * i]) << 24) |
             (uint32_t(block_[4 * i + 1]) << 16) |
             (uint32_t(block_[4 * i + 2]) << 8) | uint32_t(block_[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const auto s0 =
          rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const auto s1 =
          rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const auto t1 = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) +
                      ((e & f) ^ (~e & g)) + k[i] + w[i];
      const auto t2 = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) +
                      ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
    used_ = 0;
  }

  std::array<uint32_t, 8> state_ = {0x6a09e667,
                                    0xbb67ae85,
                                    0x3c6ef372,
                                    0xa54ff53a,
                                    0x510e527f,
                                    0x9b05688c,
                                    0x1f83d9ab,
                                    0x5be0cd19};
  std::array<uint8_t, 64> block_{};
  size_t used_ = 0;
  uint64_t length_ = 0;
};

}  // namespace

namespace detail {

c10::ScalarType container_dtype(const std::string& name) {
  return scalar_type_of(name);
}

const char* container_dtype_name(c10::ScalarType type) {
  return dtype_of(type).name;
}

std::string sha256_hex(std::string_view bytes) {
  Sha256 digest;
  digest.update(bytes);
  return digest.hex();
}

}  // namespace detail

void save_container(
    const std::string& path,
    const std::vector<std::pair<std::string, at::Tensor>>& tensors,
    const std::map<std::string, std::string>& metadata,
    bool overwrite) {
  const fs::path destination(path);
  TORCH_CHECK_VALUE(!fs::exists(destination) || overwrite,
                    "Container file ",
                    path,
                    " exists");
  std::vector<std::pair<std::string, at::Tensor>> ordered;
  for (const auto& [name, tensor] : tensors) {
    TORCH_CHECK_VALUE(name != "__metadata__",
                      "__metadata__ is not a tensor name");
    TORCH_CHECK_VALUE(tensor.layout() == c10::kStrided,
                      "Container tensor ",
                      name,
                      " is not strided");
    dtype_of(tensor.scalar_type());
    ordered.emplace_back(name, tensor.detach().cpu().contiguous());
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
    const auto ra = dtype_of(a.second.scalar_type()).rank;
    const auto rb = dtype_of(b.second.scalar_type()).rank;
    return ra != rb ? ra > rb : a.first < b.first;
  });
  for (size_t i = 1; i < ordered.size(); ++i)
    TORCH_CHECK_VALUE(ordered[i].first != ordered[i - 1].first,
                      "Container tensor name ",
                      ordered[i].first,
                      " repeats");
  std::string header = "{";
  if (!metadata.empty()) {
    header += "\"__metadata__\":{";
    bool first = true;
    for (const auto& [key, value] : metadata) {
      if (!first) header.push_back(',');
      first = false;
      append_string(header, key);
      header.push_back(':');
      append_string(header, value);
    }
    header += "}";
  }
  uint64_t offset = 0;
  bool first = metadata.empty();
  for (const auto& [name, tensor] : ordered) {
    if (!first) header.push_back(',');
    first = false;
    append_string(header, name);
    header += ":{\"dtype\":";
    append_string(header, dtype_of(tensor.scalar_type()).name);
    header += ",\"shape\":[";
    for (int64_t d = 0; d < tensor.dim(); ++d) {
      if (d) header.push_back(',');
      header += std::to_string(tensor.size(d));
    }
    const auto bytes = static_cast<uint64_t>(tensor.nbytes());
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header.push_back('}');
  header.append((8 - header.size() % 8) % 8, ' ');

  const auto temporary = temporary_name(destination);
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    TORCH_CHECK(file.good(), "Cannot create ", temporary);
    uint8_t length[8];
    for (int i = 0; i < 8; ++i)
      length[i] = static_cast<uint8_t>(uint64_t(header.size()) >> (8 * i));
    file.write(reinterpret_cast<const char*>(length), 8);
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& [name, tensor] : ordered)
      file.write(static_cast<const char*>(tensor.const_data_ptr()),
                 static_cast<std::streamsize>(tensor.nbytes()));
    file.close();
    if (!file.good()) {
      fs::remove(temporary);
      TORCH_CHECK(false, "Writing ", temporary, " failed");
    }
  }
  std::error_code error;
  fs::permissions(temporary,
                  fs::perms::owner_read | fs::perms::owner_write,
                  fs::perm_options::replace,
                  error);
  if (!overwrite && fs::exists(destination)) {
    fs::remove(temporary);
    TORCH_CHECK_VALUE(false, "Container file ", path, " exists");
  }
  fs::rename(temporary, destination, error);
  if (error) {
    fs::remove(temporary);
    TORCH_CHECK(false, "Renaming ", temporary, " failed: ", error.message());
  }
}

Container load_container(const std::string& path, const c10::Device& device) {
  std::ifstream file(path, std::ios::binary);
  TORCH_CHECK_VALUE(file.good(), "No container file at ", path);
  auto header = read_header(file, path);
  file.seekg(0, std::ios::end);
  const auto size = static_cast<int64_t>(file.tellg()) -
                    static_cast<int64_t>(header.data_offset);
  auto data = at::empty({size}, at::TensorOptions().dtype(at::kByte));
  file.seekg(static_cast<std::streamoff>(header.data_offset));
  file.read(static_cast<char*>(data.data_ptr()), size);
  TORCH_CHECK_VALUE(
      file.good() || size == 0, "Container ", path, " data is truncated");
  if (!device.is_cpu()) data = data.to(device);
  Container container;
  container.metadata = std::move(header.metadata);
  std::vector<std::pair<uint64_t, std::string>> order;
  for (const auto& [name, item] : header.tensors.items())
    order.emplace_back(item.at("data_offsets").at(0).get<uint64_t>(), name);
  std::sort(order.begin(), order.end());
  for (const auto& [begin, name] : order) {
    const auto& item = header.tensors.at(name);
    const auto type = scalar_type_of(item.at("dtype").get<std::string>());
    std::vector<int64_t> shape;
    for (const auto& extent : item.at("shape"))
      shape.push_back(extent.get<int64_t>());
    const auto end = item.at("data_offsets").at(1).get<int64_t>();
    const auto element = static_cast<int64_t>(c10::elementSize(type));
    TORCH_CHECK_VALUE(begin % element == 0,
                      "Container tensor ",
                      name,
                      " is not aligned to its element size");
    auto bytes = data.narrow(
        0, static_cast<int64_t>(begin), end - static_cast<int64_t>(begin));
    container.tensors.emplace_back(name, bytes.view(type).view(shape));
  }
  return container;
}

std::pair<std::map<std::string, std::string>, std::string>
read_container_header(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  TORCH_CHECK_VALUE(file.good(), "No container file at ", path);
  auto header = read_header(file, path);
  return {std::move(header.metadata), header.tensors.dump()};
}

std::string native_identity() {
  std::vector<std::string> schemas;
  for (const auto& name : c10::Dispatcher::singleton().getAllOpNames()) {
    if (name.name.rfind("fhelium_", 0) != 0) continue;
    const auto handle = c10::Dispatcher::singleton().findSchema(name);
    if (handle.has_value() && handle->hasSchema())
      schemas.push_back(c10::toString(handle->schema()));
  }
  std::sort(schemas.begin(), schemas.end());
  Sha256 digest;
  for (size_t i = 0; i < schemas.size(); ++i) {
    if (i) digest.update("\n");
    digest.update(schemas[i]);
  }
  nlohmann::json identity;
  identity["fhelium_version"] = FHELIUM_VERSION;
  identity["native_schema_hash"] = digest.hex();
  identity["torch_abi"] = std::to_string(TORCH_VERSION_MAJOR) + "." +
                          std::to_string(TORCH_VERSION_MINOR) + "." +
                          std::to_string(TORCH_VERSION_PATCH)
#if defined(_GLIBCXX_USE_CXX11_ABI)
                          + "-cxx11abi" + std::to_string(_GLIBCXX_USE_CXX11_ABI)
#endif
      ;
  identity["cuda_runtime"] = exec::detail::runtime_version();
  return identity.dump();
}

}  // namespace fhelium::io
