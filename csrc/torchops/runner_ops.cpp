// Registers the exec schedule Runner as torch.classes.fhelium.Runner, its
// suspended runs as torch.classes.fhelium.Execution, shared arenas as
// torch.classes.fhelium.Arena and loaded artifacts as
// torch.classes.fhelium.Artifact, and the container codec as the operators
// fhelium_io::save_container, load_container, read_container_header and
// native_identity.

#include <torch/custom_class.h>
#include <torch/library.h>

#include <cstring>

#include "fhelium/artifact.h"
#include "fhelium/container.h"
#include "fhelium/runner.h"

namespace {

using fhelium::exec::Arena;
using fhelium::exec::Execution;
using fhelium::exec::KernelRecord;
using fhelium::exec::Runner;

void save_container(const std::string& path,
                    const c10::Dict<std::string, at::Tensor>& tensors,
                    const c10::Dict<std::string, std::string>& metadata,
                    bool overwrite) {
  std::vector<std::pair<std::string, at::Tensor>> items;
  for (const auto& item : tensors) items.emplace_back(item.key(), item.value());
  std::map<std::string, std::string> map;
  for (const auto& item : metadata) map.emplace(item.key(), item.value());
  fhelium::io::save_container(path, items, map, overwrite);
}

std::tuple<c10::Dict<std::string, at::Tensor>,
           c10::Dict<std::string, std::string>>
load_container(const std::string& path, const c10::Device& device) {
  auto container = fhelium::io::load_container(path, device);
  c10::Dict<std::string, at::Tensor> tensors;
  for (auto& [name, tensor] : container.tensors) tensors.insert(name, tensor);
  c10::Dict<std::string, std::string> metadata;
  for (auto& [key, value] : container.metadata) metadata.insert(key, value);
  return {std::move(tensors), std::move(metadata)};
}

std::tuple<c10::Dict<std::string, std::string>, std::string>
read_container_header(const std::string& path) {
  auto [map, tensors] = fhelium::io::read_container_header(path);
  c10::Dict<std::string, std::string> metadata;
  for (auto& [key, value] : map) metadata.insert(key, value);
  return {std::move(metadata), std::move(tensors)};
}

// A kernel record from its image bytes in a CPU uint8 Tensor and its launch
// facts.
int64_t add_kernel(const c10::intrusive_ptr<Runner>& self,
                   std::string name,
                   const at::Tensor& image,
                   std::string image_kind,
                   int64_t num_warps,
                   int64_t warp_size,
                   int64_t num_ctas,
                   int64_t shared,
                   std::vector<std::string> parameter_types,
                   std::vector<int64_t> scratch,
                   std::string triton_version) {
  TORCH_CHECK_VALUE(image.device().is_cpu() &&
                        image.scalar_type() == at::kByte && image.dim() == 1,
                    "A kernel image is a one-dimensional CPU uint8 Tensor");
  TORCH_CHECK_VALUE(scratch.size() == 4,
                    "scratch lists the global and profile scratch size and "
                    "alignment");
  const auto bytes = image.contiguous();
  KernelRecord record;
  record.name = std::move(name);
  record.image.resize(static_cast<size_t>(bytes.numel()));
  std::memcpy(record.image.data(), bytes.data_ptr(), record.image.size());
  record.image_kind = std::move(image_kind);
  record.num_warps = num_warps;
  record.warp_size = warp_size;
  record.num_ctas = num_ctas;
  record.shared = shared;
  record.parameter_types = std::move(parameter_types);
  record.global_scratch_size = scratch[0];
  record.global_scratch_align = scratch[1];
  record.profile_scratch_size = scratch[2];
  record.profile_scratch_align = scratch[3];
  record.triton_version = std::move(triton_version);
  return self->kernel(std::move(record));
}

TORCH_LIBRARY(fhelium, m) {
  // Runner methods take an Arena and return an Execution, so both register
  // first.
  m.class_<Arena>("Arena")
      .def(torch::init<int64_t>())
      .def("reserve", &Arena::reserve)
      .def("bytes", &Arena::bytes)
      .def("device", &Arena::device)
      .def("graph_pool_bytes", &Arena::graph_pool_bytes);
  m.class_<Execution>("Execution")
      .def("pending", &Execution::pending)
      .def("arguments", &Execution::arguments)
      .def("integer_arguments", &Execution::integer_arguments)
      .def("float_arguments", &Execution::float_arguments)
      .def("resume", &Execution::resume)
      .def("outputs", &Execution::outputs);
  m.class_<Runner>("Runner")
      .def(torch::init<>())
      .def("buffer", &Runner::buffer)
      .def("int_register", &Runner::int_register)
      .def("float_register", &Runner::float_register)
      .def("inputs", &Runner::inputs)
      .def("outputs", &Runner::outputs)
      .def("bind", &Runner::bind)
      .def("place", &Runner::place)
      .def("arena_bytes", &Runner::arena_bytes)
      .def("read_in_place", &Runner::read_in_place)
      .def("place_output", &Runner::place_output)
      .def("output_bytes", &Runner::output_bytes)
      .def("share_arena", &Runner::share_arena)
      .def("graphs", &Runner::graphs)
      .def("begin_segment", &Runner::begin_segment)
      .def("end_segment", &Runner::end_segment)
      .def("graph_stats", &Runner::graph_stats)
      .def("profile_segments", &Runner::profile_segments)
      .def("segment_stats", &Runner::segment_stats)
      .def("call_op", &Runner::call_op)
      .def("kernel", &add_kernel)
      .def("launch", &Runner::launch)
      .def("view", &Runner::view)
      .def("alloc", &Runner::alloc)
      .def("copy",
           &Runner::copy,
           "",
           {torch::arg("target"),
            torch::arg("source"),
            torch::arg("non_blocking") = false})
      .def("group", &Runner::group, "", {torch::arg("size") = -1})
      .def("bind_group",
           [](const c10::intrusive_ptr<Runner>& self,
              int64_t group,
              const std::string& name) { self->bind_group(group, name); })
      .def("release_groups", &Runner::release_groups)
      .def("group_query", &Runner::group_query)
      .def("collective", &Runner::collective)
      .def("stream", &Runner::stream)
      .def("event", &Runner::event)
      .def("use_stream", &Runner::use_stream)
      .def("record", &Runner::record)
      .def("wait", &Runner::wait)
      .def("host_wait", &Runner::host_wait)
      .def("record_stream", &Runner::record_stream)
      .def("task", &Runner::task)
      .def("begin_task", &Runner::begin_task)
      .def("end_task", &Runner::end_task)
      .def("join", &Runner::join)
      .def("lane_stats", &Runner::lane_stats)
      .def("zero", &Runner::zero)
      .def("assign", &Runner::assign)
      .def("release", &Runner::release)
      .def("call", &Runner::call)
      .def("python_call", &Runner::python_call)
      .def("int_constant", &Runner::int_constant)
      .def("float_constant", &Runner::float_constant)
      .def("int_binary", &Runner::int_binary)
      .def("select", &Runner::select)
      .def("assign_register", &Runner::assign_register)
      .def("begin_for", &Runner::begin_for)
      .def("end_for", &Runner::end_for)
      .def("begin_if", &Runner::begin_if)
      .def("begin_else", &Runner::begin_else)
      .def("end_if", &Runner::end_if)
      .def("finalize", &Runner::finalize)
      .def("run", &Runner::run)
      .def("start", &Runner::start)
      .def("has_python_calls", &Runner::has_python_calls)
      .def("describe", &Runner::describe);
  m.class_<fhelium::Artifact>("Artifact")
      .def(torch::init<std::string, c10::Device>())
      .def("manifest", &fhelium::Artifact::manifest)
      .def("entries", &fhelium::Artifact::entries)
      .def("bind", &fhelium::Artifact::bind)
      .def("bind_key",
           [](const c10::intrusive_ptr<fhelium::Artifact>& self,
              std::string type,
              std::string metadata,
              c10::Dict<std::string, at::Tensor> tensors) {
             fhelium::io::Value key{std::move(type), std::move(metadata), {}};
             for (const auto& item : tensors)
               key.tensors.emplace(item.key(), item.value());
             return self->bind_key(key);
           })
      .def("material", &fhelium::Artifact::material)
      .def("seed_rounding", &fhelium::Artifact::seed_rounding)
      .def("release_groups", &fhelium::Artifact::release_groups)
      .def("bind_group",
           [](const c10::intrusive_ptr<fhelium::Artifact>& self,
              const std::string& symbol,
              const std::string& name) { self->bind_group(symbol, name); })
      .def("unbound", &fhelium::Artifact::unbound)
      .def("prepare", &fhelium::Artifact::prepare)
      .def("refresh", &fhelium::Artifact::refresh)
      .def("runner", &fhelium::Artifact::runner);
}

TORCH_LIBRARY(fhelium_io, m) {
  m.def(
      "save_container(str path, Dict(str, Tensor) tensors, Dict(str, str) "
      "metadata, bool overwrite=False) -> ()",
      &save_container);
  m.def(
      "load_container(str path, Device device) -> (Dict(str, Tensor), "
      "Dict(str, str))",
      &load_container);
  m.def("read_container_header(str path) -> (Dict(str, str), str)",
        &read_container_header);
  m.def("native_identity() -> str", &fhelium::io::native_identity);
}

}  // namespace
