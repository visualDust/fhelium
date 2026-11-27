#include "fhelium/runner.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/scalar_tensor.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Event.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <utility>

#include "graph_capture.h"
#include "kernel_loader.h"

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace fhelium::exec {

#if !defined(FHELIUM_WITH_CUDA) && !defined(FHELIUM_WITH_HIP)
namespace detail {
std::unique_ptr<LoadedKernel> load_kernel(const KernelRecord& record) {
  TORCH_CHECK(false,
              "libfhelium was built without a GPU backend and cannot load "
              "kernel ",
              record.name);
}
std::string device_architecture(int64_t) {
  TORCH_CHECK(false, "libfhelium was built without a GPU backend");
}
std::string runtime_version() { return "none"; }
bool graph_capture_available() { return false; }
int64_t current_graph_device() { return 0; }
GraphPool new_graph_pool() { return {0, 0}; }
int64_t graph_pool_bytes(int64_t, GraphPool) { return 0; }
std::unique_ptr<CapturedGraph> capture_graph(int64_t,
                                             GraphPool,
                                             const std::function<void()>&) {
  TORCH_CHECK(false, "libfhelium was built without a GPU backend");
}
}  // namespace detail
#endif

namespace {

enum class Code : uint8_t {
  call_op,
  launch,
  view,
  alloc,
  copy,
  zero,
  assign,
  release,
  call,
  python_call,
  int_constant,
  float_constant,
  int_binary,
  select,
  assign_register,
  for_init,
  for_next,
  jump_if_false,
  jump,
  segment_begin,
  segment_end,
  nop,
  group_query,
  collective,
  use_stream,
  record,
  wait,
  host_wait,
  record_stream,
  task_begin,
  task_end,
  join,
};

enum class Binary : uint8_t {
  add,
  sub,
  mul,
  div,
  rem,
  min,
  max,
  eq,
  ne,
  lt,
  le,
  gt,
  ge,
};

// One instruction: `index` selects its entry in the per-kind spec lists, and
// a-d are buffer, register or jump operands.
struct Instruction {
  Code code;
  int64_t index = -1;
  int64_t a = -1;
  int64_t b = -1;
  int64_t c = -1;
  int64_t d = -1;
};

struct BufferInfo {
  std::string storage;
  std::optional<std::vector<int64_t>> sizes;
  std::optional<std::vector<int64_t>> strides;
  std::optional<c10::ScalarType> dtype;
  std::optional<c10::Device> device;
  int64_t arena_offset = -1;   // byte offset of a placed arena buffer
  int64_t output_offset = -1;  // byte offset of a placed output buffer
  // A segment reads this buffer, or a view of it, at its run-time address.
  bool in_place = false;

  bool placed() const { return arena_offset >= 0 || output_offset >= 0; }
};

struct Argument {
  ArgumentKind kind;
  c10::IValue value;
  std::vector<int64_t> ids;
};

struct CallSpec {
  std::string schema;
  std::optional<c10::OperatorHandle> handle;
  std::vector<Argument> arguments;
  std::vector<int64_t> results;
  // Built by finalize: the argument stack with every constant and bound buffer
  // in place, and the positions a run fills.
  torch::jit::Stack stack;
  std::vector<std::pair<size_t, int64_t>> buffers;
  std::vector<std::pair<size_t, size_t>> lists;  // position, argument
  std::vector<std::tuple<size_t, int64_t, bool, bool>> registers;
  // Positions of the unaliased Tensor and Tensor list returns copied into
  // `results`; with `unpack`, a list contributes each of its Tensors.
  std::vector<size_t> returned;
  bool unpack = false;
};

struct LaunchParameter {
  enum class Type : uint8_t {
    pointer,
    i1,
    i8,
    i16,
    i32,
    i64,
    u32,
    u64,
    f32,
    f64
  };
  Type type;
  ArgumentKind kind;
  c10::IValue value;
  int64_t id = -1;
  uint64_t bits = 0;  // constant payload
};

struct LaunchSpec {
  int64_t kernel;
  int64_t device;
  std::array<int64_t, 3> grid = {1, 1, 1};
  std::vector<LaunchParameter> parameters;
};

struct KernelEntry {
  KernelRecord record;
  std::shared_ptr<detail::LoadedKernel> loaded;  // loaded by finalize
};

// Kernels loaded by the live Runners of the process, by name and image, so
// that Runners of one schedule load each image once.
std::shared_ptr<detail::LoadedKernel> shared_kernel(
    const KernelRecord& record) {
  struct Loaded {
    std::vector<uint8_t> image;
    std::string kind;
    std::weak_ptr<detail::LoadedKernel> kernel;
  };
  static std::mutex mutex;
  static std::map<std::string, std::vector<Loaded>> loaded;
  std::lock_guard<std::mutex> lock(mutex);
  auto& candidates = loaded[record.name];
  std::shared_ptr<detail::LoadedKernel> kernel;
  candidates.erase(std::remove_if(candidates.begin(),
                                  candidates.end(),
                                  [&](const Loaded& item) {
                                    if (item.kernel.expired()) return true;
                                    if (!kernel &&
                                        item.kind == record.image_kind &&
                                        item.image == record.image)
                                      kernel = item.kernel.lock();
                                    return false;
                                  }),
                   candidates.end());
  if (kernel) return kernel;
  kernel = detail::load_kernel(record);
  candidates.push_back({record.image, record.image_kind, kernel});
  return kernel;
}

struct ViewSpec {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
};

struct CallScheduleSpec {
  c10::intrusive_ptr<Runner> callee;
  std::vector<int64_t> buffers;
  std::vector<int64_t> ints;
  std::vector<int64_t> floats;
  std::vector<int64_t> results;
};

struct Control {
  bool loop;
  size_t start;   // for_init or jump_if_false
  size_t branch;  // jump at the end of the then block
};

struct LiveBuffer {
  int64_t buffer;
  bool written;  // an instruction of the segment may write it in place
};

// The static facts of one segment, derived by finalize from its
// instructions in program order.
struct SegmentSpec {
  size_t begin = 0;  // pc of segment_begin
  size_t end = 0;    // pc of segment_end
  int64_t device = -1;
  // Where begin_segment was called: control depth, block start and branch.
  size_t depth = 0;
  size_t block = 0;
  size_t branch = 0;
  // Unbound buffers and registers read before the segment writes them.
  std::vector<LiveBuffer> live_buffers;
  std::vector<int64_t> live_ints;
  std::vector<int64_t> live_floats;
  // Buffers and registers that the segment assigns.
  std::vector<int64_t> written_buffers;
  std::vector<int64_t> written_ints;
  std::vector<int64_t> written_floats;
  // Whether each written buffer is read after the segment or is an output.
  std::vector<bool> written_live;
  // Groups of the segment's collectives.
  std::vector<int64_t> groups;
};

enum class Collective : uint8_t {
  all_gather,
  broadcast,
  reduce_scatter,
  gather
};

struct CollectiveSpec {
  Collective kind;
  int64_t group;
  std::vector<int64_t> inputs;
  std::vector<int64_t> outputs;
  std::vector<int64_t> options;
};

struct GroupSlot {
  int64_t size = -1;  // -1: any size
  c10::intrusive_ptr<c10d::ProcessGroup> value;
  bool bound = false;
  bool nccl = false;  // has an NCCL backend for CUDA tensors
};

struct StreamSpec {
  int64_t device;
  std::string role;  // "compute" or a pool stream role
};

struct TaskSpec {
  int64_t task;
  std::vector<int64_t> wait_events;
  size_t begin = 0;  // pc of task_begin
  size_t end = 0;    // pc of task_end
};

// The last start of one task in a run.
struct TaskState {
  std::mutex mutex;
  std::condition_variable done;
  bool running = false;
  std::exception_ptr error;
};

// The CPU lane: one worker thread of the process that runs the tasks of all
// Runners in issue order. One persistent thread keeps one OpenMP team, which
// stays warm across tasks.
class LaneWorker {
 public:
  LaneWorker() : thread_([this] { loop(); }) {}
  void post(std::function<void()> function) {
    {
      std::lock_guard lock(mutex_);
      queue_.push_back(std::move(function));
    }
    ready_.notify_one();
  }

 private:
  void loop() {
    for (;;) {
      std::function<void()> function;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return !queue_.empty(); });
        function = std::move(queue_.front());
        queue_.pop_front();
      }
      function();
    }
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::function<void()>> queue_;
  std::thread thread_;
};

// Created on the first task; it lives until the process exits.
LaneWorker& process_lane() {
  static auto* lane = new LaneWorker();
  return *lane;
}

// The streams of one thread during a run: the compute stream and the
// stream current before the run of each device that a use_stream changed.
struct Lane {
  bool active = false;
  std::map<int64_t, c10::Stream> compute;
  std::map<int64_t, c10::Stream> original;
};

Lane& lane() {
  thread_local Lane value;
  return value;
}

const c10::impl::DeviceGuardImplInterface* cuda_guard() {
  return c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);
}

c10::Device cuda_device(int64_t device) {
  return {c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)};
}

// Restores the streams of the thread when the outermost run on it returns.
class LaneScope {
 public:
  LaneScope() : owner_(!lane().active) { lane().active = true; }
  ~LaneScope() {
    if (!owner_) return;
    auto& state = lane();
    for (const auto& [device, stream] : state.original)
      cuda_guard()->exchangeStream(stream);
    state.compute.clear();
    state.original.clear();
    state.active = false;
  }
  LaneScope(const LaneScope&) = delete;
  LaneScope& operator=(const LaneScope&) = delete;

 private:
  bool owner_;
};

int64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

// A Tensor's identity for replay: definedness, address and layout.
struct TensorKey {
  bool defined = false;
  const void* data = nullptr;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  c10::ScalarType dtype = c10::ScalarType::Undefined;
  c10::Device device{c10::kCPU};

  static TensorKey of(const at::Tensor& value) {
    TensorKey key;
    if (!value.defined()) return key;
    key.defined = true;
    key.data = value.data_ptr();
    key.sizes = value.sizes().vec();
    key.strides = value.strides().vec();
    key.dtype = value.scalar_type();
    key.device = value.device();
    return key;
  }
  bool same_layout(const TensorKey& other) const {
    return defined == other.defined && sizes == other.sizes &&
           strides == other.strides && dtype == other.dtype &&
           device == other.device;
  }
};

// One captured graph of a segment and the values it was captured with.
// A use of a graph pool. A pool identifier is valid while one of its graphs
// exists, so the graphs of a pool hold its lease and a capture after the last
// one ends takes a new pool.
struct PoolLease {
  detail::GraphPool id;
};

struct SegmentVariant {
  // The data pointer of each live buffer that the graph reads in place, null
  // for staged, undefined and CPU live buffers.
  std::vector<const void*> key;
  std::shared_ptr<PoolLease> pool;
  std::unique_ptr<detail::CapturedGraph> graph;
  int64_t device = -1;
  std::vector<int64_t> ints;
  std::vector<double> floats;
  std::vector<TensorKey> live;
  // Staging storage of each live buffer whose storage is not Runner-owned.
  std::vector<at::Tensor> staging;
  // written_buffers after capture; undefined for buffers that are dead after
  // the segment.
  std::vector<at::Tensor> buffers;
  // Pool storage of the live written buffers and the Runner-owned storage that
  // receives it after each replay, as byte Tensors.
  std::vector<std::pair<at::Tensor, at::Tensor>> copies;
  std::vector<int64_t> written_ints;
  std::vector<double> written_floats;
};

// Device time between two events recorded around one execution.
struct Interval {
  std::unique_ptr<c10::Event> start;
  std::unique_ptr<c10::Event> end;
};

// Whether an adaptive segment executes through its graph: measuring until
// the decision, then graph or direct for the rest of the Runner's life.
enum class SegmentChoice { measuring, graph, direct };

// Timed replays that an adaptive segment compares with its direct execution.
constexpr size_t kDecisionReplays = 3;

struct SegmentState {
  bool warmed = false;
  SegmentChoice choice = SegmentChoice::measuring;
  // Device time of the first direct execution and of the replays after the
  // capturing one, recorded while measuring and read at the next run after
  // the last of them has completed.
  Interval decision_direct;
  std::array<Interval, kDecisionReplays> decision_replays;
  size_t timed_replays = 0;
  double decision_direct_seconds = std::numeric_limits<double>::quiet_NaN();
  double decision_replay_seconds = std::numeric_limits<double>::quiet_NaN();
  // Captured variants, least recently replayed first.
  std::vector<SegmentVariant> variants;
  // Keys seen once without a variant; a key captures on its second run.
  std::vector<std::vector<const void*>> seen;
  // Facts of the latest capture and, when profiling, of the latest direct
  // execution and replay.
  int64_t nodes = 0;
  double capture_seconds = std::numeric_limits<double>::quiet_NaN();
  double instantiate_seconds = std::numeric_limits<double>::quiet_NaN();
  Interval direct;
  Interval replayed;
};

// Variants kept per segment and keys remembered without one.
constexpr size_t kSegmentVariants = 4;

// Orders reuse of Runner-owned device memory when consecutive runs use
// different streams.
struct Fence {
  int64_t device;
  std::optional<c10::Stream> stream;
  std::unique_ptr<c10::Event> event;
};

const c10::StorageImpl* storage_of(const at::Tensor& value) {
  return value.defined() && value.has_storage()
             ? value.storage().unsafeGetStorageImpl()
             : nullptr;
}

// Byte interval [first, last) of the elements of a nonempty Tensor.
std::pair<uintptr_t, uintptr_t> byte_extent(const at::Tensor& value) {
  int64_t low = 0;
  int64_t high = 0;
  for (int64_t axis = 0; axis < value.dim(); ++axis) {
    const int64_t reach = (value.size(axis) - 1) * value.stride(axis);
    (reach < 0 ? low : high) += reach;
  }
  const auto base = reinterpret_cast<uintptr_t>(value.data_ptr());
  const auto item = static_cast<int64_t>(value.element_size());
  return {base + low * item, base + (high + 1) * item};
}

// The DataPtr context of an output range returned by a run: it keeps the
// Runner-owned storage alive while the caller references the range.
struct OutputRange {
  c10::Storage storage;
};

void delete_output_range(void* context) {
  delete static_cast<OutputRange*>(context);
}

bool is_output_range(const at::Tensor& value) {
  return value.defined() && value.has_storage() &&
         value.storage().data_ptr().get_deleter() == &delete_output_range;
}

// Gives a returned output range newly allocated memory with a copy of its
// contents, enqueued on the current stream.
void move_to_new_memory(c10::StorageImpl& range) {
  const auto bytes = static_cast<int64_t>(range.nbytes());
  const auto device = range.device();
  const auto options = at::TensorOptions().dtype(at::kByte).device(device);
  auto fresh = at::empty({bytes}, options);
  const auto source =
      at::empty({0}, options)
          .set_(c10::Storage(
                    c10::intrusive_ptr<c10::StorageImpl>::reclaim_copy(&range)),
                0,
                {bytes},
                {1});
  fresh.copy_(source);
  range.set_data_ptr_noswap(
      fresh.storage().unsafeGetStorageImpl()->set_data_ptr(
          c10::DataPtr(nullptr, device)));
}

c10::Stream device_stream(int64_t device) {
  return c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)
      ->getStream(c10::Device(c10::DeviceType::CUDA,
                              static_cast<c10::DeviceIndex>(device)));
}

const std::array<std::pair<const char*, Binary>, 13> kBinary = {{
    {"add", Binary::add},
    {"sub", Binary::sub},
    {"mul", Binary::mul},
    {"div", Binary::div},
    {"rem", Binary::rem},
    {"min", Binary::min},
    {"max", Binary::max},
    {"eq", Binary::eq},
    {"ne", Binary::ne},
    {"lt", Binary::lt},
    {"le", Binary::le},
    {"gt", Binary::gt},
    {"ge", Binary::ge},
}};

const char* code_name(Code code) {
  static const char* names[] = {
      "call_op",     "launch",        "view",
      "alloc",       "copy",          "zero",
      "assign",      "release",       "call",
      "python_call", "int_constant",  "float_constant",
      "int_binary",  "select",        "assign_register",
      "for_init",    "for_next",      "jump_if_false",
      "jump",        "segment_begin", "segment_end",
      "nop",         "group_query",   "collective",
      "use_stream",  "record",        "wait",
      "host_wait",   "record_stream", "task_begin",
      "task_end",    "join"};
  return names[static_cast<size_t>(code)];
}

LaunchParameter::Type parameter_type(const std::string& name) {
  using Type = LaunchParameter::Type;
  if (!name.empty() && name[0] == '*') return Type::pointer;
  if (name == "i1" || name == "u1") return Type::i1;
  if (name == "i8" || name == "u8") return Type::i8;
  if (name == "i16" || name == "u16") return Type::i16;
  if (name == "i32") return Type::i32;
  if (name == "u32") return Type::u32;
  if (name == "i64") return Type::i64;
  if (name == "u64") return Type::u64;
  if (name == "fp32") return Type::f32;
  if (name == "fp64") return Type::f64;
  TORCH_CHECK_VALUE(false, "Unsupported kernel parameter type ", name);
}

// Stores `value` in the low bytes of `bits` with the parameter's C type.
uint64_t pack(LaunchParameter::Type type, int64_t integer, double real) {
  using Type = LaunchParameter::Type;
  uint64_t bits = 0;
  auto store = [&](auto value) { std::memcpy(&bits, &value, sizeof(value)); };
  switch (type) {
    case Type::i1:
    case Type::i8:
      store(static_cast<int8_t>(integer));
      break;
    case Type::i16:
      store(static_cast<int16_t>(integer));
      break;
    case Type::i32:
      store(static_cast<int32_t>(integer));
      break;
    case Type::u32:
      store(static_cast<uint32_t>(integer));
      break;
    case Type::i64:
      store(integer);
      break;
    case Type::u64:
      store(static_cast<uint64_t>(integer));
      break;
    case Type::f32:
      store(static_cast<float>(real));
      break;
    case Type::f64:
      store(real);
      break;
    case Type::pointer:
      break;
  }
  return bits;
}

int64_t floor_div(int64_t a, int64_t b) {
  TORCH_CHECK_VALUE(b != 0, "Host integer division by zero");
  const int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int64_t floor_rem(int64_t a, int64_t b) {
  TORCH_CHECK_VALUE(b != 0, "Host integer remainder by zero");
  const int64_t r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}

}  // namespace

struct Arena::State {
  int64_t device;
  at::Tensor region;
  // Incremented when reserve replaces the region.
  int64_t generation = 0;
  Fence fence;
  // Output ranges in the region returned by runs of attached Runners.
  std::vector<c10::intrusive_ptr<c10::StorageImpl>> returned;
  // Graph pool of the attached Runners per device.
  std::map<int64_t, std::weak_ptr<PoolLease>> pools;
};

Arena::Arena(int64_t device) : state_(std::make_unique<State>()) {
  TORCH_CHECK_VALUE(device >= -1, "An Arena device is a CUDA index or -1");
  state_->device = device;
  if (device >= 0)
    state_->fence.event = std::make_unique<c10::Event>(c10::DeviceType::CUDA);
  state_->fence.device = device;
}

Arena::~Arena() = default;

namespace {

// A byte region whose first element is aligned to kRegionAlignment, so
// placed workspaces satisfy the operation workspace alignment on every device.
constexpr int64_t kRegionAlignment = 256;

at::Tensor aligned_region(int64_t bytes, c10::Device device) {
  auto raw = at::empty({bytes + kRegionAlignment},
                       at::TensorOptions().dtype(at::kByte).device(device));
  const auto address = reinterpret_cast<uintptr_t>(raw.data_ptr());
  const auto shift = static_cast<int64_t>(
      (kRegionAlignment - address % kRegionAlignment) % kRegionAlignment);
  return raw.narrow(0, shift, bytes);
}

}  // namespace

void Arena::reserve(int64_t bytes) {
  auto& state = *state_;
  TORCH_CHECK_VALUE(bytes >= 0, "Arena sizes are nonnegative");
  if (state.region.defined() && state.region.numel() >= bytes) return;
  const auto device =
      state.device < 0
          ? c10::Device(c10::kCPU)
          : c10::Device(c10::DeviceType::CUDA,
                        static_cast<c10::DeviceIndex>(state.device));
  state.region = aligned_region((bytes + 255) / 256 * 256, device);
  ++state.generation;
}

int64_t Arena::bytes() const {
  return state_->region.defined() ? state_->region.numel() : 0;
}

int64_t Arena::device() const { return state_->device; }

int64_t Arena::graph_pool_bytes() const {
  int64_t bytes = 0;
  for (const auto& [device, lease] : state_->pools)
    if (const auto pool = lease.lock())
      bytes += detail::graph_pool_bytes(device, pool->id);
  return bytes;
}

// The values of one run.
struct Frame {
  std::vector<at::Tensor> buffers;
  std::vector<int64_t> ints;
  std::vector<double> floats;
  std::vector<std::shared_ptr<TaskState>> tasks;

  Frame() = default;
  Frame(Frame&&) = default;
  Frame& operator=(Frame&&) = default;
  // A running task uses the frame until it ends.
  ~Frame() {
    for (const auto& task : tasks) {
      if (!task) continue;
      std::unique_lock lock(task->mutex);
      task->done.wait(lock, [&] { return !task->running; });
    }
  }
};

struct Program {
  std::vector<BufferInfo> buffers;
  std::vector<at::Tensor> bound;
  std::vector<bool> is_bound;
  int64_t int_count = 0;
  int64_t float_count = 0;
  std::vector<int64_t> input_buffers;
  std::vector<int64_t> input_ints;
  std::vector<int64_t> input_floats;
  std::vector<int64_t> output_buffers;

  std::vector<Instruction> code;
  std::vector<CallSpec> calls;
  std::vector<LaunchSpec> launches;
  std::vector<std::unique_ptr<KernelEntry>> kernels;
  std::vector<ViewSpec> views;
  std::vector<CallScheduleSpec> schedules;
  std::vector<double> float_constants;
  std::vector<Control> control;
  std::vector<GroupSlot> groups;
  std::vector<CollectiveSpec> collectives;
  std::vector<StreamSpec> streams;
  mutable std::vector<std::optional<c10::Stream>> pool_streams;
  int64_t event_count = 0;
  std::vector<std::unique_ptr<c10::Event>> events;
  int64_t task_count = 0;
  std::vector<TaskSpec> tasks;
  std::optional<size_t> open_task;
  size_t task_depth = 0;
  mutable std::atomic<int64_t> tasks_run{0};
  mutable std::atomic<int64_t> task_ns{0};
  mutable int64_t join_ns = 0;
  mutable std::atomic<int64_t> event_waits{0};
  mutable std::atomic<int64_t> collective_calls{0};
  mutable std::atomic<int64_t> collective_wait_ns{0};
  bool finalized = false;
  bool python_calls = false;

  // Arenas, output regions and graph segments; device -1 is the CPU.
  std::vector<std::pair<int64_t, at::Tensor>> arenas;  // device, region
  std::vector<std::pair<int64_t, at::Tensor>> output_regions;
  int64_t arena_requirement = 0;
  c10::intrusive_ptr<Arena> shared_arena;
  int64_t arena_generation = 0;  // of shared_arena when the views were made
  bool graphs = false;
  // Each segment keeps its graph only if its replays are faster than its
  // direct execution.
  bool adaptive = false;
  bool profile = false;
  std::vector<SegmentSpec> segments;
  std::optional<size_t> open_segment;
  std::unordered_set<const c10::StorageImpl*> bound_storages;
  std::vector<int64_t> fence_devices;
  mutable std::vector<SegmentState> segment_states;
  // Graph pool per device without a shared Arena.
  mutable std::map<int64_t, std::weak_ptr<PoolLease>> pools;
  // Arena, output and captured storage; outputs in it are returned as ranges.
  mutable std::unordered_set<const c10::StorageImpl*> retained;
  // Output ranges returned by the last run outside a shared Arena.
  mutable std::vector<c10::intrusive_ptr<c10::StorageImpl>> returned;
  mutable std::vector<Fence> fences;
  // Set when a run drops the graphs; segments then run directly until the
  // run returns.
  mutable bool direct_until_return = false;
  mutable int64_t captures = 0;
  mutable int64_t replays = 0;
  mutable int64_t staged_copies = 0;
  mutable int64_t detached_outputs = 0;
  mutable int64_t live_out_copies = 0;

  const at::Tensor& tensor(const Frame& frame, int64_t buffer) const {
    return is_bound[buffer] ? bound[buffer] : frame.buffers[buffer];
  }

  void check_buffer(int64_t buffer) const {
    TORCH_CHECK_INDEX(
        buffer >= 0 && buffer < static_cast<int64_t>(buffers.size()),
        "Buffer ",
        buffer,
        " is not defined");
  }
  void check_int(int64_t id) const {
    TORCH_CHECK_INDEX(
        id >= 0 && id < int_count, "Int register ", id, " is not defined");
  }
  void check_float(int64_t id) const {
    TORCH_CHECK_INDEX(
        id >= 0 && id < float_count, "Float register ", id, " is not defined");
  }
  void check_building() const {
    TORCH_CHECK(!finalized, "The schedule is finalized");
  }

  Argument argument(ArgumentKind kind, c10::IValue value) const {
    Argument result{kind, std::move(value), {}};
    switch (kind) {
      case ArgumentKind::constant:
        break;
      case ArgumentKind::buffer:
        check_buffer(result.value.toInt());
        break;
      case ArgumentKind::buffer_list:
        for (const auto id : result.value.toIntVector()) {
          if (id != -1) check_buffer(id);
          result.ids.push_back(id);
        }
        break;
      case ArgumentKind::int_register:
        check_int(result.value.toInt());
        break;
      case ArgumentKind::float_register:
        check_float(result.value.toInt());
        break;
      case ArgumentKind::empty_list:
        break;
      default:
        TORCH_CHECK_VALUE(
            false, "Unknown argument kind ", static_cast<int64_t>(kind));
    }
    return result;
  }

  // The stream `id` stands for on the calling thread.
  c10::Stream resolve_stream(int64_t id) const;
  void use_stream(int64_t id) const;
  void run_collective(const CollectiveSpec& spec, Frame& frame) const;
  void start_task(const TaskSpec& spec, Frame& frame) const;
  // Waits for the frame's tasks; rethrows the first error when `rethrow`.
  void join_task(const Frame& frame, int64_t task) const;
  void join_tasks(const Frame& frame) const;
  // Whether the groups of a segment's collectives allow its capture.
  bool capturable(const SegmentSpec& spec) const;
  void build_stack(CallSpec& spec);
  void build_launch(LaunchSpec& spec);
  void analyze_segment(SegmentSpec& spec);
  void allocate_arenas();
  void bind_placed();
  void reset_retained() const;
  // Moves the returned ranges that the caller still references to new memory
  // and forgets all of them.
  void detach(std::vector<c10::intrusive_ptr<c10::StorageImpl>>& ranges) const;
  // Before a top-level run: detaches returned outputs and follows the shared
  // Arena, here and in callees.
  void prepare_run();
  void collect_bound_storages();
  // Runs instructions [pc, end); returns the index of a suspending python
  // call, or `end`. Inside a capture, segments of this schedule and its
  // callees run directly.
  size_t execute(Frame& frame, size_t pc, size_t end, bool capturing) const;
  void run_segment(size_t index, Frame& frame) const;
  // The variant key of the segment's live buffers in `frame`.
  std::vector<const void*> variant_key(const SegmentSpec& spec,
                                       const Frame& frame) const;
  bool replayable(const SegmentSpec& spec,
                  const SegmentVariant& variant,
                  const Frame& frame,
                  int64_t device) const;
  void capture(const SegmentSpec& spec,
               SegmentState& state,
               SegmentVariant& variant,
               Frame& frame,
               int64_t device) const;
  void replay(const SegmentSpec& spec,
              const SegmentVariant& variant,
              Frame& frame) const;
  // Runs `body`, recording `interval` around it on `device` when profiling.
  template <typename Body>
  void timed(Interval& interval, int64_t device, Body&& body) const;
  // Settles the choice of an adaptive segment whose timed runs have
  // completed; a direct choice releases the segment's graphs.
  void decide(SegmentState& state) const;
  void release_variants(SegmentState& state) const;
  void drop_graphs() const;
  // The pools of this Runner's graphs: the shared Arena's or its own.
  std::map<int64_t, std::weak_ptr<PoolLease>>& graph_pools() const {
    return shared_arena ? shared_arena->state_->pools : pools;
  }
  bool owned(const at::Tensor& value) const {
    const auto* storage = storage_of(value);
    return storage != nullptr &&
           (retained.count(storage) != 0 || bound_storages.count(storage) != 0);
  }
  // Stream ordering of top-level runs.
  void begin_run() const;
  void end_run() const;
  void run_call(const CallSpec& spec, Frame& frame) const;
  void run_launch(const LaunchSpec& spec, Frame& frame) const;
  Frame enter(std::vector<at::Tensor> inputs,
              std::vector<int64_t> ints,
              std::vector<double> floats) const;
  std::vector<at::Tensor> leave(const Frame& frame) const;
};

namespace {

// Converts a constant to the schema type where Python and C++ spell it
// differently: an integer for a float argument, and an empty list.
c10::IValue coerce(const c10::IValue& value, const c10::TypePtr& type) {
  auto target = type;
  if (auto optional = target->castRaw<c10::OptionalType>()) {
    if (value.isNone()) return value;
    target = optional->getElementType();
  }
  if (target->kind() == c10::TypeKind::FloatType && value.isInt())
    return static_cast<double>(value.toInt());
  // A Python number in a Tensor argument becomes a wrapped-number Tensor, as
  // the Python argument parser makes it, so it takes part in type promotion
  // as a scalar.
  if (target->kind() == c10::TypeKind::TensorType &&
      (value.isInt() || value.isDouble() || value.isBool())) {
    auto tensor = value.isInt() ? at::scalar_tensor(value.toInt(), at::kLong)
                  : value.isDouble()
                      ? at::scalar_tensor(value.toDouble(), at::kDouble)
                      : at::scalar_tensor(value.toBool(), at::kBool);
    tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
    return tensor;
  }
  if (auto list = target->castRaw<c10::ListType>()) {
    if (value.isList() || value.isIntList() || value.isDoubleList() ||
        value.isBoolList()) {
      const auto element = list->getElementType();
      if (element->kind() == c10::TypeKind::IntType)
        return c10::IValue(value.toList().empty()
                               ? c10::List<int64_t>()
                               : c10::List<int64_t>(value.toIntVector()));
      if (element->kind() == c10::TypeKind::FloatType) {
        std::vector<double> items;
        for (const auto& item : value.toListRef())
          items.push_back(item.isInt() ? static_cast<double>(item.toInt())
                                       : item.toDouble());
        return c10::IValue(c10::List<double>(items));
      }
      if (element->kind() == c10::TypeKind::BoolType) {
        c10::List<bool> items;
        for (const auto& item : value.toListRef())
          items.push_back(item.toBool());
        return c10::IValue(items);
      }
    }
  }
  return value;
}

// An empty list of the list type `type`, or of the list inside an optional.
c10::IValue empty_list(const c10::TypePtr& type) {
  auto target = type;
  if (auto optional = target->castRaw<c10::OptionalType>())
    target = optional->getElementType();
  const auto* list = target->castRaw<c10::ListType>();
  TORCH_CHECK_VALUE(list != nullptr,
                    "An empty-list argument needs a list type, not ",
                    type->str());
  return c10::IValue(c10::impl::GenericList(list->getElementType()));
}

bool optional_tensor_list(const c10::TypePtr& type) {
  auto list = type->castRaw<c10::ListType>();
  return list != nullptr &&
         list->getElementType()->kind() == c10::TypeKind::OptionalType;
}

}  // namespace

void Program::build_stack(CallSpec& spec) {
  const auto& schema = spec.handle->schema();
  spec.stack.clear();
  spec.buffers.clear();
  spec.lists.clear();
  spec.registers.clear();
  for (size_t position = 0; position < spec.arguments.size(); ++position) {
    const auto& argument = spec.arguments[position];
    const auto& type = schema.arguments()[position].type();
    switch (argument.kind) {
      case ArgumentKind::constant:
        spec.stack.push_back(coerce(argument.value, type));
        break;
      case ArgumentKind::buffer: {
        const auto buffer = argument.value.toInt();
        if (is_bound[buffer]) {
          spec.stack.emplace_back(bound[buffer]);
        } else {
          spec.stack.emplace_back();
          spec.buffers.emplace_back(position, buffer);
        }
        break;
      }
      case ArgumentKind::buffer_list: {
        bool dynamic = false;
        for (const auto id : argument.ids)
          dynamic = dynamic || (id != -1 && !is_bound[id]);
        if (dynamic) {
          spec.stack.emplace_back();
          spec.lists.emplace_back(position, position);
        } else if (optional_tensor_list(type)) {
          c10::List<std::optional<at::Tensor>> list;
          list.reserve(argument.ids.size());
          for (const auto id : argument.ids)
            list.push_back(id == -1 ? std::optional<at::Tensor>()
                                    : std::optional<at::Tensor>(bound[id]));
          spec.stack.emplace_back(std::move(list));
        } else {
          c10::List<at::Tensor> list;
          list.reserve(argument.ids.size());
          for (const auto id : argument.ids) list.push_back(bound[id]);
          spec.stack.emplace_back(std::move(list));
        }
        break;
      }
      case ArgumentKind::empty_list:
        spec.stack.push_back(empty_list(type));
        break;
      case ArgumentKind::int_register:
      case ArgumentKind::float_register: {
        auto target = type;
        if (auto optional = target->castRaw<c10::OptionalType>())
          target = optional->getElementType();
        spec.stack.emplace_back();
        spec.registers.emplace_back(
            position,
            argument.value.toInt(),
            argument.kind == ArgumentKind::float_register,
            target->kind() == c10::TypeKind::FloatType);
        break;
      }
    }
  }
}

void Program::run_call(const CallSpec& spec, Frame& frame) const {
  torch::jit::Stack stack = spec.stack;
  for (const auto& [position, buffer] : spec.buffers)
    stack[position] = frame.buffers[buffer];
  const auto& schema = spec.handle->schema();
  for (const auto& [position, index] : spec.lists) {
    const auto& ids = spec.arguments[index].ids;
    if (optional_tensor_list(schema.arguments()[position].type())) {
      c10::List<std::optional<at::Tensor>> list;
      list.reserve(ids.size());
      for (const auto id : ids)
        list.push_back(id == -1 ? std::optional<at::Tensor>()
                                : std::optional<at::Tensor>(tensor(frame, id)));
      stack[position] = std::move(list);
    } else {
      c10::List<at::Tensor> list;
      list.reserve(ids.size());
      for (const auto id : ids) list.push_back(tensor(frame, id));
      stack[position] = std::move(list);
    }
  }
  for (const auto& [position, id, is_float, as_float] : spec.registers) {
    if (is_float) {
      stack[position] = frame.floats[id];
    } else if (as_float) {
      stack[position] = static_cast<double>(frame.ints[id]);
    } else {
      stack[position] = frame.ints[id];
    }
  }
  spec.handle->callBoxed(stack);
  if (spec.unpack) {
    // Tensor list returns contribute each of their Tensors in order.
    size_t next = 0;
    const auto store = [&](at::Tensor value) {
      TORCH_CHECK(next < spec.results.size(),
                  spec.schema,
                  " returned more Tensors than its ",
                  spec.results.size(),
                  " results");
      frame.buffers[spec.results[next++]] = std::move(value);
    };
    for (const auto position : spec.returned) {
      auto& value = stack[position];
      if (value.isTensorList()) {
        for (const auto& item : value.toTensorList()) store(item);
      } else {
        store(value.isNone() ? at::Tensor() : std::move(value).toTensor());
      }
    }
    TORCH_CHECK(next == spec.results.size(),
                spec.schema,
                " returned ",
                next,
                " Tensors for ",
                spec.results.size(),
                " results");
    return;
  }
  for (size_t i = 0; i < spec.results.size(); ++i) {
    auto& value = stack[spec.returned[i]];
    frame.buffers[spec.results[i]] =
        value.isNone() ? at::Tensor() : std::move(value).toTensor();
  }
}

void Program::build_launch(LaunchSpec& spec) {
  for (auto& parameter : spec.parameters) {
    if (parameter.kind != ArgumentKind::constant) continue;
    if (parameter.type == LaunchParameter::Type::pointer) {
      TORCH_CHECK_VALUE(parameter.value.isNone(),
                        "A constant pointer parameter must be None");
      parameter.bits = 0;
    } else if (parameter.value.isDouble()) {
      parameter.bits = pack(parameter.type,
                            static_cast<int64_t>(parameter.value.toDouble()),
                            parameter.value.toDouble());
    } else {
      const auto integer = parameter.value.isBool()
                               ? static_cast<int64_t>(parameter.value.toBool())
                               : parameter.value.toInt();
      parameter.bits =
          pack(parameter.type, integer, static_cast<double>(integer));
    }
  }
}

void Program::run_launch(const LaunchSpec& spec, Frame& frame) const {
  const auto& entry = *kernels[spec.kernel];
  const auto count = spec.parameters.size();
  // Every parameter fits in eight bytes; the last two are the scratch
  // pointers.
  std::vector<uint64_t> values(count + 2, 0);
  std::vector<void*> pointers(count + 2);
  for (size_t i = 0; i < count; ++i) {
    const auto& parameter = spec.parameters[i];
    switch (parameter.kind) {
      case ArgumentKind::constant:
        values[i] = parameter.bits;
        break;
      case ArgumentKind::buffer: {
        const auto& value = tensor(frame, parameter.id);
        const auto address = reinterpret_cast<uintptr_t>(
            value.defined() ? value.data_ptr() : nullptr);
        std::memcpy(&values[i], &address, sizeof(address));
        break;
      }
      case ArgumentKind::int_register:
        values[i] = pack(parameter.type,
                         frame.ints[parameter.id],
                         static_cast<double>(frame.ints[parameter.id]));
        break;
      case ArgumentKind::float_register:
        values[i] = pack(parameter.type,
                         static_cast<int64_t>(frame.floats[parameter.id]),
                         frame.floats[parameter.id]);
        break;
      default:
        break;
    }
    pointers[i] = &values[i];
  }
  const c10::Device device(c10::DeviceType::CUDA,
                           static_cast<c10::DeviceIndex>(spec.device));
  c10::DeviceGuard guard(device);
  const auto& record = entry.record;
  const auto programs = spec.grid[0] * spec.grid[1] * spec.grid[2];
  // Scratch is allocated per launch from the caching allocator, as Triton's
  // launcher does: grid size × blocks per program × per-block size.
  at::Tensor scratch[2];
  const int64_t sizes[2] = {record.global_scratch_size,
                            record.profile_scratch_size};
  for (int k = 0; k < 2; ++k) {
    uintptr_t address = 0;
    if (sizes[k] > 0) {
      scratch[k] =
          at::empty({programs * record.num_ctas * sizes[k]},
                    at::TensorOptions().dtype(at::kByte).device(device));
      address = reinterpret_cast<uintptr_t>(scratch[k].data_ptr());
    }
    std::memcpy(&values[count + k], &address, sizeof(address));
    pointers[count + k] = &values[count + k];
  }
  const auto stream =
      c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)->getStream(device);
  entry.loaded->launch(
      spec.device, spec.grid, pointers.data(), stream.native_handle());
}

Frame Program::enter(std::vector<at::Tensor> inputs,
                     std::vector<int64_t> ints,
                     std::vector<double> floats) const {
  TORCH_CHECK(finalized, "Finalize the schedule before running it");
  TORCH_CHECK_VALUE(inputs.size() == input_buffers.size() &&
                        ints.size() == input_ints.size() &&
                        floats.size() == input_floats.size(),
                    "The schedule takes ",
                    input_buffers.size(),
                    " Tensors, ",
                    input_ints.size(),
                    " integers and ",
                    input_floats.size(),
                    " floats; got ",
                    inputs.size(),
                    ", ",
                    ints.size(),
                    " and ",
                    floats.size());
  Frame frame;
  frame.buffers.resize(buffers.size());
  frame.ints.resize(int_count);
  frame.floats.resize(float_count);
  for (size_t i = 0; i < inputs.size(); ++i)
    frame.buffers[input_buffers[i]] = std::move(inputs[i]);
  for (size_t i = 0; i < ints.size(); ++i) frame.ints[input_ints[i]] = ints[i];
  for (size_t i = 0; i < floats.size(); ++i)
    frame.floats[input_floats[i]] = floats[i];
  frame.tasks.resize(task_count);
  return frame;
}

std::vector<at::Tensor> Program::leave(const Frame& frame) const {
  join_tasks(frame);
  std::vector<at::Tensor> result(output_buffers.size());
  // Outputs in Runner-owned storage, by storage.
  std::map<const c10::StorageImpl*, std::vector<size_t>> groups;
  for (size_t i = 0; i < output_buffers.size(); ++i) {
    const auto& value = tensor(frame, output_buffers[i]);
    const auto* storage = storage_of(value);
    if (storage != nullptr && retained.count(storage) != 0 && value.numel() > 0)
      groups[storage].push_back(i);
    else
      result[i] = value;
  }
  const auto* arena_storage =
      shared_arena ? storage_of(shared_arena->state_->region) : nullptr;
  for (const auto& [storage, members] : groups) {
    const auto& region = tensor(frame, output_buffers[members[0]]).storage();
    const auto base = reinterpret_cast<uintptr_t>(region.data_ptr().get());
    uintptr_t first = std::numeric_limits<uintptr_t>::max();
    uintptr_t last = 0;
    for (const auto i : members) {
      const auto [a, b] = byte_extent(tensor(frame, output_buffers[i]));
      first = std::min(first, a);
      last = std::max(last, b);
    }
    // A 16-byte aligned start keeps every member's offset a whole number of
    // its elements.
    const auto start = (first - base) / 16 * 16;
    auto range = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        static_cast<int64_t>(last - base - start),
        c10::DataPtr(reinterpret_cast<char*>(base) + start,
                     new OutputRange{region},
                     &delete_output_range,
                     region.device()),
        nullptr,
        false);
    for (const auto i : members) {
      const auto& value = tensor(frame, output_buffers[i]);
      const auto item = static_cast<int64_t>(value.element_size());
      const auto offset =
          (value.storage_offset() * item - static_cast<int64_t>(start)) / item;
      result[i] =
          at::empty({0}, value.options())
              .set_(
                  c10::Storage(range), offset, value.sizes(), value.strides());
    }
    (storage == arena_storage ? shared_arena->state_->returned : returned)
        .push_back(std::move(range));
  }
  return result;
}

void Program::detach(
    std::vector<c10::intrusive_ptr<c10::StorageImpl>>& ranges) const {
  for (const auto& range : ranges) {
    if (range.use_count() <= 1) continue;
    move_to_new_memory(*range);
    ++detached_outputs;
  }
  ranges.clear();
}

void Program::prepare_run() {
  // Python call specs have no callee.
  for (auto& call : schedules)
    if (call.callee) call.callee->program_->prepare_run();
  if (shared_arena) {
    auto& state = *shared_arena->state_;
    detach(state.returned);
    if (state.generation != arena_generation) {
      // The Arena grew: the arena views move to its new region.
      arenas = {{state.device, state.region}};
      arena_generation = state.generation;
      bind_placed();
      for (auto& spec : calls) build_stack(spec);
      drop_graphs();
    }
  }
  detach(returned);
}

void Program::begin_run() const {
  direct_until_return = false;
  if (shared_arena && shared_arena->state_->fence.event) {
    auto& fence = shared_arena->state_->fence;
    const auto current = device_stream(fence.device);
    if (fence.stream.has_value() && *fence.stream != current)
      fence.event->block(current);
  }
  for (auto& fence : fences) {
    const auto current = device_stream(fence.device);
    if (fence.stream.has_value() && *fence.stream != current)
      fence.event->block(current);
  }
}

void Program::end_run() const {
  for (auto& fence : fences) {
    const auto current = device_stream(fence.device);
    fence.event->record(current);
    fence.stream = current;
  }
  if (shared_arena && shared_arena->state_->fence.event) {
    auto& fence = shared_arena->state_->fence;
    const auto current = device_stream(fence.device);
    fence.event->record(current);
    fence.stream = current;
  }
}

void Program::drop_graphs() const {
  for (auto& state : segment_states) {
    // The keys stay known, so that each captures again on its next run.
    for (auto& variant : state.variants)
      if (state.seen.size() < kSegmentVariants)
        state.seen.push_back(std::move(variant.key));
    state.variants.clear();
  }
  reset_retained();
  direct_until_return = true;
}

void Program::reset_retained() const {
  retained.clear();
  for (const auto& [device, region] : arenas)
    retained.insert(storage_of(region));
  for (const auto& [device, region] : output_regions)
    retained.insert(storage_of(region));
}

std::vector<const void*> Program::variant_key(const SegmentSpec& spec,
                                              const Frame& frame) const {
  std::vector<const void*> key(spec.live_buffers.size(), nullptr);
  for (size_t i = 0; i < key.size(); ++i) {
    const auto& value = frame.buffers[spec.live_buffers[i].buffer];
    // Runner-owned storage and output ranges returned by Runners hold their
    // contents at a fixed address until their owner runs again, so the graph
    // reads them in place; other device Tensors are staged.
    if (value.defined() && !value.is_cpu() &&
        (buffers[spec.live_buffers[i].buffer].in_place || owned(value) ||
         is_output_range(value)))
      key[i] = value.data_ptr();
  }
  return key;
}

bool Program::replayable(const SegmentSpec& spec,
                         const SegmentVariant& variant,
                         const Frame& frame,
                         int64_t device) const {
  if (variant.device != device) return false;
  for (size_t i = 0; i < spec.live_ints.size(); ++i)
    if (frame.ints[spec.live_ints[i]] != variant.ints[i]) return false;
  for (size_t i = 0; i < spec.live_floats.size(); ++i)
    if (std::memcmp(&frame.floats[spec.live_floats[i]],
                    &variant.floats[i],
                    sizeof(double)) != 0)
      return false;
  for (size_t i = 0; i < spec.live_buffers.size(); ++i)
    if (!TensorKey::of(frame.buffers[spec.live_buffers[i].buffer])
             .same_layout(variant.live[i]))
      return false;
  return true;
}

void Program::capture(const SegmentSpec& spec,
                      SegmentState& state,
                      SegmentVariant& variant,
                      Frame& frame,
                      int64_t device) const {
  variant.device = device;
  variant.ints.clear();
  for (const auto id : spec.live_ints) variant.ints.push_back(frame.ints[id]);
  variant.floats.clear();
  for (const auto id : spec.live_floats)
    variant.floats.push_back(frame.floats[id]);
  const auto count = spec.live_buffers.size();
  variant.live.assign(count, TensorKey());
  variant.staging.assign(count, at::Tensor());
  std::vector<at::Tensor> originals(count);
  // Storage that the graph reads but did not allocate.
  std::unordered_set<const c10::StorageImpl*> outside;
  for (size_t i = 0; i < count; ++i) {
    auto& value = frame.buffers[spec.live_buffers[i].buffer];
    variant.live[i] = TensorKey::of(value);
    if (!value.defined() || value.is_cpu() || variant.key[i] != nullptr) {
      if (const auto* storage = storage_of(value)) outside.insert(storage);
      continue;
    }
    // The graph reads a Runner-owned copy that each replay refreshes.
    variant.staging[i] =
        value.is_non_overlapping_and_dense()
            ? at::empty_strided(value.sizes(), value.strides(), value.options())
            : at::empty(value.sizes(), value.options());
    originals[i] = value;
    value = variant.staging[i];
    outside.insert(storage_of(value));
  }
  auto& lease = graph_pools()[device];
  variant.pool = lease.lock();
  if (!variant.pool) {
    variant.pool =
        std::make_shared<PoolLease>(PoolLease{detail::new_graph_pool()});
    lease = variant.pool;
  }
  try {
    variant.graph = detail::capture_graph(device, variant.pool->id, [&] {
      // Role compute is the capture stream inside the capture.
      auto& state = lane();
      auto compute = std::exchange(state.compute, {});
      auto original = std::exchange(state.original, {});
      const auto restore = [&] {
        for (const auto& [index, stream] : state.compute)
          cuda_guard()->exchangeStream(stream);
        state.compute = std::move(compute);
        state.original = std::move(original);
      };
      try {
        execute(frame, spec.begin + 1, spec.end, true);
        for (const auto& [index, stream] : state.compute)
          TORCH_CHECK_VALUE(
              cuda_guard()->getStream(cuda_device(index)) == stream,
              "Segment ",
              spec.begin,
              " ends on a stream other than compute");
      } catch (...) {
        restore();
        throw;
      }
      restore();
    });
  } catch (...) {
    for (size_t i = 0; i < count; ++i)
      if (originals[i].defined())
        frame.buffers[spec.live_buffers[i].buffer] = originals[i];
    throw;
  }
  ++captures;
  state.nodes = variant.graph->nodes;
  state.capture_seconds = variant.graph->capture_seconds;
  state.instantiate_seconds = variant.graph->instantiate_seconds;
  // Values that outlive the replay leave the pool: the graph's pool memory
  // is free for the other graphs of the pool once the replay ends, so each
  // replay copies those values to Runner-owned storage, one copy per pool
  // storage.
  std::map<const c10::StorageImpl*, c10::Storage> moved;
  variant.buffers.clear();
  variant.copies.clear();
  for (size_t i = 0; i < spec.written_buffers.size(); ++i) {
    const auto& value = frame.buffers[spec.written_buffers[i]];
    if (!spec.written_live[i] || !value.defined()) {
      variant.buffers.emplace_back();
      continue;
    }
    const auto* storage = storage_of(value);
    if (storage == nullptr || value.is_cpu() || owned(value) ||
        outside.count(storage) != 0) {
      variant.buffers.push_back(value);
      continue;
    }
    auto found = moved.find(storage);
    if (found == moved.end()) {
      const auto bytes = static_cast<int64_t>(storage->nbytes());
      const auto options =
          at::TensorOptions().dtype(at::kByte).device(value.device());
      auto source =
          at::from_blob(const_cast<void*>(storage->data()), {bytes}, options);
      auto target = at::empty({bytes}, options);
      variant.copies.emplace_back(source, target);
      found = moved.emplace(storage, target.storage()).first;
      retained.insert(storage_of(target));
    }
    variant.buffers.push_back(at::empty({0}, value.options())
                                  .set_(found->second,
                                        value.storage_offset(),
                                        value.sizes(),
                                        value.strides()));
  }
  variant.written_ints.clear();
  for (const auto id : spec.written_ints)
    variant.written_ints.push_back(frame.ints[id]);
  variant.written_floats.clear();
  for (const auto id : spec.written_floats)
    variant.written_floats.push_back(frame.floats[id]);
  for (size_t i = 0; i < count; ++i)
    if (originals[i].defined())
      frame.buffers[spec.live_buffers[i].buffer] = originals[i];
}

void Program::replay(const SegmentSpec& spec,
                     const SegmentVariant& variant,
                     Frame& frame) const {
  for (size_t i = 0; i < spec.live_buffers.size(); ++i) {
    if (!variant.staging[i].defined()) continue;
    const_cast<at::Tensor&>(variant.staging[i])
        .copy_(frame.buffers[spec.live_buffers[i].buffer]);
    ++staged_copies;
  }
  variant.graph->replay();
  ++replays;
  for (const auto& [source, target] : variant.copies) {
    const_cast<at::Tensor&>(target).copy_(source);
    ++live_out_copies;
  }
  for (size_t i = 0; i < spec.live_buffers.size(); ++i) {
    if (!variant.staging[i].defined() || !spec.live_buffers[i].written)
      continue;
    frame.buffers[spec.live_buffers[i].buffer].copy_(variant.staging[i]);
    ++staged_copies;
  }
  for (size_t i = 0; i < spec.written_buffers.size(); ++i)
    frame.buffers[spec.written_buffers[i]] = variant.buffers[i];
  for (size_t i = 0; i < spec.written_ints.size(); ++i)
    frame.ints[spec.written_ints[i]] = variant.written_ints[i];
  for (size_t i = 0; i < spec.written_floats.size(); ++i)
    frame.floats[spec.written_floats[i]] = variant.written_floats[i];
}

namespace {

template <typename Body>
void record_interval(Interval& interval, int64_t device, Body&& body) {
  for (auto* event : {&interval.start, &interval.end})
    if (!*event)
      *event = std::make_unique<c10::Event>(c10::DeviceType::CUDA,
                                            c10::EventFlag::BACKEND_DEFAULT);
  interval.start->record(device_stream(device));
  body();
  interval.end->record(device_stream(device));
}

double interval_seconds(const Interval& interval) {
  return interval.start->elapsedTime(*interval.end) / 1000.0;
}

}  // namespace

template <typename Body>
void Program::timed(Interval& interval, int64_t device, Body&& body) const {
  if (profile)
    record_interval(interval, device, std::forward<Body>(body));
  else
    body();
}

void Program::release_variants(SegmentState& state) const {
  for (const auto& variant : state.variants)
    for (const auto& [source, target] : variant.copies)
      retained.erase(storage_of(target));
  state.variants.clear();
  state.seen.clear();
}

void Program::decide(SegmentState& state) const {
  if (state.timed_replays < kDecisionReplays) return;
  // Events recorded by earlier runs are read once they have completed, so
  // the decision adds no synchronization.
  if (!state.decision_direct.end->query()) return;
  for (const auto& interval : state.decision_replays)
    if (!interval.end->query()) return;
  std::array<double, kDecisionReplays> replays{};
  for (size_t i = 0; i < kDecisionReplays; ++i)
    replays[i] = interval_seconds(state.decision_replays[i]);
  std::sort(replays.begin(), replays.end());
  state.decision_direct_seconds = interval_seconds(state.decision_direct);
  state.decision_replay_seconds = replays[kDecisionReplays / 2];
  if (state.decision_replay_seconds < state.decision_direct_seconds) {
    state.choice = SegmentChoice::graph;
    return;
  }
  state.choice = SegmentChoice::direct;
  release_variants(state);
}

void Program::run_segment(size_t index, Frame& frame) const {
  const auto& spec = segments[index];
  auto& state = segment_states[index];
  const auto device =
      spec.device >= 0 ? spec.device : detail::current_graph_device();
  const bool measuring = adaptive && state.choice == SegmentChoice::measuring;
  const auto direct = [&] {
    const auto body = [&] {
      timed(state.direct, device, [&] {
        execute(frame, spec.begin + 1, spec.end, true);
      });
    };
    if (measuring && !state.decision_direct.end)
      record_interval(state.decision_direct, device, body);
    else
      body();
  };
  // Replays after the capturing one are timed while measuring.
  const auto replay_variant = [&](const SegmentVariant& variant) {
    const auto body = [&] {
      timed(state.replayed, device, [&] { replay(spec, variant, frame); });
    };
    if (measuring && state.decision_direct.end &&
        state.timed_replays < kDecisionReplays)
      record_interval(
          state.decision_replays[state.timed_replays++], device, body);
    else
      body();
  };
  if (measuring) decide(state);
  if (adaptive && state.choice == SegmentChoice::direct) {
    direct();
    return;
  }
  auto key = variant_key(spec, frame);
  if (!state.warmed || direct_until_return) {
    // The first execution loads lazily initialized state outside a capture.
    direct();
    if (!state.warmed && state.seen.size() < kSegmentVariants)
      state.seen.push_back(std::move(key));
    state.warmed = true;
    return;
  }
  auto& variants = state.variants;
  const auto found =
      std::find_if(variants.begin(), variants.end(), [&](const auto& variant) {
        return variant.key == key;
      });
  if (found != variants.end()) {
    if (!replayable(spec, *found, frame, device)) {
      // Variants share the Runner's pool, so all of them capture again.
      drop_graphs();
      direct();
      return;
    }
    std::rotate(found, found + 1, variants.end());
    replay_variant(variants.back());
    return;
  }
  const auto seen = std::find(state.seen.begin(), state.seen.end(), key);
  if (seen == state.seen.end()) {
    if (state.seen.size() == kSegmentVariants)
      state.seen.erase(state.seen.begin());
    state.seen.push_back(std::move(key));
    direct();
    return;
  }
  state.seen.erase(seen);
  if (adaptive && variants.size() == kSegmentVariants) {
    // Under auto, a segment whose in-place addresses outgrow its variants
    // runs directly from then on instead of capturing again.
    state.choice = SegmentChoice::direct;
    release_variants(state);
    direct();
    return;
  }
  SegmentVariant variant;
  variant.key = std::move(key);
  capture(spec, state, variant, frame, device);
  variants.push_back(std::move(variant));
  // The least recently replayed variant leaves after the capture, so that the
  // pool keeps a graph.
  if (variants.size() > kSegmentVariants) variants.erase(variants.begin());
  timed(state.replayed, device, [&] { replay(spec, variants.back(), frame); });
}

size_t Program::execute(Frame& frame,
                        size_t pc,
                        size_t end,
                        bool capturing) const {
  while (pc < end) {
    const auto& instruction = code[pc];
    switch (instruction.code) {
      case Code::call_op:
        run_call(calls[instruction.index], frame);
        break;
      case Code::launch:
        run_launch(launches[instruction.index], frame);
        break;
      case Code::view: {
        const auto& spec = views[instruction.index];
        const auto& source = tensor(frame, instruction.b);
        frame.buffers[instruction.a] = source.as_strided(
            spec.sizes, spec.strides, source.storage_offset() + spec.offset);
        break;
      }
      case Code::alloc: {
        const auto& buffer = buffers[instruction.a];
        frame.buffers[instruction.a] =
            at::empty_strided(*buffer.sizes,
                              *buffer.strides,
                              at::TensorOptions()
                                  .dtype(*buffer.dtype)
                                  .device(*buffer.device)
                                  .pinned_memory(buffer.storage == "pinned"));
        break;
      }
      case Code::copy:
        // copy_ is a method of the Tensor handle; the buffer keeps its storage.
        const_cast<at::Tensor&>(tensor(frame, instruction.a))
            .copy_(tensor(frame, instruction.b), instruction.c > 0);
        break;
      case Code::zero:
        const_cast<at::Tensor&>(tensor(frame, instruction.a)).zero_();
        break;
      case Code::assign:
        frame.buffers[instruction.a] = tensor(frame, instruction.b);
        break;
      case Code::release:
        frame.buffers[instruction.a].reset();
        break;
      case Code::call: {
        const auto& spec = schedules[instruction.index];
        const auto& callee = *spec.callee->program_;
        std::vector<at::Tensor> inputs;
        inputs.reserve(spec.buffers.size());
        for (const auto buffer : spec.buffers)
          inputs.push_back(tensor(frame, buffer));
        std::vector<int64_t> ints;
        for (const auto id : spec.ints) ints.push_back(frame.ints[id]);
        std::vector<double> floats;
        for (const auto id : spec.floats) floats.push_back(frame.floats[id]);
        auto inner =
            callee.enter(std::move(inputs), std::move(ints), std::move(floats));
        callee.execute(inner, 0, callee.code.size(), capturing);
        callee.join_tasks(inner);
        for (size_t i = 0; i < spec.results.size(); ++i)
          frame.buffers[spec.results[i]] =
              callee.tensor(inner, callee.output_buffers[i]);
        break;
      }
      case Code::python_call:
        return pc;
      case Code::int_constant:
        frame.ints[instruction.a] = instruction.b;
        break;
      case Code::float_constant:
        frame.floats[instruction.a] = float_constants[instruction.index];
        break;
      case Code::int_binary: {
        const auto a = frame.ints[instruction.b];
        const auto b = frame.ints[instruction.c];
        int64_t value = 0;
        switch (static_cast<Binary>(instruction.index)) {
          case Binary::add:
            value = a + b;
            break;
          case Binary::sub:
            value = a - b;
            break;
          case Binary::mul:
            value = a * b;
            break;
          case Binary::div:
            value = floor_div(a, b);
            break;
          case Binary::rem:
            value = floor_rem(a, b);
            break;
          case Binary::min:
            value = std::min(a, b);
            break;
          case Binary::max:
            value = std::max(a, b);
            break;
          case Binary::eq:
            value = a == b;
            break;
          case Binary::ne:
            value = a != b;
            break;
          case Binary::lt:
            value = a < b;
            break;
          case Binary::le:
            value = a <= b;
            break;
          case Binary::gt:
            value = a > b;
            break;
          case Binary::ge:
            value = a >= b;
            break;
        }
        frame.ints[instruction.a] = value;
        break;
      }
      case Code::select:
        frame.ints[instruction.a] = frame.ints[instruction.b]
                                        ? frame.ints[instruction.c]
                                        : frame.ints[instruction.d];
        break;
      case Code::assign_register:
        frame.ints[instruction.a] = frame.ints[instruction.b];
        break;
      case Code::for_init: {
        // a: induction, b: lower, c: upper, d: step; index: loop exit.
        const auto step = frame.ints[instruction.d];
        TORCH_CHECK_VALUE(step != 0, "A host loop has step zero");
        const auto value = frame.ints[instruction.b];
        frame.ints[instruction.a] = value;
        const auto upper = frame.ints[instruction.c];
        if (step > 0 ? value >= upper : value <= upper) {
          pc = static_cast<size_t>(instruction.index);
          continue;
        }
        break;
      }
      case Code::for_next: {
        // a: induction, c: upper, d: step; index: first body instruction.
        const auto step = frame.ints[instruction.d];
        const auto value = frame.ints[instruction.a] + step;
        frame.ints[instruction.a] = value;
        const auto upper = frame.ints[instruction.c];
        if (step > 0 ? value < upper : value > upper) {
          pc = static_cast<size_t>(instruction.index);
          continue;
        }
        break;
      }
      case Code::jump_if_false:
        if (!frame.ints[instruction.a]) {
          pc = static_cast<size_t>(instruction.index);
          continue;
        }
        break;
      case Code::jump:
        pc = static_cast<size_t>(instruction.index);
        continue;
      case Code::segment_begin:
        if (capturing || !graphs || !detail::graph_capture_available() ||
            !capturable(segments[instruction.index]))
          break;
        run_segment(static_cast<size_t>(instruction.index), frame);
        pc = segments[instruction.index].end + 1;
        continue;
      case Code::segment_end:
      case Code::nop:
      case Code::task_end:
        break;
      case Code::group_query: {
        const auto& group = groups[instruction.b];
        TORCH_CHECK_VALUE(
            group.bound, "Group ", instruction.b, " is not bound");
        frame.ints[instruction.a] = instruction.index == 0
                                        ? group.value->getRank()
                                        : group.value->getSize();
        break;
      }
      case Code::collective:
        run_collective(collectives[instruction.index], frame);
        break;
      case Code::use_stream:
        use_stream(instruction.a);
        break;
      case Code::record:
        events[instruction.a]->record(resolve_stream(instruction.b));
        break;
      case Code::wait:
        events[instruction.a]->block(resolve_stream(instruction.b));
        ++event_waits;
        break;
      case Code::host_wait:
        events[instruction.a]->synchronize();
        break;
      case Code::record_stream: {
        const auto& value = tensor(frame, instruction.a);
        if (value.defined() && value.has_storage())
          cuda_guard()->recordDataPtrOnStream(value.storage().data_ptr(),
                                              resolve_stream(instruction.b));
        break;
      }
      case Code::task_begin: {
        const auto& spec = tasks[instruction.index];
        start_task(spec, frame);
        pc = spec.end + 1;
        continue;
      }
      case Code::join:
        join_task(frame, instruction.a);
        break;
    }
    ++pc;
  }
  return end;
}

c10::Stream Program::resolve_stream(int64_t id) const {
  const auto& spec = streams[id];
  if (spec.role == "compute") {
    const auto& state = lane();
    const auto found = state.compute.find(spec.device);
    return found != state.compute.end()
               ? found->second
               : cuda_guard()->getStream(cuda_device(spec.device));
  }
  auto& stream = pool_streams[id];
  if (!stream)
    stream = cuda_guard()->getStreamFromGlobalPool(cuda_device(spec.device));
  return *stream;
}

void Program::use_stream(int64_t id) const {
  auto& state = lane();
  const auto device = streams[id].device;
  // Other devices return to their compute streams.
  for (const auto& [other, stream] : state.compute)
    if (other != device) cuda_guard()->exchangeStream(stream);
  const auto target = resolve_stream(id);
  const auto previous = cuda_guard()->exchangeStream(target);
  if (state.compute.find(device) == state.compute.end())
    state.compute.emplace(device, previous);
  if (state.original.find(device) == state.original.end())
    state.original.emplace(device, previous);
}

void Program::run_collective(const CollectiveSpec& spec, Frame& frame) const {
  const auto& group = groups[spec.group];
  TORCH_CHECK_VALUE(group.bound, "Group ", spec.group, " is not bound");
  auto& output = const_cast<at::Tensor&>(tensor(frame, spec.outputs[0]));
  auto input = tensor(frame, spec.inputs[0]);
  if (!input.is_contiguous()) input = input.contiguous();
  c10::intrusive_ptr<c10d::Work> work;
  switch (spec.kind) {
    case Collective::all_gather: {
      // Backends split the output into one chunk per rank shaped like the
      // input; both are flat.
      auto gathered = output.view({-1});
      auto flat = input.view({-1});
      work = group.value->all_gather_single(gathered, flat);
      break;
    }
    case Collective::broadcast: {
      const auto root = spec.options.at(0);
      if (group.value->getRank() == root && !output.is_same(input))
        output.copy_(input);
      std::vector<at::Tensor> values{output};
      c10d::BroadcastOptions options;
      options.rootRank = root;
      work = group.value->broadcast(values, options);
      break;
    }
    case Collective::reduce_scatter: {
      // Integer sums wrap in two's complement.
      auto reduced = output.view({-1});
      auto flat = input.view({-1});
      c10d::ReduceScatterOptions options;
      options.reduceOp = c10d::ReduceOp::SUM;
      work = group.value->reduce_scatter_single(reduced, flat, options);
      break;
    }
    case Collective::gather: {
      const auto root = spec.options.at(0);
      std::vector<at::Tensor> inputs{input.view({-1})};
      std::vector<std::vector<at::Tensor>> outputs;
      if (group.value->getRank() == root)
        outputs.push_back(output.view({group.value->getSize(), -1}).unbind(0));
      c10d::GatherOptions options;
      options.rootRank = root;
      work = group.value->gather(outputs, inputs, options);
      break;
    }
  }
  ++collective_calls;
  if (group.nccl) {
    work->wait();
  } else {
    const auto start = std::chrono::steady_clock::now();
    work->wait();
    collective_wait_ns += elapsed_ns(start);
  }
}

void Program::start_task(const TaskSpec& spec, Frame& frame) const {
  auto& slot = frame.tasks[spec.task];
  if (!slot) slot = std::make_shared<TaskState>();
  join_task(frame, spec.task);
  // The body begins on the compute streams of this thread.
  std::vector<c10::Stream> compute;
  std::set<int64_t> devices;
  for (const auto& stream : streams) devices.insert(stream.device);
  for (const auto device : devices) {
    const auto& state = lane();
    const auto found = state.compute.find(device);
    compute.push_back(found != state.compute.end()
                          ? found->second
                          : cuda_guard()->getStream(cuda_device(device)));
  }
  slot->running = true;
  auto task = slot;
  process_lane().post(
      [this, &spec, &frame, task, compute = std::move(compute)] {
        const auto start = std::chrono::steady_clock::now();
        std::exception_ptr error;
        try {
          LaneScope scope;
          auto& state = lane();
          for (const auto& stream : compute) {
            state.compute.emplace(stream.device_index(), stream);
            state.original.emplace(stream.device_index(),
                                   cuda_guard()->exchangeStream(stream));
          }
          for (const auto event : spec.wait_events)
            events[event]->synchronize();
          execute(frame, spec.begin + 1, spec.end, false);
        } catch (...) {
          error = std::current_exception();
        }
        task_ns += elapsed_ns(start);
        ++tasks_run;
        {
          std::lock_guard lock(task->mutex);
          task->error = error;
          task->running = false;
        }
        task->done.notify_all();
      });
}

void Program::join_task(const Frame& frame, int64_t task) const {
  const auto& state = frame.tasks[task];
  if (!state) return;
  std::exception_ptr error;
  {
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock lock(state->mutex);
    if (state->running) {
      state->done.wait(lock, [&] { return !state->running; });
      join_ns += elapsed_ns(start);
    }
    error = std::exchange(state->error, nullptr);
  }
  if (error) std::rethrow_exception(error);
}

void Program::join_tasks(const Frame& frame) const {
  for (size_t task = 0; task < frame.tasks.size(); ++task)
    join_task(frame, static_cast<int64_t>(task));
}

bool Program::capturable(const SegmentSpec& spec) const {
  for (const auto group : spec.groups)
    if (!groups[group].nccl) return false;
  return true;
}

// Runner -----------------------------------------------------------------

Runner::Runner() : program_(std::make_unique<Program>()) {}
Runner::~Runner() = default;

int64_t Runner::buffer(const std::string& storage,
                       std::optional<std::vector<int64_t>> sizes,
                       std::optional<std::vector<int64_t>> strides,
                       std::optional<c10::ScalarType> dtype,
                       std::optional<c10::Device> device) {
  auto& p = *program_;
  p.check_building();
  static const char* classes[] = {"input",
                                  "output",
                                  "arena",
                                  "material",
                                  "state",
                                  "internal",
                                  "pinned",
                                  "view"};
  bool known = false;
  for (const auto* name : classes) known = known || storage == name;
  TORCH_CHECK_VALUE(known, "Unknown buffer storage class ", storage);
  TORCH_CHECK_VALUE(!sizes || !strides || sizes->size() == strides->size(),
                    "Buffer sizes and strides differ in rank");
  p.buffers.push_back(
      {storage, std::move(sizes), std::move(strides), dtype, device});
  p.bound.emplace_back();
  p.is_bound.push_back(false);
  return static_cast<int64_t>(p.buffers.size()) - 1;
}

int64_t Runner::int_register() {
  program_->check_building();
  return program_->int_count++;
}

int64_t Runner::float_register() {
  program_->check_building();
  return program_->float_count++;
}

void Runner::inputs(std::vector<int64_t> buffers,
                    std::vector<int64_t> int_registers,
                    std::vector<int64_t> float_registers) {
  auto& p = *program_;
  p.check_building();
  for (const auto id : buffers) p.check_buffer(id);
  for (const auto id : int_registers) p.check_int(id);
  for (const auto id : float_registers) p.check_float(id);
  p.input_buffers = std::move(buffers);
  p.input_ints = std::move(int_registers);
  p.input_floats = std::move(float_registers);
}

void Runner::outputs(std::vector<int64_t> buffers) {
  auto& p = *program_;
  p.check_building();
  for (const auto id : buffers) p.check_buffer(id);
  p.output_buffers = std::move(buffers);
}

void Runner::bind(int64_t buffer, const at::Tensor& value) {
  auto& p = *program_;
  p.check_buffer(buffer);
  const auto& storage = p.buffers[buffer].storage;
  TORCH_CHECK_VALUE(storage == "material" || storage == "state",
                    "Only material and state buffers are bound; buffer ",
                    buffer,
                    " is ",
                    storage);
  TORCH_CHECK_VALUE(
      value.defined(), "Bind a defined Tensor to buffer ", buffer);
  const bool rebinding = p.finalized && p.is_bound[buffer];
  TORCH_CHECK(!p.finalized || p.is_bound[buffer],
              "Buffer ",
              buffer,
              " was unbound when the schedule was finalized");
  const bool same =
      rebinding &&
      TensorKey::of(p.bound[buffer]).same_layout(TensorKey::of(value)) &&
      p.bound[buffer].data_ptr() == value.data_ptr();
  p.bound[buffer] = value;
  p.is_bound[buffer] = true;
  if (!rebinding) return;
  p.collect_bound_storages();
  if (!same) p.drop_graphs();
  // Bound buffers are part of the prebuilt argument stacks.
  for (auto& spec : p.calls) {
    bool uses = false;
    for (const auto& argument : spec.arguments) {
      if (argument.kind == ArgumentKind::buffer)
        uses = uses || argument.value.toInt() == buffer;
      if (argument.kind == ArgumentKind::buffer_list)
        for (const auto id : argument.ids) uses = uses || id == buffer;
    }
    if (uses) p.build_stack(spec);
  }
}

void Runner::call_op(const std::string& schema,
                     std::vector<int64_t> kinds,
                     std::vector<c10::IValue> values,
                     std::vector<int64_t> results) {
  auto& p = *program_;
  p.check_building();
  const auto dot = schema.rfind('.');
  TORCH_CHECK_VALUE(
      dot != std::string::npos && schema.find("::") != std::string::npos,
      "Operator schema ",
      schema,
      " is not namespace::name.overload");
  const auto name = schema.substr(0, dot);
  auto overload = schema.substr(dot + 1);
  if (overload == "default") overload.clear();
  auto handle = c10::Dispatcher::singleton().findSchema({name, overload});
  TORCH_CHECK_VALUE(handle.has_value(),
                    "Operator ",
                    schema,
                    " is not registered with the dispatcher");
  const auto& function = handle->schema();
  TORCH_CHECK_VALUE(kinds.size() == values.size() &&
                        kinds.size() == function.arguments().size(),
                    "Operator ",
                    schema,
                    " takes ",
                    function.arguments().size(),
                    " arguments; got ",
                    kinds.size(),
                    " kinds and ",
                    values.size(),
                    " values");
  CallSpec spec;
  spec.schema = schema;
  spec.handle = handle;
  for (size_t i = 0; i < kinds.size(); ++i)
    spec.arguments.push_back(
        p.argument(static_cast<ArgumentKind>(kinds[i]), std::move(values[i])));
  for (const auto id : results) p.check_buffer(id);
  // Results are the unaliased Tensor returns; other returns, such as the
  // field names of table builders, are not kept.
  const auto& returns = function.returns();
  for (size_t i = 0; i < returns.size(); ++i) {
    const auto* alias = returns[i].alias_info();
    const auto& type = returns[i].type();
    const bool tensor = type->isSubtypeOf(*c10::OptionalType::ofTensor()) ||
                        type->isSubtypeOf(*c10::ListType::ofTensors());
    if (tensor && (alias == nullptr || !alias->isWrite()))
      spec.returned.push_back(i);
  }
  spec.unpack = std::any_of(
      spec.returned.begin(), spec.returned.end(), [&](size_t position) {
        return returns[position].type()->kind() == c10::TypeKind::ListType;
      });
  if (!spec.unpack) {
    TORCH_CHECK_VALUE(spec.returned.size() == results.size(),
                      "Operator ",
                      schema,
                      " returns ",
                      spec.returned.size(),
                      " Tensors; got ",
                      results.size(),
                      " result buffers");
  }
  spec.results = std::move(results);
  p.code.push_back({Code::call_op, static_cast<int64_t>(p.calls.size())});
  p.calls.push_back(std::move(spec));
}

int64_t Runner::kernel(KernelRecord record) {
  auto& p = *program_;
  p.check_building();
  for (const auto& type : record.parameter_types) parameter_type(type);
  TORCH_CHECK_VALUE(record.num_warps > 0 && record.warp_size > 0 &&
                        record.num_ctas > 0 && record.shared >= 0,
                    "Kernel ",
                    record.name,
                    " has invalid launch dimensions");
  auto entry = std::make_unique<KernelEntry>();
  entry->record = std::move(record);
  p.kernels.push_back(std::move(entry));
  return static_cast<int64_t>(p.kernels.size()) - 1;
}

void Runner::launch(int64_t kernel,
                    int64_t device,
                    std::vector<int64_t> grid,
                    std::vector<int64_t> kinds,
                    std::vector<c10::IValue> values) {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK_INDEX(
      kernel >= 0 && kernel < static_cast<int64_t>(p.kernels.size()),
      "Kernel ",
      kernel,
      " is not defined");
  const auto& record = p.kernels[kernel]->record;
  TORCH_CHECK_VALUE(kinds.size() == values.size() &&
                        kinds.size() == record.parameter_types.size(),
                    "Kernel ",
                    record.name,
                    " takes ",
                    record.parameter_types.size(),
                    " runtime parameters; got ",
                    kinds.size());
  TORCH_CHECK_VALUE(!grid.empty() && grid.size() <= 3,
                    "A launch grid has one to three dimensions");
  LaunchSpec spec;
  spec.kernel = kernel;
  spec.device = device;
  for (size_t i = 0; i < grid.size(); ++i) spec.grid[i] = grid[i];
  for (size_t i = 0; i < kinds.size(); ++i) {
    const auto kind = static_cast<ArgumentKind>(kinds[i]);
    const auto type = parameter_type(record.parameter_types[i]);
    auto argument = p.argument(kind, values[i]);
    TORCH_CHECK_VALUE(kind != ArgumentKind::buffer_list,
                      "Kernel parameters do not take buffer lists");
    TORCH_CHECK_VALUE(
        (type == LaunchParameter::Type::pointer) ==
            (kind == ArgumentKind::buffer ||
             (kind == ArgumentKind::constant && argument.value.isNone())),
        "Kernel ",
        record.name,
        " parameter ",
        i,
        " of type ",
        record.parameter_types[i],
        " does not match its argument kind");
    LaunchParameter parameter{type, kind, argument.value};
    if (kind != ArgumentKind::constant) parameter.id = argument.value.toInt();
    spec.parameters.push_back(std::move(parameter));
  }
  p.build_launch(spec);
  p.code.push_back({Code::launch, static_cast<int64_t>(p.launches.size())});
  p.launches.push_back(std::move(spec));
}

void Runner::view(int64_t target,
                  int64_t source,
                  std::vector<int64_t> sizes,
                  std::vector<int64_t> strides,
                  int64_t offset) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(target);
  p.check_buffer(source);
  TORCH_CHECK_VALUE(sizes.size() == strides.size(),
                    "View sizes and strides differ in rank");
  p.code.push_back(
      {Code::view, static_cast<int64_t>(p.views.size()), target, source});
  p.views.push_back({std::move(sizes), std::move(strides), offset});
}

void Runner::alloc(int64_t target) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(target);
  const auto& buffer = p.buffers[target];
  TORCH_CHECK_VALUE(
      buffer.sizes && buffer.strides && buffer.dtype && buffer.device,
      "Buffer ",
      target,
      " has no static layout to allocate");
  p.code.push_back({Code::alloc, -1, target});
}

void Runner::copy(int64_t target, int64_t source, bool non_blocking) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(target);
  p.check_buffer(source);
  p.code.push_back({Code::copy, -1, target, source, non_blocking ? 1 : 0});
}

// Process groups, streams and tasks ----------------------------------------

namespace {
void check_index(int64_t id, size_t count, const char* what) {
  TORCH_CHECK_INDEX(id >= 0 && id < static_cast<int64_t>(count),
                    what,
                    " ",
                    id,
                    " is not defined");
}
}  // namespace

int64_t Runner::group(int64_t size) {
  auto& p = *program_;
  p.check_building();
  p.groups.push_back({size});
  return static_cast<int64_t>(p.groups.size()) - 1;
}

void Runner::bind_group(int64_t group, const std::string& name) {
  bind_group(group, c10d::resolve_process_group(name));
}

void Runner::bind_group(int64_t group,
                        c10::intrusive_ptr<c10d::ProcessGroup> value) {
  auto& p = *program_;
  check_index(group, p.groups.size(), "Group");
  TORCH_CHECK_VALUE(value, "Group ", group, " is bound to no process group");
  auto& slot = p.groups[group];
  TORCH_CHECK_VALUE(slot.size < 0 || value->getSize() == slot.size,
                    "Group ",
                    group,
                    " has ",
                    slot.size,
                    " ranks; process group ",
                    value->getGroupName(),
                    " has ",
                    value->getSize());
  if (slot.bound && slot.value.get() != value.get() && p.finalized)
    p.drop_graphs();
  slot.nccl =
      value->hasBackendForDeviceType(c10::DeviceType::CUDA) &&
      value->getBackend(c10::DeviceType::CUDA)->getBackendName() == "nccl";
  slot.value = std::move(value);
  slot.bound = true;
}

void Runner::release_groups() {
  auto& p = *program_;
  TORCH_CHECK(p.finalized, "Finalize the schedule before releasing its groups");
  p.drop_graphs();
  for (auto& slot : p.groups) {
    slot.value.reset();
    slot.bound = false;
    slot.nccl = false;
  }
  for (auto& spec : p.schedules)
    if (spec.callee) spec.callee->release_groups();
}

void Runner::group_query(const std::string& kind,
                         int64_t target,
                         int64_t group) {
  auto& p = *program_;
  p.check_building();
  p.check_int(target);
  check_index(group, p.groups.size(), "Group");
  TORCH_CHECK_VALUE(
      kind == "rank" || kind == "size", "Unknown group query ", kind);
  p.code.push_back({Code::group_query, kind == "rank" ? 0 : 1, target, group});
}

void Runner::collective(const std::string& kind,
                        int64_t group,
                        std::vector<int64_t> inputs,
                        std::vector<int64_t> outputs,
                        std::vector<int64_t> options) {
  auto& p = *program_;
  p.check_building();
  check_index(group, p.groups.size(), "Group");
  for (const auto id : inputs) p.check_buffer(id);
  for (const auto id : outputs) p.check_buffer(id);
  CollectiveSpec spec{Collective::all_gather,
                      group,
                      std::move(inputs),
                      std::move(outputs),
                      std::move(options)};
  if (kind == "broadcast" || kind == "gather") {
    spec.kind =
        kind == "broadcast" ? Collective::broadcast : Collective::gather;
    TORCH_CHECK_VALUE(
        spec.options.size() == 1, kind, " takes the root rank as its option");
  } else if (kind == "reduce_scatter") {
    spec.kind = Collective::reduce_scatter;
  } else {
    TORCH_CHECK_VALUE(kind == "all_gather", "Unknown collective ", kind);
  }
  TORCH_CHECK_VALUE(spec.inputs.size() == 1 && spec.outputs.size() == 1,
                    kind,
                    " takes one input and one output");
  p.code.push_back(
      {Code::collective, static_cast<int64_t>(p.collectives.size())});
  p.collectives.push_back(std::move(spec));
}

int64_t Runner::stream(int64_t device, const std::string& role) {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK_VALUE(device >= 0, "Streams belong to a GPU device");
  for (size_t id = 0; id < p.streams.size(); ++id)
    if (p.streams[id].device == device && p.streams[id].role == role)
      return static_cast<int64_t>(id);
  p.streams.push_back({device, role});
  return static_cast<int64_t>(p.streams.size()) - 1;
}

int64_t Runner::event() {
  auto& p = *program_;
  p.check_building();
  return p.event_count++;
}

void Runner::use_stream(int64_t stream) {
  auto& p = *program_;
  p.check_building();
  check_index(stream, p.streams.size(), "Stream");
  p.code.push_back({Code::use_stream, -1, stream});
}

void Runner::record(int64_t event, int64_t stream) {
  auto& p = *program_;
  p.check_building();
  check_index(event, static_cast<size_t>(p.event_count), "Event");
  check_index(stream, p.streams.size(), "Stream");
  p.code.push_back({Code::record, -1, event, stream});
}

void Runner::wait(int64_t event, int64_t stream) {
  auto& p = *program_;
  p.check_building();
  check_index(event, static_cast<size_t>(p.event_count), "Event");
  check_index(stream, p.streams.size(), "Stream");
  p.code.push_back({Code::wait, -1, event, stream});
}

void Runner::host_wait(int64_t event) {
  auto& p = *program_;
  p.check_building();
  check_index(event, static_cast<size_t>(p.event_count), "Event");
  p.code.push_back({Code::host_wait, -1, event});
}

void Runner::record_stream(int64_t buffer, int64_t stream) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(buffer);
  check_index(stream, p.streams.size(), "Stream");
  p.code.push_back({Code::record_stream, -1, buffer, stream});
}

int64_t Runner::task() {
  auto& p = *program_;
  p.check_building();
  return p.task_count++;
}

void Runner::begin_task(int64_t task, std::vector<int64_t> wait_events) {
  auto& p = *program_;
  p.check_building();
  check_index(task, static_cast<size_t>(p.task_count), "Task");
  for (const auto event : wait_events)
    check_index(event, static_cast<size_t>(p.event_count), "Event");
  TORCH_CHECK(!p.open_task.has_value(), "Tasks do not nest");
  TORCH_CHECK(!p.open_segment.has_value(), "A segment contains a task");
  TaskSpec spec{task, std::move(wait_events), p.code.size()};
  p.open_task = p.tasks.size();
  p.code.push_back(
      {Code::task_begin, static_cast<int64_t>(p.tasks.size()), task});
  p.tasks.push_back(std::move(spec));
  p.task_depth = p.control.size();
}

void Runner::end_task() {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(p.open_task.has_value(), "end_task without begin_task");
  auto& spec = p.tasks[*p.open_task];
  TORCH_CHECK(p.control.size() == p.task_depth,
              "A task closes in the block that opened it");
  for (size_t pc = spec.begin + 1; pc < p.code.size(); ++pc) {
    switch (p.code[pc].code) {
      case Code::call_op:
      case Code::launch:
      case Code::view:
      case Code::alloc:
      case Code::copy:
      case Code::zero:
      case Code::assign:
      case Code::release:
      case Code::nop:
      case Code::collective:
      case Code::use_stream:
      case Code::record:
      case Code::wait:
      case Code::host_wait:
      case Code::record_stream:
        break;
      default:
        TORCH_CHECK_VALUE(false,
                          "A task body contains ",
                          code_name(p.code[pc].code),
                          " at instruction ",
                          pc);
    }
  }
  spec.end = p.code.size();
  p.code.push_back({Code::task_end, static_cast<int64_t>(*p.open_task)});
  p.open_task.reset();
}

void Runner::join(int64_t task) {
  auto& p = *program_;
  p.check_building();
  check_index(task, static_cast<size_t>(p.task_count), "Task");
  p.code.push_back({Code::join, -1, task});
}

std::vector<double> Runner::lane_stats() const {
  const auto& p = *program_;
  return {static_cast<double>(p.tasks_run.load()),
          static_cast<double>(p.task_ns.load()) * 1e-9,
          static_cast<double>(p.join_ns) * 1e-9,
          static_cast<double>(p.event_waits.load()),
          static_cast<double>(p.collective_calls.load()),
          static_cast<double>(p.collective_wait_ns.load()) * 1e-9};
}

void Runner::zero(int64_t target) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(target);
  p.code.push_back({Code::zero, -1, target});
}

void Runner::assign(int64_t target, int64_t source) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(target);
  p.check_buffer(source);
  p.code.push_back({Code::assign, -1, target, source});
}

void Runner::release(int64_t buffer) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(buffer);
  const auto& storage = p.buffers[buffer].storage;
  TORCH_CHECK_VALUE(storage != "material" && storage != "state",
                    "Bound buffer ",
                    buffer,
                    " cannot be released");
  p.code.push_back({Code::release, -1, buffer});
}

void Runner::call(c10::intrusive_ptr<Runner> callee,
                  std::vector<int64_t> buffers,
                  std::vector<int64_t> int_registers,
                  std::vector<int64_t> float_registers,
                  std::vector<int64_t> results) {
  auto& p = *program_;
  p.check_building();
  const auto& inner = *callee->program_;
  TORCH_CHECK_VALUE(inner.finalized, "Finalize a callee before calling it");
  TORCH_CHECK_VALUE(!inner.python_calls,
                    "A called schedule cannot contain python calls");
  TORCH_CHECK_VALUE(buffers.size() == inner.input_buffers.size() &&
                        int_registers.size() == inner.input_ints.size() &&
                        float_registers.size() == inner.input_floats.size() &&
                        results.size() == inner.output_buffers.size(),
                    "Schedule call arguments do not match the callee");
  for (const auto id : buffers) p.check_buffer(id);
  for (const auto id : int_registers) p.check_int(id);
  for (const auto id : float_registers) p.check_float(id);
  for (const auto id : results) p.check_buffer(id);
  p.code.push_back({Code::call, static_cast<int64_t>(p.schedules.size())});
  p.schedules.push_back({std::move(callee),
                         std::move(buffers),
                         std::move(int_registers),
                         std::move(float_registers),
                         std::move(results)});
}

void Runner::python_call(int64_t callee,
                         std::vector<int64_t> buffers,
                         std::vector<int64_t> int_registers,
                         std::vector<int64_t> float_registers,
                         std::vector<int64_t> results) {
  auto& p = *program_;
  p.check_building();
  for (const auto id : buffers) p.check_buffer(id);
  for (const auto id : int_registers) p.check_int(id);
  for (const auto id : float_registers) p.check_float(id);
  for (const auto id : results) p.check_buffer(id);
  // A python call is a schedule call whose callee the caller runs.
  p.code.push_back(
      {Code::python_call, static_cast<int64_t>(p.schedules.size()), callee});
  p.schedules.push_back({c10::intrusive_ptr<Runner>(),
                         std::move(buffers),
                         std::move(int_registers),
                         std::move(float_registers),
                         std::move(results)});
  p.python_calls = true;
}

void Runner::int_constant(int64_t target, int64_t value) {
  auto& p = *program_;
  p.check_building();
  p.check_int(target);
  p.code.push_back({Code::int_constant, -1, target, value});
}

void Runner::float_constant(int64_t target, double value) {
  auto& p = *program_;
  p.check_building();
  p.check_float(target);
  p.code.push_back({Code::float_constant,
                    static_cast<int64_t>(p.float_constants.size()),
                    target});
  p.float_constants.push_back(value);
}

void Runner::int_binary(const std::string& op,
                        int64_t target,
                        int64_t a,
                        int64_t b) {
  auto& p = *program_;
  p.check_building();
  p.check_int(target);
  p.check_int(a);
  p.check_int(b);
  for (const auto& [name, code] : kBinary) {
    if (op == name) {
      p.code.push_back(
          {Code::int_binary, static_cast<int64_t>(code), target, a, b});
      return;
    }
  }
  TORCH_CHECK_VALUE(false, "Unknown host integer operation ", op);
}

void Runner::select(int64_t target, int64_t condition, int64_t a, int64_t b) {
  auto& p = *program_;
  p.check_building();
  for (const auto id : {target, condition, a, b}) p.check_int(id);
  p.code.push_back({Code::select, -1, target, condition, a, b});
}

void Runner::assign_register(int64_t target, int64_t source) {
  auto& p = *program_;
  p.check_building();
  p.check_int(target);
  p.check_int(source);
  p.code.push_back({Code::assign_register, -1, target, source});
}

void Runner::begin_for(int64_t induction,
                       int64_t lower,
                       int64_t upper,
                       int64_t step) {
  auto& p = *program_;
  p.check_building();
  for (const auto id : {induction, lower, upper, step}) p.check_int(id);
  p.control.push_back({true, p.code.size(), 0});
  p.code.push_back({Code::for_init, -1, induction, lower, upper, step});
}

void Runner::end_for() {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(!p.control.empty() && p.control.back().loop,
              "end_for without begin_for");
  const auto start = p.control.back().start;
  p.control.pop_back();
  auto init = p.code[start];
  p.code.push_back({Code::for_next,
                    static_cast<int64_t>(start + 1),
                    init.a,
                    init.b,
                    init.c,
                    init.d});
  p.code[start].index = static_cast<int64_t>(p.code.size());
}

void Runner::begin_if(int64_t condition) {
  auto& p = *program_;
  p.check_building();
  p.check_int(condition);
  p.control.push_back({false, p.code.size(), 0});
  p.code.push_back({Code::jump_if_false, -1, condition});
}

void Runner::begin_else() {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(!p.control.empty() && !p.control.back().loop &&
                  p.control.back().branch == 0,
              "begin_else without begin_if");
  auto& control = p.control.back();
  control.branch = p.code.size();
  p.code.push_back({Code::jump});
  p.code[control.start].index = static_cast<int64_t>(p.code.size());
}

void Runner::end_if() {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(!p.control.empty() && !p.control.back().loop,
              "end_if without begin_if");
  const auto control = p.control.back();
  p.control.pop_back();
  const auto end = static_cast<int64_t>(p.code.size());
  if (control.branch == 0) {
    p.code[control.start].index = end;
  } else {
    p.code[control.branch].index = end;
  }
}

void Runner::place(int64_t buffer, int64_t offset) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(buffer);
  auto& info = p.buffers[buffer];
  TORCH_CHECK_VALUE(info.storage == "arena",
                    "Only arena buffers are placed; buffer ",
                    buffer,
                    " is ",
                    info.storage);
  TORCH_CHECK_VALUE(info.sizes && info.strides && info.dtype && info.device,
                    "Arena buffer ",
                    buffer,
                    " has no static layout");
  const auto element = static_cast<int64_t>(c10::elementSize(*info.dtype));
  TORCH_CHECK_VALUE(offset >= 0 && offset % element == 0,
                    "Arena offset ",
                    offset,
                    " of buffer ",
                    buffer,
                    " is not a multiple of its element size ",
                    element);
  info.arena_offset = offset;
}

int64_t Runner::arena_bytes() const { return program_->arena_requirement; }

void Runner::read_in_place(std::vector<int64_t> buffers) {
  auto& p = *program_;
  p.check_building();
  for (const auto buffer : buffers) {
    p.check_buffer(buffer);
    auto& info = p.buffers[buffer];
    TORCH_CHECK_VALUE(info.storage == "input",
                      "Only input buffers are read in place; buffer ",
                      buffer,
                      " is ",
                      info.storage);
    info.in_place = true;
  }
}

void Runner::place_output(int64_t buffer, int64_t offset) {
  auto& p = *program_;
  p.check_building();
  p.check_buffer(buffer);
  auto& info = p.buffers[buffer];
  TORCH_CHECK_VALUE(info.storage == "output",
                    "Only output buffers are placed with place_output; buffer ",
                    buffer,
                    " is ",
                    info.storage);
  TORCH_CHECK_VALUE(info.sizes && info.strides && info.dtype && info.device,
                    "Output buffer ",
                    buffer,
                    " has no static layout");
  const auto element = static_cast<int64_t>(c10::elementSize(*info.dtype));
  TORCH_CHECK_VALUE(offset >= 0 && offset % element == 0,
                    "Output offset ",
                    offset,
                    " of buffer ",
                    buffer,
                    " is not a multiple of its element size ",
                    element);
  info.output_offset = offset;
}

int64_t Runner::output_bytes() const {
  int64_t total = 0;
  for (const auto& [device, region] : program_->output_regions)
    total += region.numel();
  return total;
}

void Runner::share_arena(c10::intrusive_ptr<Arena> arena) {
  program_->check_building();
  TORCH_CHECK_VALUE(arena, "share_arena requires an Arena");
  program_->shared_arena = std::move(arena);
}

void Runner::graphs(const std::string& mode) {
  program_->check_building();
  TORCH_CHECK_VALUE(mode == "off" || mode == "on" || mode == "auto",
                    "graphs requires mode off, on or auto, not ",
                    mode);
  program_->graphs = mode != "off";
  program_->adaptive = mode == "auto";
}

void Runner::profile_segments(bool enable) {
  program_->check_building();
  program_->profile = enable;
}

std::vector<std::vector<double>> Runner::segment_stats() const {
  const auto& p = *program_;
  const auto seconds = [](const Interval& interval) {
    if (!interval.end || !interval.end->was_marked_for_recording())
      return std::numeric_limits<double>::quiet_NaN();
    interval.end->synchronize();
    return interval.start->elapsedTime(*interval.end) / 1000.0;
  };
  std::vector<std::vector<double>> rows;
  for (const auto& state : p.segment_states) {
    // 0 measuring, 1 graph, 2 direct; segments of an always-capturing Runner
    // report graph.
    const auto choice = p.adaptive ? static_cast<double>(state.choice)
                                   : static_cast<double>(SegmentChoice::graph);
    rows.push_back({static_cast<double>(state.nodes),
                    state.capture_seconds,
                    state.instantiate_seconds,
                    seconds(state.direct),
                    seconds(state.replayed),
                    static_cast<double>(state.variants.size()),
                    choice,
                    state.decision_direct_seconds,
                    state.decision_replay_seconds});
  }
  return rows;
}

void Runner::begin_segment(int64_t device) {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(!p.open_segment.has_value(), "Segments do not nest");
  TORCH_CHECK(!p.open_task.has_value(), "A task contains a segment");
  SegmentSpec spec;
  spec.begin = p.code.size();
  spec.device = device;
  spec.depth = p.control.size();
  if (!p.control.empty()) {
    spec.block = p.control.back().start;
    spec.branch = p.control.back().branch;
  }
  p.open_segment = p.segments.size();
  p.code.push_back(
      {Code::segment_begin, static_cast<int64_t>(p.segments.size())});
  p.segments.push_back(std::move(spec));
}

void Runner::end_segment() {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(p.open_segment.has_value(), "end_segment without begin_segment");
  auto& spec = p.segments[*p.open_segment];
  const bool same_block =
      p.control.size() == spec.depth &&
      (p.control.empty() || (p.control.back().start == spec.block &&
                             p.control.back().branch == spec.branch));
  TORCH_CHECK(same_block, "A segment closes in the block that opened it");
  spec.end = p.code.size();
  p.code.push_back({Code::segment_end, static_cast<int64_t>(*p.open_segment)});
  p.open_segment.reset();
}

std::vector<int64_t> Runner::graph_stats() const {
  const auto& p = *program_;
  int64_t graphs = 0;
  for (const auto& state : p.segment_states)
    graphs += static_cast<int64_t>(state.variants.size());
  int64_t pool_bytes = 0;
  for (const auto& [device, lease] : p.graph_pools())
    if (const auto pool = lease.lock())
      pool_bytes += detail::graph_pool_bytes(device, pool->id);
  return {static_cast<int64_t>(p.segments.size()),
          p.captures,
          p.replays,
          p.staged_copies,
          p.detached_outputs,
          graphs,
          pool_bytes,
          p.live_out_copies};
}

void Program::collect_bound_storages() {
  bound_storages.clear();
  for (size_t buffer = 0; buffer < buffers.size(); ++buffer)
    if (is_bound[buffer] && !buffers[buffer].placed())
      if (const auto* storage = storage_of(bound[buffer]))
        bound_storages.insert(storage);
}

namespace {

// The byte extent of the buffers placed at `offset`, by device (-1 for CPU).
std::map<int64_t, std::pair<c10::Device, int64_t>> placed_extents(
    const std::vector<BufferInfo>& buffers, int64_t BufferInfo::* offset) {
  std::map<int64_t, std::pair<c10::Device, int64_t>> extents;
  for (const auto& info : buffers) {
    if (info.*offset < 0) continue;
    int64_t elements = 1;
    for (size_t d = 0; d < info.sizes->size(); ++d) {
      if ((*info.sizes)[d] == 0) {
        elements = 0;
        break;
      }
      elements += ((*info.sizes)[d] - 1) * (*info.strides)[d];
    }
    const auto bytes =
        info.*offset +
        elements * static_cast<int64_t>(c10::elementSize(*info.dtype));
    const auto key = info.device->is_cpu() ? -1 : info.device->index();
    auto& extent = extents.emplace(key, std::make_pair(*info.device, 0))
                       .first->second.second;
    extent = std::max(extent, bytes);
  }
  return extents;
}

int64_t round_region(int64_t bytes) { return (bytes + 255) / 256 * 256; }

}  // namespace

void Program::allocate_arenas() {
  const auto arena_extents = placed_extents(buffers, &BufferInfo::arena_offset);
  arena_requirement = 0;
  for (const auto& [key, extent] : arena_extents)
    arena_requirement += round_region(extent.second);
  if (shared_arena) {
    auto& state = *shared_arena->state_;
    for (const auto& [key, extent] : arena_extents)
      TORCH_CHECK_VALUE(key == state.device,
                        "Arena buffers on device ",
                        key,
                        " cannot use the shared Arena of device ",
                        state.device);
    if (!arena_extents.empty()) {
      shared_arena->reserve(arena_requirement);
      arenas = {{state.device, state.region}};
      arena_generation = state.generation;
    } else {
      shared_arena.reset();
    }
  } else {
    for (const auto& [key, extent] : arena_extents)
      arenas.emplace_back(
          key, aligned_region(round_region(extent.second), extent.first));
  }
  for (const auto& [key, extent] :
       placed_extents(buffers, &BufferInfo::output_offset))
    output_regions.emplace_back(
        key, aligned_region(round_region(extent.second), extent.first));
  for (const auto* regions : {&arenas, &output_regions})
    for (const auto& [key, region] : *regions)
      if (key >= 0 &&
          std::find(fence_devices.begin(), fence_devices.end(), key) ==
              fence_devices.end())
        fence_devices.push_back(key);
  bind_placed();
  reset_retained();
}

void Program::bind_placed() {
  const auto region_of = [](const auto& regions, int64_t key) {
    return std::find_if(regions.begin(),
                        regions.end(),
                        [&](const auto& entry) { return entry.first == key; })
        ->second;
  };
  for (size_t buffer = 0; buffer < buffers.size(); ++buffer) {
    const auto& info = buffers[buffer];
    if (!info.placed()) continue;
    const auto key = info.device->is_cpu() ? -1 : info.device->index();
    const bool arena = info.arena_offset >= 0;
    const auto& region = region_of(arena ? arenas : output_regions, key);
    const auto element = static_cast<int64_t>(c10::elementSize(*info.dtype));
    const auto offset = arena ? info.arena_offset : info.output_offset;
    bound[buffer] = at::empty({0}, region.options().dtype(*info.dtype))
                        .set_(region.storage(),
                              (region.storage_offset() + offset) / element,
                              *info.sizes,
                              *info.strides);
    is_bound[buffer] = true;
  }
}

void Program::analyze_segment(SegmentSpec& spec) {
  std::vector<int8_t> defined(buffers.size(), 0);
  std::vector<int8_t> defined_ints(int_count, 0);
  std::vector<int8_t> defined_floats(float_count, 0);
  std::vector<int64_t> live_index(buffers.size(), -1);
  std::set<int64_t> written;
  std::set<int64_t> written_ints;
  std::set<int64_t> written_floats;
  const auto read = [&](int64_t buffer, bool in_place) {
    if (buffer < 0 || is_bound[buffer]) return;
    if (!defined[buffer]) {
      if (live_index[buffer] < 0) {
        live_index[buffer] = static_cast<int64_t>(spec.live_buffers.size());
        spec.live_buffers.push_back({buffer, false});
      }
    }
    if (in_place && live_index[buffer] >= 0)
      spec.live_buffers[live_index[buffer]].written = true;
  };
  const auto write = [&](int64_t buffer) {
    defined[buffer] = 1;
    written.insert(buffer);
  };
  const auto read_int = [&](int64_t id) {
    if (!defined_ints[id] &&
        std::find(spec.live_ints.begin(), spec.live_ints.end(), id) ==
            spec.live_ints.end())
      spec.live_ints.push_back(id);
  };
  const auto write_int = [&](int64_t id) {
    defined_ints[id] = 1;
    written_ints.insert(id);
  };
  const auto read_float = [&](int64_t id) {
    if (!defined_floats[id] &&
        std::find(spec.live_floats.begin(), spec.live_floats.end(), id) ==
            spec.live_floats.end())
      spec.live_floats.push_back(id);
  };
  const auto write_float = [&](int64_t id) {
    defined_floats[id] = 1;
    written_floats.insert(id);
  };
  for (size_t pc = spec.begin + 1; pc < spec.end; ++pc) {
    const auto& instruction = code[pc];
    switch (instruction.code) {
      case Code::call_op: {
        const auto& call = calls[instruction.index];
        const auto& schema = call.handle->schema().arguments();
        for (size_t i = 0; i < call.arguments.size(); ++i) {
          const auto& argument = call.arguments[i];
          const auto* alias = schema[i].alias_info();
          const bool in_place = alias != nullptr && alias->isWrite();
          switch (argument.kind) {
            case ArgumentKind::buffer:
              read(argument.value.toInt(), in_place);
              break;
            case ArgumentKind::buffer_list:
              for (const auto id : argument.ids) read(id, in_place);
              break;
            case ArgumentKind::int_register:
              read_int(argument.value.toInt());
              break;
            case ArgumentKind::float_register:
              read_float(argument.value.toInt());
              break;
            default:
              break;
          }
        }
        for (const auto id : call.results) write(id);
        break;
      }
      case Code::launch:
        for (const auto& parameter : launches[instruction.index].parameters) {
          if (parameter.kind == ArgumentKind::buffer) read(parameter.id, true);
          if (parameter.kind == ArgumentKind::int_register)
            read_int(parameter.id);
          if (parameter.kind == ArgumentKind::float_register)
            read_float(parameter.id);
        }
        break;
      case Code::view:
      case Code::assign:
        read(instruction.b, false);
        write(instruction.a);
        break;
      case Code::alloc:
      case Code::release:
        write(instruction.a);
        break;
      case Code::copy:
        read(instruction.b, false);
        read(instruction.a, true);
        break;
      case Code::zero:
        read(instruction.a, true);
        break;
      case Code::call: {
        const auto& call = schedules[instruction.index];
        for (const auto id : call.buffers) read(id, true);
        for (const auto id : call.ints) read_int(id);
        for (const auto id : call.floats) read_float(id);
        for (const auto id : call.results) write(id);
        break;
      }
      case Code::python_call:
        TORCH_CHECK_VALUE(false, "A segment contains a python call");
        break;
      case Code::int_constant:
        write_int(instruction.a);
        break;
      case Code::float_constant:
        write_float(instruction.a);
        break;
      case Code::int_binary:
        read_int(instruction.b);
        read_int(instruction.c);
        write_int(instruction.a);
        break;
      case Code::select:
        read_int(instruction.b);
        read_int(instruction.c);
        read_int(instruction.d);
        write_int(instruction.a);
        break;
      case Code::assign_register:
        read_int(instruction.b);
        write_int(instruction.a);
        break;
      case Code::for_init:
        read_int(instruction.b);
        read_int(instruction.c);
        read_int(instruction.d);
        write_int(instruction.a);
        break;
      case Code::for_next:
        read_int(instruction.a);
        read_int(instruction.c);
        read_int(instruction.d);
        write_int(instruction.a);
        break;
      case Code::jump_if_false:
        read_int(instruction.a);
        break;
      case Code::segment_begin:
        TORCH_CHECK_VALUE(false, "Segments do not nest");
        break;
      case Code::group_query:
        write_int(instruction.a);
        break;
      case Code::collective: {
        const auto& collective = collectives[instruction.index];
        for (const auto id : collective.inputs) read(id, false);
        for (const auto id : collective.outputs) read(id, true);
        if (std::find(spec.groups.begin(),
                      spec.groups.end(),
                      collective.group) == spec.groups.end())
          spec.groups.push_back(collective.group);
        break;
      }
      case Code::record_stream:
        read(instruction.a, false);
        break;
      case Code::task_begin:
      case Code::join:
      case Code::host_wait:
        TORCH_CHECK_VALUE(false,
                          "A segment contains ",
                          code_name(instruction.code),
                          " at instruction ",
                          pc);
        break;
      default:
        break;
    }
  }
  spec.written_buffers.assign(written.begin(), written.end());
  spec.written_ints.assign(written_ints.begin(), written_ints.end());
  spec.written_floats.assign(written_floats.begin(), written_floats.end());
  // A written buffer is live when a later instruction reads it, in place
  // included, before assigning it, or when it is an output. Segments are
  // top-level, so the instructions after the segment run in program order.
  std::map<int64_t, int8_t> state;  // 1 live, -1 assigned first
  for (const auto buffer : written) state[buffer] = 0;
  const auto use = [&](int64_t buffer) {
    auto found = state.find(buffer);
    if (found != state.end() && found->second == 0) found->second = 1;
  };
  const auto assign = [&](int64_t buffer) {
    auto found = state.find(buffer);
    if (found != state.end() && found->second == 0) found->second = -1;
  };
  for (size_t pc = spec.end + 1; pc < code.size(); ++pc) {
    const auto& instruction = code[pc];
    switch (instruction.code) {
      case Code::call_op: {
        const auto& call = calls[instruction.index];
        for (const auto& argument : call.arguments) {
          if (argument.kind == ArgumentKind::buffer)
            use(argument.value.toInt());
          if (argument.kind == ArgumentKind::buffer_list)
            for (const auto id : argument.ids) use(id);
        }
        for (const auto id : call.results) assign(id);
        break;
      }
      case Code::launch:
        for (const auto& parameter : launches[instruction.index].parameters)
          if (parameter.kind == ArgumentKind::buffer) use(parameter.id);
        break;
      case Code::view:
      case Code::assign:
        use(instruction.b);
        assign(instruction.a);
        break;
      case Code::alloc:
      case Code::release:
        assign(instruction.a);
        break;
      case Code::copy:
        use(instruction.b);
        use(instruction.a);
        break;
      case Code::zero:
        use(instruction.a);
        break;
      case Code::call:
      case Code::python_call: {
        const auto& call = schedules[instruction.index];
        for (const auto id : call.buffers) use(id);
        for (const auto id : call.results) assign(id);
        break;
      }
      case Code::collective: {
        const auto& collective = collectives[instruction.index];
        for (const auto id : collective.inputs) use(id);
        for (const auto id : collective.outputs) use(id);
        break;
      }
      case Code::record_stream:
        use(instruction.a);
        break;
      default:
        break;
    }
  }
  for (const auto buffer : output_buffers) use(buffer);
  spec.written_live.clear();
  for (const auto buffer : spec.written_buffers)
    spec.written_live.push_back(state[buffer] == 1);
}

void Runner::finalize() {
  auto& p = *program_;
  p.check_building();
  TORCH_CHECK(p.control.empty(), "A control-flow block is still open");
  TORCH_CHECK(!p.open_segment.has_value(), "A segment is still open");
  TORCH_CHECK(!p.open_task.has_value(), "A task is still open");
  for (size_t group = 0; group < p.groups.size(); ++group)
    TORCH_CHECK_VALUE(p.groups[group].bound, "Group ", group, " is not bound");
  for (int64_t event = 0; event < p.event_count; ++event)
    p.events.push_back(std::make_unique<c10::Event>(c10::DeviceType::CUDA));
  p.pool_streams.assign(p.streams.size(), std::nullopt);
  for (size_t buffer = 0; buffer < p.buffers.size(); ++buffer) {
    const auto& storage = p.buffers[buffer].storage;
    TORCH_CHECK_VALUE(
        p.is_bound[buffer] || (storage != "material" && storage != "state"),
        "Buffer ",
        buffer,
        " (",
        storage,
        ") is not bound");
  }
  // Placed buffers take no per-run storage: their alloc and release are no-ops,
  // and no instruction assigns them.
  p.allocate_arenas();
  const auto placed = [&](int64_t buffer) {
    return buffer >= 0 && p.buffers[buffer].placed();
  };
  for (auto& instruction : p.code) {
    const auto code = instruction.code;
    if ((code == Code::alloc || code == Code::release) && placed(instruction.a))
      instruction = {Code::nop};
    bool assigns = false;
    if (code == Code::view || code == Code::assign)
      assigns = placed(instruction.a);
    if (code == Code::call_op)
      for (const auto id : p.calls[instruction.index].results)
        assigns = assigns || placed(id);
    if (code == Code::call || code == Code::python_call)
      for (const auto id : p.schedules[instruction.index].results)
        assigns = assigns || placed(id);
    TORCH_CHECK_VALUE(
        !assigns,
        "An instruction assigns a placed buffer; placed buffers are "
        "written in place");
    // Views of an input read in place are read in place too.
    if ((code == Code::view || code == Code::assign) &&
        p.buffers[instruction.b].in_place)
      p.buffers[instruction.a].in_place = true;
  }
  p.collect_bound_storages();
  for (auto& spec : p.segments) p.analyze_segment(spec);
  p.segment_states.resize(p.segments.size());
  if (p.graphs && !p.segments.empty() && detail::graph_capture_available()) {
    for (const auto& spec : p.segments) {
      const auto device =
          spec.device >= 0 ? spec.device : detail::current_graph_device();
      if (std::find(p.fence_devices.begin(), p.fence_devices.end(), device) ==
          p.fence_devices.end())
        p.fence_devices.push_back(device);
    }
  }
  for (const auto device : p.fence_devices)
    p.fences.push_back({device,
                        std::nullopt,
                        std::make_unique<c10::Event>(c10::DeviceType::CUDA)});
  // Every launched kernel loads now, so that no load happens in a run.
  std::set<std::pair<int64_t, int64_t>> prepared;
  for (const auto& spec : p.launches) {
    auto& entry = *p.kernels[spec.kernel];
    if (!entry.loaded) entry.loaded = shared_kernel(entry.record);
    if (prepared.emplace(spec.kernel, spec.device).second)
      entry.loaded->prepare(spec.device);
  }
  for (auto& spec : p.calls) p.build_stack(spec);
  p.finalized = true;
}

std::vector<at::Tensor> Runner::run(std::vector<at::Tensor> inputs,
                                    std::vector<int64_t> ints,
                                    std::vector<double> floats) {
  TORCH_CHECK(!program_->python_calls,
              "The schedule contains python calls; run it with start()");
  TORCH_CHECK(program_->finalized, "Finalize the schedule before running it");
  program_->prepare_run();
  const auto& p = *program_;
  LaneScope lanes;
  auto frame = p.enter(std::move(inputs), std::move(ints), std::move(floats));
  p.begin_run();
  p.execute(frame, 0, p.code.size(), false);
  auto outputs = p.leave(frame);
  p.end_run();
  return outputs;
}

bool Runner::has_python_calls() const { return program_->python_calls; }

std::string Runner::describe() const {
  const auto& p = *program_;
  std::ostringstream out;
  out << "buffers " << p.buffers.size() << ", int registers " << p.int_count
      << ", float registers " << p.float_count << "\n";
  for (size_t pc = 0; pc < p.code.size(); ++pc) {
    const auto& instruction = p.code[pc];
    out << pc << ": " << code_name(instruction.code);
    if (instruction.code == Code::call_op)
      out << " " << p.calls[instruction.index].schema;
    if (instruction.code == Code::launch)
      out << " "
          << p.kernels[p.launches[instruction.index].kernel]->record.name;
    out << " " << instruction.index << " " << instruction.a << " "
        << instruction.b << " " << instruction.c << " " << instruction.d
        << "\n";
  }
  return out.str();
}

// Execution --------------------------------------------------------------

struct Execution::State {
  Frame frame;
  size_t pc = 0;
  std::vector<at::Tensor> outputs;  // set when the schedule returns
};

Execution::Execution(c10::intrusive_ptr<Runner> runner)
    : runner_(std::move(runner)), state_(std::make_unique<State>()) {}
Execution::~Execution() = default;

c10::intrusive_ptr<Execution> Runner::start(std::vector<at::Tensor> inputs,
                                            std::vector<int64_t> ints,
                                            std::vector<double> floats) {
  auto execution = c10::intrusive_ptr<Execution>::reclaim(
      new Execution(c10::intrusive_ptr<Runner>::reclaim_copy(this)));
  auto& state = *execution->state_;
  TORCH_CHECK(program_->finalized, "Finalize the schedule before running it");
  program_->prepare_run();
  const auto& p = *program_;
  LaneScope lanes;
  state.frame = p.enter(std::move(inputs), std::move(ints), std::move(floats));
  p.begin_run();
  state.pc = p.execute(state.frame, 0, p.code.size(), false);
  if (state.pc >= p.code.size()) {
    state.outputs = p.leave(state.frame);
    p.end_run();
  }
  return execution;
}

int64_t Execution::pending() const {
  const auto& p = *runner_->program_;
  if (state_->pc >= p.code.size()) return -1;
  return p.code[state_->pc].a;
}

namespace {
const CallScheduleSpec& pending_call(const Program& p, size_t pc) {
  TORCH_CHECK(pc < p.code.size(), "The execution has no pending python call");
  return p.schedules[p.code[pc].index];
}
}  // namespace

std::vector<at::Tensor> Execution::arguments() const {
  const auto& p = *runner_->program_;
  const auto& spec = pending_call(p, state_->pc);
  std::vector<at::Tensor> result;
  for (const auto buffer : spec.buffers)
    result.push_back(p.tensor(state_->frame, buffer));
  return result;
}

std::vector<int64_t> Execution::integer_arguments() const {
  const auto& p = *runner_->program_;
  const auto& spec = pending_call(p, state_->pc);
  std::vector<int64_t> result;
  for (const auto id : spec.ints) result.push_back(state_->frame.ints[id]);
  return result;
}

std::vector<double> Execution::float_arguments() const {
  const auto& p = *runner_->program_;
  const auto& spec = pending_call(p, state_->pc);
  std::vector<double> result;
  for (const auto id : spec.floats) result.push_back(state_->frame.floats[id]);
  return result;
}

void Execution::resume(std::vector<at::Tensor> results) {
  const auto& p = *runner_->program_;
  const auto& spec = pending_call(p, state_->pc);
  TORCH_CHECK_VALUE(results.size() == spec.results.size(),
                    "The python call returned ",
                    results.size(),
                    " Tensors for ",
                    spec.results.size(),
                    " results");
  for (size_t i = 0; i < results.size(); ++i)
    state_->frame.buffers[spec.results[i]] = std::move(results[i]);
  LaneScope lanes;
  state_->pc = p.execute(state_->frame, state_->pc + 1, p.code.size(), false);
  if (state_->pc >= p.code.size()) {
    state_->outputs = p.leave(state_->frame);
    p.end_run();
  }
}

std::vector<at::Tensor> Execution::outputs() const {
  const auto& p = *runner_->program_;
  TORCH_CHECK(state_->pc >= p.code.size(),
              "The execution is suspended at a python call");
  return state_->outputs;
}

}  // namespace fhelium::exec
