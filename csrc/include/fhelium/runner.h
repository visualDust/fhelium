#pragma once

// Execution schedules over typed storage buffers.
//
// A Runner holds one linked exec schedule. A caller builds it through the
// methods below, calls finalize() once, and then runs it any number of times.
// Builder methods append operations to the current block; begin_for/end_for
// and begin_if/begin_else/end_if open and close nested blocks. finalize()
// translates the blocks into one instruction list with jumps, resolves every
// dispatcher operator, and fixes the constant part of each operator's argument
// stack, so that a run only fills buffer Tensors and host registers.
//
// Values live in three index spaces allocated by the Runner: Tensor buffers,
// int64 registers and double registers. `inputs` and `outputs` fix the
// argument and result order of `run`. Material and state buffers are bound once
// with `bind` and stay bound across runs; every other buffer is released when a
// run returns. `exec.python_call` operations suspend execution: `start`
// returns an Execution that reports the pending call, receives its results with
// `resume`, and continues. `run` rejects schedules that contain python calls.
//
// Arena buffers placed with `place` are views of one region per device that the
// Runner allocates at finalize and keeps, or of a shared Arena attached with
// `share_arena`; they behave like bound buffers, and the caller assigns byte
// offsets so that buffers with overlapping ranges have disjoint lifetimes.
// Output buffers placed with `place_output` are views of a per-Runner output
// region in the same way. Between `begin_segment` and `end_segment`, a Runner
// with graphs enabled captures the instructions as one device graph: the
// first run that reaches a segment executes it directly, the next one
// captures and replays it, and later runs replay it. In mode `auto` a segment
// decides once whether to keep its graph: the device time of its first direct
// execution, measured with events on its stream and so including idle gaps
// from host launch latency, is compared with the median of the next three
// replays after the capturing one; the events are read at the first run after
// they have completed. A segment whose median replay is not faster releases
// its graphs and executes directly from then on; the choice outlasts later
// captures of the segment. After a replay, buffers and
// registers written in the segment hold their capture-time values; written
// buffers that no later instruction reads and that are not outputs are
// released. The graphs of the Runners attached to one Arena, or of one Runner
// without an Arena, share one memory pool per device, and a graph's
// temporaries may reuse memory that another graph of the pool uses during its
// replay. Written buffers that are read after the segment therefore hold
// Runner-owned storage: a value that the graph allocated in the pool is copied
// to it after each replay, one copy per pool storage. Placing such buffers
// (`place`, `place_output`) avoids the copy. A segment
// reads Runner-owned Tensors (arena, bound, output or captured storage) and
// output ranges returned by Runners and inputs marked with `read_in_place` in
// place, and other device Tensors through staging storage that each replay
// refreshes, copying them back when the segment may write them. A segment
// keeps up to four captured variants, one per combination of live-buffer
// addresses read in place; a combination captures on its second run, and the
// least recently replayed variant leaves first. In mode `auto`, a segment
// that would have to evict a variant runs directly from then on. A changed
// live-in register value or layout drops every graph of the Runner; the
// segments then run directly for the rest of that run and capture again on the
// next one.
//
// Outputs that refer to Runner-owned storage (arena, output region or
// captured storage) are returned without a copy, each storage range through
// its own StorageImpl. Before a later run can overwrite that range (the next
// run of this Runner or, for a shared Arena, of any Runner attached to it),
// a returned StorageImpl that the caller still references is moved to newly
// allocated memory holding a copy of its contents; views follow because they
// share the StorageImpl. Returned outputs that are no longer referenced cost
// nothing. A Runner is not reentrant: one Runner does not execute on two
// threads at once, and runs of Runners that share an Arena do not interleave.
//
// Process groups, streams and tasks: `group` declares a c10d process group
// binding, `bind_group` supplies it, `group_query` reads the rank or size at
// run time and `collective` runs one c10d collective on its buffers. Device
// work issues on the current stream of its device; `use_stream` switches it
// between declared streams (`stream`, where role `compute` is the stream
// current when the run begins, or the capture stream inside a capture), and
// `record`/`wait` order streams through declared events. Between
// `begin_task` and `end_task`, instructions form a task that runs on the
// process's CPU lane, one worker thread shared by all Runners that runs tasks
// in issue order, after its wait events complete, while the host thread
// continues with the instructions after the task; `join` waits for it. Inside a
// segment, work on a stream other than `compute` follows a `wait` on an event
// recorded on `compute` and is joined back before the segment ends; collectives
// over a group without an NCCL CUDA backend run their segment directly.
//
// Torch registers Arena, Runner and Execution as torch.classes.fhelium.Arena,
// torch.classes.fhelium.Runner and torch.classes.fhelium.Execution.

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fhelium/torch_library.h"

namespace c10d {
class ProcessGroup;
}  // namespace c10d

namespace fhelium::exec {

// Argument kinds of call_op and launch.
enum class ArgumentKind : int64_t {
  constant = 0,        // the value itself
  buffer = 1,          // a Tensor buffer id
  buffer_list = 2,     // a list of Tensor buffer ids; -1 is None
  int_register = 3,    // an int64 register id
  float_register = 4,  // a double register id
  // An empty list of the schema argument's list type; the value is ignored.
  // (Python cannot convert an empty list to an element-typed IValue.)
  empty_list = 5,
};

// A compiled device kernel: one image for one target architecture and the
// launch facts recorded by its compiler. `parameter_types` lists the runtime
// parameters in order, without compile-time constants, as Triton type strings
// such as "*i32", "i64" or "fp32". Launches append the global and profile
// scratch pointers after the runtime parameters; a scratch pointer is null
// when its size is zero.
struct KernelRecord {
  std::string name;
  std::vector<uint8_t> image;
  std::string image_kind;  // cubin, ptx or hsaco
  int64_t num_warps = 4;
  int64_t warp_size = 32;
  int64_t num_ctas = 1;
  int64_t shared = 0;
  std::vector<std::string> parameter_types;
  int64_t global_scratch_size = 0;
  int64_t global_scratch_align = 1;
  int64_t profile_scratch_size = 0;
  int64_t profile_scratch_align = 1;
  std::string triton_version;
};

class Execution;
struct Program;

// A grow-only allocation for the arena buffers of the Runners attached to it,
// on CUDA device `device` or on the CPU for -1. Runners executed in sequence on
// one device share it to bound their temporary memory by the largest
// requirement. Growth replaces the region; each attached Runner moves its arena
// views to the new region and captures its segments again on its next run.
class FHELIUM_TORCH_API Arena : public torch::CustomClassHolder {
 public:
  explicit Arena(int64_t device);
  ~Arena() override;
  // Grows the region to at least `bytes`.
  void reserve(int64_t bytes);
  int64_t bytes() const;
  int64_t device() const;
  // Bytes that the graph pools of the attached Runners hold in the caching
  // allocator, 0 without captured graphs.
  int64_t graph_pool_bytes() const;

 private:
  friend struct Program;
  struct State;
  std::unique_ptr<State> state_;
};

class FHELIUM_TORCH_API Runner : public torch::CustomClassHolder {
 public:
  Runner();
  ~Runner() override;

  // Buffers and registers ------------------------------------------------

  // A Tensor buffer. `storage` is input, output, arena, material, state,
  // internal, pinned (page-locked host memory) or view; alloc requires sizes,
  // strides, dtype and device.
  int64_t buffer(const std::string& storage,
                 std::optional<std::vector<int64_t>> sizes,
                 std::optional<std::vector<int64_t>> strides,
                 std::optional<c10::ScalarType> dtype,
                 std::optional<c10::Device> device);
  int64_t int_register();
  int64_t float_register();
  void inputs(std::vector<int64_t> buffers,
              std::vector<int64_t> int_registers,
              std::vector<int64_t> float_registers);
  void outputs(std::vector<int64_t> buffers);
  // Binds a material or state buffer. A bound buffer may be rebound between
  // runs; rebinding to a different pointer or layout drops the captured graphs.
  void bind(int64_t buffer, const at::Tensor& value);
  // Places arena buffer `buffer`, which has a static layout, at byte `offset`
  // of its device's arena; `offset` is a multiple of the element size.
  void place(int64_t buffer, int64_t offset);
  // Bytes of arena storage the schedule requires, set by finalize.
  int64_t arena_bytes() const;
  // Marks input buffers whose contents stay at their address across runs,
  // such as keys and plaintexts. Segments read them, and views of them, in
  // place, so their addresses join the variant key instead of being staged.
  void read_in_place(std::vector<int64_t> buffers);
  // Places output buffer `buffer`, which has a static layout, at byte `offset`
  // of its device's output region; `offset` is a multiple of the element size.
  void place_output(int64_t buffer, int64_t offset);
  // Total bytes of the output regions allocated by finalize.
  int64_t output_bytes() const;
  // Places the arena buffers in `arena` instead of a Runner-owned region; all
  // placed arena buffers are on the arena's device. Call before finalize, which
  // reserves arena_bytes() in it.
  void share_arena(c10::intrusive_ptr<Arena> arena);

  // Operations -----------------------------------------------------------

  // Calls the dispatcher operator `schema` ("namespace::name.overload",
  // "default" for the unnamed overload) with one (kind, value) pair per schema
  // argument. `results` receives the returned Tensors, a returned Tensor list
  // unpacked; returns that alias a written argument and returns of other types
  // are not listed.
  void call_op(const std::string& schema,
               std::vector<int64_t> kinds,
               std::vector<c10::IValue> values,
               std::vector<int64_t> results);
  int64_t kernel(KernelRecord record);
  // Launches `kernel` on CUDA or HIP device `device` on its current stream.
  // Buffer arguments pass the buffer's data pointer.
  void launch(int64_t kernel,
              int64_t device,
              std::vector<int64_t> grid,
              std::vector<int64_t> kinds,
              std::vector<c10::IValue> values);
  // target = source.as_strided(sizes, strides, source offset + offset).
  void view(int64_t target,
            int64_t source,
            std::vector<int64_t> sizes,
            std::vector<int64_t> strides,
            int64_t offset);
  void alloc(int64_t target);
  // target.copy_(source, non_blocking).
  void copy(int64_t target, int64_t source, bool non_blocking = false);
  void zero(int64_t target);
  // target refers to the Tensor of source.
  void assign(int64_t target, int64_t source);
  // Drops the buffer's Tensor reference after its last use; storage returns
  // to the allocator when no other buffer or view refers to it.
  void release(int64_t buffer);
  // Runs the finalized schedule `callee` with arguments in its input order.
  void call(c10::intrusive_ptr<Runner> callee,
            std::vector<int64_t> buffers,
            std::vector<int64_t> int_registers,
            std::vector<int64_t> float_registers,
            std::vector<int64_t> results);
  // Suspends execution until the caller supplies the results of Python
  // callable `callee`.
  void python_call(int64_t callee,
                   std::vector<int64_t> buffers,
                   std::vector<int64_t> int_registers,
                   std::vector<int64_t> float_registers,
                   std::vector<int64_t> results);

  // Host scalars and control flow ----------------------------------------

  void int_constant(int64_t target, int64_t value);
  void float_constant(int64_t target, double value);
  // op is add, sub, mul, div, rem, min, max, eq, ne, lt, le, gt or ge on
  // signed values; div and rem round toward negative infinity.
  void int_binary(const std::string& op, int64_t target, int64_t a, int64_t b);
  void select(int64_t target, int64_t condition, int64_t a, int64_t b);
  void assign_register(int64_t target, int64_t source);
  // for induction in range(lower, upper, step): body until end_for.
  void begin_for(int64_t induction, int64_t lower, int64_t upper, int64_t step);
  void end_for();
  void begin_if(int64_t condition);
  void begin_else();
  void end_if();

  // Graph segments -------------------------------------------------------

  // Segment capture mode: `off` ignores segment markers, `on` captures every
  // segment and `auto` keeps a segment's graph only when its replays are
  // faster than its direct execution.
  void graphs(const std::string& mode);
  // Opens a segment on CUDA device `device`, or on the current device for -1.
  // A segment closes in the block that opened it, does not nest and contains
  // no python calls.
  void begin_segment(int64_t device);
  void end_segment();
  // [segments, captures, replays, staged live-in copies, detached outputs,
  //  captured graphs held, bytes held by the graph pools the Runner uses,
  //  copies of live values out of pool memory]. Runners attached to one
  //  Arena report the same shared pool bytes.
  std::vector<int64_t> graph_stats() const;
  // Records device time around direct executions and replays of segments;
  // call before finalize.
  void profile_segments(bool enable);
  // One row per segment: [graph nodes, capture seconds and instantiate
  // seconds of the latest capture, device seconds of the latest direct
  // execution and replay, variants held, choice (0 measuring, 1 graph,
  // 2 direct), device seconds of the decision's direct execution and median
  // replay]. Times not recorded are NaN.
  std::vector<std::vector<double>> segment_stats() const;

  // Process groups, streams and tasks -------------------------------------

  // A process group binding of `size` ranks; -1 accepts any size.
  int64_t group(int64_t size = -1);
  // Binds `group` to the c10d group registered as `name`, or to `value`.
  // Every group is bound before finalize; rebinding to another group drops
  // the captured graphs.
  void bind_group(int64_t group, const std::string& name);
  void bind_group(int64_t group, c10::intrusive_ptr<c10d::ProcessGroup> value);
  // Drops the captured graphs and unbinds every group, here and in called
  // schedules, so that the process groups can be destroyed; a run needs
  // `bind_group` again first.
  void release_groups();
  // Writes the group's rank (`kind` rank) or size (`kind` size) of this
  // process to int register `target`.
  void group_query(const std::string& kind, int64_t target, int64_t group);
  // Runs one collective over `group` and waits for it; an NCCL collective
  // makes the current stream wait without blocking the host. `kind`:
  //   all_gather  inputs {x}, outputs {y}: y[r] = x of rank r; y is contiguous
  //               with shape [size, *x.shape].
  //   broadcast   inputs {x}, outputs {y}, options {root}: y = x of rank
  //               root; y is contiguous.
  //   reduce_scatter  inputs {x}, outputs {y}: x and y are contiguous, x has
  //               size * y.numel() elements, and y = sum over ranks r of
  //               chunk `rank` of x of rank r; integer sums wrap in two's
  //               complement.
  //   gather      inputs {x}, outputs {y}, options {root}: on rank root,
  //               y[r] = x of rank r with y contiguous of size * x.numel()
  //               elements; other ranks leave y unwritten.
  // Outputs are written in place.
  void collective(const std::string& kind,
                  int64_t group,
                  std::vector<int64_t> inputs,
                  std::vector<int64_t> outputs,
                  std::vector<int64_t> options);
  // A stream of GPU device `device`: role `compute`, or one stream from the
  // device's pool per other role.
  int64_t stream(int64_t device, const std::string& role);
  int64_t event();
  // Device work after this instruction issues on `stream`; a run and each
  // task begin on `compute` and the run restores the caller's stream.
  void use_stream(int64_t stream);
  void record(int64_t event, int64_t stream);
  // `stream` waits for the work recorded by `event`.
  void wait(int64_t event, int64_t stream);
  // The host thread waits for the work recorded by `event`.
  void host_wait(int64_t event);
  // Tells the caching allocator that `buffer`'s storage is used on `stream`.
  void record_stream(int64_t buffer, int64_t stream);
  int64_t task();
  // Opens the body of `task`, which waits on the host for `wait_events` and
  // runs on the process's CPU lane. A body contains no segments, tasks, joins,
  // calls of other schedules, python calls, control flow or register writes;
  // instructions after the task must not use the buffers the body writes
  // before the `join`.
  void begin_task(int64_t task, std::vector<int64_t> wait_events);
  void end_task();
  // Waits for the task's last start and rethrows its error.
  void join(int64_t task);
  // [tasks run, task seconds, host seconds waiting in join, event waits,
  //  collectives, host seconds waiting for non-NCCL collectives].
  std::vector<double> lane_stats() const;

  // Execution ------------------------------------------------------------

  void finalize();
  std::vector<at::Tensor> run(std::vector<at::Tensor> inputs,
                              std::vector<int64_t> ints,
                              std::vector<double> floats);
  c10::intrusive_ptr<Execution> start(std::vector<at::Tensor> inputs,
                                      std::vector<int64_t> ints,
                                      std::vector<double> floats);
  bool has_python_calls() const;
  std::string describe() const;

 private:
  friend class Execution;
  friend struct Program;
  std::unique_ptr<Program> program_;
};

// One run of a schedule that may contain python calls.
class FHELIUM_TORCH_API Execution : public torch::CustomClassHolder {
 public:
  ~Execution() override;
  // The callee of the pending python call, or -1 after the schedule returned.
  int64_t pending() const;
  std::vector<at::Tensor> arguments() const;
  std::vector<int64_t> integer_arguments() const;
  std::vector<double> float_arguments() const;
  void resume(std::vector<at::Tensor> results);
  std::vector<at::Tensor> outputs() const;

 private:
  friend class Runner;
  struct State;
  explicit Execution(c10::intrusive_ptr<Runner> runner);
  c10::intrusive_ptr<Runner> runner_;
  std::unique_ptr<State> state_;
};

}  // namespace fhelium::exec
