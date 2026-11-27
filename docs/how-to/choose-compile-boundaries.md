# Choose compile boundaries

A compiled function is the unit that `fc.compile` captures, transforms, links, and executes as one specialization. Where an application places `@fc.compile` determines the size of each Program, the number of specializations, the CUDA Graph segments, the storage shared between calls, and whether a new process can reuse the compiled result without capture. This guide gives the rules that follow from capture and specialization; [Compile a callable](compile-callable.md) lists the options.

## 1. One call captures the whole function body

Calling a compiled function with new input conditions executes its Python body once under capture. Every FHElium operation reached from the body, including operations inside helper functions and library calls such as `fhelium.algorithms.linear.apply`, becomes part of one Program, and the result is one specialization for those conditions. Python control flow runs during capture only: the Program records the operations it produced.

## 2. Nested compiled functions are inlined

A compiled function called from another compiled function is captured as part of the outer computation. Its operations are inlined and compiled with the outer function's options, including `graph`, `cache`, and the pipeline; the inner function's own specializations are not used. The inner function's source enters the cache identity of the outer entry, so editing it invalidates the outer entry as well. Compile a helper separately when it is also called on its own, and nest it when it is part of a larger computation.

## 3. Python loops unroll

A `for` loop over a Python range executes once per iteration during capture, so a loop of $k$ iterations produces $k$ copies of its operations. Program size, pass time, and kernel compilation time grow with the iteration count. Structured operations keep one operation per group regardless of the term count: `Engine.sum_plaintext_products` for a weighted sum of ciphertext-plaintext products, `Engine.sum_rotations` and `Engine.rotate_many_by_steps` for hoisted rotations, and `fhelium.algorithms.linear` for a diagonal linear transform with a baby-step/giant-step plan.

## 4. Scalars specialize by value

Python `bool`, `int`, `float`, `complex`, `str`, and `None` arguments are part of the call signature by value. Each distinct value prepares and stores another specialization; a gain that takes two values produces two specializations. Pass a value that changes per request as a Tensor or a plaintext, whose contents may change without a new specialization, and keep Python scalars for values that select the computation's structure, such as a polynomial degree.

Tensor and CKKS value arguments are specialized by their conditions: shape, dtype, device, and, for CKKS values, depth, scale, polynomial domain, modulus basis, and component count. A ciphertext at another depth prepares another specialization.

## 5. Methods compile per instance

Decorating a method compiles it for each instance: accessing the method on an instance returns a compiled callable bound to that instance, with its own specializations, stored in the instance `__dict__`. Attributes the method reads, such as weights, prepared plaintexts, and keys, become materials of that instance's specializations. Attributes assigned inside a compiled method raise `CaptureError` ("Captured computation cannot mutate bound object state"); set them before the call.

The cache identity of a method contains its function and the attributes it reads: Engines by configuration, Tensors and CKKS values by layout, immutable values by value. Instances whose attributes have equal layouts share one cache entry, and each binds its own materials. A method that passes `self` as a value, reads a property or another descriptor, or reads an object without a supported identity is found by its captured Program instead.

## 6. Graph segments follow the schedule

Compile divides each linked schedule into CUDA Graph segments at instructions that cannot be captured, such as Python calls and CPU operations, and with `graph="auto"` keeps each segment's Graph only when its replay is faster than its direct execution. The boundary of a compiled function is also a segment boundary: one larger function can form fewer, longer segments, while a chain of smaller functions replays one Graph per function and segment.

## 7. Weigh function size

| Larger compiled functions | Smaller compiled functions |
| --- | --- |
| More fusion opportunities across operations | Shorter capture and pass time |
| Fewer Graph launches per request | More reuse of one specialization in several places |
| Longer first call and more specializations when any input condition varies | One more call and result per boundary |

Compiled functions on one device share one arena for their temporaries, so splitting a computation does not multiply temporary memory. The output of one compiled function passed to the next is read where it lies, without a copy. Each function keeps its own output storage for the results it returns.

## 8. Keep runtime decisions outside

A compiled function records arithmetic; decisions that depend on runtime state belong in the calling code. Capture rejects a branch on encrypted or Tensor data with `CaptureError: Captured computation cannot branch on Tensor or ciphertext data`, and a call to a Residency manager, scope, controller, or lease with `CaptureError: ResidencyManager.<method> cannot run inside a compiled computation; call it between compiled calls and pass the resulting values as arguments`. Request routing, admission, I/O, key selection per request, and placement decisions run between compiled calls, and their results enter as arguments.

## 9. Write cache-friendly functions

A new process finds a cached entry without capture when the function source lies in the statically checked subset: assignments to local names, `if`, `for` over supported iterables, `return`, expressions, calls of FHElium Engine operations, Tensor methods, other Python functions in the subset, and compiled functions, with global and closure values read through their attributes. Module attributes from `os`, `sys`, `time`, `random`, `numpy`, and `secrets`, assignments to attributes or subscripts, dynamic call targets, `**kwargs` expansion, and other statements such as `import`, `with`, and `try` fall outside the subset; the subset applies to every function the source calls. A function outside it is captured in each process and found by its captured Program, which skips the pass pipeline and linking.

`compiled.stats()["cache"]` reports the reason when the source falls outside the subset, for example `{'ineligible': 'Unsupported Python cache syntax: ImportFrom'}`.

## Verify the outcome

After representative calls, `len(compiled.specializations)` gives the number of specializations, `stats()["graphs"]` the segment choices, and `stats()["cache"]` the cache key in use. [Example 12](../tutorial/compile-layer.md) compiles a layer class with nested methods and a scalar gain, and [Measure and optimize a workload](measure-and-optimize.md) separates first-call and steady-state latency.
