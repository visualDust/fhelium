# Values, operation semantics, and Eager dispatch

Eager evaluates individual CKKS operations over public values. An `Engine` owns a CKKS configuration and lazily prepared device services; its methods calculate result metadata and dispatch Tensor payloads without constructing a Program. This page traces the value-state and direct-call mechanism. The [architecture overview](engine-native-stack.md) places it alongside manual and callable Compile execution.

## What gives a Tensor its CKKS meaning?

`fhelium.values` defines `Ciphertext`, `Plaintext`, `CompressedPlaintext`, and key values. Their Tensor storage is interpreted through polynomial domain, residue representation, ordered `prime_ids`, and modulus basis. Ciphertexts and plaintexts additionally carry depth and actual scale. Ciphertext component count describes the polynomial in the secret key; it is independent of the batch axes.

For a ciphertext with components $c_i$, decryption evaluates $\sum_i c_i s^i$ modulo the active basis. Scale $\Delta$ relates that encoded polynomial to the approximate message. Tensor shape alone cannot establish the scale or the meaning of a prime row. Equal row counts do not establish equal ordered primes.

`values/state.py` and the value classes describe public state. `ir/ckks_state.py` implements operation-specific metadata equations shared with symbolic Eager capture. `CkksConfig` supplies mathematical parameters alongside that per-value state.

## Which transitions belong to an operation?

For compatible operands at one depth, addition preserves scale and active rows. Multiplication multiplies scales. Multiplying two two-component ciphertexts produces

$$
(c_0,c_1,c_2)=(a_0b_0,\ a_0b_1+a_1b_0,\ a_1b_1),\qquad \Delta_c=\Delta_a\Delta_b.
$$

The polynomial product is implemented in NTT/Montgomery form. Relinearization reduces three components to two using a supplied evaluation key; its output keeps the represented depth and scale. The [arithmetic article](multiplication-keyswitch-rescale.md) gives the key-switch decomposition and rounding equations.

If the next depth group contains primes $G_d$, rescale uses their complete product $D_d=\prod_{q\in G_d}q$. It advances depth from $d$ to $d+1$, removes those rows, and changes scale to $\Delta/D_d$. Domain-specific implementations preserve the requested coefficient/standard or NTT/Montgomery result representation. A depth transition can remove several primes.

Modulus restriction selects a smaller active basis without performing the rescale quotient. Reinterpreting scale changes the represented message interpretation without changing residue bytes. These are distinct operations with distinct state transitions.

## How does an Engine call reach arithmetic?

```text
Engine method in eager/_engine.py
  → public input adaptation and CKKS metadata equations
  → _EagerOperationDispatcher in eager/_operation_dispatch.py
  → OperationInvocation and registry-selected implementation
  → Tensor arithmetic with numerical operands and bound handles
  → public value construction or in-place replacement
```

`OperationInvocation` carries an operation class, operand/result counts, and attributes needed by the implementation. It has no SSA region. `_EagerOperationDispatcher` caches invocations and resolved direct calls using operation metadata. Its `_prepare_call` resolves non-Tensor requirements from device services; numerical parameters, transform tables, and key payloads are passed as Tensor operands.

The chosen operation class identifies the computation being executed. A whole CKKS implementation can compose native RNS/NTT arithmetic internally. Lower-level operations can also be dispatched directly. Registry lookup selects an implementation of the requested operation.

An in-place Engine method computes the new state, performs arithmetic, then updates the public value. Metadata is committed after successful execution. Native mutation and storage aliasing follow the called operator's contract.

## Who selects placement and key storage?

Factories follow PyTorch defaults unless a device is supplied. An Engine has no default-device ownership. Operations on materialized values follow operand placement and select the corresponding lazy device services. Public conversion methods with a device argument may perform their documented transfer; ordinary arithmetic requires compatible operand placement.

`eager/_key_inventory.py` retains supplied evaluation keys and resolves the requested key kind or rotation step. Cross-device key replication is opt-in. A direct dispatch cache retains prepared implementation and handle objects.

## How does capture reuse these semantics?

`compile/frontend/_eager_capture.py` substitutes symbolic adapters for supported Engine calls. `compile/frontend/_eager_values.py` constructs typed SSA values from the same metadata transitions while preserving each input's independent state. Numerical kernels are not run for those symbolic CKKS calls. The result is a Compilation that can be inspected or transformed manually, or prepared through a compiled callable.

This shared mathematical code keeps immediate execution and capture aligned. Graph scheduling remains a Compile responsibility: Eager executes the rescale, relinearization, and representation transitions requested by the caller. See [IR and capture](compiler-stack-internals.md) for capture scope and [Compilation and passes](compilation-and-passes.md) for graph transformations.

## Extending a public operation

Define the input/output equations and storage behavior in the public method and dialect operation. Reuse an existing metadata transition when the mathematics matches, or add the corresponding operation-specific rule in `ir/ckks_state.py`. Connect Tensor execution through the owning Backend family. For capture support, implement the symbolic adaptation of the same public arguments and result state; do not run the Eager kernel to infer those results.

Check the public result state and numerical payload together, including functional versus in-place behavior.
