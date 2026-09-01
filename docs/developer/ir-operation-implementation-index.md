# IR operator implementation index

This index maps the operations used by FHElium's intermediate representation (IR) to the files that directly transform or execute them. An **IR abstraction level** describes how much cryptographic and execution detail an operation represents; it is independent of a CKKS modulus-chain level.

The main inventory contains all 84 operations in `REGISTERED_DIALECTS` and groups them by mathematical or execution responsibility. Each group proceeds from higher-level intent toward lower-level arithmetic where applicable. A separate final section lists the 18 upstream xDSL operations that FHElium's current Backend executes directly.

[Operation declaration and implementation selection](./operation-registration-and-selection.md) describes how Compile records transformations and how execution owners assemble Backend implementations.

## Reading the tables

**Operator** is the operation's textual IR name. **Schema** gives its operands, behavior-relevant attributes, regions, and results. **File** names the direct transformation or execution owner using only its basename. A high-level lowering does not repeat every transitive lower-level kernel; those kernels appear on the rows for the resulting RNS or NTT operations.

The compact schemas use these symbols:

- `T` and `U` are arbitrary value types and may differ. Repeated `T` positions have the same type unless stated otherwise. `T...` is a possibly heterogeneous type sequence; corresponding `T...` positions match element by element.
- `X` is either `CT` or `PT`, with the same concrete value kind at the corresponding operand and result positions.
- `P_message`, `P_plaintext`, and `P_static` are public source values whose represented roles are message, plaintext, and static.
- `S` is `!fhelium_semantic.secret` or `!fhelium_semantic.public`; `LE` and `LP` are `!fhelium_logical.encrypted` and `!fhelium_logical.public`.
- `M`, `CT`, `PT`, and `CPT` are `!fhelium.message`, `!fhelium_ckks.ciphertext`, `!fhelium_ckks.plaintext`, and `!fhelium_ckks.compressed_plaintext`.
- `R`, `RP`, `RS`, `KP`, and `EK` are `!fhelium_rns.bundle`, `!fhelium_rns.parameters`, `!fhelium_rns.rescale_plan`, `!fhelium_rns.key_switch_plan`, and `!fhelium_rns.evaluation_key_resource`.
- `NP`, `D`, and `G` are `!fhelium_ntt.plan`, `!fhelium_memory.device`, and `!fhelium_dist.group`.
- `I` is an index or integer scalar type, `J` is another index or integer scalar type, and `F` is a floating scalar type.
- `?` marks an optional operand or attribute. A region schema inside braces gives its block arguments and yielded values.

## Program structure, binding, and interoperability

These operations introduce literals and graph-external bindings or preserve a PyTorch call for a selected consumer.

| Operator | Schema | File |
|---|---|---|
| `fhelium.constant` | `() {fhelium.literal: JSON} -> T` | `_interpreter_runtime.py` |
| `fhelium.material.ref` | `() {symbol: str, kind?: str} -> T` | `_interpreter_runtime.py` |
| `fhelium.resource.ref` | `() {symbol: str, kind?: str} -> T` | `execution.py`<br>`_interpreter_runtime.py` |
| `torch.call` | `(T...) {fhelium.call.kind: str, fhelium.call.target: str, fhelium.call.arguments: JSON, fhelium.role?: str} -> T` | `_interpreter_runtime.py` |

## Frontend intent and operand roles

Semantic operations state source-level arithmetic. Logical operations make encrypted and public operand roles visible before a CKKS representation is selected.

| Operator | Schema | File |
|---|---|---|
| `fhelium_semantic.add` | `(S, S) -> S` | `_lower_semantic_to_logical.py`<br>`_triton.py` |
| `fhelium_semantic.subtract` | `(S, S) -> S` | `_lower_semantic_to_logical.py` |
| `fhelium_semantic.multiply` | `(S, S) -> S` | `_lower_semantic_to_logical.py`<br>`_triton.py` |
| `fhelium_semantic.negate` | `(S) -> S` | `_lower_semantic_to_logical.py`<br>`_triton.py` |
| `fhelium_semantic.roll` | `(S) {shift: i64, dimension?: i64} -> S` | `_lower_semantic_to_logical.py` |
| `fhelium_logical.add.encrypted_encrypted` | `(LE, LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.add.encrypted_public` | `(LE, LP) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.add.public_encrypted` | `(LP, LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.subtract.encrypted_encrypted` | `(LE, LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.subtract.encrypted_public` | `(LE, LP) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.subtract.public_encrypted` | `(LP, LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.multiply.encrypted_encrypted` | `(LE, LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.multiply.encrypted_public` | `(LE, LP) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.multiply.public_encrypted` | `(LP, LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.negate.encrypted` | `(LE) -> LE` | `_lower_logical_to_ckks.py` |
| `fhelium_logical.roll.encrypted` | `(LE) {shift: i64} -> LE` | `_resolve_rotation_keys.py` |

## Boundary conversion and cryptography

Boundary operations adapt messages, plaintexts, ciphertexts, and key resources to Tensor numerical work. They are normally invoked outside an encrypted computation, but remain representable for caller-selected protocols and passes.

| Operator | Schema | File |
|---|---|---|
| `fhelium_ckks.encode` | `(M) {level: i64, scale: f64} -> PT` | `_implementation.py`<br>`_codec.py` |
| `fhelium_ckks.decode` | `(PT) {is_real: i64} -> M` | `_implementation.py`<br>`_codec.py` |
| `fhelium_ckks.integer_coefficients_to_rns` | `(PT) {modulus_basis: str, level: i64} -> PT` | `_implementation.py` |
| `fhelium_ckks.encrypt` | `(PT) {key_symbol: str} -> CT` | `_encryption.py` |
| `fhelium_ckks.decrypt` | `(CT) {key_symbol: str} -> PT` | `_decryption.py` |

## Ciphertext arithmetic and component composition

This group connects CKKS ciphertext arithmetic to RNS component operations. CKKS multiplication has both a lowering and a registered whole-operation implementation.

| Operator | Schema | File |
|---|---|---|
| `fhelium_ckks.add` | `(CT, CT) -> CT` | `_arithmetic.py` |
| `fhelium_ckks.subtract` | `(CT, CT) -> CT` | `_arithmetic.py` |
| `fhelium_ckks.negate` | `(CT) -> CT` | `_arithmetic.py` |
| `fhelium_ckks.multiply` | `(CT, CT) -> CT` | `_arithmetic.py`<br>`operations.py`<br>`multiply_cpu.cpp`<br>`multiply_cuda.cu` |
| `fhelium_ckks.add_scalar` | `(CT) {scalar: f64, scalar_scale: f64} -> CT` | `scalar.py` |
| `fhelium_ckks.multiply_scalar` | `(CT) {scalar: f64, scalar_scale: f64} -> CT` | `scalar.py` |
| `fhelium_ckks.multiply_integer_scalar` | `(CT) {scalar: I} -> CT` | `scalar.py` |
| `fhelium_rns.add_standard` | `(R, R, RP) -> R` | `operations.py`<br>`rns_standard_arithmetic_cpu.cpp`<br>`rns_standard_arithmetic_cuda.cu` |
| `fhelium_rns.subtract_standard` | `(R, R, RP) -> R` | `operations.py`<br>`rns_standard_arithmetic_cpu.cpp`<br>`rns_standard_arithmetic_cuda.cu` |
| `fhelium_rns.negate_standard` | `(R, RP) -> R` | `operations.py` |
| `fhelium_rns.montgomery_multiply` | `(R, R, RP) -> R` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`rns_arithmetic_cuda.cu` |
| `fhelium_rns.extract_component` | `(R) {component: i64} -> R` | `operations.py` |
| `fhelium_rns.pack_two_components` | `(R, R) -> R` | `operations.py` |
| `fhelium_rns.pack_three_components` | `(R, R, R) -> R` | `operations.py` |

## Public and plaintext preparation and mixed arithmetic

Preparation operations convert a public source into an operation-ready plaintext for one ciphertext. The operation name identifies whether the source is a message, existing plaintext, or specialized static value.

| Operator | Schema | File |
|---|---|---|
| `fhelium_ckks.prepare.add.message` | `(P_message, CT) {operation: "add", source_role: "message", scale_mode: "ciphertext_scale"} -> PT` | `_lower_message_preparation.py`<br>`_engine.py` |
| `fhelium_ckks.prepare.add.plaintext` | `(P_plaintext, CT) {operation: "add", source_role: "plaintext", scale_mode: "ciphertext_scale"} -> PT` | `_engine.py`<br>`_interpreter_runtime.py` |
| `fhelium_ckks.prepare.add.static` | `(P_static, CT) {operation: "add", source_role: "static", scale_mode: "ciphertext_scale"} -> PT` | `_engine.py`<br>`_interpreter_runtime.py` |
| `fhelium_ckks.prepare.multiply.message` | `(P_message, CT) {operation: "multiply", source_role: "message", scale_mode: "default_scale"} -> PT` | `_lower_message_preparation.py`<br>`_engine.py` |
| `fhelium_ckks.prepare.multiply.plaintext` | `(P_plaintext, CT) {operation: "multiply", source_role: "plaintext", scale_mode: "runtime_plaintext_scale"} -> PT` | `_engine.py`<br>`_interpreter_runtime.py` |
| `fhelium_ckks.prepare.multiply.static` | `(P_static, CT) {operation: "multiply", source_role: "static", scale_mode: "default_scale"} -> PT` | `_engine.py`<br>`_interpreter_runtime.py` |
| `fhelium_ckks.add_plaintext` | `(CT, PT) -> CT` | `_arithmetic.py` |
| `fhelium_ckks.multiply_plaintext` | `(CT, PT) -> CT` | `_arithmetic.py` |
| `fhelium_ckks.add_compressed_plaintext` | `(CT, CPT) {inplace?: i64} -> CT` | `operations.py`<br>`plaintext_cpu.cpp`<br>`plaintext_cuda.cu` |
| `fhelium_ckks.multiply_compressed_plaintext` | `(CT, CPT) {inplace?: i64} -> CT` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`rns_arithmetic_cuda.cu` |
| `fhelium_rns.add_plaintext` | `(R, R, RP) -> R` | `operations.py`<br>`plaintext_cpu.cpp`<br>`plaintext_cuda.cu` |
| `fhelium_rns.multiply_plaintext` | `(R, R, RP) -> R` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`rns_arithmetic_cuda.cu` |

## Polynomial and residue representation transforms

These operations change polynomial domain or residue representation while preserving the represented CKKS value. `X` below is a ciphertext or plaintext and the result has the same value kind.

| Operator | Schema | File |
|---|---|---|
| `fhelium_ckks.to_ntt` | `(X) -> X` | `_representation.py` |
| `fhelium_ckks.from_ntt` | `(X) -> X` | `_representation.py` |
| `fhelium_ckks.to_montgomery_residues` | `(X) -> X` | `_representation.py` |
| `fhelium_ckks.to_standard_residues` | `(X) -> X` | `_representation.py` |
| `fhelium_rns.standard_to_montgomery` | `(R, RP) -> R` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`rns_arithmetic_cuda.cu` |
| `fhelium_rns.montgomery_to_standard` | `(R, RP) -> R` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`rns_arithmetic_cuda.cu` |
| `fhelium_ntt.coefficient_standard_to_ntt_montgomery` | `(R, NP) -> R` | `operations.py`<br>`indexed_radix2.py`<br>`compact_radix2.py`<br>`power_of_two_radix.py`<br>`indexed_ntt_cpu.cpp`<br>`forward_ntt_cuda.cu` |
| `fhelium_ntt.coefficient_montgomery_to_ntt_montgomery` | `(R, NP) -> R` | `operations.py`<br>`indexed_radix2.py`<br>`compact_radix2.py`<br>`power_of_two_radix.py`<br>`indexed_ntt_cpu.cpp`<br>`forward_ntt_cuda.cu` |
| `fhelium_ntt.ntt_montgomery_to_coefficient_standard` | `(R, NP) -> R` | `operations.py`<br>`indexed_radix2.py`<br>`compact_radix2.py`<br>`power_of_two_radix.py`<br>`indexed_ntt_cpu.cpp`<br>`inverse_ntt_cuda.cu` |
| `fhelium_ntt.inverse_montgomery` | `(R, NP) -> R` | `operations.py`<br>`indexed_radix2.py`<br>`compact_radix2.py`<br>`power_of_two_radix.py`<br>`indexed_ntt_cpu.cpp`<br>`inverse_ntt_cuda.cu` |

## Modulus-chain and scale transitions

These operations change the active modulus basis, level metadata, or scale metadata. CKKS rescale accepts one ciphertext operand and carries `rounding: "nearest"` or `"floor"`.

| Operator | Schema | File |
|---|---|---|
| `fhelium_ckks.rescale` | `(CT, PT?) {condition: str, rounding: str} -> CT` | `_representation.py` |
| `fhelium_ckks.mod_switch` | `(CT) {target_level: i64} -> CT` | `_representation.py` |
| `fhelium_ckks.reinterpret_scale` | `(CT) {scale: f64} -> CT` | `_representation.py` |
| `fhelium_rns.rescale_drop_leading_prime` | `(R, RS) {rounding?: str} -> R` | `rescale.py`<br>`rescale_cpu.cpp`<br>`rescale_cuda.cu` |
| `fhelium_rns.restrict_level` | `(R, RP) {target_level: i64} -> R` | `operations.py` |
| `fhelium_rns.reinterpret_scale` | `(R) {scale: f64} -> R` | `operations.py` |

## Key switching, automorphisms, and rotation

This group contains evaluation-key-dependent CKKS operations and their RNS key-switch primitives. Whole-operation relinearization and rotation coexist with lower-level compositions.

| Operator | Schema | File |
|---|---|---|
| `fhelium_ckks.rotate` | `(CT, EK) -> CT` | `_resolve_rotation_keys.py`<br>`_keyswitch.py`<br>`operations.py` |
| `fhelium_ckks.hoisted_rotate_many` | `(CT, EK...) -> CT...` | `_hoist_rotations.py`<br>`operations.py`<br>`_hoisted.py` |
| `fhelium_ckks.relinearize` | `(CT) -> CT` | `_keyswitch.py`<br>`operations.py` |
| `fhelium_ckks.switch_key` | `(CT) {key_symbol: str} -> CT` | `_keyswitch.py` |
| `fhelium_ckks.conjugate` | `(CT) -> CT` | `_keyswitch.py` |
| `fhelium_rns.hybrid_modup_digit` | `(R, RP, KP) {digit_index: i64} -> R` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`mixed_radix_cuda.cu`<br>`rns_arithmetic_cuda.cu` |
| `fhelium_rns.key_switch_digit_product` | `(R, EK, RP, KP) {key_digit_index: i64} -> R` | `operations.py`<br>`keyswitch_cpu.cpp`<br>`keyswitch_cuda.cu` |
| `fhelium_rns.add_montgomery_lazy` | `(R, R, RP) -> R` | `operations.py`<br>`rns_arithmetic_cpu.cpp`<br>`rns_arithmetic_cuda.cu` |
| `fhelium_rns.moddown_qp_to_q` | `(R, RP, KP) -> R` | `operations.py`<br>`keyswitch_cpu.cpp`<br>`keyswitch_cuda.cu` |
| `fhelium_rns.coefficient_automorphism` | `(R, RP) {galois_element: i64} -> R` | `operations.py`<br>`automorphism_cpu.cpp`<br>`automorphism_cuda.cu` |

## Placement and memory movement

Existing values change placement through a transfer operation. The destination is a launch-bound device resource; `memory_space` may select default, pageable-host, or pinned-host storage.

| Operator | Schema | File |
|---|---|---|
| `fhelium_memory.transfer` | `(T, D) {memory_space?: str} -> T` | `operations.py` |

## Rank-local SPMD and collectives

A distributed Program represents one rank. Group resources supply rank and process-group identity at launch. Local verification does not prove collective uniformity, cross-rank order, associativity, or deadlock freedom. Generic all-reduce exposes its combine region; specialized ciphertext addition may remain a whole operation or lower to that generic form.

| Operator | Schema | File |
|---|---|---|
| `fhelium_dist.rank` | `(G) -> index` | `operations.py` |
| `fhelium_dist.group_size` | `(G) -> index` | `operations.py` |
| `fhelium_dist.broadcast` | `(T, G) {root: i64} -> T` | `operations.py` |
| `fhelium_dist.all_reduce` | `(T, G) {combine(T, T) -> T} -> T` | `operations.py` |
| `fhelium_dist.all_reduce_add_ciphertext` | `(CT, G) -> CT` | `_lower_specialized_collectives.py`<br>`operations.py` |
| `fhelium_dist.yield` | `(T) -> ()` | `execution.py` |

## Upstream structured execution substrate

The FHElium context loads xDSL Builtin, Func, Arith, and SCF dialects. The table
below lists the upstream operations that the current `ProgramExecutable`
handles directly.

| Operator | Schema | File |
|---|---|---|
| `builtin.unrealized_conversion_cast` | `(T) -> U` | `execution.py` |
| `func.return` | `(T...) -> ()` | `execution.py` |
| `arith.constant` | `() {value: I or F} -> I or F` | `execution.py` |
| `arith.addi` | `(I, I) -> I` | `execution.py` |
| `arith.subi` | `(I, I) -> I` | `execution.py` |
| `arith.muli` | `(I, I) -> I` | `execution.py` |
| `arith.divui` | `(I, I) -> I` | `execution.py` |
| `arith.divsi` | `(I, I) -> I` | `execution.py` |
| `arith.remui` | `(I, I) -> I` | `execution.py` |
| `arith.remsi` | `(I, I) -> I` | `execution.py` |
| `arith.minui` | `(I, I) -> I` | `execution.py` |
| `arith.maxui` | `(I, I) -> I` | `execution.py` |
| `arith.index_cast` | `(I) -> J` | `execution.py` |
| `arith.select` | `(i1, T, T) -> T` | `execution.py` |
| `arith.cmpi` | `(I, I) {predicate} -> i1` | `execution.py` |
| `scf.for` | `(index, index, index, T...) {body(index, T...) -> T...} -> T...` | `execution.py` |
| `scf.if` | `(i1) {then() -> T..., else() -> T...} -> T...` | `execution.py` |
| `scf.yield` | `(T...) -> ()` | `execution.py` |

Unknown application and vendor operations may coexist with this vocabulary in permissive Programs. Their schemas and implementations belong to the extension that defines them and therefore do not appear in this static index.

## Continue

- [Operation declaration and implementation selection](./operation-registration-and-selection.md)
- [Compiler stack internals](./compiler-stack-internals.md)
- [Compiler state and Eager execution](./compiler-state-and-eager-execution.md)
- [Distributed internals](./distributed-internals.md)
- [RNS and NTT internals](./rns-and-ntt.md)
