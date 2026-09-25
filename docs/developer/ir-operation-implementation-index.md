# IR operations and implementations

This index maps operation families to their Compile transformations and Backend execution owners. Paths below are relative to `fhelium/`. An IR abstraction level describes the amount of computational detail represented by an operation; CKKS modulus-chain depth is a separate value property.

The operation definitions specify operands, attributes, equations, and state transitions: [CKKS](/api/fhelium/ir/dialects/ckks), [RNS](/api/fhelium/ir/dialects/rns), [NTT](/api/fhelium/ir/dialects/ntt), and [core data references](/api/fhelium/ir/dialects/core). Numerical parameters, tables, and keys enter execution as Tensor operands. Non-Tensor execution handles use resource references.

## Program structure, binding, and interoperability

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium.constant` | Materialize a represented literal in generated host code | `compile/passes/codegen/_host.py` |
| `fhelium.material.ref` | Read the Tensor bound to a Program symbol | `compile/passes/backend/_operations.py`, `_prepare_host.py` |
| `fhelium.resource.ref` | Bind a supplied execution handle | `compile/passes/backend/_operations.py`, `_link_program.py` |
| `torch.call` | Execute supported public Tensor calls | `backend/torch.py` |
| `torch.tensor_call` | Execute supported public Tensor calls | `backend/torch.py` |
| `fhelium_fusion.yield` | Return the region's selected result values | `compile/passes/codegen/_host.py`, `backend/triton/_expressions.py` |
| `fhelium_fusion.execute` | Execute a selected generated region | `backend/triton/_pointwise.py`, `_tensor.py` |

Compile selects fusion regions in `compile/passes/fusion.py`. The Triton backend owns expression lowering, generated kernels, compilation, and launch. A region may contain several kernels, including native NTT middle stages.

## Frontend intent and operand roles

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_semantic.add` | Classify addition operands by encrypted and public roles | `compile/passes/frontend/_lower_semantic_to_logical.py` |
| `fhelium_semantic.subtract` | Classify subtraction operands by encrypted and public roles | `compile/passes/frontend/_lower_semantic_to_logical.py` |
| `fhelium_semantic.multiply` | Classify multiplication operands by encrypted and public roles | `compile/passes/frontend/_lower_semantic_to_logical.py` |
| `fhelium_semantic.negate` | Classify unary negation by its input role | `compile/passes/frontend/_lower_semantic_to_logical.py` |
| `fhelium_semantic.roll` | Classify slot rotation by its input role | `compile/passes/frontend/_lower_semantic_to_logical.py` |
| `fhelium_logical.add.encrypted_encrypted` | Construct CKKS add for encrypted/encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.add.encrypted_public` | Construct CKKS add for encrypted/public operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.add.public_encrypted` | Construct CKKS add for public/encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.subtract.encrypted_encrypted` | Construct CKKS subtract for encrypted/encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.subtract.encrypted_public` | Construct CKKS subtract for encrypted/public operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.subtract.public_encrypted` | Construct CKKS subtract for public/encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.multiply.encrypted_encrypted` | Construct CKKS multiply for encrypted/encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.multiply.encrypted_public` | Construct CKKS multiply for encrypted/public operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.multiply.public_encrypted` | Construct CKKS multiply for public/encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.negate.encrypted` | Construct CKKS negate for encrypted operands | `compile/passes/ckks/_lower_logical_to_ckks.py` |
| `fhelium_logical.roll.encrypted` | Resolve rotation keys and the coefficient rotation route | `compile/passes/ckks/_resolve_rotation_keys.py` |

The shared construction helpers preserve value identity through transparent type references. Arithmetic lowering retains unknown facts until they can be derived or assigned; a type cast does not perform an NTT or change residue data.

## Boundary conversion and cryptography

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_ckks.encode` | Embed message slots and quantize scaled coefficients | `backend/ckks/codec/_implementation.py`, `_codec.py`, `_embedding.py` |
| `fhelium_ckks.decode` | Reconstruct message slots from scaled coefficients | `backend/ckks/codec/_implementation.py`, `_codec.py`, `_embedding.py` |
| `fhelium_ckks.integer_coefficients_to_rns` | Reduce integer coefficients into supplied prime rows | `backend/ckks/codec/_implementation.py`, `backend/rns/context.py` |
| `fhelium_ckks.prepare_compressed_plaintext` | Prepare periodic compact plaintexts | `backend/ckks/codec/_periodic.py` |
| `fhelium_ckks.encrypt` | Sample and assemble ciphertext components | `backend/ckks/crypto/_encryption.py` |
| `fhelium_ckks.decrypt` | Evaluate the ciphertext phase and reconstruct coefficients | `backend/ckks/crypto/_decryption.py` |

Key generation is a data-provision API in `backend/ckks/crypto/_key_generation.py`. Decryption's numerical reconstruction tables live in `crypto/_tables.py`; `crypto/_resources.py` defines the encryption sampler's execution-resource names.

## Ciphertext arithmetic and component composition

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_ckks.add` | Lower ciphertext addition to row-wise RNS addition | `compile/passes/lowering/_arithmetic.py` |
| `fhelium_ckks.subtract` | Lower ciphertext subtraction to row-wise RNS subtraction | `compile/passes/lowering/_arithmetic.py` |
| `fhelium_ckks.negate` | Lower ciphertext negation to row-wise RNS negation | `compile/passes/lowering/_arithmetic.py` |
| `fhelium_ckks.multiply` | Whole CT2 convolution or RNS component composition | `backend/ckks/arithmetic.py`, `compile/passes/lowering/_arithmetic.py` |
| `fhelium_ckks.add_scalar` | Encode and add a scalar at the ciphertext scale | `backend/ckks/scalar.py` |
| `fhelium_ckks.multiply_scalar` | Encode and multiply by a scalar with its represented scale | `backend/ckks/scalar.py` |
| `fhelium_ckks.multiply_integer_scalar` | Multiply by an unscaled integer while preserving actual scale | `backend/ckks/scalar.py` |
| `fhelium_rns.add_standard` | Add corresponding residues modulo each supplied prime | `backend/rns/operations.py` |
| `fhelium_rns.subtract_standard` | Subtract corresponding residues modulo each supplied prime | `backend/rns/operations.py` |
| `fhelium_rns.negate_standard` | Negate residues modulo each supplied prime | `backend/rns/operations.py` |
| `fhelium_rns.montgomery_multiply` | Multiply corresponding Montgomery residues | `backend/rns/operations.py` |
| `fhelium_rns.add_montgomery_lazy` | Accumulate Montgomery residues in the declared lazy range | `backend/rns/operations.py` |
| `fhelium_rns.extract_component` | Select one polynomial component from a Tensor bundle | `backend/rns/operations.py` |
| `fhelium_rns.pack_two_components` | Assemble a two-component Tensor bundle | `backend/rns/operations.py` |
| `fhelium_rns.pack_three_components` | Assemble a three-component Tensor bundle | `backend/rns/operations.py` |

## Public and plaintext preparation and mixed arithmetic

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_ckks.prepare.add.message` | Prepare message data for CKKS add arithmetic | `compile/passes/ckks/_lower_message_preparation.py` |
| `fhelium_ckks.prepare.add.plaintext` | Prepare plaintext data for CKKS add arithmetic | `compile/passes/ckks/_lower_message_preparation.py` |
| `fhelium_ckks.prepare.add.static` | Prepare static data for CKKS add arithmetic | `compile/passes/ckks/_lower_message_preparation.py` |
| `fhelium_ckks.prepare.multiply.message` | Prepare message data for CKKS multiply arithmetic | `compile/passes/ckks/_lower_message_preparation.py` |
| `fhelium_ckks.prepare.multiply.plaintext` | Prepare plaintext data for CKKS multiply arithmetic | `compile/passes/ckks/_lower_message_preparation.py` |
| `fhelium_ckks.prepare.multiply.static` | Prepare static data for CKKS multiply arithmetic | `compile/passes/ckks/_lower_message_preparation.py` |
| `fhelium_ckks.add_plaintext` | Lower addition of a supplied plaintext to component zero | `compile/passes/lowering/_arithmetic.py` |
| `fhelium_ckks.multiply_plaintext` | Lower multiplication of ciphertext components by a supplied plaintext | `compile/passes/lowering/_arithmetic.py` |
| `fhelium_ckks.add_compressed_plaintext` | Add compact plaintext values to component zero through encoded-axis indexing | `backend/ckks/arithmetic.py`, `backend/triton/_expressions.py` |
| `fhelium_ckks.multiply_compressed_plaintext` | Multiply ciphertext components by compact plaintext values through encoded-axis indexing | `backend/ckks/arithmetic.py`, `backend/triton/_expressions.py` |
| `fhelium_rns.add_plaintext` | Add the supplied plaintext polynomial to component zero | `backend/rns/operations.py` |
| `fhelium_rns.multiply_plaintext` | Multiply each ciphertext component by the supplied plaintext polynomial | `backend/rns/operations.py` |
| `fhelium_rns.montgomery_weighted_sum` | Accumulate one weighted sum of polynomial Tensor products | `backend/rns/operations.py` |
| `fhelium_rns.montgomery_weighted_sums` | Apply several plaintext weight rows to shared polynomial Tensor inputs | `backend/rns/operations.py` |

## Polynomial and residue representation transforms

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_ckks.to_ntt` | Lower a forward polynomial-domain transition to NTT operations | `compile/passes/lowering/_representation.py` |
| `fhelium_ckks.from_ntt` | Lower an inverse polynomial-domain transition to NTT operations | `compile/passes/lowering/_representation.py` |
| `fhelium_ckks.to_montgomery_residues` | Lower standard-to-Montgomery residue conversion | `compile/passes/lowering/_representation.py` |
| `fhelium_ckks.to_standard_residues` | Lower Montgomery-to-standard residue conversion | `compile/passes/lowering/_representation.py` |
| `fhelium_rns.standard_to_montgomery` | Convert standard residues to Montgomery form | `backend/rns/operations.py` |
| `fhelium_rns.montgomery_to_standard` | Convert Montgomery residues to standard form | `backend/rns/operations.py` |
| `fhelium_ntt.coefficient_standard_to_ntt_montgomery` | Forward NTT with standard input | `backend/ntt/operations.py`, `backend/ntt/executors/` |
| `fhelium_ntt.coefficient_montgomery_to_ntt_montgomery` | Forward NTT with Montgomery input | `backend/ntt/operations.py`, `backend/ntt/executors/` |
| `fhelium_ntt.ntt_montgomery_to_coefficient_standard` | Normalized inverse NTT with standard output | `backend/ntt/operations.py`, `backend/ntt/executors/` |
| `fhelium_ntt.ntt_montgomery_to_coefficient_montgomery` | Normalized inverse NTT retaining Montgomery output | `backend/ntt/operations.py`, `backend/ntt/executors/` |

`backend/ntt/plans/` constructs twiddle and index tables; `backend/ntt/tables.py` materializes their device-specific views. Native transitions execute through `backend/ntt/operations.py`. Generated NTT regions use `backend/triton/_ntt.py` and `_ntt_codegen.py`.

## Modulus-chain and scale transitions

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_ckks.rescale` | Lower a rounded quotient over the next complete Q depth group | `compile/passes/lowering/_representation.py` |
| `fhelium_ckks.mod_switch` | Lower restriction to the selected active Q rows | `compile/passes/lowering/_representation.py` |
| `fhelium_ckks.reinterpret_scale` | Lower a represented-scale change that preserves residue data | `compile/passes/lowering/_representation.py` |
| `fhelium_rns.rescale_drop_leading_primes` | Divide by a dropped Q group with the selected rounding | `backend/rns/rescale.py` |
| `fhelium_rns.restrict_depth` | Select the active rows at the requested depth | `backend/rns/operations.py` |
| `fhelium_rns.reinterpret_scale` | Update represented scale while preserving residue data | `backend/rns/operations.py` |

Modulus switching to fewer Q rows and ModDown by the auxiliary P product have different equations. They remain separate operations. Numerical inverse tables for rescaling and ModDown are constructed in `backend/rns/tables.py`.

## Key switching, automorphisms, and rotation

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_ckks.switch_key` | Assemble ciphertext components under the supplied destination key relation | `backend/ckks/key_switch.py` |
| `fhelium_ckks.conjugate` | Compose conjugation with the supplied key-switch relation | `backend/ckks/key_switch.py` |
| `fhelium_ckks.relinearize` | Reduce a three-component ciphertext through a supplied relinearization key | `backend/ckks/key_switch.py` |
| `fhelium_ckks.rotate` | Compose a slot-selected automorphism and key switch | `backend/ckks/rotation/operations.py` |
| `fhelium_ckks.hoisted_rotate_many` | Reuse input-derived digit preparation across keys | `backend/ckks/rotation/operations.py`, `_hoisted.py` |
| `fhelium_ckks.grouped_rotation_weighted_sum` | Compose a selected rotation group with plaintext products | `backend/ckks/rotation/operations.py` |
| `fhelium_rns.hybrid_modup_digit` | Decompose and extend one digit into QP | `backend/rns/modup.py` |
| `fhelium_rns.key_switch_digit_product` | Multiply one prepared digit by its supplied key rows | `backend/rns/key_product.py` |
| `fhelium_rns.moddown_qp_to_q` | Remove the auxiliary P basis in coefficient representation | `backend/rns/moddown.py` |
| `fhelium_rns.moddown_ntt_qp_to_q` | Remove the auxiliary P basis while retaining Q NTT evaluations | `backend/rns/moddown.py` |
| `fhelium_rns.coefficient_automorphism` | Apply a supplied polynomial Galois element | `backend/rns/automorphism.py` |

The CKKS slot-step convention is defined in `backend/ckks/rotation/_galois.py`. Coefficient and NTT permutation builders accept the polynomial Galois element in `backend/rns/automorphism.py` and `backend/ntt/automorphism.py` respectively. Neither requires a key Tensor or a slot-rotation convention.

`compile/passes/lowering/_keyswitch.py` exposes key switching as RNS/NTT dataflow. The whole-operation and lowered routes use the same numerical owners. `backend/rns/_preparation.py` describes hybrid arithmetic tables and ModDown requirements; `backend/ckks/_key_switch_preparation.py` declares whole CKKS implementation requirements. `backend/ckks/tables.py` assembles the table views used by those whole implementations. Material lookup does not generate keys.

## Placement and memory movement

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_memory.transfer` | Move a Tensor to the supplied destination handle | `backend/memory/operations.py` |

`backend/memory/resources.py` owns the device resource kind. Tensor placement and storage policy are caller choices.

## Rank-local SPMD and collectives

| Operation | Responsibility | Direct owner |
|---|---|---|
| `fhelium_dist.rank` | Read the process-group-relative rank | `backend/distributed/operations.py` |
| `fhelium_dist.group_size` | Read the participating process-group size | `backend/distributed/operations.py` |
| `fhelium_dist.broadcast` | Broadcast a rank-local Tensor from the selected group root | `backend/distributed/operations.py` |
| `fhelium_dist.all_reduce` | Reduce rank-local Tensors through a supplied combine region | `backend/distributed/operations.py` |
| `fhelium_dist.all_reduce_add_ciphertext` | Execute ciphertext-add reduction or expose a generic combine region | `backend/distributed/operations.py`, `compile/passes/distributed/_lower_specialized_collectives.py` |
| `fhelium_dist.yield` | Return a combine region's Tensor result | `compile/passes/codegen/_host.py` |

`backend/distributed/resources.py` owns process-group handles. The distributed schedule must preserve cross-rank operation order and the combine region's algebraic requirements.

## Upstream structured execution substrate

| Operation | Responsibility | Direct owner |
|---|---|---|
| `builtin.unrealized_conversion_cast` | Forward a one-input, one-result type reference | `compile/passes/codegen/_host.py` |
| `func.return` | Return function results | `compile/passes/codegen/_host.py` |
| `arith.constant` | Materialize a scalar integer or floating-point constant | `compile/passes/codegen/_host.py` |
| `arith.addi` | Add scalar indices | `compile/passes/codegen/_host.py` |
| `arith.subi` | Subtract scalar indices | `compile/passes/codegen/_host.py` |
| `arith.muli` | Multiply scalar indices | `compile/passes/codegen/_host.py` |
| `arith.divui` | Evaluate unsigned scalar quotient control flow | `compile/passes/codegen/_host.py` |
| `arith.divsi` | Evaluate signed scalar quotient control flow | `compile/passes/codegen/_host.py` |
| `arith.remui` | Evaluate unsigned scalar remainder control flow | `compile/passes/codegen/_host.py` |
| `arith.remsi` | Evaluate signed scalar remainder control flow | `compile/passes/codegen/_host.py` |
| `arith.minui` | Select the minimum scalar index | `compile/passes/codegen/_host.py` |
| `arith.maxui` | Select the maximum scalar index | `compile/passes/codegen/_host.py` |
| `arith.index_cast` | Adapt a scalar index | `compile/passes/codegen/_host.py` |
| `arith.select` | Select one scalar value from a condition | `compile/passes/codegen/_host.py` |
| `arith.cmpi` | Compare scalar indices using the represented predicate | `compile/passes/codegen/_host.py` |
| `scf.if` | Execute the selected conditional region | `compile/passes/codegen/_host.py` |
| `scf.for` | Execute a loop with carried values | `compile/passes/codegen/_host.py` |
| `scf.yield` | Return region values to structured control flow | `compile/passes/codegen/_host.py` |

`compile/passes/backend/_prepare_host.py` binds the emitted host function. `ProgramExecutable` invokes it with the linked Tensor materials and execution resources.

## Continue

- [Operation registration and implementation selection](./operation-registration-and-selection.md)
- [IR, capture, effects, and open state](./compiler-stack-internals.md)
- [Values, operation semantics, and Eager dispatch](./compiler-state-and-eager-execution.md)
- [Distributed internals](./distributed-internals.md)
- [RNS and NTT internals](./rns-and-ntt.md)
