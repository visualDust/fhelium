# Glossary

This glossary defines FHElium's canonical value-state and rank-local
terminology. A glossary entry does not replace expansion at first use on a
standalone page. API signatures remain authoritative for identifiers and
types.

## Construction and identity

### `Preset`

A maintained CKKS parameter baseline whose name records complex slot capacity,
default scale-prime width, public-level count, and integral tensor dtype. Each
preset also selects the corresponding residue buffer width, ring dimension,
and P-prime count. `CkksConfig.parse` resolves the baseline and applies any
overrides.

### `CkksConfig`

The resolved mathematical and security configuration: ring, Q/P prime chains,
integral residue buffer width, default planning scale, Gaussian error standard
deviation, classical security category, and modulus-bit-budget enforcement
policy. FHElium secret keys use uniform-ternary sampling.

### `CkksContextSpec`

Immutable, device-independent compatibility metadata derived from a
`CkksConfig`. Its `context_id` identifies the complete mathematical context.

### `CkksEngine`

One process-local evaluator binding a resolved configuration to one device, NTT
policy, random-number generator, precomputed tables, keys, and operations.

### `context_id`

A stable identity derived from all mathematical context parameters. It prevents
values from different contexts from being combined. It does not identify a
particular secret key or encode device placement.

## Cryptographic context and arithmetic

### Active rows

The Q or QP residue rows represented at a value's current level. `prime_ids`
map the physical row order to canonical moduli.

### Basis (`Q` / `QP`)

The modulus set represented by a value. Q is the ciphertext modulus chain; QP
also includes special P moduli used by hybrid key switching. `modulus_basis` is
independent of level, active-row completeness, and polynomial domain.

### CKKS

An approximate homomorphic-encryption scheme that encodes complex or real
values into polynomial slots and supports approximate arithmetic on encrypted
values.

### Coefficient domain

A polynomial represented by coefficients. This axis is distinct from plaintext
representation and from standard/Montgomery residue representation.

### CRT reconstruction

Chinese Remainder Theorem (CRT) reconstruction combines residues modulo
coprime primes into an integer representative modulo their product. Tail-Q
binary64 decrypt reconstruction is approximate and must not be described as
exact CRT.

### Default scale

The context value `config.default_scale`, equal to `2**config.scale_bits`.
Value creation selects it when the scale argument is omitted. Arithmetic reads
and updates the actual scale stored on each value.

### Level

The number of leading scale primes consumed from the Q chain. Level zero uses
the complete configured Q chain; a larger level has fewer active Q rows. Level
and actual scale are independent state coordinates. Public values use levels
from zero through `engine.final_public_level`.

### Final public level

The greatest ordinary public CKKS level, exposed as
`engine.final_public_level`. Its active Q rows contain the final scale prime
and structural base prime. `rescale_to_next_level` and `mod_switch_to_next_level` require a
source level strictly below it. See
[Scale and level lifecycle](ckks/scale-and-level-lifecycle.md).

### Limb

One residue-polynomial row for one modulus. A value's limb axis is interpreted
through its ordered `prime_ids`.

### ModUp

Hybrid key-switch basis extension from active Q rows into QP while preserving
the represented polynomial. It is distinct from bootstrap ModRaise.

### ModDown

Hybrid key-switch down-conversion from QP to Q, including division by the P
product according to the documented rounding law.

### ModRaise

Centered bootstrap basis extension from the single active structural base Q row
to the larger bootstrap Q basis. It relies on an application-owned input bound.

### Montgomery representation

A residue representation for efficient modular multiplication. It is tracked
independently from coefficient/NTT domain, although valid FHElium NTT
ciphertexts use Montgomery form.

### NTT

The Number Theoretic Transform converts negacyclic polynomial multiplication
into pointwise modular multiplication. NTT backend policy controls table layout
and grouped radix-2 stages.

### `N = 2^logN`

The polynomial-ring dimension. CKKS exposes `N/2` complex slots. Increasing
`N` increases slots and arithmetic/key storage cost.

### P

Special auxiliary moduli used by hybrid key switching. P rows are normally
temporary for ciphertext data and appear in QP evaluation-key layouts.

### Q

The ordinary ciphertext modulus chain. Rescale removes leading Q scale primes
as level increases.

### RNS

A Residue Number System (RNS) represents an integer polynomial by its residues
modulo several coprime primes. Each prime contributes one limb/RNS row.

### Rescale

A state transition that approximately divides by the leading active scale
prime and drops its RNS row. It changes level, actual scale, `prime_ids`, and row
count. The public one-level operation is named `rescale_to_next_level`.

### Rescale drop prime

The leading active Q prime used as the divisor and removed by one rescale.
`engine.rescale_to_next_drop_prime(level=...)` returns this integer for a
non-final public level.

### Actual scale

The positive finite binary64 CKKS encoding factor stored as `value.scale`.
Multiplication multiplies actual scales; rescale divides by the actual dropped
Q prime.

### Scale reinterpretation

A metadata transition that records a different actual scale while preserving
ciphertext residues. It changes the decoded message by the old-to-new scale
ratio. The public operation is named `reinterpret_at_scale`.

### Slot

One packed complex CKKS message position. A ring of dimension `N` has `N/2`
slots. Applications define how vectors, matrices, padding, and masks map to
slots.

## Values and key relations

### Ciphertext component

One polynomial component in an encrypted value. Fresh ciphertexts have two;
ciphertext-ciphertext multiplication naturally creates three until
relinearization.

### CKKS value state

The stored combination of concrete type, tensor topology, context, level,
actual scale, `prime_ids`, plaintext representation where applicable,
polynomial domain, modulus basis, residue representation, and component or key
specialization. Device placement is separate. Ciphertext/key lineage is an
external cryptographic relation unless a concrete type stores a specialization
such as `rotation_step`.

### External cryptographic relation

An application-maintained compatibility condition between ciphertexts and key
material when no symbolic lineage field is stored. Examples include a public
key's destination secret and a generic key-switch key's source/destination
secret relation.

### Homogeneous batch

Zero or more local message dimensions whose members share one value's
CKKS metadata. RNS plaintexts use
`[*batch, limb, coefficient_or_ntt_index]`; ciphertexts use
`[component, *batch, limb, coefficient_or_ntt_index]`. A homogeneous batch is
not packed slots, RNS-row placement, hybrid digits, or distributed processes.

### Key switch

A transformation that changes a ciphertext component's secret dependency.
Relinearization, rotation, and conjugation use specialized key-switch material.

### Operation-ready plaintext

A plaintext already encoded at a level and arithmetic state suitable for
an evaluator operation. It avoids repeated preparation but usually occupies
more memory than a semantic slots value or `integer_coefficients` value.

### Relinearization

A specialized key switch that transforms a three-component multiplication
result back to two components.

### Rotation

A CKKS slot permutation implemented as a Galois automorphism plus a key switch.
Each direct rotation uses a key specialized for one canonical signed step.

### Rotation hoisting

Reuse of decomposition, ModUp, and NTT preparation shared by several rotations
of one input component. Step-specific automorphism, key products, ModDown, and
output storage remain.

### `TensorResident`

The protocol for tensor-resident values whose declared tensor fields move together. It
supplies device/byte inspection and .to(...) without placement
history or cache policy.

## Bootstrapping

### Bootstrapping

The CKKS refresh operation that raises available modulus, maps coefficients to
slots, applies periodic reduction, and maps slots back to coefficients. A
bootstrap callable/object executes the operation; a refreshed ciphertext is its
output.

### CoeffsToSlots / SlotsToCoeffs

The linear transforms between coefficient embedding and two conjugate slot
vectors used by full-slot bootstrapping. `CoeffsToSlots` is the forward map;
`SlotsToCoeffs` is its inverse composition stage.

### BSGS

Baby-step/giant-step (BSGS) decomposes a diagonal linear transform into two
rotation levels to trade rotation-key families and repeated work against
intermediate computation.

### Periodic reduction

The bootstrap component approximating the periodic map that removes the
encoded large-modulus quotient. The corresponding composition field is named
`modular_reduction` in code.

### Structural base Q prime

The single Q prime remaining after all public scale primes are dropped. At the
private bootstrap source level, `[q_b]` is the one active Q row; it is not a new
`modulus_basis` value.

## Distributed execution

### Additive partial

A process-local ciphertext representing one summand of a shared logical result.
Partials combine through CKKS modular addition, not structural concatenation.

### Additive-term parallelism

Processes own disjoint summands of one result and reduce compatible ciphertext
partials. Rotation/diagonal offsets are concrete examples; heads or experts fit
this category only when the mathematical output is defined as their sum.

### Collective

A communication operation that every participating process group member enters
in compatible order. A communication collective is distinct from a
cryptographic collective key or decryption protocol.

### Data parallelism

Processes evaluate independent requests or samples. Results are gathered as a
list rather than reduced into one ciphertext.

### Descriptor/payload separation

A typed transport pattern that exchanges value metadata first, allocates and
validates the receiver, and then transfers dense tensor payloads.

### Gather

Transport preserving independent process-local objects in a list. It performs
no CKKS arithmetic.

### Limb parallelism

Processes own disjoint RNS rows of one logical value. Row-local work is
possible, but an operation requiring every expected active row needs full
reconstruction first.

### Process group

An ordered set of participating processes used by a communication backend.
World size is the number of processes in the group.

### Process rank

A process's integer position. Distinguish global process rank,
process-group-relative rank, and node-local process rank (`LOCAL_RANK`). A CUDA
device index is not a rank, and tensor dimensionality should be written `ndim`.

### Reconstruction

Structural concatenation of disjoint RNS rows into one value with the complete
active-row layout. It is not addition.

### SPMD

Single Program, Multiple Data: each process executes the same worker function
on process-local data and a local device, with defined collective order and
ownership.

### Typed ciphertext reduction

A collective combining additive ciphertext partials with modulus-aware engine
addition. It is not raw machine-integer NCCL `SUM`.

## Multiparty CKKS

### Cryptographic party

An independent trust-domain participant. A party is not a CKKS slot, process
rank, or CUDA device; an application maps party identities to processes and
transport endpoints.

### CKG and RKG

Collective key generation (CKG) constructs collective public-key material.
Relinearization-key generation (RKG) constructs the evaluation key for
three-to-two-component relinearization. These arithmetic phases do not provide
a coordinator, authenticated membership, or transport.

### Collective decryption and public-key switching

Collective-decryption shares fuse into an output under the documented unsafe
arithmetic scope. Public-key-switch shares fuse into a ciphertext under a
destination public key. The current experimental namespace provides no
privacy or production-security guarantees for these output operations.

## Execution and lifecycle

### Borrowed output

An output backed by storage retained and reused by another owner, such as a
CUDA Graph program. A later replay may overwrite it; request an owned copy to
retain old contents.

### `CopyHandle`

An object representing an enqueued reusable-buffer copy. It retains source
storage and exposes completion/wait operations for stream-safe consumption.

### CUDA Graph

A captured fixed GPU execution schedule replayed with stable addresses. In
FHElium, graph capture is a process-local mechanism with dynamic-input value
signatures.

### Eager execution

Immediate Python/operator dispatch performed call by call, outside a captured
CUDA Graph replay.

### Value signature

A device-independent description of nested tensor structure and value
state used to validate reusable copies and graph inputs. It cannot encode an
external cryptographic relation that a concrete value does not store.

### Hold

Longer-lived residency protection expressing that an application wants a
resource retained. A hold does not expose an evaluator-value mapping.

### Lease

Short-lived protection for active evaluator reads. A lease must remain active
until every consumer, including asynchronous CUDA readers, has completed.

### Materialization

A ready managed value in one local residency location, such as pageable CPU,
pinned CPU, or an indexed CUDA device.

### Residency budget

An optional application-supplied strict admission limit for managed
materialization charges and reservations at one location. Unbudgeted locations
retain complete current and peak byte accounting.

### Residency plan

Immutable ordered low-level intermediate representation (IR) of explicit
`ensure`, `move`, `drop`, and `discard` actions plus scoped memory accounting
reservations. The application specifies each handle, destination, optional
move-source constraint, and removal.

### `ReusableValueBuffer`

Fixed-address storage for one value-tree signature on one target device.
It supports scheduled copying and synchronization for eager streaming or graph
input staging.

## Implementation and build terms

### ABI

An application binary interface (ABI) is the compiled compatibility boundary
among CPython, PyTorch, its CUDA variant, the local CUDA toolkit, C++ ABI, and
native binaries. Ordinary wheel tags do not encode the complete PyTorch/CUDA
ABI.

### Native operator stack

The PyTorch dispatcher schemas, C++ bindings, and CUDA implementations loaded
by FHElium. This is distinct from NTT backend policy, process-group
communication backend, and Python build backend.

## Documentation terms

### Concept

An explanation of a stable mental model, invariant, or design rationale.

### Developer Guide

Implementation-oriented documentation explaining how the engine, native
operator stack, ABI, and source layers fit together.

### How-to guide

A task-oriented procedure for completing or diagnosing one concrete goal.

### Tutorial

A learning-oriented end-to-end workflow, usually aligned with a maintained
runnable example.
