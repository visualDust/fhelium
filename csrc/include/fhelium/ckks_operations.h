#pragma once

#include <ATen/core/Tensor.h>
#include <fhelium/ckks_keys.h>
#include <fhelium/export.h>
#include <fhelium/operation_tables.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

//! Whole CKKS operations over prepared tables: key switching,
//! conjugation, relinearization with optional joint rescale, rotation,
//! hoisted rotation groups, rotation sums, grouped rotation weighted sums,
//! modulus raising, compressed-plaintext and scalar arithmetic, slot
//! encoding and decoding, encryption, and decryption.
//!
//! A ciphertext has shape `[component, limb, N]` over the active Q rows. The
//! hybrid key switch decomposes component one into the `digits` of
//! `KeySwitchOptions`, lifts each digit to QP, multiplies it with the key
//! digit in NTT Montgomery form, and divides the accumulated products by the
//! auxiliary product $P$ (ModDown). Key-switching functions read
//! `KeySwitchTables` (`fhelium/operation_tables.h`): the fields read by
//! `fhelium/rns_operations.h` (`hybrid_modup_digit`, `moddown_qp_to_q`,
//! `moddown_ntt_qp_to_q`, `moddown_add_rescale`), the `crt` tables of each
//! digit for the CRT ModUp methods, `q_parameters`, and the `q`, `qp`, and
//! `p` NTT schedules. A key has shape `[key_digit, 2, limb, N]` with NTT
//! Montgomery QP residues. Rotation steps follow
//! `torch.roll(slots, shifts=step)`; `rotation_galois_element` in
//! `fhelium/ckks_tables.h` maps them to polynomial automorphisms, and each
//! rotation reads the `GaloisTables` of its automorphism. The `_out` forms
//! write the result into caller-supplied Tensors with the result's shape,
//! dtype, and device that do not overlap the operands.

namespace fhelium::ckks {

/// Selection of the decomposition, ModUp, ModDown, and domains of one key
/// switch.
struct KeySwitchOptions {
  /// NTT policy name (`fhelium/ntt_operations.h`).
  std::string ntt_backend;
  /// Flattened `(start, stop, key_digit)` triples of the hybrid digits.
  std::vector<int64_t> digits;
  /// First key row that corresponds to QP row zero.
  int64_t key_row_start = 0;
  /// First key row of the trailing auxiliary rows, or $-1$ when the active
  /// QP rows are one key interval starting at `key_row_start`.
  int64_t key_p_row_start = -1;
  /// Key rows of the active Q rows, in operand order, when they do not form
  /// one interval starting at `key_row_start`; empty otherwise. A nonempty
  /// list requires `key_p_row_start`, and each run of consecutive key rows
  /// is one contiguous digit product.
  std::vector<int64_t> key_q_rows;
  /// Digit lift: `exact` (Garner) reads the mixed-radix fields of
  /// `KeySwitchTables::digits`; `crt` (raw CRT) and `normalized_crt`
  /// (centered CRT) read their `crt` fields. ModDown divides by $P$ with the
  /// centered CRT remainder when `moddown_crt` is defined
  /// (`fhelium::rns::moddown_qp_to_q`) and exactly otherwise.
  std::string basis_conversion = "exact";
  /// Number of auxiliary rows that trail the Q rows of the QP accumulator.
  int64_t p_count = 0;
  /// `nearest` or `floor` ModDown rounding.
  std::string moddown_rounding = "nearest";
  /// Domain of the input ciphertext: `coefficient` (standard residues) or
  /// `ntt` (Montgomery residues).
  std::string input_domain = "coefficient";
  /// Domain of the result, encoded as for `input_domain`; NTT results hold
  /// lazy residues.
  std::string output_domain = "coefficient";
  /// Seed rows `[key_digit, 6]` of seeded keys, one per key operand in
  /// operand order; empty when the keys are dense. A seeded key stores only
  /// $k_0$ as `[key_digit, 1, key_rows, N]`, and the products generate $k_1$
  /// (`keyswitch_accumulate_digit_products_`).
  std::vector<at::Tensor> key_seeds;
  /// ChaCha rounds of the seeded keys; 0 for dense keys.
  int64_t key_rounds = 0;
};

/// Joint ModDown and rescale of a relinearization
/// (`fhelium::rns::moddown_add_rescale`).
struct JointRescaleOptions {
  int64_t q_count = 0;
  int64_t joint_drop_count = 0;
  bool joint_drop_is_prefix = false;
};

/// Accumulates the hybrid key products of coefficient standard Q polynomial
/// `component` into NTT Montgomery QP accumulators of shape
/// `[2, *batch, limb, N]`.
///
/// When `accumulator` is supplied, products are added to it in place and it
/// is returned; otherwise a zero accumulator is allocated.
FHELIUM_API at::Tensor key_switch_products(
    const at::Tensor& component,
    const at::Tensor& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<at::Tensor>& accumulator = std::nullopt);

/// Switches CT2 `source` from the key's source secret to its destination
/// secret.
///
/// For input phase $c_0 + c_1 s_{src}$, the key products give corrections
/// $(d_0, d_1)$ under $s_{dst}$, and the result is $(c_0 + d_0, d_1)$. With
/// `conjugation`, the Galois tables of $X \mapsto X^{-1}$, both components
/// first take that automorphism. The input is coefficient standard Q
/// residues; the output domain follows `options`.
FHELIUM_API at::Tensor key_switch(
    const at::Tensor& source,
    const at::Tensor& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<GaloisTables>& conjugation = std::nullopt);

/// Out form of `key_switch`.
FHELIUM_API void key_switch_out(const at::Tensor& source,
                                const at::Tensor& key,
                                const KeySwitchTables& tables,
                                const KeySwitchOptions& options,
                                const std::optional<GaloisTables>& conjugation,
                                at::Tensor& out);

/// Relinearizes NTT Montgomery CT3 `source` to CT2.
///
/// Component two is switched with the relinearization key and the
/// corrections are added to components zero and one. With `rescale`, the
/// division by $P$ is combined with the rescale of `rescale_options`.
FHELIUM_API at::Tensor relinearize(
    const at::Tensor& source,
    const at::Tensor& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options = std::nullopt);

/// Out form of `relinearize`. A defined `workspace` holds the temporaries: a
/// contiguous uint8 Tensor of at least `relinearize_workspace_bytes` bytes on
/// the source device with a 256-byte aligned base. With an undefined
/// `workspace`, the temporaries are allocated per call.
FHELIUM_API void relinearize_out(
    const at::Tensor& source,
    const at::Tensor& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `relinearize_out` for these operands; reads only their
/// sizes, dtypes and devices.
FHELIUM_API int64_t relinearize_workspace_bytes(
    const at::Tensor& source,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options = std::nullopt);

/// Rotates CT2 `source` by one slot step: applies the Galois automorphism
/// `galois` of the step to both components and key-switches component one.
FHELIUM_API at::Tensor rotate(const at::Tensor& source,
                              const at::Tensor& key,
                              const GaloisTables& galois,
                              const KeySwitchTables& tables,
                              const KeySwitchOptions& options);

/// Out form of `rotate`.
FHELIUM_API void rotate_out(const at::Tensor& source,
                            const at::Tensor& key,
                            const GaloisTables& galois,
                            const KeySwitchTables& tables,
                            const KeySwitchOptions& options,
                            at::Tensor& out);

/// Returns the hoisted digits `[digit, *batch, qp_limb, N]` of coefficient
/// standard Q polynomial `component`: every hybrid digit lifted to QP once
/// and transformed to NTT Montgomery form, shared by the key products of
/// several rotations (`rotate_hoisted`).
FHELIUM_API at::Tensor rotation_digits(const at::Tensor& component,
                                       const KeySwitchTables& tables,
                                       const KeySwitchOptions& options);

/// Out form of `rotation_digits`.
FHELIUM_API void rotation_digits_out(const at::Tensor& component,
                                     const KeySwitchTables& tables,
                                     const KeySwitchOptions& options,
                                     at::Tensor& out);

/// Rotates a CT2 by the automorphism `galois` of one slot step from its
/// component zero `c0` and the hoisted digits `hoisted_digits` of its
/// component one (`rotation_digits`).
///
/// The NTT gather of the step permutes each digit in the key-product load.
/// With coefficient output, `c0` holds coefficient standard residues; with
/// NTT output (`options.output_domain`), `c0` holds NTT Montgomery residues
/// and the result holds lazy NTT Montgomery residues.
FHELIUM_API at::Tensor rotate_hoisted(const at::Tensor& c0,
                                      const at::Tensor& hoisted_digits,
                                      const at::Tensor& key,
                                      const GaloisTables& galois,
                                      const KeySwitchTables& tables,
                                      const KeySwitchOptions& options);

/// Out form of `rotate_hoisted`. A defined `workspace` holds the temporaries:
/// a contiguous uint8 Tensor of at least `rotate_hoisted_workspace_bytes`
/// bytes on the device of `c0` with a 256-byte aligned base. With an
/// undefined `workspace`, the temporaries are allocated per call.
FHELIUM_API void rotate_hoisted_out(const at::Tensor& c0,
                                    const at::Tensor& hoisted_digits,
                                    const at::Tensor& key,
                                    const GaloisTables& galois,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options,
                                    at::Tensor& out,
                                    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `rotate_hoisted_out`; reads only the sizes, dtypes and
/// devices of the operands.
FHELIUM_API int64_t
rotate_hoisted_workspace_bytes(const at::Tensor& c0,
                               const at::Tensor& hoisted_digits,
                               const KeySwitchTables& tables,
                               const KeySwitchOptions& options);

/// Rotates coefficient standard CT2 `source` by every step of `galois`,
/// decomposing component one once (`rotation_digits`, `rotate_hoisted`).
/// `keys[i]` belongs to `galois[i]`.
FHELIUM_API std::vector<at::Tensor> rotate_many(
    const at::Tensor& source,
    at::TensorList keys,
    c10::ArrayRef<GaloisTables> galois,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options);

/// Out form of `rotate_many`. `workspace` is a contiguous uint8 Tensor on the
/// device of `source` with a 256-byte aligned base and at least
/// `rotate_many_workspace_bytes` bytes; it holds the hoisted digits, the
/// accumulators and the ModDown temporaries. With an undefined `workspace`,
/// the temporaries are allocated per call.
FHELIUM_API void rotate_many_out(const at::Tensor& source,
                                 at::TensorList keys,
                                 c10::ArrayRef<GaloisTables> galois,
                                 const KeySwitchTables& tables,
                                 const KeySwitchOptions& options,
                                 at::TensorList out,
                                 at::Tensor& workspace);

/// Workspace bytes of `rotate_many_out`. Only the sizes and dtypes of the
/// operands are read, so Meta and fake Tensors qualify.
FHELIUM_API int64_t
rotate_many_workspace_bytes(const at::Tensor& source,
                            at::TensorList keys,
                            c10::ArrayRef<GaloisTables> galois,
                            const KeySwitchTables& tables,
                            const KeySwitchOptions& options);

/// Adds the rotation $\operatorname{rot}_{s}(c)$ of CT2 `source` to the
/// rotation-sum accumulators.
///
/// `ordinary` (`[2, *batch, limb, N]` over Q) and `products`
/// (`[2, *batch, qp_limb, N]`) start as zeros. A nonzero step applies its
/// Galois automorphism `galois`, adds component zero to `ordinary[0]`, and
/// adds the key products of component one to `products` in NTT Montgomery
/// form; step zero (no `galois`) adds both components to `ordinary` and
/// leaves `products` unread. `ordinary` keeps the input domain of `options`:
/// lazy NTT Montgomery or coefficient standard residues. A defined
/// `workspace` holds the temporaries of a nonzero step: a contiguous uint8
/// Tensor of at least `rotation_sum_add_workspace_bytes` bytes on the device
/// of `source` with a 256-byte aligned base. With an undefined `workspace`,
/// the temporaries are allocated per call.
FHELIUM_API void rotation_sum_add_(at::Tensor& ordinary,
                                   at::Tensor& products,
                                   const at::Tensor& source,
                                   const std::optional<at::Tensor>& key,
                                   const std::optional<GaloisTables>& galois,
                                   const KeySwitchTables& tables,
                                   const KeySwitchOptions& options,
                                   const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `rotation_sum_add_`; zero for step zero (no `galois`).
/// Reads only the sizes, dtypes and devices of the operands.
FHELIUM_API int64_t
rotation_sum_add_workspace_bytes(const at::Tensor& source,
                                 const std::optional<GaloisTables>& galois,
                                 const KeySwitchTables& tables,
                                 const KeySwitchOptions& options);

/// Returns the coefficient standard rotation sum of the accumulators of
/// `rotation_sum_add_` with one shared quotient.
///
/// `ordinary` is multiplied by $P$ in NTT Montgomery form and added to the
/// Q rows of `products`; the result is $\lfloor (z + h) / D \rfloor$ on the
/// surviving `remaining_q_count` rows, with $D = P$ or $P$ times the dropped
/// Q group and $h = 0$ or $(D - 1)/2$ by `options.moddown_rounding`. Reads
/// `sum_p_montgomery`, `sum_quotient_parameters`, `sum_quotient_inverses`,
/// and the optional `sum_quotient_order` and `sum_quotient_half`. Both
/// accumulators are overwritten.
FHELIUM_API at::Tensor rotation_sum_finish_(at::Tensor& ordinary,
                                            at::Tensor& products,
                                            const KeySwitchTables& tables,
                                            const KeySwitchOptions& options,
                                            int64_t remaining_q_count);

/// Out form of `rotation_sum_finish_`. A defined `workspace` holds the
/// temporaries: a contiguous uint8 Tensor of at least
/// `rotation_sum_finish_workspace_bytes` bytes on the device of `products`
/// with a 256-byte aligned base. With an undefined `workspace`, the
/// temporaries are allocated per call.
FHELIUM_API void rotation_sum_finish_out_(
    at::Tensor& ordinary,
    at::Tensor& products,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    int64_t remaining_q_count,
    at::Tensor& out,
    const at::Tensor& workspace = at::Tensor());

/// Workspace bytes of `rotation_sum_finish_out_`; reads only the sizes, dtypes
/// and devices of the operands.
FHELIUM_API int64_t
rotation_sum_finish_workspace_bytes(const at::Tensor& products,
                                    const KeySwitchTables& tables,
                                    int64_t remaining_q_count);

/// Returns $\sum_i \operatorname{rot}_{s_i}(c_i)$ with one shared quotient
/// (`rotation_sum_add_` for every source, then `rotation_sum_finish_`).
/// `keys` lists the keys of the nonzero steps in order, and `galois[i]`
/// holds the automorphism of `steps[i]`; entries of zero steps are unread.
FHELIUM_API at::Tensor rotation_sum(at::TensorList sources,
                                    at::TensorList keys,
                                    c10::IntArrayRef steps,
                                    c10::ArrayRef<GaloisTables> galois,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options,
                                    int64_t remaining_q_count);

/// Out form of `rotation_sum`.
FHELIUM_API void rotation_sum_out(at::TensorList sources,
                                  at::TensorList keys,
                                  c10::IntArrayRef steps,
                                  c10::ArrayRef<GaloisTables> galois,
                                  const KeySwitchTables& tables,
                                  const KeySwitchOptions& options,
                                  int64_t remaining_q_count,
                                  at::Tensor& out);

/// Returns `group_count` sums $\sum_j \operatorname{rot}_{b_j}(c) \cdot
/// m_{g,j}$ over the baby steps $b_j$ with plaintexts $m_{g,j}$
/// (`plaintexts[g * term_count + j]`) in NTT Montgomery form.
///
/// Component one of the coefficient standard CT2 `source` is decomposed once;
/// `keys` lists the keys of the nonzero baby steps in order, and `galois[j]`
/// holds the automorphism of baby step $b_j$. The result has shape
/// `[group, component, limb, N]` with NTT Montgomery residues.
FHELIUM_API at::Tensor grouped_rotation_weighted_sum(
    const at::Tensor& source,
    at::TensorList keys,
    c10::IntArrayRef baby_steps,
    c10::ArrayRef<GaloisTables> galois,
    at::TensorList plaintexts,
    int64_t group_count,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options);

/// Out form of `grouped_rotation_weighted_sum`.
FHELIUM_API void grouped_rotation_weighted_sum_out(
    const at::Tensor& source,
    at::TensorList keys,
    c10::IntArrayRef baby_steps,
    c10::ArrayRef<GaloisTables> galois,
    at::TensorList plaintexts,
    int64_t group_count,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    at::Tensor& out);

/// Raises CT2 `source` from its low basis through the encapsulation keys.
///
/// The source is key-switched with `low_key` under `low_options` and the
/// `low` tables into the low secret relation, extended to the target basis
/// by `fhelium::rns::centered_basis_extend` with the `raise` tables, and
/// returned to the main secret with the
/// one-digit `return_key`, which encodes $D t$ for $D$ the guard prime times
/// the return auxiliary product. The result is
/// $(\tilde c_0, 0) + \lfloor \tilde c_1 (H_0, H_1) / D \rceil$ on Q rows
/// $[$`target_row_start`$, $`q_count`$)$ with coefficient standard residues.
/// Additionally reads the `return` NTT schedules, `quotient_order`,
/// `quotient_parameters`, `quotient_inverses`, `target_parameters`, and
/// `quotient_half` for nearest rounding.
FHELIUM_API at::Tensor encapsulated_modulus_raise(
    const at::Tensor& source,
    const at::Tensor& low_key,
    const at::Tensor& return_key,
    const EncapsulationTables& tables,
    const KeySwitchOptions& low_options,
    int64_t return_drop_count,
    int64_t target_row_start,
    int64_t q_count);

/// Key operands of `KeySwitchKey` type. Each overload equals its Tensor form
/// with the key data and, for compact keys, `options.key_seeds` and
/// `options.key_rounds` taken from the keys; the keys of one call share one
/// form.
FHELIUM_API at::Tensor key_switch_products(
    const at::Tensor& component,
    const KeySwitchKey& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<at::Tensor>& accumulator = std::nullopt);
FHELIUM_API at::Tensor key_switch(
    const at::Tensor& source,
    const KeySwitchKey& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<GaloisTables>& conjugation = std::nullopt);
FHELIUM_API at::Tensor relinearize(
    const at::Tensor& source,
    const KeySwitchKey& key,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options,
    const std::optional<JointRescaleOptions>& rescale_options = std::nullopt);
FHELIUM_API at::Tensor rotate(const at::Tensor& source,
                              const KeySwitchKey& key,
                              const GaloisTables& galois,
                              const KeySwitchTables& tables,
                              const KeySwitchOptions& options);
FHELIUM_API at::Tensor rotate_hoisted(const at::Tensor& c0,
                                      const at::Tensor& hoisted_digits,
                                      const KeySwitchKey& key,
                                      const GaloisTables& galois,
                                      const KeySwitchTables& tables,
                                      const KeySwitchOptions& options);
FHELIUM_API std::vector<at::Tensor> rotate_many(
    const at::Tensor& source,
    const std::vector<KeySwitchKey>& keys,
    c10::ArrayRef<GaloisTables> galois,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options);
FHELIUM_API at::Tensor rotation_sum(at::TensorList sources,
                                    const std::vector<KeySwitchKey>& keys,
                                    c10::IntArrayRef steps,
                                    c10::ArrayRef<GaloisTables> galois,
                                    const KeySwitchTables& tables,
                                    const KeySwitchOptions& options,
                                    int64_t remaining_q_count);
FHELIUM_API at::Tensor grouped_rotation_weighted_sum(
    const at::Tensor& source,
    const std::vector<KeySwitchKey>& keys,
    c10::IntArrayRef baby_steps,
    c10::ArrayRef<GaloisTables> galois,
    at::TensorList plaintexts,
    int64_t group_count,
    const KeySwitchTables& tables,
    const KeySwitchOptions& options);
FHELIUM_API at::Tensor encapsulated_modulus_raise(
    const at::Tensor& source,
    const KeySwitchKey& low_key,
    const KeySwitchKey& return_key,
    const EncapsulationTables& tables,
    const KeySwitchOptions& low_options,
    int64_t return_drop_count,
    int64_t target_row_start,
    int64_t q_count);

/// Returns `ciphertext` with the compressed plaintext `compressed` added to
/// component zero.
///
/// `layout` is `cyclic` (one period repeated over the coefficients),
/// `contiguous` (leading coefficients), or `strided_sparse` (every $N/U$-th
/// coefficient, with `implicit` holding the value of the other positions).
/// Coefficient-domain plaintexts hold $pR$; NTT-domain plaintexts are first
/// converted to $pR^2$, so the addition yields $cR + pR$.
FHELIUM_API at::Tensor add_compressed_plaintext(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout,
    std::string_view polynomial_domain);

/// In-place form of `add_compressed_plaintext`; returns `ciphertext`.
FHELIUM_API at::Tensor& add_compressed_plaintext_(
    at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout,
    std::string_view polynomial_domain);

/// Out form of `add_compressed_plaintext`. `out` has the shape, dtype and
/// device of `ciphertext` and is `ciphertext` or disjoint from it; the
/// multiplication and scalar out forms below have the same requirement.
FHELIUM_API void add_compressed_plaintext_out(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout,
    std::string_view polynomial_domain,
    at::Tensor& out);

/// Returns every component of `ciphertext` multiplied by the compressed
/// Montgomery plaintext `compressed` with the layouts of
/// `add_compressed_plaintext`. Strided plaintexts fill the implicit positions
/// with the product of the one-value cyclic row `implicit`.
FHELIUM_API at::Tensor multiply_compressed_plaintext(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout);

/// In-place form of `multiply_compressed_plaintext`; returns `ciphertext`.
FHELIUM_API at::Tensor& multiply_compressed_plaintext_(
    at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout);

/// Out form of `multiply_compressed_plaintext`.
FHELIUM_API void multiply_compressed_plaintext_out(
    const at::Tensor& ciphertext,
    const at::Tensor& compressed,
    const std::optional<at::Tensor>& implicit,
    const at::Tensor& parameters,
    std::string_view layout,
    at::Tensor& out);

/// Adds or multiplies the real scalar $a$ at scale $\Delta$.
///
/// The integer $\lfloor a\Delta \rceil$ is rounded stochastically with the
/// stream `seed` at its current nonce and lifted to every row of `parameters`
/// with `rns::lift_integer_coefficients`, and the nonce of `seed` advances
/// by one. `add` adds it to the constant coefficient of component zero;
/// otherwise every component is multiplied by it.
FHELIUM_API at::Tensor scalar_arithmetic(const at::Tensor& ciphertext,
                                         const at::Tensor& parameters,
                                         at::Tensor& seed,
                                         double scalar,
                                         double scalar_scale,
                                         bool add);

/// Out form of `scalar_arithmetic`.
FHELIUM_API void scalar_arithmetic_out(const at::Tensor& ciphertext,
                                       const at::Tensor& parameters,
                                       at::Tensor& seed,
                                       double scalar,
                                       double scalar_scale,
                                       bool add,
                                       at::Tensor& out);

/// Encodes complex slots `[*batch, slot]` into stochastically rounded
/// integer coefficients `[*batch, N]` at scale $\Delta$.
///
/// The message is padded to the slot count, placed by the slot permutation
/// `pre`, extended conjugate-symmetrically, transformed by a forward FFT, and
/// multiplied by the twist `twister` (`slot_permutations`, `slot_twist`).
/// Rounding reads the stream `seed` at its current nonce and advances the
/// nonce by one. Every scaled coefficient must satisfy $|c\Delta| <
/// 2^{63}$; otherwise the call raises `c10::ValueError` with the largest
/// magnitude and the scale.
FHELIUM_API at::Tensor encode(const at::Tensor& message,
                              const at::Tensor& pre,
                              const at::Tensor& twister,
                              at::Tensor& seed,
                              double scale);

/// Decodes coefficients `[*batch, N]` to the leading $N/2$ complex slots
/// divided by $\Delta$, or their real parts when `is_real`.
FHELIUM_API at::Tensor decode(const at::Tensor& coefficients,
                              const at::Tensor& post,
                              const at::Tensor& skewer,
                              double scale,
                              bool is_real);

/// Prepares the compact plaintext of the periodic message $a(X^{N/U})$,
/// where $U$ is the extent of `pre`.
///
/// Rounding of coefficient $j$ uses word $jN/U$ of the stream `seed` at its
/// current nonce, and the nonce advances by one. NTT output applies
/// the $U$-point indexed schedule `even`, `odd`, `twiddles`; coefficient
/// output returns Montgomery residues and a zero implicit row.
FHELIUM_API std::vector<at::Tensor> prepare_periodic_plaintext(
    const at::Tensor& message,
    const at::Tensor& pre,
    const at::Tensor& twister,
    at::Tensor& seed,
    const at::Tensor& parameters,
    const at::Tensor& twice_moduli,
    const at::Tensor& even,
    const at::Tensor& odd,
    const at::Tensor& twiddles,
    int64_t ring_dimension,
    double scale,
    std::string_view polynomial_domain);

/// Encrypts integer coefficients `[*batch, N]` under public key
/// $(k_0, k_1)$ of shape `[2, limb, N]`.
///
/// With ternary mask $v$ ($\Pr[\pm 1] = 1/4$) and discrete-Gaussian errors
/// $e_0, e_1$ drawn with table `cdt_table` from the stream `seed` at nonces
/// $n, n+1, n+2$ in the order $e_0, v, e_1$, the result is $(v k_0 + m + e_0, v
/// k_1 + e_1)$ over the rows of `tables.forward.parameters` starting at key
/// row `key_row_start`, and the nonce of `seed` advances by three.
FHELIUM_API at::Tensor encrypt(const at::Tensor& coefficients,
                               const at::Tensor& public_key,
                               const RingTables& tables,
                               at::Tensor& seed,
                               const at::Tensor& cdt_table,
                               std::string_view ntt_backend,
                               int64_t key_row_start,
                               std::string_view output_domain);

/// Evaluates the phase $\sum_k c_k s^k$ of CT2 or CT3 `ciphertext` as
/// coefficient standard residues.
FHELIUM_API at::Tensor decrypt_phase(const at::Tensor& ciphertext,
                                     const at::Tensor& secret_key,
                                     const RingTables& tables,
                                     std::string_view ntt_backend,
                                     int64_t key_row_start,
                                     std::string_view input_domain);

/// Returns the centered integer class modulo the product of the primes of
/// `source_parameters` as float64 `[*batch, N]`.
FHELIUM_API at::Tensor reconstruct_q_coefficients(
    const at::Tensor& residues,
    const at::Tensor& source_parameters,
    const at::Tensor& normalizers,
    const at::Tensor& propagation,
    const at::Tensor& half_digits);

/// Decrypts `ciphertext` and reconstructs its centered Q coefficients
/// (`decrypt_phase` with `tables.ring`, then `reconstruct_q_coefficients`).
FHELIUM_API at::Tensor decrypt(const at::Tensor& ciphertext,
                               const at::Tensor& secret_key,
                               const DecryptionTables& tables,
                               std::string_view ntt_backend,
                               int64_t key_row_start,
                               std::string_view input_domain);

}  // namespace fhelium::ckks
