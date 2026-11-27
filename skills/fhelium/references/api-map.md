# API map

`import fhelium as fh`; `from fhelium import compile as fc`. Full reference: https://fhelium.550w.host/api/.

| Module | Responsibility | Main entry points |
| --- | --- | --- |
| `fhelium` (`fh`) | Configuration, value and key types, value files | `fh.Preset`, `fh.CkksConfig.parse(preset, **overrides)`, `fh.Ciphertext`, `fh.Plaintext`, `fh.CompressedPlaintext`, `fh.EvaluationKeySet`, `fh.EvaluationKeyRequirements`, `fh.save_value`, `fh.load_value`, `fh.inspect_value`, `fh.compatible_ntt_backends(logN)`, `fh.__version__` |
| `fhelium.eager` | Eager CKKS evaluation | `Engine(config, *, ntt_backend=None, rns_dtype=None, rng_seed=None, allow_automatic_key_generation=True, allow_automatic_key_replication=False, expand_keys=True)` |
| `fhelium.values` | Value and key classes (re-exported by `fh`) | `Ciphertext`, `Plaintext`, `CompressedPlaintext`, `SecretKey`, `PublicKey`, `RelinearizationKey`, `RotationKey`, `RotationKeySet`, `ConjugationKey`, `KeySwitchKey`, `EvaluationKeySet` |
| `fhelium.algorithms.linear` | Plaintext-matrix × ciphertext transforms | `LinearTransform(diagonals, slots)`, `plan(t)` → `DirectPlan`/`BsgsPlan`, `prepare(engine, like, t, plan=None, *, keys)`, `apply(engine, x, t=None, plan=None, *, keys=None, prepared=None)`, `radix2_dft` |
| `fhelium.algorithms.polynomial` | Polynomial evaluation | `chebyshev_fit(f, degree)`, `Polynomial`, `plan(p)`, `evaluate(engine, x, p, plan=None, *, keys=None)`; plans `PatersonStockmeyerPlan`, `HornerPlan`, `BalancedPowerPlan`, Chebyshev variants |
| `fhelium.bootstrap` | CKKS bootstrapping | `plan(engine, profile='3/38', ...)` → `BootstrapPlan` (`key_requirements`, `output_depth`, `required_rotations`), `create_evaluation_keys(engine, plan, secret_key)`, `prepare(engine, plan, like, *, keys)` → `PreparedBootstrap`, `bootstrap(engine, x, plan=None, *, keys=None, prepared=None)`; stages `coeffs_to_slots`, `eval_mod`, `slots_to_coeffs`, `modulus_raise` |
| `fhelium.compile` (`fc`) | Capture, lowering, linking, compiled execution, export | `fc.compile`, `fc.export`, `fc.load`, `CompiledCallable` (`prepare`, `stats`, `clear`, `refresh_materials`, `with_backend`, `specializations`), `LoadedFunction`, `CaptureError`, `CompileError`, `ExportError`, `SpecializationMiss`, `fc.capture`, `Pipeline`, pass classes |
| `fhelium.ir` | Program IR | `Program`, `fhelium.ir.parse`, `format_program`, `analyze_evaluation_key_requirements(program)`, `analyze_value_states` |
| `fhelium.runtime` | Execution helpers | `DoubleBuffer(example, *, count=2, device=None)` with `put`/`feed`, `ReusableValueBuffer`, `pin_value_tree`, `value_tree_nbytes` |
| `fhelium.residency` | Host/CUDA placement under budgets | `ResidencyManager` (`adopt`, `ensure`, `move`, `scope`, `acquire`, `explain`, `snapshot`, `discard`, `close`), `ResidencyPlan`, `PINNED_HOST`, `cuda_location` |
| `fhelium.store` | Named value storage | `ValueStore`, `ValueCollection`, `StoredValueRef` |
| `fhelium.serialization` | Value and Compilation files | `save_value`, `load_value`, `save_compilation`, `load_compilation` |
| `fhelium.distributed` | Multi-GPU collectives on CKKS values | `init_process_group`, `broadcast_ciphertext`, `all_reduce_ciphertext` and `reduce_ciphertext` (captured by `fc.compile`), `scatter_ciphertext_limbs`, `gather_ciphertext_limbs`, `all_gather_ciphertexts` |
| `fhelium.errors` | Exception classes | `FHEliumError`, `StateError`, `MaximumDepthError`, `ScaleMismatchError`, `PolynomialDomainError`, `ResidencyBudgetError`, `ValueStoreError` |
| `fhelium.native` | Native library status | `native_status()`, `native_available()`, `cmake_prefix_path()` |

## Engine methods by task

- Keys: `create_secret_key`, `create_public_key`, `create_relinearization_key`, `create_rotation_key(step, sk)`, `create_conjugation_key`, `install_evaluation_key`, `set_rotation_key`, `rotation_key_steps`.
- Encode and encrypt: `encrypt_message(m, pk, *, depth=0, scale=None, device=None)`, `decrypt_message(c, sk, *, is_real=False)`, `encode`, `decode`, `encrypt`, `decrypt`, `prepare_compressed_plaintext(m, *, depth, device)`, `prepare_plaintext_for_multiplication`.
- Arithmetic: `add`, `subtract`, `negate`, `add_plaintext`, `add_scalar`, `multiply` (NTT inputs), `multiply_plaintext`, `multiply_constant`, `multiply_integer_scalar`, `relinearize(v, key, *, rescale=False)`.
- Depth and scale: `rescale_to_next_depth`, `mod_switch_to_depth`, `mod_switch_to_next_depth`, `reinterpret_at_scale`, `depth_remaining`, `max_depth`, `rescale_output_scale`.
- Domain: `coefficient_domain_to_ntt_domain`, `ntt_domain_to_coefficient_domain`.
- Rotation: `rotate_with_key`, `rotate_many_with_keys` (hoisted), `rotate_many_by_steps`, `rotate_by_step`, `conjugate`.
- Fused sums: `sum_plaintext_products(cts, pts)`, `sum_plaintext_product_groups`, `sum_rotated_plaintext_product_groups`, `sum_rotations(terms)`, `sum_ciphertexts`, `sum_ciphertext_batch`.
- Properties: `num_slots`, `ring_dimension`, `dtype`, `config`.
