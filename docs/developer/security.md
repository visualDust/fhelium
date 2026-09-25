# Security

FHElium implements CKKS encryption, approximate homomorphic arithmetic, and decryption over polynomial residue rings. Its security depends on the configured lattice parameters, random sampling, custody of keys and plaintexts, and the environment that executes the computation. Eager execution and compiled Programs share key relations and registered numerical implementations.

## Presets and custom parameters

CKKS confidentiality is based on the assumed hardness of Ring Learning With Errors for the selected ring dimension, modulus, secret distribution, and error distribution. FHElium generates ternary secret keys and samples integer errors using the configured Gaussian sigma parameter. Public keys and evaluation keys contain noisy relations involving the secret key; their generation parameters are part of the cryptographic configuration.

For hybrid key switching, the parameter assessment includes the complete QP modulus:

$$
M=\prod_i q_i\prod_j p_j,\qquad b_M=\lceil\log_2 M\rceil.
$$

`fhelium.config.security` compares this bit width with the published parameter tables in [Security Guidelines for Implementing Homomorphic Encryption](https://doi.org/10.62056/anxra69p1). The tables use the classical RC.MATZOV lattice-estimation model and provide 128-, 192-, and 256-bit target categories for supported ring dimensions, secret distributions, and error standard deviation 3.19. The returned status is `meets`, `exceeds`, or `unsupported`, according to the matching row and modulus budget. These categories describe the published attack-cost estimates under the table's assumptions.

The built-in presets use ternary secrets, Gaussian sigma 3.19, and the table's 128-bit classical target category. Their complete QP products fit the corresponding published modulus budgets. These recipes provide ready-to-use parameter choices under that model.

Custom configurations expose the ring dimension, Q/P primes, error parameter, and requested security category. `enforce_security_budget=True` is the default: Engine construction checks the complete QP product and rejects exceeded or unsupported table budgets. Setting `enforce_security_budget=False` allows Engine construction with parameters outside that table policy. Applications preparing resources or Programs directly can invoke `config.validate_security_budget()` when they want the same assessment. The general assessment API also exposes Gaussian-secret table rows.

Parameter choice remains under programmer control. Attack experiments may deliberately select weak parameters; hybrid security constructions may use assumptions and protection mechanisms that differ from the built-in table model. Such uses can select their own parameters and decide whether to apply the table check. The experiment or application defines its security assumptions, acceptable exposure, and evaluation criteria; disabling the check leaves those choices with its author.

[Parameter and depth selection](../how-to/choose-preset-and-depth.md) explains how the arithmetic schedule determines the required chain. Increasing the modulus provides arithmetic capacity while changing the lattice-security estimate, so Q and auxiliary P must be planned together.

## Randomness and key material

`fhelium.rng.Csprng` uses the [`triton-csprng`](https://github.com/visualDust/triton-csprng) ChaCha20 streams for CPU and CUDA sampling. When key and nonce inputs are omitted, that dependency obtains their initial bytes from the operating system through `os.urandom`. User-supplied `rng_seed` and `rng_nonce` make the stream reproducible. Hashing an integer seed expands its representation but preserves the entropy of the supplied seed.

Secret-key generation samples ternary coefficients. Public-key and evaluation-key generation use uniform polynomials and Gaussian errors; encryption uses a binary masking polynomial and Gaussian errors. Integer Gaussian-error sampling uses finite-precision cumulative-distribution tables. Encoding consumes a separate rounding stream. Repeating a key/nonce/counter state repeats the corresponding samples, including after a process fork, a restored generator snapshot, or a copied stream-state Tensor. Independent cryptographic streams require distinct state, and seed material must have sufficient entropy.

Secret keys, encryption randomness, encoded plaintexts, and decoded results reside in ordinary Tensor storage. Code with access to the owning process or device can read that storage. Key cloning, device transfer, and optional automatic key replication create additional copies. PyTorch allocation and object deletion provide no secure-erasure guarantee; deployments that require erasure must account for allocator caches, aliases, device memory, and persisted copies.

An evaluator can receive ciphertexts and the public evaluation keys needed for its operations while the secret key remains with the decrypting party. Applications should distribute only the keys needed by the selected circuit and record which parameters and secret-key relation produced each key. Runtime value metadata describes shapes and arithmetic state; it carries no authenticated key-lineage identifier.

## Evaluation and decryption

Homomorphic evaluation deliberately permits ciphertext modification. CKKS ciphertexts therefore provide no authentication of the evaluated function, sender, or result. A service that accepts ciphertexts or returns decrypted results needs authenticated requests and a defined output-release policy. In particular, exposing a secret-key decryption endpoint to arbitrary attacker-selected ciphertexts changes the security setting from public-key encryption to an interactive decryption service.

The numerical implementation follows the represented operation schedule. Eager applies operation-specific state transitions; Compile transforms the Program and binds numerical materials; native operators check their execution ABI requirements. Binding a different Tensor under the same material symbol intentionally changes the computation. Programs, key assignments, and data provenance must be controlled by the application executing them.

CKKS decryption is approximate. Encoding roundoff, encryption noise, key switching, and rescale contribute to the output error, while modular wrap can change the recovered message when intermediate coefficients exceed the intended centered range. Record the circuit's input bounds, actual scales, depth schedule, and output precision. Security-table assessment and numerical-error analysis measure different properties of those same parameters. [Encoding and key construction](encoding-randomness-and-keys.md) gives the implemented equations.

## Files and stored Programs

FHElium value files and Compilation files store unencrypted safetensors payloads. Typed secret-key serialization requires `allow_secret=True`. Compilation persistence can include selected arbitrary Tensor materials, so the selected symbols must be reviewed for secret keys, plaintexts, and random state before saving. Restoring a saved random state can repeat a cryptographic stream.

ArtifactStore records payload hashes and generation identities. SHA-256 detects accidental corruption; a party able to replace both catalog and payload can also replace the recorded digest. Sensitivity labels describe contents and do not enforce access permissions. Protect stored secrets with filesystem permissions, authenticated encryption where required, and controlled access to backups.

Compilation persistence preserves Program IR and selected numerical data. Linking and executing imported IR can invoke registered implementations and generated code with the process's privileges. Use trusted Program sources and implementation registries, and isolate execution when the submitting party is outside that trust domain. Payload-format validation checks storage extents and metadata, not the behavior of executable code.

See [Compilation persistence](compilation-persistence.md) and [ArtifactStore](artifact-store-v1.md) for the formats, storage-sharing behavior, and publication lifecycle.

## Host, accelerator, and communication

FHElium executes through Python, PyTorch, native C++/CUDA operators, and generated Triton kernels. These paths have no end-to-end constant-time execution guarantee. Scheduling, memory access, transfers, allocation, and shared CPU/GPU resources can expose timing and access-pattern information. Secret-bearing processes and devices belong inside the deployment's trusted execution environment; isolation and side-channel requirements must be evaluated for the actual hardware and software stack.

Distributed ciphertext transport uses PyTorch process groups and their configured communication backend. Typed collectives validate and transport value descriptors and Tensor payloads. Authenticate participating ranks and protect the communication channel according to the data being sent, including any keys or plaintexts. Ciphertext encryption does not authenticate collective membership or operation order.

Experimental multiparty CKKS exposes arithmetic for collective keys, shares, and key switching. An application protocol supplies participant authentication, share freshness, transcript binding, error-distribution choices, and authorized output release. The current arithmetic APIs do not implement malicious-party validation of contributed shares. Secret-dependent output operations accept caller-supplied randomness and errors. Output privacy requires an analyzed output-error distribution; FHElium currently supplies no reviewed output-error sampler or supported noise-smudging and precision profile. [Multiparty CKKS](../how-to/use-multiparty-ckks.md) describes the individual protocol steps and their sampling requirements.

Install native extensions and generated-code dependencies from trusted build and package sources. The [binary packaging](binary-packaging-and-release.md) page documents wheel identities, dependency checks, and the artifact publication process.

## Report a vulnerability

Use [GitHub Private Vulnerability Reporting](https://github.com/VisualDust/fhelium/security/advisories/new) for suspected key disclosure, unsafe parsing or native memory access, cryptographic randomness defects, or compromised build artifacts. Include the affected version, execution environment, security impact, and a reproduction with synthetic data and disposable keys. Keep production secrets out of reports. The repository's [security policy](https://github.com/VisualDust/fhelium/blob/main/SECURITY.md) describes coordinated disclosure.
