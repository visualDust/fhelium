# Encoding, randomness, and key construction

The CKKS codec maps ordered complex slots to scaled integer coefficients and reconstructs slots after decryption. Cryptographic data provision constructs secret, public, and evaluation keys, while execution consumes their Tensor payloads and live randomness. These mechanisms are implemented by the libfhelium functions of `fhelium/ckks_operations.h`, called from `backend/ckks/codec/` and `backend/ckks/crypto/`, with randomness from `backend/random/`.

## How does the embedding become integer data?

For ring dimension $N$, a dense message has a final slot axis of length at most $N/2$ and arbitrary leading batch axes. `fhelium::ckks::encode` pads the slot axis to $N/2$. Generator-dependent permutation tables select generator-3 or generator-5 ordering and embed the slots into conjugate-symmetric length-$N$ data $z$.

Encoding computes real coefficients using the supplied permutation and twister tables:

$$
a_j=\operatorname{Re}\left(\operatorname{FFT}_{\mathrm{forward\ norm}}(z)_j e^{-\pi i j/N}\right).
$$

Encoding stochastically quantizes $\Delta a_j$ using the device's random-stream seed Tensor as its rounding-state operand. For $|x|=k+f$ with integer $k$ and $0\le f<1$, the implementation adds one to $k$ when a 32-bit random word is below $\lfloor 2^{32}f\rfloor$, then restores the sign of $x$. Coefficient $j$ reads key-stream word $j$ of the current nonce, and the caller advances the nonce on the device after each call.

`fhelium::ckks::decode` multiplies by the inverse twister, applies the corresponding inverse FFT, restores slot order, and divides the resulting values by the actual scale $\Delta$. The codec's use of Torch FFT with `norm="forward"` is part of the implemented normalization; changing it requires compensating the embedding equations.

`_embedding.py` caches the permutation and twister tables, `_implementation.py` registers the encode, decode and integer-to-RNS operations, and `_periodic.py` registers compact periodic plaintext preparation. The permutation, twister, and rounding-state arrays are ordinary material operands.

## How do coefficients enter and leave the residue basis?

Integer-to-RNS conversion (`fhelium::rns::lift_integer_coefficients`) reduces each signed int32 or int64 coefficient into the ordered supplied prime rows in one elementwise kernel: row $j$ receives $x+q_j$ when $-q_j\le x<q_j$ and $x \bmod q_j$ otherwise, a standard residue in $[0,2q_j)$ in both cases. The selection is made per element on the device, so encoding, encryption and scalar arithmetic do not read coefficient bounds back to the host. After decryption, `fhelium::ckks::decrypt` reconstructs the centered coefficient class modulo the full active Q product using mixed-radix data and half-product comparison. Multiple prime rows cannot generally be replaced by reading one convenient residue.

The mathematical message must remain within the intended centered range modulo Q; otherwise modular wrap changes the recovered value. FFT approximation, stochastic quantization, encryption noise, key switching, and rescale contribute different errors. Decode with the ciphertext's actual scale rather than an assumed prime width or configuration default.

The [compressed plaintext article](compressed-plaintext-internals.md) explains encoded-axis layouts, which are distinct from visible slot repetition. The [RNS article](rns-and-ntt.md) defines row ordering and representation transitions.

## Which random streams are used?

`fhelium.backend.random.stream.RandomStream` holds one device's int64 seed Tensor of shape `[5]`: four ChaCha20 key words and the nonce. Its samplers call the libfhelium `fhelium_random_ops` operators, which read a stateless coordinate-addressed key stream: sample $i$ of a call uses words $i \cdot w$ through $i \cdot w + w - 1$ of the stream selected by the current nonce, and the stream then increments the nonce in place on the device. Residue outputs take a `[parameter, limb]` RNS parameter table and return `[*batch, limb, coefficient]` residues modulo every row; without the table, samplers return centered int64 coefficients. The stream is bound as the `ckks-random-stream` Backend resource, and its seed Tensor is the `rounding_state` material of captured Programs.

Secret-key generation samples ternary coefficients with nonzero probability 2/3, or a fixed-weight support from the smallest 127-bit labels when a Hamming weight is supplied. Public-key generation samples Gaussian error and then uniform residues; each key-switching digit samples one such public key. Encryption samples $e_0$, the zero-mean ZO(0.5) mask $v$, and $e_1$ in this order: $e_0$ joins the integer plaintext before lifting, $v$ is sampled directly as residues, and $e_1$ is added to the second component in place. Encoding consumes the same stream through stochastic rounding. Each call advances the nonce once.

Use fixed seeds and nonces for reproducible experiments. Cryptographic use requires suitable seed entropy and unique nonces for independent streams under the same key. Reusing a seed/nonce or restoring the same stream state can repeat samples. CKKS security estimates depend on ring dimension, modulus chain, secret distribution, and error parameters.

## What relation does each key represent?

libfhelium constructs every key (`fhelium/ckks_keys.h`): `create_secret_key`, `prepare_secret_key`, `create_public_key`, `create_key_switch_key`, `create_relinearization_key`, `create_galois_key` for rotation and conjugation keys, and `convert_secret_key` for the change of RNS basis that encapsulation keys use. Each function samples its components at consecutive nonces of the stream without modifying it, and the caller advances the stream by the number of sampled components. The default digit partition comes from `key_digit_sizes` (`fhelium/ckks_parameters.h`), which groups consecutive Q primes while their product stays below $P$. `KeyGenerationResource` groups the device's RNS/NTT contexts and random stream and provides the named tables of these functions. `CkksKeyGenerator` checks key states, selects the key rows and digits, calls the native function, and returns the public key value classes.

| Value | Relation or purpose | Stored Tensor axes |
| --- | --- | --- |
| Secret key | Ternary polynomial $s$ | `[limb, ntt_index]` |
| Public key | $k_0+k_1s=e$ modulo the selected basis | `[key_component=2, limb, ntt_index]` |
| Hybrid key-switch key | Digit relation from $s_{\rm src}$ to $s_{\rm dst}$ | `[key_digit, key_component=2, QP_limb, ntt_index]` |
| Relinearization key | Switch the $s^2$ term to $s$ | Hybrid key-switch layout |
| Rotation/conjugation key | Switch the automorphed secret relation to the target secret | Hybrid key-switch layout with operation identity |

Secret and public key payloads are constructed at depth zero in NTT/Montgomery representation. A hybrid key-switch key stores the Q suffix selected by `source_depth`, its selected auxiliary P rows and the corresponding digit planes. The default P comes from the configuration; `p_prime_ids` selects a disjoint ordered auxiliary basis from the same catalog. For digit $d$, the embedded source rows satisfy

$$
k_{d,0}+k_{d,1}s_{\mathrm{dst}}=P s_{\mathrm{src}}+e_d,
$$

where $P$ is the auxiliary-basis product. The source term is present only in the digit's embedded Q rows. Active-depth digit selection preserves the stable key-digit mapping.

## How does encryption use the public key?

For encoded message polynomial $m$, ZO(0.5) mask $v$, and sampled errors $e_0,e_1$, encryption constructs

$$
c_0=vk_0+m+e_0,\qquad c_1=vk_1+e_1.
$$

The mask follows Section 3.4 of [Cheon, Kim, Kim, and Song, *Homomorphic Encryption for Arithmetic of Approximate Numbers*](https://yongsoosong.github.io/files/papers/HEAAN.pdf) (ePrint 2016/421): ZO($\rho$) assigns probability $\rho/2$ to each of $-1$ and $+1$, and $1-\rho$ to zero; encryption selects $\rho=0.5$. The shared RNG helper draws uniform integers $u\in\{0,1,2,3\}$ and maps them to $0,+1,-1,0$, respectively, using the difference of the two bits. The power-of-two bound is unbiased, giving probabilities $1/4,1/2,1/4$ for $-1,0,+1$ while preserving Tensor device, integral dtype, and channel layout. Backend and Legacy encryption use the same helper and retain the existing stream/nonce mechanism. Correcting the former binary mask changes encryption samples even with the same fixed seed and nonce.

Writing the public-key error as $e_{\mathrm{pk}}=k_0+k_1s$, decryption yields $m+ve_{\mathrm{pk}}+e_0+e_1s$. Conditional on a fixed public key, the zero-mean mask removes the contribution $\mathbb{E}[v]e_{\mathrm{pk}}$ caused by the former nonzero mask mean. The remaining noise and the encoding error determine slot precision; individual slots need not have equal errors.

`crypto/_encryption.py` performs the supplied NTT transitions and modular products. It consumes public-key and parameter/table Tensors plus a bound sampler handle. Coefficient output performs the inverse transition and standard-residue addition; NTT output retains the selected NTT/Montgomery arithmetic form.

`crypto/_decryption.py` evaluates $c_0+c_1s$ or $c_0+c_1s+c_2s^2$, converts the phase to coefficient/standard RNS, and reconstructs centered Q coefficients. Slot decoding then applies the inverse scale and embedding.

## How is data provision connected to execution?

Engine factories select device services and invoke key construction. Existing keys can be supplied to Eager or bound as Tensor materials in a Compilation. Capture retains actual supplied key dataflow; optional material preparation can choose among caller-provided keys but never creates an absent evaluation key.

Extend codecs beside their table and embedding owners, and extend key constructions beside the relevant secret relation. Preserve sample distributions, scale and row conventions, and the declared Tensor ABI. For persistence, review included material symbols carefully: [Compilation files](compilation-persistence.md) are unencrypted and can contain secret or stream-state data when requested.
