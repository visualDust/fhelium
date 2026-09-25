# Encoding, randomness, and key construction

The CKKS codec maps ordered complex slots to scaled integer coefficients and reconstructs slots after decryption. Cryptographic data provision constructs secret, public, and evaluation keys, while execution consumes their Tensor payloads and live randomness. These mechanisms are implemented in `backend/ckks/codec/`, `backend/ckks/crypto/`, and `rng/`.

## How does the embedding become integer data?

For ring dimension $N$, a dense message has a final slot axis of length at most $N/2$ and arbitrary leading batch axes. `_embedding.make_slot_tensor` forms the full slot axis. Generator-dependent permutation tables select generator-3 or generator-5 ordering and embed the slots into conjugate-symmetric length-$N$ data $z$.

`inverse_embed_slots` computes real coefficients using the supplied permutation and twister tables:

$$
a_j=\operatorname{Re}\left(\operatorname{FFT}_{\mathrm{forward\ norm}}(z)_j e^{-\pi i j/N}\right).
$$

Encoding stochastically quantizes $\Delta a_j$ using a live rounding-state Tensor. For $|x|=k+f$ with integer $k$ and $0\le f<1$, the implementation adds one to $k$ when a 32-bit random word is below $\lfloor 2^{32}f\rfloor$, then restores the sign of $x$. The rounding stream advances during execution.

`embed_coefficients` multiplies by the inverse twister, applies the corresponding inverse FFT, and restores slot order. `decode_slots` divides the resulting values by the actual scale $\Delta$. The codec's use of Torch FFT with `norm="forward"` is part of the implemented normalization; changing it requires compensating the embedding equations.

`_codec.py` composes these numerical functions, `_implementation.py` supplies registered encode/decode and integer-to-RNS operations, and `_periodic.py` handles compact periodic plaintext preparation. The permutation, twister, and rounding-state arrays are ordinary material operands.

## How do coefficients enter and leave the residue basis?

Integer-to-RNS conversion reduces an integer coefficient into the ordered supplied prime rows. Centered lifting must retain the sign convention before reduction. After decryption, `_decryption.py` reconstructs the centered coefficient class modulo the full active Q product using mixed-radix data and half-product comparison. Multiple prime rows cannot generally be replaced by reading one convenient residue.

The mathematical message must remain within the intended centered range modulo Q; otherwise modular wrap changes the recovered value. FFT approximation, stochastic quantization, encryption noise, key switching, and rescale contribute different errors. Decode with the ciphertext's actual scale rather than an assumed prime width or configuration default.

The [compressed plaintext article](compressed-plaintext-internals.md) explains encoded-axis layouts, which are distinct from visible slot repetition. The [RNS article](rns-and-ntt.md) defines row ordering and representation transitions.

## Which random streams are used?

`rng.Csprng` configures CKKS sampling through the `triton-csprng` package's ChaCha20 generators and RNS stream interface. It supplies coefficient counts, channel counts, repeated channels, integral dtype, and discrete-Gaussian standard deviation for CPU/CUDA streams.

Secret-key generation samples ternary coefficients. Public-key generation samples uniform components and Gaussian error. Encryption samples a binary masking polynomial and Gaussian errors. Encoding consumes its mutable rounding-state Tensor. These uses have distinct distributions and must retain their intended stream advancement.

Use fixed seeds for reproducible experiments. Cryptographic use requires suitable seed entropy and unique nonces for independent streams under the same key. Reusing a seed/nonce or restoring the same stream state can repeat samples. CKKS security estimates depend on ring dimension, modulus chain, secret distribution, and error parameters.

## What relation does each key represent?

`KeyGenerationResource` groups supplied RNS/NTT contexts, sampler, and numerical construction tables. `CkksKeyGenerator` exposes key construction as data provision and returns public key value classes.

| Value | Relation or purpose | Stored Tensor axes |
| --- | --- | --- |
| Secret key | Ternary polynomial $s$ | `[limb, ntt_index]` |
| Public key | $k_0+k_1s=e$ modulo the selected basis | `[key_component=2, limb, ntt_index]` |
| Hybrid key-switch key | Digit relation from $s_{\rm src}$ to $s_{\rm dst}$ | `[key_digit, key_component=2, QP_limb, ntt_index]` |
| Relinearization key | Switch the $s^2$ term to $s$ | Hybrid key-switch layout |
| Rotation/conjugation key | Switch the automorphed secret relation to the target secret | Hybrid key-switch layout with operation identity |

Secret and public key payloads are constructed at depth zero in NTT/Montgomery representation. Hybrid key-switch keys use depth-zero QP rows and stable key-digit identities. For digit $d$, the embedded source rows satisfy

$$
k_{d,0}+k_{d,1}s_{\mathrm{dst}}=P s_{\mathrm{src}}+e_d,
$$

where $P$ is the auxiliary-basis product. The source term is present only in the digit's embedded Q rows. Active-depth digit selection preserves the stable key-digit mapping.

## How does encryption use the public key?

For encoded message polynomial $m$, binary mask $v$, and sampled errors $e_0,e_1$, encryption constructs

$$
c_0=vk_0+m+e_0,\qquad c_1=vk_1+e_1.
$$

`crypto/_encryption.py` performs the supplied NTT transitions and modular products. It consumes public-key and parameter/table Tensors plus a bound sampler handle. Coefficient output performs the inverse transition and standard-residue addition; NTT output retains the selected NTT/Montgomery arithmetic form.

`crypto/_decryption.py` evaluates $c_0+c_1s$ or $c_0+c_1s+c_2s^2$, converts the phase to coefficient/standard RNS, and reconstructs centered Q coefficients. Slot decoding then applies the inverse scale and embedding.

## How is data provision connected to execution?

Engine factories select device services and invoke key construction. Existing keys can be supplied to Eager or bound as Tensor materials in a Compilation. Capture retains actual supplied key dataflow; optional material preparation can choose among caller-provided keys but never creates an absent evaluation key.

Extend codecs beside their table and embedding owners, and extend key constructions beside the relevant secret relation. Preserve sample distributions, scale and row conventions, and the declared Tensor ABI. For persistence, review included material symbols carefully: [Compilation files](compilation-persistence.md) are unencrypted and can contain secret or stream-state data when requested.
