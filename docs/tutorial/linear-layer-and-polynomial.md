# Linear layer and polynomial activation

**Example source:** [`examples/08_algorithms_linear_polynomial.py`](https://github.com/VisualDust/fhelium/blob/main/examples/08_algorithms_linear_polynomial.py)

This example evaluates one encrypted network layer $z=p(Wx)$: a dense $n\times n$ matrix-vector product with `fhelium.algorithms.linear`, followed by a Chebyshev approximation $p$ of GELU with `fhelium.algorithms.polynomial`. The same Python function runs directly on the Engine and under `fc.compile`, and the example compares their depth, scale, error, and latency.

## Run the example

```bash
python examples/08_algorithms_linear_polynomial.py
python examples/08_algorithms_linear_polynomial.py --device cuda:0 --features 32 --degree 31
```

`--features` sets the matrix width $n$, which must divide the slot count, and `--degree` sets the Chebyshev degree of the activation.

## 1. Represent the matrix as cyclic diagonals

The input vector $x\in\mathbb{R}^n$ is packed periodically, so slot $i$ holds $x_{i \bmod n}$. Rotation $\operatorname{Rot}_k$ follows `torch.roll(..., shifts=k)` and places $x_{(i-k) \bmod n}$ in slot $i$. With the cyclic diagonals

$$
d_k[i]=W_{i \bmod n,\,(i-k) \bmod n},\qquad k=0,\dots,n-1,
$$

every period of $n$ slots satisfies

$$
\sum_{k=0}^{n-1} d_k\odot\operatorname{Rot}_k(x)=Wx.
$$

`weight_diagonals` builds this mapping from rotation offset to slot vector, and `linear.LinearTransform(diagonals, slots, name="dense")` records it. The transform contains plaintext data only; it holds no keys or encoded plaintexts.

## 2. Plan the evaluation and provision keys

`linear.plan(transform)` returns a baby-step giant-step (BSGS) plan. With baby step $b$, each offset is written $k=gb+j$ with $0\le j<b$, and

$$
Wx=\sum_g \operatorname{Rot}_{gb}\Bigl(\sum_j \operatorname{Rot}_{-gb}(d_{gb+j})\odot\operatorname{Rot}_j(x)\Bigr),
$$

so the evaluator performs about $b+n/b$ rotations instead of $n-1$. The plan reports the offsets it rotates by, and the application creates exactly those keys:

```python
linear_plan = linear.plan(transform)
steps = linear_plan.required_rotation_offsets(transform)
keys = EvaluationKeySet(
    [engine.create_rotation_key(step, secret_key) for step in steps]
    + [engine.create_relinearization_key(secret_key)]
)
```

For $n=16$ the plan selects $b=4$ and six rotation keys. The relinearization key serves the ciphertext products of the polynomial.

## 3. Fit the activation

`polynomial.chebyshev_fit(gelu, degree)` interpolates GELU at Chebyshev nodes on $[-1,1]$ and returns a `Polynomial` with Chebyshev coefficients $c_m$:

$$
p(t)=\sum_{m=0}^{\text{degree}} c_m T_m(t).
$$

`max_error` reports the interpolation error sampled on the interval. The approximation is valid only for inputs in $[-1,1]$, so the example normalizes each row of $W$ to unit $\ell_1$ norm: $|x_j|\le 1$ then implies $|(Wx)_i|\le 1$. `polynomial.plan(activation)` selects the evaluation schedule, and `activation_plan.required_depths(activation)` reports the depth it consumes; the degree-15 schedule consumes five depths.

## 4. Prepare once, then evaluate

`linear.prepare` encodes the shifted diagonals $\operatorname{Rot}_{-gb}(d_{gb+j})$ as plaintexts at the depth and scale of the example input, and selects the keys for each rotation:

```python
prepared = linear.prepare(engine, x, transform, keys=keys)

def layer(value):
    hidden = linear.apply(engine, value, prepared=prepared)
    return polynomial.evaluate(engine, hidden, activation, activation_plan, keys=keys)
```

Diagonal encoding draws stochastic-rounding samples. Preparing outside `layer` encodes once and reuses the same plaintexts for every call, Eager or compiled. `linear.apply` consumes one depth: the plaintext products are followed by one rescale. `polynomial.evaluate` multiplies ciphertexts, relinearizes the products, and rescales according to its plan.

## 5. Compile the same function

`fc.compile(layer)` captures the Engine calls of `layer`, including those inside `linear.apply` and `polynomial.evaluate`. The prepared plaintexts and keys enter the compiled Program as captured Tensor data, so the compiled callable needs no key generation or encoding at call time. The first call compiles for the input's value state; later calls with the same state reuse the compiled function.

## 6. Read the output

The example prints the depth and actual scale after each step:

| value | depth | log2 scale |
| --- | --- | --- |
| input | 0 | 44.976 |
| $Wx$ | 1 | 40.008 |
| $p(Wx)$, Eager | 6 | 40.014 |
| $p(Wx)$, compiled | 6 | 40.014 |

These values are from the default `slots8192-scale40-depth7` preset, $n=16$, and degree 15. The linear layer consumes one depth and the activation five, so the layer uses six of the seven depths. Each rescale divides by the product of the primes in the dropped depth group, and the actual scales therefore stay near $2^{40}$ without equaling it. The Eager and compiled results have the same depth and scale; their maximum absolute error against the cleartext reference $\operatorname{GELU}(Wx)$ was about $2\times10^{-8}$ in both cases. The compiled and Eager ciphertexts need not be bit-identical, so the example compares them after decryption.

The final table reports the mean and minimum latency of repeated `layer(x)` and `compiled_layer(x)` calls with CUDA synchronization.

::: details Source
<<< @/../examples/08_algorithms_linear_polynomial.py
:::

## Related concepts and guides

- [Depth and scale](depth-and-scale.md)
- [Evaluator operation transitions](../concepts/ckks/evaluator-operation-transitions.md)
- [Provision the minimum required keyset](../how-to/provision-keyset.md)
- [Homogeneous batching](homogeneous-batching.md)
