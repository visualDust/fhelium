# Benchmark methodology

The FHElium benchmark suite compares complete workloads, CKKS operations, and
RNS/NTT execution across hardware. It measures the same task definitions at
sampled input depths and batch sizes, with correctness checks before and after
timing. There is no aggregate hardware score.

[Explore results](/benchmarks/) · [Run and submit results](/benchmarks/run-and-submit)

## What is measured

| Layer | Tasks | Timed work |
| --- | --- | --- |
| Complete workloads | PT×CT and CT×CT matrix multiplication; polynomial evaluation | The complete prepared computation, including its rotations, products and rescaling |
| CKKS operations | Rotation, relinearization, rescale, arithmetic and complete key switching | An Eager call or the stated operation composition, including host dispatch and all native execution it launches |
| RNS arithmetic | Modular addition and Montgomery multiplication | A functional Backend operation on one polynomial across the active Q basis |
| NTT | Complete forward and inverse negacyclic transforms | The complete selected transform, including output allocation; inverse includes an input-to-output copy |

An NTT result is not one butterfly stage or one isolated GPU kernel. RNS/NTT
speedups do not imply equal improvements in matrix multiplication or polynomial
evaluation. CPU/GPU comparisons include their host execution environment,
thread settings and recorded device power limits; they are not controlled
experiments isolating GPU architecture alone.

## Configuration and sampling

The current suite uses `slots32768_scale50_depth27_int64`: ring dimension
$N=65{,}536$, default scale $2^{50}$ and maximum depth 27. The report embeds the
complete Q depth groups and P primes, rather than relying on the preset name
as their identity. Depth selects the active Q-group suffix; it is not a count
of multiplications already performed.

| Dimension | Values |
| --- | --- |
| Entry depth | 0, 4, 8, 12, 16, 20, 24 |
| Batch | 1, 2, 4, 8, 16 independent tasks |
| Matrix size | 16×16, 32×32, 64×64 |
| Matrix rotation plan | Independent rotations or shared baby-rotation preparation (hoisting) |
| Polynomial method | Balanced, Horner, Paterson–Stockmeyer with group size four |
| NTT direction | Forward or inverse |
| NTT implementation | Indexed radix-2 stages; compact grouped radix-2 stages with a shared-memory tail |

These dimensions form the defined products, not unrelated one-dimensional
sweeps. Polynomial cells are included only where the selected method's depth
requirement fits. The current full inventory has **1,165 cells**: 420 matrix,
420 CKKS operation, 80 polynomial and 245 additional NTT/key-switch/RNS cells.
One run record represents one suite execution, not one configuration.

Every-fourth-depth sampling limits execution time while showing changes as the
active basis shrinks. Unsampled depths have no measurements; connecting lines
are visual guides, not inferred data or a guarantee that local performance
steps have been captured.

## Workload definitions

### Matrix multiplication

Each batch member computes an independently generated $C_b=A_bB_b$. Columns
are packed using two periodic copies. The baby-step/giant-step decomposition
uses the largest power of two not exceeding the square root of matrix size.
The suite retains prepared NTT diagonals. Each giant group is rescaled; CT×CT
products are relinearized once per group. The two rotation plans differ in
whether baby rotations share preparation, while computing the same matrix
product.

### Polynomial evaluation

The task is one dense degree-twelve power polynomial on $[-1,1]$. All methods
use the same coefficients, stored in ascending order in the report. Balanced
and Paterson–Stockmeyer require five depths, and Horner requires twelve in the
current implementation. The reference is evaluated with NumPy `longdouble`
arithmetic and converted to binary64 for comparison; its precision depends on
the platform. This measures the polynomial itself, not its approximation error
to a named activation function.

### CKKS operations

The operation inventory distinguishes a ciphertext self-product from a product
of distinct encrypted inputs. Multiplication alone does not include
relinearization or rescale. Rescale starts from a prepared two-component
ciphertext. The eight-term plaintext-product sum is measured as one operation;
scalar multiplication and depth advancement include their stated rescale.

Rotate-many computes seven offsets from one input. Its throughput unit is an
input group producing seven outputs, not seven independent input tasks.
Both rotate-many plans start from coefficient-domain input. A single rotation
starts in NTT form. Complete key switching uses distinct source/destination
secrets, coefficient-domain input and NTT output; its ModUp, transforms,
key-product accumulation and ModDown are all timed.

### RNS and NTT

Each task processes one polynomial across every active Q prime. RNS inputs are
dense residues with independent integer formulas for modular addition and
Montgomery multiplication. NTT inputs represent
$a_b(X)=(b+1)+X-X^2$; independent modular evaluation at bit-reversed odd roots
supplies the expected transform. Inverse input is this analytic transform,
not the result of a forward implementation under test. The arithmetic schedules
do not branch on these coefficient values.

The grouped NTT route combines radix-2 stages and a shared-memory tail. Its
name does not mean every stage uses a uniform radix eight. It is implemented
for CUDA only; the CPU records its cells as unsupported. Indexed radix-2
supports both CPU and CUDA.

## Timing and qualification

- Five warmup calls precede twenty timed samples.
- Wall-clock timing brackets the complete call. CUDA is synchronized before
  the timer starts and after the call returns.
- Context construction, keys, encoding/encryption, diagonal preparation,
  transfers and decryption are excluded. Inputs and required materials are
  prepared before timing; polynomial constants are reused between calls.
- Functional output allocation is included. Each timed output is released
  after synchronization, outside that sample's timed interval.
- CKKS outputs must be finite and satisfy maximum absolute error $\le10^{-5}$
  before and after sampling. RNS/NTT outputs must match every canonical integer
  residue exactly. Tolerances are not adjusted for hardware or failures.
- The runner preserves the invoking environment's PyTorch thread settings
  unless the caller supplies `--threads`. Effective intra-op/inter-op counts
  and relevant thread environment settings are recorded.

Do not overlap controlled measurements with builds or other work on the same
execution resources. Independent hosts may run concurrently. Record shared-host
conditions, other activity and intentional power or clock settings. A scheduler
allocation alone does not prove the host or every GPU is idle.

## Reading the plots

Both views use batch on the horizontal axis:

- **Scaling** shows median batch latency divided by batch size, in ms/task.
- **Throughput** shows batch size divided by median batch latency, in tasks/s.

Error bars show the interquartile range. Throughput bounds invert the latency
quartiles. Each hardware curve highlights its best measured point: lowest
amortized latency or highest throughput. CPU/GPU buttons filter displayed
hardware; Log Y changes only the axis scale.

When Grouped stages is selected, GPU curves use that NTT route and the CPU curve
uses its available Staged radix-2 route. The info tooltip identifies this mixed
implementation comparison. Selecting Staged radix-2 compares that route across
all supported hardware.

Missing points are not zero performance. Capacity limits, unsupported
implementations and unmeasured combinations are explained in the info tooltip.
Only correctness-qualified measurements enter performance curves. A failed or
interrupted run is diagnostic evidence, not a completed comparison record.

## Memory and provenance

CUDA reports include resident, peak and additional temporary PyTorch allocated
bytes around the timed calls. These counters do not measure all driver or
system memory. CPU allocation counters are unavailable and remain null.
Required evaluation-key bytes describe the task's key tensors; they are not a
claim about total process memory.

Reports retain individual samples, task identities, exact parameters, software
and hardware fields, source identity where available, and a hash of the suite
definition. A matching name alone does not establish comparable measurements.
The package version may remain `0.20.0` when a report was collected from a
source checkout; use its commit and tracked-diff identity as well. Historical
records retain their original source and version identities rather than being
relabeled to a planned release.
When definitions change, reuse requires review of the unchanged task and timing
semantics; old results are not silently relabeled.

Additional measurements remain separately dated reports attached to their
hardware record. The website shows qualified summaries, not raw log downloads.
See [Run and submit results](/benchmarks/run-and-submit) for collection and review.

