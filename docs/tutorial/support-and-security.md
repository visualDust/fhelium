# Support, maturity, and security scope

Use this page to decide whether FHElium fits a research or evaluation task.
FHElium implements direct CKKS arithmetic and validates selected structural
and parameter requirements. It does not establish the security or numerical
correctness of an application protocol.

::: warning Active development
FHElium is under active development. APIs may change significantly between
releases. Record the exact FHElium, PyTorch, compiler, and—when applicable—CUDA
and driver versions used to produce a result.
:::

## What FHElium validates

FHElium validates operation-specific state before native execution, including
context identity, tensor device and dtype, CKKS level and actual scale,
polynomial domain, residue representation, RNS layout, component count, and
required key material.

When `enforce_security_budget=True`, `CkksEngine` also checks the complete
configured QP modulus against an exact published parameter row for the selected
ring dimension, security category, secret distribution, and error model. It
does not interpolate or extrapolate an unsupported parameter set.

These checks can reject an internally inconsistent or unsupported evaluator.
They do not prove that an application is secure, private, or numerically useful.
See [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md) for
parameter planning.

## What remains application-owned

The application is responsible for:

- defining the threat model and deciding whether CKKS is appropriate;
- generating, authorizing, transporting, storing, rotating, and deleting key
  material;
- bounding cleartext inputs and intermediate values;
- selecting an operation schedule with sufficient levels and precision;
- comparing decrypted results with a representative cleartext oracle and a
  justified error criterion;
- protecting serialized values and keys at rest and in transit;
- assessing host, process, dependency, CPU, GPU, driver, communication, and
  side-channel risks;
- determining whether each experimental capability is permitted by the
  intended use.

FHElium does not provide a key-management service, access-control system,
secure transport, encrypted artifact store, remote execution protocol, or
application threat model. Metadata such as an artifact sensitivity label does
not encrypt its payload.

## Numerical correctness is workload-specific

CKKS is approximate. A successful operation and a parameter-budget check do not
establish that the decrypted result meets an application's accuracy target.
Document the input range, circuit, scale and level schedule, cleartext
reference, observed error distribution, and acceptance threshold for each
workload. Do not weaken a threshold to hide an unexplained discrepancy.

## Before using private data

Confirm all of the following:

1. the scheme and parameters match a documented threat model;
2. key custody, transport, persistence, backup, rotation, and deletion have
   been reviewed;
3. the application has numerical bounds for packed inputs and circuit
   intermediates;
4. numerical error has been measured on representative data against a
   cleartext oracle;
5. the complete software, host, accelerator, communication, and side-channel
   environment has been assessed;
6. every experimental feature in the path is allowed by its own documented
   scope.

If these conditions are not established, use synthetic data and throwaway keys
only.

## Continue

- Build the selected native backend: [Installation](installation.md)
- Run one local evaluator: [Quickstart](tutorials.md)
- Plan parameters and depth: [Choose a preset and chain depth](../how-to/choose-preset-and-depth.md)
- Manage key lifecycles: [Key lifecycle](../concepts/ckks/key-lifecycle.md)
- Inspect exact value state: [Programming model](../concepts/programming-model.md)
