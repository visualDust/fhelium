# Run and submit a benchmark

Run the complete suite on an available device and submit its JSON report through
a GitHub issue. Maintainers review the measurement conditions and validate the
report before adding its qualified results to the website.

[Methodology](/benchmarks/methodology) ·
[Submit results](https://github.com/visualDust/fhelium/issues/new?template=benchmark_results.yml)

## 1. Prepare the environment

Follow the [installation instructions](/tutorial/installation). Use the same environment
for installing and running FHElium. Record the release or source commit; if you
have modified code affecting the benchmark, describe those changes in the issue.
An installed package may not contain Git checkout identity, so provide the
source commit yourself when applicable.

Choose a device without competing computation. Avoid building, testing or
running another benchmark on the same resources during measurement. Different
physical hosts can run independently. On a shared cluster, obtain a compute
allocation first and follow the site's scheduling rules.

The full suite covers matrix products, polynomial evaluation, CKKS operations
and RNS/NTT calls. It can take substantially longer on CPU than GPU. You do not
need to generate keys or supply application inputs; the runner prepares them.

## 2. Run one complete suite per device

```bash
# CPU
fhelium benchmark run --device cpu --output benchmark-cpu.json

# One GPU; choose the visible CUDA device index
fhelium benchmark run --device cuda:0 --output benchmark-gpu.json
```

Choose the relevant command; do not run both simultaneously on a shared host
for a controlled comparison. Each command writes one record for the full suite,
including cells unavailable on that device.

By default, the runner preserves PyTorch's thread settings. If you intentionally
want a different intra-op thread count, supply it and include the command in the
submission:

```bash
fhelium benchmark run --device cpu --threads 16 --output benchmark-cpu.json
```

There is no recommended universal thread cap. The report records effective
intra-op/inter-op counts and relevant environment variables.

## 3. Check the outcome

The CLI prints progress and a final status/count summary. The JSON is also a
checkpoint updated during execution, so its existence alone does not mean the
run completed.

| Result | Submission use |
| --- | --- |
| `status: completed`, `coverage: full` | Candidate for performance publication |
| `capacity` cells | Valid capacity exclusions; retain them without timing values |
| `unsupported` cells | Valid implementation exclusions, such as grouped NTT on CPU |
| `status: failed` or `interrupted`, or pending cells | Submit for diagnosis, not as a complete performance run |

Keep all task results and timing samples. Do not delete slow or failed cells,
change tolerance values, or edit the specification hash to make a report pass
validation. If you rerun, use a new output filename so the earlier evidence is
not overwritten. Checkpoint files preserve progress for inspection; the CLI
does not currently resume an interrupted run from them.

## 4. Review and package the report

The primary artifact is the JSON produced by `--output`. It contains the task
inventory, parameters, timing samples, correctness results, platform information
and source metadata where available. It does not need to include generated keys
or application data.

Review the artifact before publishing it. Machine/software details, source
identity and error messages can reveal information you do not want public;
error text may include local paths. Do not attach environment dumps, secrets,
private source patches or keys. If required information cannot be disclosed,
ask maintainers before publishing rather than silently changing the report.

Package the original JSON in a ZIP using Python's standard library:

```bash
python -m zipfile -c benchmark.zip benchmark-gpu.json
```

For a CPU result, substitute `benchmark-cpu.json`. One submission can include
multiple device reports, each retaining its own run identity. Logs are optional
for diagnosing an error; they are not required for successful performance
submissions. If included, review them for private information first.

## 5. Open a results issue

Use the [Benchmark results template](https://github.com/visualDust/fhelium/issues/new?template=benchmark_results.yml)
and drag `benchmark.zip` into its attachment field. Do not paste the JSON into
the issue body. If the attachment exceeds GitHub's current limit, provide a
stable downloadable archive link and its SHA-256 instead.

The form asks only for information the report cannot establish by itself:

- the command and intentional environment/thread overrides;
- whether the host/device was shared or other work overlapped;
- power/clock changes and relevant deployment conditions;
- release/commit and any benchmark-affecting source modifications;
- whether the report is a completed run or diagnostic evidence.

Hardware specifications and individual scores are already in the report and
need not be transcribed. Submitting an issue does not automatically publish a
result. Reports are reviewed as data; maintainers do not execute attached scripts
or binaries as part of importing results.

## How results enter the website

Maintainers check the original report's inventory, suite identity, qualification,
sampling and stated execution conditions. Capacity and unsupported cells remain
visible; failures and incomplete runs do not become performance points.
Cross-platform comparisons require compatible task definitions. Raw samples and
submitted logs stay with the evidence, while the website receives a compact
projection. Accepted catalog changes are reviewed through a pull request.

For maintainers working in the source checkout, the projection command is:

```bash
python scripts/publish_benchmarks.py benchmark-gpu.json benchmark-cpu.json \
  --output docs/.vitepress/theme/benchmarks/explorer/measuredRuns.json \
  --pages docs/benchmarks/records
```

The command **replaces** the destination catalog with the listed reports; it does
not append automatically. Include the complete intended catalog inputs and
review the diff so existing hardware records are not accidentally removed.
The reports compared by this command must use the same measurement definition.
Validation establishes structural and numerical qualification, not the truth of
all externally reported experimental conditions.

When a device is measured again for additional tasks, preserve a separate report.
`--supplement RUN_ID REPORT` attaches qualified partial measurements to an existing
hardware page with their own date and source identity. The public run command
always runs the full suite; contributors should submit complete reports rather
than editing one into a partial report. Maintainer-directed supplements require
a reviewed measurement plan and do not rewrite the original run.

## Independent investigations

A specialist workload run is useful for tuning, but does not replace the full
suite submission:

```bash
fhelium benchmark list
fhelium benchmark workload NAME --profile PROFILE --output investigation.json
```

Use profiles shown by the current checkout. See
[Screen NTT backends](/how-to/screen-ntt-backends) and
[Analyze and choose an NTT backend](/how-to/choose-ntt-backend) for focused NTT
investigations. Custom workload registration is documented in the
[benchmark framework API](/api/fhelium/benchmarks/model).
