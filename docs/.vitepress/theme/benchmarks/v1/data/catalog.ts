import {
  benchmarkV1CaseDefinitions,
  benchmarkV1CaseIds,
  benchmarkV1ManifestSha256,
  isBenchmarkV1CaseId,
  type BenchmarkV1CaseId,
} from './specification'

export const benchmarkVersion = 'v1' as const

export type BenchmarkScalar = boolean | number | string | null
export type BenchmarkJson = BenchmarkScalar | BenchmarkJson[] | BenchmarkJsonObject
export interface BenchmarkJsonObject { [name: string]: BenchmarkJson }
export type BenchmarkDirection = 'higher' | 'lower' | 'none'
export type BenchmarkCaseStatus = 'measured' | 'unavailable'

export interface BenchmarkMetric {
  category: 'latency' | 'throughput' | 'memory'
  dimensions: Record<string, BenchmarkScalar>
  direction: BenchmarkDirection
  name: string
  statistic: string
  unit: string
  value: number
}

export interface BenchmarkCorrectness {
  comparison: string
  details: BenchmarkJsonObject
  limit: BenchmarkScalar
  metric: string
  name: string
  observed: BenchmarkScalar
  oracle: string
  passed: boolean
  unit: string
}

export interface BenchmarkTimedBoundary {
  description: string
  excludes: string[]
  id: string
  includes: string[]
  synchronization: string
}

export interface BenchmarkUnavailable {
  details: BenchmarkJsonObject
  reason: string
}

export interface BenchmarkCase {
  benchmark: string
  benchmark_context: BenchmarkJsonObject | null
  category: string
  comparison: BenchmarkJsonObject
  correctness: BenchmarkCorrectness[]
  effective_parameters: BenchmarkJsonObject
  id: BenchmarkV1CaseId
  metadata: BenchmarkJsonObject | null
  metrics: BenchmarkMetric[]
  profile: string
  requirements: BenchmarkJsonObject
  status: BenchmarkCaseStatus
  timed_boundary: BenchmarkTimedBoundary | null
  title: string
  unavailable: BenchmarkUnavailable | null
  workload_id: string
}

export interface BenchmarkHighlight {
  case_id: BenchmarkV1CaseId
  direction: BenchmarkDirection
  id: string
  label: string
  unit: string
  value: BenchmarkScalar
}

export interface BenchmarkBuildIdentity {
  commit: string
  dirty: boolean
  native: BenchmarkJsonObject
  version: string
}

export interface BenchmarkPlatform {
  cpu: BenchmarkJsonObject
  cuda: BenchmarkJsonObject
  environment: BenchmarkJsonObject
  memory: BenchmarkJsonObject
  python: BenchmarkJsonObject
  system: BenchmarkJsonObject
  torch: BenchmarkJsonObject
}

export interface BenchmarkCaseCounts {
  failed: number
  interrupted: number
  measured: number
  unavailable: number
}

export interface BenchmarkV1Run {
  case_counts: BenchmarkCaseCounts
  cases: BenchmarkCase[]
  fhelium: BenchmarkBuildIdentity
  execution: {
    backend: 'cpu' | 'cuda'
    device: string
  }
  highlights: BenchmarkHighlight[]
  id: string
  manifest_sha256: string
  platform: BenchmarkPlatform
  published_at: string
  raw_path: string
  raw_sha256: string
  recorded_at: string
  slug: string
  status: 'completed'
}

export interface BenchmarkCatalog {
  benchmark_version: typeof benchmarkVersion
  generated_at: string | null
  runs: BenchmarkV1Run[]
}

function fail(message: string): never {
  throw new Error(`Invalid benchmark catalog: ${message}`)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function record(value: unknown, field: string): Record<string, unknown> {
  return isRecord(value) ? value : fail(`${field} must be an object`)
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  field: string,
): void {
  const actual = Object.keys(value).sort()
  const wanted = [...expected].sort()
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    fail(`${field} fields differ: expected ${wanted.join(', ')}`)
  }
}

function text(value: unknown, field: string, allowEmpty = false): string {
  if (
    typeof value !== 'string' ||
    value !== value.trim() ||
    (!allowEmpty && value.length === 0)
  ) {
    fail(`${field} must be a${allowEmpty ? '' : ' non-empty'} trimmed string`)
  }
  return value
}

function scalar(value: unknown, field: string): BenchmarkScalar {
  if (
    value === null ||
    typeof value === 'boolean' ||
    typeof value === 'string'
  ) return value
  if (typeof value === 'number' && Number.isFinite(value)) return value
  return fail(`${field} must be a finite JSON scalar`)
}

function jsonValue(value: unknown, field: string): BenchmarkJson {
  if (Array.isArray(value)) {
    return value.map((item, index) => jsonValue(item, `${field}[${index}]`))
  }
  if (isRecord(value)) {
    const result: BenchmarkJsonObject = {}
    for (const [name, item] of Object.entries(value)) {
      result[text(name, `${field} key`)] = jsonValue(item, `${field}.${name}`)
    }
    return result
  }
  return scalar(value, field)
}

function jsonObject(value: unknown, field: string): BenchmarkJsonObject {
  const parsed = jsonValue(record(value, field), field)
  if (!isRecord(parsed)) fail(`${field} must be an object`)
  return parsed as BenchmarkJsonObject
}

function nullableObject(value: unknown, field: string): BenchmarkJsonObject | null {
  return value === null ? null : jsonObject(value, field)
}

function integer(value: unknown, field: string, minimum = 0): number {
  if (!Number.isInteger(value) || typeof value !== 'number' || value < minimum) {
    fail(`${field} must be an integer greater than or equal to ${minimum}`)
  }
  return value
}

function isoDate(value: unknown, field: string): string {
  const result = text(value, field)
  if (Number.isNaN(Date.parse(result))) fail(`${field} must be an ISO-8601 timestamp`)
  return result
}

function direction(value: unknown, field: string): BenchmarkDirection {
  if (value !== 'higher' && value !== 'lower' && value !== 'none') {
    fail(`${field} must be higher, lower, or none`)
  }
  return value
}

function stringArray(value: unknown, field: string): string[] {
  if (!Array.isArray(value)) fail(`${field} must be an array`)
  return value.map((item, index) => text(item, `${field}[${index}]`))
}

function parseMetric(value: unknown, field: string): BenchmarkMetric {
  const metric = record(value, field)
  exactKeys(metric, ['category', 'dimensions', 'direction', 'name', 'statistic', 'unit', 'value'], field)
  if (metric.category !== 'latency' && metric.category !== 'throughput' && metric.category !== 'memory') {
    fail(`${field}.category must be latency, throughput, or memory`)
  }
  const dimensionsValue = record(metric.dimensions, `${field}.dimensions`)
  const dimensions: Record<string, BenchmarkScalar> = {}
  for (const [name, item] of Object.entries(dimensionsValue)) {
    dimensions[text(name, `${field}.dimensions key`)] = scalar(item, `${field}.dimensions.${name}`)
  }
  if (typeof metric.value !== 'number' || !Number.isFinite(metric.value)) {
    fail(`${field}.value must be a finite number`)
  }
  return {
    category: metric.category,
    dimensions,
    direction: direction(metric.direction, `${field}.direction`),
    name: text(metric.name, `${field}.name`),
    statistic: text(metric.statistic, `${field}.statistic`),
    unit: text(metric.unit, `${field}.unit`, true),
    value: metric.value,
  }
}

function parseCorrectness(value: unknown, field: string): BenchmarkCorrectness {
  const item = record(value, field)
  exactKeys(item, ['comparison', 'details', 'limit', 'metric', 'name', 'observed', 'oracle', 'passed', 'unit'], field)
  if (item.passed !== true) fail(`${field}.passed must be true for a published Benchmark v1 run`)
  return {
    comparison: text(item.comparison, `${field}.comparison`),
    details: jsonObject(item.details, `${field}.details`),
    limit: scalar(item.limit, `${field}.limit`),
    metric: text(item.metric, `${field}.metric`),
    name: text(item.name, `${field}.name`),
    observed: scalar(item.observed, `${field}.observed`),
    oracle: text(item.oracle, `${field}.oracle`),
    passed: item.passed,
    unit: text(item.unit, `${field}.unit`, true),
  }
}

function parseTimedBoundary(value: unknown, field: string): BenchmarkTimedBoundary {
  const item = record(value, field)
  exactKeys(item, ['description', 'excludes', 'id', 'includes', 'synchronization'], field)
  return {
    description: text(item.description, `${field}.description`),
    excludes: stringArray(item.excludes, `${field}.excludes`),
    id: text(item.id, `${field}.id`),
    includes: stringArray(item.includes, `${field}.includes`),
    synchronization: text(item.synchronization, `${field}.synchronization`),
  }
}

function parseUnavailable(value: unknown, field: string): BenchmarkUnavailable | null {
  if (value === null) return null
  const item = record(value, field)
  exactKeys(item, ['details', 'reason'], field)
  return {
    details: jsonObject(item.details, `${field}.details`),
    reason: text(item.reason, `${field}.reason`),
  }
}

function parseCase(value: unknown, field: string): BenchmarkCase {
  const item = record(value, field)
  exactKeys(item, [
    'benchmark', 'benchmark_context', 'category', 'comparison', 'correctness',
    'effective_parameters', 'id', 'metadata', 'metrics', 'profile',
    'requirements', 'status', 'timed_boundary', 'title', 'unavailable',
    'workload_id',
  ], field)
  if (item.status !== 'measured' && item.status !== 'unavailable') {
    fail(`${field}.status must be measured or unavailable`)
  }
  if (!Array.isArray(item.metrics)) fail(`${field}.metrics must be an array`)
  if (!Array.isArray(item.correctness)) fail(`${field}.correctness must be an array`)
  const metrics = item.metrics.map((metric, index) => parseMetric(metric, `${field}.metrics[${index}]`))
  const correctness = item.correctness.map((check, index) => parseCorrectness(check, `${field}.correctness[${index}]`))
  const unavailable = parseUnavailable(item.unavailable, `${field}.unavailable`)
  const benchmarkContext = nullableObject(item.benchmark_context, `${field}.benchmark_context`)
  const metadata = nullableObject(item.metadata, `${field}.metadata`)
  const timedBoundary = item.timed_boundary === null ? null : parseTimedBoundary(item.timed_boundary, `${field}.timed_boundary`)
  const id = text(item.id, `${field}.id`)
  if (!isBenchmarkV1CaseId(id)) fail(`${field}.id is not a Benchmark v1 case`)
  const definition = benchmarkV1CaseDefinitions[id]
  if (item.benchmark !== definition.benchmark) fail(`${field}.benchmark differs from the Benchmark v1 case definition`)
  if (item.profile !== definition.profile) fail(`${field}.profile differs from the Benchmark v1 case definition`)
  if (item.workload_id !== definition.workloadId) fail(`${field}.workload_id differs from the Benchmark v1 case definition`)
  const requirements = jsonObject(item.requirements, `${field}.requirements`)
  const effectiveParameters = jsonObject(item.effective_parameters, `${field}.effective_parameters`)
  if (item.status === 'measured' && (unavailable !== null || metrics.length === 0 || correctness.length === 0 || timedBoundary === null || benchmarkContext === null || metadata === null)) {
    fail(`${field} measured case must have metrics, correctness, timing, context, metadata, and no unavailability`)
  }
  if (item.status === 'unavailable' && (unavailable === null || metrics.length !== 0 || correctness.length !== 0 || timedBoundary !== null || benchmarkContext !== null || metadata !== null)) {
    fail(`${field} unavailable case must have only unavailability evidence`)
  }
  return {
    benchmark: text(item.benchmark, `${field}.benchmark`),
    benchmark_context: benchmarkContext,
    category: text(item.category, `${field}.category`),
    comparison: jsonObject(item.comparison, `${field}.comparison`),
    correctness,
    effective_parameters: effectiveParameters,
    id,
    metadata,
    metrics,
    profile: text(item.profile, `${field}.profile`),
    requirements,
    status: item.status,
    timed_boundary: timedBoundary,
    title: text(item.title, `${field}.title`),
    unavailable,
    workload_id: text(item.workload_id, `${field}.workload_id`),
  }
}

function parseHighlight(value: unknown, field: string): BenchmarkHighlight {
  const item = record(value, field)
  exactKeys(item, ['case_id', 'direction', 'id', 'label', 'unit', 'value'], field)
  const caseId = text(item.case_id, `${field}.case_id`)
  if (!isBenchmarkV1CaseId(caseId)) fail(`${field}.case_id is not a Benchmark v1 case`)
  return {
    case_id: caseId,
    direction: direction(item.direction, `${field}.direction`),
    id: text(item.id, `${field}.id`),
    label: text(item.label, `${field}.label`),
    unit: text(item.unit, `${field}.unit`, true),
    value: scalar(item.value, `${field}.value`),
  }
}

function parseRun(value: unknown, index: number): BenchmarkV1Run {
  const field = `runs[${index}]`
  const run = record(value, field)
  exactKeys(run, [
    'case_counts', 'cases', 'execution', 'fhelium', 'highlights', 'id', 'manifest_sha256',
    'platform', 'published_at', 'raw_path', 'raw_sha256', 'recorded_at', 'slug',
    'status',
  ], field)
  if (run.status !== 'completed') fail(`${field}.status must be completed`)
  const digest = text(run.raw_sha256, `${field}.raw_sha256`)
  if (!/^[0-9a-f]{64}$/u.test(digest)) fail(`${field}.raw_sha256 must be a SHA-256 digest`)
  if (run.id !== digest) fail(`${field}.id must equal raw_sha256`)
  if (run.slug !== `sha256-${digest}`) fail(`${field}.slug does not match the report digest`)
  if (run.raw_path !== `/benchmarks/v1/runs/sha256-${digest}.json`) fail(`${field}.raw_path does not match the report digest`)
  if (run.manifest_sha256 !== benchmarkV1ManifestSha256) fail(`${field}.manifest_sha256 differs from the Benchmark v1 specification`)

  const build = record(run.fhelium, `${field}.fhelium`)
  exactKeys(build, ['commit', 'dirty', 'native', 'version'], `${field}.fhelium`)
  if (typeof build.dirty !== 'boolean') fail(`${field}.fhelium.dirty must be boolean`)
  const execution = record(run.execution, `${field}.execution`)
  exactKeys(execution, ['backend', 'device'], `${field}.execution`)
  if (execution.backend !== 'cpu' && execution.backend !== 'cuda') fail(`${field}.execution.backend must be cpu or cuda`)
  if (execution.backend === 'cpu' && execution.device !== 'cpu') fail(`${field}.execution.device must be cpu for CPU execution`)
  if (execution.backend === 'cuda' && (typeof execution.device !== 'string' || !/^cuda:\d+$/u.test(execution.device))) fail(`${field}.execution.device must be an indexed cuda:N device`)
  const platform = record(run.platform, `${field}.platform`)
  exactKeys(platform, ['cpu', 'cuda', 'environment', 'memory', 'python', 'system', 'torch'], `${field}.platform`)
  const counts = record(run.case_counts, `${field}.case_counts`)
  exactKeys(counts, ['failed', 'interrupted', 'measured', 'unavailable'], `${field}.case_counts`)
  if (!Array.isArray(run.cases)) fail(`${field}.cases must be an array`)
  if (!Array.isArray(run.highlights)) fail(`${field}.highlights must be an array`)
  const cases = run.cases.map((item, caseIndex) => parseCase(item, `${field}.cases[${caseIndex}]`))
  if (
    cases.length !== benchmarkV1CaseIds.length ||
    cases.some((item, caseIndex) => item.id !== benchmarkV1CaseIds[caseIndex])
  ) {
    fail(`${field}.cases must contain the five Benchmark v1 case ids in canonical order`)
  }
  const caseIds = new Set(cases.map((item) => item.id))
  const caseCounts: BenchmarkCaseCounts = {
    failed: integer(counts.failed, `${field}.case_counts.failed`),
    interrupted: integer(counts.interrupted, `${field}.case_counts.interrupted`),
    measured: integer(counts.measured, `${field}.case_counts.measured`),
    unavailable: integer(counts.unavailable, `${field}.case_counts.unavailable`),
  }
  if (caseCounts.measured !== cases.filter((item) => item.status === 'measured').length || caseCounts.unavailable !== cases.filter((item) => item.status === 'unavailable').length) {
    fail(`${field}.case_counts does not match cases`)
  }
  if (caseCounts.failed !== 0 || caseCounts.interrupted !== 0) fail(`${field} completed publication cannot contain failed/interrupted cases`)
  const highlights = run.highlights.map((item, highlightIndex) => parseHighlight(item, `${field}.highlights[${highlightIndex}]`))
  if (highlights.some((item) => !caseIds.has(item.case_id))) fail(`${field}.highlights references an unknown case`)

  return {
    case_counts: caseCounts,
    cases,
    fhelium: {
      commit: text(build.commit, `${field}.fhelium.commit`),
      dirty: build.dirty,
      native: jsonObject(build.native, `${field}.fhelium.native`),
      version: text(build.version, `${field}.fhelium.version`),
    },
    execution: {
      backend: execution.backend,
      device: execution.device as string,
    },
    highlights,
    id: digest,
    manifest_sha256: text(run.manifest_sha256, `${field}.manifest_sha256`),
    platform: {
      cpu: jsonObject(platform.cpu, `${field}.platform.cpu`),
      cuda: jsonObject(platform.cuda, `${field}.platform.cuda`),
      environment: jsonObject(platform.environment, `${field}.platform.environment`),
      memory: jsonObject(platform.memory, `${field}.platform.memory`),
      python: jsonObject(platform.python, `${field}.platform.python`),
      system: jsonObject(platform.system, `${field}.platform.system`),
      torch: jsonObject(platform.torch, `${field}.platform.torch`),
    },
    published_at: isoDate(run.published_at, `${field}.published_at`),
    raw_path: text(run.raw_path, `${field}.raw_path`),
    raw_sha256: digest,
    recorded_at: isoDate(run.recorded_at, `${field}.recorded_at`),
    slug: text(run.slug, `${field}.slug`),
    status: 'completed',
  }
}

export function parseBenchmarkCatalog(value: unknown): BenchmarkCatalog {
  const catalog = record(value, 'root')
  exactKeys(catalog, ['benchmark_version', 'generated_at', 'runs'], 'root')
  if (catalog.benchmark_version !== benchmarkVersion) {
    fail(`benchmark_version must be ${benchmarkVersion}`)
  }
  if (!Array.isArray(catalog.runs)) fail('runs must be an array')
  const generatedAt = catalog.generated_at === null ? null : isoDate(catalog.generated_at, 'generated_at')
  const runs = catalog.runs.map(parseRun)
  if (new Set(runs.map((run) => run.id)).size !== runs.length) fail('runs contains duplicate report identities')
  return { benchmark_version: benchmarkVersion, generated_at: generatedAt, runs }
}
