import { withBase } from 'vitepress'

import type {
  BenchmarkCase,
  BenchmarkCorrectness,
  BenchmarkDirection,
  BenchmarkJson,
  BenchmarkJsonObject,
  BenchmarkMetric,
  BenchmarkScalar,
  BenchmarkV1Run,
} from './catalog'
import type { BenchmarkV1CaseId } from './specification'
import {
  type BenchmarkV1ExecutionBackend,
} from './specification'

export type BenchmarkMetricCategory = 'latency' | 'throughput' | 'memory' | 'other'

export interface BenchmarkGpuDevice {
  computeCapability: string
  index: string
  memoryBusWidth: number | null
  multiProcessorCount: number | null
  name: string
  totalGlobalMem: number | null
}

const fixedLabels: Record<string, string> = {
  add: 'Add',
  add_plaintext: 'Add plaintext',
  coefficient_domain_to_ntt_domain: 'Coefficient → NTT',
  decrypt: 'Decrypt',
  encrypt: 'Encrypt',
  forward_ntt: 'Forward NTT',
  inverse_ntt: 'Inverse NTT',
  mod_switch_to_next_level: 'Modulus switch',
  multiply: 'CT × CT multiply',
  multiply_plaintext: 'PT × CT multiply',
  ntt_domain_to_coefficient_domain: 'NTT → coefficient',
  relinearize: 'Relinearize',
  rescale_to_next_level: 'Rescale',
  radix2_indexed: 'Radix-2 indexed reference',
  radix4_compact: 'Radix-4 compact',
  radix8_compact: 'Radix-8 compact',
  radix16_compact: 'Radix-16 compact',
  rotate_with_key: 'Direct rotation',
  roundtrip: 'Roundtrip',
}

const acronymLabels: Record<string, string> = {
  api: 'API',
  ckks: 'CKKS',
  ct: 'CT',
  ctct: 'CT × CT',
  cuda: 'CUDA',
  gpu: 'GPU',
  json: 'JSON',
  ntt: 'NTT',
  pt: 'PT',
  ptct: 'PT × CT',
  qp: 'QP',
  rns: 'RNS',
  smem: 'SMEM',
}

export function humanizeIdentifier(value: string): string {
  const fixed = fixedLabels[value]
  if (fixed) return fixed
  const compactBackend = /^radix(\d+)_compact_group(\d+)_smem(\d+)$/u.exec(value)
  if (compactBackend) {
    return `Radix-${compactBackend[1]} compact · group ${compactBackend[2]} · SMEM ${compactBackend[3]}`
  }
  const words = value
    .replace(/([a-z\d])([A-Z])/gu, '$1 $2')
    .split(/[-_.\s]+/u)
    .filter(Boolean)
    .map((part) => {
      const lower = part.toLocaleLowerCase()
      if (acronymLabels[lower]) return acronymLabels[lower]
      const radix = /^radix(\d+)$/u.exec(lower)
      if (radix) return `Radix-${radix[1]}`
      const group = /^group(\d+)$/u.exec(lower)
      if (group) return `group ${group[1]}`
      const smem = /^smem(\d+)$/u.exec(lower)
      if (smem) return `SMEM ${smem[1]}`
      if (/^d\d+$/u.test(lower)) return lower.toLocaleUpperCase()
      return lower
    })
  if (!words.length) return value
  return `${words[0].charAt(0).toLocaleUpperCase()}${words[0].slice(1)}${words.length > 1 ? ` ${words.slice(1).join(' ')}` : ''}`
}

export const displayIdentifier = humanizeIdentifier

const numberFormatter = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 6 })

export function formatScalar(value: BenchmarkScalar): string {
  if (typeof value === 'number') {
    const absolute = Math.abs(value)
    return absolute !== 0 && (absolute < 0.001 || absolute >= 1_000_000)
      ? value.toExponential(3)
      : numberFormatter.format(value)
  }
  if (value === null) return 'Not reported'
  return String(value)
}

export function formatMetric(metric: BenchmarkMetric): string {
  return `${formatScalar(metric.value)}${metric.unit ? ` ${metric.unit}` : ''}`
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'Not reported'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let amount = value
  let unit = 0
  while (Math.abs(amount) >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${new Intl.NumberFormat('en-US', { maximumFractionDigits: amount >= 100 ? 0 : amount >= 10 ? 1 : 2 }).format(amount)} ${units[unit]}`
}

export function formatDate(value: string, includeTime = false): string {
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    ...(includeTime ? { timeStyle: 'short' as const } : {}),
    timeZone: 'UTC',
  }).format(new Date(value))
}

export function directionLabel(direction: BenchmarkDirection): string {
  if (direction === 'lower') return 'lower is better'
  if (direction === 'higher') return 'higher is better'
  return 'reported without ranking'
}

export function directionArrow(direction: BenchmarkDirection): string {
  if (direction === 'lower') return '↓'
  if (direction === 'higher') return '↑'
  return ''
}

export function metricCategory(metric: BenchmarkMetric): BenchmarkMetricCategory {
  if (metric.category) return metric.category
  const declared = metric.dimensions.category
  if (declared === 'latency' || declared === 'throughput' || declared === 'memory') return declared
  if (['ns', 'us', 'ms', 's'].includes(metric.unit)) return 'latency'
  if (/^(?:bytes|KiB|MiB|GiB|KB|MB|GB)$/u.test(metric.unit)) return 'memory'
  if (metric.unit.endsWith('/s')) return 'throughput'
  return 'other'
}

export function metricKey(metric: BenchmarkMetric): string {
  const dimensions = Object.entries(metric.dimensions)
    .filter(([name]) => name !== 'category')
    .sort(([left], [right]) => left.localeCompare(right))
  return `${metric.name}:${JSON.stringify(dimensions)}`
}

export function metricAt(
  metrics: readonly BenchmarkMetric[],
  name: string,
  dimensions: Record<string, BenchmarkScalar> = {},
): BenchmarkMetric | undefined {
  return metrics.find((metric) => metric.name === name && Object.entries(dimensions).every(
    ([dimension, value]) => metric.dimensions[dimension] === value,
  ))
}

export function uniqueInOrder<T>(values: readonly T[]): T[] {
  return [...new Set(values)]
}

export function jsonObject(value: BenchmarkJson | undefined): BenchmarkJsonObject | null {
  return value !== null && value !== undefined && typeof value === 'object' && !Array.isArray(value)
    ? value as BenchmarkJsonObject
    : null
}

export function jsonString(value: BenchmarkJson | undefined): string | null {
  return typeof value === 'string' ? value : null
}

export function jsonNumber(value: BenchmarkJson | undefined): number | null {
  return typeof value === 'number' ? value : null
}

export function jsonBoolean(value: BenchmarkJson | undefined): boolean | null {
  return typeof value === 'boolean' ? value : null
}

export function gpuDevices(run: BenchmarkV1Run): BenchmarkGpuDevice[] {
  const raw = run.platform.cuda.devices
  const rows: Array<[string, BenchmarkJson]> = Array.isArray(raw)
    ? raw.map((value, index) => [String(index), value])
    : raw && typeof raw === 'object'
      ? Object.entries(raw)
      : []
  return rows.flatMap(([index, value]) => {
    const device = jsonObject(value)
    if (!device) return []
    return [{
      computeCapability: jsonString(device.computeCapability) ?? (
        [jsonNumber(device.major), jsonNumber(device.minor)].every(
          (part) => part !== null,
        )
          ? `${jsonNumber(device.major)}.${jsonNumber(device.minor)}`
          : 'Not reported'
      ),
      index,
      memoryBusWidth: jsonNumber(device.memoryBusWidth),
      multiProcessorCount: jsonNumber(device.multiProcessorCount),
      name: jsonString(device.name) ?? jsonString(device.device_name) ?? `CUDA device ${index}`,
      totalGlobalMem: jsonNumber(device.totalGlobalMem),
    }]
  })
}

export function gpuModels(run: BenchmarkV1Run): string[] {
  return uniqueInOrder(gpuDevices(run).map((device) => device.name))
}

export function gpuSummary(run: BenchmarkV1Run): string {
  const devices = gpuDevices(run)
  if (!devices.length) return 'CUDA platform not reported'
  const models = uniqueInOrder(devices.map((device) => device.name))
  return `${devices.length} × ${models.join(' / ')}`
}

export function cpuSummary(run: BenchmarkV1Run): string {
  return jsonString(run.platform.cpu.model) ?? jsonString(run.platform.cpu.architecture) ?? 'CPU not reported'
}

export function executionBackendLabel(backend: BenchmarkV1ExecutionBackend): string {
  return backend === 'cuda' ? 'CUDA GPU' : 'CPU'
}

export function runExecutionBackends(run: BenchmarkV1Run): BenchmarkV1ExecutionBackend[] {
  return [run.execution.backend]
}

export function runExecutionBackendLabel(run: BenchmarkV1Run): string {
  return executionBackendLabel(run.execution.backend)
}

export function runExecutionHardwareSummary(run: BenchmarkV1Run): string {
  if (run.execution.backend === 'cpu') return cpuSummary(run)
  const selectedIndex = run.execution.device.slice('cuda:'.length)
  const selected = gpuDevices(run).find((device) => device.index === selectedIndex)
  return selected?.name ?? `CUDA device ${selectedIndex}`
}

export function ramSummary(run: BenchmarkV1Run): string {
  return formatBytes(jsonNumber(run.platform.memory.total_bytes))
}

export function caseSummary(run: BenchmarkV1Run): string {
  const parts = [`${run.case_counts.measured} measured`]
  if (run.case_counts.unavailable) parts.push(`${run.case_counts.unavailable} unavailable`)
  return parts.join(' · ')
}

export function correctnessSummary(check: BenchmarkCorrectness): string {
  const symbols: Record<string, string> = { '<=': '≤', '>=': '≥', '==': '=' }
  const observed = `${formatScalar(check.observed)}${check.unit ? ` ${check.unit}` : ''}`
  const limit = `${symbols[check.comparison] ?? check.comparison} ${formatScalar(check.limit)}${check.unit ? ` ${check.unit}` : ''}`
  return `${observed} · limit ${limit}`
}

export function resultUrl(run: BenchmarkV1Run): string {
  return withBase(`/benchmarks/v1/results/${encodeURIComponent(run.slug)}`)
}

export function compareUrl(ids: readonly string[]): string {
  const params = new URLSearchParams()
  params.set('runs', ids.join(','))
  return `${withBase('/benchmarks/v1/compare')}?${params.toString()}`
}

export function shortCommit(commit: string): string {
  return commit.slice(0, 12)
}

export function caseById(run: BenchmarkV1Run, id: BenchmarkV1CaseId): BenchmarkCase | undefined {
  return run.cases.find((entry) => entry.id === id)
}
