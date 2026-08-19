<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'

import type {
  BenchmarkCase,
  BenchmarkCorrectness,
  BenchmarkJson,
  BenchmarkMetric,
} from '../data/catalog'
import { benchmarkCatalog } from '../data/catalogClient'
import { benchmarkV1CaseLabels } from '../data/specification'
import {
  compareUrl,
  correctnessSummary,
  cpuSummary,
  directionArrow,
  directionLabel,
  displayIdentifier,
  executionBackendLabel,
  formatBytes,
  formatDate,
  formatMetric,
  formatScalar,
  gpuDevices,
  humanizeIdentifier,
  jsonNumber,
  jsonString,
  metricAt,
  resultUrl,
  runExecutionBackendLabel,
  shortCommit,
  uniqueInOrder,
} from '../data/ui'
import CkksParameterSummary from './CkksParameterSummary.vue'
import BenchmarkPortalHeader from './BenchmarkPortalHeader.vue'

interface Fact { label: string; value: string }
interface PolynomialRow {
  basis: string
  caseId: string
  degree: number | null
  latency?: BenchmarkMetric
  methodId: string
  requiredLevels: number | null
  throughput?: BenchmarkMetric
}

const props = defineProps<{ reportSlug: string }>()
const runs = benchmarkCatalog.runs
const report = computed(() => runs.find((run) => run.slug === props.reportSlug))
const otherRun = computed(() => runs.find((run) => run.id !== report.value?.id))
const singleMetricName = ref('depth-aware-ckks-operation-latency')
const singleLevelWindow = ref(0)
const nttBasis = ref('Q')
const nttOperation = ref('forward_ntt')
const nttLevel = ref(0)

const singleMetricChoices = [
  ['depth-aware-ckks-operation-latency', 'Latency', 'lower'],
  ['depth-aware-ckks-inverse-serial-rate', 'Serial rate', 'higher'],
  ['depth-aware-ckks-packed-slot-rate', 'Packed-slot rate', 'higher'],
] as const

function caseIndexLabel(caseEntry: BenchmarkCase): string {
  return benchmarkV1CaseLabels[caseEntry.id]
}
function rawValue(value: BenchmarkJson | undefined): string {
  if (value === null || value === undefined) return 'Not reported'
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') return formatScalar(value)
  return JSON.stringify(value)
}
function metricDimensions(metric: BenchmarkMetric): string {
  return Object.entries(metric.dimensions)
    .filter(([name]) => name !== 'category')
    .map(([name, value]) => `${humanizeIdentifier(name)} ${formatScalar(value)}`)
    .join(' · ')
}
function levelValues(caseEntry: BenchmarkCase): number[] {
  return uniqueInOrder(caseEntry.metrics
    .filter((metric) => metric.name === singleMetricName.value)
    .map((metric) => Number(metric.dimensions.entry_level)))
    .sort((left, right) => left - right)
}
function levelWindows(caseEntry: BenchmarkCase): number[][] {
  const levels = levelValues(caseEntry)
  const result: number[][] = []
  for (let index = 0; index < levels.length; index += 8) {
    result.push(levels.slice(index, index + 8))
  }
  return result
}
function levelWindowLabel(levels: number[]): string {
  if (!levels.length) return 'No levels'
  return levels.length === 1
    ? `L${levels[0]}`
    : `L${levels[0]}–L${levels[levels.length - 1]}`
}
function visibleLevelValues(caseEntry: BenchmarkCase): number[] {
  return levelWindows(caseEntry)[singleLevelWindow.value] ?? levelWindows(caseEntry)[0] ?? []
}
function operationValues(caseEntry: BenchmarkCase): string[] {
  const configured = caseEntry.metadata?.operation_order
  if (Array.isArray(configured)) return configured.filter((item): item is string => typeof item === 'string')
  return uniqueInOrder(caseEntry.metrics
    .filter((metric) => metric.name === singleMetricName.value)
    .map((metric) => String(metric.dimensions.operation)))
}
function singleMetric(caseEntry: BenchmarkCase, operation: string, level: number): BenchmarkMetric | undefined {
  return metricAt(caseEntry.metrics, singleMetricName.value, { operation, entry_level: level })
}
function levelBits(caseEntry: BenchmarkCase, level: number): string {
  const metric = caseEntry.metrics.find((item) => item.dimensions.entry_level === level)
  const bits = metric?.dimensions.active_q_product_bits
  return typeof bits === 'number' ? `${bits} bits` : ''
}
function nttLevels(caseEntry: BenchmarkCase): number[] {
  return uniqueInOrder(caseEntry.metrics
    .filter((metric) => metric.name === 'indexed-ntt-latency')
    .map((metric) => Number(metric.dimensions.entry_level)))
    .sort((left, right) => left - right)
}
function nttCell(caseEntry: BenchmarkCase): BenchmarkMetric | undefined {
  return metricAt(caseEntry.metrics, 'indexed-ntt-latency', {
    entry_level: nttLevel.value,
    modulus_basis: nttBasis.value,
    operation: nttOperation.value,
  })
}
function matrixMetrics(caseEntry: BenchmarkCase): BenchmarkMetric[] {
  const names = [
    ['dense-matrix-multiplication-latency', 'end-to-end'],
    ['dense-matrix-logical-macs-rate', null],
  ]
  return names.flatMap(([name, phase]) => {
    const metric = caseEntry.metrics.find((item) => item.name === name && (phase === null || item.dimensions.phase === phase))
    return metric ? [metric] : []
  })
}
function matrixMetricLabel(metric: BenchmarkMetric): string {
  const labels: Record<string, string> = {
    'dense-matrix-multiplication-latency': 'End-to-end latency',
    'dense-matrix-logical-macs-rate': 'Logical MAC rate',
  }
  return labels[metric.name] ?? humanizeIdentifier(metric.name)
}
function polynomialRows(caseEntry: BenchmarkCase): PolynomialRow[] {
  const latencies = caseEntry.metrics.filter((metric) => metric.name === 'polynomial-evaluation-latency')
  return latencies.map((latency) => {
    const methodId = String(latency.dimensions.method_id)
    const caseId = String(latency.dimensions.case_id)
    return {
      basis: String(latency.dimensions.basis),
      caseId,
      degree: typeof latency.dimensions.degree === 'number' ? latency.dimensions.degree : null,
      latency,
      methodId,
      requiredLevels: typeof latency.dimensions.required_levels === 'number' ? latency.dimensions.required_levels : null,
      throughput: metricAt(caseEntry.metrics, 'polynomial-evaluation-throughput', { method_id: methodId, case_id: caseId }),
    }
  })
}
function polynomialCases(caseEntry: BenchmarkCase): string[] {
  return uniqueInOrder(polynomialRows(caseEntry).map((row) => row.caseId))
}
function rowsForPolynomial(caseEntry: BenchmarkCase, caseId: string): PolynomialRow[] {
  return polynomialRows(caseEntry).filter((row) => row.caseId === caseId)
}
function methodLabel(row: PolynomialRow): string {
  return displayIdentifier(row.methodId.replace(`${row.caseId.replace(/-v1$/u, '')}-`, ''))
}
function correctnessObserved(check: BenchmarkCorrectness): string {
  return `${formatScalar(check.observed)}${check.unit ? ` ${check.unit}` : ''}`
}
function softwareFacts(): Fact[] {
  const run = report.value
  if (!run) return []
  return [
    { label: 'FHElium', value: `${run.fhelium.version} · ${shortCommit(run.fhelium.commit)}${run.fhelium.dirty ? ' · local changes' : ''}` },
    { label: 'Torch', value: jsonString(run.platform.torch.version) ?? 'Not reported' },
    { label: 'Torch CUDA build', value: jsonString(run.platform.torch.cuda_build_version) ?? 'Not reported' },
    { label: 'NCCL', value: rawValue(run.platform.torch.nccl_version) },
    { label: 'Python', value: jsonString(run.platform.python.version) ?? 'Not reported' },
    { label: 'Operating system', value: jsonString(run.platform.system.platform) ?? jsonString(run.platform.system.system) ?? 'Not reported' },
  ]
}
</script>

<template>
  <div class="benchmark-detail-shell">
    <BenchmarkPortalHeader />
    <main v-if="report" id="portal-content" class="detail-main">
      <nav class="detail-breadcrumb" aria-label="Breadcrumb"><a :href="withBase('/benchmarks/')">Results</a><span>/</span><span>FHElium Benchmark v1</span></nav>
      <header class="detail-hero">
        <div><span class="detail-pill">{{ runExecutionBackendLabel(report) }} benchmark run</span><h1>FHElium Benchmark v1</h1><p>FHElium {{ report.fhelium.version }} · <time :datetime="report.recorded_at">{{ formatDate(report.recorded_at, true) }} UTC</time></p></div>
        <div class="detail-actions"><a :href="withBase(report.raw_path)" download>Download raw JSON</a><a v-if="otherRun" :href="compareUrl([report.id, otherRun.id])">Compare run</a></div>
      </header>

      <section class="platform-section">
        <header><p>System under test</p><h2>Execution hardware and host</h2></header>
        <div class="platform-grid">
          <article class="cpu-card"><span>Host CPU</span><strong>{{ cpuSummary(report) }}</strong><dl><div><dt>Logical CPUs</dt><dd>{{ rawValue(report.platform.cpu.logical_count) }}</dd></div><div><dt>Architecture</dt><dd>{{ rawValue(report.platform.cpu.architecture) }}</dd></div><div><dt>System RAM</dt><dd>{{ formatBytes(jsonNumber(report.platform.memory.total_bytes)) }}</dd></div></dl></article>
          <article v-for="device in gpuDevices(report)" :key="device.index" class="gpu-card"><span>CUDA execution GPU {{ device.index }}</span><strong>{{ device.name }}</strong><dl><div><dt>Memory</dt><dd>{{ formatBytes(device.totalGlobalMem) }}</dd></div><div><dt>Compute</dt><dd>{{ device.computeCapability }}</dd></div><div><dt>SMs</dt><dd>{{ device.multiProcessorCount ?? '—' }}</dd></div><div><dt>Memory bus</dt><dd>{{ device.memoryBusWidth ? `${device.memoryBusWidth}-bit` : '—' }}</dd></div></dl></article>
        </div>
        <dl class="software-strip"><div v-for="item in softwareFacts()" :key="item.label"><dt>{{ item.label }}</dt><dd>{{ item.value }}</dd></div></dl>
      </section>

      <nav class="case-index" aria-label="Benchmark cases"><a v-for="entry in report.cases" :key="entry.id" :href="`#case-${entry.id}`" :class="{ 'is-gap': entry.status === 'unavailable' }" :title="`${entry.title} — ${entry.status}`" :aria-label="`${entry.title} — ${entry.status}`">{{ caseIndexLabel(entry) }}</a></nav>

      <section v-for="(entry, index) in report.cases" :id="`case-${entry.id}`" :key="entry.id" class="case-section">
        <header class="case-header"><div><p>Benchmark {{ index + 1 }} · {{ entry.category }}</p><h2>{{ entry.title }}</h2><span><strong class="case-backend">{{ executionBackendLabel(report.execution.backend) }} execution</strong>{{ entry.workload_id }} · {{ entry.profile }}</span></div><b :class="entry.status">{{ entry.status }}</b></header>

        <div v-if="entry.status === 'unavailable'" class="unavailable-panel"><strong>No data for this Benchmark v1 case</strong><p>{{ entry.unavailable?.reason }}</p><code>{{ JSON.stringify(entry.unavailable?.details) }}</code></div>
        <template v-else>
          <CkksParameterSummary
            :context="entry.benchmark_context"
          />

          <section v-if="entry.workload_id === 'ckks-depth-aware-single-operations'" class="workload-view">
            <header><div><p>All public entry levels</p><h3>Depth-aware operation matrix</h3></div><div class="operation-controls"><label><span>Level range</span><select v-model.number="singleLevelWindow"><option v-for="(levels, windowIndex) in levelWindows(entry)" :key="levelWindowLabel(levels)" :value="windowIndex">{{ levelWindowLabel(levels) }}</option></select></label><div class="metric-switch"><button v-for="choice in singleMetricChoices" :key="choice[0]" type="button" :aria-pressed="singleMetricName === choice[0]" @click="singleMetricName = choice[0]">{{ choice[1] }} <b :class="`direction-${choice[2]}`">{{ directionArrow(choice[2]) }}</b></button></div></div></header>
            <div class="wide-table operation-table"><table><caption>Selected operation metric for the chosen CKKS entry-level range</caption><thead><tr><th scope="col">Operation</th><th v-for="level in visibleLevelValues(entry)" :key="level" scope="col"><span>Level {{ level }}</span><small>{{ levelBits(entry, level) }} active Q</small></th></tr></thead><tbody><tr v-for="operation in operationValues(entry)" :key="operation"><th scope="row">{{ humanizeIdentifier(operation) }}</th><td v-for="level in visibleLevelValues(entry)" :key="level"><strong v-if="singleMetric(entry, operation, level)">{{ formatMetric(singleMetric(entry, operation, level)!) }}</strong><span v-else title="Not applicable at this entry level">—</span></td></tr></tbody></table></div>
          </section>

          <section v-else-if="entry.workload_id === 'indexed-ntt-operations'" class="workload-view">
            <header><div><p>Fixed cross-backend primitive</p><h3>Indexed radix-2 NTT operations</h3></div></header>
            <div class="ntt-controls"><label><span>Basis</span><select v-model="nttBasis"><option>Q</option><option>QP</option></select></label><label><span>Operation</span><select v-model="nttOperation"><option value="forward_ntt">Forward NTT</option><option value="inverse_ntt">Inverse NTT</option><option value="roundtrip">Roundtrip</option></select></label><label><span>Entry level</span><select v-model.number="nttLevel"><option v-for="level in nttLevels(entry)" :key="level" :value="level">Level {{ level }}</option></select></label></div>
            <div class="performance-grid"><article><span>Selected-cell latency</span><strong>{{ nttCell(entry) ? formatMetric(nttCell(entry)!) : '—' }}</strong><small>{{ displayIdentifier(String(nttCell(entry)?.dimensions.backend ?? 'radix2_indexed')) }}</small></article></div>
          </section>

          <section v-else-if="entry.workload_id.startsWith('dense-matrix-multiplication-')" class="workload-view">
            <header><div><p>{{ entry.workload_id.endsWith('ptct') ? 'Plaintext × ciphertext' : 'Ciphertext × ciphertext' }} · one local device</p><h3>Fixed 16 × 16 dense matrix multiplication</h3></div></header>
            <div class="performance-grid"><article v-for="metric in matrixMetrics(entry)" :key="`${metric.name}-${String(metric.dimensions.phase)}`"><span>{{ matrixMetricLabel(metric) }}</span><strong>{{ formatMetric(metric) }}</strong><small>{{ metricDimensions(metric) }}</small></article></div>
          </section>

          <section v-else-if="entry.workload_id === 'polynomial-evaluation'" class="workload-view">
            <header><div><p>Affine and degree-four methods</p><h3>Polynomial method matrix</h3></div></header>
            <section v-for="caseId in polynomialCases(entry)" :key="caseId" class="polynomial-group"><header><h4>{{ displayIdentifier(caseId) }}</h4><span>degree {{ rowsForPolynomial(entry, caseId)[0]?.degree }}</span></header><div class="wide-table"><table><caption>Polynomial methods for {{ caseId }}</caption><thead><tr><th scope="col">Method</th><th scope="col">Required levels</th><th scope="col" class="direction-lower">Latency ↓</th></tr></thead><tbody><tr v-for="row in rowsForPolynomial(entry, caseId)" :key="row.methodId"><th scope="row">{{ methodLabel(row) }}</th><td>{{ row.requiredLevels }}</td><td>{{ row.latency ? formatMetric(row.latency) : '—' }}</td></tr></tbody></table></div></section>
          </section>

          <section v-else class="workload-view"><header><div><p>Measurements</p><h3>Performance</h3></div></header><div class="performance-grid"><article v-for="metric in entry.metrics" :key="`${metric.name}-${JSON.stringify(metric.dimensions)}`"><span>{{ humanizeIdentifier(metric.name) }}</span><strong>{{ formatMetric(metric) }}</strong><small>{{ metricDimensions(metric) }}</small></article></div></section>

          <details class="correctness"><summary><span>Validation evidence</span><strong>{{ entry.correctness.length }} {{ entry.correctness.length === 1 ? 'check' : 'checks' }} passed</strong></summary><div class="wide-table"><table><caption>Correctness evidence</caption><thead><tr><th scope="col">Check</th><th scope="col">Observed</th><th scope="col">Criterion</th><th scope="col">Status</th></tr></thead><tbody><tr v-for="check in entry.correctness" :key="check.name"><th scope="row">{{ displayIdentifier(check.name) }}</th><td>{{ correctnessObserved(check) }}</td><td>{{ correctnessSummary(check) }}</td><td class="pass">✓ Passed</td></tr></tbody></table></div></details>

          <details v-if="entry.timed_boundary" class="provenance"><summary>Timed work and provenance</summary><div><article><span>{{ entry.timed_boundary.id }}</span><strong>{{ entry.timed_boundary.description }}</strong><p>{{ entry.timed_boundary.synchronization }}</p></article><article><span>Includes</span><ul><li v-for="item in entry.timed_boundary.includes" :key="item">{{ item }}</li></ul></article><article><span>Excludes</span><ul><li v-for="item in entry.timed_boundary.excludes" :key="item">{{ item }}</li></ul></article></div></details>
        </template>
      </section>

      <footer class="raw-footer"><div><span>Raw report</span><code>sha256:{{ report.raw_sha256 }}</code></div><a :href="withBase(report.raw_path)" download>Download JSON</a></footer>
    </main>
    <main v-else class="missing"><h1>Benchmark run not found</h1><a :href="withBase('/benchmarks/')">Return to results</a></main>
  </div>
</template>

<style scoped>
.benchmark-detail-shell { min-height: calc(100vh - var(--vp-nav-height)); color: var(--vp-c-text-1); background: var(--vp-c-bg); }.detail-main { max-width: 1380px; margin: 0 auto; padding: 16px 24px 68px; }.detail-breadcrumb { display: flex; gap: 8px; color: var(--vp-c-text-2); font-size: 11px; }.detail-breadcrumb a { color: var(--vp-c-brand-1); text-decoration: none; }.detail-hero { display: flex; gap: 24px; align-items: end; justify-content: space-between; padding: 28px 0 18px; }.detail-pill { display: inline-block; color: var(--vp-c-brand-1); font-size: 10px; font-weight: 700; }.detail-hero h1 { margin: 8px 0 0; border: 0; font-size: clamp(34px, 4vw, 46px); line-height: 1.05; letter-spacing: -.035em; }.detail-hero p { margin: 9px 0 0; color: var(--vp-c-text-2); font-size: 12px; }.detail-actions { display: flex; flex: 0 0 auto; gap: 7px; flex-direction: row; }.detail-actions a, .raw-footer > a { padding: 8px 12px; border: 1px solid var(--vp-c-brand-1); border-radius: 4px; color: white; background: var(--vp-c-brand-3); font-size: 11px; font-weight: 700; text-align: center; text-decoration: none; }.detail-actions a + a { color: var(--vp-c-text-1); background: var(--vp-c-bg); }
.platform-grid article { min-width: 0; padding: 3px 0 8px; background: transparent; }.platform-grid article > span, .raw-footer span { display: block; color: var(--vp-c-text-2); font-size: 10px; font-weight: 650; }
.platform-section { margin-top: 28px; padding: 18px 20px 16px; border-radius: 12px; background: color-mix(in srgb, var(--vp-c-bg-soft) 72%, var(--vp-c-bg)); box-shadow: 0 9px 28px color-mix(in srgb, var(--vp-c-text-1) 6%, transparent); }.case-section { margin-top: 42px; }.platform-section > header p, .case-header p, .workload-view > header p { margin: 0 0 4px; color: var(--vp-c-brand-1); font-size: 10px; font-weight: 700; }.platform-section h2, .case-header h2 { margin: 0; border: 0; font-size: 26px; }.platform-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 34px; margin-top: 15px; }.platform-grid strong { display: block; margin-top: 6px; font-size: 14px; }.platform-grid dl { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 12px 0 0; }.cpu-card dl { grid-template-columns: repeat(3, minmax(0, 1fr)); }.platform-grid dt, .software-strip dt { color: var(--vp-c-text-2); font-size: 10px; }.platform-grid dd, .software-strip dd { margin: 3px 0 0; overflow-wrap: anywhere; font-size: 12px; }.software-strip { display: flex; gap: 12px 26px; flex-wrap: wrap; margin: 19px 0 0; }.software-strip div { min-width: 90px; max-width: 360px; }.software-strip div:last-child { flex: 1 1 280px; }
.case-index { display: flex; gap: 2px; margin-top: 22px; padding: 6px; overflow-x: auto; border-radius: 10px; background: color-mix(in srgb, var(--vp-c-bg-soft) 66%, transparent); scrollbar-width: thin; }.case-index a { display: inline-flex; flex: 0 0 auto; gap: 7px; align-items: center; padding: 8px 10px; border-radius: 7px; color: var(--vp-c-text-1); font-size: 11px; text-decoration: none; }.case-index a::before { width: 6px; height: 6px; flex: 0 0 auto; border-radius: 50%; background: #267a5d; box-shadow: 0 0 0 3px color-mix(in srgb, #267a5d 12%, transparent); content: ''; }.case-index a:hover, .case-index a:focus-visible { background: color-mix(in srgb, var(--vp-c-bg) 72%, transparent); }.case-index a:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: -1px; }.case-index a.is-gap { color: var(--vp-c-text-2); }.case-index a.is-gap::before { background: transparent; box-shadow: inset 0 0 0 1px var(--vp-c-text-3); }.case-header { display: flex; gap: 18px; align-items: end; justify-content: space-between; }.case-header > div > span { display: block; margin-top: 5px; color: var(--vp-c-text-2); font-size: 11px; }.case-header > b { padding: 4px 7px; border-radius: 999px; color: #267a5d; background: color-mix(in srgb, #267a5d 10%, transparent); font-size: 10px; }.case-header > b.unavailable { color: var(--vp-c-text-2); background: var(--vp-c-bg-soft); }.unavailable-panel { margin-top: 14px; padding: 18px; border-radius: 9px; background: var(--vp-c-bg-soft); }.unavailable-panel p { margin: 6px 0; color: var(--vp-c-text-2); font-size: 12px; }.unavailable-panel code { font-size: 10px; }
.case-backend { display: inline-flex; margin-right: 8px; padding: 2px 5px; border-radius: 999px; color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-size: 9px; text-transform: uppercase; }
.workload-view > header { display: flex; gap: 18px; align-items: end; justify-content: space-between; }.workload-view h3 { margin: 0; border: 0; font-size: 22px; }.correctness summary, .provenance summary { padding-top: 10px; color: var(--vp-c-brand-1); font-size: 11px; font-weight: 700; cursor: pointer; }
.workload-view { margin-top: 28px; }.operation-controls { display: flex; gap: 10px; align-items: end; justify-content: flex-end; }.operation-controls label span { display: block; margin-bottom: 3px; color: var(--vp-c-text-2); font-size: 10px; }.operation-controls select { height: 34px; padding: 0 9px; border: 1px solid var(--vp-c-divider); border-radius: 4px; color: var(--vp-c-text-1); background: var(--vp-c-bg); font-size: 11px; }.metric-switch { display: flex; gap: 4px; flex-wrap: wrap; justify-content: flex-end; }.metric-switch button { min-height: 34px; padding: 6px 8px; border: 1px solid var(--vp-c-divider); border-radius: 4px; color: var(--vp-c-text-2); background: var(--vp-c-bg); font: inherit; font-size: 11px; cursor: pointer; }.metric-switch button[aria-pressed='true'] { color: var(--vp-c-text-1); border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }.direction-higher { color: #267a5d !important; }.direction-lower { color: #5d72cc !important; }.dark .direction-higher { color: #6fd2ad !important; }.dark .direction-lower { color: #a9b8ff !important; }
.wide-table { margin-top: 12px; overflow-x: auto; border-radius: 8px; background: color-mix(in srgb, var(--vp-c-bg-soft) 55%, transparent); }.wide-table table { width: 100%; min-width: 760px; margin: 0; border-collapse: collapse; font-size: 12px; }.wide-table caption { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }.wide-table th, .wide-table td { height: 42px; padding: 7px 11px; border: 0; border-bottom: 1px solid color-mix(in srgb, var(--vp-c-divider) 68%, transparent); text-align: left; white-space: nowrap; }.wide-table tbody tr:last-child th, .wide-table tbody tr:last-child td { border-bottom: 0; }.wide-table thead th { color: var(--vp-c-text-2); background: color-mix(in srgb, var(--vp-c-bg-soft) 86%, transparent); font-size: 11px; }.wide-table thead th span, .wide-table thead th small { display: block; }.wide-table thead th small { margin-top: 2px; font-size: 9px; font-weight: 400; }.wide-table td strong { font-family: var(--vp-font-family-mono); font-size: 11px; }.operation-table table { table-layout: fixed; }.operation-table th:first-child { width: 190px; }.ranking-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 7px; margin-top: 12px; }.ranking-grid article { padding: 10px; border-radius: 7px; background: var(--vp-c-bg-soft); }.ranking-grid article.winner { background: var(--vp-c-brand-soft); }.ranking-grid span, .ranking-grid strong, .ranking-grid b { display: block; }.ranking-grid span { color: var(--vp-c-brand-1); font-size: 10px; }.ranking-grid strong { margin-top: 3px; font-size: 12px; }.ranking-grid b { margin-top: 4px; font-family: var(--vp-font-family-mono); font-size: 11px; }.ntt-controls { display: flex; gap: 8px; margin-top: 14px; }.ntt-controls label span { display: block; margin-bottom: 3px; color: var(--vp-c-text-2); font-size: 10px; }.ntt-controls select { height: 34px; padding: 0 8px; border: 0; border-radius: 6px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); font-size: 11px; }
.performance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 12px; }.performance-grid article { min-width: 0; padding: 12px 13px; border-radius: 8px; background: var(--vp-c-bg-soft); }.performance-grid span, .performance-grid strong, .performance-grid small { display: block; }.performance-grid span { color: var(--vp-c-text-2); font-size: 11px; }.performance-grid strong { margin-top: 5px; font-family: var(--vp-font-family-mono); font-size: 15px; }.performance-grid small { margin-top: 4px; overflow: hidden; color: var(--vp-c-text-2); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }.polynomial-group { margin-top: 18px; }.polynomial-group > header { display: flex; align-items: baseline; justify-content: space-between; }.polynomial-group h4 { margin: 0; font-size: 15px; }.polynomial-group header span { color: var(--vp-c-text-2); font-size: 11px; }
.correctness, .provenance { margin-top: 20px; padding: 0 13px 11px; border-radius: 8px; background: var(--vp-c-bg-soft); }.correctness summary { display: flex; align-items: center; justify-content: space-between; }.correctness summary strong { color: #267a5d; }.correctness .pass { color: #267a5d; font-weight: 700; }.provenance > div { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 11px; }.provenance article { padding: 11px; border-radius: 7px; background: color-mix(in srgb, var(--vp-c-bg) 74%, transparent); }.provenance span { color: var(--vp-c-brand-1); font-size: 10px; }.provenance strong { display: block; margin-top: 5px; font-size: 12px; }.provenance p, .provenance ul { color: var(--vp-c-text-2); font-size: 11px; }.provenance ul { padding-left: 15px; }
.raw-footer { display: flex; gap: 25px; align-items: center; justify-content: space-between; margin-top: 48px; padding: 19px 20px; border-radius: 10px; background: var(--vp-c-bg-soft); }.raw-footer code { display: block; margin-top: 6px; overflow-wrap: anywhere; font-size: 9px; }.missing { display: grid; min-height: 60vh; place-content: center; text-align: center; }
@media (max-width: 900px) { .provenance > div { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 650px) { .detail-main { padding-inline: 14px; }.detail-hero { align-items: flex-start; flex-direction: column; padding-block: 20px 16px; }.detail-hero h1 { font-size: 32px; }.case-header, .workload-view > header { align-items: flex-start; flex-direction: column; }.detail-actions { align-items: stretch; flex-direction: row; flex-wrap: wrap; }.platform-section { margin-top: 20px; padding: 16px; }.platform-grid { grid-template-columns: minmax(0, 1fr); gap: 10px; }.platform-grid article { padding: 7px 0; }.software-strip { gap: 14px 22px; }.software-strip div { flex: 1 1 110px; }.operation-controls { align-items: flex-start; flex-direction: column; }.metric-switch { justify-content: flex-start; }.ntt-controls { align-items: stretch; flex-direction: column; }.raw-footer { align-items: stretch; flex-direction: column; } }
@media (forced-colors: active) { .platform-section, .wide-table, .correctness, .provenance, .raw-footer { border: 1px solid CanvasText; } }
</style>
