<script setup lang="ts">
import {
  computed,
  defineComponent,
  h,
  onMounted,
  ref,
  watch,
  type PropType,
} from 'vue'
import { withBase } from 'vitepress'

import type { BenchmarkCase, BenchmarkMetric, BenchmarkV1Run } from '../data/catalog'
import { benchmarkCatalog } from '../data/catalogClient'
import {
  benchmarkV1CaseIds,
  benchmarkV1CaseLabels,
  type BenchmarkV1CaseId,
} from '../data/specification'
import {
  caseById,
  caseSummary,
  correctnessSummary,
  directionLabel,
  displayIdentifier,
  formatDate,
  formatMetric,
  humanizeIdentifier,
  metricAt,
  metricKey,
  resultUrl,
  runExecutionBackendLabel,
  runExecutionHardwareSummary,
  shortCommit,
  uniqueInOrder,
} from '../data/ui'
import BenchmarkPortalHeader from './BenchmarkPortalHeader.vue'

interface CompareRow {
  direction: BenchmarkMetric['direction']
  key: string
  label: string
  metric: (entry: BenchmarkCase) => BenchmarkMetric | undefined
}

function runGpuCountLabel(run: BenchmarkV1Run): string {
  return `${runExecutionBackendLabel(run)} · ${run.execution.device}`
}

const CompareTable = defineComponent({
  name: 'CompareTable',
  props: {
    caseId: { type: String as PropType<BenchmarkV1CaseId>, required: true },
    gapLabel: {
      type: Function as PropType<(run: BenchmarkV1Run, id: BenchmarkV1CaseId) => string>,
      required: true,
    },
    rows: { type: Array as PropType<CompareRow[]>, required: true },
    runs: { type: Array as PropType<BenchmarkV1Run[]>, required: true },
    value: {
      type: Function as PropType<
        (run: BenchmarkV1Run, id: BenchmarkV1CaseId, row: CompareRow) => string
      >,
      required: true,
    },
  },
  setup(props) {
    return () => h('div', { class: 'table-wrap' }, [
      h('table', [
        h('thead', [
          h('tr', [
            h('th', { scope: 'col' }, 'Measurement'),
            ...props.runs.map((run, index) =>
              h('th', { scope: 'col', key: run.id }, [
                h('span', `Run ${index + 1}`),
                h('small', runGpuCountLabel(run)),
              ]),
            ),
          ]),
        ]),
        h(
          'tbody',
          props.rows.map((row) =>
            h('tr', { key: row.key }, [
              h('th', { scope: 'row' }, row.label),
              ...props.runs.map((run) => {
                const measured = props.value(run, props.caseId, row)
                return h('td', { key: run.id }, measured
                  ? h('strong', measured)
                  : h('span', {
                      class: 'blank',
                      title: props.gapLabel(run, props.caseId),
                      'aria-label': props.gapLabel(run, props.caseId),
                    }))
              }),
            ]),
          ),
        ),
      ]),
    ])
  },
})

const runs = benchmarkCatalog.runs
const byId = new Map(runs.map((run) => [run.id, run]))
const selectedIds = ref<string[]>([])
const mounted = ref(false)
const operationLevel = ref(0)
const nttLevel = ref(0)
const nttBasis = ref('Q')
const nttOperation = ref('forward_ntt')
const selectedRuns = computed(() => selectedIds.value.map((id) => byId.get(id)).filter((run): run is BenchmarkV1Run => run !== undefined))
const valid = computed(() => selectedRuns.value.length >= 2 && selectedRuns.value.length <= 4)

const caseIds = benchmarkV1CaseIds

function firstCase(id: BenchmarkV1CaseId): BenchmarkCase | undefined {
  for (const run of selectedRuns.value) {
    const entry = caseById(run, id)
    if (entry) return entry
  }
  return undefined
}
function anchorCase(id: BenchmarkV1CaseId): BenchmarkCase | undefined {
  for (const run of selectedRuns.value) {
    const entry = caseById(run, id)
    if (entry?.status === 'measured') return entry
  }
  return firstCase(id)
}
function comparableCase(run: BenchmarkV1Run, id: BenchmarkV1CaseId): BenchmarkCase | undefined {
  const entry = caseById(run, id)
  return entry?.status === 'measured' ? entry : undefined
}
function gapLabel(run: BenchmarkV1Run, id: BenchmarkV1CaseId): string {
  const entry = caseById(run, id)
  if (!entry) return 'Benchmark v1 case absent from this run'
  if (entry.status === 'unavailable') return entry.unavailable?.reason ?? 'Unavailable'
  return 'Not reported'
}
function removeRun(id: string): void { selectedIds.value = selectedIds.value.filter((item) => item !== id) }
function value(run: BenchmarkV1Run, id: BenchmarkV1CaseId, row: CompareRow): string {
  const entry = comparableCase(run, id)
  if (!entry) return ''
  const metric = row.metric(entry)
  return metric ? formatMetric(metric) : ''
}
function operations(id: BenchmarkV1CaseId): string[] {
  const entry = anchorCase(id)
  return entry ? uniqueInOrder(entry.metrics.filter((metric) => metric.name === 'depth-aware-ckks-operation-latency').map((metric) => String(metric.dimensions.operation))) : []
}
function operationLevels(id: BenchmarkV1CaseId): number[] {
  const levels = selectedRuns.value.flatMap((run) => comparableCase(run, id)?.metrics
    .filter((metric) => metric.name === 'depth-aware-ckks-operation-latency')
    .map((metric) => Number(metric.dimensions.entry_level)) ?? [])
  return uniqueInOrder(levels).sort((left, right) => left - right)
}
function singleRows(id: BenchmarkV1CaseId): CompareRow[] {
  return operations(id).map((operation) => ({
    direction: 'lower', key: operation, label: humanizeIdentifier(operation),
    metric: (entry) => metricAt(entry.metrics, 'depth-aware-ckks-operation-latency', { operation, entry_level: operationLevel.value }),
  }))
}
function nttRows(id: BenchmarkV1CaseId): CompareRow[] {
  return [{
    direction: 'lower', key: 'indexed-ntt-latency', label: 'Indexed radix-2',
    metric: (item) => metricAt(item.metrics, 'indexed-ntt-latency', { entry_level: nttLevel.value, modulus_basis: nttBasis.value, operation: nttOperation.value }),
  }]
}
function nttLevels(id: BenchmarkV1CaseId): number[] {
  const entry = anchorCase(id)
  return entry ? uniqueInOrder(entry.metrics.filter((metric) => metric.name === 'indexed-ntt-latency').map((metric) => Number(metric.dimensions.entry_level))).sort((left, right) => left - right) : []
}
function matrixRows(id: BenchmarkV1CaseId, direction: 'lower' | 'higher'): CompareRow[] {
  const lower = [
    ['dense-matrix-multiplication-latency', 'End-to-end latency', { phase: 'end-to-end' }],
  ] as const
  const higher = [
    ['dense-matrix-logical-macs-rate', 'Logical MAC rate', {}],
  ] as const
  return (direction === 'lower' ? lower : higher).map(([name, label, dimensions]) => ({
    direction, key: name, label,
    metric: (entry) => metricAt(entry.metrics, name, dimensions),
  }))
}
function polynomialRows(id: BenchmarkV1CaseId, metricName: string, direction: 'lower' | 'higher'): CompareRow[] {
  const entry = anchorCase(id)
  if (!entry) return []
  return entry.metrics.filter((metric) => metric.name === metricName).map((metric) => {
    const method = String(metric.dimensions.method_id)
    const caseId = String(metric.dimensions.case_id)
    return {
      direction, key: `${caseId}:${method}`, label: `${displayIdentifier(caseId)} · ${displayIdentifier(method)}`,
      metric: (item: BenchmarkCase) => metricAt(item.metrics, metricName, { case_id: caseId, method_id: method }),
    }
  })
}
function genericRows(id: BenchmarkV1CaseId): CompareRow[] {
  const entry = anchorCase(id)
  return entry ? entry.metrics.slice(0, 30).map((metric) => ({
    direction: metric.direction,
    key: metricKey(metric),
    label: `${displayIdentifier(metric.name)} · ${Object.values(metric.dimensions).filter((value) => value !== 'latency' && value !== 'throughput' && value !== 'memory').join(' · ')}`,
    metric: (item) => item.metrics.find((candidate) => metricKey(candidate) === metricKey(metric)),
  })) : []
}
function correctnessNames(id: BenchmarkV1CaseId): string[] {
  const entry = anchorCase(id)
  return entry ? entry.correctness.map((check) => check.name) : []
}
function correctnessValue(run: BenchmarkV1Run, id: BenchmarkV1CaseId, name: string): string {
  const entry = comparableCase(run, id)
  const check = entry?.correctness.find((item) => item.name === name)
  return check ? `✓ ${correctnessSummary(check)}` : ''
}
function syncQuery(): void {
  if (!mounted.value) return
  const url = new URL(window.location.href)
  url.searchParams.delete('runs')
  if (selectedIds.value.length) url.searchParams.set('runs', selectedIds.value.join(','))
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`)
}
onMounted(() => {
  const requestedIds = (new URLSearchParams(window.location.search).get('runs') ?? '')
    .split(',')
    .filter((id) => byId.has(id))
  selectedIds.value = [...new Set(requestedIds)].slice(0, 4)
  mounted.value = true
  syncQuery()
})
watch(selectedIds, syncQuery, { deep: true })
</script>

<template>
  <div class="benchmark-compare-shell">
    <BenchmarkPortalHeader />
    <main id="portal-content" class="compare-main">
      <header class="compare-hero">
        <div><p>Run comparison</p><h1>Compare runs</h1></div>
        <a :href="withBase('/benchmarks/')">Change selection</a>
      </header>
      <section v-if="!mounted || selectedRuns.length < 2" class="compare-empty"><h2>Select at least two runs</h2><p>Choose two to four runs from Results.</p><a :href="withBase('/benchmarks/')">Browse results</a></section>
      <section v-else-if="!valid" class="compare-empty"><h2>Use two to four runs</h2><a :href="withBase('/benchmarks/')">Revise selection</a></section>

      <template v-else>
        <div class="run-deck-scroll">
          <section class="run-deck" :style="{ '--compare-columns': selectedRuns.length, '--compare-min-width': `${220 + selectedRuns.length * 260}px` }" aria-label="Selected benchmark runs">
            <header class="run-deck-intro"><p>Comparison set</p><strong>{{ selectedRuns.length }} runs</strong></header>
            <article v-for="(run, index) in selectedRuns" :key="run.id">
              <header><span>Run {{ index + 1 }}</span><button type="button" :aria-label="`Remove run ${index + 1}`" @click="removeRun(run.id)">Remove</button></header>
              <strong class="run-execution-badge">{{ runExecutionBackendLabel(run) }} execution</strong>
              <h2><a :href="resultUrl(run)">{{ runExecutionHardwareSummary(run) }}</a></h2>
              <p>FHElium Benchmark v1 · FHElium {{ run.fhelium.version }} · <code>{{ shortCommit(run.fhelium.commit) }}</code><em v-if="run.fhelium.dirty"> · local changes</em></p>
              <footer><span><i :class="{ gaps: run.case_counts.unavailable > 0 }" aria-hidden="true"></i>{{ caseSummary(run) }}</span><time :datetime="run.recorded_at">{{ formatDate(run.recorded_at, true) }} UTC</time></footer>
            </article>
          </section>
        </div>

        <nav class="compare-case-rail" aria-label="Comparison sections">
          <a v-for="(id, index) in caseIds" :key="id" :href="`#compare-${id}`"><i :class="{ gap: selectedRuns.some((run) => !comparableCase(run, id)) }" aria-hidden="true"></i><span>{{ String(index + 1).padStart(2, '0') }}</span>{{ benchmarkV1CaseLabels[id] }}</a>
        </nav>

        <section v-for="(id, caseIndex) in caseIds" :id="`compare-${id}`" :key="id" class="case-group">
          <header>
            <div><p>{{ String(caseIndex + 1).padStart(2, '0') }} / {{ String(caseIds.length).padStart(2, '0') }} · {{ firstCase(id)?.category }}</p><h2>{{ firstCase(id)?.title ?? displayIdentifier(id) }}</h2></div>
            <div class="availability" aria-label="Run availability"><span v-for="(run, index) in selectedRuns" :key="run.id" :class="{ gap: !comparableCase(run, id) }" :title="comparableCase(run, id) ? `Run ${index + 1}: measured` : `Run ${index + 1}: ${gapLabel(run, id)}`" :aria-label="comparableCase(run, id) ? `Run ${index + 1}: measured` : `Run ${index + 1}: ${gapLabel(run, id)}`"><i aria-hidden="true"></i>R{{ index + 1 }}</span></div>
          </header>
          <details v-if="anchorCase(id)" class="case-identity"><summary>Benchmark v1 case identity</summary><code>{{ id }}</code></details>

          <template v-if="anchorCase(id)?.workload_id === 'ckks-depth-aware-single-operations'">
            <div class="controls"><label><span>Entry level</span><select v-model.number="operationLevel"><option v-for="level in operationLevels(id)" :key="level" :value="level">Level {{ level }}</option></select></label></div>
            <section class="metric-section is-lower"><header><h3>Operation latency <b class="direction-lower" :title="directionLabel('lower')">↓</b></h3></header><CompareTable :rows="singleRows(id)" :runs="selectedRuns" :case-id="id" :value="value" :gap-label="gapLabel" /></section>
          </template>

          <template v-else-if="anchorCase(id)?.workload_id === 'indexed-ntt-operations'">
            <div class="controls"><label><span>Basis</span><select v-model="nttBasis"><option>Q</option><option>QP</option></select></label><label><span>Operation</span><select v-model="nttOperation"><option value="forward_ntt">Forward NTT</option><option value="inverse_ntt">Inverse NTT</option><option value="roundtrip">Roundtrip</option></select></label><label><span>Entry level</span><select v-model.number="nttLevel"><option v-for="level in nttLevels(id)" :key="level" :value="level">Level {{ level }}</option></select></label></div>
            <section class="metric-section is-lower"><header><h3>Selected-cell latency <b class="direction-lower" :title="directionLabel('lower')">↓</b></h3></header><CompareTable :rows="nttRows(id)" :runs="selectedRuns" :case-id="id" :value="value" :gap-label="gapLabel" /></section>
          </template>

          <template v-else-if="anchorCase(id)?.workload_id.startsWith('dense-matrix-multiplication-')">
            <section class="metric-section is-lower"><header><h3>Latency <b class="direction-lower" :title="directionLabel('lower')">↓</b></h3></header><CompareTable :rows="matrixRows(id, 'lower')" :runs="selectedRuns" :case-id="id" :value="value" :gap-label="gapLabel" /></section>
            <section class="metric-section is-higher"><header><h3>Throughput <b class="direction-higher" :title="directionLabel('higher')">↑</b></h3></header><CompareTable :rows="matrixRows(id, 'higher')" :runs="selectedRuns" :case-id="id" :value="value" :gap-label="gapLabel" /></section>
          </template>

          <template v-else-if="anchorCase(id)?.workload_id === 'polynomial-evaluation'">
            <section class="metric-section is-lower"><header><h3>Method latency <b class="direction-lower" :title="directionLabel('lower')">↓</b></h3></header><CompareTable :rows="polynomialRows(id, 'polynomial-evaluation-latency', 'lower')" :runs="selectedRuns" :case-id="id" :value="value" :gap-label="gapLabel" /></section>
          </template>

          <section v-else class="metric-section"><header><h3>Reported performance</h3></header><CompareTable :rows="genericRows(id)" :runs="selectedRuns" :case-id="id" :value="value" :gap-label="gapLabel" /></section>

          <details v-if="correctnessNames(id).length" class="correctness"><summary>Validation evidence (collapsed by default)</summary><div class="table-wrap"><table><thead><tr><th scope="col">Check</th><th v-for="(run, index) in selectedRuns" :key="run.id" scope="col">Run {{ index + 1 }}</th></tr></thead><tbody><tr v-for="name in correctnessNames(id)" :key="name"><th scope="row">{{ displayIdentifier(name) }}</th><td v-for="run in selectedRuns" :key="run.id"><span v-if="correctnessValue(run, id, name)">{{ correctnessValue(run, id, name) }}</span><span v-else class="blank" :title="gapLabel(run, id)"></span></td></tr></tbody></table></div></details>
        </section>
      </template>
    </main>
  </div>
</template>

<style scoped>
.benchmark-compare-shell { min-height: calc(100vh - var(--vp-nav-height)); color: var(--vp-c-text-1); background: var(--vp-c-bg); }
.compare-main { max-width: 1440px; margin: 0 auto; padding: 22px 24px 76px; }
.compare-hero { display: flex; gap: 28px; align-items: end; justify-content: space-between; padding: 8px 0 12px; }
.compare-hero p, .case-group > header p { margin: 0 0 4px; color: var(--vp-c-brand-1); font-size: 10px; font-weight: 750; letter-spacing: .025em; text-transform: uppercase; }
.compare-hero h1 { margin: 0; border: 0; font-size: 30px; letter-spacing: -.025em; }
.compare-hero > a, .compare-empty a { flex: none; padding: 8px 11px; border: 1px solid var(--vp-c-brand-1); border-radius: 5px; color: white; background: var(--vp-c-brand-3); font-size: 11px; font-weight: 700; text-decoration: none; }
.compare-empty { display: grid; min-height: 360px; row-gap: 8px; place-content: center; justify-items: center; text-align: center; }
.compare-empty h2 { margin: 0; border: 0; }
.compare-empty p { margin: 0; color: var(--vp-c-text-2); }
.compare-empty > a { margin-top: 8px; }

.run-deck-scroll { margin-top: 15px; overflow-x: auto; border-radius: 13px; box-shadow: 0 5px 15px color-mix(in srgb, var(--vp-c-text-1) 5%, transparent), 0 18px 40px color-mix(in srgb, var(--vp-c-text-1) 8%, transparent); scrollbar-color: color-mix(in srgb, var(--vp-c-brand-1) 45%, transparent) transparent; scrollbar-width: thin; }
.run-deck { display: grid; min-width: var(--compare-min-width); grid-template-columns: 220px repeat(var(--compare-columns), minmax(0, 1fr)); overflow: hidden; border-radius: inherit; background: radial-gradient(circle at 98% -18%, color-mix(in srgb, #3152c7 10%, transparent), transparent 39%), linear-gradient(145deg, var(--vp-c-bg) 15%, color-mix(in srgb, var(--vp-c-bg-soft) 72%, var(--vp-c-bg))); }
.run-deck-intro { display: flex; min-width: 0; padding: 19px; flex-direction: column; justify-content: center; background: color-mix(in srgb, var(--vp-c-brand-soft) 32%, transparent); }
.run-deck-intro p { margin: 0 0 4px; color: var(--vp-c-brand-1); font-size: 10px; font-weight: 750; text-transform: uppercase; }
.run-deck-intro strong { font-size: 17px; }
.run-deck article { display: flex; min-width: 0; padding: 16px 18px 14px; border-left: 1px solid color-mix(in srgb, var(--vp-c-divider) 72%, transparent); flex-direction: column; }
.run-deck article > header, .run-deck article > footer { display: flex; gap: 12px; align-items: center; justify-content: space-between; }
.run-deck article > header > span { color: var(--vp-c-brand-1); font-size: 10px; font-weight: 750; text-transform: uppercase; }
.run-deck button { padding: 3px 6px; border: 0; border-radius: 4px; color: var(--vp-c-text-3); background: transparent; font: inherit; font-size: 9px; cursor: pointer; }
.run-deck button:hover, .run-deck button:focus-visible { color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); }
.run-deck h2 { margin: 11px 0 0; border: 0; font-size: 16px; line-height: 1.28; }
.run-execution-badge { width: fit-content; margin-top: 9px; padding: 3px 6px; border-radius: 999px; color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-size: 9px; letter-spacing: .025em; text-transform: uppercase; }
.run-deck h2 a { color: var(--vp-c-text-1); text-decoration: none; }
.run-deck h2 a:hover { color: var(--vp-c-brand-1); }
.run-deck article > p { margin: 7px 0 15px; color: var(--vp-c-text-2); font-size: 11px; }
.run-deck article > p code { color: inherit; font-size: inherit; }
.run-deck article > p em { color: var(--vp-c-warning-1); font-style: normal; }
.run-deck article > footer { margin-top: auto; padding-top: 10px; border-top: 1px solid color-mix(in srgb, var(--vp-c-divider) 62%, transparent); color: var(--vp-c-text-2); font-size: 10px; }
.run-deck article > footer span { display: inline-flex; gap: 6px; align-items: center; }
.run-deck article > footer i { width: 6px; height: 6px; border-radius: 50%; background: #267a5d; box-shadow: 0 0 0 3px color-mix(in srgb, #267a5d 12%, transparent); }
.run-deck article > footer i.gaps { border: 1px solid var(--vp-c-text-3); background: transparent; box-shadow: none; }

.compare-case-rail { display: flex; gap: 2px; margin: 16px 0 6px; padding: 5px; overflow-x: auto; border-radius: 8px; background: color-mix(in srgb, var(--vp-c-bg-soft) 72%, transparent); scrollbar-color: color-mix(in srgb, var(--vp-c-brand-1) 45%, transparent) transparent; scrollbar-width: thin; }
.compare-case-rail a { display: inline-flex; gap: 7px; align-items: center; min-width: max-content; padding: 7px 9px; border-radius: 5px; color: var(--vp-c-text-2); font-size: 11px; text-decoration: none; }
.compare-case-rail a:hover, .compare-case-rail a:focus-visible { color: var(--vp-c-text-1); background: var(--vp-c-bg); }
.compare-case-rail a > span { color: var(--vp-c-text-3); font-family: var(--vp-font-family-mono); font-size: 8px; }
.compare-case-rail i, .availability i { width: 7px; height: 7px; border-radius: 50%; background: #267a5d; }
.compare-case-rail i.gap, .availability span.gap i { border: 1px solid var(--vp-c-text-3); background: transparent; }

.case-group { margin-top: 34px; padding-top: 25px; border-top: 1px solid color-mix(in srgb, var(--vp-c-divider) 75%, transparent); scroll-margin-top: calc(var(--vp-nav-height) + 20px); }
.case-group > header { display: flex; gap: 22px; align-items: end; justify-content: space-between; }
.case-group h2 { margin: 0; border: 0; font-size: 24px; letter-spacing: -.015em; }
.availability { display: flex; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }
.availability span { display: inline-flex; gap: 6px; align-items: center; color: var(--vp-c-text-2); font-family: var(--vp-font-family-mono); font-size: 10px; }
.case-identity { width: fit-content; max-width: 100%; margin-top: 7px; color: var(--vp-c-text-3); font-size: 10px; }
.case-identity summary { cursor: pointer; }
.case-identity[open] { display: flex; gap: 5px; flex-wrap: wrap; }
.case-identity code { overflow-wrap: anywhere; color: var(--vp-c-text-2); font-size: 10px; }

.controls { display: flex; gap: 8px; margin-top: 16px; padding: 9px; width: fit-content; max-width: 100%; border-radius: 7px; background: color-mix(in srgb, var(--vp-c-bg-soft) 70%, transparent); }
.controls label span { display: block; margin-bottom: 3px; color: var(--vp-c-text-2); font-size: 10px; font-weight: 650; }
.controls select { height: 32px; padding: 0 8px; border: 1px solid var(--vp-c-divider); border-radius: 4px; color: var(--vp-c-text-1); background: var(--vp-c-bg); font: inherit; font-size: 11px; }
.metric-section { margin-top: 22px; }
.metric-section > header { display: flex; align-items: center; min-height: 27px; margin-bottom: 7px; }
.metric-section h3 { margin: 0; border: 0; font-size: 17px; }
.direction-higher { color: #267a5d; }
.direction-lower { color: #5d72cc; }
.dark .direction-higher { color: #6fd2ad; }
.dark .direction-lower { color: #a9b8ff; }

.table-wrap, .metric-section :deep(.table-wrap) { overflow-x: auto; border: 1px solid color-mix(in srgb, var(--vp-c-divider) 76%, transparent); border-radius: 9px; background: color-mix(in srgb, var(--vp-c-bg-soft) 22%, var(--vp-c-bg)); box-shadow: 0 7px 20px color-mix(in srgb, var(--vp-c-text-1) 4%, transparent); scrollbar-color: color-mix(in srgb, var(--vp-c-brand-1) 45%, transparent) transparent; scrollbar-width: thin; }
.table-wrap table, .metric-section :deep(.table-wrap table) { width: 100%; min-width: 720px; table-layout: fixed; margin: 0; border-collapse: separate; border-spacing: 0; font-size: 12px; }
.table-wrap th, .table-wrap td, .metric-section :deep(.table-wrap th), .metric-section :deep(.table-wrap td) { height: 42px; padding: 8px 14px; border: 0; border-bottom: 1px solid color-mix(in srgb, var(--vp-c-divider) 62%, transparent); vertical-align: middle; }
.table-wrap tr:last-child th, .table-wrap tr:last-child td, .metric-section :deep(.table-wrap tr:last-child th), .metric-section :deep(.table-wrap tr:last-child td) { border-bottom: 0; }
.table-wrap th:not(:first-child), .table-wrap td, .metric-section :deep(.table-wrap th:not(:first-child)), .metric-section :deep(.table-wrap td) { border-left: 1px solid color-mix(in srgb, var(--vp-c-divider) 48%, transparent); text-align: right; }
.table-wrap th:first-child, .metric-section :deep(.table-wrap th:first-child) { width: 34%; text-align: left; }
.table-wrap thead th, .metric-section :deep(.table-wrap thead th) { color: var(--vp-c-text-2); background: color-mix(in srgb, var(--vp-c-bg-soft) 70%, transparent); font-size: 11px; font-weight: 650; }
.table-wrap thead span, .table-wrap thead small, .metric-section :deep(.table-wrap thead span), .metric-section :deep(.table-wrap thead small) { display: block; }
.table-wrap thead small, .metric-section :deep(.table-wrap thead small) { margin-top: 2px; color: var(--vp-c-text-3); font-size: 10px; font-weight: 450; }
.table-wrap td strong, .metric-section :deep(.table-wrap td strong) { font-family: var(--vp-font-family-mono); font-size: 12px; white-space: nowrap; }
.blank, .metric-section :deep(.blank) { display: block; width: 100%; min-height: 1em; }
.correctness { margin-top: 20px; padding: 0 13px 12px; border: 1px solid color-mix(in srgb, var(--vp-c-divider) 72%, transparent); border-radius: 8px; background: color-mix(in srgb, var(--vp-c-bg-soft) 54%, transparent); }
.correctness summary { padding-top: 11px; color: var(--vp-c-text-2); font-size: 9px; font-weight: 700; cursor: pointer; }
.correctness .table-wrap { margin-top: 10px; box-shadow: none; }

@media (max-width: 760px) {
  .compare-main { padding-inline: 14px; }
  .compare-hero, .case-group > header { align-items: flex-start; flex-direction: column; }
  .compare-hero { padding-top: 5px; }
  .compare-hero h1 { font-size: 26px; }
  .run-deck-scroll { margin-inline: -3px; }
  .case-group { margin-top: 27px; padding-top: 20px; }
  .case-group h2 { font-size: 21px; }
  .availability { justify-content: flex-start; }
  .controls { align-items: stretch; flex-direction: column; width: 100%; }
  .controls select { width: 100%; }
}
@media (forced-colors: active) { .run-deck, .compare-case-rail, .table-wrap, .correctness { border: 1px solid CanvasText; } }
</style>
