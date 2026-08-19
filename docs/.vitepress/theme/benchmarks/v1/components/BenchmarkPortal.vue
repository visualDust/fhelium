<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import type { BenchmarkV1Run } from '../data/catalog'
import { benchmarkCatalog } from '../data/catalogClient'
import type { BenchmarkV1ExecutionBackend } from '../data/specification'
import {
  compareUrl,
  cpuSummary,
  executionBackendLabel,
  gpuModels,
  runExecutionBackendLabel,
  runExecutionBackends,
} from '../data/ui'
import BenchmarkPortalHeader from './BenchmarkPortalHeader.vue'
import BenchmarkRunRecord from './BenchmarkRunRecord.vue'

const maximumComparisons = 4
const runs = [...benchmarkCatalog.runs].sort((left, right) => right.recorded_at.localeCompare(left.recorded_at))
const query = ref('')
const versionFilter = ref('')
const gpuFilter = ref('')
const backendFilter = ref<BenchmarkV1ExecutionBackend | ''>('')
const sortOrder = ref<'newest' | 'oldest'>('newest')
const selectedIds = ref<string[]>([])
const mounted = ref(false)

function unique<T extends string>(values: T[]): T[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right, 'en', { numeric: true }))
}

const versions = unique(runs.map((run) => run.fhelium.version))
const gpus = unique(runs.flatMap(gpuModels))
const executionBackends: BenchmarkV1ExecutionBackend[] = ['cpu', 'cuda']

const filteredRuns = computed(() => {
  const needle = query.value.trim().toLocaleLowerCase()
  const selected = runs.filter((run) => {
    const searchable = [
      'FHElium Benchmark v1',
      run.fhelium.version,
      runExecutionBackendLabel(run),
      cpuSummary(run),
      ...gpuModels(run),
      ...run.cases.flatMap((entry) => [entry.title, entry.category, entry.workload_id]),
    ].join(' ').toLocaleLowerCase()
    return (
      (!needle || searchable.includes(needle)) &&
      (!versionFilter.value || run.fhelium.version === versionFilter.value) &&
      (!backendFilter.value || runExecutionBackends(run).includes(backendFilter.value)) &&
      (!gpuFilter.value || gpuModels(run).includes(gpuFilter.value))
    )
  })
  return [...selected].sort((left, right) => sortOrder.value === 'oldest'
    ? left.recorded_at.localeCompare(right.recorded_at)
    : right.recorded_at.localeCompare(left.recorded_at))
})

function canSelect(run: BenchmarkV1Run): boolean {
  return selectedIds.value.includes(run.id) || selectedIds.value.length < maximumComparisons
}

function setSelected(run: BenchmarkV1Run, checked: boolean): void {
  if (checked && canSelect(run)) selectedIds.value = [...selectedIds.value, run.id]
  else selectedIds.value = selectedIds.value.filter((id) => id !== run.id)
}

function reset(): void {
  query.value = ''
  versionFilter.value = ''
  gpuFilter.value = ''
  backendFilter.value = ''
  sortOrder.value = 'newest'
}

function syncQuery(): void {
  if (!mounted.value) return
  const url = new URL(window.location.href)
  for (const key of ['q', 'version', 'backend', 'gpu', 'sort', 'select']) url.searchParams.delete(key)
  if (query.value.trim()) url.searchParams.set('q', query.value.trim())
  if (versionFilter.value) url.searchParams.set('version', versionFilter.value)
  if (backendFilter.value) url.searchParams.set('backend', backendFilter.value)
  if (gpuFilter.value) url.searchParams.set('gpu', gpuFilter.value)
  if (sortOrder.value !== 'newest') url.searchParams.set('sort', sortOrder.value)
  if (selectedIds.value.length) url.searchParams.set('select', selectedIds.value.join(','))
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`)
}

onMounted(() => {
  const params = new URLSearchParams(window.location.search)
  query.value = params.get('q') ?? ''
  versionFilter.value = versions.includes(params.get('version') ?? '') ? params.get('version')! : ''
  const requestedBackend = params.get('backend')
  backendFilter.value = executionBackends.includes(requestedBackend as BenchmarkV1ExecutionBackend)
    ? requestedBackend as BenchmarkV1ExecutionBackend
    : ''
  gpuFilter.value = gpus.includes(params.get('gpu') ?? '') ? params.get('gpu')! : ''
  if (params.get('sort') === 'oldest') sortOrder.value = 'oldest'
  const available = new Set(runs.map((run) => run.id))
  const requestedIds = (params.get('select') ?? '')
    .split(',')
    .filter((id) => available.has(id))
  selectedIds.value = [...new Set(requestedIds)].slice(0, maximumComparisons)
  mounted.value = true
  syncQuery()
})
watch([query, versionFilter, backendFilter, gpuFilter, sortOrder, selectedIds], syncQuery, { deep: true })
</script>

<template>
  <div class="benchmark-portal">
    <BenchmarkPortalHeader />
    <main id="portal-content" class="portal-main">
      <section class="portal-toolbar" aria-label="Find Benchmark v1 runs">
        <label class="portal-search"><span>Search benchmark runs</span><input v-model="query" type="search" placeholder="GPU, CPU, workload, FHElium…"></label>
        <div class="portal-filters">
          <label><span>FHElium</span><select v-model="versionFilter"><option value="">All versions</option><option v-for="version in versions" :key="version" :value="version">{{ version }}</option></select></label>
          <label><span>Execution</span><select v-model="backendFilter"><option value="">All backends</option><option v-for="backend in executionBackends" :key="backend" :value="backend">{{ executionBackendLabel(backend) }}</option></select></label>
          <label><span>Accelerator</span><select v-model="gpuFilter"><option value="">All accelerators</option><option v-for="gpu in gpus" :key="gpu" :value="gpu">{{ gpu }}</option></select></label>
          <label><span>Sort</span><select v-model="sortOrder"><option value="newest">Newest</option><option value="oldest">Oldest</option></select></label>
          <button type="button" @click="reset">Reset</button>
        </div>
      </section>

      <header class="portal-results-heading">
        <div><p>Published runs</p><h1>{{ filteredRuns.length }} {{ filteredRuns.length === 1 ? 'run' : 'runs' }}</h1></div><p>Every card is one complete formal Benchmark v1 report.</p>
      </header>

      <section v-if="!runs.length" class="portal-empty" role="status"><h2>No benchmark runs</h2></section>
      <section v-else-if="!filteredRuns.length" class="portal-empty" role="status"><h2>No run matches these filters</h2><button type="button" @click="reset">Reset filters</button></section>

      <section v-else class="portal-card-grid" aria-label="Benchmark v1 runs">
        <BenchmarkRunRecord
          v-for="run in filteredRuns"
          :key="run.id"
          :run="run"
          :selected="selectedIds.includes(run.id)"
          :can-select="canSelect(run)"
          @select="setSelected(run, $event)"
        />
      </section>
    </main>

    <aside v-if="selectedIds.length" class="portal-compare-bar" aria-live="polite">
      <strong>{{ selectedIds.length }} {{ selectedIds.length === 1 ? 'run' : 'runs' }} selected</strong>
      <div><button type="button" @click="selectedIds = []">Clear</button><a v-if="selectedIds.length >= 2" :href="compareUrl(selectedIds)">Compare runs</a><span v-else>Select one more run</span></div>
    </aside>
  </div>
</template>

<style scoped>
.benchmark-portal { min-height: calc(100vh - var(--vp-nav-height)); color: var(--vp-c-text-1); background: var(--vp-c-bg); }
.portal-main { max-width: 1440px; margin: 0 auto; padding: 22px 24px 84px; }
.portal-toolbar { display: flex; gap: 20px; align-items: end; justify-content: space-between; }
.portal-search { width: min(460px, 100%); }
.portal-search span, .portal-filters label span { display: block; margin-bottom: 5px; color: var(--vp-c-text-2); font-size: 10px; font-weight: 650; }
.portal-search input, .portal-filters select { width: 100%; height: 38px; padding: 0 11px; border: 1px solid var(--vp-c-border); border-radius: 4px; color: var(--vp-c-text-1); background: var(--vp-c-bg); font: inherit; font-size: 12px; }
.portal-filters { display: grid; grid-template-columns: repeat(4, minmax(110px, 1fr)) auto; gap: 8px; align-items: end; }
.portal-filters button, .portal-empty button { height: 40px; padding: 0 13px; border: 1px solid var(--vp-c-border); border-radius: 5px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); font: inherit; font-size: 10px; cursor: pointer; }
.portal-results-heading { display: flex; gap: 24px; align-items: end; justify-content: space-between; margin: 32px 0 15px; }
.portal-results-heading p { margin: 0; color: var(--vp-c-text-2); font-size: 12px; }
.portal-results-heading > div > p { margin-bottom: 4px; color: var(--vp-c-brand-1); font-size: 10px; font-weight: 700; }
.portal-results-heading h1 { margin: 0; border: 0; font-size: 25px; }
.portal-empty { min-height: 340px; padding: 60px 24px; border: 1px dashed var(--vp-c-border); border-radius: 8px; text-align: center; }
.portal-empty h2 { margin: 0; border: 0; }.portal-empty p { color: var(--vp-c-text-2); }
.portal-card-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 13px; }
.portal-compare-bar { position: fixed; right: 24px; bottom: 24px; left: 24px; z-index: 20; display: flex; max-width: 840px; align-items: center; justify-content: space-between; margin: auto; padding: 12px 15px; border: 1px solid var(--vp-c-brand-1); border-radius: 7px; background: color-mix(in srgb, var(--vp-c-bg) 94%, transparent); box-shadow: var(--fhe-shadow-panel); backdrop-filter: blur(14px); }.portal-compare-bar div { display: flex; gap: 8px; align-items: center; }.portal-compare-bar button, .portal-compare-bar a { padding: 7px 10px; border: 0; border-radius: 4px; color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); font: inherit; font-size: 10px; text-decoration: none; cursor: pointer; }.portal-compare-bar a { color: white; background: var(--vp-c-brand-3); }.portal-compare-bar span { color: var(--vp-c-text-2); font-size: 9px; }
.visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
@media (max-width: 1050px) { .portal-toolbar { align-items: stretch; flex-direction: column; }.portal-search { width: 100%; }.portal-filters { grid-template-columns: repeat(4, minmax(0, 1fr)) auto; } }
@media (max-width: 720px) { .portal-main { padding-inline: 14px; }.portal-filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }.portal-results-heading { align-items: start; flex-direction: column; }.portal-compare-bar { right: 10px; bottom: 10px; left: 10px; } }
@media (max-width: 460px) { .portal-compare-bar { align-items: stretch; flex-direction: column; gap: 8px; }.portal-compare-bar div { justify-content: space-between; } }
@media (forced-colors: active) { .portal-empty, .portal-compare-bar { border: 1px solid CanvasText; } }
</style>
