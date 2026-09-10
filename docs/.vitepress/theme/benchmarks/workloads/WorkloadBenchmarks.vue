<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { withBase } from 'vitepress'
import BenchmarkTrend from '../explorer/BenchmarkTrend.vue'
import PrimitiveOptions from './PrimitiveOptions.vue'
import NavArrow from '../../components/NavArrow.vue'
import { hardwareLabel, runs, runUrl } from '../explorer/records'
import { batches, curves, depths, sizes, workloads, requiredPolynomialDepths, operationNotes, isMatrix, isPrimitive, type Mode, type WorkloadId } from './workloadData'

const props = defineProps<{ workloadId?: WorkloadId; recordId?: string }>()
const record = computed(() => runs.find(r => r.id === props.recordId))
const runMethods = ref<Record<WorkloadId, number>>({ ptct: 0, ctct: 0, polynomial: 0, ntt: 0, keyswitch: 0, rns: 0 })
const polynomialMethods = ['Balanced', 'Horner', 'Paterson–Stockmeyer']
const matrixOperations = [...new Set(workloads.filter(w => isMatrix(w.id)).flatMap(w => w.operations))].map(name => {
  const owner = workloads.find(w => isMatrix(w.id) && w.operations.includes(name))!
  return { name, workload: owner.id, index: owner.operations.indexOf(name) }
})
const polynomialOperations = workloads[2].operations.filter(name => !matrixOperations.some(op => op.name === name)).map(name => ({ name, workload: 'polynomial' as WorkloadId, index: workloads[2].operations.indexOf(name) }))
const current = computed(() => workloads.find(w => w.id === props.workloadId))
const mode = ref<Mode>('throughput')
const logarithmic = ref(false)
const showCpu = ref(true)
const showGpu = ref(true)
function toggleDevice(kind: 'cpu' | 'gpu') {
  if (kind === 'cpu') {
    if (!showCpu.value || showGpu.value) showCpu.value = !showCpu.value
  } else if (!showGpu.value || showCpu.value) showGpu.value = !showGpu.value
}
const depthIndex = ref(0)
const depth = computed(() => depths[depthIndex.value])
const sizeIndex = ref(2)
const size = computed(() => sizes[sizeIndex.value])
const selected = ref<{ hardware: string; x: number; value: number; low: number; high: number; title: string }>()
const methods = computed(() => current.value?.id === 'polynomial' ? ['Balanced', 'Horner', 'Paterson–Stockmeyer'] : ['Shared preparation', 'Independent rotations'])
const method = ref(0)
const rotationHoisting = ref(true)
const inverseNtt = ref(false)
watch([mode, depthIndex, sizeIndex, method, runMethods, rotationHoisting, inverseNtt, showCpu, showGpu], () => { selected.value = undefined }, { deep: true })
function panel(id: WorkloadId, operation?: number) {
  const w = workloads.find(w => w.id === id)!
  const xValues = batches
  const chosenMethod = record.value || isPrimitive(id) ? runMethods.value[id] : props.workloadId ? method.value : 0
  const series = curves(id, mode.value, depth.value, size.value, chosenMethod, props.recordId, operation, rotationHoisting.value, inverseNtt.value).filter(s => {
    const cpu = runs.find(r => r.id === s.id)!.platform.device === 'cpu'
    return cpu ? showCpu.value : showGpu.value
  })
  const insufficientDepth = id === 'polynomial' && operation === undefined && depth.value + requiredPolynomialDepths[chosenMethod] > 27
  return {
    title: operation === undefined ? (props.workloadId && !isPrimitive(id) ? 'Complete workload' : w.title) : w.operations[operation],
    context: `${!isMatrix(id) || operation !== undefined ? '' : `${size.value} × ${size.value} · `}Entry depth ${depth.value}${operation === undefined ? '' : ` · ${operationNotes[w.operations[operation]]}`}`,
    xLabel: 'Batch size',
    yLabel: mode.value === 'throughput' ? 'Throughput (tasks/s)' : 'Amortized latency (ms/task)',
    xValues, xTicks: xValues, ordinal: true, series, logarithmic: logarithmic.value,
    highlightBest: mode.value === 'throughput' ? 'max' as const : 'min' as const,
    emptyLabel: insufficientDepth ? `Requires ${requiredPolynomialDepths[chosenMethod]} remaining depths` : 'Not measured',
  }
}
function infoFor(id: WorkloadId): string {
  const choice = record.value || isPrimitive(id) ? runMethods.value[id] : props.workloadId ? method.value : 0
  if (isMatrix(id)) return choice === 0 ? 'Hoisting enabled' : 'Independent rotations'
  if (id === 'polynomial') return `${polynomialMethods[choice]} evaluation`
  if (id === 'ntt') {
    const cpuUsesStaged = choice === 0 && showCpu.value && runs.some(r => (!props.recordId || r.id === props.recordId) && r.platform.device === 'cpu')
    return `One RNS polynomial per task · ${inverseNtt.value ? 'NTT/Montgomery to coefficient/standard' : 'Coefficient/standard to NTT/Montgomery'} · ${choice === 0 ? 'Grouped radix-2 stages with a shared-memory tail (CUDA)' : 'Indexed radix-2 stages'} · output allocation included${cpuUsesStaged ? '. CPU uses Staged radix-2 because Grouped stages is CUDA-only; the CPU and GPU curves use different implementations.' : ''}`
  }
  return workloads.find(w => w.id === id)!.method
}
function pick(id: WorkloadId, seriesId: string, pointId: string, operation?: number) {
  const chart = panel(id, operation)
  const series = chart.series.find(s => s.id === seriesId)!
  const point = series.points.find(p => p.id === pointId)!
  selected.value = { hardware: series.label, x: point.x, value: point.value!, low: point.low!, high: point.high!, title: chart.title }
}
</script>

<template>
  <main class="workload-benchmarks">
    <a v-if="current || record" class="back" :href="withBase('/benchmarks/')">‹ All benchmarks</a>
    <header class="heading">
      <div class="heading-top"><h1>{{ record ? hardwareLabel(record) : current?.title ?? 'Benchmarks' }}</h1>
        <nav class="benchmark-links" aria-label="Benchmark documentation">
          <a :href="withBase('/benchmarks/methodology')">Methodology <NavArrow /></a>
          <a :href="withBase('/benchmarks/run-and-submit')">Run &amp; submit <NavArrow /></a>
          <a class="submit-results" href="https://github.com/visualDust/fhelium/issues/new?template=benchmark_results.yml">Submit results <NavArrow /></a>
        </nav>
      </div>
      <p v-if="record">{{ record.started_at.slice(0, 10) }} <span>·</span> {{ record.platform.os }} <span>·</span> PyTorch {{ record.platform.torch }}</p>
      <p>N = 65,536 <span>·</span> Scale 2<sup>50</sup><template v-if="current"> <span>·</span> {{ current.subtitle }}</template></p>
    </header>

    <div class="pinned-controls">
      <section class="controls" aria-label="Benchmark controls">
        <div class="mode-switch">
          <span :class="{ active: mode === 'scaling' }">Scaling</span>
          <button class="mode-toggle" role="switch" aria-label="Throughput mode" :aria-checked="mode === 'throughput'" @click="mode = mode === 'scaling' ? 'throughput' : 'scaling'; selected = undefined"><span class="toggle-thumb" /></button>
          <span :class="{ active: mode === 'throughput' }">Throughput</span>
        </div>
        <div class="log-control"><span>Log Y</span><button class="mode-toggle" role="switch" aria-label="Logarithmic Y axis" :aria-checked="logarithmic" @click="logarithmic = !logarithmic"><span class="toggle-thumb" /></button></div>
        <div v-if="!record" class="device-buttons" role="group" aria-label="Displayed device types">
          <button type="button" :aria-pressed="showCpu" :disabled="showCpu && !showGpu" @click="toggleDevice('cpu')">CPU</button>
          <button type="button" :aria-pressed="showGpu" :disabled="showGpu && !showCpu" @click="toggleDevice('gpu')">GPU</button>
        </div>
        <label class="slider"><span>Entry depth <output>{{ depth }}</output></span><input v-model.number="depthIndex" type="range" min="0" :max="depths.length - 1" step="1" :aria-valuetext="`Entry depth ${depth}`"><span class="ticks"><span v-for="d in depths" :key="d">{{ d }}</span></span></label>
        <label v-if="!current || isMatrix(current.id)" class="slider"><span>Matrix size <output>{{ size }} × {{ size }}</output></span><input v-model.number="sizeIndex" type="range" min="0" :max="sizes.length - 1" step="1" :aria-valuetext="`Matrix ${size} by ${size}`"><span class="ticks"><span v-for="s in sizes" :key="s">{{ s }}</span></span></label>
      </section>
    </div>

    <template v-if="record">
      <section class="run-workloads">
        <div class="section-heading"><h2>Workloads and operations</h2></div>
        <div class="workload-grid">
          <article v-for="w in workloads" :key="w.id" class="workload-card">
            <BenchmarkTrend v-bind="panel(w.id)" :info="infoFor(w.id)" interactive @select="(series, point) => pick(w.id, series, point)"><template v-if="isPrimitive(w.id)" #controls><PrimitiveOptions :id="w.id" v-model:method="runMethods[w.id]" v-model:inverse="inverseNtt" /></template></BenchmarkTrend>
            <footer v-if="!isPrimitive(w.id)">
              <div v-if="isMatrix(w.id)" class="hoisting-control"><span>Hoisting</span><button class="mode-toggle" role="switch" :aria-label="`${w.title} hoisting`" :aria-checked="runMethods[w.id] === 0" @click="runMethods[w.id] = runMethods[w.id] === 0 ? 1 : 0"><span class="toggle-thumb" /></button><span class="toggle-state">{{ runMethods[w.id] === 0 ? 'On' : 'Off' }}</span></div>
              <div v-else class="methods" role="group" aria-label="Polynomial method"><button v-for="(name, index) in polynomialMethods" :key="name" :class="{ active: runMethods.polynomial === index }" :aria-pressed="runMethods.polynomial === index" @click="runMethods.polynomial = index">{{ name }}</button></div>
            </footer>
          </article>
        </div>
      </section>
      <section class="operations-section">
        <div class="section-heading"><h2>Matrix operations</h2><span>{{ matrixOperations.length }} operations</span></div>
        <div class="operation-grid"><BenchmarkTrend v-for="op in matrixOperations" :key="op.name" v-bind="panel(op.workload, op.index)" interactive @select="(series, point) => pick(op.workload, series, point, op.index)"><template v-if="op.name === 'Rotate-many (7 offsets)'" #controls><div class="hoisting-control"><span>Hoisting</span><button class="mode-toggle" role="switch" aria-label="Rotate-many hoisting" :aria-checked="rotationHoisting" @click="rotationHoisting = !rotationHoisting"><span class="toggle-thumb" /></button><span class="toggle-state">{{ rotationHoisting ? 'On' : 'Off' }}</span></div></template></BenchmarkTrend></div>
      </section>
      <section class="operations-section">
        <div class="section-heading"><h2>Polynomial operations</h2><span>Additional operations</span></div>
        <div class="operation-grid"><BenchmarkTrend v-for="op in polynomialOperations" :key="op.name" v-bind="panel(op.workload, op.index)" interactive @select="(series, point) => pick(op.workload, series, point, op.index)"><template v-if="op.name === 'Rotate-many (7 offsets)'" #controls><div class="hoisting-control"><span>Hoisting</span><button class="mode-toggle" role="switch" aria-label="Rotate-many hoisting" :aria-checked="rotationHoisting" @click="rotationHoisting = !rotationHoisting"><span class="toggle-thumb" /></button><span class="toggle-state">{{ rotationHoisting ? 'On' : 'Off' }}</span></div></template></BenchmarkTrend></div>
      </section>
      <section v-if="selected" class="point-details" aria-live="polite"><div><h3>{{ selected.title }}</h3><p>Batch {{ selected.x }} · entry depth {{ depth }}</p><p>Interquartile range {{ selected.low.toFixed(2) }}–{{ selected.high.toFixed(2) }} {{ mode === 'scaling' ? 'ms/task' : 'tasks/s' }}</p></div><strong>{{ selected.value.toFixed(2) }} <small>{{ mode === 'scaling' ? 'ms/task' : 'tasks/s' }}</small></strong></section>
      <details class="conditions"><summary>Measurement conditions</summary><div class="condition-grid">
        <div><span>Hardware</span><strong>{{ record.platform.name }}</strong></div>
        <div><span>Host CPU</span><strong>{{ record.platform.cpu }}</strong></div>
        <div><span>Intra-op / inter-op threads</span><strong>{{ record.platform.torch_threads }} / {{ record.platform.torch_interop_threads }}</strong></div>
        <div><span>PyTorch / CUDA</span><strong>{{ record.platform.torch }} / {{ record.platform.cuda }}</strong></div>
        <div v-if="record.platform.gpu_power_limit_w"><span>GPU power limit</span><strong>{{ record.platform.gpu_power_limit_w }} W</strong></div>
        <div v-if="record.platform.host_allocation"><span>Host allocation</span><strong>{{ record.platform.host_allocation }}</strong></div>
        <div><span>Execution</span><strong>Eager CKKS and Backend RNS/NTT · prepared inputs and resources</strong></div>
        <div v-for="extra in record.additional_runs ?? []" :key="extra.id"><span>Additional measurements · {{ extra.started_at.slice(0, 10) }}</span><strong>{{ extra.recorded_case_count }} cases · {{ extra.id }}</strong></div>
      </div></details>
    </template>
    <template v-else-if="!current">
      <div class="workload-grid">
        <article v-for="w in workloads" :key="w.id" class="workload-card">
          <BenchmarkTrend v-bind="panel(w.id)" :info="infoFor(w.id)"><template v-if="isPrimitive(w.id)" #controls><PrimitiveOptions :id="w.id" v-model:method="runMethods[w.id]" v-model:inverse="inverseNtt" /></template><template #footer><a class="view-link" :href="withBase(`/benchmarks/workloads/${w.id}`)">View <NavArrow /></a></template></BenchmarkTrend>
        </article>
      </div>
      <section class="runs-section">
        <div class="section-heading"><h2>Benchmark runs</h2><span>{{ runs.length }} runs</span></div>
        <div class="run-grid">
          <a v-for="run in runs" :key="run.id" class="run-card" :href="withBase(runUrl(run.id))">
            <div class="run-top"><span>{{ run.started_at.slice(0, 10) }}</span><NavArrow /></div>
            <h3>{{ hardwareLabel(run) }}</h3>
            <p>{{ run.platform.os }} · PyTorch {{ run.platform.torch }}</p>
            <span class="run-count">{{ run.recorded_case_count }} cases in original run</span>
          </a>
        </div>
      </section>
    </template>
    <template v-else>
      <section class="workload-section">
        <div class="section-heading"><h2>{{ isPrimitive(current.id) ? 'Operation performance' : 'Workload performance' }}</h2><PrimitiveOptions v-if="isPrimitive(current.id)" :id="current.id" v-model:method="runMethods[current.id]" v-model:inverse="inverseNtt" /><div v-else-if="isMatrix(current.id)" class="hoisting-control"><span>Hoisting</span><button class="mode-toggle" role="switch" aria-label="Hoisting" :aria-checked="method === 0" @click="method = method === 0 ? 1 : 0; selected = undefined"><span class="toggle-thumb" /></button><span class="toggle-state">{{ method === 0 ? 'On' : 'Off' }}</span></div><div v-else-if="current.id === 'polynomial'" class="methods" role="group" aria-label="Execution method"><button v-for="(name, index) in methods" :key="name" :class="{ active: method === index }" :aria-pressed="method === index" @click="method = index; selected = undefined">{{ name }}</button></div></div>
        <BenchmarkTrend v-bind="panel(current.id)" :info="infoFor(current.id)" interactive @select="(series, point) => pick(current!.id, series, point)" />
      </section>
      <section v-if="current.operations.length" class="operations-section">
        <div class="section-heading"><h2>Operations</h2><span>{{ current.operations.length }} operations</span></div>
        <div class="operation-grid"><BenchmarkTrend v-for="(name, index) in current.operations" :key="name" v-bind="panel(current.id, index)" interactive @select="(series, point) => pick(current!.id, series, point, index)"><template v-if="name === 'Rotate-many (7 offsets)'" #controls><div class="hoisting-control"><span>Hoisting</span><button class="mode-toggle" role="switch" aria-label="Rotate-many hoisting" :aria-checked="rotationHoisting" @click="rotationHoisting = !rotationHoisting"><span class="toggle-thumb" /></button><span class="toggle-state">{{ rotationHoisting ? 'On' : 'Off' }}</span></div></template></BenchmarkTrend></div>
      </section>
      <section v-if="selected" class="point-details"><div><h3>{{ selected.title }}</h3><p>{{ selected.hardware }}</p></div><strong>{{ selected.value.toFixed(2) }} <small>{{ mode === 'scaling' ? 'ms/task' : 'tasks/s' }}</small></strong></section>
      <section class="conditions"><h2>Measurement conditions</h2><div class="condition-grid"><div><span>Configuration</span><strong>N = 65,536 · scale 2<sup>50</sup></strong></div><div v-if="current.id === 'polynomial'"><span>Polynomial</span><strong>Degree twelve · same coefficients across methods</strong></div><div><span>Execution</span><strong>{{ isPrimitive(current.id) && current.id !== 'keyswitch' ? 'RNS/NTT execution · prepared inputs and tables' : 'Eager · prepared inputs and keys' }}</strong></div><div><span>Platforms</span><strong>{{ runs.map(hardwareLabel).join(' / ') }}</strong></div></div></section>
    </template>
  </main>
</template>

<style scoped>
.workload-benchmarks { max-width:1440px; margin:auto; padding:32px 32px 70px; color:var(--vp-c-text-1); }
.heading { margin-bottom:18px; }
.heading-top { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px 24px; }
.benchmark-links { display:flex; align-items:center; flex-wrap:wrap; gap:12px 20px; }
.benchmark-links a { display:inline-flex; align-items:center; gap:6px; color:var(--vp-c-brand-1); font-size:12px; font-weight:550; }
.benchmark-links .submit-results { padding:8px 12px; border:1px solid var(--vp-c-divider); border-radius:7px; background:transparent; }
.benchmark-links .submit-results:hover { border-color:var(--vp-c-brand-1); background:var(--vp-c-brand-soft); }
h1 { margin:0; font-size:32px; font-weight:650; letter-spacing:-.035em; line-height:1.35; }
.heading p { margin:9px 0 0; font-size:13px; color:var(--vp-c-text-2); }
.heading p span { padding:0 7px; }
.back { display:inline-block; margin-bottom:16px; color:var(--vp-c-text-2); font-size:12px; }
.pinned-controls { position:sticky; top:var(--vp-nav-height,64px); z-index:20; padding:12px 0 18px; background:var(--vp-c-bg); }
.controls { display:flex; flex-wrap:wrap; align-items:center; gap:28px; padding:20px 24px; border:1px solid var(--vp-c-divider); border-radius:12px; background:var(--vp-c-bg-soft); }
.controls>.slider { flex:1 1 180px; }
.log-control { display:flex; align-items:center; gap:14px; font-size:12px; font-weight:600; color:var(--vp-c-text-2); }
.device-buttons { display:flex; gap:6px; }
.device-buttons button { padding:6px 12px; border:1px solid var(--vp-c-divider); border-radius:6px; color:var(--vp-c-text-2); background:var(--vp-c-bg); font-size:12px; font-weight:600; cursor:pointer; }
.device-buttons button[aria-pressed="true"] { color:var(--vp-c-brand-1); background:var(--vp-c-brand-soft); border-color:var(--vp-c-brand-1); }
.device-buttons button:disabled { cursor:default; }
.mode-switch { display:grid; grid-template-columns:max-content 42px max-content; align-items:center; justify-self:start; gap:14px; }
.mode-switch>span { color:var(--vp-c-text-2); font-size:12px; font-weight:600; white-space:nowrap; }
.mode-switch>span:first-child { text-align:right; }
.mode-switch>span.active { color:var(--vp-c-text-1); }
.mode-toggle { position:relative; flex-shrink:0; width:42px; height:24px; border:1px solid var(--vp-c-divider); border-radius:12px; background:var(--vp-c-bg); cursor:pointer; }
.toggle-thumb { position:absolute; top:3px; left:3px; width:16px; height:16px; border-radius:50%; background:var(--vp-c-text-2); transition:transform .15s, background .15s; }
.mode-toggle[aria-checked="true"] { background:var(--vp-c-brand-soft); border-color:var(--vp-c-brand-1); }
.mode-toggle[aria-checked="true"] .toggle-thumb { transform:translateX(18px); background:var(--vp-c-brand-1); }
@media(prefers-reduced-motion:reduce) { .toggle-thumb { transition:none; } }
.slider { display:flex; flex-direction:column; gap:8px; min-width:0; font-size:12px; color:var(--vp-c-text-2); }
.slider>span:first-child { display:flex; justify-content:space-between; }
.slider output { color:var(--vp-c-text-1); font-weight:600; font-variant-numeric:tabular-nums; }
.slider input { width:100%; margin:0; accent-color:var(--vp-c-brand-1); cursor:pointer; }
.ticks { display:flex; justify-content:space-between; font-size:10px; padding:0 5px; }
.workload-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:20px; margin-top:8px; }
.workload-card { min-width:0; border:1px solid var(--vp-c-divider); border-radius:12px; overflow:hidden; background:var(--vp-c-bg); }
.workload-card :deep(.trend-panel) { border:0; border-radius:0; }
.workload-card :deep(header) { min-height:116px; }
.workload-card footer { display:flex; flex-direction:column; align-items:flex-start; gap:12px; border-top:1px solid var(--vp-c-divider); padding:17px 18px; }
.view-link { display:flex; align-items:center; gap:8px; margin-left:auto; font-size:13px; color:var(--vp-c-brand-1); font-weight:550; }
h2 { margin:0; border:0; padding:0; font-size:20px; font-weight:600; letter-spacing:-.02em; }
.section-heading { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:15px; margin:12px 0 17px; }
.section-heading>span { color:var(--vp-c-text-2); font-size:12px; }
.methods { display:flex; flex-wrap:wrap; gap:4px; border-radius:7px; padding:4px; background:var(--vp-c-bg-soft); }
.hoisting-control { display:flex; align-items:center; gap:14px; font-size:12px; color:var(--vp-c-text-2); }
.toggle-state { width:22px; }
.methods button { font-size:12px; padding:7px 12px; cursor:pointer; border-radius:4px; color:var(--vp-c-text-2); }
.methods button.active { background:var(--vp-c-bg); color:var(--vp-c-brand-1); }
.workload-section :deep(.plot) { max-width:900px; margin:16px auto; }
.operations-section { margin-top:32px; }
.runs-section { margin-top:36px; }
.run-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }
.run-card { display:block; padding:20px; border:1px solid var(--vp-c-divider); border-radius:10px; background:var(--vp-c-bg); }
.run-card:hover { border-color:var(--vp-c-brand-1); }
.run-top { display:flex; align-items:center; justify-content:space-between; color:var(--vp-c-text-2); font-size:11px; }
.run-card h3 { margin:12px 0 8px; font-size:15px; font-weight:600; line-height:1.5; }
.run-card p,.run-count { font-size:12px; color:var(--vp-c-text-2); }
.run-card p { margin:0 0 14px; }
@media(max-width:700px) { .run-grid { grid-template-columns:1fr; } }
.operation-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }
.point-details { display:flex; justify-content:space-between; align-items:center; margin-top:24px; border:1px solid var(--vp-c-divider); background:var(--vp-c-bg-soft); border-radius:9px; padding:18px; }
.point-details h3 { margin:0; font-size:14px; }.point-details p { margin:4px 0 0; font-size:12px; color:var(--vp-c-text-2); }.point-details strong { font-size:25px; }.point-details small { font-size:12px; font-weight:400; }
.conditions { margin-top:32px; }.conditions summary { cursor:pointer; font-size:18px; font-weight:600; }.condition-grid { display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:20px; }.condition-grid>div { display:flex; flex-direction:column; gap:5px; font-size:12px; }.condition-grid span { color:var(--vp-c-text-2); }.condition-grid strong { font-weight:500; }
a:focus-visible,button:focus-visible,input:focus-visible { outline:2px solid var(--vp-c-brand-1); outline-offset:3px; }
@media(max-width:1100px) { .workload-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }.controls { gap:20px; } }
@media(max-width:700px) { .workload-benchmarks { padding:24px 16px 40px; }h1 { font-size:27px; }.controls { padding:16px; gap:18px; }.controls>.slider { flex-basis:100%; }.workload-grid,.operation-grid,.condition-grid { grid-template-columns:1fr; }.workload-card :deep(header) { min-height:0; }.section-heading { align-items:flex-start; }.pinned-controls { padding:8px 0 12px; } }
</style>
