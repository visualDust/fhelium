<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'

export interface TrendPoint { id: string; x: number; value: number | null; low?: number; high?: number; reason?: string }
export interface TrendSeries { id: string; label: string; color: string; points: TrendPoint[] }
const props = defineProps<{
  title: string
  context: string
  xLabel: string
  yLabel: string
  xValues: number[]
  xTicks: number[]
  ordinal?: boolean
  logarithmic?: boolean
  series: TrendSeries[]
  selectedId?: string
  selectedSeries?: string
  emptyLabel?: string
  interactive?: boolean
  highlightBest?: 'min' | 'max'
  info?: string
}>()
const emit = defineEmits<{ select: [series: string, point: string] }>()
const host = ref<HTMLElement | null>(null)
const width = ref(550)
let observer: ResizeObserver | undefined
onMounted(() => {
  observer = new ResizeObserver(entries => { width.value = Math.max(260, Math.floor(entries[0].contentRect.width)) })
  if (host.value) observer.observe(host.value)
})
onUnmounted(() => observer?.disconnect())
const points = computed(() => props.series.flatMap(s => s.points).filter(p => p.value !== null))
const bestValues = computed(() => new Map(props.series.map(s => {
  const values = s.points.filter(p => p.value !== null).map(p => p.value!)
  return [s.id, props.highlightBest === 'min' ? Math.min(...values) : Math.max(...values)]
})))
function highlighted(series: TrendSeries, point: TrendPoint): boolean {
  return (props.selectedId === point.id && props.selectedSeries === series.id)
    || (!!props.highlightBest && point.value !== null && point.value === bestValues.value.get(series.id))
}
const top = 24, bottom = 250, left = 58
const right = computed(() => width.value - 20)
const high = computed(() => {
  const max = Math.max(...points.value.map(p => p.high ?? p.value!), 0.01) * 1.12
  if (props.logarithmic) return 10 ** Math.ceil(Math.log10(max))
  const step = 10 ** Math.floor(Math.log10(max)) / 2
  return Math.ceil(max / step) * step
})
const low = computed(() => props.logarithmic ? 10 ** Math.floor(Math.log10((points.value.length ? Math.min(...points.value.map(p => p.low ?? p.value!)) : 0.1))) : 0)
const yTicks = computed(() => props.logarithmic
  ? Array.from({ length: Math.round(Math.log10(high.value / low.value)) + 1 }, (_, i) => low.value * 10 ** i)
  : Array.from({ length: 5 }, (_, i) => high.value * i / 4))
function x(value: number): number {
  const start = props.xValues[0], end = props.xValues[props.xValues.length - 1]
  const ratio = props.ordinal ? props.xValues.indexOf(value) / Math.max(1, props.xValues.length - 1) : (value - start) / Math.max(1, end - start)
  return left + ratio * (right.value - left)
}
function y(value: number): number {
  const ratio = props.logarithmic ? Math.log10(value / low.value) / Math.log10(high.value / low.value) : value / high.value
  return bottom - ratio * (bottom - top)
}
function line(values: TrendPoint[]): string {
  let connected = false
  return [...values].sort((a, b) => a.x - b.x).map(p => {
    if (p.value === null) { connected = false; return '' }
    const command = connected ? 'L' : 'M'; connected = true
    return `${command}${x(p.x)},${y(p.value)}`
  }).join(' ')
}
const exclusions = computed(() => props.series.flatMap(s => {
  const reasons = new Map<string, Set<number>>()
  for (const p of s.points) {
    if (p.value !== null || !p.reason) continue
    const positions = reasons.get(p.reason) ?? new Set<number>()
    positions.add(p.x)
    reasons.set(p.reason, positions)
  }
  return [...reasons].map(([reason, positions]) =>
    `${s.label}: ${reason} (${props.xLabel} ${[...positions].sort((a, b) => a - b).join(', ')})`)
}))
const number = (v: number) => Number(v.toPrecision(3)).toLocaleString()
</script>

<template>
  <section class="trend-panel">
    <header><div class="title-row"><h3>{{ title }}</h3><slot name="controls" /></div><span>{{ context }}</span><div class="legend"><span v-for="s in series" :key="s.id"><i :style="{ background: s.color }" />{{ s.label }}</span></div></header>
    <div ref="host" class="plot">
      <div v-if="!points.length" class="empty">{{ emptyLabel ?? 'No measurements for this selection' }}</div>
      <svg v-else :viewBox="`0 0 ${width} 304`" role="group" :aria-label="title">
        <text x="5" y="13" class="axis-title">{{ yLabel }}{{ logarithmic ? ' · log scale' : '' }}</text>
        <g v-for="tick in yTicks" :key="`y${tick}`"><line :x1="left" :x2="right" :y1="y(tick)" :y2="y(tick)" class="grid" /><text :x="left - 9" :y="y(tick) + 4" text-anchor="end" class="tick">{{ number(tick) }}</text></g>
        <text v-for="tick in xTicks" :key="`x${tick}`" :x="x(tick)" y="275" text-anchor="middle" class="tick">{{ tick }}</text>
        <text :x="(left + right) / 2" y="299" text-anchor="middle" class="axis-title">{{ xLabel }}</text>
        <g v-for="s in series" :key="s.id">
          <path :d="line(s.points)" :stroke="s.color" fill="none" stroke-width="2" />
          <template v-for="p in s.points" :key="p.id"><g v-if="p.value !== null">
            <template v-if="p.low !== undefined && p.high !== undefined"><line :x1="x(p.x)" :x2="x(p.x)" :y1="y(p.low)" :y2="y(p.high)" :stroke="s.color" stroke-width="1.5" /><line v-for="bound in [p.low, p.high]" :key="bound" :x1="x(p.x)-4" :x2="x(p.x)+4" :y1="y(bound)" :y2="y(bound)" :stroke="s.color" stroke-width="1.5" /></template>
            <circle :cx="x(p.x)" :cy="y(p.value)" :r="highlighted(s, p) ? 6 : 4.5" :fill="s.color" :class="['point', { interactive, selected: highlighted(s, p) }]" :tabindex="interactive ? 0 : undefined" :role="interactive ? 'button' : undefined" :aria-label="`${s.label}: ${number(p.value)}; ${xLabel} ${p.x}`" @click="interactive && emit('select', s.id, p.id)" @keydown.enter.prevent="interactive && emit('select', s.id, p.id)" @keydown.space.prevent="interactive && emit('select', s.id, p.id)"><title>{{ s.label }} · {{ number(p.value) }}</title></circle>
          </g></template>
        </g>
      </svg>
    </div>
    <footer v-if="info || exclusions.length || $slots.footer" class="chart-footer">
      <div v-if="info || exclusions.length" class="missing-info">
        <button type="button" class="info-button" :aria-label="[info, exclusions.length ? `Missing points. ${exclusions.join('; ')}` : ''].filter(Boolean).join('. ')"><svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.5" /><path d="M10 9v5" /><circle cx="10" cy="6" r=".7" class="info-dot" /></svg></button>
        <div class="info-tooltip" role="tooltip"><p v-if="info">{{ info }}</p><strong v-if="exclusions.length">Missing points</strong><p v-for="item in exclusions" :key="item">{{ item }}</p></div>
      </div>
      <slot name="footer" />
    </footer>
  </section>
</template>

<style scoped>
.trend-panel { min-width:0; border:1px solid var(--vp-c-divider); border-radius:10px; overflow:hidden; }
header { position:relative; padding:17px 18px 13px; border-bottom:1px solid var(--vp-c-divider); }
.title-row { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.chart-footer { position:relative; display:flex; align-items:center; justify-content:space-between; gap:12px; border-top:1px solid var(--vp-c-divider); padding:12px 18px; }
.info-button { display:flex; align-items:center; justify-content:center; width:22px; height:22px; color:var(--vp-c-text-3); cursor:help; border-radius:50%; }
.info-button svg { width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.4; stroke-linecap:round; }
.info-button .info-dot { fill:currentColor; stroke:none; }
.info-button:hover,.info-button:focus-visible { color:var(--vp-c-text-1); }
.info-button:focus-visible { outline:2px solid var(--vp-c-brand-1); outline-offset:2px; }
.info-tooltip { position:absolute; z-index:5; bottom:calc(100% - 6px); left:12px; right:12px; padding:12px 14px; border:1px solid var(--vp-c-divider); border-radius:8px; background:var(--vp-c-bg-elv); box-shadow:var(--vp-shadow-2); color:var(--vp-c-text-2); font-size:11px; line-height:1.6; visibility:hidden; opacity:0; pointer-events:none; }
.missing-info:hover .info-tooltip,.missing-info:focus-within .info-tooltip { visibility:visible; opacity:1; pointer-events:auto; }
.info-tooltip strong { color:var(--vp-c-text-1); font-weight:600; }
.info-tooltip p { margin:5px 0 0; font-size:inherit; line-height:inherit; overflow-wrap:anywhere; }
h3 { margin:0; font-size:15px; font-weight:600; line-height:1.5; }
header>span { display:block; margin-top:3px; font-size:11px; color:var(--vp-c-text-2); }
.legend { display:flex; flex-wrap:wrap; gap:7px 15px; margin-top:9px; font-size:10px; color:var(--vp-c-text-2); }
.legend span { display:flex; align-items:center; gap:6px; line-height:1.5; }
.legend i { flex-shrink:0; width:7px; height:7px; border-radius:50%; }
.plot { margin:16px 9px 13px; }
svg { display:block; width:100%; }
.axis-title,.tick { fill:var(--vp-c-text-2); font-size:10px; }
.grid { stroke:var(--vp-c-divider); stroke-width:1; opacity:.6; }
.point { stroke:var(--vp-c-bg); stroke-width:2; }
.point.interactive { cursor:pointer; }
.point.selected,.point:hover,.point:focus-visible { stroke:var(--vp-c-text-1); outline:none; }
.empty { min-height:304px; display:grid; place-content:center; text-align:center; color:var(--vp-c-text-3); font-size:12px; padding:20px; }
</style>
