<script setup lang="ts">
import { withBase } from 'vitepress'
import { computed, onMounted, ref } from 'vue'

type KernelRow = [family: string, startUs: number, durationUs: number]

interface TimelineMode {
  label: string
  window_us: number
  busy_us: number
  kernel_count: number
  kernels: KernelRow[]
}

interface TimelinePayload {
  workload: {
    name: string
    shape: string
    depth: number
    mode: string
    domain: string
    device: string
    source: string
  }
  modes: TimelineMode[]
  implementations?: { compiled?: string[] | null }
}

const FAMILIES = [
  'NTT',
  'Key switch',
  'Fused',
  'Arithmetic',
  'Copies',
  'Other',
] as const

const COLORS: Record<string, string> = {
  NTT: '--timeline-ntt',
  'Key switch': '--timeline-keyswitch',
  Fused: '--timeline-fused',
  Arithmetic: '--timeline-rns',
  Copies: '--timeline-elementwise',
  Other: '--timeline-other',
}

const VIEW_WIDTH = 1000
const VIEW_HEIGHT = 46

const data = ref<TimelinePayload>()
const status = ref<'loading' | 'ready' | 'error'>('loading')

const maximumWindow = computed(() =>
  Math.max(...(data.value?.modes ?? []).map(mode => mode.window_us), 1),
)

const ticks = computed(() => {
  const maximum = maximumWindow.value / 1000
  const step = maximum > 12 ? 2 : 1
  const values: number[] = []
  for (let value = 0; value <= Math.ceil(maximum); value += step) values.push(value)
  return values
})

function scale(value: number): number {
  return (value / maximumWindow.value) * VIEW_WIDTH
}

function barWidth(durationUs: number): number {
  return Math.max(0.6, scale(durationUs))
}

function color(family: string): string {
  return `var(${COLORS[family] ?? COLORS.Other})`
}

function milliseconds(value: number): string {
  return (value / 1000).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function kernelTitle(mode: TimelineMode, kernel: KernelRow): string {
  return `${mode.label}: ${kernel[0]}, ${(kernel[2] / 1000).toFixed(3)} milliseconds`
}

onMounted(async () => {
  try {
    const response = await fetch(withBase('/assets/compile-timeline.json'))
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    data.value = (await response.json()) as TimelinePayload
    status.value = 'ready'
  } catch {
    status.value = 'error'
  }
})
</script>

<template>
  <section class="compile-timeline" aria-label="Eager and compiled kernel timelines for one BSGS matrix-vector call">
    <div class="chart-toolbar">
      <div class="chart-legend" aria-label="Kernel families">
        <span v-for="family in FAMILIES" :key="family">
          <i class="legend-dot" :style="{ background: color(family) }" />{{ family }}
        </span>
      </div>
      <div class="chart-units">
        <span>Time within one call · ms</span>
        <span>Both rows share one scale</span>
      </div>
    </div>

    <p v-if="status === 'loading'" class="timeline-status">Loading measured timelines…</p>
    <p v-else-if="status === 'error'" class="timeline-status">The timeline data could not be loaded.</p>

    <template v-else-if="data">
      <p class="timeline-context">
        {{ data.workload.name }} · Depth {{ data.workload.depth }} · {{ data.workload.device }}
      </p>
      <div class="timeline-rows" role="group">
        <div v-for="mode in data.modes" :key="mode.label" class="timeline-row">
          <div class="timeline-label">
            <strong>{{ mode.label }}</strong>
          </div>
          <div class="timeline-track">
            <svg
              :viewBox="`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`"
              preserveAspectRatio="none"
              role="img"
              :aria-label="`${mode.label}: ${milliseconds(mode.window_us)} milliseconds per call`"
            >
              <rect
                v-for="(kernel, index) in mode.kernels"
                :key="index"
                :x="scale(kernel[1])"
                y="6"
                :width="barWidth(kernel[2])"
                height="34"
                :fill="color(kernel[0])"
              >
                <title>{{ kernelTitle(mode, kernel) }}</title>
              </rect>
            </svg>
          </div>
          <div class="timeline-stats">
            <strong>{{ milliseconds(mode.window_us) }} ms</strong>
            <small>{{ milliseconds(mode.busy_us) }} ms GPU busy</small>
          </div>
        </div>
        <div class="timeline-axis" aria-hidden="true">
          <span
            v-for="tick in ticks"
            :key="tick"
            :style="{ left: `${(tick * 1000 / maximumWindow) * 100}%` }"
          >
            {{ tick }}
          </span>
        </div>
      </div>
    </template>
  </section>
</template>

<style scoped>
.compile-timeline {
  --timeline-ntt: #5b7fc7;
  --timeline-keyswitch: #c07a2c;
  --timeline-fused: #087e6d;
  --timeline-rns: #8a4f2e;
  --timeline-elementwise: #7c8698;
  --timeline-other: #b6bdc9;
  margin: 20px 0 30px;
  padding: 14px;
  border: 1px solid color-mix(in srgb, var(--fhe-c-border) 72%, var(--fhe-c-divider));
  border-radius: 12px;
  background:
    radial-gradient(circle at 84% 0%, color-mix(in srgb, var(--fhe-c-brand) 6%, transparent), transparent 36%),
    var(--fhe-c-surface);
  box-shadow: 0 2px 4px color-mix(in srgb, var(--fhe-c-text-1) 4%, transparent), 0 22px 54px color-mix(in srgb, var(--fhe-c-text-1) 9%, transparent);
}

:global(.dark .compile-timeline) {
  --timeline-ntt: #8fb0ef;
  --timeline-keyswitch: #e0a463;
  --timeline-fused: #69d6bb;
  --timeline-rns: #d79a72;
  --timeline-elementwise: #9aa5b4;
  --timeline-other: #66707e;
}

.chart-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px 20px;
  padding: 2px 4px 12px;
  line-height: 1.4;
}
.chart-legend { display: flex; flex-wrap: wrap; gap: 8px 16px; color: var(--vp-c-text-2); font-size: 11px; }
.chart-legend span { display: inline-flex; align-items: center; gap: 6px; }
.chart-units { display: flex; flex-wrap: wrap; gap: 4px 14px; color: var(--vp-c-text-3); font-size: 10px; }
.legend-dot { width: 8px; height: 8px; border-radius: 2px; }

.timeline-status { margin: 0; padding: 8px 4px; color: var(--vp-c-text-3); font-size: 12px; }
.timeline-context { margin: 0 0 10px; padding: 0 4px; color: var(--vp-c-text-3); font-size: 10px; }

.timeline-rows {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr) 132px;
  gap: 6px 12px;
  align-items: center;
}
.timeline-row { display: contents; }
.timeline-label { color: var(--vp-c-text-2); font-size: 12px; text-align: right; }
.timeline-label strong { font-weight: 600; }
.timeline-track {
  border: 1px solid color-mix(in srgb, var(--fhe-c-border) 60%, transparent);
  border-radius: 6px;
  background: color-mix(in srgb, var(--fhe-c-surface-tint) 70%, var(--fhe-c-surface));
  overflow: hidden;
}
.timeline-track svg { display: block; width: 100%; height: 46px; }
.timeline-stats { display: flex; flex-direction: column; gap: 1px; color: var(--vp-c-text-3); font-size: 10px; }
.timeline-stats strong { color: var(--vp-c-text-1); font-size: 12px; font-weight: 600; }

.timeline-axis { grid-column: 2; position: relative; height: 14px; margin-top: 2px; }
.timeline-axis span {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  color: var(--vp-c-text-3);
  font-size: 9px;
  white-space: nowrap;
}
.timeline-axis span:first-child { transform: none; }

@media (max-width: 720px) {
  .timeline-rows { grid-template-columns: 60px minmax(0, 1fr); }
  .timeline-stats { grid-column: 2; flex-direction: row; flex-wrap: wrap; gap: 4px 12px; }
}
</style>
