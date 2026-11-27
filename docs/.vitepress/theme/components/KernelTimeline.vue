<script setup lang="ts">
import { useData } from 'vitepress'
import { computed } from 'vue'

import type { KernelRow, KernelTimelineChart, KernelTimelineRow } from './kernelTimeline'

const props = defineProps<{ chart: KernelTimelineChart }>()
const { isDark } = useData()

const VIEW_WIDTH = 1000
const VIEW_HEIGHT = 46

const maximumWindowUs = computed(() =>
  Math.max(...props.chart.rows.map(row => row.windowMs * 1000), 1),
)

const ticks = computed(() => {
  const maximum = maximumWindowUs.value / 1000
  const step = maximum > 12 ? 2 : 1
  const values: number[] = []
  for (let value = 0; value <= Math.floor(maximum); value += step) values.push(value)
  return values
})

function scale(value: number): number {
  return (value / maximumWindowUs.value) * VIEW_WIDTH
}

function barWidth(durationUs: number): number {
  return Math.max(0.6, scale(durationUs))
}

function color(family: number): string {
  const entry = props.chart.families[family]
  return isDark.value ? entry.dark : entry.light
}

function milliseconds(value: number): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function kernelTitle(row: KernelTimelineRow, kernel: KernelRow): string {
  return `${row.label}: ${props.chart.families[kernel[0]].label}, ${(kernel[2] / 1000).toFixed(3)} milliseconds`
}
</script>

<template>
  <section class="kernel-timeline" :aria-label="chart.ariaLabel">
    <div class="chart-toolbar">
      <div class="chart-legend" aria-label="Kernel families">
        <span v-for="(family, index) in chart.families" :key="family.label">
          <i class="legend-dot" :style="{ background: color(index) }" />{{ family.label }}
        </span>
      </div>
      <div class="chart-units">
      </div>
    </div>

    <p class="timeline-context">{{ chart.context }}</p>
    <div class="timeline-rows" role="group">
      <div v-for="row in chart.rows" :key="row.label" class="timeline-row">
        <div class="timeline-label">
          <strong>{{ row.label }}</strong>
        </div>
        <div class="timeline-track">
          <svg
            :viewBox="`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`"
            preserveAspectRatio="none"
            role="img"
            :aria-label="`${row.label}: ${milliseconds(row.medianMs)} milliseconds median per call`"
          >
            <rect
              v-for="(kernel, index) in row.kernels"
              :key="index"
              :x="scale(kernel[1])"
              y="6"
              :width="barWidth(kernel[2])"
              height="34"
              :fill="color(kernel[0])"
            >
              <title>{{ kernelTitle(row, kernel) }}</title>
            </rect>
          </svg>
        </div>
        <div class="timeline-stats">
          <strong>{{ milliseconds(row.medianMs) }} ms</strong>
          <small>median per call</small>
          <small>{{ milliseconds(row.busyMs) }} ms GPU busy</small>
        </div>
      </div>
      <div class="timeline-axis" aria-hidden="true">
        <span
          v-for="tick in ticks"
          :key="tick"
          :style="{ left: `${(tick * 1000 / maximumWindowUs) * 100}%` }"
        >
          {{ tick }}
        </span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.kernel-timeline {
  margin: 20px 0 30px;
  padding: 14px;
  border: 1px solid color-mix(in srgb, var(--fhe-c-border) 72%, var(--fhe-c-divider));
  border-radius: 12px;
  background:
    radial-gradient(circle at 84% 0%, color-mix(in srgb, var(--fhe-c-brand) 6%, transparent), transparent 36%),
    var(--fhe-c-surface);
  box-shadow: 0 2px 4px color-mix(in srgb, var(--fhe-c-text-1) 4%, transparent), 0 22px 54px color-mix(in srgb, var(--fhe-c-text-1) 9%, transparent);
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
