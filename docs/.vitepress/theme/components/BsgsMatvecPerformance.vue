<script setup lang="ts">
import {
  depthConfigurations,
  platformMeasurements,
  type PlatformMeasurement,
  type WorkloadKind,
  type WorkloadMeasurement,
} from '../data/bsgsMatvecPerformance'

const workloads = [
  { key: 'ptCt', label: 'PT×CT' },
  { key: 'ctCt', label: 'CT×CT' },
] as const satisfies readonly { key: WorkloadKind; label: string }[]

function formatInteger(value: number): string {
  return value.toLocaleString('en-US')
}

function formatLatency(value: number): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: value < 100 ? 2 : 1,
    maximumFractionDigits: value < 100 ? 2 : 1,
  })
}

function series(measurement: WorkloadMeasurement) {
  return [
    { key: 'reference', label: 'Reference', value: measurement.referenceMs },
    { key: 'jit', label: 'JIT', value: measurement.fheliumJitMs },
  ] as const
}

function speedup(measurement: WorkloadMeasurement, value: number): string {
  return `${(measurement.referenceMs / value).toFixed(2)}×`
}

function cellMaximum(
  platform: PlatformMeasurement,
  depth: 7 | 16 | 34,
): number {
  return Math.max(
    ...workloads.flatMap(({ key }) => {
      const measurement = platform.results[depth][key]
      return series(measurement).map(entry => entry.value)
    }),
  )
}

function barWidth(value: number, maximum: number): string {
  return `${(value / maximum) * 100}%`
}

function metricLabel(
  platform: PlatformMeasurement,
  workload: string,
  measurement: WorkloadMeasurement,
): string {
  return `${workload}, ${platform.reference}: ${series(measurement).map(entry =>
    `${entry.label} ${formatLatency(entry.value)} milliseconds`,
  ).join('; ')}`
}
</script>

<template>
  <section class="performance-chart" aria-label="Measured BSGS matrix-vector performance">
    <div class="chart-toolbar">
      <div class="chart-legend" aria-label="Implementations">
        <span><i class="legend-dot reference-dot" />Reference</span>
        <span><i class="legend-dot jit-dot" />FHElium JIT</span>
      </div>
      <div class="chart-units">
        <span>Latency · ms</span>
        <span>Speedup · relative to reference</span>
      </div>
    </div>
    <div
      class="chart-scroll"
      role="region"
      aria-label="BSGS packed matrix-vector latency chart"
      tabindex="0"
    >
      <div class="chart-grid">
        <div class="axis-corner">Platform / depth</div>

        <div
          v-for="configuration in depthConfigurations"
          :key="`header-${configuration.depth}`"
          class="depth-heading"
        >
          <strong>
            Depth {{ configuration.depth }}
            <span>(N={{ formatInteger(configuration.ringDimension) }})</span>
          </strong>
          <small>
            Q {{ configuration.qBits[0] }}/{{ configuration.qBits[1] }} ·
            QP {{ configuration.qpBits[0] }}/{{ configuration.qpBits[1] }} bits
          </small>
        </div>

        <template v-for="platform in platformMeasurements" :key="platform.key">
          <div class="platform-cell">
            <strong>{{ platform.label }}</strong>
            <span>{{ platform.hardware }}</span>
            <small>{{ platform.workload }}</small>
            <small>{{ platform.reference }} / {{ platform.measured }}</small>
          </div>

          <div
            v-for="configuration in depthConfigurations"
            :key="`${platform.key}-${configuration.depth}`"
            class="depth-cell"
          >
            <div
              v-for="workload in workloads"
              :key="workload.key"
              class="metric"
              :aria-label="metricLabel(
                platform,
                workload.label,
                platform.results[configuration.depth][workload.key],
              )"
            >
              <div class="metric-heading">
                <strong>{{ workload.label }}</strong>
              </div>
              <div
                v-for="entry in series(platform.results[configuration.depth][workload.key])"
                :key="entry.key"
                class="series-row"
                :class="`${entry.key}-series`"
              >
                <span class="series-label visually-hidden">{{ entry.label }}</span>
                <span class="bar-track" aria-hidden="true">
                  <i :style="{
                    width: barWidth(entry.value, cellMaximum(platform, configuration.depth)),
                  }" />
                </span>
                <span class="series-value">{{ formatLatency(entry.value) }}</span>
                <small class="series-speedup">
                  {{ entry.key === 'reference' ? '' : speedup(platform.results[configuration.depth][workload.key], entry.value) }}
                </small>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>

  </section>
</template>

<style scoped>
.performance-chart {
  --chart-reference: #8a4f2e;
  --chart-jit: #087e6d;
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
.legend-dot { width: 8px; height: 8px; border-radius: 999px; }
.reference-dot { background: var(--chart-reference); }
.jit-dot { background: var(--chart-jit); }
:global(.dark .performance-chart) { --chart-reference: #e4a273; --chart-jit: #69d6bb; }

.chart-scroll { overflow-x: auto; padding: 2px; border-radius: 10px; }
.chart-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.chart-grid {
  display: grid;
  grid-template-columns: 158px repeat(3, minmax(226px, 1fr));
  gap: 8px;
  min-width: 860px;
  line-height: 1.4;
}
.axis-corner { align-self: end; padding: 10px 12px; color: var(--vp-c-text-3); font-size: 10px; }
.depth-heading,
.platform-cell,
.depth-cell {
  border: 1px solid color-mix(in srgb, var(--fhe-c-border) 72%, var(--fhe-c-divider));
  border-radius: 9px;
  background: var(--fhe-c-surface);
}
.depth-heading {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 3px;
  padding: 9px 10px;
  background: var(--fhe-c-surface-tint);
  text-align: center;
}
.depth-heading strong { font-size: 13px; }
.depth-heading strong span { color: var(--vp-c-text-3); font-size: 10px; font-weight: 500; }
.depth-heading small { color: var(--vp-c-text-3); font-size: 9px; }
.platform-cell {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 2px;
  padding: 12px;
  background: color-mix(in srgb, var(--fhe-c-brand) 5%, var(--fhe-c-surface));
}
.platform-cell strong { margin-bottom: 3px; font-size: 15px; }
.platform-cell span { color: var(--vp-c-text-2); font-size: 10px; }
.platform-cell small { color: var(--vp-c-text-3); font-size: 9px; }
.depth-cell {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  padding: 12px 10px;
  background: color-mix(in srgb, var(--fhe-c-surface) 94%, var(--fhe-c-surface-tint));
}
.metric { min-width: 0; padding-right: 10px; }
.metric + .metric { padding-right: 0; padding-left: 10px; border-left: 1px solid var(--vp-c-divider); }
.metric-heading { margin-bottom: 8px; }
.metric-heading > strong { font-size: 10px; }
.series-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  grid-template-areas: 'value speedup' 'bar bar';
  align-items: baseline;
  column-gap: 4px;
  row-gap: 4px;
  font-variant-numeric: tabular-nums;
  font-size: 11px;
}
.series-row + .series-row { margin-top: 9px; }
.series-value { grid-area: value; font-weight: 650; white-space: nowrap; }
.timing-note { display: block; margin-top: 7px; color: var(--vp-c-text-3); font-size: 9px; line-height: 1.4; }
.series-speedup { grid-area: speedup; text-align: right; font-size: 9px; white-space: nowrap; }
.reference-series { color: var(--chart-reference); }
.jit-series { color: var(--chart-jit); }
.bar-track {
  grid-area: bar;
  display: block;
  height: 5px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--vp-c-text-3) 12%, transparent);
}
.bar-track i { display: block; height: 100%; border-radius: inherit; background: currentColor; }
.visually-hidden {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}
@media (max-width: 700px) {
  .performance-chart { padding: 12px 8px; }
  .chart-toolbar { gap: 8px; }
}
</style>
