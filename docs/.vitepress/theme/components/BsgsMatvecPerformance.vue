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

function speedup(measurement: WorkloadMeasurement): string {
  return `${(measurement.referenceMs / measurement.fheliumMs).toFixed(2)}×`
}

function cellMaximum(
  platform: PlatformMeasurement,
  depth: 7 | 16 | 34,
): number {
  return Math.max(
    ...workloads.flatMap(({ key }) => {
      const measurement = platform.results[depth][key]
      return [measurement.referenceMs, measurement.fheliumMs]
    }),
  )
}

function barWidth(value: number, maximum: number): string {
  return `${Math.max(3, (value / maximum) * 100)}%`
}

function metricLabel(
  platform: PlatformMeasurement,
  workload: string,
  measurement: WorkloadMeasurement,
): string {
  return `${workload}: ${platform.reference} ${formatLatency(measurement.referenceMs)} milliseconds; ${platform.measured} ${formatLatency(measurement.fheliumMs)} milliseconds`
}
</script>

<template>
  <section class="performance-chart" aria-label="Measured BSGS matrix-vector performance">
    <div
      class="chart-scroll"
      role="region"
      aria-label="BSGS packed matrix-vector latency chart"
      tabindex="0"
    >
      <div class="chart-grid">
        <div class="legend-cell">
          <div>
            <span><i class="legend-dot reference-dot" />reference</span>
            <span><i class="legend-dot fhelium-dot" />FHElium</span>
          </div>
          <small>lower is better</small>
        </div>

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
                <span class="latencies">
                  <span
                    class="reference-value"
                  >
                    {{ formatLatency(platform.results[configuration.depth][workload.key].referenceMs) }}
                  </span>
                  <span aria-hidden="true">/</span>
                  <span class="fhelium-value">
                    {{ formatLatency(platform.results[configuration.depth][workload.key].fheliumMs) }}
                  </span>
                  <small>ms</small>
                  <em>
                    {{ speedup(platform.results[configuration.depth][workload.key]) }}
                  </em>
                </span>
              </div>
              <div class="bar-pair" aria-hidden="true">
                <span
                  class="bar-track reference-track"
                >
                  <i
                    :style="{
                      width: barWidth(
                        platform.results[configuration.depth][workload.key].referenceMs,
                        cellMaximum(platform, configuration.depth),
                      ),
                    }"
                  />
                </span>
                <span class="bar-track fhelium-track">
                  <i
                    :style="{
                      width: barWidth(
                        platform.results[configuration.depth][workload.key].fheliumMs,
                        cellMaximum(platform, configuration.depth),
                      ),
                    }"
                  />
                </span>
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
  margin: 20px 0 30px;
  padding: 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background:
    radial-gradient(circle at 96% 4%, color-mix(in srgb, var(--vp-c-brand-1) 10%, transparent), transparent 32%),
    color-mix(in srgb, var(--vp-c-bg-soft) 64%, var(--vp-c-bg));
  box-shadow: 0 14px 34px color-mix(in srgb, var(--vp-c-text-1) 7%, transparent);
}

.chart-scroll { overflow-x: auto; padding: 2px; border-radius: 12px; }
.chart-scroll:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }

.chart-grid {
  display: grid;
  grid-template-columns: 190px repeat(3, minmax(218px, 1fr));
  gap: 8px;
  min-width: 900px;
}

.legend-cell,
.depth-heading,
.platform-cell,
.depth-cell {
  border: 1px solid color-mix(in srgb, var(--vp-c-divider) 78%, transparent);
  border-radius: 10px;
  background: color-mix(in srgb, var(--vp-c-bg) 90%, transparent);
}

.legend-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  padding: 10px;
  color: var(--vp-c-text-2);
  font-size: 10px;
}

.legend-cell div { display: flex; gap: 14px; }
.legend-cell span { display: inline-flex; align-items: center; gap: 5px; }
.legend-cell small { color: var(--vp-c-text-3); font-size: 9px; }
.legend-dot { width: 8px; height: 8px; border-radius: 999px; }
.reference-dot { background: #a8663f; }
.fhelium-dot { background: var(--vp-c-brand-1); }
:global(.dark) .reference-dot { background: #e4a273; }

.depth-heading { padding: 9px 12px; text-align: center; }
.depth-heading strong,
.depth-heading small { display: block; }
.depth-heading strong { font-size: 13px; }
.depth-heading strong span { color: var(--vp-c-text-3); font-size: 10px; font-weight: 500; }
.depth-heading small { margin-top: 1px; color: var(--vp-c-text-3); font-size: 9px; }

.platform-cell {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 104px;
  padding: 12px 14px;
  border-left: 3px solid var(--vp-c-brand-1);
}

.platform-cell strong { font-size: 15px; }
.platform-cell span { color: var(--vp-c-text-2); font-size: 10px; line-height: 1.4; }
.platform-cell small { color: var(--vp-c-text-3); font-size: 9px; line-height: 1.4; }

.depth-cell {
  display: grid;
  align-content: center;
  gap: 9px;
  min-height: 104px;
  padding: 10px 12px;
}

.metric + .metric { padding-top: 8px; border-top: 1px dashed var(--vp-c-divider); }
.metric-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.metric-heading > strong { font-size: 10px; }
.latencies { display: inline-flex; align-items: baseline; gap: 4px; font-variant-numeric: tabular-nums; font-size: 10px; }
.reference-value { color: #8a4f2e; font-weight: 650; }
:global(.dark) .reference-value { color: #e4a273; }
.fhelium-value { color: var(--vp-c-brand-1); font-weight: 700; }
.latencies small { color: var(--vp-c-text-3); font-size: 8px; }
.latencies em {
  margin-left: 3px;
  padding: 1px 5px;
  border-radius: 999px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-size: 8px;
  font-style: normal;
  font-weight: 700;
}

.bar-pair { display: grid; gap: 3px; margin-top: 4px; }
.bar-track {
  position: relative;
  display: block;
  height: 4px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--vp-c-text-3) 12%, transparent);
}
.bar-track i { display: block; height: 100%; border-radius: inherit; }
.reference-track i { background: #a8663f; }
:global(.dark) .reference-track i { background: #e4a273; }
.fhelium-track i { background: linear-gradient(90deg, var(--vp-c-brand-1), #7f94ff); }

@media (max-width: 700px) {
  .performance-chart { padding: 15px 10px; border-radius: 13px; }
  .chart-grid { grid-template-columns: 176px repeat(3, 214px); }
}
</style>
