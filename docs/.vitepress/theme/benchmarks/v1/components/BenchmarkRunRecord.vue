<script setup lang="ts">
import type { BenchmarkHighlight, BenchmarkV1Run } from '../data/catalog'
import {
  cpuSummary,
  displayIdentifier,
  formatDate,
  formatScalar,
  resultUrl,
  runExecutionBackendLabel,
  runExecutionBackends,
  runExecutionHardwareSummary,
  shortCommit,
} from '../data/ui'

const props = defineProps<{
  canSelect: boolean
  run: BenchmarkV1Run
  selected: boolean
}>()
const emit = defineEmits<{ select: [checked: boolean] }>()

function highlightValue(highlight: BenchmarkHighlight): string {
  const backend = typeof highlight.value === 'string'
    ? /^radix(\d+)_compact_group(\d+)_smem\d+$/u.exec(highlight.value)
    : null
  const value = backend
    ? `Radix-${backend[1]} · group ${backend[2]}`
    : typeof highlight.value === 'string'
      ? displayIdentifier(highlight.value)
      : formatScalar(highlight.value)
  return `${value}${highlight.unit ? ` ${highlight.unit}` : ''}`
}

function updateSelection(event: Event): void {
  const input = event.target
  if (input instanceof HTMLInputElement) emit('select', input.checked)
}
</script>

<template>
  <article class="run-record">
    <a class="run-record__cover" :href="resultUrl(run)"><span class="visually-hidden">Open FHElium Benchmark v1 run</span></a>

    <section class="run-record__identity">
      <div class="backend-mark" :data-backend="runExecutionBackends(run).join('-')" aria-hidden="true">
        <svg viewBox="0 0 24 24">
          <template v-if="runExecutionBackends(run).includes('cuda')">
            <path d="M3 12c2.7-4.2 6-6.2 9.8-6.2c3.1 0 5.8 1.3 8.2 3.9c-2.4 3.5-5.3 5.3-8.6 5.3c-2.2 0-4.1-.8-5.6-2.4c1.2-1.8 2.8-2.7 4.7-2.7c1.6 0 2.9.7 4 2.1" />
            <path d="M3 12c1.8 4.1 5.2 6.2 10.1 6.2c2.8 0 5.4-.7 7.9-2.2" />
          </template>
          <template v-else>
            <rect x="6" y="6" width="12" height="12" rx="2" />
            <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
            <path d="M9.5 10h5v4h-5z" />
          </template>
        </svg>
      </div>
      <div>
        <p class="run-kicker"><span>{{ runExecutionBackendLabel(run) }}</span> execution</p>
        <h2>{{ runExecutionHardwareSummary(run) }}</h2>
        <p class="host-line"><b>Device</b>{{ run.execution.device }}<template v-if="run.execution.backend === 'cuda'"><i>·</i><b>Host</b>{{ cpuSummary(run) }}</template></p>
      </div>
    </section>

    <section class="run-record__measurements" aria-label="Run highlights">
      <header><span>Selected measurements</span><small>No aggregate score</small></header>
      <div>
        <article v-for="highlight in run.highlights.slice(0, 4)" :key="highlight.id" :title="typeof highlight.value === 'string' ? highlight.value : highlight.label">
          <span>{{ highlight.label }}</span>
          <strong>{{ highlightValue(highlight) }}</strong>
        </article>
      </div>
    </section>

    <footer class="run-record__footer">
      <div class="run-provenance">
        <span>v1</span>
        <span>FHElium {{ run.fhelium.version }}</span>
        <code>{{ shortCommit(run.fhelium.commit) }}</code>
        <em v-if="run.fhelium.dirty">local changes</em>
        <time :datetime="run.recorded_at">{{ formatDate(run.recorded_at) }}</time>
      </div>
      <div class="run-actions">
        <label :class="{ selected }"><input type="checkbox" :checked="selected" :disabled="!canSelect" :aria-label="selected ? 'Remove Benchmark v1 run from comparison' : 'Select Benchmark v1 run for comparison'" @change="updateSelection"><span>{{ selected ? 'Selected' : 'Compare' }}</span></label>
        <a :href="resultUrl(run)">Open report <span aria-hidden="true">→</span></a>
      </div>
    </footer>
  </article>
</template>

<style scoped>
.run-record { position: relative; display: grid; min-width: 0; grid-template-columns: minmax(320px, .9fr) minmax(470px, 1.35fr); overflow: hidden; border: 1px solid color-mix(in srgb, var(--vp-c-divider) 78%, transparent); border-radius: 11px; background: var(--vp-c-bg); box-shadow: 0 1px 2px color-mix(in srgb, var(--vp-c-text-1) 4%, transparent), 0 12px 34px color-mix(in srgb, var(--vp-c-text-1) 7%, transparent); transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease; }
.run-record:hover { border-color: color-mix(in srgb, var(--vp-c-brand-1) 38%, var(--vp-c-divider)); box-shadow: 0 2px 5px color-mix(in srgb, var(--vp-c-text-1) 5%, transparent), 0 17px 42px color-mix(in srgb, var(--vp-c-text-1) 10%, transparent); transform: translateY(-2px); }
.run-record__cover { position: absolute; inset: 0; z-index: 1; }.run-record__cover:focus-visible { outline: 3px solid var(--vp-c-brand-1); outline-offset: -3px; }
.run-record__identity { display: grid; min-width: 0; grid-template-columns: 46px minmax(0, 1fr); gap: 16px; align-items: start; padding: 25px 26px 23px; border-right: 1px solid color-mix(in srgb, var(--vp-c-divider) 66%, transparent); background: linear-gradient(135deg, color-mix(in srgb, var(--vp-c-brand-soft) 22%, var(--vp-c-bg)), var(--vp-c-bg) 70%); }
.backend-mark { display: grid; width: 46px; height: 46px; border: 1px solid color-mix(in srgb, var(--vp-c-brand-1) 30%, var(--vp-c-divider)); border-radius: 10px; color: var(--vp-c-brand-1); background: color-mix(in srgb, var(--vp-c-brand-soft) 48%, var(--vp-c-bg)); place-items: center; }.backend-mark svg { width: 26px; height: 26px; fill: none; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.55; }
.run-kicker { margin: 1px 0 0; color: var(--vp-c-text-2); font-size: 10px; font-weight: 650; letter-spacing: .035em; text-transform: uppercase; }.run-kicker span { color: var(--vp-c-brand-1); }
.run-record h2 { max-width: 31ch; margin: 8px 0 0; border: 0; font-size: clamp(17px, 1.45vw, 21px); line-height: 1.25; letter-spacing: -.015em; }
.host-line { display: flex; gap: 7px; align-items: baseline; margin: 11px 0 0; color: var(--vp-c-text-2); font-size: 11px; }.host-line b { color: var(--vp-c-text-3); font-size: 9px; letter-spacing: .045em; text-transform: uppercase; }.host-line i { color: var(--vp-c-text-3); font-style: normal; }
.run-record__measurements { min-width: 0; padding: 20px 23px 19px; }.run-record__measurements > header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 12px; }.run-record__measurements > header span { color: var(--vp-c-text-2); font-size: 10px; font-weight: 700; letter-spacing: .035em; text-transform: uppercase; }.run-record__measurements > header small { color: var(--vp-c-text-3); font-size: 9px; }
.run-record__measurements > div { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }.run-record__measurements article { min-width: 0; min-height: 61px; padding: 9px 13px; border-left: 1px solid color-mix(in srgb, var(--vp-c-divider) 62%, transparent); }.run-record__measurements article:first-child { padding-left: 0; border-left: 0; }.run-record__measurements article > span, .run-record__measurements strong { display: block; overflow: hidden; text-overflow: ellipsis; }.run-record__measurements article > span { color: var(--vp-c-text-2); font-size: 9px; line-height: 1.35; white-space: nowrap; }.run-record__measurements strong { margin-top: 9px; overflow-wrap: anywhere; font-family: var(--vp-font-family-mono); font-size: 12px; line-height: 1.25; }
.run-record__footer { position: relative; z-index: 2; display: flex; grid-column: 1 / -1; gap: 18px; align-items: center; justify-content: space-between; min-height: 48px; padding: 8px 13px 8px 25px; border-top: 1px solid color-mix(in srgb, var(--vp-c-divider) 70%, transparent); background: color-mix(in srgb, var(--vp-c-bg-soft) 52%, var(--vp-c-bg)); pointer-events: none; }
.run-provenance { display: flex; gap: 5px 15px; flex-wrap: wrap; color: var(--vp-c-text-2); font-size: 9px; }.run-provenance code { color: inherit; font-size: inherit; }.run-provenance em { color: var(--vp-c-warning-1); font-style: normal; }
.run-actions { display: flex; flex: 0 0 auto; gap: 7px; align-items: center; pointer-events: auto; }.run-actions label, .run-actions a { display: inline-flex; height: 31px; align-items: center; padding: 0 10px; border: 1px solid var(--vp-c-divider); border-radius: 5px; color: var(--vp-c-text-2); background: var(--vp-c-bg); font-size: 10px; font-weight: 700; text-decoration: none; cursor: pointer; }.run-actions label.selected { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }.run-actions label:has(input:focus-visible), .run-actions a:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }.run-actions input { position: absolute; width: 1px; height: 1px; opacity: 0; }.run-actions a { gap: 7px; border-color: var(--vp-c-brand-1); color: white; background: var(--vp-c-brand-3); }
.visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
@media (max-width: 980px) { .run-record { grid-template-columns: minmax(0, 1fr); }.run-record__identity { border-right: 0; border-bottom: 1px solid color-mix(in srgb, var(--vp-c-divider) 66%, transparent); }.run-record__measurements > div { grid-template-columns: repeat(2, minmax(0, 1fr)); }.run-record__measurements article:nth-child(3) { padding-left: 0; border-left: 0; }.run-record__measurements article:nth-child(n + 3) { margin-top: 12px; padding-top: 12px; border-top: 1px solid color-mix(in srgb, var(--vp-c-divider) 62%, transparent); } }
@media (max-width: 600px) { .run-record__identity { grid-template-columns: 38px minmax(0, 1fr); padding: 20px 18px; }.backend-mark { width: 38px; height: 38px; }.backend-mark svg { width: 22px; height: 22px; }.run-record__measurements { padding-inline: 18px; }.run-record__footer { align-items: stretch; flex-direction: column; padding: 12px 18px; }.run-actions { justify-content: flex-end; } }
@media (forced-colors: active) { .run-record { border: 1px solid CanvasText; } }
</style>
