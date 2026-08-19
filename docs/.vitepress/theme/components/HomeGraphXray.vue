<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { highlightPython } from './pythonHighlight'

type StageId = 'capture' | 'transform' | 'execute'
type NodeTone = 'input' | 'semantic' | 'mechanic' | 'output'

interface GraphNode {
  id: string
  x: number
  y: number
  width: number
  label: string
  detail: string
  tone: NodeTone
}

interface GraphEdge {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
}

interface CodeLensLine {
  html: string
  focus: boolean
}

interface GraphStage {
  id: StageId
  label: string
  eyebrow: string
  description: string
  path: readonly string[]
  nodes: readonly GraphNode[]
  edges: readonly GraphEdge[]
  facts: readonly { value: string, label: string }[]
  diagnostic: string
  code: readonly CodeLensLine[]
}

function codeLensLines(
  contextBefore: string,
  focus: string,
  contextAfter: string,
): readonly CodeLensLine[] {
  return [
    ...contextBefore.split('\n').map((line) => ({
      html: highlightPython(line),
      focus: false,
    })),
    ...focus.split('\n').map((line) => ({
      html: highlightPython(line),
      focus: true,
    })),
    ...contextAfter.split('\n').map((line) => ({
      html: highlightPython(line),
      focus: false,
    })),
  ]
}

const codePrelude = `import torch

import fhelium as fh
from fhelium.experimental import jit


def encrypted_quadratic(x, gain, bias, rotation):
    mixed = (x + torch.roll(x, shifts=rotation, dims=-1)) * gain
    return mixed * mixed + bias`

const captureCode = `captured = jit.trace(
    encrypted_quadratic,
    inputs={
        "x": jit.encrypted(),
        "gain": jit.message(),
        "bias": jit.message(),
        "rotation": jit.static(3),
    },
)
source = captured.program
workspace = captured.workspace`

const transformCode = `lowered = jit.default_pipeline().run(source, workspace)
program = lowered.program
workspace = lowered.workspace
for report in lowered.reports:
    print(report.name, report.stats, report.diagnostics)`

const executeCode = `key_plan = jit.analyze_evaluation_key_requirements(program)
evaluation_keys = fh.EvaluationKeySet(
    rotations=fh.RotationKeySet({
        step: engine.rotation_key(step)
        for step in key_plan.rotation_steps
    }),
    relinearization=(
        engine.relinearization_key
        if key_plan.requires_relinearization else None
    ),
)
workspace.update({
    "engine": engine,
    "evaluation_keys": evaluation_keys,
})
ready = program.readiness(workspace)
if not ready.runnable:
    raise jit.ProgramNotReadyError(ready)
result = program.run(encrypted_x, 0.625, bias, workspace=workspace)`

function codeBlocks(...blocks: readonly string[]): string {
  return blocks.join('\n\n')
}

const stages: readonly GraphStage[] = [
  {
    id: 'capture',
    label: 'Create',
    eyebrow: 'One source-independent Program',
    description: 'PyTorch tracing emits one structurally valid mixed-dialect xDSL Program and retains frontend evidence and graph-external materials beside it.',
    path: ['jit.trace', 'CaptureResult', 'Program + Workspace'],
    nodes: [
      { id: 'ci', x: 70, y: 80, width: 90, label: 'typed inputs', detail: '4 roles', tone: 'input' },
      { id: 'ct', x: 230, y: 70, width: 110, label: 'torch.call', detail: 'preserved call', tone: 'semantic' },
      { id: 'cs', x: 230, y: 190, width: 125, label: 'semantic ops', detail: 'encrypted path', tone: 'semantic' },
      { id: 'cp', x: 430, y: 130, width: 125, label: 'xDSL Program', detail: 'mixed dialects', tone: 'output' },
      { id: 'cw', x: 605, y: 130, width: 110, label: 'Workspace', detail: 'live state', tone: 'output' },
    ],
    edges: [
      { id: 'ce1', x1: 115, y1: 80, x2: 175, y2: 70 },
      { id: 'ce2', x1: 115, y1: 80, x2: 168, y2: 190 },
      { id: 'ce3', x1: 285, y1: 70, x2: 367, y2: 118 },
      { id: 'ce4', x1: 293, y1: 190, x2: 367, y2: 142 },
      { id: 'ce5', x1: 493, y1: 130, x2: 550, y2: 130 },
    ],
    facts: [
      { value: 'one class', label: 'capture, text, and xDSL construction converge' },
      { value: 'open IR', label: 'FHElium, Torch, and extension dialects coexist' },
      { value: 'external', label: 'live materials remain in the Workspace' },
    ],
    diagnostic: 'Structural verification establishes an interchange Program. Runtime readiness remains an independent decision.',
    code: codeLensLines(
      codePrelude,
      captureCode,
      codeBlocks(transformCode, executeCode),
    ),
  },
  {
    id: 'transform',
    label: 'Transform',
    eyebrow: 'Selected local pass policy',
    description: 'An ordered pipeline transforms one clone, preserves unknown operations, shares the retained Workspace, and reports local activity.',
    path: ['Program.clone', 'default_pipeline', 'PipelineResult'],
    nodes: [
      { id: 'ti', x: 60, y: 130, width: 100, label: 'Program', detail: 'source clone', tone: 'input' },
      { id: 'ts', x: 205, y: 72, width: 115, label: 'semantic', detail: 'role classify', tone: 'semantic' },
      { id: 'tp', x: 350, y: 72, width: 115, label: 'prepare + NTT', detail: 'insert mechanics', tone: 'mechanic' },
      { id: 'tc', x: 350, y: 188, width: 115, label: 'lowered CKKS', detail: 'lower locally', tone: 'semantic' },
      { id: 'tr', x: 500, y: 130, width: 125, label: 'relin + rescale', detail: 'clear obligations', tone: 'mechanic' },
      { id: 'to', x: 640, y: 130, width: 75, label: 'reports', detail: 'evidence', tone: 'output' },
    ],
    edges: [
      { id: 'te1', x1: 110, y1: 130, x2: 148, y2: 84 },
      { id: 'te2', x1: 263, y1: 72, x2: 292, y2: 72 },
      { id: 'te3', x1: 408, y1: 72, x2: 438, y2: 118 },
      { id: 'te4', x1: 110, y1: 130, x2: 292, y2: 188 },
      { id: 'te5', x1: 408, y1: 188, x2: 438, y2: 142 },
      { id: 'te6', x1: 563, y1: 130, x2: 603, y2: 130 },
    ],
    facts: [
      { value: 'ordered', label: 'pass names and composition stay inspectable' },
      { value: 'legal no-op', label: 'unmatched local patterns remain valid' },
      { value: 'retained', label: 'every pass receives the same Workspace' },
    ],
    diagnostic: 'Pipeline completion records transformations; the independent readiness check determines executability.',
    code: codeLensLines(
      codeBlocks(codePrelude, captureCode),
      transformCode,
      executeCode,
    ),
  },
  {
    id: 'execute',
    label: 'Execute',
    eyebrow: 'Independent readiness check',
    description: 'The selected entry, transformed operations, and bound Workspace capabilities must satisfy the exact interpreter schemas before execution.',
    path: ['requirements', 'Workspace bindings', 'readiness', 'Program.run'],
    nodes: [
      { id: 'ei', x: 65, y: 85, width: 100, label: 'Program', detail: 'selected entry', tone: 'input' },
      { id: 'ew', x: 65, y: 188, width: 110, label: 'Workspace', detail: 'engine + keys', tone: 'input' },
      { id: 'er', x: 260, y: 136, width: 125, label: 'readiness', detail: 'pure check', tone: 'mechanic' },
      { id: 'eh', x: 430, y: 136, width: 125, label: 'exact handlers', detail: 'trusted extension', tone: 'mechanic' },
      { id: 'ex', x: 600, y: 136, width: 105, label: 'Program.run', detail: 'run request', tone: 'output' },
    ],
    edges: [
      { id: 'ee1', x1: 115, y1: 85, x2: 198, y2: 124 },
      { id: 'ee2', x1: 120, y1: 188, x2: 198, y2: 148 },
      { id: 'ee3', x1: 323, y1: 136, x2: 367, y2: 136 },
      { id: 'ee4', x1: 493, y1: 136, x2: 548, y2: 136 },
    ],
    facts: [
      { value: 'pure', label: 'readiness invokes no resolver or operation' },
      { value: 'exact', label: 'versions, schemas, keys, and references checked' },
      { value: 'trusted', label: 'extension handlers require trusted bindings' },
    ],
    diagnostic: 'ProgramNotReadyError preserves the complete readiness report when a Program.run(...) call is blocked.',
    code: codeLensLines(
      codeBlocks(codePrelude, captureCode, transformCode),
      executeCode,
      '',
    ),
  },
]

const activeStageId = ref<StageId>('execute')
const activeStage = computed(
  () => stages.find((stage) => stage.id === activeStageId.value) ?? stages[0],
)
const lensPre = ref<HTMLPreElement | null>(null)

async function centerCodeLens(): Promise<void> {
  await nextTick()
  const pre = lensPre.value
  if (pre === null) return
  const focusLines = Array.from(
    pre.querySelectorAll<HTMLElement>('.lens-line.is-focus'),
  )
  const first = focusLines[0]
  const last = focusLines.at(-1)
  if (first === undefined || last === undefined) return
  const focusTop = first.offsetTop
  const focusBottom = last.offsetTop + last.offsetHeight
  pre.scrollTop = Math.max(
    0,
    (focusTop + focusBottom - pre.clientHeight) / 2,
  )
  pre.scrollLeft = 0
}

watch(activeStageId, centerCodeLens)
onMounted(centerCodeLens)
</script>

<template>
  <section class="jit-xray" aria-label="Inspectable JIT Program pipeline">
    <section class="graph-workbench" aria-label="Encrypted evaluator graph">
      <div class="stage-picker" aria-label="Evaluator code and graph stages">
        <button
          v-for="stage in stages"
          :key="stage.id"
          type="button"
          :class="{ 'is-active': activeStageId === stage.id }"
          :aria-pressed="activeStageId === stage.id"
          @click="activeStageId = stage.id"
        >
          <span>{{ stage.eyebrow }}</span>
          <strong>{{ stage.label }}</strong>
        </button>
      </div>

      <div class="stage-inspection">
        <section class="code-lens" :aria-label="`${activeStage.label} Code Lens`">
          <div class="lens-window">
            <pre
              ref="lensPre"
              tabindex="0"
              :aria-label="`Complete evaluator flow focused on the ${activeStage.label} stage`"
            ><code class="home-python-code"><span
              v-for="(line, index) in activeStage.code"
              :key="index"
              class="lens-line"
              :class="line.focus ? 'is-focus' : 'is-context'"
            ><span v-html="line.html" /></span></code></pre>
          </div>
        </section>

      </div>

      <div class="stage-route" tabindex="0" :aria-label="`${activeStage.label} graph route`">
        <template v-for="(step, index) in activeStage.path" :key="step">
          <code>{{ step }}</code>
          <i v-if="index < activeStage.path.length - 1" aria-hidden="true">→</i>
        </template>
      </div>

      <div class="graph-body">
        <div class="graph-viewport" tabindex="0" :aria-label="`${activeStage.label} graph visualization; scroll horizontally when needed`">
          <svg viewBox="0 0 680 260" role="img" :aria-label="`${activeStage.label}: ${activeStage.description}`">
            <defs>
              <marker id="jit-xray-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M 0 0 L 8 4 L 0 8 z" />
              </marker>
            </defs>
            <g class="edges">
              <line
                v-for="edge in activeStage.edges"
                :key="edge.id"
                :x1="edge.x1"
                :y1="edge.y1"
                :x2="edge.x2"
                :y2="edge.y2"
                marker-end="url(#jit-xray-arrow)"
              />
            </g>
            <g
              v-for="node in activeStage.nodes"
              :key="node.id"
              class="node"
              :class="`is-${node.tone}`"
              :transform="`translate(${node.x - node.width / 2} ${node.y - 25})`"
            >
              <rect :width="node.width" height="50" rx="5" />
              <text :x="node.width / 2" y="21" text-anchor="middle">{{ node.label }}</text>
              <text class="detail" :x="node.width / 2" y="37" text-anchor="middle">{{ node.detail }}</text>
            </g>
          </svg>
        </div>

        <aside class="facts" aria-label="Selected graph stage facts">
          <div v-for="fact in activeStage.facts" :key="fact.label">
            <strong>{{ fact.value }}</strong>
            <span>{{ fact.label }}</span>
          </div>
          <p>{{ activeStage.diagnostic }}</p>
        </aside>
      </div>
    </section>
  </section>
</template>

<style scoped>
.jit-xray {
  --xray-amber: #d99018;
  --code-bg: #f5f7fc;
  --code-ink: #172033;
  --code-muted: #5e687c;
  --code-line: #cbd3e3;
  --code-blue: #3159d5;
  margin: 28px 0 36px;
}

:global(.dark .jit-xray) {
  --code-bg: #11151f;
  --code-ink: #e8ebf5;
  --code-muted: #9fa9be;
  --code-line: rgba(151, 169, 220, 0.24);
  --code-blue: #8da4ff;
}

.graph-workbench {
  min-width: 0;
  overflow: hidden;
  border-radius: 7px;
}

.stage-picker {
  display: grid;
}

.stage-picker button {
  appearance: none;
  border: 0;
  color: inherit;
  background: transparent;
  cursor: pointer;
  font: inherit;
}

.code-lens pre:focus-visible,
.stage-route:focus-visible,
.graph-viewport:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: -3px;
}

.graph-workbench {
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
}

.stage-picker {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  border-bottom: 1px solid var(--vp-c-divider);
}

.stage-picker button {
  display: grid;
  gap: 3px;
  padding: 12px 14px;
  color: var(--vp-c-text-2);
  text-align: left;
}

.stage-picker button + button {
  border-left: 1px solid var(--vp-c-divider);
}

.stage-picker button.is-active {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.stage-picker span {
  font-family: var(--vp-font-family-mono);
  font-size: 8px;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.stage-picker strong {
  color: var(--vp-c-text-1);
  font-size: 11px;
}

.stage-picker button:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: -3px;
}

.stage-inspection {
  min-width: 0;
  border-bottom: 1px solid var(--vp-c-divider);
}

.code-lens {
  min-width: 0;
  height: 260px;
  overflow: hidden;
  color: var(--code-ink);
  background: var(--code-bg);
}

.lens-window {
  position: relative;
  height: 100%;
  overflow: hidden;
}

.lens-window::before,
.lens-window::after {
  position: absolute;
  z-index: 2;
  right: 0;
  left: 0;
  height: 28px;
  pointer-events: none;
  content: '';
}

.lens-window::before {
  top: 0;
  background: linear-gradient(var(--code-bg), transparent);
}

.lens-window::after {
  bottom: 0;
  background: linear-gradient(transparent, var(--code-bg));
}

.code-lens pre {
  position: absolute;
  inset: 0;
  margin: 0;
  overflow: auto;
  padding: 8px 0 10px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  scrollbar-width: none;
}

.code-lens pre::-webkit-scrollbar {
  display: none;
}

.code-lens pre code {
  color: var(--code-ink);
  background: transparent !important;
  font-family: var(--vp-font-family-mono);
  font-size: 10px;
  line-height: 1.25;
}

.lens-line {
  display: block;
  min-height: 1.25em;
  padding: 0 14px;
  white-space: pre;
}

.lens-line.is-focus {
  background: color-mix(in srgb, var(--code-blue) 9%, transparent);
  box-shadow: inset 2px 0 var(--code-blue);
}

.lens-line.is-context {
  color: var(--code-muted);
  opacity: 0.62;
}

.lens-line.is-context :deep(span) {
  color: inherit;
}

.stage-route {
  display: flex;
  align-items: center;
  gap: 7px;
  min-height: 49px;
  overflow-x: auto;
  padding: 10px 13px;
  border-bottom: 1px solid var(--vp-c-divider);
  white-space: nowrap;
}

.stage-route code {
  flex: 0 0 auto;
  padding: 5px 7px;
  border: 1px solid color-mix(in srgb, var(--vp-c-brand-1) 38%, var(--vp-c-divider));
  border-radius: 4px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-brand-soft);
  font-family: var(--vp-font-family-mono);
  font-size: 8.5px;
  font-weight: 700;
}

.stage-route i {
  flex: 0 0 auto;
  color: var(--vp-c-brand-1);
  font-style: normal;
}

.graph-body {
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(14rem, 0.45fr);
  min-width: 0;
}

.graph-viewport {
  min-width: 0;
  overflow-x: auto;
  border-right: 1px solid var(--vp-c-divider);
  background:
    linear-gradient(color-mix(in srgb, var(--vp-c-divider) 34%, transparent) 1px, transparent 1px),
    linear-gradient(90deg, color-mix(in srgb, var(--vp-c-divider) 34%, transparent) 1px, transparent 1px);
  background-size: 20px 20px;
}

.graph-viewport svg {
  display: block;
  width: 100%;
  min-width: 620px;
  height: auto;
}

.graph-viewport marker path { fill: var(--vp-c-brand-1); }

.edges line {
  stroke: color-mix(in srgb, var(--vp-c-brand-1) 68%, var(--vp-c-divider));
  stroke-dasharray: 5 4;
  stroke-width: 1.4;
  animation: xray-flow 2.7s linear infinite;
}

.node rect { stroke: var(--vp-c-divider); fill: var(--vp-c-bg); }
.node text {
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 10px;
  font-weight: 700;
}
.node .detail { fill: var(--vp-c-text-3); font-size: 8px; font-weight: 550; }
.node.is-semantic rect {
  stroke: var(--vp-c-brand-1);
  fill: color-mix(in srgb, var(--vp-c-brand-soft) 78%, var(--vp-c-bg));
}
.node.is-mechanic rect {
  stroke: color-mix(in srgb, var(--xray-amber) 68%, var(--vp-c-divider));
  fill: color-mix(in srgb, var(--xray-amber) 9%, var(--vp-c-bg));
}
.node.is-output rect { stroke: var(--vp-c-brand-1); stroke-width: 1.6; }

.facts {
  display: grid;
  align-content: start;
  background: var(--vp-c-bg-soft);
}

.facts > div {
  display: grid;
  gap: 2px;
  padding: 12px 13px;
  border-bottom: 1px solid var(--vp-c-divider);
}
.facts strong {
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
}
.facts span { color: var(--vp-c-text-3); font-size: 8.5px; }
.facts p {
  margin: 0;
  padding: 13px;
  color: var(--vp-c-text-2);
  font-size: 9.5px;
  line-height: 1.65;
}

@keyframes xray-flow { to { stroke-dashoffset: -18; } }

@media (max-width: 900px) {
  .graph-body {
    grid-template-columns: minmax(0, 1fr);
  }

  .graph-viewport {
    border-right: 0;
    border-bottom: 1px solid var(--vp-c-divider);
  }
}

@media (max-width: 640px) {
  .stage-route { flex-wrap: wrap; white-space: normal; }
}

@media (forced-colors: active) {
  .lens-window::before,
  .lens-window::after {
    display: none;
  }

  .lens-line.is-focus {
    border-left: 2px solid Highlight;
    color: CanvasText;
    background: Canvas;
    box-shadow: none;
  }

  .lens-line.is-context {
    color: GrayText;
    opacity: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .edges line { animation: none; }
}
</style>
