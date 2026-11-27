<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter, withBase } from 'vitepress'
import NavArrow from './NavArrow.vue'
import { useAutoAdvance } from '../composables/useAutoAdvance'

type ControlLevel = 'eager' | 'compile' | 'deploy' | 'serve' | 'distributed'

interface LevelEntry {
  id: ControlLevel
  label: string
  summary: string
  phases: readonly string[]
  title: string
  detail: string
  link: string
  href: string
}

const levels: readonly LevelEntry[] = [
  {
    id: 'eager',
    label: 'Eager',
    summary: 'per-operation calls',
    phases: ['Call', 'Transition', 'Dispatch', 'Return'],
    title: 'Run operations immediately.',
    detail: 'Each Engine call updates CKKS state and dispatches its kernels.',
    link: 'Example 01',
    href: '/tutorial/basic-ckks-workflow',
  },
  {
    id: 'compile',
    label: 'Compile',
    summary: 'cached schedules',
    phases: ['Look up', 'Capture', 'Lower', 'Store', 'Run'],
    title: 'Compile once per signature.',
    detail: 'A miss builds and stores the schedule; a hit runs it directly.',
    link: 'Example 12',
    href: '/tutorial/compile-layer',
  },
  {
    id: 'deploy',
    label: 'Deploy',
    summary: 'exported artifacts',
    phases: ['Export', 'Load', 'Bind keys', 'Run'],
    title: 'Run compiled code elsewhere.',
    detail: 'Python or C++ loads the artifact, binds keys, and runs it.',
    link: 'Example 15',
    href: '/tutorial/export-and-load',
  },
  {
    id: 'serve',
    label: 'Serve',
    summary: 'request streams',
    phases: ['Queue', 'Prefetch', 'Call', 'Respond'],
    title: 'Serve every user with one model.',
    detail: 'Keys arrive with each request; the next one copies in during the call.',
    link: 'Serving guide',
    href: '/how-to/serve-compiled-requests',
  },
  {
    id: 'distributed',
    label: 'Distributed',
    summary: 'rank-local programs',
    phases: ['Launch', 'Evaluate', 'Combine', 'Continue'],
    title: 'Run one program per rank.',
    detail: 'Compiled collectives combine ciphertexts across ranks.',
    link: 'Multi-GPU guide',
    href: '/how-to/choose-multi-gpu-partition',
  },
]

const activeLevel = ref<ControlLevel>('eager')
const animationRun = ref(0)
const router = useRouter()

const { stop: stopRotation } = useAutoAdvance(
  levels.map((level) => level.id),
  activeLevel,
  4500,
)

function selectLevel(level: ControlLevel): void {
  stopRotation()
  activeLevel.value = level
  animationRun.value += 1
}

function openExample(event: MouseEvent, href: string): void {
  if (
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) return

  event.preventDefault()
  void router.go(withBase(href))
}

const activeKey = computed(() => `${activeLevel.value}-${animationRun.value}`)
const active = computed(
  () => levels.find((level) => level.id === activeLevel.value) ?? levels[0],
)
</script>

<template>
  <section class="control-explorer" aria-label="FHElium execution routes">
    <div class="level-picker" role="tablist" aria-label="FHElium execution routes">
      <button
        v-for="(level, index) in levels"
        :key="level.id"
        type="button"
        role="tab"
        :aria-selected="activeLevel === level.id"
        :class="{ 'is-active': activeLevel === level.id }"
        @click="selectLevel(level.id)"
      >
        <span>{{ String(index + 1).padStart(2, '0') }}</span>
        <strong>{{ level.label }}</strong>
        <small>{{ level.summary }}</small>
      </button>
    </div>

    <div class="graph-shell">
      <div class="graph-viewport" tabindex="0" aria-label="Selected FHElium execution route; scroll horizontally when needed">
        <svg
          v-if="activeLevel === 'eager'"
          :key="activeKey"
          class="control-graph eager-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Eager operation route from an Engine call to a result value"
        >
          <defs>
            <marker id="eager-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>
          <path class="edge eager-edge" style="--delay: 120ms" d="M 220 165 H 280" />
          <path class="edge eager-edge" style="--delay: 320ms" d="M 480 165 H 540" />
          <path class="edge eager-edge" style="--delay: 520ms" d="M 760 165 H 820" />

          <g class="node eager-node" style="--delay: 0ms" transform="translate(20 125)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="34" text-anchor="middle">Engine call</text>
            <text class="detail" x="100" y="57" text-anchor="middle">values · keys · options</text>
          </g>
          <g class="node eager-node" style="--delay: 200ms" transform="translate(280 125)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="34" text-anchor="middle">State transition</text>
            <text class="detail" x="100" y="57" text-anchor="middle">depth · scale · domain</text>
          </g>
          <g class="node backend-node" style="--delay: 400ms" transform="translate(540 125)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Implementation</text>
            <text class="detail" x="110" y="57" text-anchor="middle">native CPU · CUDA · Triton</text>
          </g>
          <g class="node native-node" style="--delay: 600ms" transform="translate(820 125)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="34" text-anchor="middle">Result value</text>
            <text class="detail" x="100" y="57" text-anchor="middle">Tensor storage + CKKS state</text>
          </g>
        </svg>

        <svg
          v-else-if="activeLevel === 'compile'"
          :key="activeKey"
          class="control-graph compile-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Compile route: a specialization cache hit runs the stored schedule; a miss captures, transforms, lowers, and stores it"
        >
          <defs>
            <marker id="compile-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
            <marker id="compile-hit-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>
          <path class="edge compile-edge" style="--delay: 120ms" d="M 170 70 H 220" />
          <path class="edge hit-edge" style="--delay: 320ms" d="M 430 70 H 810" />
          <path class="edge miss-edge" style="--delay: 520ms" d="M 290 105 V 205" />
          <path class="edge compile-edge" style="--delay: 720ms" d="M 360 240 H 385" />
          <path class="edge compile-edge" style="--delay: 920ms" d="M 525 240 H 550" />
          <path class="edge compile-edge" style="--delay: 1120ms" d="M 690 240 H 715" />
          <path class="edge compile-edge" style="--delay: 1320ms" d="M 855 240 C 900 240 920 190 920 120" />
          <path class="edge store-edge" style="--delay: 1520ms" d="M 785 205 C 760 150 480 160 410 105" />
          <text class="edge-label" x="620" y="58" text-anchor="middle">hit</text>
          <text class="edge-label" x="302" y="160">miss</text>
          <text class="edge-label" x="600" y="138" text-anchor="middle">store</text>

          <g class="node eager-node" style="--delay: 0ms" transform="translate(20 35)">
            <rect width="150" height="70" rx="8" />
            <text x="75" y="30" text-anchor="middle">Call</text>
            <text class="detail" x="75" y="51" text-anchor="middle">argument signature</text>
          </g>
          <g class="node resource-node" style="--delay: 200ms" transform="translate(220 35)">
            <rect width="210" height="70" rx="8" />
            <text x="105" y="30" text-anchor="middle">Specialization cache</text>
            <text class="detail" x="105" y="51" text-anchor="middle">resident · on-disk artifact files</text>
          </g>
          <g class="node native-node" style="--delay: 400ms" transform="translate(810 20)">
            <rect width="210" height="100" rx="8" />
            <text x="105" y="36" text-anchor="middle">libfhelium Runner</text>
            <text class="detail" x="105" y="60" text-anchor="middle">arena · CUDA Graph segments</text>
            <text class="detail" x="105" y="78" text-anchor="middle">transfer overlap</text>
          </g>
          <g class="node compile-node" style="--delay: 600ms" transform="translate(220 205)">
            <rect width="140" height="70" rx="8" />
            <text x="70" y="30" text-anchor="middle">Capture</text>
            <text class="detail" x="70" y="51" text-anchor="middle">Python → Program</text>
          </g>
          <g class="node compile-node" style="--delay: 800ms" transform="translate(385 205)">
            <rect width="140" height="70" rx="8" />
            <text x="70" y="30" text-anchor="middle">Passes</text>
            <text class="detail" x="70" y="51" text-anchor="middle">depth · scale · reuse</text>
          </g>
          <g class="node compile-node" style="--delay: 1000ms" transform="translate(550 205)">
            <rect width="140" height="70" rx="8" />
            <text x="70" y="30" text-anchor="middle">Lowering</text>
            <text class="detail" x="70" y="51" text-anchor="middle">RNS · NTT · fusion</text>
          </g>
          <g class="node executable-node" style="--delay: 1200ms" transform="translate(715 205)">
            <rect width="140" height="70" rx="8" />
            <text x="70" y="30" text-anchor="middle">Exec schedule</text>
            <text class="detail" x="70" y="51" text-anchor="middle">kernels · buffers · streams</text>
          </g>
        </svg>

        <svg
          v-else-if="activeLevel === 'deploy'"
          :key="activeKey"
          class="control-graph deploy-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Deployment route from an exported artifact to Python and C++ loaders"
        >
          <defs>
            <marker id="deploy-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>
          <path class="edge deploy-edge" style="--delay: 120ms" d="M 200 165 H 270" />
          <path class="edge deploy-edge" style="--delay: 420ms" d="M 480 165 C 520 165 510 85 560 85" />
          <path class="edge deploy-edge" style="--delay: 420ms" d="M 480 165 C 520 165 510 245 560 245" />
          <path class="edge deploy-edge" style="--delay: 820ms" d="M 760 85 C 800 85 790 165 830 165" />
          <path class="edge deploy-edge" style="--delay: 820ms" d="M 760 245 C 800 245 790 165 830 165" />
          <text class="edge-label" x="235" y="150" text-anchor="middle">fc.export</text>

          <g class="node compile-node" style="--delay: 0ms" transform="translate(20 125)">
            <rect width="180" height="80" rx="8" />
            <text x="90" y="34" text-anchor="middle">Compiled callable</text>
            <text class="detail" x="90" y="57" text-anchor="middle">signatures to serve</text>
          </g>
          <g class="node executable-node" style="--delay: 250ms" transform="translate(270 125)">
            <rect width="210" height="80" rx="8" />
            <text x="105" y="34" text-anchor="middle">Artifact file</text>
            <text class="detail" x="105" y="57" text-anchor="middle">schedules · kernels · constants</text>
          </g>
          <g class="node owner-node" style="--delay: 600ms" transform="translate(560 45)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="34" text-anchor="middle">fc.load</text>
            <text class="detail" x="100" y="57" text-anchor="middle">Python process</text>
          </g>
          <g class="node owner-node" style="--delay: 600ms" transform="translate(560 205)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="34" text-anchor="middle">fhelium::Artifact</text>
            <text class="detail" x="100" y="57" text-anchor="middle">C++ process</text>
          </g>
          <g class="node native-node" style="--delay: 1000ms" transform="translate(830 125)">
            <rect width="190" height="80" rx="8" />
            <text x="95" y="34" text-anchor="middle">Runner</text>
            <text class="detail" x="95" y="57" text-anchor="middle">keys supplied at load</text>
          </g>
        </svg>

        <svg
          v-else-if="activeLevel === 'serve'"
          :key="activeKey"
          class="control-graph serve-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Serving route: queued requests with their keys pass through DoubleBuffer to one compiled model while the next request is copied"
        >
          <defs>
            <marker id="serve-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>
          <path class="edge serve-edge" style="--delay: 120ms" d="M 200 165 H 260" />
          <path class="edge serve-edge dashed-edge" style="--delay: 320ms" d="M 110 205 C 110 270 360 270 360 205" />
          <path class="edge serve-edge" style="--delay: 520ms" d="M 460 165 H 520" />
          <path class="edge serve-edge" style="--delay: 720ms" d="M 740 165 H 800" />
          <text class="edge-label" x="235" y="285" text-anchor="middle">next request</text>

          <circle class="flow-pulse" r="5">
            <animate attributeName="opacity" begin="0.9s" dur="0.15s" from="0" to="0.9" fill="freeze" />
            <animateMotion begin="0.9s" dur="2.4s" path="M 110 205 C 110 270 360 270 360 205" repeatCount="indefinite" />
          </circle>

          <g class="node input-node" style="--delay: 0ms" transform="translate(20 125)">
            <rect width="180" height="80" rx="8" />
            <text x="90" y="34" text-anchor="middle">Request queue</text>
            <text class="detail" x="90" y="57" text-anchor="middle">ciphertext · user keys</text>
          </g>
          <g class="node buffer-node" style="--delay: 200ms" transform="translate(260 125)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="34" text-anchor="middle">DoubleBuffer</text>
            <text class="detail" x="100" y="57" text-anchor="middle">copies on a transfer stream</text>
          </g>
          <g class="node compile-node" style="--delay: 450ms" transform="translate(520 125)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Compiled model</text>
            <text class="detail" x="110" y="57" text-anchor="middle">keys as call arguments</text>
          </g>
          <g class="node native-node" style="--delay: 700ms" transform="translate(800 125)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Response</text>
            <text class="detail" x="110" y="57" text-anchor="middle">consumed before the next call</text>
          </g>
        </svg>

        <svg
          v-else
          :key="activeKey"
          class="control-graph distributed-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Rank-local programs joined by a compiled collective"
        >
          <defs>
            <marker id="distributed-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>
          <path class="edge distributed-edge" style="--delay: 220ms" d="M 240 85 C 330 85 330 150 410 150" />
          <path class="edge distributed-edge" style="--delay: 220ms" d="M 240 245 C 330 245 330 180 410 180" />
          <path class="edge distributed-edge" style="--delay: 620ms" d="M 630 150 C 710 150 710 85 800 85" />
          <path class="edge distributed-edge" style="--delay: 620ms" d="M 630 180 C 710 180 710 245 800 245" />

          <g class="node owner-node" style="--delay: 0ms" transform="translate(20 45)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Rank 0 program</text>
            <text class="detail" x="110" y="57" text-anchor="middle">local values on cuda:0</text>
          </g>
          <g class="node owner-node" style="--delay: 0ms" transform="translate(20 205)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Rank 1 program</text>
            <text class="detail" x="110" y="57" text-anchor="middle">local values on cuda:1</text>
          </g>
          <g class="node backend-node" style="--delay: 400ms" transform="translate(410 120)">
            <rect width="220" height="90" rx="8" />
            <text x="110" y="36" text-anchor="middle">Compiled collective</text>
            <text class="detail" x="110" y="58" text-anchor="middle">ciphertext all-reduce</text>
            <text class="detail" x="110" y="75" text-anchor="middle">in each rank's exec schedule</text>
          </g>
          <g class="node native-node" style="--delay: 800ms" transform="translate(800 45)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Rank 0 result</text>
            <text class="detail" x="110" y="57" text-anchor="middle">combined ciphertext</text>
          </g>
          <g class="node native-node" style="--delay: 800ms" transform="translate(800 205)">
            <rect width="220" height="80" rx="8" />
            <text x="110" y="34" text-anchor="middle">Rank 1 result</text>
            <text class="detail" x="110" y="57" text-anchor="middle">combined ciphertext</text>
          </g>
        </svg>
      </div>

      <div :key="`phases-${activeKey}`" class="route-phases" aria-hidden="true">
        <template v-for="(phase, index) in active.phases" :key="phase">
          <i v-if="index > 0" />
          <span :style="{ '--delay': `${index * 200}ms` }">{{ phase }}</span>
        </template>
      </div>
    </div>

    <div class="graph-caption" role="status" aria-live="polite">
      <div class="graph-caption-copy">
        <strong>{{ active.title }}</strong>
        <span>{{ active.detail }}</span>
      </div>
      <a :href="withBase(active.href)" @click="openExample($event, active.href)">
        {{ active.link }}
        <NavArrow />
      </a>
    </div>
  </section>
</template>

<style scoped>
.control-explorer {
  margin: 28px 0 52px;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--fhe-c-border) 72%, var(--fhe-c-divider));
  border-radius: 12px;
  background: var(--fhe-c-surface);
  box-shadow: 0 2px 4px color-mix(in srgb, var(--fhe-c-text-1) 4%, transparent), 0 22px 54px color-mix(in srgb, var(--fhe-c-text-1) 9%, transparent);
}

.level-picker {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 7px;
  padding: 8px;
  border-bottom: 1px solid var(--fhe-c-divider);
  background: var(--fhe-c-canvas-alt);
}

.level-picker button {
  position: relative;
  display: grid;
  min-height: 70px;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 3px 12px;
  align-content: center;
  padding: 11px 14px;
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--fhe-c-text-2);
  background: transparent;
  cursor: pointer;
  font-family: var(--vp-font-family-base);
  text-align: left;
}

.level-picker button::after {
  position: absolute;
  right: 13px;
  bottom: 5px;
  left: 13px;
  height: 3px;
  border-radius: 999px;
  background: transparent;
  content: "";
}

.level-picker button:is(:hover, :focus-visible) {
  color: var(--fhe-c-text-1);
  border-color: color-mix(in srgb, var(--fhe-c-brand) 25%, var(--fhe-c-divider));
  background: color-mix(in srgb, var(--fhe-c-brand) 5%, var(--fhe-c-surface));
}

.level-picker button.is-active {
  color: var(--fhe-c-text-1);
  border-color: color-mix(in srgb, var(--fhe-c-brand) 30%, var(--fhe-c-divider));
  background: var(--fhe-c-surface);
  box-shadow: 0 1px 2px color-mix(in srgb, var(--fhe-c-text-1) 5%, transparent), 0 8px 18px color-mix(in srgb, var(--fhe-c-text-1) 7%, transparent);
}

.level-picker button.is-active::after {
  background: linear-gradient(90deg, var(--fhe-c-brand), var(--fhe-c-helium));
}

.level-picker button > span {
  grid-row: 1 / span 2;
  align-self: center;
  color: var(--fhe-c-brand);
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  font-weight: 800;
}

.level-picker strong {
  align-self: end;
  font-size: 17px;
  line-height: 1.2;
}

.level-picker small {
  align-self: start;
  overflow: hidden;
  color: var(--fhe-c-text-3);
  font-size: 12px;
  line-height: 1.3;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.graph-shell {
  background: var(--fhe-c-surface);
}

.graph-viewport {
  width: 100%;
  height: 340px;
  overflow-x: auto;
  background:
    radial-gradient(circle at 52% 48%, color-mix(in srgb, var(--fhe-c-brand) 6%, transparent), transparent 46%),
    radial-gradient(circle, color-mix(in srgb, var(--fhe-c-text-3) 24%, transparent) 1px, transparent 1.35px),
    var(--fhe-c-surface);
  background-size: auto, 24px 24px, auto;
}

.control-graph {
  display: block;
  width: 100%;
  min-width: 820px;
  height: 100%;
}

.edge {
  fill: none;
  opacity: 0;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
  animation: reveal-edge 320ms ease-out forwards;
  animation-delay: var(--delay);
}

.eager-edge {
  stroke: var(--fhe-c-brand);
  marker-end: url(#eager-control-arrow);
}

.compile-edge {
  stroke: var(--fhe-c-helium);
  marker-end: url(#compile-control-arrow);
}

.hit-edge {
  stroke: var(--fhe-c-brand);
  stroke-width: 2.6;
  marker-end: url(#compile-hit-arrow);
}

.miss-edge {
  stroke: var(--fhe-c-helium);
  stroke-dasharray: 7 5;
  marker-end: url(#compile-control-arrow);
}

.store-edge {
  stroke: var(--fhe-c-helium);
  stroke-dasharray: 2 5;
  marker-end: url(#compile-control-arrow);
}

.deploy-edge {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 50%, var(--fhe-c-helium));
  marker-end: url(#deploy-control-arrow);
}

.serve-edge {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 68%, var(--fhe-c-helium));
  marker-end: url(#serve-control-arrow);
}

.dashed-edge {
  stroke-dasharray: 5 5;
}

.distributed-edge {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 35%, var(--fhe-c-helium));
  marker-end: url(#distributed-control-arrow);
}

.control-graph marker path {
  fill: none;
  stroke-width: 2.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.eager-graph marker path {
  stroke: var(--fhe-c-brand);
}

.compile-graph marker path {
  stroke: var(--fhe-c-helium);
}

.compile-graph #compile-hit-arrow path {
  stroke: var(--fhe-c-brand);
}

.deploy-graph marker path {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 50%, var(--fhe-c-helium));
}

.serve-graph marker path {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 68%, var(--fhe-c-helium));
}

.distributed-graph marker path {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 35%, var(--fhe-c-helium));
}

.node {
  opacity: 0;
  animation: enter-node 420ms ease-out forwards;
  animation-delay: var(--delay);
}

.node rect {
  rx: 10px;
  fill: color-mix(in srgb, var(--fhe-c-surface) 94%, var(--fhe-c-surface-tint));
  stroke: var(--fhe-c-border);
  stroke-width: 1.6;
  vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 6px 7px color-mix(in srgb, var(--fhe-c-text-1) 11%, transparent));
}

.eager-node rect,
.owner-node rect {
  fill: color-mix(in srgb, var(--fhe-c-brand) 5%, var(--fhe-c-surface));
  stroke: var(--fhe-c-brand);
}

.compile-node rect,
.executable-node rect {
  fill: color-mix(in srgb, var(--fhe-c-helium) 5%, var(--fhe-c-surface));
  stroke: var(--fhe-c-helium);
}

.backend-node rect {
  fill: color-mix(in srgb, var(--fhe-c-brand) 4%, var(--fhe-c-surface));
  stroke: color-mix(in srgb, var(--fhe-c-brand) 58%, var(--fhe-c-helium));
}

.resource-node rect {
  fill: var(--fhe-c-surface-tint);
  stroke: var(--fhe-c-border);
}

.native-node rect {
  fill: color-mix(in srgb, var(--fhe-c-helium) 7%, var(--fhe-c-surface));
  stroke: var(--fhe-c-helium);
}

.buffer-node rect {
  fill: color-mix(in srgb, var(--fhe-c-brand) 4%, var(--fhe-c-surface));
  stroke: color-mix(in srgb, var(--fhe-c-brand) 58%, var(--fhe-c-helium));
}

.input-node rect {
  fill: color-mix(in srgb, var(--fhe-c-brand) 5%, var(--fhe-c-surface));
  stroke: var(--fhe-c-brand);
}

.node text {
  fill: var(--fhe-c-text-1);
  font-family: var(--vp-font-family-base);
  font-size: 14px;
  font-weight: 760;
}

.node text.detail {
  fill: var(--fhe-c-text-2);
  font-size: 11px;
  font-weight: 560;
}

.flow-pulse {
  fill: var(--fhe-c-brand);
  opacity: 0;
  filter: drop-shadow(0 0 4px color-mix(in srgb, var(--fhe-c-brand) 55%, transparent));
}

.flow-pulse.is-two {
  fill: color-mix(in srgb, var(--fhe-c-brand) 55%, var(--fhe-c-helium));
}

.flow-pulse.is-three {
  fill: var(--fhe-c-helium);
}

.edge-label {
  fill: var(--fhe-c-text-3);
  font-family: var(--vp-font-family-base);
  font-size: 12px;
  font-weight: 700;
  opacity: 1;
}

.route-phases {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 48px;
  padding: 0 20px;
  border-top: 1px solid var(--fhe-c-divider);
  background: color-mix(in srgb, var(--fhe-c-canvas-alt) 52%, transparent);
}

.route-phases span {
  opacity: 0;
  color: var(--fhe-c-text-2);
  font-size: 12px;
  font-weight: 750;
  animation: phase-in 300ms ease-out forwards;
  animation-delay: var(--delay);
}

.route-phases i {
  height: 2px;
  flex: 1;
  background: linear-gradient(90deg, var(--fhe-c-brand), var(--fhe-c-helium));
}

.graph-caption {
  display: flex;
  gap: 16px;
  align-items: center;
  min-height: 64px;
  padding: 14px 22px;
  border-top: 1px solid var(--fhe-c-divider);
  background: color-mix(in srgb, var(--fhe-c-canvas-alt) 50%, transparent);
}

.graph-caption-copy {
  min-width: 0;
  flex: 1;
}

.graph-caption-copy strong {
  margin-right: 8px;
  color: var(--fhe-c-text-1);
  font-size: 17px;
  line-height: 1.35;
}

.graph-caption-copy span {
  color: var(--fhe-c-text-2);
  font-size: 14px;
  line-height: 1.55;
}

.graph-caption > a {
  display: inline-flex;
  height: 34px;
  flex: 0 0 auto;
  gap: 7px;
  align-items: center;
  padding: 0 12px;
  border: 1px solid color-mix(in srgb, var(--fhe-c-brand) 34%, var(--fhe-c-border));
  border-radius: var(--fhe-radius-control);
  color: var(--fhe-c-brand);
  background: var(--fhe-c-surface);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.01em;
  text-decoration: none;
  transition: border-color 140ms ease, background-color 140ms ease, transform 140ms ease;
}

.graph-caption > a:hover {
  border-color: var(--fhe-c-brand);
  color: var(--fhe-c-brand-hover);
  background: var(--fhe-c-surface-tint);
  transform: translateY(-1px);
}

.graph-caption > a:focus-visible {
  outline: 2px solid var(--fhe-c-focus);
  outline-offset: 2px;
}

@keyframes reveal-edge {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes enter-node {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes phase-in {
  from { opacity: 0; transform: translateX(-5px); }
  to { opacity: 1; transform: translateX(0); }
}

@media (max-width: 760px) {
  .level-picker {
    display: flex;
    overflow-x: auto;
  }

  .level-picker button {
    flex: 0 0 210px;
  }

  .graph-caption {
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .graph-caption-copy {
    flex-basis: 100%;
  }

}

@media (prefers-reduced-motion: reduce) {
  .edge,
  .node,
  .route-phases span {
    animation-delay: 0ms;
    animation-duration: 1ms;
  }

  .flow-pulse {
    display: none;
  }
}

@media (forced-colors: active) {
  .control-explorer,
  .level-picker button,
  .node rect {
    border-color: CanvasText;
    background: Canvas;
    stroke: CanvasText;
  }
}
</style>
