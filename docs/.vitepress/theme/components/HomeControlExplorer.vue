<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter, withBase } from 'vitepress'
import NavArrow from './NavArrow.vue'

type ControlLevel = 'stack' | 'eager' | 'compile' | 'execution'

const activeLevel = ref<ControlLevel>('stack')
const animationRun = ref(0)
const router = useRouter()

function selectLevel(level: ControlLevel): void {
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
const caption = computed(() => {
  if (activeLevel.value === 'stack') {
    return {
      title: 'Trace the shared stack.',
      detail: 'Eager and Program meet Backend resources and native CPU/CUDA dispatch.',
      example: 'Example 17',
      href: '/tutorial/compose-and-execute-compile-pipeline',
    }
  }
  if (activeLevel.value === 'compile') {
    return {
      title: 'Transform before execution.',
      detail: 'Preserve or lower CKKS operations, then link implementations and resources.',
      example: 'Example 19',
      href: '/tutorial/customize-compile-pass-and-pipeline',
    }
  }
  if (activeLevel.value === 'execution') {
    return {
      title: 'Capture once and replay.',
      detail: 'Each Input refreshes the same stable Buffer before CUDA Graph replay.',
      example: 'Example 11',
      href: '/tutorial/cuda-graph-matvec',
    }
  }
  return {
    title: 'Control each operation.',
    detail: 'Engine applies its CKKS transition and dispatches through Backend immediately.',
    example: 'Example 01',
    href: '/tutorial/basic-ckks-workflow',
  }
})
</script>

<template>
  <section class="control-explorer" aria-label="Choose a FHElium level of control">
    <div class="level-picker" role="tablist" aria-label="FHElium control levels">
      <button
        type="button"
        role="tab"
        :aria-selected="activeLevel === 'stack'"
        :class="{ 'is-active': activeLevel === 'stack' }"
        @click="selectLevel('stack')"
      >
        <span>01</span>
        <strong>Stack</strong>
        <small>owners, Backend, resources</small>
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeLevel === 'eager'"
        :class="{ 'is-active': activeLevel === 'eager' }"
        @click="selectLevel('eager')"
      >
        <span>02</span>
        <strong>Eager</strong>
        <small>operation-level control</small>
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeLevel === 'compile'"
        :class="{ 'is-active': activeLevel === 'compile' }"
        @click="selectLevel('compile')"
      >
        <span>03</span>
        <strong>Compile</strong>
        <small>graph and scheduling control</small>
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="activeLevel === 'execution'"
        :class="{ 'is-active': activeLevel === 'execution' }"
        @click="selectLevel('execution')"
      >
        <span>04</span>
        <strong>Execution</strong>
        <small>captured repeated execution</small>
      </button>
    </div>

    <div class="graph-shell">
      <div class="graph-viewport" tabindex="0" aria-label="Selected FHElium control graph; scroll horizontally when needed">
        <svg
          v-if="activeLevel === 'eager'"
          :key="activeKey"
          class="control-graph animated-graph eager-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Eager operation execution route"
        >
          <defs>
            <marker id="eager-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>
          <path class="edge eager-edge" style="--delay: 80ms" d="M 176 165 H 220" />
          <path class="edge eager-edge" style="--delay: 260ms" d="M 376 165 H 420" />
          <path class="edge eager-edge" style="--delay: 440ms" d="M 576 165 H 620" />
          <path class="edge eager-edge" style="--delay: 620ms" d="M 776 165 H 820" />

          <g class="node eager-node" style="--delay: 0ms" transform="translate(20 125)">
            <rect width="156" height="80" rx="8" />
            <text x="78" y="34" text-anchor="middle">Engine call</text>
            <text class="detail" x="78" y="57" text-anchor="middle">typed values + key</text>
          </g>
          <g class="node eager-node" style="--delay: 180ms" transform="translate(220 125)">
            <rect width="156" height="80" rx="8" />
            <text x="78" y="34" text-anchor="middle">Value transition</text>
            <text class="detail" x="78" y="57" text-anchor="middle">level · scale · domain</text>
          </g>
          <g class="node eager-node" style="--delay: 360ms" transform="translate(420 125)">
            <rect width="156" height="80" rx="8" />
            <text x="78" y="34" text-anchor="middle">Eager dispatcher</text>
            <text class="detail" x="78" y="57" text-anchor="middle">graph-free call</text>
          </g>
          <g class="node backend-node" style="--delay: 540ms" transform="translate(620 125)">
            <rect width="156" height="80" rx="8" />
            <text x="78" y="34" text-anchor="middle">Implementation</text>
            <text class="detail" x="78" y="57" text-anchor="middle">registered operation</text>
          </g>
          <g class="node native-node" style="--delay: 720ms" transform="translate(820 125)">
            <rect width="190" height="80" rx="8" />
            <text x="95" y="32" text-anchor="middle">RNS / NTT</text>
            <text class="detail" x="95" y="56" text-anchor="middle">native CPU · CUDA</text>
          </g>
        </svg>

        <svg
          v-else-if="activeLevel === 'compile'"
          :key="activeKey"
          class="control-graph animated-graph compile-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Compile graph transforming from captured operations into a linked executable"
        >
          <defs>
            <marker id="compile-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>

          <path class="edge compile-edge" style="--delay: 100ms" d="M 150 165 H 190" />
          <path class="edge compile-edge" style="--delay: 430ms" d="M 330 165 C 370 165 368 92 410 92" />
          <path class="edge compile-edge" style="--delay: 430ms" d="M 330 165 C 370 165 368 238 410 238" />
          <path class="edge compile-edge" style="--delay: 1500ms" d="M 550 238 H 590" />
          <path class="edge compile-edge" style="--delay: 1950ms" d="M 730 238 C 750 238 748 165 770 165" />
          <path class="edge compile-edge" style="--delay: 1950ms" d="M 550 92 C 655 92 650 165 770 165" />
          <path class="edge compile-edge" style="--delay: 2450ms" d="M 900 165 H 930" />

          <text class="edge-label" x="365" y="116" text-anchor="middle">Preserve</text>
          <text class="edge-label" x="365" y="220" text-anchor="middle">Lower</text>

          <g class="node compile-node" style="--delay: 0ms" transform="translate(20 125)">
            <rect width="130" height="80" rx="8" />
            <text x="65" y="34" text-anchor="middle">Capture / import</text>
            <text class="detail" x="65" y="57" text-anchor="middle">PyTorch · text · xDSL</text>
          </g>
          <g class="node compile-node" style="--delay: 260ms" transform="translate(190 125)">
            <rect width="140" height="80" rx="8" />
            <text x="70" y="34" text-anchor="middle">Program</text>
            <text class="detail" x="70" y="57" text-anchor="middle">mixed-level IR</text>
          </g>

          <g class="node compile-node transform-node" style="--delay: 620ms" transform="translate(410 52)">
            <rect width="140" height="80" rx="8" />
            <text class="semantic-label" x="70" y="34" text-anchor="middle">torch.roll</text>
            <text class="ckks-label" x="70" y="34" text-anchor="middle">CKKS RotateOp</text>
            <text class="detail" x="70" y="57" text-anchor="middle">preserved</text>
          </g>
          <g class="node compile-node transform-node" style="--delay: 620ms" transform="translate(410 198)">
            <rect width="140" height="80" rx="8" />
            <text class="semantic-label" x="70" y="34" text-anchor="middle">torch.multiply</text>
            <text class="ckks-label" x="70" y="34" text-anchor="middle">CKKS MultiplyOp</text>
            <text class="detail" x="70" y="57" text-anchor="middle">three components</text>
          </g>

          <g class="node transition-node" style="--delay: 1450ms" transform="translate(590 198)">
            <rect width="140" height="80" rx="8" />
            <text x="70" y="31" text-anchor="middle">RelinearizeOp</text>
            <text x="70" y="52" text-anchor="middle">RescaleOp</text>
            <text class="detail" x="70" y="69" text-anchor="middle">placed policy</text>
          </g>

          <g class="node backend-node" style="--delay: 2050ms" transform="translate(770 125)">
            <rect width="130" height="80" rx="8" />
            <text x="65" y="32" text-anchor="middle">Lowered graph</text>
            <text class="detail" x="65" y="55" text-anchor="middle">RNS · NTT · key switch</text>
          </g>
          <g class="node executable-node" style="--delay: 2550ms" transform="translate(930 125)">
            <rect width="90" height="80" rx="8" />
            <text x="45" y="32" text-anchor="middle">Link</text>
            <text class="detail" x="45" y="55" text-anchor="middle">Executable</text>
          </g>
        </svg>

        <svg
          v-else-if="activeLevel === 'stack'"
          :key="activeKey"
          class="control-graph animated-graph stack-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="FHElium execution components and Backend relationships"
        >
          <defs>
            <marker id="stack-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>

          <g class="stack-route stack-eager-route">
            <path class="edge stack-edge" style="--delay: 120ms" d="M 155 82 H 205" />
            <path class="edge stack-edge" style="--delay: 520ms" d="M 375 82 C 430 82 420 165 475 165" />
            <path class="edge stack-edge" style="--delay: 1150ms" d="M 550 125 C 565 95 610 80 650 80" />
            <path class="edge stack-edge" style="--delay: 1550ms" d="M 810 80 C 855 80 855 148 900 148" />

            <g class="node owner-node" style="--delay: 0ms" transform="translate(15 47)">
              <rect width="140" height="70" rx="8" />
              <text x="70" y="29" text-anchor="middle">Eager Engine</text>
              <text class="detail" x="70" y="51" text-anchor="middle">public operations</text>
            </g>
            <g class="node dispatcher-node" style="--delay: 300ms" transform="translate(205 47)">
              <rect width="170" height="70" rx="8" />
              <text x="85" y="29" text-anchor="middle">Eager dispatcher</text>
              <text class="detail" x="85" y="51" text-anchor="middle">operation call cache</text>
            </g>
            <g class="node resource-node" style="--delay: 1050ms" transform="translate(650 45)">
              <rect width="160" height="70" rx="8" />
              <text x="80" y="28" text-anchor="middle">Runtime Resource</text>
              <text class="detail" x="80" y="50" text-anchor="middle">keys · materials · bindings</text>
            </g>
          </g>

          <g class="stack-route stack-program-route">
            <path class="edge stack-edge" style="--delay: 120ms" d="M 155 248 H 205" />
            <path class="edge stack-edge" style="--delay: 520ms" d="M 375 248 C 430 248 420 165 475 165" />
            <path class="edge stack-edge" style="--delay: 1150ms" d="M 550 205 C 565 235 610 250 650 250" />
            <path class="edge stack-edge" style="--delay: 1550ms" d="M 810 250 C 855 250 855 182 900 182" />

            <g class="node owner-node" style="--delay: 0ms" transform="translate(15 213)">
              <rect width="140" height="70" rx="8" />
              <text x="70" y="29" text-anchor="middle">Program</text>
              <text class="detail" x="70" y="51" text-anchor="middle">linked Program</text>
            </g>
            <g class="node dispatcher-node" style="--delay: 300ms" transform="translate(205 213)">
              <rect width="170" height="70" rx="8" />
              <text x="85" y="29" text-anchor="middle">Dispatch table</text>
              <text class="detail" x="85" y="51" text-anchor="middle">operation → implementation</text>
            </g>
            <g class="node resource-node" style="--delay: 1050ms" transform="translate(650 215)">
              <rect width="160" height="70" rx="8" />
              <text x="80" y="28" text-anchor="middle">Arithmetic contexts</text>
              <text class="detail" x="80" y="50" text-anchor="middle">RnsContext · NttContext</text>
            </g>
          </g>

          <g class="node backend-node" style="--delay: 700ms" transform="translate(475 125)">
            <rect width="150" height="80" rx="8" />
            <text x="75" y="32" text-anchor="middle">Backend</text>
            <text class="detail" x="75" y="55" text-anchor="middle">registry + workspace</text>
          </g>

          <g class="node native-node" style="--delay: 1500ms" transform="translate(900 125)">
            <rect width="130" height="80" rx="8" />
            <text x="65" y="31" text-anchor="middle">Tensor execution</text>
            <text class="detail" x="65" y="54" text-anchor="middle">native CPU · CUDA</text>
          </g>
        </svg>

        <svg
          v-else
          :key="activeKey"
          class="control-graph animated-graph replay-graph"
          viewBox="0 0 1040 330"
          role="img"
          aria-label="Program CUDA Graph capture and repeated buffered input execution"
        >
          <defs>
            <marker id="replay-control-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
              <path d="M 1.5 1.5 L 8.5 5 L 1.5 8.5" />
            </marker>
          </defs>

          <path class="edge replay-edge" style="--delay: 320ms" d="M 335 78 H 435" />
          <path class="edge replay-edge" style="--delay: 320ms" d="M 225 205 C 265 205 270 225 315 225" />
          <path class="edge replay-edge" style="--delay: 620ms" d="M 415 195 C 415 145 520 155 520 115" />
          <path class="edge replay-edge" style="--delay: 1050ms" d="M 225 285 C 265 285 270 245 315 245" />
          <path class="edge replay-edge" style="--delay: 1300ms" d="M 515 235 H 625" />
          <path class="edge replay-edge" style="--delay: 1300ms" d="M 560 115 C 560 155 705 155 705 195" />
          <path class="edge replay-edge" style="--delay: 1650ms" d="M 785 235 H 835" />

          <text class="edge-label" x="385" y="68" text-anchor="middle">capture once</text>
          <text class="loop-label" x="585" y="315" text-anchor="middle">refresh buffer → replay graph → consume output</text>

          <circle class="flow-pulse input-flow-pulse" r="5">
            <animate attributeName="opacity" begin="1.15s" dur="0.15s" from="0" to="0.9" fill="freeze" />
            <animateMotion
              begin="1.15s"
              dur="1.25s"
              path="M 225 285 C 265 285 270 245 315 245"
              repeatCount="indefinite"
            />
          </circle>
          <circle class="flow-pulse is-one" r="5">
            <animate attributeName="opacity" begin="2.2s" dur="0.15s" from="0" to="0.9" fill="freeze" />
            <animateMotion begin="2.2s" dur="2.8s" path="M 515 235 H 835" repeatCount="indefinite" />
          </circle>
          <circle class="flow-pulse is-two" r="5">
            <animate attributeName="opacity" begin="3.1s" dur="0.15s" from="0" to="0.9" fill="freeze" />
            <animateMotion begin="3.1s" dur="2.8s" path="M 515 235 H 835" repeatCount="indefinite" />
          </circle>
          <circle class="flow-pulse is-three" r="5">
            <animate attributeName="opacity" begin="4s" dur="0.15s" from="0" to="0.9" fill="freeze" />
            <animateMotion begin="4s" dur="2.8s" path="M 515 235 H 835" repeatCount="indefinite" />
          </circle>

          <g class="node owner-node" style="--delay: 0ms" transform="translate(145 40)">
            <rect width="190" height="75" rx="8" />
            <text x="95" y="31" text-anchor="middle">Program</text>
            <text class="detail" x="95" y="54" text-anchor="middle">linked Program callable</text>
          </g>
          <g class="node graph-node" style="--delay: 620ms" transform="translate(435 40)">
            <rect width="210" height="75" rx="8" />
            <text x="105" y="31" text-anchor="middle">Captured CUDA Graph</text>
            <text class="detail" x="105" y="54" text-anchor="middle">Program work at stable addresses</text>
          </g>

          <g class="node input-node" style="--delay: 0ms" transform="translate(75 175)">
            <rect width="150" height="60" rx="8" />
            <text x="75" y="25" text-anchor="middle">Input</text>
            <text class="detail" x="75" y="45" text-anchor="middle">initial Tensor payload</text>
          </g>
          <g class="node input-node" style="--delay: 950ms" transform="translate(75 255)">
            <rect width="150" height="60" rx="8" />
            <text x="75" y="25" text-anchor="middle">Next input</text>
            <text class="detail" x="75" y="45" text-anchor="middle">new Tensor payload</text>
          </g>
          <g class="node buffer-node" style="--delay: 260ms" transform="translate(315 195)">
            <rect width="200" height="80" rx="8" />
            <text x="100" y="32" text-anchor="middle">Buffer</text>
            <text class="detail" x="100" y="55" text-anchor="middle">one stable value view</text>
          </g>
          <g class="node replay-node" style="--delay: 1250ms" transform="translate(625 195)">
            <rect width="160" height="80" rx="8" />
            <text x="80" y="32" text-anchor="middle">Graph replay</text>
            <text class="detail" x="80" y="55" text-anchor="middle">fixed addresses</text>
          </g>
          <g class="node native-node" style="--delay: 1600ms" transform="translate(835 195)">
            <rect width="130" height="80" rx="8" />
            <text x="65" y="32" text-anchor="middle">Output view</text>
            <text class="detail" x="65" y="55" text-anchor="middle">consume result</text>
          </g>
        </svg>
      </div>

      <div :key="`phases-${activeKey}`" class="route-phases" aria-hidden="true">
        <template v-if="activeLevel === 'eager'">
          <span style="--delay: 0ms">Call</span><i />
          <span style="--delay: 180ms">Transition</span><i />
          <span style="--delay: 360ms">Dispatch</span><i />
          <span style="--delay: 720ms">Execute</span>
        </template>
        <template v-else-if="activeLevel === 'compile'">
          <span style="--delay: 0ms">Capture</span><i />
          <span style="--delay: 620ms">Recognize</span><i />
          <span style="--delay: 1450ms">Schedule</span><i />
          <span style="--delay: 2050ms">Lower</span><i />
          <span style="--delay: 2550ms">Link</span>
        </template>
        <template v-else-if="activeLevel === 'stack'">
          <span style="--delay: 0ms">Owner</span><i />
          <span style="--delay: 300ms">Dispatch</span><i />
          <span style="--delay: 700ms">Backend</span><i />
          <span style="--delay: 1500ms">Native</span>
        </template>
        <template v-else>
          <span style="--delay: 0ms">Prepare</span><i />
          <span style="--delay: 300ms">Capture</span><i />
          <span style="--delay: 1150ms">Refresh</span><i />
          <span style="--delay: 1450ms">Replay</span>
        </template>
      </div>
    </div>

    <div class="graph-caption" role="status" aria-live="polite">
      <div class="graph-caption-copy">
        <strong>{{ caption.title }}</strong>
        <span>{{ caption.detail }}</span>
      </div>
      <a :href="withBase(caption.href)" @click="openExample($event, caption.href)">
        {{ caption.example }}
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
  grid-template-columns: repeat(4, minmax(0, 1fr));
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

.stack-edge {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 50%, var(--fhe-c-helium));
  marker-end: url(#stack-control-arrow);
}

.stack-route {
  animation: alternate-stack-route 6s ease-in-out infinite;
  will-change: opacity;
}

.stack-program-route {
  animation-delay: -3s;
}

.replay-edge {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 68%, var(--fhe-c-helium));
  marker-end: url(#replay-control-arrow);
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

.stack-graph marker path {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 50%, var(--fhe-c-helium));
}

.replay-graph marker path {
  stroke: color-mix(in srgb, var(--fhe-c-brand) 68%, var(--fhe-c-helium));
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
.transition-node rect,
.executable-node rect {
  fill: color-mix(in srgb, var(--fhe-c-helium) 5%, var(--fhe-c-surface));
  stroke: var(--fhe-c-helium);
}

.backend-node rect {
  fill: color-mix(in srgb, var(--fhe-c-brand) 4%, var(--fhe-c-surface));
  stroke: color-mix(in srgb, var(--fhe-c-brand) 58%, var(--fhe-c-helium));
}

.dispatcher-node rect,
.resource-node rect {
  fill: var(--fhe-c-surface-tint);
  stroke: var(--fhe-c-border);
}

.native-node rect {
  fill: color-mix(in srgb, var(--fhe-c-helium) 7%, var(--fhe-c-surface));
  stroke: var(--fhe-c-helium);
}

.capture-node rect,
.graph-node rect,
.buffer-node rect,
.replay-node rect {
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

.semantic-label {
  animation: cycle-semantic-label 4s ease-in-out infinite;
}

.ckks-label {
  opacity: 0;
  animation: cycle-ckks-label 4s ease-in-out infinite;
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

.loop-label {
  fill: var(--fhe-c-text-3);
  font-family: var(--vp-font-family-base);
  font-size: 12px;
  font-weight: 700;
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
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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

@keyframes cycle-semantic-label {
  0%, 24%, 100% { opacity: 1; }
  38%, 84% { opacity: 0; }
}

@keyframes cycle-ckks-label {
  0%, 24%, 100% { opacity: 0; }
  38%, 84% { opacity: 1; }
}

@keyframes phase-in {
  from { opacity: 0; transform: translateX(-5px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes alternate-stack-route {
  0%, 42%, 100% { opacity: 1; }
  50%, 92% { opacity: 0.28; }
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
    white-space: normal;
  }

}

@media (prefers-reduced-motion: reduce) {
  .edge,
  .node,
  .semantic-label,
  .ckks-label,
  .route-phases span {
    animation-delay: 0ms;
    animation-duration: 1ms;
  }

  .stack-route {
    animation: none;
    opacity: 1;
  }

  .semantic-label {
    animation: none;
    opacity: 1;
  }

  .ckks-label {
    animation: none;
    opacity: 0;
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
