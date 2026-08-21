<script setup lang="ts">
import { computed, ref } from 'vue'

type AuthoringId = 'direct' | 'jit'
type PlanningId = 'direct-explicit' | 'jit-default' | 'jit-custom'
type ExecutionId = 'eager' | 'cuda-graph'
type PlacementId = 'cpu' | 'cuda' | 'spmd'

interface AuthoringOption {
  id: AuthoringId
  label: string
  detail: string
  signal: string
}

interface PlanningOption {
  id: PlanningId
  label: string
  directDetail: string
  compatibleWith: readonly AuthoringId[]
  requirement: string
}

interface ExecutionOption {
  id: ExecutionId
  label: string
  directDetail: string
  jitDetail: string
  compatibleWith: readonly AuthoringId[]
  requirement: string
}

interface PlacementOption {
  id: PlacementId
  label: string
  detail: string
}

interface RouteNode {
  label: string
  detail: string
  tone: 'source' | 'plan' | 'execute' | 'place' | 'collective'
}

const authoringOptions: readonly AuthoringOption[] = [
  {
    id: 'direct',
    label: 'Direct evaluator',
    detail: 'Write CkksEngine operations and own the arithmetic schedule.',
    signal: 'Python → CkksEngine',
  },
  {
    id: 'jit',
    label: 'JIT',
    detail: 'Trace PyTorch or import textual IR into one xDSL Program.',
    signal: 'PyTorch or IR → Program',
  },
]

const planningOptions: readonly PlanningOption[] = [
  {
    id: 'direct-explicit',
    label: 'Direct operations',
    directDetail: 'The caller selects NTT, relinearization, rescale, and alignment.',
    compatibleWith: ['direct'],
    requirement: 'Direct evaluator route only',
  },
  {
    id: 'jit-default',
    label: 'Default passes',
    directDetail: 'Apply the inspectable general lowering policy to one clone.',
    compatibleWith: ['jit'],
    requirement: 'Requires a JIT Program',
  },
  {
    id: 'jit-custom',
    label: 'Selected passes',
    directDetail: 'Compose local transforms, analyses, and selected validators.',
    compatibleWith: ['jit'],
    requirement: 'Requires a JIT Program',
  },
]

const executionOptions: readonly ExecutionOption[] = [
  {
    id: 'eager',
    label: 'Eager evaluation',
    directDetail: 'Ordinary direct Python calls into one local engine.',
    jitDetail: 'Program.run checks readiness against one retained Workspace.',
    compatibleWith: ['direct', 'jit'],
    requirement: '',
  },
  {
    id: 'cuda-graph',
    label: 'CUDA Graph replay',
    directDetail: 'Capture a fixed direct callable with bound resident artifacts.',
    jitDetail: '',
    compatibleWith: ['direct'],
    requirement: 'Use a fixed direct callable for CUDA Graph capture',
  },
]

const placementOptions: readonly PlacementOption[] = [
  {
    id: 'cpu',
    label: 'Local CPU',
    detail: 'The engine and every resident value target process-local CPU memory.',
  },
  {
    id: 'cuda',
    label: 'Local CUDA',
    detail: 'The engine and every resident value target one selected CUDA device.',
  },
  {
    id: 'spmd',
    label: 'Rank-local SPMD',
    detail: 'The application partitions work and composes results with typed collectives.',
  },
]

const authoringId = ref<AuthoringId>('jit')
const planningId = ref<PlanningId>('jit-default')
const executionId = ref<ExecutionId>('eager')
const placementId = ref<PlacementId>('cuda')

const authoring = computed(
  () => authoringOptions.find((option) => option.id === authoringId.value) ?? authoringOptions[0],
)
const planning = computed(
  () => planningOptions.find((option) => option.id === planningId.value) ?? planningOptions[0],
)
const execution = computed(
  () => executionOptions.find((option) => option.id === executionId.value) ?? executionOptions[0],
)
const placement = computed(
  () => placementOptions.find((option) => option.id === placementId.value) ?? placementOptions[0],
)

function planningCompatible(option: PlanningOption): boolean {
  return option.compatibleWith.includes(authoringId.value)
}

function executionCompatible(option: ExecutionOption): boolean {
  return (
    option.compatibleWith.includes(authoringId.value)
    && (option.id !== 'cuda-graph' || placementId.value !== 'cpu')
  )
}

function planningDetail(option: PlanningOption): string {
  return option.directDetail
}

function executionDetail(option: ExecutionOption): string {
  if (option.id === 'cuda-graph' && placementId.value === 'cpu') {
    return 'CUDA Graph capture requires local CUDA placement'
  }
  return authoringId.value === 'direct' ? option.directDetail : option.jitDetail
}

function selectAuthoring(next: AuthoringId): void {
  authoringId.value = next
  if (next === 'direct') {
    planningId.value = 'direct-explicit'
    return
  }
  planningId.value = 'jit-default'
  executionId.value = 'eager'
}

function selectExecution(next: ExecutionId): void {
  executionId.value = next
  if (next === 'cuda-graph') {
    placementId.value = 'cuda'
  }
}

function selectPlacement(next: PlacementId): void {
  placementId.value = next
  if (next === 'cpu') {
    executionId.value = 'eager'
  }
}

const routeNodes = computed<readonly RouteNode[]>(() => {
  const nodes: RouteNode[] = [
    {
      label: authoringId.value === 'direct' ? 'Direct Python' : 'PyTorch or textual IR',
      detail: authoringId.value === 'direct' ? 'CkksEngine API' : 'one Program class',
      tone: 'source',
    },
  ]

  if (authoringId.value === 'direct') {
    nodes.push({
      label: 'Lowered CKKS',
      detail: 'caller-owned schedule',
      tone: 'plan',
    })
  } else {
    nodes.push(
      {
        label: 'xDSL Program',
        detail: planningId.value === 'jit-custom' ? 'selected local passes' : 'default pipeline',
        tone: 'plan',
      },
      {
        label: 'Workspace',
        detail: 'materials + capabilities',
        tone: 'plan',
      },
      {
        label: 'Readiness',
        detail: 'selected entry + interpreter schemas',
        tone: 'plan',
      },
    )
  }

  if (executionId.value === 'eager') {
    nodes.push({
      label: authoringId.value === 'direct' ? 'Engine calls' : 'Program.run',
      detail: 'ordinary eager launch',
      tone: 'execute',
    })
  } else {
    nodes.push({
      label: 'CUDA Graph',
      detail: 'fixed direct callable',
      tone: 'execute',
    })
  }

  nodes.push({
    label:
      placementId.value === 'cpu'
        ? 'cpu'
        : placementId.value === 'cuda'
          ? 'cuda:N'
          : 'Rank-local CUDA',
    detail:
      placementId.value === 'cpu'
        ? 'process-local execution'
        : placementId.value === 'cuda'
          ? 'one selected device'
          : 'application-owned work',
    tone: 'place',
  })

  if (placementId.value === 'spmd') {
    nodes.push({
      label: 'Typed collective',
      detail: 'application-owned collective',
      tone: 'collective',
    })
  }

  return nodes
})

const routeStatus = computed(
  () => `${authoring.value.label}; ${planning.value.label}; ${execution.value.label}; ${placement.value.label}`,
)
</script>

<template>
  <section class="stack-builder" aria-label="Interactive FHElium stack builder">
    <div class="patch-bay">
      <div class="stage-grid">
        <fieldset class="stack-stage is-authoring">
          <legend><span>01</span> Author</legend>
          <button
            v-for="option in authoringOptions"
            :key="option.id"
            type="button"
            :class="{ 'is-active': authoringId === option.id }"
            :aria-pressed="authoringId === option.id"
            @click="selectAuthoring(option.id)"
          >
            <i aria-hidden="true" />
            <span>
              <strong>{{ option.label }}</strong>
              <small>{{ option.detail }}</small>
            </span>
          </button>
        </fieldset>

        <fieldset class="stack-stage is-planning">
          <legend><span>02</span> Plan</legend>
          <button
            v-for="option in planningOptions"
            :key="option.id"
            type="button"
            :class="{ 'is-active': planningId === option.id }"
            :disabled="!planningCompatible(option)"
            :aria-pressed="planningId === option.id"
            @click="planningId = option.id"
          >
            <i aria-hidden="true" />
            <span>
              <strong>{{ option.label }}</strong>
              <small>{{ planningCompatible(option) ? planningDetail(option) : option.requirement }}</small>
            </span>
          </button>
        </fieldset>

        <fieldset class="stack-stage is-execution">
          <legend><span>03</span> Execute</legend>
          <button
            v-for="option in executionOptions"
            :key="option.id"
            type="button"
            :class="{ 'is-active': executionId === option.id }"
            :disabled="!executionCompatible(option)"
            :aria-pressed="executionId === option.id"
            @click="selectExecution(option.id)"
          >
            <i aria-hidden="true" />
            <span>
              <strong>{{ option.label }}</strong>
              <small>{{ executionDetail(option) || option.requirement }}</small>
            </span>
          </button>
        </fieldset>

        <fieldset class="stack-stage is-placement">
          <legend><span>04</span> Place</legend>
          <button
            v-for="option in placementOptions"
            :key="option.id"
            type="button"
            :class="{ 'is-active': placementId === option.id }"
            :aria-pressed="placementId === option.id"
            @click="selectPlacement(option.id)"
          >
            <i aria-hidden="true" />
            <span>
              <strong>{{ option.label }}</strong>
              <small>{{ option.detail }}</small>
            </span>
          </button>
        </fieldset>
      </div>

      <div class="signal-rail" tabindex="0" aria-label="Selected evaluator route">
        <template v-for="(node, index) in routeNodes" :key="`${node.label}-${index}`">
          <div class="route-node" :class="`is-${node.tone}`">
            <span>{{ node.label }}</span>
            <small>{{ node.detail }}</small>
          </div>
          <div v-if="index < routeNodes.length - 1" class="route-wire" aria-hidden="true">
            <i />
          </div>
        </template>
      </div>

      <span class="route-status" role="status" aria-live="polite">
        Selected route: {{ routeStatus }}.
      </span>
    </div>

  </section>
</template>

<style scoped>
.stack-builder {
  --builder-night: #f5f7fc;
  --builder-panel: #edf1f9;
  --builder-line: #cbd3e3;
  --builder-ink: #172033;
  --builder-muted: #626d82;
  --builder-blue: #3159d5;
  --builder-blue-soft: rgba(49, 89, 213, 0.1);
  --builder-amber: #875404;
  --builder-grid: rgba(49, 89, 213, 0.035);
  --builder-button: rgba(49, 89, 213, 0.025);
  --builder-node: rgba(255, 255, 255, 0.66);
  margin: 28px 0 38px;
}

:global(.dark .stack-builder) {
  --builder-night: #11141c;
  --builder-panel: #181d29;
  --builder-line: rgba(155, 173, 225, 0.22);
  --builder-ink: #edf0fa;
  --builder-muted: #9aa5bb;
  --builder-blue: #7a94ff;
  --builder-blue-soft: rgba(122, 148, 255, 0.14);
  --builder-amber: #e4a53a;
  --builder-grid: rgba(122, 148, 255, 0.025);
  --builder-button: rgba(255, 255, 255, 0.025);
  --builder-node: rgba(17, 20, 28, 0.86);
}

.patch-bay {
  position: relative;
  margin: 0;
  overflow: hidden;
  border: 1px solid var(--builder-line);
  border-radius: 8px;
  color: var(--builder-ink);
  background:
    linear-gradient(var(--builder-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--builder-grid) 1px, transparent 1px),
    var(--builder-night);
  background-size: 22px 22px;
}

.stage-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0;
}

.stack-stage {
  position: relative;
  min-width: 0;
  margin: 0;
  padding: 18px 14px 16px;
  border: 0;
}

.stack-stage + .stack-stage {
  border-left: 1px solid var(--builder-line);
}

.stack-stage:not(:last-child)::after {
  position: absolute;
  z-index: 2;
  top: 32px;
  right: -5px;
  width: 10px;
  height: 10px;
  border: 2px solid var(--builder-blue);
  border-radius: 50%;
  background: var(--builder-night);
  box-shadow: 0 0 13px rgba(122, 148, 255, 0.46);
  content: "";
}

.stack-stage legend {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  padding: 14px 0 0;
  color: var(--builder-muted);
  font-family: var(--vp-font-family-mono);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.1em;
  line-height: 1.4;
  text-transform: uppercase;
}

.stack-stage legend span {
  color: var(--builder-blue);
}

.stack-stage button {
  position: relative;
  display: grid;
  width: 100%;
  min-height: 76px;
  grid-template-columns: 12px minmax(0, 1fr);
  gap: 9px;
  align-items: start;
  margin: 0 0 7px;
  padding: 11px;
  border: 1px solid transparent;
  border-radius: 5px;
  color: var(--builder-muted);
  background: var(--builder-button);
  cursor: pointer;
  font: inherit;
  text-align: left;
  transition: border-color 160ms ease, background-color 160ms ease, transform 160ms ease;
}

.stack-stage button:hover:not(:disabled) {
  border-color: var(--builder-line);
  color: var(--builder-ink);
  background: var(--builder-blue-soft);
  transform: translateY(-1px);
}

.stack-stage button.is-active {
  border-color: color-mix(in srgb, var(--builder-blue) 52%, var(--builder-line));
  color: var(--builder-ink);
  background: var(--builder-blue-soft);
  box-shadow: inset 3px 0 0 var(--builder-blue);
}

.stack-stage button:disabled {
  cursor: not-allowed;
  opacity: 0.36;
}

.stack-stage button:focus-visible {
  z-index: 3;
  outline: 2px solid var(--builder-amber);
  outline-offset: 2px;
}

.stack-stage button > i {
  display: block;
  width: 9px;
  height: 9px;
  margin-top: 3px;
  border: 1px solid var(--builder-line);
  border-radius: 50%;
  background: var(--builder-panel);
}

.stack-stage button.is-active > i {
  border-color: var(--builder-blue);
  background: var(--builder-blue);
  box-shadow: 0 0 0 3px rgba(122, 148, 255, 0.12), 0 0 12px rgba(122, 148, 255, 0.58);
}

.stack-stage button > span {
  display: grid;
  min-width: 0;
  gap: 5px;
}

.stack-stage button strong {
  color: inherit;
  font-size: 11px;
  line-height: 1.35;
}

.stack-stage button small {
  color: var(--builder-muted);
  font-size: 9px;
  line-height: 1.45;
}

.signal-rail {
  display: flex;
  overflow-x: auto;
  align-items: stretch;
  padding: 16px;
  border-top: 1px solid var(--builder-line);
  background: var(--builder-panel);
  scrollbar-width: thin;
}

.signal-rail:focus-visible {
  outline: 2px solid var(--builder-blue);
  outline-offset: -3px;
}

.route-node {
  display: grid;
  min-width: 112px;
  flex: 1 0 112px;
  align-content: center;
  gap: 4px;
  padding: 10px 11px;
  border: 1px solid var(--builder-line);
  border-radius: 5px;
  background: var(--builder-node);
}

.route-node span {
  color: var(--builder-ink);
  font-family: var(--vp-font-family-mono);
  font-size: 10px;
  font-weight: 700;
  line-height: 1.35;
}

.route-node small {
  color: var(--builder-muted);
  font-size: 8px;
  line-height: 1.35;
}

.route-node.is-plan {
  border-color: color-mix(in srgb, var(--builder-blue) 48%, var(--builder-line));
  background: color-mix(in srgb, var(--builder-blue) 8%, var(--builder-night));
}

.route-node.is-execute {
  border-color: color-mix(in srgb, var(--builder-amber) 48%, var(--builder-line));
  background: color-mix(in srgb, var(--builder-amber) 7%, var(--builder-night));
}

.route-node.is-place,
.route-node.is-collective {
  border-color: color-mix(in srgb, #766ccf 54%, var(--builder-line));
  background: rgba(118, 108, 207, 0.08);
}

.route-wire {
  position: relative;
  display: grid;
  width: 30px;
  flex: 0 0 30px;
  place-items: center;
}

.route-wire::before {
  width: 100%;
  height: 1px;
  background: var(--builder-line);
  content: "";
}

.route-wire i {
  position: absolute;
  left: 0;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--builder-blue);
  box-shadow: 0 0 8px rgba(122, 148, 255, 0.72);
  animation: stack-signal 2s linear infinite;
}

.route-status {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  white-space: nowrap;
}

@keyframes stack-signal {
  from {
    transform: translateX(0);
  }

  to {
    transform: translateX(30px);
  }
}

@media (max-width: 1000px) {
  .stage-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .stack-stage:nth-child(3) {
    border-top: 1px solid var(--builder-line);
    border-left: 0;
  }

  .stack-stage:nth-child(4) {
    border-top: 1px solid var(--builder-line);
  }

  .stack-stage:nth-child(2)::after {
    content: none;
  }

}

@media (max-width: 720px) {
  .stage-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .stack-stage + .stack-stage {
    border-left: 0;
  }

  .stack-stage + .stack-stage {
    border-top: 1px solid var(--builder-line);
  }

  .stack-stage:not(:last-child)::after {
    top: auto;
    right: 50%;
    bottom: -5px;
  }

  .signal-rail {
    align-items: center;
  }

}

@media (prefers-reduced-motion: reduce) {
  .stack-stage button {
    transition: none;
  }

  .route-wire i {
    animation: none;
    transform: translateX(12px);
  }
}
</style>
