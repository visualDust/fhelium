<script setup lang="ts">
import { computed, ref } from 'vue'
import { highlightPython } from './pythonHighlight'

type RouteId = 'direct' | 'tuned' | 'jit' | 'spmd'

interface RouteOption {
  id: RouteId
  number: string
  label: string
  qualifier: string
  path: readonly string[]
  code: string
  state: readonly { label: string, value: string }[]
  note: string
}

const routes: readonly RouteOption[] = [
  {
    id: 'direct',
    number: '01',
    label: 'Direct CKKS',
    qualifier: 'Core API · eager CPU or CUDA',
    path: ['Python', 'CkksEngine', 'Typed state transitions', 'cpu / cuda:0'],
    code: `import fhelium as fh


def diagonal_matvec(engine, source, packed_diagonals, rotation_keys):
    rotated_values = []
    plaintexts = []
    for step, diagonal in enumerate(packed_diagonals):
        rotated = (
            source
            if step == 0
            else engine.rotate_with_key(source, rotation_keys[step])
        )
        plaintext = engine.prepare_plaintext_for_multiplication(
            engine.plaintext(
                diagonal,
                level=source.level,
                scale=engine.config.default_scale,
            ),
            modulus_basis=source.modulus_basis,
        )
        rotated_values.append(rotated)
        plaintexts.append(plaintext)

    rotated_ntt = engine.coefficient_domain_to_ntt_domain(
        fh.Ciphertext.stack_batch(rotated_values)
    )
    products = engine.multiply_plaintext(
        rotated_ntt, fh.Plaintext.stack_batch(plaintexts)
    )
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.sum_ciphertext_batch(products)
        )
    )`,
    state: [
      { label: 'Value', value: 'core Ciphertext · Q / NTT · 2 components' },
      { label: 'Schedule', value: 'rotate → batch → NTT → multiply → reduce → rescale' },
      { label: 'Keys', value: 'caller-owned rotation-key mapping' },
      { label: 'Placement', value: 'one process-local engine and CPU or CUDA device' },
    ],
    note: 'The application names every CKKS transition. Public operations validate level, scale, basis, domain, and key requirements.',
  },
  {
    id: 'tuned',
    number: '02',
    label: 'Manually tuned',
    qualifier: 'Hoisted rotation · CUDA Graph',
    path: ['CKKS schedule', 'Hoisted rotation', 'Fixed input buffer', 'CUDA Graph'],
    code: `import fhelium as fh
from fhelium.execution import CudaGraphProgram


def tuned_matvec(source, *, engine, diagonals, rotation_keys):
    # Hoist one hybrid decomposition across every direct rotation.
    rotated = fh.Ciphertext.stack_batch(
        [source, *engine.rotate_many_with_keys(source, rotation_keys)]
    )
    products = engine.multiply_plaintext(
        engine.coefficient_domain_to_ntt_domain(rotated),
        fh.Plaintext.stack_batch(diagonals),
    )
    # Keep equal-scale products at one level and rescale once after the sum.
    return engine.rescale_to_next_level(
        engine.ntt_domain_to_coefficient_domain(
            engine.sum_ciphertext_batch(products)
        )
    )


@CudaGraphProgram.capture(example_inputs=(prototype,))
def graph(source):
    return tuned_matvec(
        source,
        engine=engine,
        diagonals=diagonals,
        rotation_keys=rotation_keys,
    )

try:
    output = graph.replay(next_ciphertext, copy_output=True)
finally:
    graph.close()`,
    state: [
      { label: 'Value', value: 'the same core Ciphertext input and output' },
      { label: 'Schedule', value: 'hoisted decomposition · add-only reduction · one rescale' },
      { label: 'Keys', value: 'caller-owned rotation-key mapping bound as static state' },
      { label: 'Placement', value: 'one CUDA device · fixed graph input/output storage' },
    ],
    note: 'The callable remains valid eagerly. Capture adds fixed-address input staging and launch replay without changing the CKKS operation schedule.',
  },
  {
    id: 'jit',
    number: '03',
    label: 'JIT',
    qualifier: 'PyTorch or IR · one Program',
    path: ['Typed PyTorch', 'xDSL Program', 'Selected passes', 'Readiness + run'],
    code: `import torch
import fhelium as fh
from fhelium.experimental import jit


def matrix_vector_quadratic(x, weight, bias, size, repeats):
    result = x * torch.diagonal(weight).repeat(repeats)
    for step in range(1, size):
        diagonal = torch.diagonal(
            torch.roll(weight, shifts=step, dims=1)
        ).repeat(repeats)
        result += torch.roll(x, shifts=step, dims=-1) * diagonal

    affine = result + bias.repeat(repeats)
    return (affine + 0.25) * (affine - 0.5)


captured = jit.trace(
    matrix_vector_quadratic,
    inputs={
        "x": jit.encrypted(),
        "weight": jit.message(),
        "bias": jit.message(),
        "size": jit.static(8),
        "repeats": jit.static(engine.num_slots // 8),
    },
)
lowered = jit.default_pipeline().run(
    captured.program, captured.workspace,
)
requirements = jit.analyze_evaluation_key_requirements(lowered.program)
evaluation_keys = fh.EvaluationKeySet(
    rotations=fh.RotationKeySet({
        step: engine.rotation_key(step)
        for step in requirements.rotation_steps
    }),
    relinearization=engine.relinearization_key,
)
lowered.workspace.update({
    "engine": engine,
    "evaluation_keys": evaluation_keys,
})
ct_y = lowered.program.run(
    ct_x, weight, bias, workspace=lowered.workspace,
)`,
    state: [
      { label: 'Value', value: 'one mixed-dialect xDSL Program' },
      { label: 'Schedule', value: 'ordered pass tuple · reported changes' },
      { label: 'Keys', value: 'rotations 1…7 · relinearization' },
      { label: 'Runtime', value: 'retained Workspace · independent readiness gate' },
    ],
    note: 'Tracing and textual import produce the same Program class. Live engines, keys, materials, resources, and handlers remain outside the serializable IR.',
  },
  {
    id: 'spmd',
    number: '04',
    label: 'Multi-GPU SPMD',
    qualifier: 'torchrun · rank ownership',
    path: ['torchrun', 'Rank-local CkksEngine', 'Local diagonals', 'Typed reduce'],
    code: `# Launch with:
# torchrun --standalone --nproc-per-node=2 worker.py
import fhelium as fh
import fhelium.distributed as dist


dist.init()
rank = dist.get_rank()
world_size = dist.get_world_size()
engine = fh.CkksEngine(
    fh.Preset.slots32768_scale40_levels34_int64,
    device=dist.local_device(),
    allow_sk_gen=False,
)

# Rank 0 owns encryption. Every process receives the same packed input,
# then evaluates only the application-selected diagonal partition.
source = dist.broadcast_ciphertext(
    root_source if rank == 0 else None,
    src=0,
)
local_steps = range(rank, matrix_size, world_size)
local_terms = []
for step in local_steps:
    rotated = (
        source.clone()
        if step == 0
        else engine.rotate_with_key(source, local_rotation_keys[step])
    )
    diagonal = engine.prepare_plaintext_for_multiplication(
        engine.encode(cyclic_diagonals[step], level=source.level)
    )
    local_terms.append(
        engine.rescale_to_next_level(
            engine.ntt_domain_to_coefficient_domain(
                engine.multiply_plaintext(
                    engine.coefficient_domain_to_ntt_domain(rotated),
                    diagonal,
                )
            )
        )
    )

local_partial = engine.sum_ciphertexts(local_terms)
# Only dst is guaranteed to hold the complete sum; other rank-local values
# are partial or unchanged after this typed collective.
dist.reduce_ciphertext(local_partial, dst=0, engine=engine)
dist.shutdown()`,
    state: [
      { label: 'Value', value: 'rank-local Ciphertext values with typed communication' },
      { label: 'Schedule', value: 'application partition → local evaluator → typed reduce' },
      { label: 'Keys', value: 'each rank receives only its selected rotation keys' },
      { label: 'Placement', value: 'rank = get_rank() · device = local_device()' },
    ],
    note: 'torchrun starts the processes; the application assigns rank work, key placement, collectives, and shutdown for this evaluator.',
  },
]

const activeRouteId = ref<RouteId>('jit')
const activeRoute = computed(
  () => routes.find((route) => route.id === activeRouteId.value) ?? routes[0],
)
const highlightedRouteCode = computed(() => highlightPython(activeRoute.value.code))
</script>

<template>
  <section class="control-levels" aria-label="FHElium evaluator control levels">
    <div class="route-picker" aria-label="Evaluator control levels">
      <button
        v-for="route in routes"
        :key="route.id"
        type="button"
        :class="{ 'is-active': activeRouteId === route.id }"
        :aria-pressed="activeRouteId === route.id"
        @click="activeRouteId = route.id"
      >
        <span>{{ route.number }}</span>
        <strong>{{ route.label }}</strong>
        <small>{{ route.qualifier }}</small>
      </button>
    </div>

    <div class="route-console">
      <div class="route-lane" tabindex="0" :aria-label="`${activeRoute.label} execution route`">
        <template v-for="(step, index) in activeRoute.path" :key="step">
          <span>{{ step }}</span>
          <i v-if="index < activeRoute.path.length - 1" aria-hidden="true">→</i>
        </template>
      </div>

      <div class="route-console-body">
        <pre class="route-code" tabindex="0"><code v-html="highlightedRouteCode" class="home-python-code" /></pre>

        <aside class="state-readout" aria-label="Selected route state">
          <span class="readout-title">Value state</span>
          <dl>
            <div v-for="item in activeRoute.state" :key="item.label">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </div>
          </dl>
          <p>{{ activeRoute.note }}</p>
        </aside>
      </div>
    </div>
  </section>
</template>

<style scoped>
.control-levels {
  --deck-ink: #172033;
  --deck-muted: #626d82;
  --deck-night: #f5f7fc;
  --deck-panel: #edf1f9;
  --deck-line: #cbd3e3;
  --deck-blue: #3159d5;
  --deck-blue-soft: rgba(49, 89, 213, 0.1);
  --deck-amber: #875404;
  --deck-amber-text: #875404;
  --deck-grid: rgba(49, 89, 213, 0.035);
  margin: 30px 0 36px;
}

:global(.dark .control-levels) {
  --deck-ink: #eef1fb;
  --deck-muted: #9ca6bb;
  --deck-night: #11141d;
  --deck-panel: #181d29;
  --deck-line: rgba(154, 172, 225, 0.23);
  --deck-blue: #7590ff;
  --deck-blue-soft: rgba(117, 144, 255, 0.15);
  --deck-amber: #e3a742;
  --deck-amber-text: #ffd58c;
  --deck-grid: rgba(117, 144, 255, 0.025);
}

.route-picker {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  margin: 0;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px 7px 0 0;
  background: var(--vp-c-divider);
}

.route-picker button {
  position: relative;
  display: grid;
  min-width: 0;
  min-height: 92px;
  appearance: none;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 2px 10px;
  align-content: center;
  padding: 14px 15px;
  border: 0;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.route-picker button::after {
  position: absolute;
  right: 14px;
  bottom: 0;
  left: 14px;
  height: 3px;
  background: transparent;
  content: "";
  transform: scaleX(0.25);
  transition: transform 180ms ease, background-color 180ms ease;
}

.route-picker button:hover {
  color: var(--vp-c-text-1);
  background: color-mix(in srgb, var(--vp-c-bg) 82%, var(--vp-c-brand-soft));
}

.route-picker button.is-active {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
}

.route-picker button.is-active::after {
  background: var(--vp-c-brand-1);
  transform: scaleX(1);
}

.route-picker button:focus-visible {
  z-index: 2;
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: -3px;
}

.route-picker button > span {
  grid-row: 1 / 3;
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
  font-size: 10px;
  font-weight: 800;
  line-height: 1.5;
}

.route-picker strong,
.route-picker small {
  overflow: hidden;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.route-picker strong {
  color: inherit;
  font-size: 13px;
}

.route-picker small {
  color: var(--vp-c-text-3);
  font-size: 10px;
}

.route-console {
  margin: 0;
  overflow: hidden;
  border: 1px solid var(--deck-line);
  border-top: 0;
  border-radius: 0 0 8px 8px;
  color: var(--deck-ink);
  background:
    linear-gradient(var(--deck-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--deck-grid) 1px, transparent 1px),
    var(--deck-night);
  background-size: 24px 24px;
}

.route-lane {
  display: flex;
  overflow-x: auto;
  align-items: center;
  gap: 9px;
  padding: 13px 20px;
  border-bottom: 1px solid var(--deck-line);
  scrollbar-width: thin;
}

.route-lane span {
  flex: 0 0 auto;
  padding: 5px 8px;
  border: 1px solid var(--deck-line);
  border-radius: 4px;
  color: var(--deck-ink);
  background: var(--deck-blue-soft);
  font-family: var(--vp-font-family-mono);
  font-size: 10px;
  font-weight: 650;
  line-height: 1.35;
}

.route-lane span:last-of-type {
  border-color: color-mix(in srgb, var(--deck-amber) 55%, transparent);
  color: var(--deck-amber-text);
  background: color-mix(in srgb, var(--deck-amber) 12%, transparent);
}

.route-lane i {
  flex: 0 0 auto;
  color: var(--deck-blue);
  font-size: 11px;
  font-style: normal;
}

.route-console-body {
  display: grid;
  grid-template-columns: minmax(0, 1.42fr) minmax(15rem, 0.58fr);
  height: 500px;
}

.route-code {
  min-width: 0;
  height: 100%;
  margin: 0;
  overflow: auto;
  padding: 20px 22px 24px;
  border: 0;
  border-right: 1px solid var(--deck-line);
  border-radius: 0;
  color: var(--deck-ink);
  background: transparent !important;
  box-shadow: none;
  tab-size: 4;
}

.route-code code {
  color: inherit;
  background: transparent !important;
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  line-height: 1.72;
  white-space: pre;
}

.route-code:focus-visible {
  outline: 2px solid var(--deck-blue);
  outline-offset: -3px;
}

.route-lane:focus-visible {
  outline: 2px solid var(--deck-blue);
  outline-offset: -3px;
}

.state-readout {
  min-width: 0;
  height: 100%;
  overflow-y: auto;
  padding: 18px;
  background: var(--deck-panel);
}

.readout-title {
  display: block;
  margin-bottom: 11px;
  color: var(--deck-amber);
  font-family: var(--vp-font-family-mono);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  line-height: 1.4;
  text-transform: uppercase;
}

.state-readout dl {
  display: grid;
  margin: 0;
}

.state-readout dl > div {
  padding: 12px 0;
  border-top: 1px solid var(--deck-line);
}

.state-readout dt {
  margin-bottom: 4px;
  color: var(--deck-muted);
  font-family: var(--vp-font-family-mono);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.state-readout dd {
  margin: 0;
  color: var(--deck-ink);
  font-size: 11px;
  line-height: 1.55;
}

.state-readout > p {
  margin: 16px 0 0;
  padding: 12px;
  border-left: 2px solid var(--deck-amber);
  color: var(--deck-muted);
  background: color-mix(in srgb, var(--deck-amber) 7%, var(--deck-panel));
  font-size: 10px;
  line-height: 1.62;
}

@media (max-width: 900px) {
  .route-picker {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .route-console-body {
    grid-template-columns: minmax(0, 1fr);
    height: auto;
  }

  .route-code {
    height: 420px;
    border-right: 0;
    border-bottom: 1px solid var(--deck-line);
  }

  .state-readout {
    height: auto;
    overflow-y: visible;
  }
}

@media (max-width: 640px) {
  .route-picker {
    grid-template-columns: minmax(0, 1fr);
  }

  .route-picker button {
    min-height: 72px;
  }

  .route-code {
    height: 360px;
    padding-right: 16px;
    padding-left: 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .route-picker button::after {
    transition: none;
  }
}
</style>
