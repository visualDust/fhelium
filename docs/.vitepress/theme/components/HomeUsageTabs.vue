<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'

import NavArrow from './NavArrow.vue'
import { useAutoAdvance } from '../composables/useAutoAdvance'

type UsageId = 'eager' | 'compile' | 'deploy' | 'serve' | 'distributed'

interface UsageTab {
  id: UsageId
  label: string
  title: string
  description: string
  link: string
  href: string
  code: string
}

const tabs: readonly UsageTab[] = [
  {
    id: 'eager',
    label: 'Eager',
    title: 'Execute one CKKS operation at a time',
    description:
      'Create keys and Tensor-backed values, then apply each operation and its depth, scale, and domain transition immediately on CPU or CUDA.',
    link: 'Open tutorial',
    href: '/tutorial/basic-ckks-workflow',
    code: `import torch
import fhelium as fh
from fhelium.eager import Engine

engine = Engine(fh.Preset.slots8192_scale40_depth7)
secret = engine.create_secret_key()
public = engine.create_public_key(secret)
rotation = engine.create_rotation_key(1, secret)
relin = engine.create_relinearization_key(secret)

message = torch.rand(engine.num_slots)
x = engine.encrypt_message(message, public)
rotated = engine.rotate_with_key(x, rotation)
product = engine.multiply(
    engine.coefficient_domain_to_ntt_domain(x),
    engine.coefficient_domain_to_ntt_domain(rotated),
)
y = engine.relinearize(product, relin, rescale=True)
print(y.depth, y.scale)`,
  },
  {
    id: 'compile',
    label: 'Compile',
    title: 'Compile the same code into a schedule',
    description:
      'The first call captures a Program, applies the default passes, and links an exec schedule; later calls with the same signature run that schedule.',
    link: 'Open tutorial',
    href: '/tutorial/compile-layer',
    code: `from fhelium import compile as fc

@fc.compile
def layer(x):
    rotated = engine.rotate_with_key(x, rotation)
    product = engine.multiply(
        engine.coefficient_domain_to_ntt_domain(x),
        engine.coefficient_domain_to_ntt_domain(rotated),
    )
    return engine.relinearize(product, relin, rescale=True)

y = layer(x)  # capture, passes, lowering, linking
y = layer(x)  # Runner executes the linked schedule
print(len(layer.specializations))`,
  },
  {
    id: 'deploy',
    label: 'Deploy',
    title: 'Export once and load in another process',
    description:
      'The artifact stores schedules, kernel images, and constants. The serving process supplies the keys and runs without capture or compilation.',
    link: 'Open tutorial',
    href: '/tutorial/export-and-load',
    code: `fc.export(layer, "layer.fhelium", signatures=[(x,)])
fh.save_value(rotation, "rotation.fhelium")
fh.save_value(relin, "relin.fhelium")

# Serving process
keys = [
    fh.load_value("rotation.fhelium"),
    fh.load_value("relin.fhelium"),
]
loaded = fc.load("layer.fhelium", keys=keys)
y = loaded(x)`,
  },
  {
    id: 'serve',
    label: 'Serve',
    title: 'Serve a stream of user requests',
    description:
      'Each request brings its ciphertext and keys. One compiled model serves every user while DoubleBuffer copies the next request.',
    link: 'Open guide',
    href: '/how-to/serve-compiled-requests',
    code: `from fhelium.runtime import DoubleBuffer

@fc.compile
def model(x, rotation, relin):
    rotated = engine.rotate_with_key(x, rotation)
    product = engine.multiply(
        engine.coefficient_domain_to_ntt_domain(x),
        engine.coefficient_domain_to_ntt_domain(rotated),
    )
    return engine.relinearize(product, relin, rescale=True)

# requests: (ciphertext, rotation key, relinearization key)
with DoubleBuffer(requests[0], device=x.device) as buffers:
    for _, (ct, rotation, relin) in buffers.feed(requests):
        respond(model(ct, rotation, relin))  # consumed here`,
  },
  {
    id: 'distributed',
    label: 'Distributed',
    title: 'Combine rank-local results with collectives',
    description:
      'Each process evaluates its own values; a ciphertext collective inside the compiled function becomes part of every rank\'s exec schedule.',
    link: 'Open guide',
    href: '/how-to/choose-multi-gpu-partition',
    code: `import fhelium.distributed as dist
from fhelium import compile as fc

dist.init()

@fc.compile
def total(partial):
    dist.all_reduce_ciphertext(partial, engine=engine)
    return partial

local = engine.encrypt_message(local_values, public)
result = total(local)  # every rank receives the sum
dist.shutdown()`,
  },
]

const activeId = ref<UsageId>('eager')
const activeTab = computed(
  () => tabs.find((tab) => tab.id === activeId.value) ?? tabs[0],
)

const { stop: stopRotation } = useAutoAdvance(
  tabs.map((tab) => tab.id),
  activeId,
  3000,
)

function selectTab(id: UsageId): void {
  stopRotation()
  activeId.value = id
}

const pythonKeywords = new Set([
  'as',
  'def',
  'del',
  'for',
  'else',
  'False',
  'from',
  'if',
  'import',
  'in',
  'is',
  'None',
  'return',
  'with',
  'True',
])

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
}

function highlightPython(source: string): string {
  let result = ''
  let index = 0

  while (index < source.length) {
    const character = source[index]
    if (character === '#') {
      const newline = source.indexOf('\n', index)
      const end = newline === -1 ? source.length : newline
      result += `<span class="code-comment">${escapeHtml(source.slice(index, end))}</span>`
      index = end
      continue
    }
    if (character === '"' || character === "'") {
      const quote = character
      let end = index + 1
      while (end < source.length) {
        if (source[end] === '\\') {
          end += 2
          continue
        }
        if (source[end] === quote) {
          end += 1
          break
        }
        end += 1
      }
      result += `<span class="code-string">${escapeHtml(source.slice(index, end))}</span>`
      index = end
      continue
    }
    if (/[0-9]/u.test(character)) {
      const match = source.slice(index).match(/^[0-9]+(?:\.[0-9]+)?/u)
      const token = match?.[0] ?? character
      result += `<span class="code-number">${token}</span>`
      index += token.length
      continue
    }
    if (/[A-Za-z_]/u.test(character)) {
      const match = source.slice(index).match(/^[A-Za-z_][A-Za-z0-9_]*/u)
      const token = match?.[0] ?? character
      const remaining = source.slice(index + token.length)
      let className = ''
      if (pythonKeywords.has(token)) className = 'code-keyword'
      else if (/^\s*\(/u.test(remaining)) className = 'code-call'
      else if (/^[A-Z]/u.test(token)) className = 'code-type'
      result += className
        ? `<span class="${className}">${token}</span>`
        : token
      index += token.length
      continue
    }
    result += escapeHtml(character)
    index += 1
  }

  return result
}

const highlightedCode = computed(() => highlightPython(activeTab.value.code))
</script>

<template>
  <section class="home-usage-tabs" aria-label="FHElium usage models">
    <div class="usage-tab-list" role="tablist" aria-label="Usage model">
      <button
        v-for="(tab, index) in tabs"
        :id="`usage-tab-${tab.id}`"
        :key="tab.id"
        type="button"
        role="tab"
        :aria-controls="`usage-panel-${tab.id}`"
        :aria-selected="activeId === tab.id"
        :tabindex="activeId === tab.id ? 0 : -1"
        :class="{ 'is-active': activeId === tab.id }"
        @click="selectTab(tab.id)"
      >
        <span>{{ String(index + 1).padStart(2, '0') }}</span>
        {{ tab.label }}
      </button>
    </div>

    <div
      :id="`usage-panel-${activeTab.id}`"
      class="usage-tab-panel"
      role="tabpanel"
      :aria-labelledby="`usage-tab-${activeTab.id}`"
    >
      <div class="usage-tab-copy">
        <p class="usage-tab-eyebrow">{{ activeTab.label }}</p>
        <h3>{{ activeTab.title }}</h3>
        <p>{{ activeTab.description }}</p>
        <a :href="withBase(activeTab.href)">
          {{ activeTab.link }}
          <NavArrow />
        </a>
      </div>

      <pre class="usage-tab-code"><code v-html="highlightedCode" /></pre>
    </div>
  </section>
</template>

<style scoped>
.home-usage-tabs {
  margin: 28px 0 52px;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--fhe-c-border) 72%, var(--fhe-c-divider));
  border-radius: 12px;
  background: var(--fhe-c-surface);
  box-shadow:
    0 2px 4px color-mix(in srgb, var(--fhe-c-text-1) 4%, transparent),
    0 22px 54px color-mix(in srgb, var(--fhe-c-text-1) 9%, transparent);
}

.usage-tab-list {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 7px;
  padding: 8px;
  border-bottom: 1px solid var(--fhe-c-divider);
  background: var(--fhe-c-canvas-alt);
}

.usage-tab-list button {
  position: relative;
  display: flex;
  min-width: 0;
  min-height: 70px;
  align-items: center;
  justify-content: flex-start;
  gap: 12px;
  border: 1px solid transparent;
  border-radius: 8px;
  padding: 12px 16px;
  color: var(--fhe-c-text-2);
  background: transparent;
  cursor: pointer;
  font: inherit;
  font-size: 16px;
  font-weight: 760;
  text-align: left;
  transition:
    color 140ms ease,
    border-color 140ms ease,
    background-color 140ms ease,
    box-shadow 140ms ease;
}

.usage-tab-list button::after {
  position: absolute;
  right: 13px;
  bottom: 5px;
  left: 13px;
  height: 3px;
  border-radius: 999px;
  background: transparent;
  content: "";
}

.usage-tab-list button span {
  color: var(--fhe-c-brand);
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  font-weight: 800;
}

.usage-tab-list button:hover {
  color: var(--fhe-c-text-1);
  border-color: color-mix(in srgb, var(--fhe-c-brand) 25%, var(--fhe-c-divider));
  background: color-mix(in srgb, var(--fhe-c-brand) 5%, var(--fhe-c-surface));
}

.usage-tab-list button:focus-visible {
  outline: 2px solid var(--fhe-c-focus);
  outline-offset: 2px;
}

.usage-tab-list button.is-active {
  color: var(--fhe-c-text-1);
  border-color: color-mix(in srgb, var(--fhe-c-brand) 30%, var(--fhe-c-divider));
  background: var(--fhe-c-surface);
  box-shadow:
    0 1px 2px color-mix(in srgb, var(--fhe-c-text-1) 5%, transparent),
    0 8px 18px color-mix(in srgb, var(--fhe-c-text-1) 7%, transparent);
}

.usage-tab-list button.is-active::after {
  background: linear-gradient(90deg, var(--fhe-c-brand), var(--fhe-c-helium));
}

.usage-tab-panel {
  display: grid;
  min-height: 410px;
  grid-template-columns: minmax(230px, 0.76fr) minmax(0, 1.5fr);
  gap: 40px;
  align-items: center;
  padding: 38px;
  background:
    radial-gradient(circle at 18% 45%, color-mix(in srgb, var(--fhe-c-brand) 5%, transparent), transparent 38%),
    var(--fhe-c-surface);
}

.usage-tab-copy {
  min-width: 0;
}

.usage-tab-eyebrow {
  margin: 0 0 10px;
  color: var(--fhe-c-brand);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.usage-tab-copy h3 {
  margin: 0 0 15px;
  border: 0;
  color: var(--fhe-c-text-1);
  font-size: clamp(23px, 2.5vw, 31px);
  letter-spacing: -0.025em;
  line-height: 1.14;
}

.usage-tab-copy > p:not(.usage-tab-eyebrow) {
  margin: 0;
  color: var(--fhe-c-text-2);
  font-size: 14px;
  line-height: 1.7;
}

.usage-tab-copy a {
  display: inline-flex;
  height: 36px;
  align-items: center;
  gap: 7px;
  margin-top: 24px;
  padding: 0 13px;
  border: 1px solid color-mix(in srgb, var(--fhe-c-brand) 34%, var(--fhe-c-border));
  border-radius: var(--fhe-radius-control);
  color: var(--fhe-c-brand);
  background: var(--fhe-c-surface);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.01em;
  text-decoration: none;
  transition:
    border-color 140ms ease,
    background-color 140ms ease,
    transform 140ms ease;
}

.usage-tab-copy a:hover {
  border-color: var(--fhe-c-brand);
  color: var(--fhe-c-brand-hover);
  background: var(--fhe-c-surface-tint);
  transform: translateY(-1px);
}

.usage-tab-copy a:focus-visible {
  outline: 2px solid var(--fhe-c-focus);
  outline-offset: 2px;
}

.usage-tab-code {
  min-width: 0;
  height: 342px;
  margin: 0;
  overflow: auto;
  border: 1px solid var(--fhe-c-divider);
  border-radius: 10px;
  padding: 24px 26px;
  color: var(--vp-code-block-color);
  background: var(--vp-code-block-bg);
  font-size: 12px;
  line-height: 1.62;
  box-shadow: inset 0 1px 0 color-mix(in srgb, white 6%, transparent);
  tab-size: 4;
}

.usage-tab-code code {
  color: inherit;
  font-family: var(--vp-font-family-mono);
}

.usage-tab-code :deep(.code-comment) {
  color: var(--fhe-code-comment);
}

.usage-tab-code :deep(.code-keyword) {
  color: var(--fhe-code-keyword);
}

.usage-tab-code :deep(.code-string) {
  color: var(--fhe-code-string);
}

.usage-tab-code :deep(.code-number) {
  color: var(--fhe-code-number);
}

.usage-tab-code :deep(.code-type) {
  color: var(--fhe-code-type);
}

.usage-tab-code :deep(.code-call) {
  color: var(--fhe-code-call);
}

@media (max-width: 760px) {
  .usage-tab-list {
    display: flex;
    overflow-x: auto;
  }

  .usage-tab-list button {
    min-height: 58px;
    flex: 0 0 155px;
    gap: 8px;
    padding: 10px 12px;
    font-size: 14px;
  }

  .usage-tab-panel {
    min-height: 0;
    grid-template-columns: minmax(0, 1fr);
    gap: 26px;
    padding: 28px 20px 20px;
  }

  .usage-tab-copy a {
    margin-top: 18px;
  }

  .usage-tab-code {
    height: 325px;
    padding: 18px;
    font-size: 11px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .usage-tab-list button,
  .usage-tab-copy a {
    transition: none;
  }
}

@media (forced-colors: active) {
  .home-usage-tabs,
  .usage-tab-list button,
  .usage-tab-code {
    border-color: CanvasText;
    background: Canvas;
  }
}
</style>
