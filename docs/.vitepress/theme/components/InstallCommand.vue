<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { withBase } from 'vitepress'
import {
  computeLabel,
  methodLabels,
  osLabels,
  type ComputeId,
  type InstallSelection,
  type MethodId,
  type OsId,
} from '../data/installSelector'
import {
  changeSelection,
  defaultSelection,
  optionsFor,
  resolveInstall,
  type InstallAxis,
} from '../data/installResolver'

const props = withDefaults(defineProps<{
  showBadges?: boolean
  showDetailsLink?: boolean
}>(), {
  showBadges: true,
  showDetailsLink: true,
})

const selection = ref<InstallSelection>(defaultSelection())
const resolved = computed(() => resolveInstall(selection.value))
const command = computed(() => resolved.value.commands.join('\n'))
const copied = ref(false)
let resetTimer: ReturnType<typeof setTimeout> | undefined

const groups = computed(() => [
  {
    axis: 'os' as const,
    label: 'Operating system',
    options: optionsFor('os', selection.value).map(id => ({ id, label: osLabels[id as OsId] })),
  },
  {
    axis: 'method' as const,
    label: 'Package method',
    options: optionsFor('method', selection.value).map(id => ({ id, label: methodLabels[id as MethodId] })),
  },
  {
    axis: 'torch' as const,
    label: 'Torch version',
    options: optionsFor('torch', selection.value).map(id => ({ id, label: `PyTorch ${id}` })),
  },
  {
    axis: 'compute' as const,
    label: 'Compute platform',
    options: optionsFor('compute', selection.value).map(id => ({ id, label: computeLabel(id as ComputeId) })),
  },
])

function selectOption(axis: InstallAxis, value: string): void {
  selection.value = changeSelection(selection.value, axis, value)
  copied.value = false
}

function selected(axis: InstallAxis, value: string): boolean {
  return selection.value[axis] === value
}

async function copyCommand(): Promise<void> {
  try {
    await navigator.clipboard.writeText(command.value)
  }
  catch {
    const textarea = document.createElement('textarea')
    textarea.value = command.value
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    textarea.remove()
  }

  copied.value = true
  if (resetTimer !== undefined) clearTimeout(resetTimer)
  resetTimer = setTimeout(() => {
    copied.value = false
  }, 1800)
}

onBeforeUnmount(() => {
  if (resetTimer !== undefined) clearTimeout(resetTimer)
})
</script>

<template>
  <div id="install-fhelium" class="home-install" aria-label="Install FHElium">
    <div class="home-install-copy">
      <strong>Install FHElium</strong>
      <div v-if="props.showBadges" class="home-install-badges" aria-label="FHElium package metadata">
        <a href="https://pypi.org/project/fhelium/">
          <img src="https://img.shields.io/pypi/v/fhelium" alt="PyPI version">
        </a>
        <a href="https://pypi.org/project/fhelium/">
          <img src="https://img.shields.io/pypi/pyversions/fhelium" alt="Supported Python versions">
        </a>
        <a href="https://github.com/VisualDust/fhelium/blob/main/LICENSE">
          <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license">
        </a>
      </div>
    </div>
    <a v-if="props.showDetailsLink" :href="withBase('/tutorial/installation')">Requirements and editable builds →</a>

    <div class="home-install-selector">
      <div
        v-for="group in groups"
        :key="group.axis"
        :class="['home-install-selector-row', `is-${group.axis}`]"
      >
        <span :id="`install-${group.axis}-label`" class="home-install-selector-label">{{ group.label }}</span>
        <div class="home-install-options" role="group" :aria-labelledby="`install-${group.axis}-label`">
          <button
            v-for="option in group.options"
            :key="option.id"
            type="button"
            :class="{ 'is-active': selected(group.axis, option.id) }"
            :aria-pressed="selected(group.axis, option.id)"
            @click="selectOption(group.axis, option.id)"
          >
            {{ option.label }}
          </button>
        </div>
      </div>
    </div>

    <div class="home-install-code">
      <pre><code>{{ command }}</code></pre>
      <button type="button" :aria-label="copied ? 'Commands copied' : 'Copy install commands'" @click="copyCommand">
        <svg v-if="!copied" aria-hidden="true" viewBox="0 0 20 20">
          <rect x="7" y="3" width="9" height="11" rx="1" />
          <path d="M13 16H5a1 1 0 0 1-1-1V7" />
        </svg>
        <svg v-else aria-hidden="true" viewBox="0 0 20 20">
          <path d="m4 10 4 4 8-9" />
        </svg>
      </button>
    </div>
    <p class="home-install-development">
      <svg viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="10" cy="10" r="7.25" />
        <path d="M10 6.2v4.8M10 14h.01" />
      </svg>
      <span>FHElium is under active development. APIs may change significantly between releases.</span>
    </p>
    <span class="home-install-status" role="status" aria-live="polite">{{ copied ? 'Install commands copied to clipboard.' : '' }}</span>
  </div>
</template>
