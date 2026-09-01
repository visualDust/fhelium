<script setup lang="ts">
import type { TopLevelSpec } from 'vega-lite'
import { withBase } from 'vitepress'
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps<{
  source: string
  label: string
  specification: (data: unknown, theme: 'light' | 'dark') => TopLevelSpec
}>()

const chart = ref<HTMLElement>()
const status = ref<'waiting' | 'loading' | 'ready' | 'error'>('waiting')
let data: unknown
let renderSequence = 0
let renderedView: {
  finalize: () => void
  resize: () => { runAsync: () => Promise<unknown> }
  runAsync: () => Promise<unknown>
} | undefined
let intersectionObserver: IntersectionObserver | undefined
let resizeObserver: ResizeObserver | undefined
let themeObserver: MutationObserver | undefined
let resizeFrame: number | undefined

function theme(): 'light' | 'dark' {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

async function render(): Promise<void> {
  const element = chart.value
  if (element === undefined) return
  const sequence = ++renderSequence
  status.value = 'loading'
  try {
    if (data === undefined) {
      const response = await fetch(withBase(props.source))
      if (!response.ok) {
        throw new Error(`Chart data request failed with HTTP ${response.status}`)
      }
      data = await response.json()
    }
    const { default: embed } = await import('vega-embed')
    if (sequence !== renderSequence) return
    renderedView?.finalize()
    element.replaceChildren()
    const result = await embed(element, props.specification(data, theme()), {
      actions: false,
      renderer: 'svg',
    })
    if (sequence !== renderSequence) {
      result.view.finalize()
      return
    }
    renderedView = result.view
    status.value = 'ready'
  } catch (error) {
    if (sequence !== renderSequence) return
    console.error('Unable to render Vega-Lite chart', error)
    status.value = 'error'
  }
}

onMounted(() => {
  const element = chart.value
  if (element === undefined) return
  intersectionObserver = new IntersectionObserver(
    (entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return
      intersectionObserver?.disconnect()
      intersectionObserver = undefined
      void render()
    },
    { rootMargin: '240px 0px' },
  )
  intersectionObserver.observe(element)

  themeObserver = new MutationObserver(() => {
    if (status.value === 'ready') void render()
  })
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })

  resizeObserver = new ResizeObserver(() => {
    if (renderedView === undefined || resizeFrame !== undefined) return
    resizeFrame = requestAnimationFrame(() => {
      resizeFrame = undefined
      void nextTick(async () => {
        await renderedView?.resize().runAsync()
      })
    })
  })
  resizeObserver.observe(element)
})

onBeforeUnmount(() => {
  ++renderSequence
  intersectionObserver?.disconnect()
  resizeObserver?.disconnect()
  themeObserver?.disconnect()
  if (resizeFrame !== undefined) cancelAnimationFrame(resizeFrame)
  renderedView?.finalize()
})
</script>

<template>
  <figure class="vega-lite-figure">
    <div
      ref="chart"
      class="vega-lite-chart"
      role="img"
      :aria-label="label"
      :aria-busy="status === 'loading'"
    >
      <span v-if="status === 'waiting' || status === 'loading'">Loading chart…</span>
      <span v-else-if="status === 'error'">Chart unavailable.</span>
    </div>
    <figcaption v-if="$slots.default"><slot /></figcaption>
  </figure>
</template>

<style scoped>
.vega-lite-figure {
  margin: var(--fhe-space-5) 0;
}

.vega-lite-chart {
  width: 100%;
  min-height: 470px;
  color: var(--fhe-c-text-2);
  background: transparent;
  font-size: 12px;
}

.vega-lite-chart :deep(.vega-embed),
.vega-lite-chart :deep(.vega-embed > div),
.vega-lite-chart :deep(svg) {
  display: block;
  width: 100%;
  max-width: 100%;
  background: transparent;
}

figcaption {
  margin-top: var(--fhe-space-2);
  color: var(--fhe-c-text-2);
  font-size: 12px;
  line-height: 1.55;
}
</style>
