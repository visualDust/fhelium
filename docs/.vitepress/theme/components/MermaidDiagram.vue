<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  useId,
  watch,
} from 'vue'

const props = defineProps<{ code: string }>()
const target = ref<HTMLElement | null>(null)
const expandButton = ref<HTMLButtonElement | null>(null)
const expandedTarget = ref<HTMLElement | null>(null)
const expandedViewport = ref<HTMLElement | null>(null)
const dialog = ref<HTMLDialogElement | null>(null)
const errorMessage = ref('')
const expandedErrorMessage = ref('')
const zoom = ref(1)
const dragging = ref(false)

let renderSequence = 0
let expandedRenderSequence = 0
let themeObserver: MutationObserver | undefined
let dragStartX = 0
let dragStartY = 0
let dragScrollLeft = 0
let dragScrollTop = 0
const componentId = useId().replace(/[^A-Za-z0-9_-]/gu, '')
const dialogTitleId = `fhelium-mermaid-title-${componentId}`
const minZoom = 1
const maxZoom = 4
const zoomStep = 0.25

const darkMindmapFills = [
  '#405aa8',
  '#6847a4',
  '#8d427d',
  '#286c72',
  '#7a5529',
  '#315f87',
  '#704c72',
  '#3e704b',
  '#4d5e76',
  '#7c434b',
  '#28697a',
  '#5b4c92',
] as const

const darkMindmapLines = [
  '#91a8ff',
  '#b998f2',
  '#df8bc8',
  '#71bac0',
  '#d0a162',
  '#77a9d3',
  '#bd91bd',
  '#7fba89',
  '#93a2bd',
  '#ca858d',
  '#72b3c3',
  '#a092db',
] as const

function darkMindmapVariables() {
  const variables: Record<string, string> = {
    git0: '#5068ba',
    gitBranchLabel0: '#f7f8fc',
  }
  darkMindmapFills.forEach((fill, index) => {
    variables[`cScale${index}`] = fill
    variables[`cScaleInv${index}`] = darkMindmapLines[index]
    variables[`cScaleLabel${index}`] = '#f7f8fc'
    variables[`lineColor${index}`] = darkMindmapLines[index]
  })
  return variables
}

const zoomPercent = computed(() => `${Math.round(zoom.value * 100)}%`)
const canZoomOut = computed(() => zoom.value > minZoom)
const canZoomIn = computed(() => zoom.value < maxZoom)

function mermaidOptions(dark: boolean) {
  return {
    startOnLoad: false,
    securityLevel: 'strict' as const,
    theme: 'base' as const,
    fontFamily: 'var(--vp-font-family-base)',
    themeVariables: dark
      ? {
          background: '#1a1e29',
          primaryColor: '#222a48',
          primaryTextColor: '#f1f3f9',
          primaryBorderColor: '#9aafff',
          secondaryColor: '#202533',
          secondaryTextColor: '#f1f3f9',
          secondaryBorderColor: '#4b5368',
          tertiaryColor: '#11131a',
          tertiaryTextColor: '#f1f3f9',
          tertiaryBorderColor: '#4b5368',
          lineColor: '#b7bdcc',
          textColor: '#f1f3f9',
          mainBkg: '#222a48',
          nodeBorder: '#9aafff',
          clusterBkg: '#171b25',
          clusterBorder: '#ffc25c',
          edgeLabelBackground: '#11131a',
          actorBkg: '#222a48',
          actorBorder: '#9aafff',
          actorTextColor: '#f1f3f9',
          signalColor: '#b7bdcc',
          signalTextColor: '#f1f3f9',
          labelBoxBkgColor: '#222a48',
          labelBoxBorderColor: '#4b5368',
          labelTextColor: '#f1f3f9',
          ...darkMindmapVariables(),
        }
      : {
          background: '#ffffff',
          primaryColor: '#eef0fb',
          primaryTextColor: '#181b26',
          primaryBorderColor: '#3152c7',
          secondaryColor: '#f1f3fa',
          secondaryTextColor: '#181b26',
          secondaryBorderColor: '#b9becb',
          tertiaryColor: '#ffffff',
          tertiaryTextColor: '#181b26',
          tertiaryBorderColor: '#b9becb',
          lineColor: '#505767',
          textColor: '#181b26',
          mainBkg: '#eef0fb',
          nodeBorder: '#3152c7',
          clusterBkg: '#f1f3fa',
          clusterBorder: '#e39a13',
          edgeLabelBackground: '#f7f7f2',
          actorBkg: '#eef0fb',
          actorBorder: '#3152c7',
          actorTextColor: '#181b26',
          signalColor: '#505767',
          signalTextColor: '#181b26',
          labelBoxBkgColor: '#eef0fb',
          labelBoxBorderColor: '#b9becb',
          labelTextColor: '#181b26',
        },
  }
}

function annotateSvg(container: HTMLElement, expanded: boolean) {
  const renderedSvg = container.querySelector('svg')
  renderedSvg?.setAttribute('role', 'img')
  renderedSvg?.setAttribute(
    'aria-label',
    expanded ? 'Expanded Mermaid diagram' : 'Mermaid diagram',
  )

  if (expanded) {
    renderedSvg?.classList.add('is-expanded')
    return
  }

  const viewBox = renderedSvg
    ?.getAttribute('viewBox')
    ?.trim()
    .split(/[\s,]+/u)
    .map(Number)
  if (viewBox?.length === 4 && viewBox[3] > 0) {
    const aspectRatio = viewBox[2] / viewBox[3]
    if (aspectRatio > 2.4) {
      renderedSvg?.classList.add('is-wide')
    }
    if (aspectRatio > 6) {
      renderedSvg?.classList.add('is-ultrawide')
    }
  }
}

async function renderMermaid(id: string) {
  const { default: mermaid } = await import('mermaid')
  const dark = document.documentElement.classList.contains('dark')
  mermaid.initialize(mermaidOptions(dark))
  return mermaid.render(
    id,
    decodeURIComponent(props.code),
  )
}

async function renderDiagram() {
  const currentRender = ++renderSequence
  errorMessage.value = ''
  await nextTick()

  if (!target.value) {
    return
  }

  try {
    const id = `fhelium-mermaid-${componentId}-${currentRender}`
    const { svg, bindFunctions } = await renderMermaid(id)
    if (currentRender !== renderSequence || !target.value) {
      return
    }
    target.value.innerHTML = svg
    annotateSvg(target.value, false)
    bindFunctions?.(target.value)
  } catch (error) {
    if (currentRender === renderSequence) {
      errorMessage.value =
        error instanceof Error ? error.message : 'Unable to render diagram'
    }
  }
}

async function renderExpandedDiagram() {
  const currentRender = ++expandedRenderSequence
  expandedErrorMessage.value = ''
  await nextTick()

  if (!expandedTarget.value || !dialog.value?.open) {
    return
  }

  try {
    const id = `fhelium-mermaid-expanded-${componentId}-${currentRender}`
    const { svg, bindFunctions } = await renderMermaid(id)
    if (
      currentRender !== expandedRenderSequence ||
      !expandedTarget.value ||
      !dialog.value?.open
    ) {
      return
    }
    expandedTarget.value.innerHTML = svg
    annotateSvg(expandedTarget.value, true)
    bindFunctions?.(expandedTarget.value)
  } catch (error) {
    if (currentRender === expandedRenderSequence) {
      expandedErrorMessage.value =
        error instanceof Error ? error.message : 'Unable to render diagram'
    }
  }
}

async function openExpanded() {
  if (!dialog.value || dialog.value.open) {
    return
  }

  zoom.value = minZoom
  dialog.value.showModal()
  await nextTick()
  await renderExpandedDiagram()
  expandedViewport.value?.scrollTo({ left: 0, top: 0 })
}

function closeExpanded() {
  dialog.value?.close()
}

function handleDialogClosed() {
  dragging.value = false
  expandButton.value?.focus()
}

function handleDialogClick(event: MouseEvent) {
  if (event.target === dialog.value) {
    closeExpanded()
  }
}

async function setZoom(nextZoom: number) {
  const viewport = expandedViewport.value
  const normalized = Math.min(maxZoom, Math.max(minZoom, nextZoom))
  if (!viewport || normalized === zoom.value) {
    zoom.value = normalized
    return
  }

  const centerX = (viewport.scrollLeft + viewport.clientWidth / 2) /
    Math.max(viewport.scrollWidth, 1)
  const centerY = (viewport.scrollTop + viewport.clientHeight / 2) /
    Math.max(viewport.scrollHeight, 1)
  zoom.value = normalized
  await nextTick()
  viewport.scrollLeft = centerX * viewport.scrollWidth - viewport.clientWidth / 2
  viewport.scrollTop = centerY * viewport.scrollHeight - viewport.clientHeight / 2
}

function fitDiagram() {
  void setZoom(minZoom).then(() => {
    expandedViewport.value?.scrollTo({ left: 0, top: 0 })
  })
}

function handleDialogKeydown(event: KeyboardEvent) {
  if ((event.key === '+' || event.key === '=') && canZoomIn.value) {
    event.preventDefault()
    void setZoom(zoom.value + zoomStep)
  } else if (event.key === '-' && canZoomOut.value) {
    event.preventDefault()
    void setZoom(zoom.value - zoomStep)
  } else if (event.key === '0') {
    event.preventDefault()
    fitDiagram()
  }
}

function startPan(event: PointerEvent) {
  const viewport = expandedViewport.value
  if (
    !viewport ||
    event.button !== 0 ||
    (viewport.scrollWidth <= viewport.clientWidth &&
      viewport.scrollHeight <= viewport.clientHeight)
  ) {
    return
  }

  dragging.value = true
  dragStartX = event.clientX
  dragStartY = event.clientY
  dragScrollLeft = viewport.scrollLeft
  dragScrollTop = viewport.scrollTop
  viewport.setPointerCapture(event.pointerId)
  event.preventDefault()
}

function panDiagram(event: PointerEvent) {
  if (!dragging.value || !expandedViewport.value) {
    return
  }

  expandedViewport.value.scrollLeft =
    dragScrollLeft - (event.clientX - dragStartX)
  expandedViewport.value.scrollTop =
    dragScrollTop - (event.clientY - dragStartY)
}

function stopPan(event: PointerEvent) {
  if (!dragging.value) {
    return
  }
  dragging.value = false
  if (expandedViewport.value?.hasPointerCapture(event.pointerId)) {
    expandedViewport.value.releasePointerCapture(event.pointerId)
  }
}

function rerenderForTheme() {
  void renderDiagram().then(() => {
    if (dialog.value?.open) {
      void renderExpandedDiagram()
    }
  })
}

onMounted(() => {
  void renderDiagram()
  themeObserver = new MutationObserver(rerenderForTheme)
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })
})

onBeforeUnmount(() => {
  themeObserver?.disconnect()
  if (dialog.value?.open) {
    dialog.value.close()
  }
})

watch(() => props.code, () => rerenderForTheme())
</script>

<template>
  <div class="mermaid-diagram">
    <button
      v-if="!errorMessage"
      ref="expandButton"
      class="mermaid-expand-button"
      type="button"
      title="Open diagram"
      aria-label="Open diagram in expanded view"
      @click="openExpanded"
    >
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
      </svg>
    </button>
    <div
      ref="target"
      class="mermaid-diagram-target"
      :hidden="Boolean(errorMessage)"
    />
    <pre v-if="errorMessage" class="mermaid-error">{{ errorMessage }}</pre>

    <dialog
      ref="dialog"
      class="mermaid-lightbox"
      :aria-labelledby="dialogTitleId"
      @click="handleDialogClick"
      @close="handleDialogClosed"
      @keydown="handleDialogKeydown"
    >
      <div class="mermaid-lightbox-shell">
        <header class="mermaid-lightbox-toolbar">
          <strong :id="dialogTitleId">Expanded diagram</strong>
          <div class="mermaid-lightbox-controls">
            <button
              type="button"
              title="Zoom out"
              aria-label="Zoom out"
              :disabled="!canZoomOut"
              @click="setZoom(zoom - zoomStep)"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M5 12h14" />
              </svg>
            </button>
            <output aria-live="polite">{{ zoomPercent }}</output>
            <button
              type="button"
              title="Zoom in"
              aria-label="Zoom in"
              :disabled="!canZoomIn"
              @click="setZoom(zoom + zoomStep)"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>
            <button
              type="button"
              title="Fit diagram"
              aria-label="Fit diagram to the window"
              @click="fitDiagram"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
              </svg>
            </button>
            <button
              class="mermaid-lightbox-close"
              type="button"
              title="Close"
              aria-label="Close expanded diagram"
              autofocus
              @click="closeExpanded"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="m5 5 14 14M19 5 5 19" />
              </svg>
            </button>
          </div>
        </header>
        <div
          ref="expandedViewport"
          class="mermaid-lightbox-viewport"
          :class="{ 'is-dragging': dragging }"
          @pointerdown="startPan"
          @pointermove="panDiagram"
          @pointerup="stopPan"
          @pointercancel="stopPan"
        >
          <div
            ref="expandedTarget"
            class="mermaid-lightbox-target"
            :style="{ width: `${zoom * 100}%` }"
            :hidden="Boolean(expandedErrorMessage)"
          />
          <pre v-if="expandedErrorMessage" class="mermaid-error">
            {{ expandedErrorMessage }}
          </pre>
        </div>
      </div>
    </dialog>
  </div>
</template>
