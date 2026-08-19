<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{
  title: string
  description: string
  href: string
}>()

const resolvedHref = computed(() => {
  if (/^(?:[a-z]+:|#)/iu.test(props.href)) {
    return props.href
  }
  return withBase(props.href)
})
</script>

<template>
  <a class="doc-card" :href="resolvedHref">
    <strong>{{ title }}</strong>
    <span class="doc-card-description">{{ description }}</span>
    <span class="doc-card-arrow" aria-hidden="true">→</span>
  </a>
</template>
