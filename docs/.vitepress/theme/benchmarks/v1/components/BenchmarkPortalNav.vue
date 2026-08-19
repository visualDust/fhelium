<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, withBase } from 'vitepress'

const route = useRoute()
const active = computed(() => {
  if (route.path.startsWith('/benchmarks/v1/compare')) return 'compare'
  if (route.path.startsWith('/benchmarks/v1/methodology')) return 'methodology'
  return 'results'
})
</script>

<template>
  <nav class="benchmark-subnav" aria-label="Benchmark portal">
    <a
      :href="withBase('/benchmarks/')"
      :aria-current="active === 'results' ? 'page' : undefined"
    >Results</a>
    <a
      :href="withBase('/benchmarks/v1/compare')"
      :aria-current="active === 'compare' ? 'page' : undefined"
    >Compare</a>
    <a
      :href="withBase('/benchmarks/v1/methodology')"
      :aria-current="active === 'methodology' ? 'page' : undefined"
    >Methodology</a>
  </nav>
</template>

<style scoped>
.benchmark-subnav {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  padding: 4px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: color-mix(in srgb, var(--vp-c-bg) 84%, transparent);
}

.benchmark-subnav a {
  padding: 7px 12px;
  border-radius: 5px;
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 680;
  line-height: 1;
  text-decoration: none;
}

.benchmark-subnav a:hover {
  color: var(--vp-c-brand-1);
}

.benchmark-subnav a[aria-current='page'] {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
  box-shadow: inset 0 0 0 1px var(--vp-c-divider);
}

@media (max-width: 440px) {
  .benchmark-subnav {
    display: grid;
    width: 100%;
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .benchmark-subnav a {
    padding-inline: 6px;
    text-align: center;
  }
}

@media (forced-colors: active) {
  .benchmark-subnav,
  .benchmark-subnav a[aria-current='page'] {
    border-color: CanvasText;
  }
}
</style>
