<script setup lang="ts">
import { withBase } from 'vitepress'

import BenchmarkPortalNav from './BenchmarkPortalNav.vue'

withDefaults(defineProps<{ title?: string }>(), {
  title: 'Benchmark v1 results',
})
</script>

<template>
  <header class="portal-shared-head">
    <div class="portal-shared-head__bar">
      <a class="portal-shared-head__wordmark" :href="withBase('/benchmarks/')">
        FHElium <span>Benchmark v1</span>
      </a>
      <BenchmarkPortalNav />
    </div>
    <h1 class="portal-shared-head__title">{{ title }}</h1>
  </header>
</template>

<style scoped>
.portal-shared-head {
  padding: 12px max(24px, calc((100vw - 1440px) / 2)) 15px;
  background: linear-gradient(
    105deg,
    color-mix(in srgb, var(--vp-c-brand-soft) 28%, var(--vp-c-bg)),
    var(--vp-c-bg) 62%
  );
}

.portal-shared-head__bar {
  display: flex;
  gap: 24px;
  align-items: center;
  justify-content: space-between;
}

.portal-shared-head__wordmark {
  color: var(--vp-c-text-1);
  font-size: 15px;
  font-weight: 800;
  letter-spacing: -0.02em;
  text-decoration: none;
}

.portal-shared-head__wordmark:hover,
.portal-shared-head__wordmark:focus-visible {
  color: var(--vp-c-brand-1);
}

.portal-shared-head__wordmark span {
  color: var(--vp-c-brand-1);
  font-weight: 650;
}

.portal-shared-head__title {
  margin: 32px 0 0;
  border: 0;
  font-size: clamp(25px, 2.5vw, 32px);
  line-height: 1;
  letter-spacing: -0.035em;
}

@media (max-width: 720px) {
  .portal-shared-head {
    padding: 14px 16px 22px;
  }

  .portal-shared-head__bar {
    align-items: flex-start;
    flex-direction: column;
  }

  .portal-shared-head__title {
    margin-top: 28px;
  }
}
</style>
