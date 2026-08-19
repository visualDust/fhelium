<script setup lang="ts">
import HomeProgrammingModel from './HomeProgrammingModel.vue'
import InstallCommand from './InstallCommand.vue'
</script>

<template>
  <section class="home-opening-rail" aria-label="FHElium programming model and installation">
    <HomeProgrammingModel />
    <InstallCommand />
  </section>
</template>

<style scoped>
.home-opening-rail {
  position: relative;
  z-index: 2;
  margin: -28px 0 var(--fhe-space-8);
  overflow: hidden;
  border: 1px solid var(--fhe-c-divider);
  border-radius: var(--fhe-radius-panel);
  background: color-mix(in srgb, var(--fhe-c-surface) 32%, transparent);
  box-shadow: var(--fhe-shadow-panel);
  backdrop-filter: blur(18px) saturate(1.08);
  -webkit-backdrop-filter: blur(18px) saturate(1.08);
}

.home-opening-rail :deep(.programming-model) {
  margin: 0;
  border: 0;
  border-radius: 0;
  background: color-mix(in srgb, var(--vp-c-bg) 66%, transparent);
}

.home-opening-rail :deep(.home-install) {
  margin: 0;
  border: 0;
  border-top: 1px solid var(--fhe-c-divider);
  border-radius: 0;
  background: color-mix(in srgb, var(--fhe-c-surface) 78%, transparent);
  box-shadow: none;
}

.home-opening-rail :deep(.home-install::before) {
  display: none;
}

@media (max-width: 640px) {
  .home-opening-rail {
    margin-top: -12px;
  }
}

@media (forced-colors: active) {
  .home-opening-rail,
  .home-opening-rail :deep(.programming-model),
  .home-opening-rail :deep(.home-install) {
    background: Canvas;
    backdrop-filter: none;
  }
}
</style>
