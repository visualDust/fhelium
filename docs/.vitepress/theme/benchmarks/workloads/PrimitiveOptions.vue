<script setup lang="ts">
import type { WorkloadId } from './workloadData'
defineProps<{ id: WorkloadId; method: number; inverse: boolean }>()
const emit = defineEmits<{ 'update:method': [value: number]; 'update:inverse': [value: boolean] }>()
</script>
<template>
  <div class="primitive-options">
    <template v-if="id === 'ntt'">
      <select aria-label="NTT direction" :value="inverse ? 'inverse' : 'forward'" @change="emit('update:inverse', ($event.target as HTMLSelectElement).value === 'inverse')"><option value="forward">Forward</option><option value="inverse">Inverse</option></select>
      <select aria-label="NTT implementation" :value="method" @change="emit('update:method', Number(($event.target as HTMLSelectElement).value))"><option :value="0">Grouped stages</option><option :value="1">Staged radix-2</option></select>
    </template>
    <select v-else-if="id === 'rns'" aria-label="RNS operation" :value="method" @change="emit('update:method', Number(($event.target as HTMLSelectElement).value))"><option :value="0">Montgomery multiply</option><option :value="1">Modular add</option></select>
  </div>
</template>
<style scoped>
.primitive-options { display:flex; flex-wrap:wrap; gap:8px; }
select { max-width:100%; padding:5px 8px; border:1px solid var(--vp-c-divider); border-radius:6px; background:var(--vp-c-bg-soft); color:var(--vp-c-text-2); font-size:11px; cursor:pointer; }
select:focus-visible { outline:2px solid var(--vp-c-brand-1); outline-offset:2px; }
</style>
