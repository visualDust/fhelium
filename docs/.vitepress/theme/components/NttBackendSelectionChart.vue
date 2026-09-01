<script setup lang="ts">
import { nttBackendSelectionSpec } from '../charts/nttBackendSelection'
import VegaLiteChart from './VegaLiteChart.vue'
</script>

<template>
  <VegaLiteChart
    source="/data/ntt-backend-selection-sm120.json"
    label="Relative NTT backend latency measured with NTT-only and complete CKKS operations"
    :specification="nttBackendSelectionSpec"
  >
    Relative latency for the five tested backends. The shaded interval extends
    through 3%; the ring marks the backend selected by the complete CKKS
    operations.
  </VegaLiteChart>
</template>
