<script setup lang="ts">
import { compileTimeline } from '../charts/compileTimeline'
import KernelTimeline from './KernelTimeline.vue'
</script>

<template>
  <KernelTimeline :chart="compileTimeline" />
</template>
