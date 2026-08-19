<script setup lang="ts">
interface ModelDimension {
  label: string
  value: string
  detail: string
}

const dimensions: readonly ModelDimension[] = [
  {
    label: 'Value model',
    value: 'Core Ciphertext throughout',
    detail: 'Authoring and execution layers share one encrypted value type.',
  },
  {
    label: 'CKKS state',
    value: 'Value state · operation-validated',
    detail: 'Level, scale, prime IDs, basis, domain, residues, and components remain visible.',
  },
  {
    label: 'Evaluator artifacts',
    value: 'Keys + prepared plaintexts',
    detail: 'Rotation/relinearization keys and prepared public operands remain reusable.',
  },
  {
    label: 'Execution',
    value: 'CPU · CUDA · replayable',
    detail: 'Tensor device selects eager CPU or CUDA execution; CUDA may add Graph replay.',
  },
  {
    label: 'Placement',
    value: 'One local device · Multi-GPU',
    detail: 'Keep one engine on a CPU or GPU, or compose GPU ranks with typed collectives.',
  },
]
</script>

<template>
  <section class="programming-model" aria-label="FHElium programming model">
    <dl>
      <div v-for="dimension in dimensions" :key="dimension.label">
        <dt>{{ dimension.label }}</dt>
        <dd>
          <strong>{{ dimension.value }}</strong>
          <span>{{ dimension.detail }}</span>
        </dd>
      </div>
    </dl>

  </section>
</template>

<style scoped>
.programming-model {
  --model-line: #cbd3e3;
  --model-ink: #172033;
  --model-muted: #626d82;
  --model-blue: #3159d5;
  margin: 0;
  overflow: hidden;
  border: 1px solid var(--model-line);
  border-radius: 8px;
  color: var(--model-ink);
  background: var(--vp-c-bg);
}

:global(.dark .programming-model) {
  --model-line: rgba(154, 172, 225, 0.23);
  --model-ink: #eef1fb;
  --model-muted: #9ca6bb;
  --model-blue: #8da4ff;
}

.programming-model dl {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  margin: 0;
}

.programming-model dl > div {
  min-width: 0;
  padding: 15px 16px 17px;
}

.programming-model dl > div + div {
  border-left: 1px solid var(--model-line);
}

.programming-model dt {
  margin-bottom: 9px;
  color: var(--model-blue);
  font-family: var(--vp-font-family-mono);
  font-size: 8px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.programming-model dd {
  display: grid;
  gap: 6px;
  margin: 0;
}

.programming-model dd strong {
  color: var(--model-ink);
  font-size: 10px;
  line-height: 1.45;
}

.programming-model dd span {
  color: var(--model-muted);
  font-size: 8.5px;
  line-height: 1.52;
}

@media (max-width: 900px) {
  .programming-model dl {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .programming-model dl > div {
    border-top: 1px solid var(--model-line);
  }

  .programming-model dl > div:nth-child(-n + 2) {
    border-top: 0;
  }

  .programming-model dl > div:nth-child(odd) {
    border-left: 0;
  }

  .programming-model dl > div:last-child {
    grid-column: 1 / -1;
    border-left: 0;
  }
}

@media (max-width: 640px) {
  .programming-model dl {
    grid-template-columns: minmax(0, 1fr);
  }

  .programming-model dl > div,
  .programming-model dl > div:nth-child(-n + 2) {
    border-top: 1px solid var(--model-line);
    border-left: 0;
  }

  .programming-model dl > div:first-child {
    border-top: 0;
  }

  .programming-model dl > div:last-child {
    grid-column: auto;
  }

}
</style>
