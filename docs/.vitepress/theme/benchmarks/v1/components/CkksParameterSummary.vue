<script setup lang="ts">
import { computed } from 'vue'

import type {
  BenchmarkJson,
  BenchmarkJsonObject,
} from '../data/catalog'
import {
  formatScalar,
  humanizeIdentifier,
  jsonObject,
} from '../data/ui'

interface ParameterCluster {
  detail: string[]
  id: string
  lead: string
  title: string
}

const props = defineProps<{
  context: BenchmarkJsonObject | null
}>()

const plan = computed(() => jsonObject(props.context?.ckks_plan))
const entry = computed(() => jsonObject(props.context?.entry_state))
const selection = computed(() => jsonObject(props.context?.parameter_selection))

function valueAt(
  sources: Array<BenchmarkJsonObject | null>,
  names: readonly string[],
): BenchmarkJson | undefined {
  for (const source of sources) {
    if (!source) continue
    for (const name of names) {
      const value = source[name]
      if (value !== undefined && value !== null) return value
    }
  }
  return undefined
}

function compactValue(value: BenchmarkJson): string {
  if (Array.isArray(value)) {
    if (value.every((item) => typeof item === 'number' || typeof item === 'string')) {
      const items = value.map(String)
      return items.length <= 8
        ? items.join(', ')
        : `${items.slice(0, 3).join(', ')} … ${items.slice(-2).join(', ')} (${items.length})`
    }
    return `${value.length} entries`
  }
  if (typeof value === 'object' && value !== null) return `${Object.keys(value).length} fields`
  return formatScalar(value)
}

function display(
  sources: Array<BenchmarkJsonObject | null>,
  names: readonly string[],
): string | null {
  const value = valueAt(sources, names)
  return value === undefined ? null : compactValue(value)
}

function count(
  sources: Array<BenchmarkJsonObject | null>,
  directNames: readonly string[],
  arrayNames: readonly string[],
): string | null {
  const direct = display(sources, directNames)
  if (direct !== null) return direct
  const values = valueAt(sources, arrayNames)
  return Array.isArray(values) ? String(values.length) : null
}

function detail(...items: Array<string | null>): string[] {
  return items.filter((item): item is string => Boolean(item))
}

function labeled(value: string | null, label: string): string | null {
  return value === null ? null : `${value} ${label}`
}

function backendLabel(value: string): string {
  const match = /^radix(\d+)_compact(?:_group(\d+))?(?:_smem(\d+))?$/u.exec(value)
  if (!match) return humanizeIdentifier(value)
  return [
    `Radix-${match[1]} compact`,
    match[2] ? `group ${match[2]}` : null,
    match[3] ? `SMEM ${match[3]}` : null,
  ].filter(Boolean).join(' · ')
}

const clusters = computed<ParameterCluster[]>(() => {
  const parameterSources = [plan.value, entry.value]
  const stateSources = [entry.value, plan.value]

  const logN = display(parameterSources, ['logN'])
  const ring = display(parameterSources, ['ring_dimension'])
  const slots = display(parameterSources, ['slot_count', 'complex_slot_count'])
  const dtype = display(parameterSources, ['residue_dtype', 'torch_dtype'])

  const scaleBits = display(stateSources, ['scale_bits', 'default_scale_bits'])
  const actualScale = display(
    [entry.value, plan.value],
    ['scale', 'entry_scale'],
  )
  const defaultScale = display([plan.value], ['default_scale'])
  const level = display(stateSources, ['entry_level', 'level'])
  const depth = display(stateSources, ['entry_depth'])
  const available = display(stateSources, ['available_depth_budget', 'available_transitions'])
  const predicted = display(stateSources, ['expected_workload_depth', 'maximum_workload_depth'])
  const exitLevel = display(stateSources, ['exit_level'])
  const exitDepth = display(stateSources, ['exit_depth'])

  const qCount = count(
    [entry.value],
    ['entry_active_q_count', 'active_q_count'],
    ['entry_active_q_moduli', 'active_q_moduli', 'prime_ids'],
  ) ?? count(
    [plan.value],
    ['entry_active_q_count', 'active_q_count', 'num_q_primes'],
    ['entry_active_q_moduli', 'active_q_moduli', 'q_moduli', 'q_rows'],
  )
  const pCount = count(
    stateSources,
    ['parameter_p_count', 'num_p_primes'],
    ['parameter_p_moduli', 'p_moduli', 'p_rows'],
  )
  const activeQBits = display(stateSources, [
    'entry_active_q_product_bits',
    'entry_q_product_bits',
    'active_q_product_bits',
    'q0_product_bits',
  ])
  const completeBits = display(stateSources, [
    'complete_qp_product_bits',
    'qp_parameter_product_bits',
    'total_modulus_bits',
  ])
  const budgetBits = display(stateSources, ['maximum_qp_product_bits', 'maximum_security_budget_bits', 'security_budget_bits'])
  const securityBits = display(stateSources, ['security_bits'])

  const backendValue = valueAt(stateSources, ['ntt_backend'])
  const backend = backendValue === undefined
    ? null
    : backendLabel(String(backendValue))
  const basis = display(stateSources, ['entry_modulus_basis', 'modulus_basis', 'modulus_bases'])
  const domain = display(stateSources, ['polynomial_domain'])
  const residues = display(stateSources, ['residue_representation'])
  const securityCheck = valueAt(stateSources, ['security_budget_enforced'])

  const geometryLead = logN !== null ? `N = 2^${logN}` : `N = ${ring ?? 'unspecified'}`
  const stateLead = [
    actualScale !== null
      ? `Δ ${actualScale}`
      : scaleBits !== null
        ? `${scaleBits}-bit default scale`
        : defaultScale !== null
          ? `default Δ ${defaultScale}`
          : 'Scale unspecified',
    level !== null ? `level ${level}` : null,
  ].filter(Boolean).join(' · ')
  const modulusLead = [
    qCount !== null ? `Q ${qCount}${activeQBits !== null ? ` / ${activeQBits} bits` : ''}` : 'Q chain',
    pCount !== null ? `P ${pCount}` : null,
  ].filter(Boolean).join(' · ')

  return [
    {
      detail: detail(
        logN !== null ? labeled(ring, 'coefficients') : null,
        labeled(slots, 'complex slots'),
        dtype !== null ? `${dtype} residues` : null,
      ),
      id: 'geometry',
      lead: geometryLead,
      title: 'Ring geometry',
    },
    {
      detail: detail(
        actualScale !== null ? labeled(scaleBits, 'configured scale bits') : null,
        labeled(depth, 'entry depth'),
        labeled(available, 'available transitions'),
        labeled(predicted, 'predicted depth'),
        exitLevel !== null || exitDepth !== null
          ? `exit ${exitLevel !== null ? `L${exitLevel}` : ''}${exitLevel !== null && exitDepth !== null ? ' / ' : ''}${exitDepth !== null ? `D${exitDepth}` : ''}`
          : null,
      ),
      id: 'state',
      lead: stateLead,
      title: 'Scale and state',
    },
    {
      detail: detail(
        labeled(completeBits, 'complete Q·P bits'),
        labeled(budgetBits, 'security-budget bits'),
        securityBits !== null ? `${securityBits}-bit security target` : null,
      ),
      id: 'modulus',
      lead: modulusLead,
      title: 'Modulus budget',
    },
    {
      detail: detail(
        basis !== null ? `${basis} entry basis` : null,
        domain !== null ? `${domain} domain` : null,
        residues !== null ? `${residues} residues` : null,
        securityCheck === true ? 'security check enforced' : null,
      ),
      id: 'execution',
      lead: backend ?? basis ?? 'Configured execution policy',
      title: 'Execution policy',
    },
  ]
})

const rationale = computed(() => {
  if (!selection.value) return []
  return Object.entries(selection.value).flatMap(([name, value]) => {
    if (!/(rationale|policy|rule|selection)/u.test(name)) return []
    if (typeof value === 'string') return [{ label: humanizeIdentifier(name), value }]
    if (Array.isArray(value)) {
      return value
        .filter((item): item is string => typeof item === 'string')
        .map((item) => ({ label: humanizeIdentifier(name), value: item }))
    }
    return []
  })
})

const expandedContext = computed(() => JSON.stringify(props.context, null, 2))
</script>

<template>
  <section class="ckks-summary" aria-label="CKKS parameters">
    <h3>CKKS parameters</h3>

    <div class="ckks-summary__clusters">
      <section v-for="cluster in clusters" :key="cluster.id">
        <h4>{{ cluster.title }}</h4>
        <strong>{{ cluster.lead }}</strong>
        <p v-if="cluster.detail.length">{{ cluster.detail.join(' · ') }}</p>
      </section>
    </div>

    <div class="ckks-summary__disclosures">
      <details v-if="rationale.length">
        <summary>Parameter-selection notes</summary>
        <div class="ckks-summary__rationale">
          <p v-for="item in rationale" :key="`${item.label}:${item.value}`">
            <span>{{ item.label }}</span>{{ item.value }}
          </p>
        </div>
      </details>
      <details>
        <summary>Exact CKKS parameters and states</summary>
        <pre>{{ expandedContext }}</pre>
      </details>
    </div>
  </section>
</template>

<style scoped>
.ckks-summary {
  margin-top: 18px;
  padding: 19px 20px 15px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--vp-c-bg-soft) 74%, var(--vp-c-bg));
  box-shadow: 0 10px 30px color-mix(in srgb, var(--vp-c-text-1) 7%, transparent);
}

.ckks-summary > h3 {
  margin: 0;
  border: 0;
  font-size: 22px;
}

.ckks-summary__clusters {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 24px;
  margin-top: 20px;
}

.ckks-summary__clusters > section {
  min-width: 0;
}

.ckks-summary__clusters h4 {
  margin: 0 0 7px;
  color: var(--vp-c-brand-1);
  font-size: 10px;
  font-weight: 700;
}

.ckks-summary__clusters strong {
  display: block;
  overflow-wrap: anywhere;
  font-family: var(--vp-font-family-mono);
  font-size: 16px;
  line-height: 1.25;
}

.ckks-summary__clusters p {
  margin: 7px 0 0;
  color: var(--vp-c-text-2);
  font-size: 10px;
  line-height: 1.5;
}

.ckks-summary__disclosures {
  display: flex;
  gap: 7px 18px;
  flex-wrap: wrap;
  margin-top: 18px;
}

.ckks-summary details {
  min-width: 0;
}

.ckks-summary details[open] {
  flex-basis: 100%;
}

.ckks-summary summary {
  color: var(--vp-c-text-2);
  font-size: 10px;
  font-weight: 650;
  cursor: pointer;
}

.ckks-summary details[open] summary {
  color: var(--vp-c-brand-1);
}

.ckks-summary details > code {
  display: block;
  margin-top: 9px;
  overflow-wrap: anywhere;
  color: var(--vp-c-text-2);
  font-size: 10px;
}

.ckks-summary__rationale {
  max-width: 900px;
  margin-top: 10px;
}

.ckks-summary__rationale p {
  margin: 6px 0 0;
  color: var(--vp-c-text-2);
  font-size: 11px;
  line-height: 1.5;
}

.ckks-summary__rationale span {
  margin-right: 7px;
  color: var(--vp-c-text-1);
  font-weight: 650;
}

.ckks-summary pre {
  max-height: 440px;
  margin-top: 10px;
  padding: 12px;
  overflow: auto;
  border-radius: 6px;
  background: var(--vp-c-bg);
  font-size: 10px;
}

@media (max-width: 1050px) {
  .ckks-summary__clusters {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 650px) {
  .ckks-summary {
    padding-inline: 17px;
  }

  .ckks-summary__clusters {
    grid-template-columns: minmax(0, 1fr);
    gap: 18px;
  }

  .ckks-summary__clusters p {
    max-width: 34em;
  }
}

@media (forced-colors: active) {
  .ckks-summary { border: 1px solid CanvasText; }
}
</style>
