import type { TrendSeries } from '../explorer/BenchmarkTrend.vue'
import { hardwareLabel, palette, runs, type Workload, type Plan } from '../explorer/records'

export const depths = [0, 4, 8, 12, 16, 20, 24]
export const batches = [1, 2, 4, 8, 16]
export const sizes = [16, 32, 64]
export type WorkloadId = 'ptct' | 'ctct' | 'polynomial' | 'ntt' | 'keyswitch' | 'rns'
export type Mode = 'scaling' | 'throughput'
export const workloads: { id: WorkloadId; title: string; subtitle: string; method: string; operations: string[] }[] = [
  { id: 'ptct' as const, title: 'PT × CT matrix multiplication', subtitle: 'Plaintext matrix × encrypted matrix', method: 'Hoisting enabled',
    operations: ['Rotate (1 offset)', 'Rotate-many (7 offsets)', 'Multiply (PT × CT)', 'Sum products (8 PT × CT terms)', 'Rescale (2 components)'] },
  { id: 'ctct' as const, title: 'CT × CT matrix multiplication', subtitle: 'Encrypted matrix × encrypted matrix', method: 'Hoisting enabled',
    operations: ['Rotate (1 offset)', 'Rotate-many (7 offsets)', 'Multiply (CT × CT)', 'Add (CT + CT)', 'Relinearize', 'Rescale (2 components)'] },
  { id: 'polynomial' as const, title: 'Nonlinear polynomial evaluation', subtitle: 'Degree-twelve power polynomial', method: 'Balanced evaluation',
    operations: ['Multiply (CT × CT)', 'Square (CT)', 'Multiply scalar & rescale', 'Add (CT + CT)', 'Relinearize', 'Rescale (2 components)', 'Advance depth'] },
  { id: 'ntt', title: 'NTT', subtitle: 'Complete negacyclic transform', method: 'One RNS polynomial per task', operations: [] },
  { id: 'keyswitch', title: 'Key switching', subtitle: 'Complete ciphertext key switch', method: 'Hybrid RNS key switching · coefficient input and NTT output', operations: [] },
  { id: 'rns', title: 'RNS arithmetic', subtitle: 'Coefficient-wise modular arithmetic', method: 'One RNS polynomial per task', operations: [] },
]
export const isMatrix = (id: WorkloadId) => id === 'ptct' || id === 'ctct'
export const isPrimitive = (id: WorkloadId) => id === 'ntt' || id === 'keyswitch' || id === 'rns'
const operationIds: Record<string, Workload> = {
  'Rotate (1 offset)': 'rotate',
  'Rotate-many (7 offsets)': 'rotate_many',
  'Multiply (PT × CT)': 'multiply_plaintext',
  'Sum products (8 PT × CT terms)': 'sum_plaintext_products',
  'Rescale (2 components)': 'rescale_ct2',
  'Multiply (CT × CT)': 'multiply_ciphertexts',
  'Add (CT + CT)': 'add',
  'Relinearize': 'relinearize',
  'Square (CT)': 'square',
  'Multiply scalar & rescale': 'multiply_coefficient',
  'Advance depth': 'advance_depth',
}
export const operationNotes: Record<string, string> = {
  'Rotate (1 offset)': 'Shift +1',
  'Rotate-many (7 offsets)': 'Shifts +1…+7 · one input',
  'Multiply (PT × CT)': 'No rescale',
  'Sum products (8 PT × CT terms)': 'Sum of eight products · no rescale',
  'Rescale (2 components)': 'One Q depth group',
  'Multiply (CT × CT)': 'Distinct inputs · no relinearization or rescale',
  'Add (CT + CT)': 'Same depth and scale',
  'Relinearize': '3 → 2 components · no rescale',
  'Square (CT)': 'Same input · no relinearization or rescale',
  'Multiply scalar & rescale': 'Scalar −0.3 · one rescale',
  'Advance depth': 'Multiply by one & rescale',
}
export const requiredPolynomialDepths = [5, 12, 5]
export function curves(id: WorkloadId, mode: Mode, depth: number, size: number, method: number, recordId?: string, operation?: number, hoisting = true, inverseNtt = false): TrendSeries[] {
  const workload = workloads.find(w => w.id === id)!
  const selectedOp = operation === undefined ? undefined : operationIds[workload.operations[operation]]
  const op = selectedOp === 'rotate_many' && !hoisting ? 'rotate_many_independent' : selectedOp
  const task: Workload = op ?? (id === 'ntt' ? (inverseNtt ? 'ntt_inverse' : 'ntt_forward') : id === 'keyswitch' ? 'key_switch' : id === 'rns' ? (method === 0 ? 'rns_multiply' : 'rns_add') : id === 'ptct' ? 'matrix_ptct' : id === 'ctct' ? 'matrix_ctct' : 'polynomial')
  const requestedPlan: Plan = op ? (op === 'rotate_many' ? 'shared' : 'direct') : id === 'ntt' ? (method === 0 ? 'radix2_compact_group8_smem8' : 'radix2_indexed') : id === 'keyswitch' || id === 'rns' ? 'direct' : id === 'polynomial' ? (['balanced', 'horner', 'paterson_stockmeyer'] as const)[method] : method === 0 ? 'shared' : 'independent'
  return runs.filter(r => !recordId || r.id === recordId).map(r => {
    const plan: Plan = !op && id === 'ntt' && r.platform.device === 'cpu' ? 'radix2_indexed' : requestedPlan
    return {
    id: r.id, label: hardwareLabel(r), color: palette[runs.indexOf(r) % palette.length],
    points: batches.map(batch => {
      const point = { id: `${task}/${plan}/d${depth}/b${batch}/s${op || !isMatrix(id) ? 0 : size}`, x: batch }
      // Earlier records named the identical self-product measurement "multiply".
      const cell = [...(r.additional_runs ?? []).flatMap(extra => extra.results), ...r.results].find(c => c.config === 'slots32768-scale50-depth27-int64' && (c.workload === task || task === 'square' && c.workload === 'multiply') && c.plan === plan && c.depth === depth && c.batch === batch && c.size === (op || !isMatrix(id) ? null : size))
      if (!cell) return { ...point, value: null, reason: 'Not measured' }
      if (cell.status === 'unsupported') return { ...point, value: null, reason: cell.reason ?? 'Implementation unavailable on this device' }
      if (cell.status === 'capacity') return { ...point, value: null, reason: 'Device capacity exceeded' }
      const m = cell.measurement
      if (!m) return { ...point, value: null }
      return mode === 'scaling' ? { ...point, value: m.median_ms / batch, low: m.q25_ms / batch, high: m.q75_ms / batch }
        : { ...point, value: m.tasks_per_second, low: batch * 1000 / m.q75_ms, high: batch * 1000 / m.q25_ms }
    }),
    }
  })
}
