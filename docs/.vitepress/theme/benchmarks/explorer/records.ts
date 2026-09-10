import data from './measuredRuns.json'

export type Workload = 'matrix_ptct' | 'matrix_ctct' | 'rotate' | 'rotate_many' | 'relinearize' | 'rescale' | 'multiply_plaintext' | 'multiply' | 'square' | 'polynomial' | 'rotate_many_independent' | 'rescale_ct2' | 'multiply_ciphertexts' | 'add' | 'sum_plaintext_products' | 'multiply_coefficient' | 'advance_depth' | 'ntt_forward' | 'ntt_inverse' | 'key_switch' | 'rns_add' | 'rns_multiply'
export type Plan = 'direct' | 'shared' | 'independent' | 'balanced' | 'horner' | 'paterson_stockmeyer' | 'radix2_indexed' | 'radix2_compact_group8_smem8'
export interface Configuration {
  id: string
  ring_degree: number
  max_depth: number
  default_scale: number
  q_depth_groups: string[][]
  p_moduli: string[]
  states: { depth: number; q_rows: number; q_bits: number; qp_bits: number }[]
}
export interface Measurement {
  median_ms: number
  q25_ms: number
  q75_ms: number
  tasks_per_second: number
  required_evaluation_key_bytes: number
  resident_allocated_bytes: number | null
  peak_allocated_bytes: number | null
  temporary_allocated_bytes: number | null
  input_scale: number | null
  output_depth: number
  output_scale: number | null
  output_components: number | null
  output_domain: string
  max_absolute_error: number
  rms_error: number
  checked_elements: number
}
export interface Cell {
  id: string
  config: string
  workload: Workload
  depth: number
  batch: number
  size: number | null
  plan: Plan
  status: 'passed' | 'capacity' | 'unsupported'
  reason?: string
  measurement?: Measurement
}
export interface Run {
  additional_runs?: Run[]
  polynomial?: { degree: number; coefficients_ascending: number[]; domain: number[]; methods: Record<string, { required_depths: number }> }
  id: string
  started_at: string
  finished_at: string
  specification_hash: string
  comparison_hash: string
  recorded_case_count: number
  source: { version: string; commit?: string; tracked_diff_sha256?: string; suite_code_sha256: string; base_commit?: string; snapshot_sha256?: string }
  platform: { device: string; name: string; cpu: string; logical_processors: number; torch_threads: number; torch_interop_threads: number; torch: string; cuda: string | null; python: string; os: string; architecture: string; compute_capability: number[] | null; device_memory_bytes: number | null; gpu_power_limit_w?: number; host_allocation?: string }
  configurations: Configuration[]
  sampling: { warmups: number; samples: number; absolute_error_limit: number; timing: string; execution: string }
  results: Cell[]
}
export const runs = data as Run[]
export const palette = ['#4779cf', '#159883', '#a466c3', '#c37732', '#c4516b', '#64822e']
export const runUrl = (id: string) => `/benchmarks/records/${id}`

export const hardwareLabel = (run: Run) => run.platform.name + (run.platform.gpu_power_limit_w !== undefined ? ` · ${run.platform.gpu_power_limit_w} W` : '')
